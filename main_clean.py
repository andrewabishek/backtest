import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.rename(columns={'date': 'datetime'}, inplace=True)  # Rename for consistency
    
    # Filter for regular market hours (9:15 AM to 3:30 PM)
    df['time'] = df['datetime'].dt.time
    market_start = pd.to_datetime('09:15').time()
    market_end = pd.to_datetime('15:30').time()
    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
    df = df.drop('time', axis=1)
    
    return df

def calculate_pivots(df, leftBars=2, rightBars=2):
    pivots = []
    df_5min = df.resample('5T', on='datetime').agg({'high':'max', 'low':'min'})
    df_5min = df_5min.reset_index()
    for i in range(leftBars, len(df_5min)-rightBars):
        high = df_5min.loc[i, 'high']
        low = df_5min.loc[i, 'low']
        if high == max(df_5min.loc[i-leftBars:i+rightBars, 'high']):
            pivots.append({'type':'high', 'price':high, 'datetime':df_5min.loc[i, 'datetime']})
        if low == min(df_5min.loc[i-leftBars:i+rightBars, 'low']):
            pivots.append({'type':'low', 'price':low, 'datetime':df_5min.loc[i, 'datetime']})
    return pivots

def backtest_strategy(df, pivots):
    trades = []
    used_pivots = set()  # Track pivots that resulted in entry to prevent reuse
    open_trades = []  # Track open trades for intraday closure
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Convert pivots to a lookup by datetime for fast access
    pivot_highs = {p['datetime']: p['price'] for p in pivots if p['type'] == 'high'}
    pivot_lows = {p['datetime']: p['price'] for p in pivots if p['type'] == 'low'}
    
    logger.info(f"Starting backtest with {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows")
    
    # Group data by trading day for intraday management
    df['date'] = df['datetime'].dt.date
    trading_days = df['date'].unique()
    
    logger.info(f"Processing {len(trading_days)} trading days from {trading_days[0]} to {trading_days[-1]}")

    # Iterate over 1-min bars with progress bar
    for i in tqdm(range(len(df)), desc="Processing bars", unit="bars"):
        row = df.iloc[i]
        dt = row['datetime']
        current_time = dt.time()
        current_date = dt.date()
        
        # Close all open trades at 15:25 (intraday rule)
        if current_time >= pd.to_datetime('15:25').time():
            if open_trades:
                logger.info(f"Closing {len(open_trades)} open trades at market close (15:25)")
                for trade_info in open_trades:
                    trade = close_trade_at_market_close(df, i, trade_info)
                    if trade:
                        trades.append(trade)
                        logger.info(f"Trade #{len(trades)} closed at market close: {trade['direction']} P&L: {trade['pl_points']:.1f}")
                open_trades = []
            continue  # No new entries after 15:25
        
        # No new entries after 15:20 (5 minutes before close)
        if current_time >= pd.to_datetime('15:20').time():
            continue
            
        # Check for trade exits first (for open trades)
        remaining_trades = []
        for trade_info in open_trades:
            exit_result = check_trade_exit(row, trade_info)
            if exit_result:
                trade = create_trade_result_from_exit(df, i, trade_info, exit_result)
                trades.append(trade)
                logger.info(f"Trade #{len(trades)} exited: {trade['direction']} {exit_result['type']} P&L: {trade['pl_points']:.1f}")
            else:
                # Update max favorable/adverse
                update_trade_excursions(row, trade_info)
                remaining_trades.append(trade_info)
        open_trades = remaining_trades
        
        # Find latest pivot high/low before current bar
        pivot_high = None
        pivot_high_time = None
        pivot_low = None
        pivot_low_time = None
        
        for pdt in sorted(pivot_highs.keys()):
            if pdt < dt:
                pivot_high = pivot_highs[pdt]
                pivot_high_time = pdt
            else:
                break
        for pdt in sorted(pivot_lows.keys()):
            if pdt < dt:
                pivot_low = pivot_lows[pdt]
                pivot_low_time = pdt
            else:
                break
                
        # Check for breakout (long) - only if pivot not already used
        if (pivot_high and pivot_high_time not in used_pivots and 
            row['high'] > pivot_high and row['close'] > row['open']):
            logger.info(f"Long breakout detected at {dt} - Pivot: {pivot_high:.1f}, High: {row['high']:.1f}")
            # Breakout candle found (long)
            trade_info = simulate_trade_setup(df, i, 'long', pivot_high, logger)
            if trade_info:
                # Trade setup pending entry
                trade_info['pivot_time'] = pivot_high_time
                open_trades.append(trade_info)
                used_pivots.add(pivot_high_time)
                logger.info(f"Long trade setup pending entry - Entry: {trade_info['entry_price']:.1f}, SL: {trade_info['sl_price']:.1f}")
                
        # Check for breakout (short) - only if pivot not already used
        if (pivot_low and pivot_low_time not in used_pivots and 
            row['low'] < pivot_low and row['close'] < row['open']):
            logger.info(f"Short breakout detected at {dt} - Pivot: {pivot_low:.1f}, Low: {row['low']:.1f}")
            # Breakout candle found (short)
            trade_info = simulate_trade_setup(df, i, 'short', pivot_low, logger)
            if trade_info:
                # Trade setup pending entry
                trade_info['pivot_time'] = pivot_low_time
                open_trades.append(trade_info)
                used_pivots.add(pivot_low_time)
                logger.info(f"Short trade setup pending entry - Entry: {trade_info['entry_price']:.1f}, SL: {trade_info['sl_price']:.1f}")
        
        # Check for entries on pending trades
        for trade_info in open_trades:
            if not trade_info['entered'] and check_entry_trigger(row, trade_info):
                trade_info['entered'] = True
                trade_info['entry_idx'] = i
                trade_info['entry_time'] = dt
                logger.info(f"Entry triggered: {trade_info['direction']} @ {trade_info['entry_price']:.1f}")
        
        # Update marking candles for pending trades
        for trade_info in open_trades:
            if not trade_info['entered']:
                update_marking_candle(row, trade_info, i, logger)
                
    # Close any remaining open trades at end of backtest
    if open_trades:
        logger.info(f"Closing {len(open_trades)} remaining open trades at end of backtest")
        for trade_info in open_trades:
            if trade_info['entered']:
                trade = close_trade_at_market_close(df, len(df)-1, trade_info)
                if trade:
                    trades.append(trade)
                    
    logger.info(f"Backtest completed. Total trades: {len(trades)}")
    return trades

def simulate_trade_setup(df, breakout_idx, direction, pivot_level, logger):
    """Modified version that returns trade info for tracking"""
    breakout_row = df.iloc[breakout_idx]
    breakout_time = breakout_row['datetime']
    
    # Find marking candle within next 5 bars
    marking_idx = None
    for j in range(1, 6):
        if breakout_idx + j >= len(df):
            break
        row = df.iloc[breakout_idx + j]
        # Opposite close direction and within breakout candle range
        if direction == 'long' and row['close'] < row['open']:
            if breakout_row['low'] <= row['close'] <= breakout_row['high']:
                marking_idx = breakout_idx + j
                logger.info(f"Marking candle found at bar {j} after breakout")
                break
        elif direction == 'short' and row['close'] > row['open']:
            if breakout_row['low'] <= row['close'] <= breakout_row['high']:
                marking_idx = breakout_idx + j
                logger.info(f"Marking candle found at bar {j} after breakout")
                break
    
    if marking_idx is None:
        logger.info("No valid marking candle found within 5 bars")
        return None  # No valid marking candle

    marking_row = df.iloc[marking_idx]
    initial_entry_price = marking_row['high'] if direction == 'long' else marking_row['low']
    initial_sl_price = marking_row['low'] if direction == 'long' else marking_row['high']
    
    # Check max SL distance (25 points limit)
    initial_sl_distance = abs(initial_entry_price - initial_sl_price)
    if initial_sl_distance > 25:
        logger.info(f"Trade skipped - SL distance {initial_sl_distance:.1f} exceeds 25 points")
        return None  # Skip trade if SL distance exceeds 25 points
    
    # Calculate final trade parameters
    rr = 2  # Risk-reward ratio
    tp_price = initial_entry_price + rr * initial_sl_distance if direction == 'long' else initial_entry_price - rr * initial_sl_distance
    
    return {
        'direction': direction,
        'breakout_idx': breakout_idx,
        'breakout_time': breakout_time,
        'marking_idx': marking_idx,
        'entry_price': initial_entry_price,
        'sl_price': initial_sl_price,
        'tp_price': tp_price,
        'pivot_level': pivot_level,
        'updates': 0,
        'max_favorable': 0,
        'max_adverse': 0,
        'entered': False,
        'entry_idx': None,
        'entry_time': None
    }

def check_entry_trigger(row, trade_info):
    """Check if entry is triggered"""
    if trade_info['direction'] == 'long':
        return row['high'] > trade_info['entry_price']
    else:
        return row['low'] < trade_info['entry_price']

def update_marking_candle(row, trade_info, current_idx, logger):
    """Update marking candle if conditions are met"""
    # Check if we're within 18 bars from breakout and under 3 updates
    bars_from_breakout = current_idx - trade_info['breakout_idx']
    if bars_from_breakout > 18 or trade_info['updates'] >= 3:
        return
    
    direction = trade_info['direction']
    
    # Update marking candle if opposite close and SL extension >= 1 unit
    if direction == 'long' and row['close'] < row['open']:
        if row['low'] < trade_info['sl_price'] - 1:  # SL extension check
            new_sl_distance = abs(trade_info['entry_price'] - row['low'])
            if new_sl_distance <= 25:  # Check 25-point limit
                logger.info(f"Marking candle updated (Long) - New SL: {row['low']:.1f}, New Entry: {row['high']:.1f}")
                trade_info['sl_price'] = row['low']
                trade_info['entry_price'] = row['high']
                trade_info['updates'] += 1
                # Recalculate TP
                sl_distance = abs(trade_info['entry_price'] - trade_info['sl_price'])
                trade_info['tp_price'] = trade_info['entry_price'] + 2 * sl_distance
    elif direction == 'short' and row['close'] > row['open']:
        if row['high'] > trade_info['sl_price'] + 1:  # SL extension check
            new_sl_distance = abs(row['high'] - trade_info['entry_price'])
            if new_sl_distance <= 25:  # Check 25-point limit
                logger.info(f"Marking candle updated (Short) - New SL: {row['high']:.1f}, New Entry: {row['low']:.1f}")
                trade_info['sl_price'] = row['high']
                trade_info['entry_price'] = row['low']
                trade_info['updates'] += 1
                # Recalculate TP
                sl_distance = abs(trade_info['entry_price'] - trade_info['sl_price'])
                trade_info['tp_price'] = trade_info['entry_price'] - 2 * sl_distance

def update_trade_excursions(row, trade_info):
    """Update max favorable and adverse excursions for entered trades"""
    if not trade_info['entered']:
        return
        
    entry_price = trade_info['entry_price']
    direction = trade_info['direction']
    
    if direction == 'long':
        current_pl = row['high'] - entry_price  # Best case
        current_adverse = entry_price - row['low']  # Worst case
    else:
        current_pl = entry_price - row['low']  # Best case
        current_adverse = row['high'] - entry_price  # Worst case
    
    trade_info['max_favorable'] = max(trade_info['max_favorable'], current_pl)
    trade_info['max_adverse'] = max(trade_info['max_adverse'], current_adverse)

def check_trade_exit(row, trade_info):
    """Check if trade should exit at SL or TP"""
    if not trade_info['entered']:
        return None
        
    direction = trade_info['direction']
    sl_price = trade_info['sl_price']
    tp_price = trade_info['tp_price']
    
    if direction == 'long':
        if row['low'] <= sl_price:
            return {'type': 'SL', 'price': sl_price}
        if row['high'] >= tp_price:
            return {'type': 'TP', 'price': tp_price}
    else:
        if row['high'] >= sl_price:
            return {'type': 'SL', 'price': sl_price}
        if row['low'] <= tp_price:
            return {'type': 'TP', 'price': tp_price}
    
    return None

def create_trade_result_from_exit(df, exit_idx, trade_info, exit_result):
    """Create final trade result when exit is triggered"""
    entry_time = trade_info['entry_time']
    exit_time = df.iloc[exit_idx]['datetime']
    time_in_trade = exit_time - entry_time
    
    entry_price = trade_info['entry_price']
    exit_price = exit_result['price']
    
    # Calculate P&L
    if trade_info['direction'] == 'long':
        pl_points = exit_price - entry_price
    else:
        pl_points = entry_price - exit_price
    
    sl_distance = abs(entry_price - trade_info['sl_price'])
    
    return {
        'trade_num': None,  # Will be set later
        'breakout_time': trade_info['breakout_time'],
        'entry_time': entry_time,
        'exit_time': exit_time,
        'direction': trade_info['direction'],
        'pivot_level': trade_info['pivot_level'],
        'entry_price': entry_price,
        'sl_price': trade_info['sl_price'],
        'tp_price': trade_info['tp_price'],
        'exit_price': exit_price,
        'result': exit_result['type'],
        'pl_points': pl_points,
        'sl_distance': sl_distance,
        'rr_achieved': pl_points / sl_distance if sl_distance > 0 else 0,
        'max_favorable': trade_info['max_favorable'],
        'max_adverse': trade_info['max_adverse'],
        'max_rr_potential': trade_info['max_favorable'] / sl_distance if sl_distance > 0 else 0,
        'time_in_trade_minutes': time_in_trade.total_seconds() / 60,
        'marking_updates': trade_info['updates']
    }

def close_trade_at_market_close(df, close_idx, trade_info):
    """Close trade at market close (15:25)"""
    if not trade_info['entered']:
        return None  # Can't close a trade that never entered
        
    close_row = df.iloc[close_idx]
    close_price = close_row['close']  # Use closing price
    
    return create_trade_result_from_exit(df, close_idx, trade_info, 
                                       {'type': 'CLOSE', 'price': close_price})

def calculate_analytics(trades, df):
    if not trades:
        return {"message": "No trades to analyze"}
    
    # Basic trade statistics
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['result'] == 'TP')
    losses = total_trades - wins
    win_rate = wins / total_trades * 100
    
    # P&L statistics
    total_pl = sum(t['pl_points'] for t in trades)
    winning_trades = [t for t in trades if t['result'] == 'TP']
    losing_trades = [t for t in trades if t['result'] == 'SL']
    
    avg_win = np.mean([t['pl_points'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pl_points'] for t in losing_trades]) if losing_trades else 0
    profit_factor = (wins * avg_win) / abs(losses * avg_loss) if losses > 0 and avg_loss != 0 else float('inf')
    
    # Risk-reward statistics
    avg_rr = np.mean([t['rr_achieved'] for t in trades])
    max_rr = max([t['rr_achieved'] for t in trades])
    min_rr = min([t['rr_achieved'] for t in trades])
    
    # Drawdown and streak analysis
    cumulative_pl = np.cumsum([t['pl_points'] for t in trades])
    peak = np.maximum.accumulate(cumulative_pl)
    drawdown = peak - cumulative_pl
    max_drawdown = np.max(drawdown)
    
    # Win/loss streaks
    results = [1 if t['result'] == 'TP' else -1 for t in trades]
    streaks = []
    current_streak = 1
    for i in range(1, len(results)):
        if results[i] == results[i-1]:
            current_streak += 1
        else:
            streaks.append(current_streak * results[i-1])
            current_streak = 1
    streaks.append(current_streak * results[-1])
    
    max_win_streak = max([s for s in streaks if s > 0]) if any(s > 0 for s in streaks) else 0
    max_loss_streak = abs(min([s for s in streaks if s < 0])) if any(s < 0 for s in streaks) else 0
    
    # Time-based analysis
    first_trade = min(trades, key=lambda t: t['entry_time'])
    last_trade = max(trades, key=lambda t: t['exit_time'])
    total_days = (last_trade['exit_time'] - first_trade['entry_time']).days
    
    # CAGR calculation (assuming each point = 1 unit of currency)
    initial_capital = 100000  # Assume 1 lakh starting capital
    final_capital = initial_capital + total_pl
    if total_days > 0:
        cagr = ((final_capital / initial_capital) ** (365.25 / total_days) - 1) * 100
    else:
        cagr = 0
    
    # Directional analysis
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    
    analytics = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 2),
        'total_pl_points': round(total_pl, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 2),
        'avg_rr': round(avg_rr, 2),
        'max_rr': round(max_rr, 2),
        'min_rr': round(min_rr, 2),
        'max_drawdown': round(max_drawdown, 2),
        'max_win_streak': max_win_streak,
        'max_loss_streak': max_loss_streak,
        'cagr': round(cagr, 2),
        'total_days': total_days,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_win_rate': round(sum(1 for t in long_trades if t['result'] == 'TP') / len(long_trades) * 100, 2) if long_trades else 0,
        'short_win_rate': round(sum(1 for t in short_trades if t['result'] == 'TP') / len(short_trades) * 100, 2) if short_trades else 0,
        'avg_time_in_trade': round(np.mean([t['time_in_trade_minutes'] for t in trades]), 2),
        'max_favorable_avg': round(np.mean([t['max_favorable'] for t in trades]), 2),
        'max_adverse_avg': round(np.mean([t['max_adverse'] for t in trades]), 2)
    }
    
    return analytics

def print_trade_details(trades):
    print("\n" + "="*100)
    print("DETAILED TRADE ANALYSIS")
    print("="*100)
    
    headers = ['Trade#', 'Date', 'Dir', 'Entry', 'SL', 'TP', 'Exit', 'Result', 'P&L', 'R:R', 'MaxFav', 'Time(min)']
    print(f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<5} {headers[3]:<8} {headers[4]:<8} {headers[5]:<8} {headers[6]:<8} {headers[7]:<6} {headers[8]:<8} {headers[9]:<6} {headers[10]:<8} {headers[11]:<10}")
    print("-" * 100)
    
    for i, trade in enumerate(trades, 1):
        trade['trade_num'] = i
        date_str = trade['entry_time'].strftime('%Y-%m-%d')
        print(f"{i:<6} {date_str:<12} {trade['direction'][:4]:<5} {trade['entry_price']:<8.1f} {trade['sl_price']:<8.1f} {trade['tp_price']:<8.1f} {trade['exit_price']:<8.1f} {trade['result']:<6} {trade['pl_points']:<8.1f} {trade['rr_achieved']:<6.2f} {trade['max_favorable']:<8.1f} {trade['time_in_trade_minutes']:<10.0f}")

def print_analytics(analytics):
    print("\n" + "="*60)
    print("BACKTEST ANALYTICS SUMMARY")
    print("="*60)
    
    print(f"Total Trades: {analytics['total_trades']}")
    print(f"Wins: {analytics['wins']} | Losses: {analytics['losses']}")
    print(f"Win Rate: {analytics['win_rate']}%")
    print(f"Total P&L: {analytics['total_pl_points']} points")
    print(f"Average Win: {analytics['avg_win']} points")
    print(f"Average Loss: {analytics['avg_loss']} points")
    print(f"Profit Factor: {analytics['profit_factor']}")
    print(f"CAGR: {analytics['cagr']}%")
    print(f"Max Drawdown: {analytics['max_drawdown']} points")
    print(f"Max Win Streak: {analytics['max_win_streak']}")
    print(f"Max Loss Streak: {analytics['max_loss_streak']}")
    print(f"Average R:R: {analytics['avg_rr']}")
    print(f"Max R:R Achieved: {analytics['max_rr']}")
    print(f"Min R:R Achieved: {analytics['min_rr']}")
    print(f"Long Trades: {analytics['long_trades']} (Win Rate: {analytics['long_win_rate']}%)")
    print(f"Short Trades: {analytics['short_trades']} (Win Rate: {analytics['short_win_rate']}%)")
    print(f"Avg Time in Trade: {analytics['avg_time_in_trade']} minutes")
    print(f"Avg Max Favorable: {analytics['max_favorable_avg']} points")
    print(f"Avg Max Adverse: {analytics['max_adverse_avg']} points")
    print(f"Backtest Period: {analytics['total_days']} days")

def main():
    csv_path = 'NIFTY 50_minute_data.csv'
    print("="*60)
    print("5-MINUTE PIVOT BREAKOUT STRATEGY BACKTEST")
    print("="*60)
    print("Strategy Parameters:")
    print("- Pivots: 15,15 on 5-min candles")
    print("- Market Hours: 9:15 AM - 3:30 PM only")
    print("- Intraday Trading: No entries after 15:20, Close all at 15:25")
    print("- Max SL Distance: 25 points")
    print("- One trade per pivot (no reuse after entry)")
    print("- 18-bar limit from breakout candle")
    print("- 2:1 Risk-Reward ratio")
    print("- Detailed logging enabled")
    print("="*60)
    
    print("Loading data...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} rows from {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Check market session hours
    sample_times = df['datetime'].dt.time
    print(f"Market session: {sample_times.min()} to {sample_times.max()}")
    
    print("Calculating pivots...")
    pivots = calculate_pivots(df, leftBars=15, rightBars=15)  # Updated to 15,15
    print(f"Found {len(pivots)} pivots")
    
    print("Running backtest with progress tracking...")
    trades = backtest_strategy(df, pivots)
    
    if trades:
        print_trade_details(trades)
        analytics = calculate_analytics(trades, df)
        print_analytics(analytics)
        
        # Export to CSV
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('backtest_results.csv', index=False)
        print(f"\nTrade details exported to 'backtest_results.csv'")
        print(f"Detailed logs available in the console output above")
    else:
        print("No trades generated!")

if __name__ == '__main__':
    main()
