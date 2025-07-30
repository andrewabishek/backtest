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
    df_5min = df.resample('5min', on='datetime').agg({'high':'max', 'low':'min'})
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
    # Track state for marking candle logic (from validation script)
    marking_candle_active = False
    marking_direction = None
    marking_entry = 0
    marking_sl = 0
    marking_updates = 0
    marking_pivot = 0
    marking_breakout_idx = None  # Track breakout bar index for 5-bar window and 18-bar limit
    marking_breakout_high = 0  # Store breakout candle high for range validation
    marking_breakout_low = 0   # Store breakout candle low for range validation
    active_trade_info = None
    
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
            # Also close any active marking candle setup
            marking_candle_active = False
            active_trade_info = None
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
        
        # VALIDATION SCRIPT LOGIC - STEP 1: Check entry triggers FIRST (before any updates)
        if marking_candle_active and not active_trade_info:
            entry_triggered = False
            
            if marking_direction == 'long':
                if row['high'] > marking_entry:  # HIGH > ENTRY (not >=)
                    logger.info(f"Entry triggered: long @ {marking_entry:.1f}")
                    # Calculate TP based on current SL distance (TP = Entry + 2 * SL_distance)
                    sl_distance = abs(marking_entry - marking_sl)
                    tp_price = marking_entry + 2 * sl_distance
                    # Create trade info for tracking
                    active_trade_info = {
                        'direction': 'long',
                        'entry_price': marking_entry,
                        'sl_price': marking_sl,
                        'tp_price': tp_price,
                        'entry_time': dt,
                        'entry_idx': i,
                        'updates': marking_updates,
                        'entered': True,
                        'max_favorable': 0,
                        'max_adverse': 0,
                        'pivot_level': marking_pivot,
                        'breakout_time': dt  # Use entry time as placeholder
                    }
                    open_trades.append(active_trade_info)
                    entry_triggered = True
                    
            elif marking_direction == 'short':
                if row['low'] < marking_entry:  # LOW < ENTRY (not <=)
                    logger.info(f"Entry triggered: short @ {marking_entry:.1f}")
                    # Calculate TP based on current SL distance (TP = Entry - 2 * SL_distance)
                    sl_distance = abs(marking_sl - marking_entry)
                    tp_price = marking_entry - 2 * sl_distance
                    # Create trade info for tracking
                    active_trade_info = {
                        'direction': 'short',
                        'entry_price': marking_entry,
                        'sl_price': marking_sl,
                        'tp_price': tp_price,
                        'entry_time': dt,
                        'entry_idx': i,
                        'updates': marking_updates,
                        'entered': True,
                        'max_favorable': 0,
                        'max_adverse': 0,
                        'pivot_level': marking_pivot,
                        'breakout_time': dt  # Use entry time as placeholder
                    }
                    open_trades.append(active_trade_info)
                    entry_triggered = True
            
            # If entry triggered, stop processing this bar for marking candle
            if entry_triggered:
                marking_candle_active = False
                marking_breakout_idx = None  # Reset breakout tracking
                continue
        
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
        
        # VALIDATION SCRIPT LOGIC - STEP 2: Check for breakouts (only if no active trade)
        if not active_trade_info and not marking_candle_active:
            # Check for breakout (long) - only if pivot not already used
            if (pivot_high and pivot_high_time not in used_pivots and 
                row['high'] > pivot_high and row['close'] > row['open']):
                logger.info(f"Long breakout detected at {dt} - Pivot: {pivot_high:.1f}, High: {row['high']:.1f}")
                # Set up marking direction
                marking_direction = 'long'
                marking_pivot = pivot_high
                marking_breakout_idx = i  # Store breakout bar index for windows
                marking_breakout_high = row['high']  # Store breakout range
                marking_breakout_low = row['low']    # Store breakout range
                used_pivots.add(pivot_high_time)
                
            # Check for breakout (short) - only if pivot not already used
            elif (pivot_low and pivot_low_time not in used_pivots and 
                  row['low'] < pivot_low and row['close'] < row['open']):
                logger.info(f"Short breakout detected at {dt} - Pivot: {pivot_low:.1f}, Low: {row['low']:.1f}")
                # Set up marking direction
                marking_direction = 'short'
                marking_pivot = pivot_low
                marking_breakout_idx = i  # Store breakout bar index for windows
                marking_breakout_high = row['high']  # Store breakout range
                marking_breakout_low = row['low']    # Store breakout range
                used_pivots.add(pivot_low_time)
        
        # VALIDATION SCRIPT LOGIC - STEP 3: Check for marking candle pattern (within 5 bars of breakout)
        if marking_direction and not marking_candle_active and not active_trade_info and marking_breakout_idx is not None:
            bars_since_breakout = i - marking_breakout_idx
            
            # Only search for marking candle within NEXT 5 bars after breakout (not same bar)
            if 1 <= bars_since_breakout <= 5:
                if marking_direction == 'long' and row['close'] < row['open']:  # RED candle for LONG
                    # Check if close is within breakout candle range
                    if marking_breakout_low <= row['close'] <= marking_breakout_high:
                        logger.info(f"Marking candle found for LONG trade (bar {bars_since_breakout} after breakout)")
                        marking_entry = row['high']
                        marking_sl = row['low']
                        
                        # Check max SL distance (25 points limit)
                        sl_distance = abs(marking_entry - marking_sl)
                        if sl_distance <= 25:
                            marking_candle_active = True
                            marking_updates = 0
                            logger.info(f"Long trade setup pending entry - Entry: {marking_entry:.1f}, SL: {marking_sl:.1f}")
                        else:
                            logger.info(f"Trade skipped - SL distance {sl_distance:.1f} exceeds 25 points")
                            marking_direction = None
                            marking_breakout_idx = None
                    else:
                        logger.info(f"Red candle found but close {row['close']:.1f} not in breakout range [{marking_breakout_low:.1f}-{marking_breakout_high:.1f}]")
                    
                elif marking_direction == 'short' and row['close'] > row['open']:  # GREEN candle for SHORT
                    # Check if close is within breakout candle range
                    if marking_breakout_low <= row['close'] <= marking_breakout_high:
                        logger.info(f"Marking candle found for SHORT trade (bar {bars_since_breakout} after breakout)")
                        marking_entry = row['low']
                        marking_sl = row['high']
                        
                        # Check max SL distance (25 points limit)
                        sl_distance = abs(marking_sl - marking_entry)
                        if sl_distance <= 25:
                            marking_candle_active = True
                            marking_updates = 0
                            logger.info(f"Short trade setup pending entry - Entry: {marking_entry:.1f}, SL: {marking_sl:.1f}")
                        else:
                            logger.info(f"Trade skipped - SL distance {sl_distance:.1f} exceeds 25 points")
                            marking_direction = None
                            marking_breakout_idx = None
                    else:
                        logger.info(f"Green candle found but close {row['close']:.1f} not in breakout range [{marking_breakout_low:.1f}-{marking_breakout_high:.1f}]")
            elif bars_since_breakout > 5:
                # No marking candle found within 5 bars, reset and wait for next breakout
                logger.info(f"No marking candle found within 5 bars after breakout, resetting setup")
                marking_direction = None
                marking_breakout_idx = None
        
        # VALIDATION SCRIPT LOGIC - STEP 4: Update marking levels (only if no entry triggered this bar)
        elif marking_candle_active and not active_trade_info and marking_updates < 3 and marking_breakout_idx is not None:
            bars_from_breakout = i - marking_breakout_idx
            
            # Only allow updates within 18 bars from breakout
            if bars_from_breakout <= 18:
                old_entry = marking_entry
                old_sl = marking_sl
                
                if marking_direction == 'long':
                    # Check if SL needs to be extended (low goes below current SL - 1)
                    if row['low'] < marking_sl - 1:
                        new_entry = row['high']
                        new_sl = row['low']
                        new_sl_distance = abs(new_entry - new_sl)
                        
                        if new_sl_distance <= 25:  # Check 25-point limit
                            marking_entry = new_entry
                            marking_sl = new_sl
                            marking_updates += 1
                            candle_color = "Red" if row['close'] < row['open'] else "Green"
                            logger.info(f"Marking candle updated (Long {candle_color}) - Old Entry: {old_entry:.1f}, New Entry: {marking_entry:.1f}, Old SL: {old_sl:.1f}, New SL: {marking_sl:.1f}")
                        
                elif marking_direction == 'short':
                    # Check if SL needs to be extended (high goes above current SL + 1)
                    if row['high'] > marking_sl + 1:
                        new_entry = row['low']
                        new_sl = row['high']
                        new_sl_distance = abs(new_sl - new_entry)
                        
                        if new_sl_distance <= 25:  # Check 25-point limit
                            marking_entry = new_entry
                            marking_sl = new_sl
                            marking_updates += 1
                            candle_color = "Red" if row['close'] < row['open'] else "Green"
                            logger.info(f"Marking candle updated (Short {candle_color}) - Old Entry: {old_entry:.1f}, New Entry: {marking_entry:.1f}, Old SL: {old_sl:.1f}, New SL: {marking_sl:.1f}")
            else:
                # 18-bar limit reached, expire the marking setup
                logger.info(f"18-bar limit reached from breakout, expiring marking setup")
                marking_candle_active = False
                marking_direction = None
                marking_breakout_idx = None
                
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

# REMOVED: simulate_trade_setup() and related functions now replaced with inline validation logic
# The bar-by-bar processing loop now contains the proven validation script logic

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

def calculate_yearly_analytics(trades, df):
    """Calculate comprehensive year-wise analytics"""
    if not trades:
        return {}
    
    # Convert trades to DataFrame for easier manipulation
    trades_df = pd.DataFrame(trades)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['year'] = trades_df['entry_time'].dt.year
    
    # Get unique years
    years = sorted(trades_df['year'].unique())
    yearly_stats = {}
    
    print(f"\n{'='*80}")
    print("YEAR-WISE DETAILED ANALYTICS")
    print(f"{'='*80}")
    
    # Overall period stats
    start_date = trades_df['entry_time'].min()
    end_date = trades_df['entry_time'].max()
    total_days = (end_date - start_date).days
    
    print(f"Backtest Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({total_days} days)")
    print(f"Years Covered: {', '.join(map(str, years))}")
    print(f"{'='*80}")
    
    cumulative_pl = 0
    starting_capital = 100000  # Assuming starting capital for CAGR calculation
    
    for year in years:
        year_trades = trades_df[trades_df['year'] == year]
        
        if len(year_trades) == 0:
            continue
            
        # Basic stats
        total_trades = len(year_trades)
        wins = len(year_trades[year_trades['pl_points'] > 0])
        losses = len(year_trades[year_trades['pl_points'] < 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # P&L calculations
        year_pl = year_trades['pl_points'].sum()
        cumulative_pl += year_pl
        
        # Win/Loss analysis
        winning_trades = year_trades[year_trades['pl_points'] > 0]['pl_points']
        losing_trades = year_trades[year_trades['pl_points'] < 0]['pl_points']
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        max_win = winning_trades.max() if len(winning_trades) > 0 else 0
        max_loss = losing_trades.min() if len(losing_trades) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown calculation for the year
        year_trades_sorted = year_trades.sort_values('entry_time')
        running_pl = year_trades_sorted['pl_points'].cumsum()
        peak = running_pl.cummax()
        drawdown = peak - running_pl
        max_drawdown = drawdown.max()
        
        # Calculate year CAGR (assuming each point = 1 unit of currency)
        year_start_capital = starting_capital + (cumulative_pl - year_pl)
        year_end_capital = starting_capital + cumulative_pl
        year_return = (year_end_capital - year_start_capital) / year_start_capital * 100
        
        # Streak analysis
        win_streak = 0
        loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for _, trade in year_trades_sorted.iterrows():
            if trade['pl_points'] > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        # Trading frequency
        year_start = pd.Timestamp(f'{year}-01-01')
        year_end = pd.Timestamp(f'{year}-12-31')
        trading_days_in_year = len(pd.bdate_range(year_start, year_end))
        trades_per_month = total_trades / 12
        
        yearly_stats[year] = {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'year_pl': year_pl,
            'cumulative_pl': cumulative_pl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'year_return': year_return,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'trades_per_month': trades_per_month,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
        
        # Print year summary
        print(f"Year {year}:")
        print(f"  Trades: {total_trades:>6} | Wins: {wins:>3} | Losses: {losses:>3} | Win Rate: {win_rate:>5.1f}%")
        print(f"  P&L: {year_pl:>8.1f} pts | Cumulative: {cumulative_pl:>8.1f} pts | Return: {year_return:>6.1f}%")
        print(f"  Avg Win: {avg_win:>6.1f} | Avg Loss: {avg_loss:>6.1f} | Profit Factor: {profit_factor:>5.2f}")
        print(f"  Max DD: {max_drawdown:>6.1f} pts | Win Streak: {max_win_streak:>2} | Loss Streak: {max_loss_streak:>2}")
        print(f"  Best Trade: {max_win:>6.1f} | Worst Trade: {max_loss:>6.1f} | Monthly Avg: {trades_per_month:>4.1f}")
        print("-" * 80)
    
    # Calculate overall CAGR
    total_years = len(years)
    if total_years > 0 and cumulative_pl != 0:
        final_capital = starting_capital + cumulative_pl
        overall_cagr = ((final_capital / starting_capital) ** (1/total_years) - 1) * 100
    else:
        overall_cagr = 0
    
    # Summary statistics
    print(f"SUMMARY STATISTICS:")
    print(f"  Total Years: {total_years}")
    print(f"  Overall CAGR: {overall_cagr:>6.2f}%")
    print(f"  Best Year: {max(yearly_stats.values(), key=lambda x: x['year_return'])['year_return'] if yearly_stats else 0:>6.1f}% (Year {max(yearly_stats.keys(), key=lambda x: yearly_stats[x]['year_return']) if yearly_stats else 'N/A'})")
    print(f"  Worst Year: {min(yearly_stats.values(), key=lambda x: x['year_return'])['year_return'] if yearly_stats else 0:>6.1f}% (Year {min(yearly_stats.keys(), key=lambda x: yearly_stats[x]['year_return']) if yearly_stats else 'N/A'})")
    print(f"  Profitable Years: {sum(1 for stats in yearly_stats.values() if stats['year_pl'] > 0)}/{total_years}")
    print(f"  Average Annual Trades: {sum(stats['total_trades'] for stats in yearly_stats.values()) / total_years if total_years > 0 else 0:.1f}")
    
    return yearly_stats

def print_monthly_heatmap(trades):
    """Print a monthly performance heatmap"""
    if not trades:
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df['year'] = trades_df['entry_time'].dt.year
    trades_df['month'] = trades_df['entry_time'].dt.month
    
    # Create monthly P&L matrix
    monthly_pl = trades_df.groupby(['year', 'month'])['pl_points'].sum().reset_index()
    
    print(f"\n{'='*100}")
    print("MONTHLY P&L HEATMAP (Points)")
    print(f"{'='*100}")
    
    years = sorted(trades_df['year'].unique())
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Header
    print(f"{'Year':<6}", end='')
    for month in months:
        print(f"{month:>8}", end='')
    print(f"{'Total':>10}")
    print("-" * 100)
    
    for year in years:
        print(f"{year:<6}", end='')
        year_total = 0
        for month_num in range(1, 13):
            month_data = monthly_pl[(monthly_pl['year'] == year) & (monthly_pl['month'] == month_num)]
            month_pl = month_data['pl_points'].iloc[0] if len(month_data) > 0 else 0
            year_total += month_pl
            
            # Color coding for terminal (basic)
            if month_pl > 0:
                print(f"{month_pl:>8.1f}", end='')
            elif month_pl < 0:
                print(f"{month_pl:>8.1f}", end='')
            else:
                print(f"{'—':>8}", end='')
        
        print(f"{year_total:>10.1f}")
    
    print(f"{'='*100}")

def print_drawdown_analysis(trades):
    """Print detailed drawdown analysis"""
    if not trades:
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
    trades_df = trades_df.sort_values('entry_time')
    
    # Calculate running P&L and drawdowns
    trades_df['cumulative_pl'] = trades_df['pl_points'].cumsum()
    trades_df['peak'] = trades_df['cumulative_pl'].cummax()
    trades_df['drawdown'] = trades_df['peak'] - trades_df['cumulative_pl']
    
    print(f"\n{'='*80}")
    print("DRAWDOWN ANALYSIS")
    print(f"{'='*80}")
    
    max_dd = trades_df['drawdown'].max()
    max_dd_idx = trades_df['drawdown'].idxmax()
    max_dd_trade = trades_df.loc[max_dd_idx]
    
    # Find drawdown periods
    in_drawdown = trades_df['drawdown'] > 0
    drawdown_periods = []
    
    start_idx = None
    for idx, is_dd in enumerate(in_drawdown):
        if is_dd and start_idx is None:
            start_idx = idx
        elif not is_dd and start_idx is not None:
            drawdown_periods.append((start_idx, idx - 1))
            start_idx = None
    
    # If still in drawdown at the end
    if start_idx is not None:
        drawdown_periods.append((start_idx, len(trades_df) - 1))
    
    print(f"Maximum Drawdown: {max_dd:.1f} points")
    print(f"Max DD Date: {max_dd_trade['entry_time'].strftime('%Y-%m-%d')}")
    print(f"Max DD Peak: {max_dd_trade['peak']:.1f} points")
    print(f"Number of Drawdown Periods: {len(drawdown_periods)}")
    
    if drawdown_periods:
        print(f"\nTop 5 Drawdown Periods:")
        print(f"{'Start':<12} {'End':<12} {'Duration':<10} {'Peak':<8} {'Trough':<8} {'DD Points':<10} {'Recovery':<10}")
        print("-" * 80)
        
        # Sort by drawdown magnitude
        dd_details = []
        for start_idx, end_idx in drawdown_periods:
            period_trades = trades_df.iloc[start_idx:end_idx+1]
            start_date = period_trades.iloc[0]['entry_time']
            end_date = period_trades.iloc[-1]['entry_time']
            duration = (end_date - start_date).days
            peak_value = period_trades.iloc[0]['peak']
            trough_value = period_trades['cumulative_pl'].min()
            dd_magnitude = peak_value - trough_value
            
            # Check if recovered
            if end_idx < len(trades_df) - 1:
                recovery_status = "Yes"
            else:
                recovery_status = "Ongoing"
            
            dd_details.append({
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration,
                'peak': peak_value,
                'trough': trough_value,
                'magnitude': dd_magnitude,
                'recovery': recovery_status
            })
        
        # Sort by magnitude and show top 5
        dd_details.sort(key=lambda x: x['magnitude'], reverse=True)
        for dd in dd_details[:5]:
            print(f"{dd['start_date'].strftime('%Y-%m-%d'):<12} "
                  f"{dd['end_date'].strftime('%Y-%m-%d'):<12} "
                  f"{dd['duration']:<10} "
                  f"{dd['peak']:<8.1f} "
                  f"{dd['trough']:<8.1f} "
                  f"{dd['magnitude']:<10.1f} "
                  f"{dd['recovery']:<10}")

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
        
        # Add comprehensive yearly analytics
        yearly_stats = calculate_yearly_analytics(trades, df)
        print_monthly_heatmap(trades)
        print_drawdown_analysis(trades)
        
        # Export to CSV with additional analytics
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv('backtest_results.csv', index=False)
        
        # Export yearly statistics
        if yearly_stats:
            yearly_df = pd.DataFrame.from_dict(yearly_stats, orient='index')
            yearly_df.to_csv('yearly_analytics.csv')
            print(f"\nYearly analytics exported to 'yearly_analytics.csv'")
        
        print(f"Trade details exported to 'backtest_results.csv'")
        print(f"Detailed logs available in the console output above")
    else:
        print("No trades generated!")

if __name__ == '__main__':
    main()
