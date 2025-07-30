import pandas as pd
import numpy as np

def detailed_trade_analysis():
    """Detailed step-by-step analysis of specific trades"""
    
    print("="*80)
    print("DETAILED TRADE ANALYSIS - POTENTIAL ISSUES IDENTIFIED")
    print("="*80)
    
    # Load the data
    df = pd.read_csv('NIFTY 50_minute_data.csv', parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.rename(columns={'date': 'datetime'}, inplace=True)
    
    # Filter for regular market hours
    df['time'] = df['datetime'].dt.time
    market_start = pd.to_datetime('09:15').time()
    market_end = pd.to_datetime('15:30').time()
    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
    df = df.drop('time', axis=1)
    
    print("ISSUE 1: MARKING CANDLE LOGIC")
    print("-" * 40)
    
    # Analyze Trade 1 in detail
    print("Trade 1 Analysis:")
    print("- Breakout bar (14:52): 8268.10 / 8275.15 / 8267.50 / 8275.15")
    print("- Breakout range: 8267.50 - 8275.15")
    print("- Marking candle (14:55): 8275.05 / 8275.95 / 8269.65 / 8269.95")
    print("- Close (8269.95) is within range ✓")
    print("- Red candle ✓") 
    print("- Entry: High = 8275.95")
    print("- SL: Low = 8269.65")
    print("- But trade shows Entry: 8275.95, SL: 8269.65 - MATCHES!")
    
    print("\nChecking entry trigger...")
    # Look at next bar after marking candle
    breakout_time = pd.to_datetime('2015-01-09 14:52:00')
    marking_time = pd.to_datetime('2015-01-09 14:55:00')
    
    # Get bars after marking candle
    idx = df[df['datetime'] == marking_time].index[0]
    entry_bar = df.iloc[idx]  # Same bar or next bar?
    
    print(f"Marking candle: {entry_bar['datetime']} - High: {entry_bar['high']}")
    print(f"Entry price should be: {entry_bar['high']}")
    print(f"Entry triggered when high > {entry_bar['high']}?")
    
    # Check next few bars
    for i in range(1, 4):
        if idx + i < len(df):
            next_bar = df.iloc[idx + i]
            print(f"Bar {i} after marking ({next_bar['datetime']}): High {next_bar['high']:.2f}")
            if next_bar['high'] >= entry_bar['high']:
                print(f"  -> Entry would trigger here! But actual entry time was 14:53...")
    
    print("\n" + "="*50)
    print("ISSUE 2: ENTRY TIMING DISCREPANCY")
    print("-" * 40)
    
    print("From backtest results:")
    print("- Breakout: 14:52")
    print("- Entry: 14:53 (but marking candle is at 14:55)")
    print("- This suggests entry happened BEFORE marking candle was found!")
    
    print("\nPOSSIBLE BUG: Entry might be triggering immediately on breakout")
    print("rather than waiting for marking candle + entry trigger")
    
    print("\n" + "="*50)
    print("ISSUE 3: WIN RATE ANALYSIS")
    print("-" * 40)
    
    # Load results
    results = pd.read_csv('backtest_results.csv')
    
    # Check SL distances
    sl_distances = results['sl_distance']
    print(f"SL Distance Statistics:")
    print(f"- Mean: {sl_distances.mean():.2f}")
    print(f"- Median: {sl_distances.median():.2f}")
    print(f"- Max: {sl_distances.max():.2f}")
    print(f"- Min: {sl_distances.min():.2f}")
    print(f"- % with SL > 20 points: {(sl_distances > 20).mean()*100:.1f}%")
    
    # Check R:R ratios
    winning_trades = results[results['result'] == 'TP']
    losing_trades = results[results['result'] == 'SL']
    
    print(f"\nTrade Distribution:")
    print(f"- Total: {len(results)}")
    print(f"- TP: {len(winning_trades)} ({len(winning_trades)/len(results)*100:.1f}%)")
    print(f"- SL: {len(losing_trades)} ({len(losing_trades)/len(results)*100:.1f}%)")
    print(f"- Other: {len(results) - len(winning_trades) - len(losing_trades)}")
    
    # Look at time in trade
    print(f"\nTime in Trade:")
    print(f"- Mean: {results['time_in_trade_minutes'].mean():.1f} minutes")
    print(f"- Median: {results['time_in_trade_minutes'].median():.1f} minutes")
    print(f"- % trades < 5 min: {(results['time_in_trade_minutes'] < 5).mean()*100:.1f}%")
    
    print("\n" + "="*50)
    print("ISSUE 4: MARKING CANDLE UPDATE FREQUENCY")
    print("-" * 40)
    
    updates = results['marking_updates']
    print(f"Marking Updates:")
    print(f"- Mean: {updates.mean():.2f}")
    print(f"- 0 updates: {(updates == 0).sum()} trades ({(updates == 0).mean()*100:.1f}%)")
    print(f"- 1 update: {(updates == 1).sum()} trades ({(updates == 1).mean()*100:.1f}%)")
    print(f"- 2 updates: {(updates == 2).sum()} trades ({(updates == 2).mean()*100:.1f}%)")
    print(f"- 3 updates: {(updates == 3).sum()} trades ({(updates == 3).mean()*100:.1f}%)")
    
    print("\n" + "="*50)
    print("POTENTIAL ROOT CAUSES:")
    print("-" * 40)
    print("1. Entry timing: Entry may be triggering too early")
    print("2. Marking candle logic: May need refinement")
    print("3. SL too tight: Many small SL distances causing quick hits")
    print("4. Market noise: 1-min data very noisy for this strategy")
    print("5. Pivot quality: Need better pivot filtering")
    
    # Check specific patterns
    print("\n" + "="*50)
    print("PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Look at trades with 0 marking updates (original marking candle)
    original_trades = results[results['marking_updates'] == 0]
    print(f"\nTrades with original marking candle (0 updates): {len(original_trades)}")
    print(f"Win rate: {(original_trades['result'] == 'TP').mean()*100:.1f}%")
    print(f"Avg SL distance: {original_trades['sl_distance'].mean():.2f}")
    
    # Look at trades with updates
    updated_trades = results[results['marking_updates'] > 0]
    print(f"\nTrades with marking updates (>0 updates): {len(updated_trades)}")
    print(f"Win rate: {(updated_trades['result'] == 'TP').mean()*100:.1f}%")
    print(f"Avg SL distance: {updated_trades['sl_distance'].mean():.2f}")

if __name__ == "__main__":
    detailed_trade_analysis()
