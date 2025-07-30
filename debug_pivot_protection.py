import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df.sort_values('date', inplace=True)
    df.rename(columns={'date': 'datetime'}, inplace=True)
    
    # Filter for regular market hours (9:15 AM to 3:30 PM)
    df['time'] = df['datetime'].dt.time
    market_start = pd.to_datetime('09:15').time()
    market_end = pd.to_datetime('15:30').time()
    df = df[(df['time'] >= market_start) & (df['time'] <= market_end)]
    df = df.drop('time', axis=1)
    
    return df

def calculate_pivots_simple(df, leftBars=15, rightBars=15):
    """Fast pivot calculation for small dataset"""
    pivots = []
    df_5min = df.resample('5min', on='datetime').agg({'high':'max', 'low':'min', 'open':'first', 'close':'last'})
    df_5min = df_5min.reset_index()
    
    for i in range(leftBars, len(df_5min)-rightBars):
        high = df_5min.loc[i, 'high']
        low = df_5min.loc[i, 'low']
        
        # Check for pivot high
        window_highs = df_5min.loc[i-leftBars:i+rightBars, 'high']
        if high == max(window_highs):
            pivots.append({'type':'high', 'price':high, 'datetime':df_5min.loc[i, 'datetime']})
        
        # Check for pivot low
        window_lows = df_5min.loc[i-leftBars:i+rightBars, 'low']
        if low == min(window_lows):
            pivots.append({'type':'low', 'price':low, 'datetime':df_5min.loc[i, 'datetime']})
    
    return pivots

def debug_correct_timing():
    """Debug with proper used_pivots logic to show single trade per pivot"""
    print("="*80)
    print("CORRECTED TIMING DEBUG - With Used Pivots Logic")
    print("="*80)
    
    # Load first 5000 rows only
    df = load_data('NIFTY 50_minute_data.csv')
    df_small = df.head(5000).copy()
    
    print(f"Data range: {df_small['datetime'].min()} to {df_small['datetime'].max()}")
    print(f"Total rows: {len(df_small)}")
    
    # Calculate pivots
    pivots = calculate_pivots_simple(df_small)
    pivot_highs = {p['datetime']: p['price'] for p in pivots if p['type'] == 'high'}
    pivot_lows = {p['datetime']: p['price'] for p in pivots if p['type'] == 'low'}
    
    print(f"Found {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows")
    
    # Track used pivots (like in main.py)
    used_pivots = set()
    active_trades = []
    trade_count = 0
    
    print(f"\n🔍 TRACING BREAKOUTS WITH PIVOT PROTECTION:")
    
    for i in range(len(df_small)):
        if trade_count >= 3:  # Only trace first 3 actual trades
            break
            
        row = df_small.iloc[i]
        dt = row['datetime']
        
        # Find latest pivot high before current bar
        pivot_high = None
        pivot_high_time = None
        for pdt in sorted(pivot_highs.keys()):
            if pdt < dt:
                pivot_high = pivot_highs[pdt]
                pivot_high_time = pdt
            else:
                break
        
        # Check for long breakout with proper pivot protection
        if (pivot_high and pivot_high_time not in used_pivots and 
            row['high'] > pivot_high and row['close'] > row['open']):
            
            trade_count += 1
            used_pivots.add(pivot_high_time)  # Mark pivot as used immediately
            
            print(f"\n🚀 TRADE #{trade_count} - LONG BREAKOUT at {dt}")
            print(f"├─ Pivot: {pivot_high:.2f} at {pivot_high_time}")
            print(f"├─ Breakout High: {row['high']:.2f}")
            print(f"├─ Used Pivots: {len(used_pivots)} pivot(s) now marked as used")
            
            # Simulate marking candle search
            breakout_range = (row['low'], row['high'])
            marking_found = False
            
            for j in range(1, 6):
                if i + j >= len(df_small):
                    break
                bar = df_small.iloc[i + j]
                is_red = bar['close'] < bar['open']
                in_range = breakout_range[0] <= bar['close'] <= breakout_range[1]
                
                if is_red and in_range:
                    print(f"├─ Marking candle found at bar {j}: {bar['datetime']}")
                    print(f"├─ Entry: {bar['high']:.2f}, SL: {bar['low']:.2f}")
                    marking_found = True
                    
                    # Store trade info
                    active_trades.append({
                        'breakout_idx': i,
                        'marking_idx': i + j,
                        'pivot_time': pivot_high_time,
                        'pivot_price': pivot_high
                    })
                    break
            
            if not marking_found:
                print(f"└─ No marking candle found - Trade abandoned")
            else:
                print(f"└─ Trade setup complete - Pivot {pivot_high_time} now BLOCKED for future breakouts")
        
        # Check if any subsequent bars try to break the SAME pivot
        elif (pivot_high and pivot_high_time in used_pivots and 
              row['high'] > pivot_high and row['close'] > row['open']):
            print(f"\n⚠️  BLOCKED BREAKOUT at {dt}")
            print(f"├─ Would break Pivot: {pivot_high:.2f} at {pivot_high_time}")
            print(f"├─ High: {row['high']:.2f} > {pivot_high:.2f}")
            print(f"└─ IGNORED - Pivot already used for previous trade")
    
    print(f"\n📊 SUMMARY:")
    print(f"├─ Total valid trades: {trade_count}")
    print(f"├─ Used pivots: {len(used_pivots)}")
    print(f"└─ This shows ONE trade per pivot (correct behavior)")

if __name__ == "__main__":
    debug_correct_timing()
