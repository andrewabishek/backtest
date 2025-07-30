import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the data
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

def debug_entry_timing():
    """Debug the entry timing issue with small dataset"""
    print("="*80)
    print("ENTRY TIMING DEBUG - First 5000 rows")
    print("="*80)
    
    # Load first 5000 rows only
    df = load_data('NIFTY 50_minute_data.csv')
    df_small = df.head(5000).copy()
    
    print(f"Data range: {df_small['datetime'].min()} to {df_small['datetime'].max()}")
    print(f"Total rows: {len(df_small)}")
    
    # Calculate pivots on small dataset
    pivots = calculate_pivots_simple(df_small)
    pivot_highs = [p for p in pivots if p['type'] == 'high']
    pivot_lows = [p for p in pivots if p['type'] == 'low']
    
    print(f"Found {len(pivot_highs)} pivot highs and {len(pivot_lows)} pivot lows")
    
    # Look for the first few breakouts and trace their exact timing
    print(f"\n🔍 TRACING FIRST FEW BREAKOUTS:")
    
    breakout_count = 0
    
    for i in range(len(df_small)):
        if breakout_count >= 3:  # Only trace first 3 breakouts
            break
            
        row = df_small.iloc[i]
        dt = row['datetime']
        
        # Check for pivot high breakouts
        for pivot in pivot_highs:
            if dt <= pivot['datetime']:
                continue  # Can't trade on pivot that hasn't formed yet
                
            if row['high'] > pivot['price']:  # Breakout detected
                breakout_count += 1
                print(f"\n🚀 BREAKOUT #{breakout_count} - LONG at {dt}")
                print(f"├─ Pivot: {pivot['price']:.2f} at {pivot['datetime']}")
                print(f"├─ Breakout High: {row['high']:.2f}")
                print(f"├─ Breakout Range: {row['low']:.2f} - {row['high']:.2f}")
                
                # Simulate the strategy step by step
                breakout_idx = i
                
                # Step 1: Find marking candle
                marking_found = False
                marking_idx = None
                for j in range(1, 6):
                    if breakout_idx + j >= len(df_small):
                        break
                    bar = df_small.iloc[breakout_idx + j]
                    is_red = bar['close'] < bar['open']
                    in_range = row['low'] <= bar['close'] <= row['high']
                    
                    if is_red and in_range:
                        marking_idx = breakout_idx + j
                        marking_found = True
                        print(f"├─ Marking candle found at bar {j}: {bar['datetime']}")
                        print(f"├─ Initial Entry: {bar['high']:.2f}, Initial SL: {bar['low']:.2f}")
                        break
                
                if not marking_found:
                    print(f"└─ No marking candle found")
                    continue
                
                # Step 2: Simulate marking candle updates and entry checks
                marking_bar = df_small.iloc[marking_idx]
                current_entry = marking_bar['high']
                current_sl = marking_bar['low']
                updates = 0
                entered = False
                entry_bar = None
                
                print(f"├─ Simulating bar-by-bar updates and entry checks:")
                
                for k in range(marking_idx + 1, min(breakout_idx + 19, len(df_small))):
                    bar = df_small.iloc[k]
                    bars_from_breakout = k - breakout_idx
                    
                    print(f"│   Bar {bars_from_breakout} ({bar['datetime'].strftime('%H:%M')}): {bar['open']:.1f}/{bar['high']:.1f}/{bar['low']:.1f}/{bar['close']:.1f}")
                    
                    # Check for marking candle update FIRST
                    updated = False
                    if (bars_from_breakout <= 18 and updates < 3 and 
                        bar['low'] < current_sl - 1 and 
                        abs(bar['high'] - bar['low']) <= 25):
                        
                        old_entry = current_entry
                        old_sl = current_sl
                        current_entry = bar['high']
                        current_sl = bar['low']
                        updates += 1
                        updated = True
                        
                        candle_color = "Red" if bar['close'] < bar['open'] else "Green"
                        print(f"│     ✓ UPDATE ({candle_color}): Entry {old_entry:.1f}→{current_entry:.1f}, SL {old_sl:.1f}→{current_sl:.1f}")
                    
                    # Check for entry trigger AFTER update
                    if not entered and bar['high'] > current_entry:
                        entered = True
                        entry_bar = k
                        print(f"│     🎯 ENTRY TRIGGERED! High {bar['high']:.1f} > Entry {current_entry:.1f}")
                        print(f"│     Entry Time: {bar['datetime']}")
                        
                        if updated:
                            print(f"│     ⚠️  WARNING: Entry triggered on SAME bar as update!")
                        break
                    
                    if not updated and not entered:
                        print(f"│     - No update, no entry")
                
                if entered:
                    bars_to_entry = entry_bar - breakout_idx
                    print(f"└─ Entry after {bars_to_entry} bars from breakout")
                else:
                    print(f"└─ No entry triggered within 18 bars")
                
                print(f"   Final levels: Entry={current_entry:.2f}, SL={current_sl:.2f}, Updates={updates}")
                
                # Exit after first breakout for detailed analysis
                if breakout_count == 1:
                    break
                    
        # Check for pivot low breakouts (similar logic)
        for pivot in pivot_lows:
            if dt <= pivot['datetime']:
                continue
                
            if row['low'] < pivot['price']:  # Short breakout
                breakout_count += 1
                print(f"\n🚀 BREAKOUT #{breakout_count} - SHORT at {dt}")
                print(f"├─ Pivot: {pivot['price']:.2f} at {pivot['datetime']}")
                print(f"├─ Breakout Low: {row['low']:.2f}")
                
                # Similar logic for short trades...
                break

if __name__ == "__main__":
    debug_entry_timing()
