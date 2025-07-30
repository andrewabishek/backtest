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

def calculate_pivots_debug(df, leftBars=15, rightBars=15):
    """Calculate pivots with detailed logging"""
    pivots = []
    df_5min = df.resample('5min', on='datetime').agg({'high':'max', 'low':'min', 'open':'first', 'close':'last'})
    df_5min = df_5min.reset_index()
    
    print(f"Total 5-min candles: {len(df_5min)}")
    print(f"First few 5-min candles:")
    print(df_5min.head(10))
    
    for i in range(leftBars, len(df_5min)-rightBars):
        high = df_5min.loc[i, 'high']
        low = df_5min.loc[i, 'low']
        
        # Check for pivot high
        window_highs = df_5min.loc[i-leftBars:i+rightBars, 'high']
        if high == max(window_highs):
            pivot = {'type':'high', 'price':high, 'datetime':df_5min.loc[i, 'datetime']}
            pivots.append(pivot)
            if len(pivots) <= 5:  # Show first few pivots
                print(f"Pivot High #{len(pivots)}: {pivot}")
        
        # Check for pivot low
        window_lows = df_5min.loc[i-leftBars:i+rightBars, 'low']
        if low == min(window_lows):
            pivot = {'type':'low', 'price':low, 'datetime':df_5min.loc[i, 'datetime']}
            pivots.append(pivot)
            if len(pivots) <= 5:  # Show first few pivots
                print(f"Pivot Low #{len(pivots)}: {pivot}")
    
    return pivots

def debug_first_trade():
    """Let's manually trace the first trade from the results with EXTREME detail"""
    print("="*100)
    print("DETAILED STEP-BY-STEP TRADE ANALYSIS - Trade #1")
    print("="*100)
    
    # From CSV: Trade 1 details
    print("🔍 TRADE #1 FROM BACKTEST RESULTS:")
    print("- Breakout Time: 2015-01-09 14:52:00")
    print("- Entry Time: 2015-01-09 14:53:00") 
    print("- Exit Time: 2015-01-09 14:55:00")
    print("- Direction: long")
    print("- Pivot Level: 8272.6")
    print("- Entry Price: 8275.95")
    print("- SL Price: 8269.65")
    print("- TP Price: 8288.55")
    print("- Exit Price: 8269.65")
    print("- Result: SL")
    print("- P&L: -6.3 points")
    print("- Marking Updates: 0")
    
    # Load data around this time
    df = load_data('NIFTY 50_minute_data.csv')
    
    # Look at data around the first trade
    start_time = pd.to_datetime('2015-01-09 14:45:00')
    end_time = pd.to_datetime('2015-01-09 15:05:00')
    
    trade_data = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
    print(f"\n📊 1-MINUTE DATA AROUND TRADE:")
    print("Time                Open      High      Low       Close")
    print("-" * 60)
    for _, row in trade_data.iterrows():
        marker = ""
        if row['datetime'] == pd.to_datetime('2015-01-09 14:52:00'):
            marker = " ← BREAKOUT"
        elif row['datetime'] == pd.to_datetime('2015-01-09 14:53:00'):
            marker = " ← ENTRY TIME"
        elif row['datetime'] == pd.to_datetime('2015-01-09 14:55:00'):
            marker = " ← EXIT TIME"
        print(f"{row['datetime']}   {row['open']:8.2f}  {row['high']:8.2f}  {row['low']:8.2f}  {row['close']:8.2f}{marker}")
    
    # Calculate pivots for this period
    print(f"\n🎯 PIVOT ANALYSIS:")
    jan_data = df[df['datetime'].dt.date <= pd.to_datetime('2015-01-15').date()]
    pivots = calculate_pivots_debug(jan_data[:2000])  # First 2000 rows to see early pivots
    
    # Find pivot highs around 8272.6
    pivot_highs = [p for p in pivots if p['type'] == 'high']
    print(f"\nFirst 5 pivot highs:")
    for i, ph in enumerate(pivot_highs[:5]):
        marker = " ← TARGET PIVOT" if abs(ph['price'] - 8272.6) < 0.1 else ""
        print(f"  {i+1}. {ph['datetime']} - {ph['price']:.2f}{marker}")
    
    # Look for the specific pivot at 8272.6
    target_pivot = None
    for ph in pivot_highs:
        if abs(ph['price'] - 8272.6) < 0.1:
            target_pivot = ph
            break
    
    if target_pivot:
        print(f"\n✅ FOUND TARGET PIVOT: {target_pivot}")
        
        # Check breakout candle at 14:52
        breakout_time = pd.to_datetime('2015-01-09 14:52:00')
        breakout_bar = df[df['datetime'] == breakout_time].iloc[0]
        
        print(f"\n🚀 BREAKOUT CANDLE ANALYSIS (14:52:00):")
        print(f"├─ OHLC: {breakout_bar['open']:.2f} / {breakout_bar['high']:.2f} / {breakout_bar['low']:.2f} / {breakout_bar['close']:.2f}")
        print(f"├─ Pivot level: {target_pivot['price']:.2f}")
        print(f"├─ High > Pivot? {breakout_bar['high'] > target_pivot['price']} ({breakout_bar['high']:.2f} > {target_pivot['price']:.2f})")
        print(f"├─ Green candle? {breakout_bar['close'] > breakout_bar['open']} ({breakout_bar['close']:.2f} > {breakout_bar['open']:.2f})")
        print(f"└─ Breakout Range: {breakout_bar['low']:.2f} - {breakout_bar['high']:.2f}")
        
        # Now let's trace the marking candle search step by step
        print(f"\n🎯 MARKING CANDLE SEARCH (Next 5 bars after breakout):")
        breakout_idx = df[df['datetime'] == breakout_time].index[0]
        
        marking_candle_found = False
        marking_bar = None
        marking_bar_num = None
        
        for j in range(1, 6):
            if breakout_idx + j < len(df):
                bar = df.iloc[breakout_idx + j]
                is_red = bar['close'] < bar['open']
                in_range = breakout_bar['low'] <= bar['close'] <= breakout_bar['high']
                
                print(f"\n📍 Bar {j} after breakout ({bar['datetime']}):")
                print(f"├─ OHLC: {bar['open']:.2f} / {bar['high']:.2f} / {bar['low']:.2f} / {bar['close']:.2f}")
                print(f"├─ Red candle? {is_red} ({'✓' if is_red else '✗'})")
                print(f"├─ Close in breakout range? {in_range} ({'✓' if in_range else '✗'})")
                print(f"├─ Range check: {breakout_bar['low']:.2f} <= {bar['close']:.2f} <= {breakout_bar['high']:.2f}")
                
                if is_red and in_range and not marking_candle_found:
                    print(f"└─ *** MARKING CANDLE FOUND! ***")
                    marking_candle_found = True
                    marking_bar = bar
                    marking_bar_num = j
                    
                    initial_entry = bar['high']
                    initial_sl = bar['low']
                    initial_sl_distance = abs(initial_entry - initial_sl)
                    initial_tp = initial_entry + 2 * initial_sl_distance
                    
                    print(f"   ├─ Initial Entry Price: {initial_entry:.2f} (marking candle high)")
                    print(f"   ├─ Initial SL Price: {initial_sl:.2f} (marking candle low)")
                    print(f"   ├─ Initial SL Distance: {initial_sl_distance:.2f}")
                    print(f"   ├─ Initial TP Price: {initial_tp:.2f} (Entry + 2*SL_distance)")
                    print(f"   └─ SL within 25 points? {initial_sl_distance <= 25} ({'✓' if initial_sl_distance <= 25 else '✗'})")
                else:
                    print(f"└─ Not a marking candle")
        
        if marking_candle_found:
            print(f"\n📈 MARKING CANDLE UPDATES CHECK:")
            print(f"Checking next {18 - marking_bar_num} bars for potential marking candle updates...")
            
            current_entry = marking_bar['high']
            current_sl = marking_bar['low']
            updates_count = 0
            
            # Process each bar sequentially: Check entry FIRST, then update marking levels
            print("🔄 BAR-BY-BAR PROCESSING (Correct Sequence):")
            entry_triggered = False
            entry_bar_num = None
            actual_entry_time = None
            
            for k in range(marking_bar_num + 1, min(19, len(df) - breakout_idx)):
                if breakout_idx + k >= len(df):
                    break
                
                bar = df.iloc[breakout_idx + k]
                bars_from_breakout = k
                
                print(f"\n� Bar {k} from breakout ({bar['datetime']}):")
                print(f"├─ OHLC: {bar['open']:.2f} / {bar['high']:.2f} / {bar['low']:.2f} / {bar['close']:.2f}")
                print(f"├─ Current Entry Level: {current_entry:.2f}")
                print(f"├─ Current SL Level: {current_sl:.2f}")
                print(f"├─ Updates so far: {updates_count}")
                
                # STEP 1: Check entry trigger with CURRENT levels (before any update)
                if not entry_triggered:
                    entry_condition = bar['high'] > current_entry
                    print(f"├─ Entry check: {bar['high']:.2f} > {current_entry:.2f} = {entry_condition}")
                    
                    if entry_condition:
                        print(f"├─ 🎯 *** ENTRY TRIGGERED! ***")
                        entry_triggered = True
                        entry_bar_num = k
                        actual_entry_time = bar['datetime']
                        print(f"├─ Entry Time: {actual_entry_time}")
                        print(f"└─ Entry Price: {current_entry:.2f} (NO further updates allowed)")
                        break  # Exit loop once entry is triggered
                    else:
                        print(f"├─ No entry trigger")
                
                # STEP 2: Since no entry, check if this bar can update marking levels
                sl_extension_long = bar['low'] < current_sl - 1
                can_update = (bars_from_breakout <= 18 and updates_count < 3 and sl_extension_long)
                
                print(f"├─ SL extension check: {bar['low']:.2f} < {current_sl-1:.2f} = {sl_extension_long}")
                print(f"├─ Can update? {can_update}")
                
                if can_update:
                    new_entry = bar['high']
                    new_sl = bar['low']
                    new_sl_distance = abs(new_entry - new_sl)
                    
                    print(f"├─ Potential new levels: Entry={new_entry:.2f}, SL={new_sl:.2f}")
                    print(f"├─ SL distance: {new_sl_distance:.2f}")
                    print(f"├─ Within 25 points? {new_sl_distance <= 25}")
                    
                    if new_sl_distance <= 25:
                        candle_color = "Red" if bar['close'] < bar['open'] else "Green"
                        print(f"└─ ✅ MARKING CANDLE UPDATED ({candle_color})!")
                        print(f"   ├─ Entry: {current_entry:.2f} → {new_entry:.2f}")
                        print(f"   ├─ SL: {current_sl:.2f} → {new_sl:.2f}")
                        
                        current_entry = new_entry
                        current_sl = new_sl
                        updates_count += 1
                        
                        new_tp = current_entry + 2 * new_sl_distance
                        print(f"   └─ TP: {new_tp:.2f}")
                    else:
                        print(f"└─ ❌ Update rejected (SL distance > 25)")
                else:
                    reasons = []
                    if bars_from_breakout > 18:
                        reasons.append("beyond 18 bars")
                    if updates_count >= 3:
                        reasons.append("max updates reached")
                    if not sl_extension_long:
                        reasons.append("no SL extension")
                    
                    print(f"└─ No update: {', '.join(reasons)}")
            
            print(f"\n🎯 FINAL RESULT:")
            print(f"├─ Entry Triggered: {entry_triggered}")
            if entry_triggered:
                print(f"├─ Entry Time: {actual_entry_time}")
                print(f"├─ Entry Price: {current_entry:.2f}")
                print(f"├─ Bars to entry: {entry_bar_num} from breakout")
            print(f"├─ Final SL: {current_sl:.2f}")
            print(f"├─ Total Updates: {updates_count}")
            final_tp = current_entry + 2 * abs(current_entry - current_sl)
            print(f"└─ Final TP: {final_tp:.2f}")
            
            # Compare with backtest results
            print(f"\n🔍 COMPARISON WITH BACKTEST RESULTS:")
            print(f"├─ Expected Entry: {current_entry:.2f} vs Backtest: 8275.95")
            print(f"├─ Expected SL: {current_sl:.2f} vs Backtest: 8269.65")
            print(f"├─ Expected TP: {final_tp:.2f} vs Backtest: 8288.55")
            print(f"├─ Expected Updates: {updates_count} vs Backtest: 0")
            if entry_triggered:
                print(f"├─ Expected Entry Time: {actual_entry_time} vs Backtest: 2015-01-09 14:53:00")
                print(f"└─ Bars to entry: {entry_bar_num} from breakout")
            else:
                print(f"└─ Entry not triggered in our analysis!")
        
        else:
            print(f"\n❌ NO MARKING CANDLE FOUND in next 5 bars!")
    else:
        print(f"\n❌ Could not find pivot at 8272.6")

def debug_winning_trade():
    """Let's also check a winning trade to see if logic works correctly"""
    print("\n" + "="*80)
    print("DEBUGGING WINNING TRADE - Trade #3")
    print("="*80)
    
    # From CSV: Trade 3 details (first TP)
    print("Trade 3 from CSV (First TP):")
    print("- Breakout Time: 2015-01-12 14:24:00")
    print("- Entry Time: 2015-01-12 14:26:00") 
    print("- Exit Time: 2015-01-12 14:29:00")
    print("- Direction: long")
    print("- Pivot Level: 8293.9")
    print("- Entry Price: 8311.45")
    print("- SL Price: 8307.35")
    print("- TP Price: 8319.65")
    print("- Exit Price: 8319.65")
    print("- Result: TP")
    print("- P&L: +8.2 points")
    
    # Load and examine the data
    df = load_data('NIFTY 50_minute_data.csv')
    
    start_time = pd.to_datetime('2015-01-12 14:20:00')
    end_time = pd.to_datetime('2015-01-12 14:35:00')
    
    trade_data = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
    print(f"\n1-minute data around winning trade:")
    print(trade_data[['datetime', 'open', 'high', 'low', 'close']].to_string())

if __name__ == "__main__":
    debug_first_trade()
    debug_winning_trade()
