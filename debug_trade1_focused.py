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

def analyze_trade1_updates():
    """Focus specifically on marking candle updates for Trade 1"""
    print("FOCUSED ANALYSIS: Trade #1 Marking Candle Updates")
    print("="*70)
    
    df = load_data('NIFTY 50_minute_data.csv')
    
    # Get the exact data around Trade 1
    start_time = pd.to_datetime('2015-01-09 14:52:00')
    end_time = pd.to_datetime('2015-01-09 15:05:00')
    
    trade_data = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)]
    
    print("1-MINUTE DATA FOR TRADE 1:")
    print("Time                OHLC")
    print("-" * 50)
    for i, (_, row) in enumerate(trade_data.iterrows()):
        bar_num = f"Bar {i}" if i == 0 else f"Bar {i} "
        print(f"{bar_num:6} {row['datetime']} {row['open']:8.2f} {row['high']:8.2f} {row['low']:8.2f} {row['close']:8.2f}")
    
    # Breakout at 14:52 (Bar 0)
    breakout_bar = trade_data.iloc[0]
    print(f"\nBREAKOUT BAR (14:52):")
    print(f"Range: {breakout_bar['low']:.2f} - {breakout_bar['high']:.2f}")
    
    # Find marking candle
    marking_candle = None
    marking_bar_idx = None
    
    print(f"\nMARKING CANDLE SEARCH:")
    for i in range(1, min(6, len(trade_data))):
        bar = trade_data.iloc[i]
        is_red = bar['close'] < bar['open']
        in_range = breakout_bar['low'] <= bar['close'] <= breakout_bar['high']
        
        print(f"Bar {i} ({bar['datetime'].strftime('%H:%M')}) - Red: {is_red}, In Range: {in_range}")
        print(f"  Close: {bar['close']:.2f}, Range: {breakout_bar['low']:.2f} - {breakout_bar['high']:.2f}")
        
        if is_red and in_range and marking_candle is None:
            marking_candle = bar
            marking_bar_idx = i
            print(f"  *** MARKING CANDLE FOUND ***")
            print(f"  Entry: {bar['high']:.2f}, SL: {bar['low']:.2f}")
    
    if marking_candle is not None:
        print(f"\nMARKING CANDLE UPDATES SIMULATION:")
        print(f"Initial - Entry: {marking_candle['high']:.2f}, SL: {marking_candle['low']:.2f}")
        
        current_entry = marking_candle['high']
        current_sl = marking_candle['low']
        updates = 0
        breakout_bar_idx = 0  # 14:52 is bar 0
        
        # Check each subsequent bar for potential updates
        for i in range(marking_bar_idx + 1, len(trade_data)):
            bar = trade_data.iloc[i]
            bars_from_breakout = i  # Since breakout is at index 0
            
            print(f"\nBar {i} ({bar['datetime'].strftime('%H:%M')}) - {bars_from_breakout} bars from breakout:")
            print(f"  OHLC: {bar['open']:.2f} / {bar['high']:.2f} / {bar['low']:.2f} / {bar['close']:.2f}")
            print(f"  Current Entry: {current_entry:.2f}, Current SL: {current_sl:.2f}")
            print(f"  Updates so far: {updates}")
            
            # Check update conditions  
            sl_extension = bar['low'] < current_sl - 1
            within_18_bars = bars_from_breakout <= 18
            under_3_updates = updates < 3
            
            print(f"  SL extension? {sl_extension} ({bar['low']:.2f} < {current_sl - 1:.2f})")
            print(f"  Within 18 bars? {within_18_bars}")
            print(f"  Under 3 updates? {under_3_updates}")
            
            if sl_extension and within_18_bars and under_3_updates:
                new_entry = bar['high']
                new_sl = bar['low']
                new_sl_distance = abs(new_entry - new_sl)
                
                print(f"  New Entry: {new_entry:.2f}, New SL: {new_sl:.2f}")
                print(f"  New SL Distance: {new_sl_distance:.2f}")
                print(f"  Within 25 points? {new_sl_distance <= 25}")
                
                if new_sl_distance <= 25:
                    candle_color = "Red" if bar['close'] < bar['open'] else "Green"
                    print(f"  *** UPDATE ACCEPTED ({candle_color} candle) ***")
                    current_entry = new_entry
                    current_sl = new_sl
                    updates += 1
                else:
                    print(f"  Update rejected - SL distance > 25")
            else:
                reasons = []
                if not sl_extension: reasons.append("no SL extension")
                if not within_18_bars: reasons.append("beyond 18 bars")
                if not under_3_updates: reasons.append("max updates")
                print(f"  No update: {', '.join(reasons)}")
        
        print(f"\nFINAL SETUP:")
        print(f"Entry: {current_entry:.2f}")
        print(f"SL: {current_sl:.2f}")
        print(f"Total Updates: {updates}")
        
        final_tp = current_entry + 2 * abs(current_entry - current_sl)
        print(f"TP: {final_tp:.2f}")
        
        print(f"\nCOMPARISON TO YOUR EXPECTED VALUES:")
        print(f"Expected Entry: 8266.65 vs Calculated: {current_entry:.2f}")
        print(f"Expected SL: 8263.15 vs Calculated: {current_sl:.2f}")

if __name__ == "__main__":
    analyze_trade1_updates()
