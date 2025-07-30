#!/usr/bin/env python3
"""
Quick validation script to show the corrected bar-by-bar logic
Focus on first trade to confirm entry timing is correct
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def calculate_pivots(df, left_bars=15, right_bars=15):
    """Calculate pivot highs and lows on 5-min aggregated data"""
    # Aggregate to 5-min candles first
    df_5min = df.set_index('datetime').resample('5T').agg({
        'open': 'first',
        'high': 'max', 
        'low': 'min',
        'close': 'last'
    }).dropna().reset_index()
    
    pivot_highs = {}
    pivot_lows = {}
    
    for i in range(left_bars, len(df_5min) - right_bars):
        current_high = df_5min.iloc[i]['high']
        current_low = df_5min.iloc[i]['low']
        current_time = df_5min.iloc[i]['datetime']
        
        # Check if it's a pivot high
        is_pivot_high = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and df_5min.iloc[j]['high'] >= current_high:
                is_pivot_high = False
                break
        
        if is_pivot_high:
            pivot_highs[current_time] = current_high
            
        # Check if it's a pivot low  
        is_pivot_low = True
        for j in range(i - left_bars, i + right_bars + 1):
            if j != i and df_5min.iloc[j]['low'] <= current_low:
                is_pivot_low = False
                break
                
        if is_pivot_low:
            pivot_lows[current_time] = current_low
    
    return pivot_highs, pivot_lows

def test_corrected_logic():
    """Test the corrected bar-by-bar processing logic"""
    # Open output file for writing detailed results
    output_file = "validation_results.txt"
    with open(output_file, 'w') as f:
        f.write("🔧 TESTING CORRECTED BAR-BY-BAR LOGIC\n")
        f.write("=" * 60 + "\n")
        
        print("🔧 TESTING CORRECTED BAR-BY-BAR LOGIC")
        print("=" * 60)
    
        # Load data
        df = pd.read_csv('NIFTY 50_minute_data.csv')
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # Filter to market hours only
        df = df[df['datetime'].dt.time.between(time(9, 15), time(15, 29))]
        
        # Limit to first 5000 rows for testing
        df = df.head(5000)
        
        log_msg = f"📊 Data loaded: {len(df):,} rows (limited for testing)"
        print(log_msg)
        f.write(log_msg + "\n")
        
        # Calculate pivots
        pivot_highs, pivot_lows = calculate_pivots(df)
        log_msg = f"🎯 Found {len(pivot_highs)} pivot highs, {len(pivot_lows)} pivot lows"
        print(log_msg)
        f.write(log_msg + "\n")
        
        # Show first few pivots
        if pivot_highs or pivot_lows:
            log_msg = f"\n🔍 FIRST FEW PIVOTS:"
            print(log_msg)
            f.write(log_msg + "\n")
            all_pivots = []
            for ptime, plevel in pivot_highs.items():
                all_pivots.append((ptime, plevel, 'HIGH'))
            for ptime, plevel in pivot_lows.items():
                all_pivots.append((ptime, plevel, 'LOW'))
            
            all_pivots.sort()
            for i, (ptime, plevel, ptype) in enumerate(all_pivots[:5]):
                log_msg = f"   {i+1}. {ptime} - {ptype} @ {plevel}"
                print(log_msg)
                f.write(log_msg + "\n")
        
        # Simulate bar-by-bar processing for ALL 5000 rows
        log_msg = f"\n📈 BAR-BY-BAR PROCESSING (Corrected Logic - All {len(df)} rows):"
        print(log_msg)
        f.write(log_msg + "\n")
        log_msg = "=" * 60
        print(log_msg)
        f.write(log_msg + "\n")
        
        active_trade = None
        marking_candle_active = False
        marking_entry = None
        marking_sl = None
        marking_direction = None
        marking_updates = 0
        trades_found = 0
    
        for i in range(len(df)):
            bar = df.iloc[i]
            bar_time = bar['datetime']
            
            # Only print detailed info for first few trades or when something happens
            verbose = trades_found < 3 or marking_candle_active or active_trade
            
            if verbose:
                log_msg = f"\n📊 BAR {i+1}: {bar_time}"
                print(log_msg)
                f.write(log_msg + "\n")
                log_msg = f"   OHLC: {bar['open']:.1f}/{bar['high']:.1f}/{bar['low']:.1f}/{bar['close']:.1f}"
                print(log_msg)
                f.write(log_msg + "\n")        # STEP 1: Check entry triggers FIRST (before any updates)
        if marking_candle_active and not active_trade:
            entry_triggered = False
            
            if marking_direction == 'long':
                if bar['high'] > marking_entry:  # HIGH > ENTRY (not >=)
                    print(f"   ✅ LONG ENTRY TRIGGERED @ {marking_entry}")
                    active_trade = {
                        'direction': 'long',
                        'entry': marking_entry,
                        'sl': marking_sl,
                        'tp': marking_entry + 2 * (marking_entry - marking_sl),
                        'entry_time': bar_time,
                        'updates': marking_updates
                    }
                    entry_triggered = True
                    
            elif marking_direction == 'short':
                if bar['low'] < marking_entry:  # LOW < ENTRY (not <=)
                    print(f"   ✅ SHORT ENTRY TRIGGERED @ {marking_entry}")
                    active_trade = {
                        'direction': 'short', 
                        'entry': marking_entry,
                        'sl': marking_sl,
                        'tp': marking_entry - 2 * (marking_sl - marking_entry),
                        'entry_time': bar_time,
                        'updates': marking_updates
                    }
                    entry_triggered = True
            
            # If entry triggered, stop processing this bar
            if entry_triggered:
                marking_candle_active = False
                continue
        
        # STEP 2: Check for breakouts (only if no active trade)
        if not active_trade and not marking_candle_active:
            for ptime, plevel in pivot_highs.items():
                if ptime <= bar_time and bar['high'] > plevel:
                    if verbose:
                        print(f"   🚀 LONG BREAKOUT detected vs pivot {ptime} @ {plevel}")
                    # Look for marking candle in next few bars
                    marking_direction = 'long'
                    break
                    
            for ptime, plevel in pivot_lows.items():
                if ptime <= bar_time and bar['low'] < plevel:
                    if verbose:
                        print(f"   🚀 SHORT BREAKOUT detected vs pivot {ptime} @ {plevel}")
                    marking_direction = 'short'
                    break
        
        # STEP 3: Check for marking candle pattern
        if marking_direction and not marking_candle_active and not active_trade:
            if marking_direction == 'long' and bar['close'] < bar['open']:  # RED candle for LONG
                print(f"   📍 MARKING CANDLE FOUND (Long Red)")
                marking_entry = bar['high']
                marking_sl = bar['low']
                marking_candle_active = True
                marking_updates = 0
                print(f"   📋 Initial levels: Entry={marking_entry}, SL={marking_sl}")
                
            elif marking_direction == 'short' and bar['close'] > bar['open']:  # GREEN candle for SHORT
                print(f"   📍 MARKING CANDLE FOUND (Short Green)")
                marking_entry = bar['low']
                marking_sl = bar['high']
                marking_candle_active = True
                marking_updates = 0
                print(f"   📋 Initial levels: Entry={marking_entry}, SL={marking_sl}")
        
        # STEP 4: Update marking levels (only if no entry triggered this bar)
        elif marking_candle_active and not active_trade and marking_updates < 3:
            old_entry = marking_entry
            old_sl = marking_sl
            
            if marking_direction == 'long':
                # Check if SL needs to be extended (low goes below current SL - 1)
                if bar['low'] < marking_sl - 1:
                    new_entry = bar['high']
                    new_sl = bar['low']
                    marking_entry = new_entry
                    marking_sl = new_sl
                    marking_updates += 1
                    color = "Green" if bar['close'] > bar['open'] else "Red"
                    if verbose:
                        print(f"   🔄 MARKING UPDATE #{marking_updates} (Long {color}) - SL Extended")
                        print(f"   📋 Updated: Entry {old_entry}→{marking_entry}, SL {old_sl}→{marking_sl}")
                    
            elif marking_direction == 'short':
                # Check if SL needs to be extended (high goes above current SL + 1)
                if bar['high'] > marking_sl + 1:
                    new_entry = bar['low']
                    new_sl = bar['high']
                    marking_entry = new_entry
                    marking_sl = new_sl
                    marking_updates += 1
                    color = "Green" if bar['close'] > bar['open'] else "Red"
                    if verbose:
                        print(f"   🔄 MARKING UPDATE #{marking_updates} (Short {color}) - SL Extended")
                        print(f"   📋 Updated: Entry {old_entry}→{marking_entry}, SL {old_sl}→{marking_sl}")
        
        # STEP 5: Check trade exit
        if active_trade:
            if active_trade['direction'] == 'long':
                if bar['low'] <= active_trade['sl']:
                    print(f"   ❌ LONG SL HIT @ {active_trade['sl']}")
                    pnl = active_trade['sl'] - active_trade['entry']
                    print(f"   💰 P&L: {pnl:.1f} points")
                    print(f"   📊 TRADE SUMMARY: Entry={active_trade['entry']}, SL={active_trade['sl']}, Updates={active_trade['updates']}")
                    trades_found += 1
                    active_trade = None
                    marking_direction = None
                    if trades_found >= 3:
                        print(f"\n✅ Found {trades_found} trades. Stopping early for validation.")
                        break
                elif bar['high'] >= active_trade['tp']:
                    print(f"   ✅ LONG TP HIT @ {active_trade['tp']}")
                    pnl = active_trade['tp'] - active_trade['entry']
                    print(f"   💰 P&L: {pnl:.1f} points")
                    print(f"   📊 TRADE SUMMARY: Entry={active_trade['entry']}, SL={active_trade['sl']}, Updates={active_trade['updates']}")
                    trades_found += 1
                    active_trade = None
                    marking_direction = None
                    if trades_found >= 3:
                        print(f"\n✅ Found {trades_found} trades. Stopping early for validation.")
                        break
                    
            elif active_trade['direction'] == 'short':
                if bar['high'] >= active_trade['sl']:
                    print(f"   ❌ SHORT SL HIT @ {active_trade['sl']}")
                    pnl = active_trade['entry'] - active_trade['sl']
                    print(f"   💰 P&L: {pnl:.1f} points")
                    print(f"   📊 TRADE SUMMARY: Entry={active_trade['entry']}, SL={active_trade['sl']}, Updates={active_trade['updates']}")
                    trades_found += 1
                    active_trade = None
                    marking_direction = None
                    if trades_found >= 3:
                        print(f"\n✅ Found {trades_found} trades. Stopping early for validation.")
                        break
                elif bar['low'] <= active_trade['tp']:
                    print(f"   ✅ SHORT TP HIT @ {active_trade['tp']}")
                    pnl = active_trade['entry'] - active_trade['tp']
                    print(f"   💰 P&L: {pnl:.1f} points")
                    print(f"   📊 TRADE SUMMARY: Entry={active_trade['entry']}, SL={active_trade['sl']}, Updates={active_trade['updates']}")
                    trades_found += 1
                    active_trade = None
                    marking_direction = None
                    if trades_found >= 3:
                        print(f"\n✅ Found {trades_found} trades. Stopping early for validation.")
                        break
    
    if trades_found == 0:
        print("\n❓ No trades found in the 5000 rows")
    else:
        print(f"\n🎯 Validation complete: Found {trades_found} trades")
    return None

if __name__ == "__main__":
    test_corrected_logic()
