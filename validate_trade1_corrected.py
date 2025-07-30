#!/usr/bin/env python3
"""
Validation script specifically for Trade #1 to confirm corrected logic
Tracks the exact pivot (8272.60 at 13:10) and breakout (8275.15 at 14:52)
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

def validate_trade1_corrected():
    """Validate the corrected logic for Trade #1 specifically"""
    print("🔧 VALIDATING TRADE #1 WITH CORRECTED LOGIC")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv('NIFTY 50_minute_data.csv')
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Filter to market hours only
    df = df[df['datetime'].dt.time.between(time(9, 15), time(15, 29))]
    
    print(f"📊 Data loaded: {len(df):,} rows")
    
    # Trade #1 specific details
    pivot_time = pd.to_datetime('2015-01-09 13:10:00')
    pivot_level = 8272.60
    breakout_time = pd.to_datetime('2015-01-09 14:52:00')
    breakout_high = 8275.15
    
    print(f"🎯 TRADE #1 PARAMETERS:")
    print(f"   Pivot: {pivot_time} @ {pivot_level}")
    print(f"   Breakout: {breakout_time} @ {breakout_high}")
    print(f"   Margin: +{(breakout_high - pivot_level):.2f} points")
    print()
    
    # Find breakout bar index
    breakout_idx = df[df['datetime'] == breakout_time].index[0]
    
    print(f"📈 BAR-BY-BAR PROCESSING (CORRECTED LOGIC):")
    print("=" * 70)
    
    # Simulate the corrected logic
    active_trade = None
    marking_candle_active = False
    marking_entry = None
    marking_sl = None
    marking_direction = 'long'  # We know this is a long breakout
    marking_updates = 0
    breakout_detected = False
    
    # Start from breakout bar and process forward
    for i in range(breakout_idx, min(breakout_idx + 25, len(df))):
        bar = df.iloc[i]
        bar_time = bar['datetime']
        bars_from_breakout = i - breakout_idx
        
        print(f"\\n📊 BAR {bars_from_breakout}: {bar_time}")
        print(f"   OHLC: {bar['open']:.2f}/{bar['high']:.2f}/{bar['low']:.2f}/{bar['close']:.2f}")
        
        # STEP 1: Detect breakout (only on first bar)
        if bars_from_breakout == 0:
            if bar['high'] > pivot_level:
                print(f"   🚀 LONG BREAKOUT DETECTED @ {bar['high']:.2f} vs pivot {pivot_level}")
                breakout_detected = True
                # Look for marking candle (need red candle first)
                if bar['close'] < bar['open']:  # Red candle
                    print(f"   📍 MARKING CANDLE FOUND (Red candle on breakout bar)")
                    marking_entry = bar['high']  # Exact high, no +1
                    marking_sl = bar['low']      # Exact low, no -2
                    marking_candle_active = True
                    marking_updates = 0
                    print(f"   📋 Initial levels: Entry={marking_entry:.2f}, SL={marking_sl:.2f}")
                else:
                    print(f"   ⏳ Green breakout candle - looking for red marking candle")
            continue
        
        # STEP 2: Check entry triggers FIRST (before any updates)
        if marking_candle_active and not active_trade:
            if bar['high'] >= marking_entry:
                print(f"   ✅ LONG ENTRY TRIGGERED @ {marking_entry:.2f}")
                tp_level = marking_entry + 2 * (marking_entry - marking_sl)
                active_trade = {
                    'direction': 'long',
                    'entry': marking_entry,
                    'sl': marking_sl,
                    'tp': tp_level,
                    'entry_time': bar_time,
                    'updates': marking_updates,
                    'entry_bar': bars_from_breakout
                }
                marking_candle_active = False
                print(f"   📊 TRADE ACTIVE: Entry={marking_entry:.2f}, SL={marking_sl:.2f}, TP={tp_level:.2f}")
                print(f"   🔢 Total marking updates: {marking_updates}")
                continue  # Skip to trade management
        
        # STEP 3: Look for initial marking candle (if not found yet)
        if breakout_detected and not marking_candle_active and not active_trade and bars_from_breakout <= 5:
            if bar['close'] < bar['open']:  # Red candle
                print(f"   📍 MARKING CANDLE FOUND (Red) at bar {bars_from_breakout}")
                marking_entry = bar['high']  # Exact high
                marking_sl = bar['low']      # Exact low
                sl_distance = marking_entry - marking_sl
                
                if sl_distance <= 25:  # Validate SL distance
                    marking_candle_active = True
                    marking_updates = 0
                    print(f"   📋 Initial levels: Entry={marking_entry:.2f}, SL={marking_sl:.2f}")
                    print(f"   ✅ SL distance: {sl_distance:.2f} points (within 25 limit)")
                else:
                    print(f"   ❌ SL distance {sl_distance:.2f} > 25 points - rejected")
                continue
        
        # STEP 4: Update marking levels (only if no entry triggered this bar)
        if marking_candle_active and not active_trade and bars_from_breakout <= 18 and marking_updates < 3:
            old_entry = marking_entry
            old_sl = marking_sl
            
            # Check if this bar extends SL (any color candle can update)
            if bar['low'] < marking_sl - 1:
                new_entry = bar['high']  # Exact high
                new_sl = bar['low']      # Exact low
                new_sl_distance = new_entry - new_sl
                
                if new_sl_distance <= 25:  # Validate new SL distance
                    marking_entry = new_entry
                    marking_sl = new_sl
                    marking_updates += 1
                    color = "Green" if bar['close'] > bar['open'] else "Red"
                    print(f"   🔄 MARKING UPDATE #{marking_updates} ({color} candle)")
                    print(f"   📋 Updated: Entry {old_entry:.2f}→{marking_entry:.2f}, SL {old_sl:.2f}→{marking_sl:.2f}")
                    print(f"   ✅ New SL distance: {new_sl_distance:.2f} points")
                else:
                    print(f"   ❌ Update rejected: SL distance would be {new_sl_distance:.2f} > 25 points")
        
        # STEP 5: Check trade exit (if trade is active)
        if active_trade:
            if bar['low'] <= active_trade['sl']:
                print(f"   ❌ LONG SL HIT @ {active_trade['sl']:.2f}")
                pnl = active_trade['sl'] - active_trade['entry']
                print(f"   💰 P&L: {pnl:.2f} points")
                print(f"\\n📊 FINAL TRADE SUMMARY:")
                print(f"   Entry Time: {active_trade['entry_time']}")
                print(f"   Entry Price: {active_trade['entry']:.2f}")
                print(f"   SL Price: {active_trade['sl']:.2f}")
                print(f"   TP Price: {active_trade['tp']:.2f}")
                print(f"   Total Updates: {active_trade['updates']}")
                print(f"   Entry Bar: {active_trade['entry_bar']} (from breakout)")
                print(f"   Result: SL Hit")
                print(f"   P&L: {pnl:.2f} points")
                return active_trade
                
            elif bar['high'] >= active_trade['tp']:
                print(f"   ✅ LONG TP HIT @ {active_trade['tp']:.2f}")
                pnl = active_trade['tp'] - active_trade['entry']
                print(f"   💰 P&L: {pnl:.2f} points")
                print(f"\\n📊 FINAL TRADE SUMMARY:")
                print(f"   Entry Time: {active_trade['entry_time']}")
                print(f"   Entry Price: {active_trade['entry']:.2f}")
                print(f"   SL Price: {active_trade['sl']:.2f}")
                print(f"   TP Price: {active_trade['tp']:.2f}")
                print(f"   Total Updates: {active_trade['updates']}")
                print(f"   Entry Bar: {active_trade['entry_bar']} (from breakout)")
                print(f"   Result: TP Hit")
                print(f"   P&L: {pnl:.2f} points")
                return active_trade
    
    print("\\n❓ Trade did not complete within 25 bars")
    if active_trade:
        print(f"   Trade still active with Entry={active_trade['entry']:.2f}, SL={active_trade['sl']:.2f}")
    elif marking_candle_active:
        print(f"   Marking candle active with Entry={marking_entry:.2f}, SL={marking_sl:.2f}, Updates={marking_updates}")
    else:
        print("   No marking candle found")
    
    return None

if __name__ == "__main__":
    validate_trade1_corrected()
