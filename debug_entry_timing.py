#!/usr/bin/env python3

import pandas as pd
import numpy as np
from main import load_data, calculate_pivots, simulate_trade_setup, check_entry_trigger
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_entry_timing():
    """Debug the entry timing issue"""
    
    # Load data
    logger.info("Loading data...")
    df = load_data('NIFTY 50_minute_data.csv')
    logger.info(f"Data loaded: {len(df)} rows")
    
    # Calculate pivots
    logger.info("Calculating pivots...")
    df_with_pivots = calculate_pivots(df, 15, 15)
    
    # Find Trade #1 setup (2024-01-01 14:47)
    target_time = pd.Timestamp('2024-01-01 14:47:00')
    target_idx = df_with_pivots[df_with_pivots['datetime'] == target_time].index[0]
    
    logger.info(f"Found Trade #1 breakout at index {target_idx}, time {target_time}")
    
    # Get the trade setup
    breakout_row = df_with_pivots.iloc[target_idx]
    pivot_low = breakout_row['pivot_low']
    
    trade_info = simulate_trade_setup(df_with_pivots, target_idx, 'Long', pivot_low, logger)
    if not trade_info:
        logger.error("No trade setup created!")
        return
    
    logger.info(f"Initial trade setup: Entry={trade_info['entry_price']:.2f}, SL={trade_info['sl_price']:.2f}")
    
    # Track marking candle updates and entry checks
    logger.info("\n=== Simulating Bar-by-Bar Processing ===")
    
    current_trade = trade_info.copy()
    
    # Process bars starting from breakout+1
    for offset in range(1, 25):  # Check next 24 bars
        current_idx = target_idx + offset
        if current_idx >= len(df_with_pivots):
            break
            
        current_row = df_with_pivots.iloc[current_idx]
        current_time = current_row['datetime']
        
        logger.info(f"\nBar {offset}: {current_time} - OHLC: {current_row['open']:.1f}/{current_row['high']:.1f}/{current_row['low']:.1f}/{current_row['close']:.1f}")
        
        # Check if marking candle update would happen
        bars_from_breakout = current_idx - current_trade['breakout_idx']
        if bars_from_breakout <= 18 and current_trade['updates'] < 3:
            
            # Check if this candle extends SL by >= 1 point
            if current_trade['direction'] == 'Long':
                if current_row['low'] <= current_trade['sl_price'] - 1:
                    old_entry = current_trade['entry_price']
                    old_sl = current_trade['sl_price']
                    
                    # Update marking candle
                    current_trade['sl_price'] = current_row['low']
                    current_trade['entry_price'] = current_row['high']
                    current_trade['updates'] += 1
                    
                    candle_color = "Green" if current_row['close'] > current_row['open'] else "Red"
                    logger.info(f"  *** MARKING CANDLE UPDATE ({candle_color}) ***")
                    logger.info(f"      Entry: {old_entry:.2f} -> {current_trade['entry_price']:.2f}")
                    logger.info(f"      SL: {old_sl:.2f} -> {current_trade['sl_price']:.2f}")
                    logger.info(f"      Updates: {current_trade['updates']}")
        
        # Now check entry trigger on the SAME bar
        if not current_trade.get('entered', False):
            if check_entry_trigger(current_row, current_trade):
                logger.info(f"  *** ENTRY TRIGGERED ON SAME BAR ***")
                logger.info(f"      High {current_row['high']:.2f} >= Entry {current_trade['entry_price']:.2f}")
                current_trade['entered'] = True
                current_trade['entry_idx'] = current_idx
                break
        
        # Check if entry would trigger on NEXT bar (if update happened)
        if offset < 24 and current_idx + 1 < len(df_with_pivots):
            next_row = df_with_pivots.iloc[current_idx + 1]
            if not current_trade.get('entered', False):
                if check_entry_trigger(next_row, current_trade):
                    logger.info(f"  >> Entry would trigger on NEXT bar: High {next_row['high']:.2f} >= Entry {current_trade['entry_price']:.2f}")

if __name__ == "__main__":
    debug_entry_timing()
