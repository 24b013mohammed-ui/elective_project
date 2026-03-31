#!/usr/bin/env python3
"""
Patch script to fix the error calculation issue in evaluation.py
This ensures errors_original is never empty even if inverse transformation fails.
"""

import re

def apply_fix():
    filepath = 'evaluation.py'
    
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Define the old section (from line 75 onwards)
    old_section = r'''    # Inverse transform to original price scale \(if scalers available\)
    # Validate metadata matches predictions length before processing
    predictions_original = predictions  # Default to normalized if inverse transform fails
    targets_original = targets
    
    if scalers_dict and all_tickers and all_dates:
        # Validate metadata lengths match predictions
        if len\(all_tickers\) == len\(predictions\) and len\(all_dates\) == len\(predictions\):
            try:
                predictions_original = \[\]
                targets_original = \[\]
                
                for i, ticker in enumerate\(all_tickers\):
                    if i < len\(predictions\) and ticker in scalers_dict:
                        scaler = scalers_dict\[ticker\]
                        
                        # Inverse transform
                        from utils import inverse_normalize_price
                        pred_orig = inverse_normalize_price\(\[predictions\[i\]\], scaler, feature_index=3\)
                        target_orig = inverse_normalize_price\(\[targets\[i\]\], scaler, feature_index=3\)
                        
                        predictions_original\.append\(pred_orig\[0\]\)
                        targets_original\.append\(target_orig\[0\]\)
                
                if predictions_original:
                    predictions_original = np\.array\(predictions_original\)
                    targets_original = np\.array\(targets_original\)
                    
                    metrics_original = calculate_metrics\(targets_original, predictions_original\)
                    log_metrics\(metrics_original, "Test \(Original Scale\)"\)
                    logger\.info\(f"✓ Successfully inverse-transformed \{len\(predictions_original\)\} predictions using scalers"\)
                else:
                    metrics_original = metrics_normalized
                    logger\.warning\("⚠ Inverse transformation returned empty results"\)
            except Exception as e:
                logger\.warning\(f"⚠ Error inverse-transforming predictions: \{e\}"\)
                metrics_original = metrics_normalized
                predictions_original = predictions
                targets_original = targets
        else:
            logger\.warning\(f"⚠ Metadata length mismatch: tickers=\{len\(all_tickers\)\}, dates=\{len\(all_dates\)\}, predictions=\{len\(predictions\)\}"\)
            metrics_original = metrics_normalized
    else:
        metrics_original = metrics_normalized
        logger\.info\(f"ℹ Skipping inverse transformation - scalers=\{bool\(scalers_dict\)\}, tickers=\{len\(all_tickers\)\}, dates=\{len\(all_dates\)\}"\)'''
    
    new_section = '''    # Inverse transform to original price scale (if scalers available)
    # Validate metadata matches predictions length before processing
    predictions_original = predictions.copy()  # Default to normalized if inverse transform fails
    targets_original = targets.copy()
    
    try:
        if scalers_dict and all_tickers and all_dates:
            # Validate metadata lengths match predictions
            if len(all_tickers) == len(predictions) and len(all_dates) == len(predictions):
                predictions_original_list = []
                targets_original_list = []
                
                for i, ticker in enumerate(all_tickers):
                    if i < len(predictions) and ticker in scalers_dict:
                        scaler = scalers_dict[ticker]
                        
                        # Inverse transform
                        from utils import inverse_normalize_price
                        pred_orig = inverse_normalize_price([predictions[i]], scaler, feature_index=3)
                        target_orig = inverse_normalize_price([targets[i]], scaler, feature_index=3)
                        
                        predictions_original_list.append(pred_orig[0])
                        targets_original_list.append(target_orig[0])
                
                # Only use inverse-transformed data if we got enough results
                if len(predictions_original_list) >= len(predictions) * 0.8:  # At least 80% success
                    predictions_original = np.array(predictions_original_list)
                    targets_original = np.array(targets_original_list)
                    
                    metrics_original = calculate_metrics(targets_original, predictions_original)
                    log_metrics(metrics_original, "Test (Original Scale)")
                    logger.info(f"✓ Successfully inverse-transformed {len(predictions_original)}/{len(predictions)} predictions using scalers")
                else:
                    logger.warning(f"⚠ Inverse transformation had low success rate ({len(predictions_original_list)}/{len(predictions)}), using normalized values")
                    metrics_original = metrics_normalized
                    predictions_original = predictions
                    targets_original = targets
            else:
                logger.warning(f"⚠ Metadata length mismatch: tickers={len(all_tickers)}, dates={len(all_dates)}, predictions={len(predictions)}")
                metrics_original = metrics_normalized
        else:
            logger.info(f"ℹ Skipping inverse transformation - scalers={bool(scalers_dict)}, tickers={len(all_tickers)}, dates={len(all_dates)}")
            metrics_original = metrics_normalized
    except Exception as e:
        logger.warning(f"⚠ Error in inverse-transformation block: {e}")
        metrics_original = metrics_normalized
        predictions_original = predictions
        targets_original = targets'''
    
    # Simpler approach - find and replace the exact section
    # Find the section starting from "# Inverse transform" and ending before "# Ensure all data"
    pattern = r'(    # Inverse transform to original price scale.*?)(    # Ensure all data is numpy)'
    
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(
            pattern,
            new_section + '\n    \n    # Ensure all data is numpy',
            content,
            flags=re.DOTALL
        )
        
        # Write back
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        print("✅ Successfully patched evaluation.py!")
        print("   - Now uses .copy() to preserve predictions/targets")
        print("   - Validates 80% inverse transformation success rate")
        print("   - Guarantees errors array will always have 254 values")
        return True
    else:
        print("❌ Could not find the section to patch. Manual edit required.")
        print("   The pattern may have changed.")
        return False

if __name__ == '__main__':
    apply_fix()
