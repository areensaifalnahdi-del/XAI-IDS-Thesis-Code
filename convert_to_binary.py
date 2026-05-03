"""
Convert Multi-Class Labels to Binary Classification (Attack vs BENIGN)
 
This script converts your multi-class datasets to binary classification:
- BENIGN = 0 (Normal traffic)
- Any Attack = 1 (All attack types combined)
 
This creates:
- friday_clean_binary.csv
- wednesday_clean_binary.csv
"""
 
import pandas as pd
import numpy as np
import os
import sys
 
def convert_to_binary(input_file, output_file):
    """
    Convert multi-class labels to binary (BENIGN vs Attack).
    
    Args:
        input_file (str): Path to input CSV
        output_file (str): Path to output CSV
    """
    if not os.path.exists(input_file):
        print(f"✗ Error: File '{input_file}' not found.")
        print(f"  Current directory: {os.getcwd()}")
        return False
    
    print(f"Processing {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"  Original shape: {df.shape}")
    
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()
    
    print(f"  Original classes: {df['Label'].unique()}")
    
    # Convert to binary: BENIGN = 0, Any Attack = 1
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    print(f"  New classes: {df['Label'].unique()}")
    print(f"  Class distribution:")
    print(f"    - BENIGN (0): {(df['Label'] == 0).sum()}")
    print(f"    - Attack (1): {(df['Label'] == 1).sum()}")
    print(f"  Total samples: {len(df)}")
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to: {output_file}\n")
    return True
 
# Convert both files
if __name__ == '__main__':
    print("\n" + "="*70)
    print("CONVERTING TO BINARY CLASSIFICATION")
    print("="*70 + "\n")
    
    success = True
    success = convert_to_binary('data/friday_clean.csv', 'data/friday_clean_binary.csv') and success
    success = convert_to_binary('data/wednesday_clean.csv', 'data/wednesday_clean_binary.csv') and success
    
    if success:
        print("="*70)
        print("✓ BINARY CONVERSION COMPLETE!")
        print("="*70)
        print("\nYou now have:")
        print("  - friday_clean_binary.csv")
        print("  - wednesday_clean_binary.csv")
        print("\nNext: Run the analysis_v2.py script")
        print("  Command: python analysis_v2.py\n")
    else:
        print("\n✗ Conversion failed. Check the errors above.")
