import pandas as pd
import numpy as np
import glob
import os

def load_and_prepare_data(all_files):
    if not all_files:
        print("Error: No metric files found! Please check the ROOT_SEARCH_PATH.")
        return None

    df_list = [pd.read_csv(f) for f in all_files]
    df_combined = pd.concat(df_list, ignore_index=True)
    print(f"Successfully combined {len(all_files)} metric files from all scenarios.")

    # --- Use the correct column names ---
    REQUIRED_COLUMNS = {
        'Timestamp': 'Timestamp',
        'slice_id': 'Slice_ID',
        'ul_rssi': 'RSRP', 
        'dl_cqi': 'CQI',   
        'num_ues': 'num_ues',
        'tx_brate downlink [Mbps]': 'Tx_brate_DL_Mbps'
    }
    df_combined.rename(columns=REQUIRED_COLUMNS, inplace=True)
    
    # --- Correctly calculate mean AND standard deviation ---
    aggregations = {
        'RSRP': ['mean', 'std'],
        'CQI': ['mean', 'std'],
        'num_ues': 'first' 
    }
    df_agg = df_combined.groupby(['Timestamp', 'Slice_ID']).agg(aggregations)
    df_agg.columns = ['avg_rsrp', 'std_rsrp', 'avg_cqi', 'std_cqi', 'num_ues']
    df_agg.reset_index(inplace=True)
    df_agg = df_agg.fillna(0)
    
    # Merge aggregated features back with throughput data
    df_traffic = df_combined[['Timestamp', 'Slice_ID', 'Tx_brate_DL_Mbps']].drop_duplicates()
    df_merged = pd.merge(df_traffic, df_agg, on=['Timestamp', 'Slice_ID'])

    # Calculate throughput share
    total_throughput = df_merged.groupby('Timestamp')['Tx_brate_DL_Mbps'].transform('sum')
    df_merged['throughput_share'] = df_merged['Tx_brate_DL_Mbps'] / total_throughput
    df_merged['throughput_share'] = df_merged['throughput_share'].fillna(0)
    
    print("Preprocessing with new features complete.")
    return df_merged

if __name__ == '__main__':
    ROOT_SEARCH_PATH = r'C:\Users\Lenovo\AI_Spectrum_Allocation\colosseum-oran-coloran-dataset'
    search_pattern = os.path.join(ROOT_SEARCH_PATH, '**/slices_bs1/*_metrics.csv')
    all_metric_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(all_metric_files)} total metric files to process.")

    processed_data = load_and_prepare_data(all_metric_files)
    
    if processed_data is not None:
        processed_data.to_csv('processed_network_data.csv', index=False)
        print("\nProcessed data saved to processed_network_data.csv")
        print(f"Final dataset has {len(processed_data)} rows.")
        print(processed_data.head())