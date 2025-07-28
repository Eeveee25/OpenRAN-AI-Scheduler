import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm

from baseline_allocator import ProportionalFairnessAllocator

def jains_fairness(throughputs):
    if not any(throughputs):
        return 1.0
    sum_of_throughputs = sum(throughputs)
    sum_of_squares = sum([x**2 for x in throughputs])
    n = len(throughputs)
    if sum_of_squares == 0:
        return 1.0
    return (sum_of_throughputs**2) / (n * sum_of_squares)

def calculate_kpis(allocation, state, max_slice_throughput=50):
    throughputs = {}
    for slice_id, share in allocation.items():
        slice_state = state[state['Slice_ID'] == slice_id]
        if slice_state.empty:
            continue
        slice_cqi = slice_state['avg_cqi'].iloc[0]
        achieved_throughput = share * max_slice_throughput * (slice_cqi / 15.0)
        throughputs[slice_id] = achieved_throughput
    
    total_throughput = sum(throughputs.values())
    fairness = jains_fairness(list(throughputs.values()))
    return {'throughput': total_throughput, 'fairness': fairness, 'slice_throughputs': throughputs}

def run_simulation():
    df = pd.read_csv('processed_network_data.csv')
    ai_model = joblib.load('rf_allocator_model.pkl')
    
    allocation_map = {0: 0.1, 1: 0.35, 2: 0.75}
    slice_ids = sorted(df['Slice_ID'].unique())
    pf_allocator = ProportionalFairnessAllocator(slice_ids)
    
    results = []
    
    timestamps = sorted(df['Timestamp'].unique())
    for ts in tqdm(timestamps, desc="Running Simulation"):
        current_state = df[df['Timestamp'] == ts].copy()
        if current_state.empty:
            continue
        
        # --- CORRECTED FEATURE LIST ---
        features = ['avg_cqi', 'std_cqi', 'avg_rsrp', 'std_rsrp', 'num_ues']
        X_current = current_state[features]
        # ---
        
        ai_predictions = ai_model.predict(X_current)
        ai_allocations_raw = {row['Slice_ID']: allocation_map[pred] for (_, row), pred in zip(current_state.iterrows(), ai_predictions)}
        total_pred_share = sum(ai_allocations_raw.values())
        ai_allocation = {s: v / total_pred_share for s, v in ai_allocations_raw.items()} if total_pred_share > 0 else {s: 1/len(slice_ids) for s in slice_ids}
        
        pf_allocation = pf_allocator.decide(current_state)

        ai_kpis = calculate_kpis(ai_allocation, current_state)
        pf_kpis = calculate_kpis(pf_allocation, current_state)
        
        for slice_id, throughput in pf_kpis['slice_throughputs'].items():
            pf_allocator.update_historical_throughput(slice_id, throughput)
        
        results.append({
            'timestamp': ts,
            'ai_throughput': ai_kpis['throughput'],
            'ai_fairness': ai_kpis['fairness'],
            'pf_throughput': pf_kpis['throughput'],
            'pf_fairness': pf_kpis['fairness']
        })

    return pd.DataFrame(results)

if __name__ == '__main__':
    simulation_results = run_simulation()
    simulation_results.to_csv('simulation_results.csv', index=False)
    print("\nSimulation complete. Results saved to simulation_results.csv")
    print(simulation_results.describe())