import pandas as pd
import numpy as np

class ProportionalFairnessAllocator:
    """
    A simple Proportional Fairness allocator for network slices.
    """
    def __init__(self, slices, alpha=0.1):
        """
        Initializes the allocator.
        :param slices: A list of unique slice IDs.
        :param alpha: The smoothing factor for the moving average of throughput.
        """
        self.slices = slices
        self.alpha = alpha
        # Historical average throughput for each slice
        self.historical_throughput = {s: 1.0 for s in slices} # Start with 1 to avoid division by zero

    def update_historical_throughput(self, slice_id, current_throughput):
        """
        Updates the moving average of throughput for a slice.
        """
        self.historical_throughput[slice_id] = (1 - self.alpha) * self.historical_throughput[slice_id] + self.alpha * current_throughput

    def decide(self, current_state):
        """
        Makes an allocation decision based on the PF metric.
        :param current_state: A DataFrame containing the current state for all slices at one timestamp.
                              It must have 'Slice_ID' and 'avg_cqi'.
        :return: A dictionary of {slice_id: allocation_share}.
        """
        metrics = {}
        for _, row in current_state.iterrows():
            slice_id = row['Slice_ID']
            # Instantaneous rate proxy: avg_cqi
            instantaneous_rate = row['avg_cqi']
            # Historical rate: our moving average
            historical_rate = self.historical_throughput[slice_id]
            
            # Calculate PF metric
            metrics[slice_id] = instantaneous_rate / historical_rate

        total_metric = sum(metrics.values())
        if total_metric == 0:
            # Fallback to equal allocation if all metrics are zero
            num_slices = len(self.slices)
            return {s: 1.0 / num_slices for s in self.slices}

        # Normalize metrics to get allocation shares
        allocation = {s: m / total_metric for s, m in metrics.items()}
        return allocation

# This file is a module, so no main execution block is needed for now.
# We will import and use this class in the main simulation script.