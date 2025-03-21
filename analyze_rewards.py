import numpy as np
import torch
import os
from pathlib import Path
import argparse

def analyze_rewards(experiment_path):
    """
    Analyze the rewards_intr.memmap file from an experiment.
    
    Args:
        experiment_path (str): Path to the experiment directory containing the memmap files
    """
    # Convert to Path object
    exp_path = Path(experiment_path)
    
    # Path to the memmap buffer directory
    memmap_dir = exp_path / "version_0" / "memmap_buffer" / "rank_0"
    
    # Load the rewards_intr.memmap file
    rewards_path = memmap_dir / "rewards_intr.memmap"
    
    if not rewards_path.exists():
        print(f"Error: Could not find rewards_intr.memmap file in {memmap_dir}")
        return
    
    try:
        # Load the memmap array
        rewards = np.memmap(rewards_path, dtype=np.float32, mode='r')
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Calculate statistics
    mean = np.mean(rewards)
    variance = np.var(rewards)
    
    # Normalize the data
    normalized_rewards = (rewards - mean) / np.sqrt(variance)
    norm_min = np.min(normalized_rewards)
    norm_max = np.max(normalized_rewards)
    
    # Create results dictionary
    results = {
        'mean': float(mean),
        'variance': float(variance),
        'normalized_min': float(norm_min),
        'normalized_max': float(norm_max),
        'shape': rewards.shape,
        'dtype': str(rewards.dtype)
    }
    
    # Create stats directory if it doesn't exist
    stats_dir = exp_path / "version_0" / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results to stats folder
    results_path = stats_dir / "rewards_intr_stats.npy"
    np.save(results_path, results)
    
    # Print results
    print("\nReward Statistics:")
    print(f"Mean: {mean:.4f}")
    print(f"Variance: {variance:.4f}")
    
    print("\nNormalized Reward Statistics:")
    print(f"Min: {norm_min:.4f}")
    print(f"Max: {norm_max:.4f}")
    
    print("\nArray Dtype:", rewards.dtype)
    print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze rewards from an experiment')
    parser.add_argument('experiment_path', type=str, help='Path to the experiment directory')
    args = parser.parse_args()
    
    analyze_rewards(args.experiment_path) 