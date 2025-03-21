import numpy as np
import argparse
from pathlib import Path


def print_norm_stats(experiment_path):
    """Print the shapes and values of the norm_stats.npy file."""
    experiment_path = Path(experiment_path)
    stats_path = experiment_path / "norm_stats.npy"
    
    if not stats_path.exists():
        print(f"Error: norm_stats.npy not found in {experiment_path}")
        return
    
    try:
        stats = np.load(stats_path, allow_pickle=True).item()
        
        # Print rewards stats
        print("\n=== Rewards Statistics ===")
        for reward_type in ['rewards_intr', 'rewards_extr']:
            print(f"\n{reward_type}:")
            for stat_name, stat_value in stats[reward_type].items():
                print(f"  {stat_name}:")
                print(f"    Shape: {stat_value.shape}")
                print(f"    Value: {stat_value}")
        
        # Print states stats
        print("\n=== States Statistics ===")
        for state_key, state_stats in stats['states'].items():
            print(f"\nState: {state_key}")
            for stat_name, stat_value in state_stats.items():
                print(f"  {stat_name}:")
                print(f"    Shape: {stat_value.shape}")
                print(f"    Value: {stat_value}")
                
    except Exception as e:
        print(f"Error processing norm_stats.npy: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", type=str, help="Path to the experiment directory")
    args = parser.parse_args()
    
    print_norm_stats(args.experiment_path) 