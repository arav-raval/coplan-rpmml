"""
Evaluate trained MARL policy and compare with baselines.

Usage:
    python evaluate_marl.py models/marl_ppo/ppo_marl_final.zip --n_episodes 50
"""

import numpy as np
import torch
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from src.envs.marl_comm_env import MARLCommEnv
from src.simulation import run_batch_multiple_seeds
from src.config import EXPERIMENT_CONFIG
import matplotlib.pyplot as plt


def evaluate_policy(model_path, n_episodes=50, n_agents=4, deterministic=True, verbose=True):
    """
    Evaluate learned MARL policy.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        n_agents: Number of agents
        deterministic: Use deterministic policy
        verbose: Print progress
        
    Returns:
        List of episode results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATING MARL POLICY")
        print(f"{'='*70}")
        print(f"Model:     {model_path}")
        print(f"Episodes:  {n_episodes}")
        print(f"Agents:    {n_agents}")
        print(f"{'='*70}\n")
    
    # Load model
    model = PPO.load(model_path)
    env = MARLCommEnv(n_agents=n_agents)
    
    results = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        episode_cost = 0
        episode_rewards = []
        episode_actions = []
        episode_info = []
        
        done = False
        step = 0
        
        while not done:
            # Get action using shared policy 
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action) if np.isscalar(action) else int(action.item())
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_info.append(info)
            episode_cost = info['cumulative_cost']
            
            done = terminated or truncated
            step += 1
        
        result = {
            'episode': ep,
            'cost': episode_cost,
            'collisions': info['collisions'],
            'success': info['both_reached_goal'],
            'time': info['time'],
            'min_sep': info['min_separation'],
            'avg_sep': info['avg_separation'],
            'replans': info['replans'],
            'messages': info['messages'],
            'steps': step,
            'actions': episode_actions,
            'rewards': episode_rewards,
        }
        results.append(result)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  Completed {ep + 1}/{n_episodes} episodes...")
    
    # Analyze results
    if verbose:
        print_results_summary(results)
    
    return results


def print_results_summary(results):
    """Print summary statistics."""
    costs = [r['cost'] for r in results]
    collisions = [r['collisions'] for r in results]
    success_rate = sum(r['success'] for r in results) / len(results)
    times = [r['time'] for r in results]
    min_seps = [r['min_sep'] for r in results]
    
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY (n={len(results)})")
    print(f"{'='*70}")
    print(f"Cost:")
    print(f"  Mean:   {np.mean(costs):8.1f}")
    print(f"  Std:    {np.std(costs):8.1f}")
    print(f"  Median: {np.median(costs):8.1f}")
    print(f"  Min:    {np.min(costs):8.1f}")
    print(f"  Max:    {np.max(costs):8.1f}")
    print(f"\nSafety:")
    print(f"  Collision rate:  {np.mean(collisions):.3f}")
    print(f"  Success rate:    {success_rate:.1%}")
    print(f"  Min separation:  {np.mean(min_seps):6.1f} ± {np.std(min_seps):5.1f} px")
    print(f"\nPerformance:")
    print(f"  Avg time:        {np.mean(times):6.2f} ± {np.std(times):5.2f} s")
    print(f"  Avg replans:     {np.mean([r['replans'] for r in results]):6.1f}")
    print(f"  Avg messages:    {np.mean([r['messages'] for r in results]):6.1f}")
    print(f"{'='*70}\n")


def compare_with_baseline(rl_results, n_agents=4, num_seeds=50):
    """Compare RL policy with fixed optimal baseline from sweeps."""
    
    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISON")
    print(f"{'='*70}")
    
    # Run baseline 
    print("\n1. Running fixed baseline (freq=15 steps, msg_len=9)...")
    
    baseline_result = run_batch_multiple_seeds(
        comm_enabled=True,
        broadcast_interval_steps=15,
        comm_range=EXPERIMENT_CONFIG['comm_range'],
        msg_length=9,
        n_agents=n_agents,
        num_seeds=num_seeds,
        verbose=True,
    )
    
    # Compare
    rl_cost_mean = np.mean([r['cost'] for r in rl_results])
    rl_cost_std = np.std([r['cost'] for r in rl_results])
    rl_collision_rate = np.mean([r['collisions'] for r in rl_results])
    
    baseline_cost = baseline_result['cost_mean']
    baseline_cost_std = baseline_result['cost_std']
    baseline_collision_rate = baseline_result['collision_rate']
    
    improvement = (baseline_cost - rl_cost_mean) / baseline_cost * 100
    
    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Baseline':>15} {'RL Policy':>15} {'Change':>12}")
    print(f"{'-'*70}")
    print(f"{'Cost':<25} {baseline_cost:>15.1f} {rl_cost_mean:>15.1f} {improvement:>11.1f}%")
    print(f"{'Cost (std)':<25} {baseline_cost_std:>15.1f} {rl_cost_std:>15.1f}")
    print(f"{'Collision rate':<25} {baseline_collision_rate:>15.3f} {rl_collision_rate:>15.3f}")
    print(f"{'='*70}\n")
    
    if improvement > 0:
        print(f"RL policy achieves {improvement:.1f}% cost reduction!")
    elif improvement > -5:
        print(f"RL policy performs similarly to baseline ({improvement:.1f}%)")
    else:
        print(f"RL policy underperforms baseline ({improvement:.1f}%)")
        print(f"   → May need more training or hyperparameter tuning")
    
    return {
        'rl_cost': rl_cost_mean,
        'baseline_cost': baseline_cost,
        'improvement_pct': improvement,
        'rl_collision_rate': rl_collision_rate,
        'baseline_collision_rate': baseline_collision_rate,
    }


def analyze_action_distribution(results):
    """Analyze which communication strategies the policy learned."""
    
    print(f"\n{'='*70}")
    print(f"ACTION DISTRIBUTION ANALYSIS")
    print(f"{'='*70}\n")
    
    # Collect all actions across all episodes
    all_actions = []
    for r in results:
        for step_action in r['actions']:
            all_actions.append(step_action)
    
    if not all_actions:
        print("No actions recorded in results!")
        return
    
    # Count action frequencies
    action_counts = {}
    for action in all_actions:
        action_counts[action] = action_counts.get(action, 0) + 1
    
    # Get action pairs from environment
    env = MARLCommEnv()
    
    # Sort by frequency
    sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 most used actions:")
    print(f"{'Rank':<6} {'Action':<8} {'(Freq, Msg)':<15} {'Count':<10} {'%':<8}")
    print(f"{'-'*60}")
    
    for rank, (action_idx, count) in enumerate(sorted_actions[:10], 1):
        if 0 <= action_idx < len(env.action_pairs):
            freq, msg_len = env.action_pairs[action_idx]
            pct = count / len(all_actions) * 100
            print(f"{rank:<6} {action_idx:<8} ({freq:>2}, {msg_len:>2}){'':<8} {count:<10} {pct:>6.1f}%")
        else:
            print(f"{rank:<6} {action_idx:<8} {'INVALID':<15} {count:<10}")
    
    print(f"\nTotal unique actions used: {len(action_counts)} / {len(env.action_pairs)}")
    print(f"{'='*70}\n")


def plot_results(results, save_path="results/marl_evaluation.png"):
    """Create visualization of evaluation results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cost over episodes
    ax = axes[0, 0]
    costs = [r['cost'] for r in results]
    ax.plot(costs, alpha=0.7, linewidth=1)
    ax.axhline(np.mean(costs), color='r', linestyle='--', label=f'Mean: {np.mean(costs):.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Cost')
    ax.set_title('Cost per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Min separation distribution
    ax = axes[0, 1]
    min_seps = [r['min_sep'] for r in results]
    ax.hist(min_seps, bins=20, alpha=0.7, edgecolor='black')
    ax.axvline(10, color='r', linestyle='--', label='Safety threshold (10px)')
    ax.set_xlabel('Minimum Separation (px)')
    ax.set_ylabel('Frequency')
    ax.set_title('Safety: Minimum Separation Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Success rate over time
    ax = axes[1, 0]
    window = min(10, len(results))  
    if window > 1:
        success_rolling = np.convolve([r['success'] for r in results], 
                                       np.ones(window)/window, mode='valid')
        ax.plot(success_rolling, linewidth=2)
    else:
        ax.plot([r['success'] for r in results], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (rolling avg)')
    ax.set_title(f'Success Rate (window={window})')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)
    
    # Time to completion
    ax = axes[1, 1]
    times = [r['time'] for r in results]
    ax.plot(times, alpha=0.7, linewidth=1, label='Episode time')
    ax.axhline(np.mean(times), color='r', linestyle='--', label=f'Mean: {np.mean(times):.1f}s')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Episode Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_dir = os.path.dirname(save_path)
    if save_dir:  
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained MARL policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model (e.g., models/marl_ppo/ppo_marl_final.zip)"
    )
    parser.add_argument(
        "--n_episodes", type=int, default=50,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--n_agents", type=int, default=4,
        help="Number of agents"
    )
    parser.add_argument(
        "--compare_baseline", action="store_true",
        help="Compare with fixed baseline (requires additional simulation time)"
    )
    parser.add_argument(
        "--baseline_seeds", type=int, default=50,
        help="Number of seeds for baseline comparison"
    )
    parser.add_argument(
        "--analyze_actions", action="store_true",
        help="Analyze action distribution"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate evaluation plots"
    )
    
    args = parser.parse_args()
    
    # Evaluate RL policy
    results = evaluate_policy(
        args.model_path,
        n_episodes=args.n_episodes,
        n_agents=args.n_agents
    )
    
    # Optional analyses
    if args.analyze_actions:
        analyze_action_distribution(results)
    
    if args.compare_baseline:
        comparison = compare_with_baseline(
            results,
            n_agents=args.n_agents,
            num_seeds=args.baseline_seeds
        )
    
    if args.plot:
        plot_results(results)
    
    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()
