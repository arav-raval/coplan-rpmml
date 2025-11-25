# src/visualization/plots.py
"""Plotting utilities for experiment results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)


def plot_frequency_sweep(frequencies, results, save_path="results/frequency_sweep.png",
                         config=None):
    """Plot cost metrics vs broadcast frequency."""
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    # Build config text for subtitle
    config_parts = []
    if 'n_agents' in config: config_parts.append(f"Agents: {config['n_agents']}")
    if 'num_trials' in config: config_parts.append(f"Trials: {config['num_trials']}")
    if 'comm_range' in config: config_parts.append(f"Range: {config['comm_range']:.0f}px")
    if 'msg_length' in config: config_parts.append(f"MsgLen: {config['msg_length']}")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Title and subtitle
    fig.suptitle("Effect of Broadcast Frequency on Performance", fontsize=14, fontweight='bold', y=0.98)
    if config_text:
        fig.text(0.5, 0.94, config_text, ha='center', fontsize=10, 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Extract data
    costs = [r['cost_mean'] for r in results]
    replans = [r['replan_mean'] for r in results]
    separations = [r['min_separation_mean'] for r in results]
    times = [r['time_mean'] for r in results]
    
    # Find optimal
    opt_idx = np.argmin(costs)
    opt_freq = frequencies[opt_idx]
    
    # Plot 1: Cost vs Frequency
    ax1 = axes[0, 0]
    ax1.plot(frequencies, costs, marker='o', linewidth=2, markersize=6)
    ax1.axvline(x=opt_freq, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {opt_freq:.2f} Hz')
    ax1.scatter([opt_freq], [costs[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax1.set_xlabel("Broadcast Frequency (Hz)")
    ax1.set_ylabel("Total Cost")
    ax1.set_title("Cost vs Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Replanning Count
    ax2 = axes[0, 1]
    ax2.plot(frequencies, replans, marker='s', color='orange', linewidth=2)
    ax2.axvline(x=opt_freq, color='red', linestyle='--', alpha=0.7)
    ax2.scatter([opt_freq], [replans[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax2.set_xlabel("Broadcast Frequency (Hz)")
    ax2.set_ylabel("Replan Count")
    ax2.set_title("Replanning Frequency")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Safety Margin
    ax3 = axes[1, 0]
    ax3.plot(frequencies, separations, marker='^', color='green', linewidth=2)
    ax3.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Collision threshold')
    ax3.axvline(x=opt_freq, color='red', linestyle='--', alpha=0.7)
    ax3.scatter([opt_freq], [separations[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax3.fill_between(frequencies, 0, 30, alpha=0.2, color='red', label='Danger zone')
    ax3.set_xlabel("Broadcast Frequency (Hz)")
    ax3.set_ylabel("Min Separation (px)")
    ax3.set_title("Safety Margin (higher = safer)")
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # Plot 4: Traversal Time
    ax4 = axes[1, 1]
    ax4.plot(frequencies, times, marker='d', color='purple', linewidth=2)
    ax4.axvline(x=opt_freq, color='red', linestyle='--', alpha=0.7)
    ax4.scatter([opt_freq], [times[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax4.set_xlabel("Broadcast Frequency (Hz)")
    ax4.set_ylabel("Total Time (s)")
    ax4.set_title("Traversal Time")
    ax4.grid(True, alpha=0.3)
    
    # Optimal result at bottom
    opt_text = (f"★ Optimal: freq={opt_freq:.2f} Hz, cost={costs[opt_idx]:.1f}, "
                f"safety={separations[opt_idx]:.1f}px, replans={replans[opt_idx]:.1f}")
    fig.text(0.5, 0.02, opt_text, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_range_sweep(ranges, results, save_path="results/range_sweep.png", config=None):
    """Plot cost metrics vs communication range."""
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    config_parts = []
    if 'n_agents' in config: config_parts.append(f"Agents: {config['n_agents']}")
    if 'num_trials' in config: config_parts.append(f"Trials: {config['num_trials']}")
    if 'frequency' in config: config_parts.append(f"Freq: {config['frequency']:.2f}Hz")
    if 'msg_length' in config: config_parts.append(f"MsgLen: {config['msg_length']}")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fig.suptitle("Effect of Communication Range on Performance", fontsize=14, fontweight='bold', y=0.98)
    if config_text:
        fig.text(0.5, 0.94, config_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    costs = [r['cost_mean'] for r in results]
    replans = [r['replan_mean'] for r in results]
    separations = [r['min_separation_mean'] for r in results]
    collision_rates = [r['collision_rate'] for r in results]
    
    opt_idx = np.argmin(costs)
    opt_range = ranges[opt_idx]
    
    ax1 = axes[0, 0]
    ax1.plot(ranges, costs, marker='o', linewidth=2)
    ax1.axvline(x=opt_range, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {opt_range:.0f}px')
    ax1.scatter([opt_range], [costs[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax1.set_xlabel("Communication Range (px)")
    ax1.set_ylabel("Total Cost")
    ax1.set_title("Cost vs Range")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(ranges, replans, marker='s', color='orange', linewidth=2)
    ax2.axvline(x=opt_range, color='red', linestyle='--', alpha=0.7)
    ax2.scatter([opt_range], [replans[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax2.set_xlabel("Communication Range (px)")
    ax2.set_ylabel("Replan Count")
    ax2.set_title("Replanning Count")
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(ranges, separations, marker='^', color='green', linewidth=2)
    ax3.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Collision threshold')
    ax3.axvline(x=opt_range, color='red', linestyle='--', alpha=0.7)
    ax3.scatter([opt_range], [separations[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax3.fill_between(ranges, 0, 30, alpha=0.2, color='red', label='Danger zone')
    ax3.set_xlabel("Communication Range (px)")
    ax3.set_ylabel("Min Separation (px)")
    ax3.set_title("Safety Margin")
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    ax4 = axes[1, 1]
    bar_width = (ranges[1]-ranges[0])*0.7 if len(ranges) > 1 else 20
    ax4.bar(ranges, collision_rates, width=bar_width, color='red', alpha=0.7)
    ax4.axvline(x=opt_range, color='blue', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Communication Range (px)")
    ax4.set_ylabel("Collision Rate")
    ax4.set_title("Collision Rate (lower = better)")
    ax4.grid(True, alpha=0.3)
    
    opt_text = (f"★ Optimal: range={opt_range:.0f}px, cost={costs[opt_idx]:.1f}, "
                f"safety={separations[opt_idx]:.1f}px, collisions={collision_rates[opt_idx]:.0%}")
    fig.text(0.5, 0.02, opt_text, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_agent_sweep(agent_counts, results, save_path="results/agent_sweep.png", config=None):
    """Plot cost metrics vs number of agents."""
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    config_parts = []
    if 'num_trials' in config: config_parts.append(f"Trials: {config['num_trials']}")
    if 'frequency' in config: config_parts.append(f"Freq: {config['frequency']:.2f}Hz")
    if 'comm_range' in config: config_parts.append(f"Range: {config['comm_range']:.0f}px")
    if 'msg_length' in config: config_parts.append(f"MsgLen: {config['msg_length']}")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fig.suptitle("Effect of Agent Count on Performance", fontsize=14, fontweight='bold', y=0.98)
    if config_text:
        fig.text(0.5, 0.94, config_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    costs = [r['cost_mean'] for r in results]
    replans = [r['replan_mean'] for r in results]
    separations = [r['min_separation_mean'] for r in results]
    collision_rates = [r['collision_rate'] for r in results]
    times = [r['time_mean'] for r in results]
    
    # Plot 1: Cost vs Agents
    ax1 = axes[0, 0]
    ax1.plot(agent_counts, costs, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Agents")
    ax1.set_ylabel("Total Cost")
    ax1.set_title("Cost vs Agent Count")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(agent_counts)
    
    # Plot 2: Replanning Count
    ax2 = axes[0, 1]
    ax2.plot(agent_counts, replans, marker='s', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Agents")
    ax2.set_ylabel("Replan Count")
    ax2.set_title("Replanning Events vs Agent Count")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(agent_counts)
    
    # Plot 3: Safety Margin
    ax3 = axes[1, 0]
    ax3.plot(agent_counts, separations, marker='^', color='green', linewidth=2, markersize=8)
    ax3.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Collision threshold')
    ax3.fill_between(agent_counts, 0, 30, alpha=0.2, color='red', label='Danger zone')
    ax3.set_xlabel("Number of Agents")
    ax3.set_ylabel("Min Separation (px)")
    ax3.set_title("Safety Margin vs Agent Count")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(agent_counts)
    ax3.set_ylim(bottom=0)
    
    # Plot 4: Collision Rate
    ax4 = axes[1, 1]
    colors = ['green' if cr == 0 else 'orange' if cr < 0.3 else 'red' for cr in collision_rates]
    ax4.bar(agent_counts, collision_rates, color=colors, alpha=0.7, width=0.6)
    ax4.set_xlabel("Number of Agents")
    ax4.set_ylabel("Collision Rate")
    ax4.set_title("Collision Rate vs Agent Count")
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(agent_counts)
    ax4.set_ylim(0, max(0.1, max(collision_rates) * 1.1))
    
    # Summary at bottom
    summary = (f"Agent range: {agent_counts[0]}-{agent_counts[-1]} | "
               f"Cost range: {min(costs):.0f}-{max(costs):.0f} | "
               f"Collision rates: {min(collision_rates):.0%}-{max(collision_rates):.0%}")
    fig.text(0.5, 0.02, summary, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cost_landscape(freq_values, range_values, cost_matrix, 
                         save_path="results/cost_landscape.png", config=None):
    """Plot 2D heatmap of cost over frequency and range."""
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    config_parts = []
    if 'n_agents' in config: config_parts.append(f"Agents: {config['n_agents']}")
    if 'num_trials' in config: config_parts.append(f"Trials: {config['num_trials']}")
    if 'msg_length' in config: config_parts.append(f"MsgLen: {config['msg_length']}")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cost_matrix, aspect='auto', origin='lower',
                   extent=[range_values[0], range_values[-1], 
                          freq_values[0], freq_values[-1]],
                   cmap='RdYlGn_r')
    
    plt.colorbar(im, label='Cost')
    ax.set_xlabel("Communication Range (px)", fontsize=12)
    ax.set_ylabel("Broadcast Frequency (Hz)", fontsize=12)
    
    title = "Cost Landscape: Frequency vs Range"
    if config_text:
        title += f"\n{config_text}"
    ax.set_title(title, fontsize=12)
    
    # Find and mark minimum
    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    min_freq = freq_values[min_idx[0]]
    min_range = range_values[min_idx[1]]
    min_cost = cost_matrix[min_idx]
    
    ax.plot(min_range, min_freq, 'k*', markersize=20, 
            label=f'Optimal: f={min_freq:.2f}Hz, r={min_range:.0f}px')
    ax.plot(min_range, min_freq, 'w*', markersize=12)
    ax.legend(loc='upper right', fontsize=10)
    
    # Contours
    X, Y = np.meshgrid(range_values, freq_values)
    contours = ax.contour(X, Y, cost_matrix, levels=6, colors='black', alpha=0.3)
    ax.clabel(contours, inline=True, fontsize=8)
    
    fig.text(0.5, 0.02, 
             f"★ Optimal: freq={min_freq:.2f} Hz, range={min_range:.0f}px, cost={min_cost:.1f}",
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def identify_plateaus(values, costs, threshold=0.05):
    """Find regions where cost gradient is near zero."""
    if len(values) < 3:
        return []
    
    gradients = np.gradient(costs, values)
    cost_range = np.max(costs) - np.min(costs)
    if cost_range < 1e-6:
        return [(0, len(values)-1)]
    
    normalized_grad = np.abs(gradients) / cost_range
    
    plateaus = []
    in_plateau = False
    start_idx = 0
    
    for i, grad in enumerate(normalized_grad):
        if grad < threshold:
            if not in_plateau:
                in_plateau = True
                start_idx = i
        else:
            if in_plateau:
                in_plateau = False
                if i - start_idx >= 2:
                    plateaus.append((start_idx, i-1))
    
    if in_plateau and len(values) - start_idx >= 2:
        plateaus.append((start_idx, len(values)-1))
    
    return plateaus