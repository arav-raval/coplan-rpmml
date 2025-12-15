# src/visualization/plots.py
"""Plotting utilities for experiment results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    Path("results").mkdir(exist_ok=True)


def plot_frequency_sweep(frequencies_steps, results, save_path="results/frequency_sweep.png",
                         config=None):
    """
    Plot cost metrics vs broadcast frequency (in steps per communication).
    
    Args:
        frequencies_steps: List/array of step intervals (steps per communication)
        results: List of result dictionaries from experiments
        save_path: Path to save the plot
        config: Optional configuration dictionary for subtitle
    """
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    # Build config text for subtitle
    config_parts = []
    if 'n_agents' in config: config_parts.append(f"Agents: {config['n_agents']}")
    if 'num_trials' in config: config_parts.append(f"Trials: {config['num_trials']}")
    if 'num_seeds' in config: config_parts.append(f"Seeds: {config['num_seeds']}")
    if 'comm_range' in config: config_parts.append(f"Range: {config['comm_range']:.0f}px")
    if 'msg_length' in config: config_parts.append(f"MsgLen: {config['msg_length']}")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Title and subtitle
    fig.suptitle("Effect of Communication Frequency on Performance", fontsize=14, fontweight='bold', y=0.98)
    if config_text:
        fig.text(0.5, 0.94, config_text, ha='center', fontsize=10, 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Extract data with error bars
    costs = [r['cost_mean'] for r in results]
    cost_stds = [r.get('cost_std', 0) for r in results]
    replans = [r['replan_mean'] for r in results]
    replan_stds = [r.get('replan_std', 0) for r in results]
    separations = [r['min_separation_mean'] for r in results]
    separation_stds = [r.get('min_separation_std', 0) for r in results]
    avg_separations = [r['avg_separation_mean'] for r in results]
    avg_separation_stds = [r.get('avg_separation_std', 0) for r in results]
    collision_counts = [r['collision_count_mean'] for r in results]
    collision_count_stds = [r.get('collision_count_std', 0) for r in results]
    collision_rates = [r.get('collision_rate', 0) * 100 for r in results]  # Convert to percentage
    times = [r['time_mean'] for r in results]
    time_stds = [r.get('time_std', 0) for r in results]
    
    # Find optimal
    opt_idx = np.argmin(costs)
    opt_steps = frequencies_steps[opt_idx]
    
    # Plot 1: Cost vs Steps per Communication with error bars
    ax1 = axes[0, 0]
    ax1.plot(frequencies_steps, costs, marker='o', linewidth=2, markersize=6, label='Mean')
    ax1.fill_between(frequencies_steps, 
                     np.array(costs) - np.array(cost_stds),
                     np.array(costs) + np.array(cost_stds),
                     alpha=0.2, label='±1 std')
    ax1.axvline(x=opt_steps, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {opt_steps} steps')
    ax1.scatter([opt_steps], [costs[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax1.set_xlabel("Steps per Communication", fontsize=11)
    ax1.set_ylabel("Total Cost", fontsize=11)
    ax1.set_title("Cost vs Communication Frequency")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Replanning Count with error bars
    ax2 = axes[0, 1]
    ax2.plot(frequencies_steps, replans, marker='s', color='orange', linewidth=2, label='Mean')
    ax2.fill_between(frequencies_steps,
                     np.array(replans) - np.array(replan_stds),
                     np.array(replans) + np.array(replan_stds),
                     alpha=0.2, color='orange', label='±1 std')
    ax2.axvline(x=opt_steps, color='red', linestyle='--', alpha=0.7)
    ax2.scatter([opt_steps], [replans[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax2.set_xlabel("Steps per Communication", fontsize=11)
    ax2.set_ylabel("Replan Count", fontsize=11)
    ax2.set_title("Replanning Frequency")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Separation (typical safety) with error bars
    ax3 = axes[1, 0]
    ax3.plot(frequencies_steps, avg_separations, marker='^', color='green', linewidth=2, label='Mean')
    ax3.fill_between(frequencies_steps,
                     np.array(avg_separations) - np.array(avg_separation_stds),
                     np.array(avg_separations) + np.array(avg_separation_stds),
                     alpha=0.2, color='green', label='±1 std')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Touching threshold')
    ax3.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Safe zone (>10px)')
    ax3.axvline(x=opt_steps, color='red', linestyle='--', alpha=0.7)
    ax3.scatter([opt_steps], [avg_separations[opt_idx]], color='red', s=150, zorder=5, marker='*')
    # Removed: Light green "Safe clearance" shading (was unnecessary visual clutter)
    ax3.set_xlabel("Steps per Communication", fontsize=11)
    ax3.set_ylabel("Average Clearance (px)", fontsize=11)
    ax3.set_title("Average Agent Separation (higher = safer)", fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    y_max = max(avg_separations) + max(avg_separation_stds) + 5
    ax3.set_ylim(bottom=min(0, min(avg_separations) - max(avg_separation_stds) - 2), top=y_max)
    
    # Plot 4: Collision Analysis - CLEANED UP
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()  # Create twin axis for collision rate
    
    # Ensure collision counts are never negative (fix data issue)
    collision_counts_clean = [max(0, c) for c in collision_counts]
    collision_count_stds_clean = [max(0, s) for s in collision_count_stds]
    
    # Bar chart for collision count - simplified (solid color, clean look)
    bars = ax4.bar(frequencies_steps, collision_counts_clean, width=3, 
                   color='#8B0000', alpha=1.0, label='Collision Count', 
                   yerr=collision_count_stds_clean,
                   capsize=3, error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    # Line plot for collision rate - NO 'x' markers, cleaner line
    line = ax4_twin.plot(frequencies_steps, collision_rates, 
                         color='red', linewidth=2.5, label='Collision Rate (%)', zorder=10)
    
    ax4.axvline(x=opt_steps, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Optimal')
    ax4.set_xlabel("Steps per Communication", fontsize=11)
    ax4.set_ylabel("Collision Count (avg per run)", fontsize=11, color='#8B0000')
    ax4_twin.set_ylabel("Collision Rate (%)", fontsize=11, color='red')
    ax4.set_title("Collision Analysis (lower = safer)", fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#8B0000')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Set y-axis minimum to 0 for collision count
    ax4.set_ylim(bottom=0)
    
    # Highlight zero collision zone (only if there are actual zero collisions)
    if min(collision_counts_clean) == 0:
        zero_collision_steps = [f for f, c in zip(frequencies_steps, collision_counts_clean) if c == 0]
        if zero_collision_steps:
            max_count = max(collision_counts_clean) if max(collision_counts_clean) > 0 else 1
            ax4.fill_betweenx([0, max_count * 1.1], 
                             min(zero_collision_steps), max(zero_collision_steps),
                             alpha=0.1, color='green', label='Zero collision zone')
    
    # Convert optimal steps to Hz for display
    from src.config import SIMULATION_FPS
    opt_freq_hz = SIMULATION_FPS / opt_steps if opt_steps > 0 else 0
    
    # Optimal result at bottom
    opt_text = (f"★ Optimal: {opt_steps} steps/comm ({opt_freq_hz:.2f} Hz) | "
                f"Cost={costs[opt_idx]:.1f} | Avg Safety={avg_separations[opt_idx]:.1f}px | "
                f"Replans={replans[opt_idx]:.1f} | Collisions={collision_counts[opt_idx]:.1f} ({collision_rates[opt_idx]:.0f}%)")
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
    if 'num_seeds' in config: config_parts.append(f"Seeds: {config['num_seeds']}")
    if 'frequency' in config: config_parts.append(f"Freq: {config['frequency']:.2f}Hz")
    if 'msg_length' in config: config_parts.append(f"MsgLen: {config['msg_length']}")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fig.suptitle("Effect of Communication Range on Performance", fontsize=14, fontweight='bold', y=0.98)
    if config_text:
        fig.text(0.5, 0.94, config_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    costs = [r['cost_mean'] for r in results]
    cost_stds = [r.get('cost_std', 0) for r in results]
    replans = [r['replan_mean'] for r in results]
    replan_stds = [r.get('replan_std', 0) for r in results]
    separations = [r['min_separation_mean'] for r in results]
    separation_stds = [r.get('min_separation_std', 0) for r in results]
    collision_rates = [r['collision_rate'] for r in results]
    collision_stds = [r.get('collision_rate_std', 0) for r in results]
    
    opt_idx = np.argmin(costs)
    opt_range = ranges[opt_idx]
    
    ax1 = axes[0, 0]
    ax1.plot(ranges, costs, marker='o', linewidth=2, label='Mean')
    ax1.fill_between(ranges,
                     np.array(costs) - np.array(cost_stds),
                     np.array(costs) + np.array(cost_stds),
                     alpha=0.2, label='±1 std')
    ax1.axvline(x=opt_range, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {opt_range:.0f}px')
    ax1.scatter([opt_range], [costs[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax1.set_xlabel("Communication Range (px)")
    ax1.set_ylabel("Total Cost")
    ax1.set_title("Cost vs Range")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(ranges, replans, marker='s', color='orange', linewidth=2, label='Mean')
    ax2.fill_between(ranges,
                     np.array(replans) - np.array(replan_stds),
                     np.array(replans) + np.array(replan_stds),
                     alpha=0.2, color='orange', label='±1 std')
    ax2.axvline(x=opt_range, color='red', linestyle='--', alpha=0.7)
    ax2.scatter([opt_range], [replans[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax2.set_xlabel("Communication Range (px)")
    ax2.set_ylabel("Replan Count")
    ax2.set_title("Replanning Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(ranges, separations, marker='^', color='green', linewidth=2, label='Mean')
    ax3.fill_between(ranges,
                     np.array(separations) - np.array(separation_stds),
                     np.array(separations) + np.array(separation_stds),
                     alpha=0.2, color='green', label='±1 std')
    ax3.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Collision threshold')
    ax3.axvline(x=opt_range, color='red', linestyle='--', alpha=0.7)
    ax3.scatter([opt_range], [separations[opt_idx]], color='red', s=150, zorder=5, marker='*')
    ax3.fill_between(ranges, 0, 30, alpha=0.1, color='red', label='Danger zone')
    ax3.set_xlabel("Communication Range (px)")
    ax3.set_ylabel("Min Separation (px)")
    ax3.set_title("Safety Margin")
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    ax4 = axes[1, 1]
    bar_width = (ranges[1]-ranges[0])*0.7 if len(ranges) > 1 else 20
    ax4.bar(ranges, collision_rates, width=bar_width, color='red', alpha=0.7, label='Mean')
    # Error bars for collision rates
    ax4.errorbar(ranges, collision_rates, yerr=collision_stds, 
                fmt='none', color='black', capsize=3, capthick=1, label='±1 std')
    ax4.axvline(x=opt_range, color='blue', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Communication Range (px)")
    ax4.set_ylabel("Collision Rate")
    ax4.set_title("Collision Rate (lower = better)")
    ax4.legend()
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
    if 'num_seeds' in config: config_parts.append(f"Seeds: {config['num_seeds']}")
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
    cost_stds = [r.get('cost_std', 0) for r in results]
    replans = [r['replan_mean'] for r in results]
    replan_stds = [r.get('replan_std', 0) for r in results]
    separations = [r['min_separation_mean'] for r in results]
    separation_stds = [r.get('min_separation_std', 0) for r in results]
    collision_rates = [r['collision_rate'] for r in results]
    collision_stds = [r.get('collision_rate_std', 0) for r in results]
    times = [r['time_mean'] for r in results]
    time_stds = [r.get('time_std', 0) for r in results]
    
    # Plot 1: Cost vs Agents with error bars
    ax1 = axes[0, 0]
    ax1.plot(agent_counts, costs, marker='o', linewidth=2, markersize=8, label='Mean')
    ax1.fill_between(agent_counts,
                     np.array(costs) - np.array(cost_stds),
                     np.array(costs) + np.array(cost_stds),
                     alpha=0.2, label='±1 std')
    ax1.set_xlabel("Number of Agents")
    ax1.set_ylabel("Total Cost")
    ax1.set_title("Cost vs Agent Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(agent_counts)
    
    # Plot 2: Replanning Count with error bars
    ax2 = axes[0, 1]
    ax2.plot(agent_counts, replans, marker='s', color='orange', linewidth=2, markersize=8, label='Mean')
    ax2.fill_between(agent_counts,
                     np.array(replans) - np.array(replan_stds),
                     np.array(replans) + np.array(replan_stds),
                     alpha=0.2, color='orange', label='±1 std')
    ax2.set_xlabel("Number of Agents")
    ax2.set_ylabel("Replan Count")
    ax2.set_title("Replanning Events vs Agent Count")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(agent_counts)
    
    # Plot 3: Safety Margin with error bars
    ax3 = axes[1, 0]
    ax3.plot(agent_counts, separations, marker='^', color='green', linewidth=2, markersize=8, label='Mean')
    ax3.fill_between(agent_counts,
                     np.array(separations) - np.array(separation_stds),
                     np.array(separations) + np.array(separation_stds),
                     alpha=0.2, color='green', label='±1 std')
    ax3.axhline(y=30, color='red', linestyle='--', linewidth=2, label='Collision threshold')
    ax3.fill_between(agent_counts, 0, 30, alpha=0.1, color='red', label='Danger zone')
    ax3.set_xlabel("Number of Agents")
    ax3.set_ylabel("Min Separation (px)")
    ax3.set_title("Safety Margin vs Agent Count")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(agent_counts)
    ax3.set_ylim(bottom=0)
    
    # Plot 4: Collision Rate with error bars
    ax4 = axes[1, 1]
    colors = ['green' if cr == 0 else 'orange' if cr < 0.3 else 'red' for cr in collision_rates]
    ax4.bar(agent_counts, collision_rates, color=colors, alpha=0.7, width=0.6, label='Mean')
    ax4.errorbar(agent_counts, collision_rates, yerr=collision_stds,
                fmt='none', color='black', capsize=3, capthick=1, label='±1 std')
    ax4.set_xlabel("Number of Agents")
    ax4.set_ylabel("Collision Rate")
    ax4.set_title("Collision Rate vs Agent Count")
    ax4.legend()
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
    if 'num_seeds' in config: config_parts.append(f"Seeds: {config['num_seeds']}")
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


def plot_cost_landscape_freq_msg(freq_values, msg_length_values, cost_matrix, 
                                  save_path="results/cost_landscape.png", config=None):
    """Plot 2D heatmap of cost over frequency and message length."""
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    config_parts = []
    if 'n_agents' in config: config_parts.append(f"Agents: {config['n_agents']}")
    if 'num_trials' in config: config_parts.append(f"Trials: {config['num_trials']}")
    if 'num_seeds' in config: config_parts.append(f"Seeds: {config['num_seeds']}")
    if 'comm_range' in config: config_parts.append(f"Range: {config['comm_range']:.0f}px")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Transpose cost_matrix to match meshgrid shape
    # cost_matrix comes in as (msg_lengths, freq_steps) but meshgrid creates (freq_steps, msg_lengths)
    cost_matrix_T = cost_matrix.T
    
    im = ax.imshow(cost_matrix_T, aspect='auto', origin='lower',
                   extent=[msg_length_values[0], msg_length_values[-1], 
                          freq_values[0], freq_values[-1]],
                   cmap='RdYlGn_r')
    
    plt.colorbar(im, label='Cost')
    ax.set_xlabel("Message Length (waypoints)", fontsize=12)
    ax.set_ylabel("Broadcast Frequency (steps)", fontsize=12)
    
    title = "Cost Landscape: Frequency vs Message Length"
    if config_text:
        title += f"\n{config_text}"
    ax.set_title(title, fontsize=12)
    
    # Find and mark minimum (using original cost_matrix orientation)
    # cost_matrix is [msg_idx, freq_idx], so min_idx[0] = msg, min_idx[1] = freq
    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    min_msg = msg_length_values[min_idx[0]]
    min_freq = freq_values[min_idx[1]]
    min_cost = cost_matrix[min_idx]
    
    # Convert freq steps to Hz for display
    from src.config import SIMULATION_FPS
    min_freq_hz = SIMULATION_FPS / min_freq if min_freq > 0 else 0
    
    ax.plot(min_msg, min_freq, 'k*', markersize=20, 
            label=f'Optimal: f={min_freq:.0f} steps ({min_freq_hz:.2f}Hz), msg={min_msg}wp')
    ax.plot(min_msg, min_freq, 'w*', markersize=12)
    ax.legend(loc='upper right', fontsize=10)
    
    # Contours (use transposed matrix to match meshgrid shape)
    X, Y = np.meshgrid(msg_length_values, freq_values)
    contours = ax.contour(X, Y, cost_matrix_T, levels=6, colors='black', alpha=0.3)
    ax.clabel(contours, inline=True, fontsize=8)
    
    fig.text(0.5, 0.02, 
             f"★ Optimal: freq={min_freq:.0f} steps ({min_freq_hz:.2f} Hz), msg_length={min_msg} waypoints, cost={min_cost:.1f}",
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_2d_sweep_details(freq_steps, msg_lengths, results_matrix, 
                           save_path="results/2d_sweep_details.png", config=None):
    """
    Plot detailed 4-panel analysis for 2D sweep results (similar to frequency/msg_length sweeps).
    
    Args:
        freq_steps: Array of frequency step values
        msg_lengths: Array of message length values  
        results_matrix: Dict of matrices with keys: cost, replan, separation, collision_count, collision_rate
        save_path: Where to save the plot
        config: Optional config dict for subtitle
    """
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    # Build config text
    config_parts = []
    if 'n_agents' in config: config_parts.append(f"Agents: {config['n_agents']}")
    if 'num_seeds' in config: config_parts.append(f"Seeds: {config['num_seeds']}")
    if 'comm_range' in config: config_parts.append(f"Range: {config['comm_range']:.0f}px")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    fig.suptitle("2D Sweep: Frequency × Message Length Analysis", fontsize=14, fontweight='bold', y=0.98)
    if config_text:
        fig.text(0.5, 0.94, config_text, ha='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Convert freq steps to Hz for axis labels
    from src.config import SIMULATION_FPS
    freq_hz = SIMULATION_FPS / np.array(freq_steps)
    
    # Find optimal
    cost_matrix = results_matrix['cost']
    min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
    opt_msg = msg_lengths[min_idx[0]]
    opt_freq = freq_steps[min_idx[1]]
    opt_freq_hz = SIMULATION_FPS / opt_freq
    
    # Plot 1: Cost Heatmap
    ax1 = axes[0, 0]
    cost_T = cost_matrix.T  # Transpose for plotting
    im1 = ax1.imshow(cost_T, aspect='auto', origin='lower',
                     extent=[msg_lengths[0], msg_lengths[-1], freq_steps[0], freq_steps[-1]],
                     cmap='RdYlGn_r')
    plt.colorbar(im1, ax=ax1, label='Cost')
    ax1.plot(opt_msg, opt_freq, 'k*', markersize=15, label='Optimal')
    ax1.plot(opt_msg, opt_freq, 'w*', markersize=9)
    ax1.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax1.set_ylabel("Frequency (steps)", fontsize=11)
    ax1.set_title("Total Cost Landscape")
    ax1.legend()
    ax1.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    # Plot 2: Replanning Heatmap
    ax2 = axes[0, 1]
    replan_T = results_matrix['replan'].T
    im2 = ax2.imshow(replan_T, aspect='auto', origin='lower',
                     extent=[msg_lengths[0], msg_lengths[-1], freq_steps[0], freq_steps[-1]],
                     cmap='YlOrRd')
    plt.colorbar(im2, ax=ax2, label='Replan Count')
    ax2.plot(opt_msg, opt_freq, 'k*', markersize=15)
    ax2.plot(opt_msg, opt_freq, 'w*', markersize=9)
    ax2.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax2.set_ylabel("Frequency (steps)", fontsize=11)
    ax2.set_title("Replanning Frequency")
    ax2.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    # Plot 3: Average Separation Heatmap
    ax3 = axes[1, 0]
    sep_T = results_matrix['avg_separation'].T
    im3 = ax3.imshow(sep_T, aspect='auto', origin='lower',
                     extent=[msg_lengths[0], msg_lengths[-1], freq_steps[0], freq_steps[-1]],
                     cmap='RdYlGn')
    plt.colorbar(im3, ax=ax3, label='Separation (px)')
    ax3.plot(opt_msg, opt_freq, 'k*', markersize=15)
    ax3.plot(opt_msg, opt_freq, 'w*', markersize=9)
    ax3.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax3.set_ylabel("Frequency (steps)", fontsize=11)
    ax3.set_title("Average Agent Separation (higher = safer)")
    ax3.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    # Plot 4: Collision Rate Heatmap
    ax4 = axes[1, 1]
    coll_T = results_matrix['collision_rate'].T * 100  # Convert to percentage
    im4 = ax4.imshow(coll_T, aspect='auto', origin='lower',
                     extent=[msg_lengths[0], msg_lengths[-1], freq_steps[0], freq_steps[-1]],
                     cmap='RdYlGn_r', vmin=0, vmax=100)
    plt.colorbar(im4, ax=ax4, label='Collision Rate (%)')
    ax4.plot(opt_msg, opt_freq, 'k*', markersize=15)
    ax4.plot(opt_msg, opt_freq, 'w*', markersize=9)
    ax4.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax4.set_ylabel("Frequency (steps)", fontsize=11)
    ax4.set_title("Collision Rate (lower = safer)")
    ax4.grid(True, alpha=0.2, color='white', linewidth=0.5)
    
    # Summary at bottom
    opt_cost = cost_matrix[min_idx]
    opt_sep = results_matrix['avg_separation'][min_idx]
    opt_replan = results_matrix['replan'][min_idx]
    opt_coll_rate = results_matrix['collision_rate'][min_idx] * 100
    
    opt_text = (f"★ Optimal: freq={opt_freq} steps ({opt_freq_hz:.2f} Hz), msg={opt_msg} wp | "
                f"Cost={opt_cost:.1f} | Avg Safety={opt_sep:.1f}px | "
                f"Replans={opt_replan:.1f} | Collision Rate={opt_coll_rate:.0f}%")
    fig.text(0.5, 0.02, opt_text, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
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

def plot_msg_length_sweep(msg_lengths, results, save_path="results/msg_length_sweep.png",
                          config=None):
    """Plot cost metrics vs message length (waypoints shared per communication)."""
    ensure_results_dir()
    
    if config is None:
        config = {}
    
    # Build config text
    config_parts = []
    if 'n_agents' in config: config_parts.append(f"Agents: {config['n_agents']}")
    if 'num_trials' in config: config_parts.append(f"Trials: {config['num_trials']}")
    if 'num_seeds' in config: config_parts.append(f"Seeds: {config['num_seeds']}")
    if 'comm_range' in config: config_parts.append(f"Range: {config['comm_range']:.0f}px")
    if 'broadcast_interval_steps' in config: 
        config_parts.append(f"Freq: {config['broadcast_interval_steps']} steps")
    config_text = " | ".join(config_parts) if config_parts else ""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    fig.suptitle("Effect of Message Length on Performance", fontsize=14, fontweight='bold', y=0.98)
    if config_text:
        fig.text(0.5, 0.94, config_text, ha='center', fontsize=10, 
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    # Extract data
    costs = [r['cost_mean'] for r in results]
    cost_stds = [r.get('cost_std', 0) for r in results]
    replans = [r['replan_mean'] for r in results]
    replan_stds = [r.get('replan_std', 0) for r in results]
    avg_separations = [r['avg_separation_mean'] for r in results]
    avg_separation_stds = [r.get('avg_separation_std', 0) for r in results]
    collision_counts = [r['collision_count_mean'] for r in results]
    collision_count_stds = [r.get('collision_count_std', 0) for r in results]
    collision_rates = [r.get('collision_rate', 0) * 100 for r in results]
    
    opt_idx = np.argmin(costs)
    opt_length = msg_lengths[opt_idx]
    
    # Plot 1: Cost
    ax1 = axes[0, 0]
    ax1.plot(msg_lengths, costs, marker='o', linewidth=2, markersize=6, label='Mean')
    ax1.fill_between(msg_lengths, np.array(costs) - np.array(cost_stds),
                     np.array(costs) + np.array(cost_stds), alpha=0.2, label='±1 std')
    ax1.axvline(x=opt_length, color='red', linestyle='--', alpha=0.7, label=f'Optimal: {opt_length} wp')
    ax1.scatter([opt_length], [costs[opt_idx]], color='red', s=150, zorder=5, marker='*')
    if 0 in msg_lengths:
        zero_idx = list(msg_lengths).index(0)
        ax1.scatter([0], [costs[zero_idx]], color='blue', s=100, zorder=5, marker='D', label='Unlimited')
    ax1.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax1.set_ylabel("Total Cost", fontsize=11)
    ax1.set_title("Cost vs Message Length")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Replans
    ax2 = axes[0, 1]
    ax2.plot(msg_lengths, replans, marker='s', color='orange', linewidth=2, label='Mean')
    ax2.fill_between(msg_lengths, np.array(replans) - np.array(replan_stds),
                     np.array(replans) + np.array(replan_stds), alpha=0.2, color='orange', label='±1 std')
    ax2.axvline(x=opt_length, color='red', linestyle='--', alpha=0.7)
    ax2.scatter([opt_length], [replans[opt_idx]], color='red', s=150, zorder=5, marker='*')
    if 0 in msg_lengths:
        zero_idx = list(msg_lengths).index(0)
        ax2.scatter([0], [replans[zero_idx]], color='blue', s=100, zorder=5, marker='D')
    ax2.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax2.set_ylabel("Replan Count", fontsize=11)
    ax2.set_title("Replanning Frequency")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average Separation (match frequency_sweep style)
    ax3 = axes[1, 0]
    ax3.plot(msg_lengths, avg_separations, marker='^', color='green', linewidth=2, label='Mean')
    ax3.fill_between(msg_lengths, np.array(avg_separations) - np.array(avg_separation_stds),
                     np.array(avg_separations) + np.array(avg_separation_stds),
                     alpha=0.2, color='green', label='±1 std')
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Touching threshold')
    ax3.axhline(y=10, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Safe zone (>10px)')
    ax3.axvline(x=opt_length, color='red', linestyle='--', alpha=0.7)
    ax3.scatter([opt_length], [avg_separations[opt_idx]], color='red', s=150, zorder=5, marker='*')
    if 0 in msg_lengths:
        zero_idx = list(msg_lengths).index(0)
        ax3.scatter([0], [avg_separations[zero_idx]], color='blue', s=100, zorder=5, marker='D')
    # Removed: Light green "Safe clearance" shading (was unnecessary visual clutter)
    ax3.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax3.set_ylabel("Average Clearance (px)", fontsize=11)
    ax3.set_title("Average Agent Separation (higher = safer)", fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    y_max = max(avg_separations) + max(avg_separation_stds) + 5
    ax3.set_ylim(bottom=min(0, min(avg_separations) - max(avg_separation_stds) - 2), top=y_max)
    
    # Plot 4: Collision Analysis (match frequency_sweep style) - CLEANED UP
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    # Ensure collision counts are never negative
    collision_counts_clean = [max(0, c) for c in collision_counts]
    collision_count_stds_clean = [max(0, s) for s in collision_count_stds]
    
    # Bar chart for collision count - solid color, clean look
    width = max(msg_lengths) * 0.03 if max(msg_lengths) > 0 else 0.5
    bars = ax4.bar(msg_lengths, collision_counts_clean, width=width, 
                   color='#8B0000', alpha=1.0, label='Collision Count',
                   yerr=collision_count_stds_clean,
                   capsize=3, error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    # Line plot for collision rate - NO 'x' markers, cleaner line
    line = ax4_twin.plot(msg_lengths, collision_rates, 
                         color='red', linewidth=2.5, label='Collision Rate (%)', zorder=10)
    
    ax4.axvline(x=opt_length, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Optimal')
    if 0 in msg_lengths:
        zero_idx = list(msg_lengths).index(0)
        ax4.scatter([0], [collision_counts_clean[zero_idx]], color='blue', s=100, zorder=5, marker='D')
    
    ax4.set_xlabel("Message Length (waypoints)", fontsize=11)
    ax4.set_ylabel("Collision Count (avg per run)", fontsize=11, color='#8B0000')
    ax4_twin.set_ylabel("Collision Rate (%)", fontsize=11, color='red')
    ax4.set_title("Collision Analysis (lower = safer)", fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='#8B0000')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Set y-axis minimum to 0
    ax4.set_ylim(bottom=0)
    
    # Summary
    opt_text = (f"★ Optimal: {opt_length} wp | Cost={costs[opt_idx]:.1f} | "
                f"Avg Safety={avg_separations[opt_idx]:.1f}px | Replans={replans[opt_idx]:.1f} | "
                f"Collisions={collision_counts_clean[opt_idx]:.1f} ({collision_rates[opt_idx]:.0f}%)")
    fig.text(0.5, 0.02, opt_text, ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
