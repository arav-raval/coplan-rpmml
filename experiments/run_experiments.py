"""
Multi-Agent Path Planning Experiments (With / Without Communication)
This test runs parameter sweep experiments (across broadcast freqeuency and message length)
to evaluate how communication affects multi-agent path planning performance.
All configuration parameters are centralized in src/config.py.

Usage:
    python experiments/run_experiments.py                   # Run all experiments
    python experiments/run_experiments.py frequency         # Run specific experiment
    python experiments/run_experiments.py --quick           # Quick mode
    python experiments/run_experiments.py --visualize       # Show visualization
    python experiments/run_experiments.py --list            # List experiments

Archive: 
    This test previously included a sweep over agents, but we simplified our testing to a 
    fixed 4-agent configuration. 
"""

import sys
sys.path.insert(0, '.')

import argparse
import time
from datetime import datetime
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from itertools import combinations

from src.config import (
    NUM_SEEDS, NUM_SEEDS_QUICK, SIMULATION_FPS, EXPERIMENT_CONFIG, get_experiment_config_value,
    SCREEN_WIDTH, SCREEN_HEIGHT, MAX_SPEED, AGENT_RADIUS, AGENT_COLORS,
    PREDICTION_HORIZON, PREDICTION_DT, COLLISION_DISTANCE, REPLAN_COOLDOWN_STEPS,
    MAX_MSG_LENGTH_STEPS
)
from src.simulation import run_batch_multiple_seeds
from src.visualization.plots import (
    plot_frequency_sweep, 
    plot_range_sweep, 
    plot_cost_landscape_freq_msg
)
from src.world import create_space, create_boundaries
from src.agents import Agent


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def visualize_single_trial(broadcast_interval_steps, comm_range, msg_length, n_agents, seed=0):
    """
    Run a single trial with pygame visualization
    
    Args:
        broadcast_interval_steps: Steps between broadcasts
        comm_range: Communication range (px)
        msg_length: Message length (waypoints)
        n_agents: Number of agents
        seed: Random seed
    """
    import random
    from src.simulation import generate_random_positions, predict_collision_pair
    
    # Pygame setup
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Experiment Visualization - {n_agents} Agents - Seed {seed}")
    clock = pygame.time.Clock()
    
    # Pymunk setup
    space = create_space()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    draw_options.flags = pymunk.pygame_util.DrawOptions.DRAW_SHAPES
    
    static_obstacles = create_boundaries(space, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Generate positions
    positions = generate_random_positions(n_agents, SCREEN_WIDTH, SCREEN_HEIGHT, seed=seed)
    
    # Create agents
    agents = []
    for i, (start, goal) in enumerate(positions):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        agent = Agent(space, start, goal, static_obstacles,
                      (SCREEN_WIDTH, SCREEN_HEIGHT), color)
        agent.id = i
        agents.append(agent)
    
    # Initial planning
    print(f"  Initial planning...")
    for i, agent in enumerate(agents):
        agent.plan_path()
        if agent.path:
            print(f"    Agent {i}: {len(agent.path)} waypoints")
    
    # Per-agent cooldowns
    replan_cooldowns = {i: 0 for i in range(len(agents))}
    replan_count = 0
    frame_count = 0
    
    # Collision tracking
    collision_count = 0
    collision_pairs = set() 
    
    # Track last communication step for each agent pair
    last_comm_step = {} 
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        dt = 1.0 / SIMULATION_FPS
        space.step(dt)
        frame_count += 1
        
        # Update cooldowns
        for i in replan_cooldowns:
            if replan_cooldowns[i] > 0:
                replan_cooldowns[i] -= 1
        
        # Communication-based replanning
        for i, j in combinations(range(len(agents)), 2):
            agent_i = agents[i]
            agent_j = agents[j]
            
            # Check which agents are done
            agent_i_done = (agent_i.path is None or agent_i.path_index >= len(agent_i.path))
            agent_j_done = (agent_j.path is None or agent_j.path_index >= len(agent_j.path))
            
            # Skip if both agents are done
            if agent_i_done and agent_j_done:
                continue
            
            # Check if enough steps have passed since last communication for this pair
            pair = (min(i, j), max(i, j))
            last_step = last_comm_step.get(pair, -broadcast_interval_steps)
            
            # Bypass timing check if one agent is done 
            timing_ok = (frame_count - last_step) >= broadcast_interval_steps
            immediate_check_needed = agent_i_done or agent_j_done 
            
            # Communicate if timing ok or immediate check needed
            if timing_ok or immediate_check_needed:
                pos_i = agent_i.body.position
                pos_j = agent_j.body.position
                distance = pos_i.get_distance(pos_j)
                
                if distance < comm_range:
                    # Update last communication time
                    last_comm_step[pair] = frame_count
                    
                    # Count this broadcast as a message
                    collector.record_message()
                    
                    if predict_collision_pair(agent_i, agent_j, PREDICTION_HORIZON):
                        # Determine which agent should replan and if we should force it
                        force_replan = False
                        
                        # One agent is done
                        if agent_i_done and not agent_j_done:
                            replanner_idx = j
                            replanner = agent_j
                            force_replan = True  
                        elif agent_j_done and not agent_i_done:
                            replanner_idx = i
                            replanner = agent_i
                            force_replan = True  
                        
                        # Both active - check cooldowns
                        else:
                            agent_i_on_cooldown = replan_cooldowns[i] > 0
                            agent_j_on_cooldown = replan_cooldowns[j] > 0
                            
                            # If one is on cooldown, the other must replan
                            if agent_i_on_cooldown and not agent_j_on_cooldown:
                                replanner_idx = j
                                replanner = agent_j
                                force_replan = True  
                            elif agent_j_on_cooldown and not agent_i_on_cooldown:
                                replanner_idx = i
                                replanner = agent_i
                                force_replan = True 
                            else:
                                # Both on cooldown or both available
                                replanner_idx = j
                                replanner = agent_j
                                force_replan = False  
                        
                        # Check cooldown (bypass if forced)
                        if force_replan or replan_cooldowns[replanner_idx] <= 0:
                            # Gather obstacles 
                            replanner_pos = replanner.body.position
                            effective_msg_length = min(msg_length, MAX_MSG_LENGTH_STEPS) if msg_length > 0 else MAX_MSG_LENGTH_STEPS
                            obstacles = []
                            for k, other in enumerate(agents):
                                if k != replanner_idx:
                                    other_pos = other.body.position
                                    other_distance = replanner_pos.get_distance(other_pos)
                                    if other_distance < comm_range:
                                        # If agent is done, include its final position
                                        other_done = (other.path is None or other.path_index >= len(other.path))
                                        if other_done:
                                            obstacles.append([other_pos])
                                        elif other.path:
                                            remaining = other.get_remaining_path()
                                            if effective_msg_length > 0 and len(remaining) > effective_msg_length:
                                                remaining = remaining[:effective_msg_length]
                                            if remaining:
                                                obstacles.append(remaining)
                            
                            if obstacles:
                                success = replanner.replan(dynamic_obstacles=obstacles)
                                if success:
                                    replan_count += 1
                                    replan_cooldowns[replanner_idx] = REPLAN_COOLDOWN_STEPS
        
        # Update agents
        for agent in agents:
            agent.update()
        
        # Ground-truth collision detection 
        for i, j in combinations(range(len(agents)), 2):
            agent_i = agents[i]
            agent_j = agents[j]
            
            # Skip if both agents are done
            if (agent_i.path is None or agent_i.path_index >= len(agent_i.path) or
                agent_j.path is None or agent_j.path_index >= len(agent_j.path)):
                continue
            
            pos_i = agent_i.body.position
            pos_j = agent_j.body.position
            distance = pos_i.get_distance(pos_j)
            
            # Check if agents are colliding
            if distance < 2 * AGENT_RADIUS:
                pair = tuple(sorted([i, j]))
                if pair not in collision_pairs:
                    collision_pairs.add(pair)
                    collision_count += 1
                    print(f"  💥 COLLISION! Agents {i}-{j} at distance {distance:.1f}px")
        
        # Render
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        
        # Draw paths
        for agent in agents:
            if agent.path:
                color = agent.shape.color
                if len(color) == 4:  
                    r, g, b, a = color
                else:  
                    r, g, b = color
                
                light_color = (min(255, r + 150), min(255, g + 150), min(255, b + 150))
                bright_color = (r, g, b)  
                
                for k in range(len(agent.path) - 1):
                    start = agent.path[k]
                    end = agent.path[k + 1]
                    pygame.draw.line(screen, light_color, 
                                   (int(start.x), int(start.y)), 
                                   (int(end.x), int(end.y)), 1)
                
                if agent.path_index < len(agent.path):
                    for k in range(agent.path_index, len(agent.path) - 1):
                        start = agent.path[k]
                        end = agent.path[k + 1]
                        pygame.draw.line(screen, bright_color, 
                                       (int(start.x), int(start.y)), 
                                       (int(end.x), int(end.y)), 2)
        
        # Draw goals
        for agent in agents:
            pygame.draw.circle(screen, (0, 200, 0), 
                             (int(agent.goal.x), int(agent.goal.y)), 
                             AGENT_RADIUS // 2, 2)
        
        # Draw info
        font = pygame.font.Font(None, 24)
        collision_status = f"💥 {collision_count}" if collision_count > 0 else "✓ 0"
        info_lines = [
            f"Seed: {seed} | Freq: {broadcast_interval_steps} steps",
            f"Frame: {frame_count} | Replans: {replan_count} | Collisions: {collision_status}",
            f"Agents: {n_agents} | Range: {comm_range:.0f}px | Msg: {msg_length}",
            f"Press ESC to continue to next experiment"
        ]
        
        for i, line in enumerate(info_lines):
            text_surface = font.render(line, True, (0, 0, 0))
            screen.blit(text_surface, (10, 10 + i * 25))
        
        # Draw agent info
        small_font = pygame.font.Font(None, 18)
        for i, agent in enumerate(agents):
            status = "✓" if (agent.path is None or agent.path_index >= len(agent.path)) else "→"
            remaining = len(agent.path) - agent.path_index if agent.path and agent.path_index < len(agent.path) else 0
            cooldown = replan_cooldowns[i]
            status_text = f"A{i}: {status} ({remaining} wpts, cd={cooldown})"
            
            color = agent.shape.color
            text_color = (color[0], color[1], color[2]) if len(color) >= 3 else (0, 0, 0)
            
            text_surface = small_font.render(status_text, True, text_color)
            screen.blit(text_surface, (SCREEN_WIDTH - 200, 10 + i * 20))
        
        pygame.display.flip()
        clock.tick(SIMULATION_FPS)
        
        # Check if all done
        all_done = all(
            agent.path is None or agent.path_index >= len(agent.path)
            for agent in agents
        )
        if all_done:
            collision_status = f"💥 {collision_count}" if collision_count > 0 else "✓ 0"
            print(f"All agents reached goals! Replans: {replan_count}, Collisions: {collision_status}")
            pygame.time.wait(1000) 
            break
    
    pygame.quit()
    print(f"  Visualization closed. Continuing with experiments...\n")


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_frequency_sweep(quick=False, visualize=False):
    """
    Sweep over broadcast frequency (defined as steps per communication)
    
    Fixed parameters: comm_range, msg_length, n_agents 
    Sweep parameter: broadcast_interval_steps
    
    Args:
        quick: Use fewer trials/points
        visualize: Show pygame visualization for first trial of each parameter
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT: Broadcast Frequency Sweep")
    print("=" * 70)
    
    # Read parameters from config
    min_steps = EXPERIMENT_CONFIG['frequency_sweep_min_steps']
    max_steps = EXPERIMENT_CONFIG['frequency_sweep_max_steps']
    num_values = get_experiment_config_value('frequency_sweep_num_values', quick)
    comm_range = EXPERIMENT_CONFIG['comm_range']
    msg_length = EXPERIMENT_CONFIG['msg_length']
    n_agents = EXPERIMENT_CONFIG['n_agents']
    num_seeds = NUM_SEEDS_QUICK if quick else NUM_SEEDS
    
    # Generate sweep values
    step_intervals = np.linspace(min_steps, max_steps, num_values).astype(int)
    frequencies_hz = SIMULATION_FPS / step_intervals  
    
    print(f"\nConfiguration:")
    print(f"  Agents: {n_agents}")
    print(f"  Comm range: {comm_range:.1f}px")
    print(f"  Message length: {msg_length} waypoints")
    print(f"  Seeds per point: {num_seeds}")
    print(f"\nFrequency range:")
    print(f"  Steps: {step_intervals[0]} to {step_intervals[-1]} steps")
    print(f"  Hz: {frequencies_hz[0]:.2f} to {frequencies_hz[-1]:.2f} Hz")
    print(f"  Total points: {len(step_intervals)}\n")
    
    # Run sweep
    results = []
    for i, steps in enumerate(step_intervals):
        freq_hz = frequencies_hz[i]
        print(f"[{i+1:2d}/{len(step_intervals)}] {steps:3d} steps ({freq_hz:5.2f} Hz) ... ", 
              end="", flush=True)
        
        try:
            # Visualize if requested
            if visualize:
                viz_seed = 0  
                print(f"\nLaunching visualization for {steps} steps (seed={viz_seed})...")
                print(f"Press ESC when ready to continue to next parameter...")
                visualize_single_trial(
                    broadcast_interval_steps=steps,
            comm_range=comm_range,
            msg_length=msg_length,
            n_agents=n_agents,
                    seed=viz_seed
                )
                print(f"Visualization complete. Skipping headless simulation for this parameter.")
                print(f"Use without --visualize to collect metrics\n")
                continue
            
            result = run_batch_multiple_seeds(
                comm_enabled=True,
                broadcast_interval_steps=steps,
                comm_range=comm_range,
                msg_length=msg_length,
                n_agents=n_agents,
                num_seeds=num_seeds,
                verbose=True  
            )
            
            print(f"cost={result['cost_mean']:6.1f}±{result['cost_std']:4.1f}, "
                  f"safety={result['min_separation_mean']:5.1f}±{result['min_separation_std']:4.1f}px")
            
            result['interval_steps'] = steps
            result['frequency_hz'] = freq_hz
            results.append(result)
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(None)
    
    results = [r for r in results if r is not None]
    
    if not results:
        if visualize:
            print("\n✓ Visualization complete for all parameters!")
            print("  Note: No metrics collected in visualization mode.")
            print("  Run without --visualize to collect performance data.")
            return {'visualization_only': True}
        print("\n❌ All trials failed!")
        return None
    
    # Plot results
    if results:
        valid_steps = [r['interval_steps'] for r in results]
        config = {
            'n_agents': n_agents,
            'num_seeds': num_seeds,
            'comm_range': comm_range,
            'msg_length': msg_length
        }
        plot_frequency_sweep(valid_steps, results, config=config)
    
    # Find optimal
    if results:
        costs = [r['cost_mean'] for r in results]
        opt_idx = np.argmin(costs)
        
        print(f"\n✓ Optimal: {valid_steps[opt_idx]} steps ({results[opt_idx]['frequency_hz']:.2f} Hz), "
              f"cost={costs[opt_idx]:.1f}")
    
    return {
            'optimal_interval_steps': valid_steps[opt_idx],
            'optimal_frequency_hz': results[opt_idx]['frequency_hz'],
            'optimal_cost': costs[opt_idx],
            'results': results
        }
    
    return None

# =============================================================================
# MESSAGE LENGTH SWEEP
# =============================================================================

def run_msg_length_sweep(quick=False, visualize=False):
    """
    Sweep over message length (number of waypoints shared per communication)
    
    Fixed parameters: comm_range, broadcast_interval_steps, n_agents 
    Sweep parameter: msg_length
    """
    from experiments.sweep_msg_length import run_msg_length_sweep as msg_length_sweep
    return msg_length_sweep(quick=quick, visualize=visualize)


# =============================================================================
# 2D SWEEP
# =============================================================================

def run_2d_sweep(quick=False, visualize=False):
    """
    Sweep over frequency and message length simultaneously
    
    Fixed parameters: comm_range, n_agents 
    Sweep parameters: broadcast_interval_steps, msg_length
    """
    from experiments.sweep_2d import main as sweep_2d_main
    import sys
    
    original_argv = sys.argv
    if quick:
        sys.argv = ['sweep_2d.py', '--quick']
    else:
        sys.argv = ['sweep_2d.py']
    
    try:
        result = sweep_2d_main()
        return {'status': 'completed'}
    finally:
        sys.argv = original_argv


# =============================================================================
# COMPARISON EXPERIMENT
# =============================================================================
def run_comparison_sweep(quick=False, visualize=False):
    """Wrapper that calls the standalone sweep_comparison.py module."""
    from experiments.sweep_comparison import run_comparison_sweep as comparison_sweep
    return comparison_sweep(quick=quick, visualize=visualize)


# =============================================================================
# EXPERIMENT REGISTRY
# =============================================================================
EXPERIMENTS = {
    'frequency': {
        'name': 'Frequency Sweep',
        'description': 'Sweep broadcast frequency (steps per communication)',
        'function': run_frequency_sweep
    },
    'msg_length': {
        'name': 'Message Length Sweep',
        'description': 'Sweep message length (waypoints shared per communication)',
        'function': run_msg_length_sweep
    },
    '2d': {
        'name': '2D Cost Landscape',
        'description': 'Sweep frequency × message length to find optimal combination',
        'function': run_2d_sweep
    },
    'comparison': {
        'name': 'Communication Comparison',
        'description': 'Compare optimal policy vs no communication baseline',
        'function': run_comparison_sweep
    },
}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Path Planning Experiments with Communication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Run all experiments
  %(prog)s frequency            # Run frequency sweep only
  %(prog)s --visualize frequency # Show visualization for first trial
  %(prog)s --quick              # Quick mode (fewer trials)
  %(prog)s --list               # List available experiments

All configuration parameters are defined in src/config.py.
        """
    )
    
    parser.add_argument('experiments', nargs='*', 
                       help='Experiments to run (default: all)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available experiments')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Quick mode (fewer trials/points)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Show pygame visualization for first trial of each parameter')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("\nAvailable Experiments:")
        print("-" * 70)
        for key, exp in EXPERIMENTS.items():
            print(f"  {key:<12} {exp['name']}")
            print(f"  {' '*12} {exp['description']}")
        print()
        return
    
    # Determine which experiments to run
    if args.experiments:
        to_run = []
        for name in args.experiments:
            if name in EXPERIMENTS:
                to_run.append(name)
            else:
                print(f"❌ Unknown experiment: {name}")
                print(f"   Available: {', '.join(EXPERIMENTS.keys())}")
                print(f"   Use --list to see descriptions")
                return
    else:
        to_run = list(EXPERIMENTS.keys())
    
    # Header
    print("\n" + "=" * 70)
    print("MULTI-AGENT PATH PLANNING EXPERIMENTS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    print(f"Visualize: {'YES' if args.visualize else 'NO'}")
    print(f"Experiments: {', '.join(to_run)}")
    print("=" * 70)
    
    # Run experiments
    results = {}
    total_start = time.time()
    
    for exp_name in to_run:
        exp = EXPERIMENTS[exp_name]
        start_time = time.time()
        
        try:
            if 'visualize' in exp['function'].__code__.co_varnames:
                result = exp['function'](quick=args.quick, visualize=args.visualize)
            else:
                result = exp['function'](quick=args.quick)
            
            elapsed = time.time() - start_time
            
            if result is not None:
                results[exp_name] = {'status': 'success', 'result': result, 'time': elapsed}
                print(f"\n{exp['name']} completed in {elapsed:.1f}s")
            else:
                results[exp_name] = {'status': 'failed', 'time': elapsed}
                print(f"\n{exp['name']} failed after {elapsed:.1f}s")
                
        except Exception as e:
            elapsed = time.time() - start_time
            results[exp_name] = {'status': 'error', 'error': str(e), 'time': elapsed}
            print(f"\n{exp['name']} error after {elapsed:.1f}s: {e}")
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"Completed: {success_count}/{len(results)} experiments")
    print(f"Total time: {total_elapsed:.1f}s\n")
    
    for exp_name, res in results.items():
        exp = EXPERIMENTS[exp_name]
        status_icon = "YES" if res['status'] == 'success' else "NO"
        print(f"{status_icon} {exp['name']:<25} ({res['time']:.1f}s)")
        
        if res['status'] == 'success' and 'result' in res:
            r = res['result']
            if 'optimal_frequency_hz' in r:
                print(f"   Optimal: {r['optimal_interval_steps']} steps "
                      f"({r['optimal_frequency_hz']:.2f} Hz)")
            if 'optimal_msg_length' in r:
                print(f"   Optimal: {r['optimal_msg_length']} waypoints")
            if 'cost_reduction_pct' in r:
                print(f"   Communication reduces cost by {r['cost_reduction_pct']:.1f}%")
        elif res['status'] == 'error':
            print(f"   Error: {res['error']}")
    
    print(f"\nResults saved to: results/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
