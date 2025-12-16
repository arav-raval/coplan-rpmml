"""
Decentralized Multi-Agent RL Environment for Communication Optimization.

Each agent observes local state and chooses own (freq, msg_length).
Uses parameter sharing: one policy controls all agents with local observations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple, Optional
import random

from src.simulation import run_simulation
from src.config import (
    SIMULATION_FPS, SCREEN_WIDTH, SCREEN_HEIGHT, 
    MAX_MSG_LENGTH_STEPS, SAFETY_THRESHOLD, DEFAULT_NUM_AGENTS
)


class MARLCommEnv(gym.Env):
    """
    Decentralized MARL environment for communication optimization.
    
    Observation: Per-agent local view (15 features, normalized)
    Action: Discrete choice of (freq_steps, msg_length) pairs
    Reward: Negative cost delta (shared across agents - cooperative)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        n_agents: int = DEFAULT_NUM_AGENTS,
        control_period_steps: int = 30,  
        episode_max_time: float = 30.0,
        seed_range: Tuple[int, int] = (0, 10000),
    ):
        """
        Initialize MARL environment.
        
        Args:
            n_agents: Number of agents in simulation
            control_period_steps: Steps between RL decisions
            episode_max_time: Maximum episode duration (seconds)
            seed_range: Range of random seeds for scenarios
        """
        super().__init__()
        
        self.n_agents = n_agents
        self.control_period = control_period_steps
        self.episode_max_time = episode_max_time
        self.seed_range = seed_range
        
        # Action space: discrete grid of (freq_steps, msg_length) pairs
        self.action_pairs = self._create_action_space()
        self.action_space = spaces.Discrete(len(self.action_pairs))
        
        # Observation space: features per agent
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(15,), dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.episode_seed = None
        self.last_cost = 0.0
        self.cumulative_cost = 0.0
        self.last_metrics = None
        self.episode_start_time = 0.0
        
        # Agent-specific state tracking
        self.agent_frequencies = None
        self.agent_msg_lengths = None
    
    def _create_action_space(self) -> List[Tuple[int, int]]:
        """
        Define discrete action space from parameter sweep insights.
        
        Returns:
            List of (freq_steps, msg_length) tuples
        """
        action_pairs = [
            # High frequency options
            (1, 10), (3, 10), (5, 9), (5, 12), (5, 15),
            
            # Medium frequency
            (10, 7), (10, 9), (10, 12), (15, 7), (15, 9), (15, 12),
            (20, 8), (20, 10), (20, 12),
            
            # Low frequency
            (30, 5), (30, 10), (30, 15), (40, 8), (45, 10), (50, 12),
            
            # Very low frequency
            (60, 10), (60, 15), (60, 20),
            
            # Variable message lengths
            (15, 5), (15, 15), (15, 20),
            (25, 7), (25, 12), (35, 8),
        ]
        return action_pairs
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Start new episode with random scenario.
        
        Returns:
            observations: shape (obs_dim,) - single observation (averaged across agents for parameter sharing)
            info: dict with episode metadata
        """
        super().reset(seed=seed)
        
        if seed is None:
            seed = random.randint(*self.seed_range)
        self.episode_seed = seed
        
        self.current_step = 0
        self.last_cost = 0.0
        self.cumulative_cost = 0.0
        self.episode_start_time = 0.0
        
        self.agent_frequencies = [15] * self.n_agents  # 4 Hz
        self.agent_msg_lengths = [9] * self.n_agents   # 9 waypoints
        
        self.last_metrics = self._run_simulation_window()
        
        obs = self._get_local_observation(0)  
        
        info = {
            'episode_seed': self.episode_seed,
            'initial_cost': self.last_metrics.compute_cost() if self.last_metrics else 0.0,
        }
        
        return obs, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute actions for all agents.
        
        Args:
            actions: scalar - single action index
            
        Returns:
            observations: shape (obs_dim,)
            reward: float - shared reward for cooperation
            terminated: bool - episode ended (collision or success)
            truncated: bool - episode timeout
            info: dict with metrics
        """
        # Decode action to communication parameters (same for all agents)
        action_idx = int(actions) if np.isscalar(actions) else int(actions[0])
        freq_steps, msg_len = self.action_pairs[action_idx]
        
        # Apply same action to all agents (parameter sharing)
        for i in range(self.n_agents):
            self.agent_frequencies[i] = freq_steps
            self.agent_msg_lengths[i] = msg_len
        
        # Run simulation for control_period steps with these settings
        metrics = self._run_simulation_window()
        self.last_metrics = metrics
        
        # Compute reward (negative cost delta + shaping)
        current_cost = metrics.compute_cost()
        cost_delta = current_cost - self.last_cost
        
        # Base reward: negative cost delta
        reward = -cost_delta
        
        if metrics.collision_count > 0:
            reward -= 500.0  
        if metrics.both_reached_goal:
            reward += 100.0  
        
        self.last_cost = current_cost
        self.cumulative_cost += current_cost
        self.current_step += 1
        
        # Check termination conditions
        terminated = (
            metrics.collision_count > 0 or      # Any collision ends episode
            metrics.both_reached_goal           # Success ends episode
        )
        truncated = (metrics.total_time >= self.episode_max_time)
        
        # Get next observation 
        obs = self._get_local_observation(0)
        
        # Info dict for analysis
        info = {
            'cost': current_cost,
            'cumulative_cost': self.cumulative_cost,
            'collisions': metrics.collision_count,
            'min_separation': metrics.min_separation,
            'avg_separation': metrics.avg_separation,
            'both_reached_goal': metrics.both_reached_goal,
            'time': metrics.total_time,
            'replans': metrics.replan_count,
            'messages': metrics.messages_sent,
            'step': self.current_step,
        }
        
        return (
            obs,
            reward,
            terminated,
            truncated,
            info
        )
    
    def _run_simulation_window(self):
        """Run simulation for one control period window."""
        # Calculate simulation time for this window
        window_end_time = (self.current_step + 1) * self.control_period / SIMULATION_FPS
        
        metrics = run_simulation(
            comm_enabled=True,
            agent_broadcast_intervals=self.agent_frequencies,
            agent_msg_lengths=self.agent_msg_lengths,
            comm_range=250.0, 
            n_agents=self.n_agents,
            seed=self.episode_seed,
            max_time=window_end_time,
            verbose=False
        )
        
        return metrics
    
    def _get_local_observation(self, agent_idx: int) -> np.ndarray:
        """
        Get local observation for agent_idx.
        
        In a full implementation, this would use actual agent positions/states.
        For now, we approximate with global metrics (extensible later).
        
        Returns:
            Normalized observation vector (15 features)
        """
        if self.last_metrics is None:
            return np.zeros(15, dtype=np.float32)
        
        metrics = self.last_metrics
        
        def normalize(val, min_val, max_val):
            """Normalize value to [-1, 1] range."""
            if max_val <= min_val:
                return 0.0
            return 2.0 * (val - min_val) / (max_val - min_val) - 1.0
        
        def clamp(val, min_val, max_val):
            return max(min_val, min(max_val, val))
        
        obs = np.array([
            normalize(metrics.total_time, 0, self.episode_max_time),
            normalize(self.current_step, 0, 60),  
            
            normalize(metrics.min_separation, -30, 100),  
            normalize(metrics.avg_separation, 0, 150),
            1.0 if metrics.min_separation < SAFETY_THRESHOLD else -1.0,  
            1.0 if metrics.collision_count > 0 else -1.0,  
            
            normalize(self.agent_frequencies[agent_idx], 1, 60),
            normalize(self.agent_msg_lengths[agent_idx], 1, MAX_MSG_LENGTH_STEPS),
            
            normalize(metrics.replan_count, 0, 50),
            normalize(metrics.messages_sent, 0, 200),
            
            normalize(
                self.cumulative_cost / max(self.current_step, 1), 
                0, 1000
            ),
            
            normalize(self.n_agents - 1, 0, 10),  
            -1.0,  
            
            1.0 if metrics.both_reached_goal else -1.0,
            
            0.0,
        ], dtype=np.float32)
        
        obs = np.clip(obs, -1.0, 1.0)
        
        return obs
    
    def render(self):
        """Rendering not implemented for headless training."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


if __name__ == "__main__":
    print("Testing MARLCommEnv...")
    env = MARLCommEnv(n_agents=4)
    
    obs, info = env.reset()
    print(f"Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Episode seed: {info['episode_seed']}")
    
    # Random rollout
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}:")
        print(f"  Reward: {reward:.2f}")
        print(f"  Cost: {info['cost']:.1f}")
        print(f"  Min sep: {info['min_separation']:.1f}")
        print(f"  Done: {terminated or truncated}")
        
        if terminated or truncated:
            break
    
    print("\nMARLCommEnv test passed!")
