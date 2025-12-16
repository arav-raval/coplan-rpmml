"""
Train MARL with parameter sharing using PPO.

Usage:
    python train_marl_ppo.py --timesteps 300000 --n_agents 4 --n_envs 8
"""

import numpy as np
import torch
import os
import sys
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from src.envs.marl_comm_env import MARLCommEnv


class ProgressCallback(BaseCallback):
    """Custom callback to print training progress."""
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.start_time = None
    
    def _on_training_start(self):
        self.start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"Training started at {self.start_time.strftime('%H:%M:%S')}")
        print(f"{'='*70}\n")
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            progress = self.num_timesteps / self.locals.get('total_timesteps', 1) * 100
            print(f"[{elapsed:6.1f}min] Steps: {self.num_timesteps:7,} ({progress:5.1f}%)")
        return True


def make_env(n_agents=4, seed=None, rank=0):
    """Create wrapped environment for parallel training."""
    def _init():
        env = MARLCommEnv(
            n_agents=n_agents,
            control_period_steps=30,
            episode_max_time=30.0,
            seed_range=(rank * 10000, (rank + 1) * 10000),  
        )
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo_marl(
    total_timesteps: int = 300_000,
    n_agents: int = 4,
    n_envs: int = 8,
    save_dir: str = "./models/marl_ppo",
    eval_freq: int = 5_000,
    checkpoint_freq: int = 10_000,
    device: str = "auto",
):
    """
    Train PPO with parameter sharing for decentralized MARL 
    
    Args:
        total_timesteps: Total training steps 
        n_agents: Number of agents per episode
        n_envs: Number of parallel environments 
        save_dir: Directory to save models and logs
        eval_freq: How often to evaluate (in timesteps)
        checkpoint_freq: How often to save checkpoints
        device: 'cuda', 'cpu', or 'auto'
    """
    
    print(f"\n{'='*70}")
    print(f"MARL COMMUNICATION OPTIMIZATION - PPO TRAINING")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Total timesteps:  {total_timesteps:,}")
    print(f"  Agents per env:   {n_agents}")
    print(f"  Parallel envs:    {n_envs}")
    print(f"  Device:           {device}")
    print(f"  Save directory:   {save_dir}")
    print(f"{'='*70}\n")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/logs", exist_ok=True)
    
    # Create parallel environments
    print(f"Creating {n_envs} parallel environments...")
    if n_envs > 1:
        envs = SubprocVecEnv([
            make_env(n_agents, seed=42, rank=i) for i in range(n_envs)
        ])
    else:
        envs = DummyVecEnv([make_env(n_agents, seed=42, rank=0)])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = Monitor(MARLCommEnv(n_agents=n_agents))
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,
        save_path=save_dir,
        name_prefix="ppo_marl_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=f"{save_dir}/logs",
        eval_freq=max(eval_freq // n_envs, 2000),  
        n_eval_episodes=5,   
        deterministic=True,
        render=False,
    )
    
    progress_callback = ProgressCallback(check_freq=1000)
    
    # Initialize PPO model
    print("\nInitializing PPO model...")
    
    # Detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,
        n_steps=1024,  
        batch_size=256,
        n_epochs=5,  
        gamma=0.98,  
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  
            activation_fn=torch.nn.ReLU,
        ),
        verbose=1,
        tensorboard_log=f"{save_dir}/tensorboard",
        device=device,
    )
    
    print(f"Model initialized on device: {model.device}")
    print(f"\nPolicy network architecture:")
    print(f"  Actor:  [obs(15)] -> [128] -> [128] -> [actions({len(MARLCommEnv().action_pairs)})]")
    print(f"  Critic: [obs(15)] -> [128] -> [128] -> [value(1)]")
    
    # Save hyperparameters
    hyperparams = {
        'total_timesteps': total_timesteps,
        'n_agents': n_agents,
        'n_envs': n_envs,
        'learning_rate': 3e-4,
        'n_steps': 512,
        'batch_size': 128,
        'gamma': 0.98,
        'ent_coef': 0.02,
        'device': str(model.device),
        'start_time': datetime.now().isoformat(),
    }
    
    import json
    with open(f"{save_dir}/hyperparams.json", 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    # Train
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Monitor progress: tensorboard --logdir {save_dir}/tensorboard")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback, progress_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    # Save final model
    final_path = f"{save_dir}/ppo_marl_final"
    model.save(final_path)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Duration:     {elapsed:.1f} minutes")
    print(f"Final model:  {final_path}.zip")
    print(f"Tensorboard:  {save_dir}/tensorboard")
    print(f"{'='*70}\n")
    
    # Cleanup
    envs.close()
    eval_env.close()
    
    return model, final_path


def main():
    parser = argparse.ArgumentParser(
        description="Train MARL policy for communication optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--timesteps", type=int, default=300_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n_agents", type=int, default=4,
        help="Number of agents per episode"
    )
    parser.add_argument(
        "--n_envs", type=int, default=8,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./models/marl_ppo",
        help="Directory to save models and logs"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=5_000,
        help="Evaluation frequency (timesteps)"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=10_000,
        help="Checkpoint save frequency (timesteps)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training"
    )
    
    args = parser.parse_args()
    
    # Train
    model, model_path = train_ppo_marl(
        total_timesteps=args.timesteps,
        n_agents=args.n_agents,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device,
    )
    
    print(f"Training complete! Model saved to: {model_path}.zip")
    print(f"\nNext steps:")
    print(f"  1. Evaluate: python evaluate_marl.py {model_path}.zip")
    print(f"  2. Visualize: tensorboard --logdir {args.save_dir}/tensorboard")


if __name__ == "__main__":
    main()
