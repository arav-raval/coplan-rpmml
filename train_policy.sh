#!/bin/bash
#SBATCH --job-name=marl_comm
#SBATCH --output=logs/marl_%j.out
#SBATCH --error=logs/marl_%j.err
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ar3015@princeton.edu 

# Adroit Training Script for MARL Communication Optimization
echo "======================================================================"
echo "JOB STARTED: $(date)"
echo "======================================================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $SLURM_NODELIST"
echo "Cores:        $SLURM_CPUS_PER_TASK"
echo "GPU:          $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

# Load modules
module purge
module load anaconda3/2023.9
module load cudatoolkit/12.1

# Activate or create conda environment
ENV_NAME="marl_env"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Activating existing environment: ${ENV_NAME}"
    conda activate ${ENV_NAME}
else
    echo "Creating new environment: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
    conda activate ${ENV_NAME}
    
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install gymnasium stable-baselines3[extra] tensorboard
    pip install numpy matplotlib pymunk pygame
fi

# Verify CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Create logs directory
mkdir -p logs

# Set environment variables for better performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Training configuration 
TIMESTEPS=12000
N_AGENTS=4
N_ENVS=4
SAVE_DIR="./models/marl_ppo_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "======================================================================"
echo "TRAINING CONFIGURATION"
echo "======================================================================"
echo "Timesteps:    $TIMESTEPS"
echo "Agents:       $N_AGENTS"
echo "Parallel envs: $N_ENVS"
echo "Save dir:     $SAVE_DIR"
echo "======================================================================"
echo ""

# Run training
echo "Starting training..."
python train_marl_ppo.py \
    --timesteps $TIMESTEPS \
    --n_agents $N_AGENTS \
    --n_envs $N_ENVS \
    --save_dir $SAVE_DIR \
    --eval_freq 5000 \
    --checkpoint_freq 10000 \
    --device cuda

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "======================================================================"
    
    # Run evaluation
    echo ""
    echo "Running evaluation..."
    python evaluate_marl.py \
        ${SAVE_DIR}/ppo_marl_final.zip \
        --n_episodes 50 \
        --n_agents $N_AGENTS \
        --compare_baseline \
        --baseline_seeds 50 \
        --analyze_actions \
        --plot
    
    EVAL_EXIT_CODE=$?
    
    if [ $EVAL_EXIT_CODE -eq 0 ]; then
        echo "Evaluation completed successfully"
    else
        echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
    fi
    
    # Print summary
    echo ""
    echo "======================================================================"
    echo "RESULTS SUMMARY"
    echo "======================================================================"
    echo "Model location:   ${SAVE_DIR}/ppo_marl_final.zip"
    echo "Tensorboard logs: ${SAVE_DIR}/tensorboard"
    echo "Evaluation plots: results/marl_evaluation.png"
    echo ""
    echo "To view tensorboard:"
    echo "  tensorboard --logdir ${SAVE_DIR}/tensorboard"
    echo "======================================================================"
    
else
    echo ""
    echo "======================================================================"
    echo "TRAINING FAILED"
    echo "======================================================================"
    echo "Exit code: $TRAIN_EXIT_CODE"
    echo "Check logs/marl_${SLURM_JOB_ID}.err for errors"
    echo "======================================================================"
fi

echo ""
echo "======================================================================"
echo "JOB ENDED: $(date)"
echo "======================================================================"

exit $TRAIN_EXIT_CODE
