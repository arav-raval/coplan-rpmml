"""Shared configuration constants."""

# ============================================================================
# CORE SIMULATION PARAMETERS
# ============================================================================
# --- Simulation Settings ---
NUM_SEEDS = 15 # Total number of independent samples (previously: 5 batches × 25 trials)
NUM_SEEDS_QUICK = 5  # Fewer seeds for quick testing
SIMULATION_FPS = 60

# --- Environment ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# --- Agent Physics ---
AGENT_RADIUS = 15
DEFAULT_NUM_AGENTS = 4
AGENT_COLORS = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "yellow"]

# --- Agent Movement ---
RRT_STEP_SIZE = AGENT_RADIUS * 0.5  # 22.5px
MAX_SPEED_STEPS_PER_SEC = 20  # 150px/s ÷ 22.5px/step
MAX_SPEED = MAX_SPEED_STEPS_PER_SEC * RRT_STEP_SIZE  # 150 px/s
GOAL_THRESHOLD = RRT_STEP_SIZE * 0.8

# --- Position Generation ---
POSITION_MARGIN = AGENT_RADIUS
MIN_POSITION_SEPARATION = RRT_STEP_SIZE * 4.0  # 30px - initial agent spacing
MIN_TRAVEL_DISTANCE = RRT_STEP_SIZE * 100.0  # 75px - minimum path length

# --- RRT Planning ---
RRT_ITERATIONS = 500
RRT_ITERATIONS_REPLAN = 500
RRT_GOAL_BIAS = 0.15
AVOIDANCE_RADIUS = AGENT_RADIUS * 2.2  # 33px - matches COLLISION_DISTANCE to prevent replan loop

# ============================================================================
# COLLISION PREDICTION & REPLANNING
# ============================================================================
PREDICTION_HORIZON_STEPS = 25  # 500px = 2.5 sec lookahead (matches old config)
PREDICTION_HORIZON = PREDICTION_HORIZON_STEPS * RRT_STEP_SIZE / MAX_SPEED
PREDICTION_DT = 0.1
COLLISION_DISTANCE = AGENT_RADIUS * 2.0  # 30px
REPLAN_COOLDOWN_STEPS = 5  # Steps to wait after replanning (prevents thrashing)
REPLAN_WAIT_STEPS = 3  # Steps to wait when replanning fails (collision avoidance fallback)
MAX_MSG_LENGTH_STEPS = PREDICTION_HORIZON_STEPS

# COST FUNCTION WEIGHTS AND PARAMETERS
# Refined cost structure for meaningful trade-offs between communication and safety

# Safety threshold for risk calculation
# Defined relative to agent geometry: minimum desired edge-to-edge clearance
# Notes:
#   - COLLISION_DISTANCE = AGENT_RADIUS * 2.0 = 30px (center-to-center touching)
#   - edge_to_edge_clearance = center_to_center_distance - COLLISION_DISTANCE
#   - When edge_to_edge = 0, agents are touching (collision)
#   - SAFETY_THRESHOLD is the minimum desired clearance for "safe" operation
#   - Using 2/3 of agent radius as safety buffer (reasonable margin)
SAFETY_THRESHOLD = AGENT_RADIUS   # ~10px with AGENT_RADIUS=15

WEIGHTS = {
    # Core performance
    'time': 1.0,              # Penalize simulation time (steps)
    'distance': 0.0,          # Drop distance (redundant with time)
    
    # Communication costs (actual usage)
    'frequency': 0.75,        # Base readiness: w_freq * broadcast_frequency_hz (small overhead)
    'message': 1.0,           # Actual data sent: w_msg * (messages_sent * msg_length)
    'replan': 0.75,           # Cost per replan event (optional, can be absorbed by message cost)
    
    # Safety shaping (pre-collision)
    'risk': 200.0,            # Penalize close calls: w_risk * (gap/threshold)^2
    
    # Failure penalties
    'collision': 3000.0,      # Large penalty per collision event
    'timeout': 1000.0,        # Penalty if agents didn't reach goal (medium severity)
}

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Optimal communication policy (from 2D sweep results)
# These can be updated based on experimental findings
OPTIMAL_COMM_POLICY = {
    'frequency_steps': 15,      # Steps per communication (12 Hz at 60 FPS)
    'msg_length': 9,          # Waypoints shared per message
    'description': 'Balanced policy: moderate frequency, medium message length'
}

EXPERIMENT_CONFIG = {
    # Core experiment settings
    'n_agents': DEFAULT_NUM_AGENTS,
    
    # Fixed communication parameters (MATCHES main.py defaults)
    # comm_range tied to prediction horizon - agents need to share paths within prediction distance
    'comm_range': PREDICTION_HORIZON_STEPS * RRT_STEP_SIZE,  # 247.5px = prediction distance
    'msg_length': 10,     # Matches main.py --msg-length default
    'default_frequency_hz': 12.0,  # ~5 steps
    
    # Frequency sweep settings
    'frequency_sweep_min_steps': 1,
    'frequency_sweep_max_steps': 60,
    'frequency_sweep_num_values': 20,
    'frequency_sweep_num_values_quick': 8,
    
    # Message length sweep settings
    'msg_length_sweep_min': 1,
    'msg_length_sweep_max': 25,
    'msg_length_sweep_num_values': 10,
    'msg_length_sweep_num_values_quick': 5,
    
    # 2D landscape settings (frequency × message length)
    'landscape_freq_min_steps': 1,
    'landscape_freq_max_steps': 60,
    'landscape_freq_num_values': 20,
    'landscape_freq_num_values_quick': 5,
    'landscape_msg_length_min': 1,
    'landscape_msg_length_max': 25,
    'landscape_msg_length_num_values': 10,
    'landscape_msg_length_num_values_quick': 5,
    
    # Comparison experiment settings
    'comparison_num_seeds': NUM_SEEDS,         # Number of seeds for comparison
    'comparison_num_seeds_quick': NUM_SEEDS_QUICK,
}


def get_experiment_config_value(key, quick=False):
    """
    Get an experiment config value, with quick mode override if available.
    
    Args:
        key: Configuration key (e.g., 'num_trials', 'frequency_sweep_num_values')
        quick: If True, returns '_quick' variant if available (e.g., 'num_trials_quick')
    
    Returns:
        Configuration value, or None if key not found
    """
    if quick and f"{key}_quick" in EXPERIMENT_CONFIG:
        return EXPERIMENT_CONFIG[f"{key}_quick"]
    return EXPERIMENT_CONFIG.get(key, None)
