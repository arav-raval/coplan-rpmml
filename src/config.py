"""Shared configuration constants."""


# --- Simulation Settings ---
NUM_SEEDS = 15 
NUM_SEEDS_QUICK = 5 
SIMULATION_FPS = 60

# --- Environment ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# --- Agent Physics ---
AGENT_RADIUS = 15
DEFAULT_NUM_AGENTS = 4
AGENT_COLORS = ["blue", "red", "green", "orange", "purple", "cyan", "magenta", "yellow"]

# --- Agent Movement ---
RRT_STEP_SIZE = AGENT_RADIUS * 0.5  
MAX_SPEED_STEPS_PER_SEC = 20  
MAX_SPEED = MAX_SPEED_STEPS_PER_SEC * RRT_STEP_SIZE  # 150 px/s
GOAL_THRESHOLD = RRT_STEP_SIZE * 0.8

# --- Position Generation ---
POSITION_MARGIN = AGENT_RADIUS
MIN_POSITION_SEPARATION = RRT_STEP_SIZE * 4.0  
MIN_TRAVEL_DISTANCE = RRT_STEP_SIZE * 100.0  

# --- RRT Planning ---
RRT_ITERATIONS = 500
RRT_ITERATIONS_REPLAN = 500
RRT_GOAL_BIAS = 0.2
AVOIDANCE_RADIUS = AGENT_RADIUS * 2.2  

# --- Collision Prediction & Replanning ---
PREDICTION_HORIZON_STEPS = 50  
PREDICTION_HORIZON = PREDICTION_HORIZON_STEPS * RRT_STEP_SIZE / MAX_SPEED
PREDICTION_DT = 0.1
COLLISION_DISTANCE = AGENT_RADIUS * 2.0  
REPLAN_COOLDOWN_STEPS = 10  
REPLAN_WAIT_STEPS = 3  
MAX_MSG_LENGTH_STEPS = PREDICTION_HORIZON_STEPS

# --- Cost Function Weights and Parameters ---
SAFETY_THRESHOLD = AGENT_RADIUS  

WEIGHTS = {
    'time': 1.0,              
    'distance': 0.0,          
    'frequency': 1.5,        
    'message': 0.75,          
    'replan': 0.75,           
    'risk': 500.0,            
    'collision': 10000.0,     
    'timeout': 1000.0,        
}

# --- Experiment Configuration ---
OPTIMAL_COMM_POLICY = {
    'frequency_steps': 15,      
    'msg_length': 9,            
    'description': 'Optimal from 2D sweep: 15 steps (4Hz), 9 waypoints'
}

EXPERIMENT_CONFIG = {
    # --- Core experiment settings ---
    'n_agents': DEFAULT_NUM_AGENTS,
    
    # Fixed communication parameters 
    'comm_range': PREDICTION_HORIZON_STEPS * RRT_STEP_SIZE,  
    'msg_length': 10,     
    'default_frequency_hz': 6.0,  
    
    # --- Frequency sweep settings ---
    'frequency_sweep_min_steps': 1,
    'frequency_sweep_max_steps': 60,
    'frequency_sweep_num_values': 20,
    'frequency_sweep_num_values_quick': 8,
    
    # --- Message length sweep settings ---
    'msg_length_sweep_min': 1,
    'msg_length_sweep_max': PREDICTION_HORIZON_STEPS,
    'msg_length_sweep_num_values': 25,
    'msg_length_sweep_num_values_quick': 5,
    
    # --- 2D landscape settings ---
    'landscape_freq_min_steps': 1,
    'landscape_freq_max_steps': 60,
    'landscape_freq_num_values': 20,
    'landscape_freq_num_values_quick': 5,
    'landscape_msg_length_min': 1,
    'landscape_msg_length_max': PREDICTION_HORIZON_STEPS,
    'landscape_msg_length_num_values': 25,
    'landscape_msg_length_num_values_quick': 5,
    
    # --- Comparison experiment settings ---
    'comparison_num_seeds': NUM_SEEDS,         
    'comparison_num_seeds_quick': NUM_SEEDS_QUICK,
}


def get_experiment_config_value(key, quick=False):
    """
    Get an experiment config value, with quick mode override if available.
    
    Args:
        key: Configuration key 
        quick: If True, returns '_quick' variant if available 
    
    Returns:
        Configuration value, or None if key not found
    """
    if quick and f"{key}_quick" in EXPERIMENT_CONFIG:
        return EXPERIMENT_CONFIG[f"{key}_quick"]
    return EXPERIMENT_CONFIG.get(key, None)
