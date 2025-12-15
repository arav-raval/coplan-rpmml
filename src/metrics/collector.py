"""Metrics collection during simulation."""

from dataclasses import dataclass
import pymunk
from src.config import WEIGHTS

@dataclass
class SimulationMetrics:
    """Results from a single simulation run."""
    # Time metrics
    total_time: float = 0.0
    agent1_time: float = 0.0
    agent2_time: float = 0.0
    
    # Distance metrics
    agent1_distance: float = 0.0
    agent2_distance: float = 0.0
    
    # Safety metrics
    min_separation: float = float('inf')
    avg_separation: float = 0.0  # Average edge-to-edge clearance
    separation_samples: int = 0  # Number of samples for average
    collision_occurred: bool = False
    collision_count: int = 0  # Number of collision events (not frames)
    
    # Communication metrics
    replan_count: int = 0
    messages_sent: int = 0
    
    # Communication parameters (set by simulation)
    broadcast_frequency: float = 1.0
    comm_range: float = 250.0
    msg_length: int = 10
    
    # Outcome
    both_reached_goal: bool = False
    timed_out: bool = False

    def total_distance(self) -> float:
        return self.agent1_distance + self.agent2_distance

    def compute_cost(self, weights: dict = None) -> float:
        """
        Compute refined weighted cost for meaningful trade-offs.
        
        New structure emphasizes actual communication usage vs safety:
        - Time: simulation steps taken (direct performance metric)
        - Distance: dropped (redundant with time) or tiny weight
        - Frequency overhead: base cost for maintaining high comm frequency (Hz)
        - Message usage: actual data transmitted (messages_sent * msg_length)
        - Replan cost: per-replan event overhead
        - Risk shaping: squared penalty for close approaches < safety threshold
        - Collision: large penalty per collision event
        - Timeout: penalty if agents didn't complete
        """
        if weights is None:
            from src.config import WEIGHTS, SAFETY_THRESHOLD
            weights = WEIGHTS
            safety_thresh = SAFETY_THRESHOLD
        else:
            safety_thresh = weights.get('safety_threshold', 10.0)
        
        # 1. Time cost (steps)
        time_cost = weights['time'] * self.total_time
        
        # 2. Distance cost (tiny or zero - mostly redundant with time)
        distance_cost = weights.get('distance', 0.0) * self.total_distance()
        
        # 3. Frequency overhead (base readiness cost for high Hz)
        frequency_cost = weights['frequency'] * self.broadcast_frequency
        
        # 4. Message usage (actual data sent: messages_sent * msg_length)
        message_cost = weights['message'] * (self.messages_sent * self.msg_length)
        
        # 5. Replan cost (optional overhead per replan event)
        replan_cost = weights.get('replan', 0.0) * self.replan_count
        
        # 6. Risk cost (safety shaping: penalize close approaches)
        # gap = how much closer than safe threshold
        # risk = (gap/threshold)^2 gives quadratic penalty for close calls
        gap = max(0, safety_thresh - self.min_separation)
        risk_score = (gap / safety_thresh) ** 2 if safety_thresh > 0 else 0
        risk_cost = weights['risk'] * risk_score
        
        # 7. Collision cost (large penalty per collision event)
        collision_cost = weights['collision'] * self.collision_count
        
        # 8. Timeout cost (penalty if didn't reach goal)
        timeout_cost = weights.get('timeout', 0.0) * (0 if self.both_reached_goal else 1)
        
        total_cost = (
            time_cost + 
            distance_cost + 
            frequency_cost + 
            message_cost + 
            replan_cost + 
            risk_cost + 
            collision_cost + 
            timeout_cost
        )
        
        return total_cost
    
    def compute_cost_breakdown(self, weights: dict = None) -> dict:
        """Return individual cost components for detailed analysis."""
        if weights is None:
            from src.config import WEIGHTS, SAFETY_THRESHOLD
            weights = WEIGHTS
            safety_thresh = SAFETY_THRESHOLD
        else:
            safety_thresh = weights.get('safety_threshold', 10.0)
        
        gap = max(0, safety_thresh - self.min_separation)
        risk_score = (gap / safety_thresh) ** 2 if safety_thresh > 0 else 0
        
        return {
            'time': weights['time'] * self.total_time,
            'distance': weights.get('distance', 0.0) * self.total_distance(),
            'frequency': weights['frequency'] * self.broadcast_frequency,
            'message': weights['message'] * (self.messages_sent * self.msg_length),
            'replan': weights.get('replan', 0.0) * self.replan_count,
            'risk': weights['risk'] * risk_score,
            'collision': weights['collision'] * self.collision_count,
            'timeout': weights.get('timeout', 0.0) * (0 if self.both_reached_goal else 1),
            'total': self.compute_cost(weights)
        }


class MetricsCollector:
    """Collects metrics during simulation."""
    
    def __init__(self, broadcast_frequency=1.0, comm_range=250.0, msg_length=10):
        self.metrics = SimulationMetrics()
        # Store communication parameters for cost calculation
        self.metrics.broadcast_frequency = broadcast_frequency
        self.metrics.comm_range = comm_range
        self.metrics.msg_length = msg_length
        
        self._prev_pos1 = None
        self._prev_pos2 = None
        self._agent1_done = False
        self._agent2_done = False
        self._agent1_done_time = None
        self._agent2_done_time = None
    
    def update(self, agent1, agent2, current_time: float, collision_distance: float):
        """Call each frame to update metrics."""
        pos1 = agent1.body.position
        pos2 = agent2.body.position
        
        # Track distance traveled
        if self._prev_pos1 is not None:
            self.metrics.agent1_distance += pos1.get_distance(self._prev_pos1)
        if self._prev_pos2 is not None:
            self.metrics.agent2_distance += pos2.get_distance(self._prev_pos2)
        
        self._prev_pos1 = pymunk.Vec2d(pos1.x, pos1.y)
        self._prev_pos2 = pymunk.Vec2d(pos2.x, pos2.y)
        
        # Track minimum separation (convert to edge-to-edge clearance)
        center_to_center = pos1.get_distance(pos2)
        
        # For 2-agent case, collision_distance = 2 * AGENT_RADIUS
        agent_radius = collision_distance / 2
        edge_to_edge = center_to_center - collision_distance
        
        if edge_to_edge < self.metrics.min_separation:
            self.metrics.min_separation = edge_to_edge
        
        # Track average separation (running average)
        self.metrics.avg_separation = (
            (self.metrics.avg_separation * self.metrics.separation_samples + edge_to_edge) /
            (self.metrics.separation_samples + 1)
        )
        self.metrics.separation_samples += 1
        
        # Check for collision (uses center-to-center)
        if center_to_center < collision_distance:
            if not self.metrics.collision_occurred:
                # First collision ever
                self.metrics.collision_occurred = True
            # Only count NEW collision events, not every frame of overlap
            if not self._in_collision:
                self.metrics.collision_count += 1
                self._in_collision = True
        else:
            # Agents separated - reset collision tracking
            self._in_collision = False
        
        # Track when each agent finishes
        if not self._agent1_done and agent1.path_index >= len(agent1.path or []):
            self._agent1_done = True
            self._agent1_done_time = current_time
            self.metrics.agent1_time = current_time
            
        if not self._agent2_done and agent2.path_index >= len(agent2.path or []):
            self._agent2_done = True
            self._agent2_done_time = current_time
            self.metrics.agent2_time = current_time
        
        # Update total time
        self.metrics.total_time = current_time
        
        # Check if both done
        if self._agent1_done and self._agent2_done:
            self.metrics.both_reached_goal = True
    
    def record_replan(self):
        """Call when a replanning event occurs."""
        self.metrics.replan_count += 1
        self.metrics.messages_sent += 1  # Each replan involves sending path data
    
    def record_message(self):
        """Call when a message is transmitted (without replan)."""
        self.metrics.messages_sent += 1
    
    def finalize(self, timed_out: bool = False):
        """Call at end of simulation."""
        self.metrics.timed_out = timed_out
        return self.metrics