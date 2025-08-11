import numpy as np


# Discrete action mapping expected by environment.uav_env.UAVEnvironment
# 0=East(+X), 1=South(-Y), 2=West(-X), 3=North(+Y), 4=Hover
EAST: int = 0
SOUTH: int = 1
WEST: int = 2
NORTH: int = 3
HOVER: int = 4


def choose_greedy_action_to_goal(
    uav_position: np.ndarray,
    end_position: np.ndarray,
    distance_tolerance: float = 1.0,
) -> int:
    """
    Choose a discrete action that greedily moves the UAV toward the end position.

    This function is a deterministic baseline controller and does not depend on
    the environment implementation. It returns an integer action code compatible
    with UAVEnvironment's discrete action mapping.

    Args:
        uav_position: Current UAV position as array-like [x, y, z]
        end_position: Target end position as array-like [x, y, z]
        distance_tolerance: If within this distance to the goal, hover

    Returns:
        int: Discrete action id (0..4)
    """
    uav_pos = np.asarray(uav_position, dtype=float)
    end_pos = np.asarray(end_position, dtype=float)

    delta_xy = end_pos[:2] - uav_pos[:2]
    distance = float(np.linalg.norm(delta_xy))
    if distance <= max(0.0, distance_tolerance):
        return HOVER

    dx, dy = delta_xy[0], delta_xy[1]

    # Prefer the axis with larger absolute distance to approximate a straight line
    if abs(dx) >= abs(dy):
        if dx > 0:
            return EAST
        else:
            return WEST
    else:
        if dy > 0:
            return NORTH
        else:
            return SOUTH


__all__ = [
    "EAST",
    "SOUTH",
    "WEST",
    "NORTH",
    "HOVER",
    "choose_greedy_action_to_goal",
]


