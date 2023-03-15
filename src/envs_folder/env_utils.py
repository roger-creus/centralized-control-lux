import numpy as np
from luxai_s2.utils.utils import my_turn_to_place_factory, is_day
from luxai_s2.map.position import Position
from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree
import math

RESOURCE_MAPPING = {0:"ice", 1:"ore", 2:"water", 3:"metal"}

def water_cost(factory, env):
    game_state = env.state
    owned_lichen_tiles = (game_state.board.lichen_strains == factory.state_dict()["strain_id"]).sum()
    return np.ceil(owned_lichen_tiles / env.env_cfg.LICHEN_WATERING_COST_FACTOR)

def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

def compute_move_cost(unit, game_state, direction):
    board = game_state.board
    target_pos = unit.pos + move_deltas[direction]
    if target_pos.x < 0 or target_pos.y < 0 or target_pos.y >= len(board.rubble) or target_pos.x >= len(board.rubble[0]):
        return -1

    factory_there = board.factory_occupancy_map[target_pos.x, target_pos.y]

    unit_agent_id = "player_0"  if unit.unit_id in game_state.units["player_0"] else "player_1"

    if factory_there != -1 and factory_there not in game_state.teams[unit_agent_id].factory_strains:
        # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
        return -1

    rubble_at_target = board.rubble[target_pos.x][target_pos.y]
    
    return math.floor(unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)

def action_queue_cost(unit, env_cfg):
    unit_type = str(unit.unit_type).split(".")[1]
    cost = env_cfg.ROBOTS[unit_type].ACTION_QUEUE_POWER_COST
    return cost

def manhattan_distance(binary_mask):
    distance_map = distance_transform_cdt(binary_mask, metric='taxicab')
    return distance_map

def manhattan_dist_to_nth_closest(arr, n):
    if n == 1:
        distance_map = distance_transform_cdt(1-arr, metric='taxicab')
        return distance_map
    else:
        true_coords = np.transpose(np.nonzero(arr))
        tree = KDTree(true_coords)
        dist, _ = tree.query(np.transpose(np.nonzero(~arr)), k=n, p=1)
        return np.reshape(dist[:, n-1], arr.shape)

def count_region_cells(array, start, min_dist=2, max_dist=np.inf, exponent=1):
    def dfs(array, loc):
        distance_from_start = abs(loc[0]-start[0]) + abs(loc[1]-start[1])
        if not (0<=loc[0]<array.shape[0] and 0<=loc[1]<array.shape[1]):   # check to see if we're still inside the map
            return 0
        if (not array[loc]) or visited[loc]:     # we're only interested in low rubble, not visited yet cells
            return 0
        if not (min_dist <= distance_from_start <= max_dist):      
            return 0
        visited[loc] = True
        count = 1.0 * exponent**distance_from_start
        count += dfs(array, (loc[0]-1, loc[1]))
        count += dfs(array, (loc[0]+1, loc[1]))
        count += dfs(array, (loc[0], loc[1]-1))
        count += dfs(array, (loc[0], loc[1]+1))
        return count
    visited = np.zeros_like(array, dtype=bool)
    return dfs(array, start)