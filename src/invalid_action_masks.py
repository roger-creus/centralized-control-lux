import gym
import numpy as np
import torch
import math
import luxai_s2.unit as luxai_unit

from luxai_s2.map.position import Position
from utils import VideoWrapper
from IPython import embed

# this is only used in the scripts that end with _post.py

# this masks the TRANSFER parameters although they werent ilegal a priori if at postprocessing we see that the output action type was MOVE
# and viceversa
def post_masks(robot_masks, action_types):
    # we are going to correct the masks only where there are robots
    # where there are no robots all actions should already be set to ilegal
    
    # if there are no robots, nothing to correct, all should be ilegal already
    max_ = torch.max(robot_masks[:,0]).item()
    if max_ == 0:
        return robot_masks

    robot_idcs = torch.where(robot_masks[:,0] == max_)[0].cpu().numpy()
    for idx in robot_idcs:
        action = action_types[idx].item()
        # if sampled action is NOOP, dont learn from ANY parameter
        if action == 0:
            robot_masks[idx, 7:] = 0
        # if sampled action is MOVE, dont learn from TRANSFER parameters
        elif action == 1:
            robot_masks[idx, 11:] = 0
        # if sampled action is TRANSFER, dont learn from MOVE parameters
        elif action == 2:
            robot_masks[idx, 7:11] = 0
        else:
            continue
    return robot_masks

def post_categoricals(multi_categoricals, action_types, robot_masks):
    """
    Mask the logits and probs of actions that were not ilegal but are not going to be executed
    Can be done only if logits and logprobs dont require grad() -> During ROLLOUTs
    Use the robot_masks only to identify where there actually are robots
    """
    # if there are no robots, nothing to correct, all should be ilegal already
    max_ = torch.max(robot_masks[:,0]).item()
    if max_ == 0:
        return multi_categoricals

    robot_idcs = torch.where(robot_masks[:,0] == max_)[0].cpu().numpy()
    for idx in robot_idcs:
        action = action_types[idx].item()
        
        # if sampled action is NOOP, masks all parameters
        if action == 0:
            for i in range(len(multi_categoricals[1:])):
                multi_categoricals[1:][i].logits[idx][:] = 0
                multi_categoricals[1:][i].probs[idx][:] = 0.

        # if sampled action is MOVE, mask all transfer params
        elif action == 1:
            for i in range(len(multi_categoricals[2:])):
                multi_categoricals[2:][i].logits[idx][:] = 0
                multi_categoricals[2:][i].probs[idx][:] = 0.
        
        # if sampled action is TRANSFER, mask all move params
        elif action == 2:
            multi_categoricals[1].logits[idx] = 0
            multi_categoricals[1].probs[idx] = 0.
        
        else:
            continue

    return multi_categoricals

#### FACTORIES ####
def get_factory_invalid_action_masks(envs, player):
    if type(envs) == gym.vector.sync_vector_env.SyncVectorEnv:
        action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["factories"].nvec[1:].sum() + 1))
        envs = envs.envs

    elif player == "player_1" or type(envs) == VideoWrapper:
        action_mask = np.zeros((1, 48, 48, envs.action_space["factories"].nvec[1:].sum() + 1))
        envs = [envs]
    else:
        print("error with type", type(envs))
    
    for i in range(len(envs)):
        env = envs[i]
        game_state = env.env_.state

        for unit_id in game_state.factories[player]:
            factory = game_state.factories[player][unit_id]
            x, y = factory.pos.x, factory.pos.y

            # dim 0: factory_in_cell
            # dim 1: NOOP
            # dim 2: build light
            # dim 3: build heavy
            # dim 4: water lichen

            # start by assuming all actions are legal where there is a unit
            action_mask[i, x, y, :] = 1

            metal = factory.cargo.metal
            water = factory.cargo.water
            current_power = factory.power

            # cant build a light robot
            if metal < 10 or current_power < 50:
                action_mask[i, x, y, 2] = 0     

            # cant build a heavy robot
            if metal < 100 or current_power < 500:
                action_mask[i, x, y, 3] = 0     

            # cant water lichen if no water
            if water -1 <= water_cost(factory, env.env_):
                action_mask[i, x, y, 4] = 0     

    return action_mask


#### ROBOTS ####
def get_robot_invalid_action_masks(envs, player):
    # the returned mask will be of shape (num_envs, map_size, map_size, total_unit_actions + 1)
    # the +1 in the end indicates if if there is a unit in tile x,y. takes values in [0,1]
    if type(envs) == gym.vector.sync_vector_env.SyncVectorEnv:
        action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["robots"].nvec[1:].sum() + 1))
        envs = envs.envs

    elif player == "player_1" or type(envs) == VideoWrapper:
        action_mask = np.zeros((1, 48, 48, envs.action_space["robots"].nvec[1:].sum() + 1))
        envs = [envs]
    else:
        print("error with type", type(envs))

    # create the mask for each environment
    for i in range(len(envs)):
        env = envs[i]
        game_state = env.env_.state

        for unit_id in game_state.units[player]:
            unit = game_state.units[player][unit_id]
            x, y = unit.pos.x, unit.pos.y

            # start by assuming all actions are legal where there is a unit
            action_mask[i, x, y, :] = 1

            current_power = unit.power
            update_queue_cost = action_queue_cost(unit, env.env_.env_cfg)

            # if not enough power to update action queue then we cannot do anything but NOOP
            if current_power <= update_queue_cost:
                action_mask[i, x, y, 2:] = 0

            ###### MOVE ACTION TYPE ######
            move_cost_up = move_cost(unit, game_state, 1)
            can_move_up = True
            # no map or enemy factory on top -> cannot move or cannot transfer up
            if move_cost_up == 99999:
                action_mask[i, x, y, 7] = 0
                action_mask[i, x, y, 12] = 0
                can_move_up = False
            # if not enough power to move due to rubble cost
            if current_power <= move_cost_up + update_queue_cost:
                action_mask[i, x, y, 7] = 0
                can_move_up = False

            move_cost_right = move_cost(unit, game_state, 2)
            can_move_right = True
            # no map or enemy factory to the right -> cannot move or cannot transfer right
            if move_cost_right == 99999:
                action_mask[i, x, y, 8] = 0
                action_mask[i, x, y, 13] = 0
                can_move_right = False
            # if not enough power to move due to rubble cost
            if current_power <= move_cost_right + update_queue_cost:
                action_mask[i, x, y, 8] = 0
                can_move_right = False

            can_move_down = True
            move_cost_down = move_cost(unit, game_state, 3)
            # no map or enemy factory down -> cannot move or cannot transfer down
            if move_cost_down == 99999:
                action_mask[i, x, y, 9] = 0
                action_mask[i, x, y, 14] = 0
                can_move_down = False
            # if not enough power to move due to rubble cost
            if current_power <= move_cost_down + update_queue_cost:
                action_mask[i, x, y, 9] = 0
                can_move_down = False

            can_move_left = True
            move_cost_left = move_cost(unit, game_state, 4)
            # no map or enemy factory left -> cannot move or cannot transfer left
            if move_cost_left == 99999:
                action_mask[i, x, y, 10] = 0
                action_mask[i, x, y, 15] = 0
                can_move_left = False
            # if not enough power to move due to rubble cost
            if current_power <= move_cost_left + update_queue_cost:
                action_mask[i, x, y, 10] = 0
                can_move_left = False

            # if cannot move in any direction then simply cannot move
            if can_move_up == False and can_move_down == False and can_move_left == False and can_move_right == False:
                action_mask[i,x,y,2] = 0

            ###### TRANSFER ACTION TYPE ######
            # check if unit is on top of factory
            factory_center = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]
            can_transfer_center = True
            if factory_center == -1:
                can_transfer_center = False
                action_mask[i,x,y,11] = 0

            # check whats one tile up from the robot
            if unit.pos.y - 1 >= 0:
                factory_up = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y - 1]
                robot_up = game_state.board.get_units_at(Position(np.array([unit.pos.x, unit.pos.y - 1])))
            else:
                factory_up = -1
                robot_up = None

            can_transfer_up = True
            # if no ally factory or unit is up then mask transfer up
            if factory_up == -1 and robot_up is None:
                can_transfer_up = False
                action_mask[i,x,y,12] = 0
            
            if unit.pos.x + 1 <= 47:
                factory_right = game_state.board.factory_occupancy_map[unit.pos.x + 1][unit.pos.y]
                robot_right = game_state.board.get_units_at(Position(np.array([unit.pos.x + 1, unit.pos.y])))
            else:
                factory_right = -1
                robot_right = None

            can_transfer_right = True
            # if no ally factory or unit is right then mask transfer right
            if factory_right == -1 and robot_right is None:
                can_transfer_right = False
                action_mask[i,x,y,13] = 0

            if unit.pos.y + 1 <= 47:
                factory_down = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y + 1]
                robot_down = game_state.board.get_units_at(Position(np.array([unit.pos.x, unit.pos.y + 1])))
            else:
                factory_down = -1
                robot_down = None

            can_transfer_down = True
            # if no ally factory or unit is down then mask transfer down
            if factory_down == -1 and robot_down is None:
                can_transfer_down = False
                action_mask[i,x,y,14] = 0

            if unit.pos.x - 1 >= 0:
                factory_left = game_state.board.factory_occupancy_map[unit.pos.x - 1][unit.pos.y]
                robot_left = game_state.board.get_units_at(Position(np.array([unit.pos.x - 1, unit.pos.y])))
            else:
                factory_left = -1
                robot_left = None

            can_transfer_left = True
            # if no ally factory or unit is left then mask transfer left
            if factory_left == -1 and robot_left is None:
                can_transfer_left = False
                action_mask[i,x,y,15] = 0

            robot_ice = unit.cargo.ice
            has_ice = True
            # if unit has no ice then cannot transfer ice
            if robot_ice == 0:
                action_mask[i,x,y,20] = 0
                has_ice = False

            robot_ore = unit.cargo.ore
            has_ore = True
            # if unit has no ore then cannot transfer ore
            if robot_ore == 0:
                action_mask[i,x,y,21] = 0
                has_ore = False

            robot_power = unit.power
            has_power = True
            # if unit has no power then cannot transfer power
            if robot_power < 5:
                action_mask[i,x,y,22] = 0
                has_power = False

            # if there is no ally unit or ally factory in any adjacent tile -> mask TRANSFER and all transfer params
            # OR if there is no power, ice or ore in cargo to transfer -> mask TRANSFER and all transfer params
            if (can_transfer_left == False and can_transfer_up == False and can_transfer_down == False and can_transfer_right == False and can_transfer_center == False) or (has_power == False and has_ice == False and has_ore == False):
                action_mask[i,x,y,3] = 0
                action_mask[i,x,y,11:23] = 0
            

            ###### PICKUP DIG ACTION TYPE ######
            factory_there = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]

            # if there is no factory on top of unit -> cannot pick up power
            if factory_there == -1:
                action_mask[i, x, y, 4] = 0
                #action_mask[i, x, y, 25:30] = 0

            board_sum = (
                game_state.board.ice[unit.pos.x, unit.pos.y]
                + game_state.board.ore[unit.pos.x, unit.pos.y]
                + game_state.board.rubble[unit.pos.x, unit.pos.y]
                + game_state.board.lichen[unit.pos.x, unit.pos.y]
            )

            # mask dig if on top of factory or no resources to dig
            if board_sum == 0 or factory_there != -1:
                action_mask[i, x, y, 5] = 0

            # mask dig if no power (light robot)
            if current_power - update_queue_cost <= 5 and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 5] = 0

            # mask dig if no power (heavy robot)
            if current_power - update_queue_cost <= 60 and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 5] = 0

            # mask self destruct if no power (light robot)
            if current_power - update_queue_cost <= 10 and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 6] = 0

            # mask self destruct if no power (heavyrobot)
            if current_power - update_queue_cost  <= 100 and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 6] = 0

    return action_mask


#### UTILS FOR THE ACTION MASKS ####
def water_cost(factory, env):
    game_state = env.state
    owned_lichen_tiles = (game_state.board.lichen_strains == factory.state_dict()["strain_id"]).sum()
    return np.ceil(owned_lichen_tiles / env.env_cfg.LICHEN_WATERING_COST_FACTOR)

def action_queue_cost(unit, env_cfg):
    unit_type = str(unit.unit_type).split(".")[1]
    cost = env_cfg.ROBOTS[unit_type].ACTION_QUEUE_POWER_COST
    return cost

# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

def move_cost(unit, game_state, direction):
    board = game_state.board

    target_pos = unit.pos + move_deltas[direction]
    
    if (
        target_pos.x < 0
        or target_pos.y < 0
        or target_pos.y >= len(board.rubble)
        or target_pos.x >= len(board.rubble[0])
    ):
        return 99999

    factory_there = board.factory_occupancy_map[target_pos.x, target_pos.y]

    unit_agent_id = "player_0" if unit.team_id == 0 else "player_1"

    if (factory_there not in game_state.teams[unit_agent_id].factory_strains and factory_there != -1):
        return 99999

    rubble_at_target = board.rubble[target_pos.x][target_pos.y]

    return math.floor(
        unit.unit_cfg.MOVE_COST
        + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
    )