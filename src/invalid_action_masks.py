import gym
import numpy as np
import torch
import math
import luxai_s2.unit as luxai_unit


def get_factory_invalid_action_masks(envs, player):
    action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["factories"].nvec[1:].sum()+1))
    
    for i in range(len(envs.envs)):
        env = envs.envs[i]
        game_state = env.env_.state

        for unit_id in game_state.factories[player]:
            factory = game_state.factories[player][unit_id]
            x, y = factory.pos.x, factory.pos.y

            # start by assuming all actions are legal where there is a unit
            action_mask[i, x, y, :] = 1
    
    return action_mask

def get_robot_invalid_action_masks(envs, player):
    # the returned mask will be of shape (num_envs, map_size, map_size, total_unit_actions)
    action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["robots"].nvec[1:].sum()+1))

    for i in range(len(envs.envs)):
        env = envs.envs[i]
        game_state = env.env_.state

        for unit_id in game_state.units[player]:
            unit = game_state.units[player][unit_id]
            x, y = unit.pos.x, unit.pos.y

            # start by assuming all actions are legal where there is a unit
            action_mask[i, x, y, :] = 1

            current_power = unit.power

            # if not enough power to update action queue then we cannot do anything but NOOP
            if current_power < action_queue_cost(unit):
                action_mask[i, x, y, :] = 0
                action_mask[i, x, y, 0] = 1

            # TODO: if cannot move in any direction -> Action type: Move should be ilegal

            move_cost_up = move_cost(unit, game_state, 1)
            # no map or enemy factory on top -> cannot move or cannot transfer up
            if move_cost_up == -1:
                action_mask[i, x, y, 7] = 0
                action_mask[i, x, y, 11] = 0
                action_mask[i, x, y, 15:24] = 0

            if current_power < move_cost_up:
                action_mask[i, x, y, 7] = 0

            move_cost_right = move_cost(unit, game_state, 2)
            # no map or enemy factory to the right -> cannot move or cannot transfer right
            if move_cost_right == -1:
                action_mask[i, x, y, 8] = 0
                action_mask[i, x, y, 12] = 0
                action_mask[i, x, y, 15:24] = 0

            if current_power < move_cost_right:
                action_mask[i, x, y, 8] = 0

            move_cost_down = move_cost(unit, game_state, 3)
            # no map or enemy factory down -> cannot move or cannot transfer down
            if move_cost_down == -1:
                action_mask[i, x, y, 9] = 0
                action_mask[i, x, y, 13] = 0
                action_mask[i, x, y, 15:24] = 0
            
            if current_power < move_cost_down:
                action_mask[i, x, y, 9] = 0

            move_cost_left = move_cost(unit, game_state, 4)
            # no map or enemy factory left -> cannot move or cannot transfer left
            if move_cost_left == -1:
                action_mask[i, x, y, 10] = 0
                action_mask[i, x, y, 14:24] = 0
            
            if current_power < move_cost_left:
                action_mask[i, x, y, 10] = 0

            factory_there = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]

            # if there is no factory on top of unit -> cannot pick up (no pick up amount [4 actions] and no pick up resource [5 actions])
            if factory_there == -1:
                action_mask[i, x, y, 3] = 0
                action_mask[i, x, y, 24:33] = 0
            else:
                action_mask[i, x, y, 4] = 0

            # mask dig if no power (light robot)
            if current_power < 5 + action_queue_cost(unit) and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 4] = 0

            # mask dig if no power (heavy robot)
            if current_power < 60  + action_queue_cost(unit) and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 4] = 0

            # mask self destruct if no power (light robot)
            if current_power < 10 + action_queue_cost(unit) and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 5] = 0

            # mask self destruct if no power (heavyrobot)
            if current_power < 100 + action_queue_cost(unit) and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 5] = 0

            # mask recharge if already have enough power
            if (current_power - action_queue_cost(unit)) / unit.battery_capacity > 0.25 * unit.battery_capacity:
                action_mask[i, x, y, 33] = 0

            if (current_power - action_queue_cost(unit)) / unit.battery_capacity > 0.5 * unit.battery_capacity:
                action_mask[i, x, y, 34] = 0

            if (current_power - action_queue_cost(unit)) / unit.battery_capacity > 0.75 * unit.battery_capacity:
                action_mask[i, x, y, 35] = 0

    return action_mask

def action_queue_cost(unit):
    cost = unit.env_cfg.ROBOTS[unit.unit_type].ACTION_QUEUE_POWER_COST
    return cost


# a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

def move_cost(unit, game_state, direction):
    board = game_state.board
    target_pos = unit.pos + move_deltas[direction]
    if target_pos[0] < 0 or target_pos[1] < 0 or target_pos[1] >= len(board.rubble) or target_pos[0] >= len(board.rubble[0]):
        return -1

    factory_there = board.factory_occupancy_map[target_pos[0], target_pos[1]]

    if factory_there not in game_state.teams[unit.agent_id].factory_strains and factory_there != -1:
        # print("Warning, tried to get move cost for going onto a opposition factory", file=sys.stderr)
        return -1

    rubble_at_target = board.rubble[target_pos[0]][target_pos[1]]
    
    return math.floor(unit.unit_cfg.MOVE_COST + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)