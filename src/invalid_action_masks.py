import gym
import numpy as np
import torch
import math
import luxai_s2.unit as luxai_unit

from IPython import embed

def get_factory_invalid_action_masks(envs, player):
    if player == "player_0":
        action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["factories"].nvec[1:].sum() + 1))
        envs = envs.envs

    if player == "player_1":
        action_mask = np.zeros((1, 48, 48, envs.action_space["factories"].nvec[1:].sum() + 1))
        envs = [envs]

    for i in range(len(envs)):
        env = envs[i]
        game_state = env.env_.state

        for unit_id in game_state.factories[player]:
            factory = game_state.factories[player][unit_id]
            x, y = factory.pos.x, factory.pos.y

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

            if water < water_cost(factory, env.env_):
                action_mask[i, x, y, 4] = 0     

    return action_mask

def get_robot_invalid_action_masks(envs, player):
    # the returned mask will be of shape (num_envs, map_size, map_size, total_unit_actions)
    if player == "player_0":
        action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["robots"].nvec[1:].sum() + 1))
        envs = envs.envs

    if player == "player_1":
        action_mask = np.zeros((1, 48, 48, envs.action_space["robots"].nvec[1:].sum() + 1))
        envs = [envs]

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

            # TODO: if cannot move in any direction -> Action type: Move should be ilegal
            move_cost_up = move_cost(unit, game_state, 1)
            
            # no map or enemy factory on top -> cannot move or cannot transfer up
            if move_cost_up == 99999:
                action_mask[i, x, y, 8] = 0
                action_mask[i, x, y, 13] = 0

            if current_power <= move_cost_up + update_queue_cost:
                action_mask[i, x, y, 8] = 0
            #else:
            #    print(env.env_.state.real_env_steps,":",unit_id, "at", (unit.pos.x, unit.pos.y) , "with power:", unit.power, "moving up cuz move cost is",move_cost_up,"and update cost is", update_queue_cost)


            move_cost_right = move_cost(unit, game_state, 2)
            # no map or enemy factory to the right -> cannot move or cannot transfer right
            if move_cost_right == 99999:
                action_mask[i, x, y, 9] = 0
                action_mask[i, x, y, 14] = 0

            if current_power <= move_cost_right + update_queue_cost:
                action_mask[i, x, y, 9] = 0
            #else:
            #    print(env.env_.state.real_env_steps,":",unit_id, "at", (unit.pos.x, unit.pos.y) , "with power:", unit.power, "moving right cuz move cost is",move_cost_right,"and update cost is", update_queue_cost)

            move_cost_down = move_cost(unit, game_state, 3)
            # no map or enemy factory down -> cannot move or cannot transfer down
            if move_cost_down == 99999:
                action_mask[i, x, y, 10] = 0
                action_mask[i, x, y, 15] = 0
            
            if current_power <= move_cost_down + update_queue_cost:
                action_mask[i, x, y, 10] = 0
            #else:
            #    print(env.env_.state.real_env_steps,":",unit_id, "at", (unit.pos.x, unit.pos.y) , "with power:", unit.power, "moving down cuz move cost is",move_cost_down,"and update cost is", update_queue_cost)

            move_cost_left = move_cost(unit, game_state, 4)
            # no map or enemy factory left -> cannot move or cannot transfer left
            if move_cost_left == 99999:
                action_mask[i, x, y, 11] = 0
                action_mask[i, x, y, 16] = 0
            
            if current_power <= move_cost_left + update_queue_cost:
                action_mask[i, x, y, 11] = 0
            #else:
            #    print(env.env_.state.real_env_steps,":",unit_id, "at", (unit.pos.x, unit.pos.y) , "with power:", unit.power, "moving left cuz move cost is",move_cost_left,"and update cost is", update_queue_cost)

            factory_there = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]

            # if there is no factory on top of unit -> cannot pick up (no pick up amount [4 actions] and no pick up resource [5 actions])
            if factory_there == -1:
                action_mask[i, x, y, 4] = 0
                action_mask[i, x, y, 26:35] = 0
            else:
                action_mask[i, x, y, 5] = 0

            # mask dig if no power (light robot)
            if current_power <= 5 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 5] = 0

            # mask dig if no power (heavy robot)
            if current_power <= 60  + update_queue_cost and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 5] = 0

            # mask self destruct if no power (light robot)
            if current_power <= 10 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 6] = 0

            # mask self destruct if no power (heavyrobot)
            if current_power <= 100 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 6] = 0

            # mask recharge if already have enough power
            if (current_power - update_queue_cost) / unit.battery_capacity >= 0.25 * unit.battery_capacity:
                action_mask[i, x, y, 35] = 0

            if (current_power - update_queue_cost) / unit.battery_capacity >= 0.5 * unit.battery_capacity:
                action_mask[i, x, y, 36] = 0

            if (current_power - update_queue_cost) / unit.battery_capacity >= 0.75 * unit.battery_capacity:
                action_mask[i, x, y, 37] = 0

    return action_mask

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

    if (
        factory_there not in game_state.teams[unit_agent_id].factory_strains
        and factory_there != -1
    ):
        return 99999

    rubble_at_target = board.rubble[target_pos.x][target_pos.y]

    return math.floor(
        unit.unit_cfg.MOVE_COST
        + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
    )