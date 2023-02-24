import gym
import numpy as np
import torch
import math
import luxai_s2.unit as luxai_unit
from luxai_s2.map.position import Position

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

            if water -1 <= water_cost(factory, env.env_):
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

            ###### MOVE ACTION TYPE ######
            move_cost_up = move_cost(unit, game_state, 1)
            can_move_up = True
            # no map or enemy factory on top -> cannot move or cannot transfer up
            if move_cost_up == 99999:
                action_mask[i, x, y, 7] = 0
                action_mask[i, x, y, 12] = 0
                can_move_up = False
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
            if current_power <= move_cost_left + update_queue_cost:
                action_mask[i, x, y, 10] = 0
                can_move_left = False

            # if cannot move in any direction then simply cannot move
            if not can_move_up and not can_move_down and not can_move_left and not can_move_right:
                action_mask[i,x,y,2] = 0

            ###### TRANSFER ACTION TYPE ######
            factory_center = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]
            can_transfer_center = True
            if factory_center == -1:
                can_transfer_center = False
                action_mask[i,x,y,11] = 0

            if unit.pos.y - 1 >= 0:
                factory_up = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y - 1]
                robot_up = game_state.board.get_units_at(Position(np.array([unit.pos.x, unit.pos.y - 1])))
            else:
                factory_up = -1
                robot_up = None

            can_transfer_up = True
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
            if factory_left == -1 and robot_left is None:
                can_transfer_left = False
                action_mask[i,x,y,15] = 0

            if not can_transfer_left and not can_transfer_up and not can_transfer_down and not can_transfer_right and not can_transfer_center:
                action_mask[i,x,y,3] = 0

            ###### PICKUP DIG ACTION TYPE ######
            factory_there = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]

            # if there is no factory on top of unit -> cannot pick up (no pick up amount [4 actions] and no pick up resource [5 actions])
            if factory_there == -1:
                action_mask[i, x, y, 4] = 0
                action_mask[i, x, y, 25:30] = 0

            # mask dig if on top of factory or no resources to dig
            board_sum = (
                game_state.board.ice[unit.pos.x, unit.pos.y]
                + game_state.board.ore[unit.pos.x, unit.pos.y]
                + game_state.board.rubble[unit.pos.x, unit.pos.y]
                + game_state.board.lichen[unit.pos.x, unit.pos.y]
            )
            if board_sum == 0 or factory_there != -1:
                action_mask[i, x, y, 5] = 0

            # mask dig if no power (light robot)
            if current_power - update_queue_cost <= 5 and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 5] = 0

            # mask dig if no power (heavy robot)
            if current_power - update_queue_cost <= 60 and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 5] = 0

            # mask self destruct if no power (light robot)
            if current_power <= 10 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 6] = 0

            # mask self destruct if no power (heavyrobot)
            if current_power <= 100 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 6] = 0

    return action_mask

def get_robot_invalid_action_masks_action_type(envs, player):
    """
    Returns the mask for invalid action types [source unit pos, NOOP, MOVE, TRANSFER, PICKUP, DIG, SELF-DESTRUCT]
    """
    if player == "player_0":
        action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["robots"].nvec[1] + 1))
        envs = envs.envs

    if player == "player_1":
        action_mask = np.zeros((1, 48, 48, envs.action_space["robots"].nvec[1] + 1))
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

            # only noop is valid
            if current_power <= update_queue_cost:
                action_mask[i, x, y, 2:] = 0

            ###### MOVE ACTION TYPE ######
            move_cost_up = move_cost(unit, game_state, 1)
            can_move_up = True
            if move_cost_up == 99999:
                can_move_up = False

            if current_power - update_queue_cost <= move_cost_up:
                can_move_up = False

            move_cost_right = move_cost(unit, game_state, 2)
            can_move_right = True
            if move_cost_right == 99999:
                can_move_right = False

            if current_power - update_queue_cost <= move_cost_right:
                can_move_right = False

            can_move_down = True
            move_cost_down = move_cost(unit, game_state, 3)
            if move_cost_down == 99999:
                can_move_down = False
            
            if current_power - update_queue_cost <= move_cost_down:
                can_move_down = False

            can_move_left = True
            move_cost_left = move_cost(unit, game_state, 4)
            if move_cost_left == 99999:
                can_move_left = False
            
            if current_power - update_queue_cost <= move_cost_left:
                can_move_left = False

            #print("Unit", unit_id, "has power:", current_power - update_queue_cost, "and the costs for moving are", move_cost_up, move_cost_right, move_cost_down, move_cost_left)

            if not can_move_up and not can_move_down and not can_move_left and not can_move_right:
                action_mask[i,x,y,2] = 0
            #    print("Restricting movement for unit", unit_id)

            ###### TRANSFER ACTION TYPE ######
            factory_center = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]
            can_transfer_center = True
            if factory_center == -1:
                can_transfer_center = False

            if unit.pos.y - 1 >= 0:
                robot_up = game_state.board.get_units_at(Position(np.array([unit.pos.x, unit.pos.y - 1])))
                factory_up = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y - 1]
            else:
                robot_up = None
                factory_up = -1

            can_transfer_up = True
            if factory_up == -1 and robot_up is None:
                can_transfer_up = False
            
            if unit.pos.x + 1 <= 47:
                robot_right = game_state.board.get_units_at(Position(np.array([unit.pos.x + 1, unit.pos.y])))
                factory_right = game_state.board.factory_occupancy_map[unit.pos.x + 1][unit.pos.y]
            else:
                robot_right = None
                factory_right = -1

            can_transfer_right = True
            if factory_right == -1 and robot_right is None:
                can_transfer_right = False

            if unit.pos.y + 1 <= 47:
                robot_down = game_state.board.get_units_at(Position(np.array([unit.pos.x, unit.pos.y + 1])))
                factory_down = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y + 1]
            else:
                robot_down = None
                factory_down = -1

            can_transfer_down = True
            if factory_down == -1 and robot_down is None:
                can_transfer_down = False

            if unit.pos.x -1 >= 0:
                robot_left = game_state.board.get_units_at(Position(np.array([unit.pos.x - 1, unit.pos.y])))
                factory_left = game_state.board.factory_occupancy_map[unit.pos.x - 1][unit.pos.y]
            else:
                robot_left = None
                factory_left = -1

            can_transfer_left = True
            if factory_left == -1 and robot_left is None:
                can_transfer_left = False

            if not can_transfer_left and not can_transfer_up and not can_transfer_down and not can_transfer_right and not can_transfer_center:
                action_mask[i,x,y,3] = 0

            ore, metal, ice, water = unit.cargo.ore, unit.cargo.metal, unit.cargo.ice, unit.cargo.water
            total_cargo = ore + metal + ice + water
            
            if total_cargo <= 15 and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i,x,y,3] = 0

            if total_cargo <= 150 and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i,x,y,3] = 0
                
            ###### PICKUP AND DIG ACTION TYPE ######
            # TODO: dont pick up a resource from a factory if there is no resource

            if factory_center == -1:
                action_mask[i,x,y,4] = 0
            else:
                action_mask[i,x,y,5] = 0

            if total_cargo >= 140 and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i,x,y,4] = 0
            
            if total_cargo >= 930 and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i,x,y,4] = 0

            if current_power <= 5 + update_queue_cost  and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i,x,y,5] = 0

            if current_power <= 60 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i,x,y,5] = 0
            
            ###### SELF DESTRUCT ACTION TYPE ######
            if current_power <= 10 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.LIGHT:
                action_mask[i, x, y, 6] = 0
            if current_power <= 100 + update_queue_cost and unit.unit_type == luxai_unit.UnitType.HEAVY:
                action_mask[i, x, y, 6] = 0
    
            ###### SELF DESTRUCT ACTION TYPE ######
            if (current_power - update_queue_cost) / unit.battery_capacity >= 0.75 * unit.battery_capacity:
                action_mask[i, x, y, 7] = 0

    return action_mask    


def get_robot_invalid_action_masks_action_params(envs, player, action_types):
    # the returned mask will be of shape (num_envs, map_size, map_size, total_unit_actions)
    if player == "player_0":
        action_mask = np.zeros((len(envs.envs), 48, 48, envs.single_action_space["robots"].nvec[2:].sum()))
        envs = envs.envs

    if player == "player_1":
        action_mask = np.zeros((1, 48, 48, envs.action_space["robots"].nvec[2:].sum()))
        envs = [envs]

    for i in range(len(envs)):
        env = envs[i]
        game_state = env.env_.state

        for unit_id in game_state.units[player]:
            unit = game_state.units[player][unit_id]
            x, y = unit.pos.x, unit.pos.y

            current_power = unit.power
            update_queue_cost = action_queue_cost(unit, env.env_.env_cfg)

            # read the selected action type
            action_type = action_types[i, x, y]
            
            ###### NOOP ######
            if action_type == 0:
                action_mask[i, x, y, :] = 0
            else:
                ###### MOVE ACTION PARAMS ######
                if action_type == 1:
                    action_mask[i, x, y, 0:4] = 1
                    #print("--------------------------------")
                    move_cost_up = move_cost(unit, game_state, 1)
                    if move_cost_up == 99999:
                        action_mask[i, x, y, 0] = 0
                        #print("masked move up bc off-map")

                    if current_power - update_queue_cost <= move_cost_up:
                        action_mask[i, x, y, 0] = 0
                        #print("masked move up bc no power", "current_power is", current_power - update_queue_cost, "and cost is", move_cost_up)


                    move_cost_right = move_cost(unit, game_state, 2)
                    if move_cost_right == 99999:
                        action_mask[i, x, y, 1] = 0
                        #print("masked move right bc off-map")

                    if current_power - update_queue_cost <= move_cost_right:
                        action_mask[i, x, y, 1] = 0
                        #print("masked move right bc no power", "current_power is", current_power - update_queue_cost, "and cost is", move_cost_right)

                    move_cost_down = move_cost(unit, game_state, 3)
                    if move_cost_down == 99999:
                        action_mask[i, x, y, 2] = 0
                        #print("masked move down bc off-map")
                    
                    if current_power - update_queue_cost <= move_cost_down:
                        action_mask[i, x, y, 2] = 0
                        #print("masked move down bc no power", "current_power is", current_power - update_queue_cost, "and cost is", move_cost_down)

                    move_cost_left = move_cost(unit, game_state, 4)
                    if move_cost_left == 99999:
                        action_mask[i, x, y, 3] = 0
                        #print("masked move left bc off-map")

                    if current_power - update_queue_cost <= move_cost_left:
                        action_mask[i, x, y, 3] = 0
                        #print("masked move left bc no power", "current_power is", current_power - update_queue_cost, "and cost is", move_cost_left)

                    # all but movement params are ilegal
                    action_mask[i, x, y, 4:] = 0

                ###### TRANSFER ACTION PARAMS ######
                elif action_type == 2:

                    action_mask[i, x, y, 4:18] = 1

                    factory_center = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y]
                    
                    if factory_center == -1:
                        action_mask[i, x, y, 4] = 0

                    if unit.pos.y - 1 >= 0:
                        robot_up = game_state.board.get_units_at(Position(np.array([unit.pos.x, unit.pos.y - 1])))
                        factory_up = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y - 1]
                    else:
                        factory_up = -1
                        robot_up = None

                    if factory_up == -1 and robot_up is None:
                        action_mask[i, x, y, 5] = 0

                    if unit.pos.x + 1 <= 47:
                        robot_right = game_state.board.get_units_at(Position(np.array([unit.pos.x + 1, unit.pos.y])))
                        factory_right = game_state.board.factory_occupancy_map[unit.pos.x + 1][unit.pos.y]
                    else:
                        factory_right = -1
                        robot_right = None

                    if factory_right == -1 and robot_right is None:
                        action_mask[i, x, y, 6] = 0

                    if unit.pos.y + 1 <= 47:
                        robot_down = game_state.board.get_units_at(Position(np.array([unit.pos.x, unit.pos.y + 1])))
                        factory_down = game_state.board.factory_occupancy_map[unit.pos.x][unit.pos.y + 1]
                    else:
                        factory_down = -1
                        robot_down = None

                    if factory_down == -1 and robot_down is None:
                        action_mask[i, x, y, 7] = 0

                    if unit.pos.x -1 >= 0:
                        robot_left = game_state.board.get_units_at(Position(np.array([unit.pos.x - 1, unit.pos.y])))
                        factory_left = game_state.board.factory_occupancy_map[unit.pos.x - 1][unit.pos.y]
                    else:
                        factory_left = -1
                        robot_left = None

                    if factory_left == -1 and robot_left is None:
                        action_mask[i, x, y, 8] = 0
                        
                    power, ore, metal, ice, water = unit.power, unit.cargo.ore, unit.cargo.metal, unit.cargo.ice, unit.cargo.water
                    total_cargo = ore + metal + ice + water

                    if power < 15:
                        action_mask[i, x, y, 13] = 0
                    
                    if ore < 5:
                        action_mask[i, x, y, 14] = 0
                        
                    if metal < 5:
                        action_mask[i, x, y, 15] = 0

                    if ice < 5:
                        action_mask[i, x, y, 16] = 0

                    if water < 5:
                        action_mask[i, x, y, 17] = 0

                    action_mask[i, x, y, 0:4] = 0
                    action_mask[i, x, y, 18:] = 0

                elif action_type == 3:
                    action_mask[i, x, y, 18:] = 1

                    power, ore, metal, ice, water = unit.power, unit.cargo.ore, unit.cargo.metal, unit.cargo.ice, unit.cargo.water
                    total_cargo = ore + metal + ice + water

                    cargo_capacity = 100 if unit.unit_type == luxai_unit.UnitType.LIGHT else 1000

                    # cannot pickup resources if full cargo
                    if total_cargo >= (0.9 * cargo_capacity):
                         action_mask[i, x, y, 19:] = 0    

                    # cannot pickup power if full battery
                    if current_power >= (0.9 * unit.battery_capacity):
                        action_mask[i, x, y, 18] = 0

                    action_mask[i, x, y, 0:18] = 0

                elif action_type == 4:
                    # digging was selected (so it wsnt ilegal), and there are no action params for digging so mask them all
                    action_mask[i, x, y, :] = 0
                elif action_type == 5:
                    # self-destruct was selected (so it wsnt ilegal), and there are no action params for self-destruct so mask them all
                    action_mask[i, x, y, :] = 0

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

    if (factory_there not in game_state.teams[unit_agent_id].factory_strains and factory_there != -1):
        return 99999

    rubble_at_target = board.rubble[target_pos.x][target_pos.y]

    return math.floor(
        unit.unit_cfg.MOVE_COST
        + unit.unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
    )