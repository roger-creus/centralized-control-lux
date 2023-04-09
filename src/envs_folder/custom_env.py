
import gym
import numpy as np
import luxai_s2.env
import torch
import math
import copy

from luxai_s2.state import ObservationStateDict, StatsStateDict
from IPython import embed
from jpype.types import JArray, JInt
from gym import spaces
from luxai_s2.env import LuxAI_S2
from .env_utils import *

"""
This is a custom wrapper for the Lux Env with custom Obs and Action spaces and self-play embedded
1) Handles self-play (from outside this script, env.step() only requires the agent action, needed for VectorizedEnvironments)
2) Every reset() call makes it to the normal game phase directly
"""

class CustomLuxEnv(gym.Env):
    def __init__(
            self, 
            self_play = False, 
            sparse_reward = True,
            simple_obs = True,
            env_cfg = None, 
            device = "cuda:0", 
            PATH_AGENT_CHECKPOINTS = "checkpoints"
        ):

        # the true env
        env_id = "LuxAI_S2-v0"
        self.env_ = gym.make(env_id, verbose=-1, collect_stats=True)

        # whether to use self-play or sparse-reward or simple-obs
        self.self_play = self_play
        self.is_sparse_reward = sparse_reward
        self.simple_obs = simple_obs

        # used for the dense reward
        self.prev_step_metrics = None

        # initialized at set_enemy_agent()
        self.enemy_agent = None

        self.device = device
        self.PATH_AGENT_CHECKPOINTS = PATH_AGENT_CHECKPOINTS
        
        ##### OBS SPACE ####

        # observation space can be simple (only 1s and 0s) or detailed. Default=detailed
        if self.simple_obs:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(48, 48, 19), dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(48, 48, 22), dtype=np.float64)


        ##### ACTION SPACE ####

        # ROBOTS
        # dim 0: position in map -- LENGTH 48 * 48 (e.g. pos (3,2) is (3 + 2*48) )
        # dim 1: action type [NOOP, move, transfer, pickup, dig, self-destruct] -- LENGTH 6
        # dim 2: move direction [up, right, down, left] -- LENGTH 4
        # dim 3: transfer direction [center, up, right, down, left] -- LENGTH 5
        # dim 4: transfer amount [25%, 50%, 75%, 95%] -- LENGTH 4
        # dim 5: transfer material [ice, ore, power] --LENGTH 3
        
        # FACTORIES
        # dim 0: position in map -- LENGTH 48 * 48 (e.g. pos (3,2) is (3 + 2*48) )
        # dim 1: factory action [NOOP, build light robot, build heavy robot, grow lichen] -- LENGTH 4
        self.action_space = gym.spaces.Dict({
            'robots': gym.spaces.MultiDiscrete(
                [(48 * 48) - 1, 6, 4, 5, 4, 3]
            ),
            'factories': gym.spaces.MultiDiscrete(
                [(48 * 48) - 1, 4]
            ),
        })

        # for the reward definition
        self.num_factories = 0
        self.num_units = 0

        # keeps updated versions of the player and enemy observations (preprocessed observations)
        self.current_player_obs = None
        self.current_enemy_obs = None
        # keep the current game observation
        self.current_state = None

    # gets a learning agent action (model format not Lux format)
    def step(self, action):

        # turn the raw model outputs to game actions
        player_action = self.act_(action, "player_0")  # returs {"player_0" : actions}

        if self.self_play:
            # get enemy action (raw output model)
            enemy_action = self.enemy_step()
            # turn the raw enemy model outputs to game actions
            enemy_action = self.act_(enemy_action, "player_1") # returs {"player_1" : actions}

        # if not self play we have a very bad enemy heurisitc
        if self.self_play is False:
            enemy_action = self.enemy_heuristic_step()

        # final dict of actions from both players to send to the game
        actions = {**player_action, **enemy_action} 

        # step actions to true env
        observations, reward, done, info = self.env_.step(actions)

        # player_0 has died
        if reward["player_0"] == -1000:
            # if players have drawn
            if reward["player_1"] == -1000:
                info["player_0"]["result"] = 0
                reward_now = 0
            
            # else, player_1 has lost
            else:
                if self.is_sparse_reward:
                    reward_now = -1
                else:
                    reward_now = 0
                    
                info["player_0"]["result"] = -1
        
        # player_1 has died
        elif reward["player_1"] == -1000:
            # if players have drawn
            if reward["player_0"] == -1000:
                reward_now = 0
                info["player_0"]["result"] = 0
            
            # else, player 1 has won
            else:
                if self.is_sparse_reward is True:
                    reward_now = 1
                else:
                    reward_now = 0
                info["player_0"]["result"] = 1
        
        # no players have died
        else:
            if self.is_sparse_reward is True:
                reward_now = 0
            else:

                #### DENSE REWARD ####
                stats: StatsStateDict = self.env_.state.stats["player_0"]
                metrics = dict()

                # new ICE in robots and new WATER in factories (+)
                metrics["ice_dug"] = (stats["generation"]["ice"]["HEAVY"] + stats["generation"]["ice"]["LIGHT"])
                metrics["water_produced"] = stats["generation"]["water"]

                # new ORE in robots and new METAL in factories (+)
                metrics["ore_dug"] = (stats["generation"]["ore"]["HEAVY"] + stats["generation"]["ore"]["LIGHT"])
                metrics["metal_produced"] = stats["generation"]["metal"]

                # new lichen (+)
                metrics["lichen"] = stats["generation"]["lichen"]

                # consumption of power in robots (-)
                metrics["power_used"] = stats["consumption"]["power"]["LIGHT"] + stats["consumption"]["power"]["HEAVY"]

                # pickup power (+)
                metrics["pickup_power"] = stats["pickup"]["power"]

                # consumption of water (-)
                metrics["consumed_water"] = stats["consumption"]["water"]

                # new lights and heavies (+)
                metrics["count_heavies"] = stats["generation"]["built"]["HEAVY"]
                metrics["count_lights"] = stats["generation"]["built"]["LIGHT"]

                # lost factories, lights and heavies (-)
                metrics["destroyed_factories"] = stats["destroyed"]["FACTORY"]
                metrics["destroyed_heavies"] = stats["destroyed"]["HEAVY"]
                metrics["destroyed_lights"] = stats["destroyed"]["LIGHT"]

                reward_now = 0
                if self.prev_step_metrics is not None:
                    
                    # positive reward for mining ice and generating water
                    ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
                    water_produced_this_step = (metrics["water_produced"] - self.prev_step_metrics["water_produced"])
                    reward_now += (ice_dug_this_step / 20) + (water_produced_this_step/10)

                    # positive reward for mining ore and generating metal
                    ore_dug_this_step = metrics["ore_dug"] - self.prev_step_metrics["ore_dug"]
                    metal_produced_this_step = (metrics["metal_produced"] - self.prev_step_metrics["metal_produced"])
                    reward_now += ((ore_dug_this_step / 40) + metal_produced_this_step / 10)

                    # positive reward for creating new robots
                    new_lights = metrics["count_lights"] - self.prev_step_metrics["count_lights"]
                    reward_now += new_lights / 50
                    new_heavies = metrics["count_heavies"] - self.prev_step_metrics["count_heavies"]
                    reward_now += (new_heavies * 10) / 50

                    # negative reward for robots dying
                    destroyed_lights = metrics["destroyed_lights"] - self.prev_step_metrics["destroyed_lights"]
                    reward_now -= destroyed_lights / 50
                    destroyed_heavies = metrics["destroyed_heavies"] - self.prev_step_metrics["destroyed_heavies"]
                    reward_now -= (destroyed_heavies * 10) / 50

                    # BIG negative reward for consuming water
                    #consumed_water = metrics["consumed_water"] - self.prev_step_metrics["consumed_water"]
                    #reward_now -= consumed_water / 10

                    # SMALL positive reward for creating lichen
                    new_lichen = metrics["lichen"] - self.prev_step_metrics["lichen"]
                    
                    # its worse to lose lichen than not to win it
                    if new_lichen > 0:
                        reward_now += new_lichen / 100
                    else:
                        reward_now -= new_lichen / 10

                    # negative reward for loosing a factory
                    destroyed_factories = metrics["destroyed_factories"] - self.prev_step_metrics["destroyed_factories"]
                    reward_now -= destroyed_factories * 3

                self.prev_step_metrics = copy.deepcopy(metrics)

        done =  done["player_0"]

        # keep the current game state updated
        self.current_state = observations
    
        # update the current observations for both players and preprocess them into model input format
        self.current_player_obs = self.simple_obs_(observations, "player_0") if self.simple_obs == True else self.obs_(observations, "player_0")
        self.current_enemy_obs = self.simple_obs_(observations, "player_1")  if self.simple_obs == True else self.obs_(observations, "player_1")

        # important: only return from the point of view of player! enemy is handled as part of the environment
        return self.current_player_obs, reward_now, done, info["player_0"]

    # heurisitc for placing factories
    def placement_heuristic_pro(self, observation, agent):
        ice = observation[agent]["board"]["ice"]
        ore = observation[agent]["board"]["ore"]
        rubble = observation[agent]["board"]["rubble"]
        
        ice_distances = [manhattan_dist_to_nth_closest(ice, i) for i in range(1,5)]
        ore_distances = [manhattan_dist_to_nth_closest(ore, i) for i in range(1,5)]
        
        ICE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25]) 
        weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

        ORE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
        weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

        ICE_PREFERENCE = np.random.uniform(2,4) # if you want to make ore more important, change to 0.3 for example

        combined_resource_score = (weigthed_ice_dist * ICE_PREFERENCE + weigthed_ore_dist)
        combined_resource_score = (np.max(combined_resource_score) - combined_resource_score) * observation[agent]["board"]["valid_spawns_mask"]

        if self.is_sparse_reward:
            low_rubble = (rubble<25)
            low_rubble_scores = np.zeros_like(low_rubble, dtype=float)

            for i in range(low_rubble.shape[0]):
                for j in range(low_rubble.shape[1]):
                    low_rubble_scores[i,j] = count_region_cells(low_rubble, (i,j), min_dist=0, max_dist=8, exponent=0.9)

            overall_score = (low_rubble_scores*2 + combined_resource_score ) * observation[agent]["board"]["valid_spawns_mask"]
        else:
            overall_score = (combined_resource_score) * observation[agent]["board"]["valid_spawns_mask"]

        best_loc = np.argmax(overall_score)
        
        return np.unravel_index(best_loc, (48, 48))

    # heuristic for bidding
    def bidding_heuristic(self, resources_amount):
        # List of possible means for the Gaussian distribution
        means_list = [-resources_amount*0.50, -resources_amount*0.25, -resources_amount*0.125, 0, resources_amount*0.1250, resources_amount*0.25, resources_amount*0.5]
        means = np.random.choice(means_list, size=2)
        variances_multiplier = np.random.choice([0.025, 0.05, 0.075], size=2)
        # Variances that are used for the Gaussian distribution
        variances = resources_amount*variances_multiplier
        
        bids = np.random.randn(2)*variances+means
        # Making sure that the bids don't exceed the total resources amount
        for i in range(len(bids)):
            if bids[i] > resources_amount:
                bids[i] = resources_amount
            if bids[i] < -resources_amount:
                bids[i] = -resources_amount
        
        return bids

    # reset the env when done
    def reset(self, load_new_enemy=True):
        
        # we sample a new opponent at each game
        if self.self_play and load_new_enemy==True:
            self.update_enemy_agent()
        
        observations = self.env_.reset()
        self.num_factories = 0
        self.num_units = 0
        self.prev_step_metrics = None

        # total resources that are available to bid
        resources_amount = observations["player_0"]["board"]["factories_per_team"]*150
        
        # TODO: use a heurisitc
        #bids = self.bidding_heuristic(resources_amount)
        bids = [0,0]
        
        actions = {"player_0" : {"bid" : bids[0], "faction" : "AlphaStrike"}, "player_1" : {"bid" : bids[1], "faction" : "AlphaStrike"}}

        # step into factory placement phase
        observations, _, _, _ = self.env_.step(actions)

        factory_amount = {"player_0" : 0, "player_1" : 0}
        for player in factory_amount.keys():
            factory_amount[player] = observations["player_0"]["teams"][player]["water"]/observations["player_0"]["teams"][player]["factories_to_place"]
        
        # handle all the factory placement phase
        while self.env_.state.real_env_steps < 0:
            action = dict()
            
            for agent in self.env_.agents:
                if my_turn_to_place_factory(observations[agent]["teams"][agent]["place_first"], self.env_.state.env_steps):
                    # TODO: get action from placer model
                    spawn_loc = self.placement_heuristic_pro(observations, agent)
                    action[agent] = dict(spawn=spawn_loc, metal=factory_amount[agent], water=factory_amount[agent])
                else:
                    action[agent] = dict()
            
            observations, _, _, _ = self.env_.step(action)

        # start total_water and total_metal used in to compute rewards
        factories_water = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["water"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
        self.total_water = factories_water
        factories_metal = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["metal"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
        self.total_metal = factories_metal
        self.num_factories = observations["player_0"]["board"]["factories_per_team"]

        # keep the current game state updated
        self.current_state = observations

        # finally reset into the normal game
        self.current_player_obs = self.simple_obs_(observations, "player_0") if self.simple_obs == True else self.obs_(observations, "player_0")
        self.current_enemy_obs = self.simple_obs_(observations, "player_1") if self.simple_obs == True else self.obs_(observations, "player_1")
        
        # important: return only from players perspective! Enemy is handled as part of the environment
        return self.current_player_obs

    # rendering
    def render(self, mode):
        return self.env_.render(mode)

    # transform the raw output actions into game actions that can be passed to the real env.
    def act_(self, action, player):
        # IMPORTANT: see \luxai_s2\spaces\act_space.py
        # IMPORTANT: see \luxai_s2\spaces\act_space.py
        # IMPORTANT: see \luxai_s2\spaces\act_space.py
        
        factory_actions = action["factories"]
        robot_actions = action["robots"]

        # we will fill this with the game actions
        commited_actions = {player : {}}

        # get the current true state of game
        game_state = self.env_.state

        #### FACTORIES ####
        for action in factory_actions:
            x = action[0] // 48
            y = action[0] % 48
            pos = Position(np.array([x,y]))

            factory = game_state.board.get_factory_at(game_state, pos)

            # important action[1] == 0 is NOOP, so dont do anything

            if action[1] == 1:
                # build light robot
                commited_actions[player][factory.unit_id] = 0
            elif action[1] == 2:
                # build heavy robot
                commited_actions[player][factory.unit_id] = 1
            elif action[1] == 3:
                # grow lichen
                commited_actions[player][factory.unit_id] = 2
            else:
                if action[1] != 0:
                    print("Factory action not implemented")

        #### ROBOTS ####
        for action in robot_actions:
            x = action[0] // 48
            y = action[0] % 48
            pos = Position(np.array([x,y]))

            robot = game_state.board.get_units_at(pos)[0]

            # this will be the action in the format Lux env wants it
            crafted_action = np.zeros(6)

            # action[1] == 0 means NOOP
            if action[1] == 0:
                crafted_action = None

            elif action[1] == 1:
                # action_type = MOVE
                crafted_action[0] = 0
                crafted_action[1] = action[2] + 1
            
            elif action[1] == 2:
                # action_type = TRANSFER
                crafted_action[0] = 1
                # transfer direction
                crafted_action[1] = action[3]
                # transfer resource
                if action[5] == 0:
                    crafted_action[2] = 0
                elif action[5] == 1:
                    crafted_action[2] = 1
                elif action[5] == 2:
                    crafted_action[2] = 4
                else:
                    print("incorrect transfer resource")

                # transfer amount = 0.25
                if action[4] == 0:
                    if action[5] == 2:
                        amount = robot.power * 0.25
                    else:
                        resource = RESOURCE_MAPPING[action[5]]
                        amount = robot.cargo.state_dict()[resource] * 0.25
                
                # transfer amount = 0.5
                elif action[4] == 1:
                    if action[5] == 2:
                        amount = robot.power * 0.5
                    else:
                        resource = RESOURCE_MAPPING[action[5]]
                        amount = robot.cargo.state_dict()[resource] * 0.5
                
                # transfer amount = 0.75
                elif action[4] == 2:
                    if action[5] == 2:
                        amount = robot.power * 0.75
                    else:
                        resource = RESOURCE_MAPPING[action[5]]
                        amount = robot.cargo.state_dict()[resource] * 0.75
                
                # transfer amount = 0.95
                elif action[4] == 3:
                    if action[5] == 2:
                        amount = robot.power * 0.95
                    else:
                        resource = RESOURCE_MAPPING[action[5]]
                        amount = robot.cargo.state_dict()[resource] * 0.95
                else:
                    print("Invalid transfer amount action")
                
                # transfer amount
                crafted_action[3] = amount

            elif action[1] == 3:
                # action_type = PICKUP
                crafted_action[0] = 2
                # pickup resource is always POWER
                crafted_action[2] = 4
                # fix pickup amount to 25% of available cargo
                free_power_capacity = robot.battery_capacity - robot.power
                crafted_action[3] = free_power_capacity

            elif action[1] == 4:
                # action_type = DIG
                crafted_action[0] = 3

            elif action[1] == 5:
                # action_type = SELF-DESTRUCT
                crafted_action[0] = 4

            else:
                if action[1] != 0:
                    print("Unit action not implemented")

            # finally commit the action
            if crafted_action is not None:
                # N and repeat
                crafted_action[4] = 0
                crafted_action[5] = 1

                commited_actions[player][robot.unit_id] = [np.array(crafted_action, dtype=int)]

        return commited_actions

    # turn the raw json observations to actual model observations
    def obs_(self, obs, player):
        obs = obs[player]

        opponent = ""
        if player == "player_0":
            opponent = "player_1"
        else:
            opponent = "player_0"

        env_steps = obs["real_env_steps"] + (obs["board"]["factories_per_team"] * 2 + 1)
        is_day_ = np.tile(int(is_day(self.env_.env_cfg, env_steps)), (48,48))

        rubble = np.array(obs["board"]["rubble"], dtype=float) / 100
        ore = np.array(obs["board"]["ore"], dtype=float) 
        ice = np.array(obs["board"]["ice"], dtype=float)
        lichen = np.array(obs["board"]["lichen"], dtype=float) / 100
        is_resource = (ore.astype(bool) * 1) | (ice.astype(bool) * 1)

        light_units = np.zeros((48,48), dtype=float)
        heavy_units = np.zeros((48,48), dtype=float)
        unit_power = np.zeros((48,48), dtype=float)
        unit_ice = np.zeros((48,48), dtype=float)
        unit_ore = np.zeros((48,48), dtype=float)
        unit_on_factories = np.zeros((48,48), dtype=float)
        unit_has_ice = np.zeros((48,48), dtype=float)
        unit_has_ore = np.zeros((48,48), dtype=float)

        my_units = obs["units"][player]
        enemy_units = obs["units"][opponent]

        for unit_id in my_units:
            unit = my_units[unit_id]
            x, y = unit["pos"][0], unit["pos"][1]
            factory_there = self.env_.state.board.factory_occupancy_map[x,y]

            unit_ice[x, y] = unit["cargo"]["ice"] 
            unit_ore[x, y] = unit["cargo"]["ore"] 
            unit_has_ore[x, y] = 1 if unit["cargo"]["ore"] > 0 else 0
            unit_has_ice[x, y] = 1 if unit["cargo"]["ice"] > 0 else 0
            unit_on_factories[x, y] = 1 if (factory_there != -1 and factory_there in self.env_.state.teams[player].factory_strains) else 0
            
            if unit["unit_type"] == 'LIGHT':
                light_units[x, y] = 1
                unit_power[x, y] /= 150
                unit_ice[x, y] /= 100
                unit_ore[x, y] /= 100
            else:
                heavy_units[x, y] = 1
                unit_power[x, y] /= 3000
                unit_ice[x, y] /= 1000
                unit_ore[x, y] /= 1000

        for unit_id in enemy_units:
            unit = enemy_units[unit_id]
            x, y = unit["pos"][0], unit["pos"][1]
            factory_there = self.env_.state.board.factory_occupancy_map[x,y]

            unit_ice[x, y] = - unit["cargo"]["ice"] 
            unit_ore[x, y] = - unit["cargo"]["ore"] 
            unit_has_ore[x, y] = -1 if unit["cargo"]["ore"] > 0 else 0
            unit_has_ice[x, y] = -1 if unit["cargo"]["ice"] > 0 else 0
            unit_on_factories[x, y] = -1 if (factory_there != -1 and factory_there in self.env_.state.teams[opponent].factory_strains) else 0
            
            if unit["unit_type"] == 'LIGHT':
                light_units[x, y] = -1
                unit_power[x, y] /= 150
                unit_ice[x, y] /= 100
                unit_ore[x, y] /= 100
            else:
                heavy_units[x, y] = -1
                unit_power[x, y] /= 3000
                unit_ice[x, y] /= 1000
                unit_ore[x, y] /= 1000

        # get factories data
        factories = np.zeros((48,48), dtype=float)
        factory_ice = np.zeros((48,48), dtype=float)
        factory_ore = np.zeros((48,48), dtype=float)
        factory_water = np.zeros((48,48), dtype=float)
        factory_metal = np.zeros((48,48), dtype=float)
        factory_has_ice = np.zeros((48,48), dtype=float)
        factory_has_ore = np.zeros((48,48), dtype=float)
        factory_will_survive = np.zeros((48,48), dtype=float)

        my_factories = obs["factories"][player]
        enemy_factories = obs["factories"][opponent]

        for factory_id in my_factories:
            factory = my_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x-1:x+2, y-1:y+2] = 1
            factory_ice[x-1:x+2, y-1:y+2] = factory["cargo"]["ice"] / 500
            factory_water[x-1:x+2, y-1:y+2] = factory["cargo"]["water"] / 500
            factory_metal[x-1:x+2, y-1:y+2] = factory["cargo"]["metal"] / 500
            factory_ore[x-1:x+2, y-1:y+2] = factory["cargo"]["ore"] / 500
            factory_has_ice[x-1:x+2, y-1:y+2] = 1 if factory["cargo"]["ice"] > 0 else 0
            factory_has_ore[x-1:x+2, y-1:y+2] = 1 if factory["cargo"]["ore"] > 0 else 0
            factory_will_survive[x-1:x+2, y-1:y+2] = 1 if factory["cargo"]["water"] > 15 else 0
            

        for factory_id in enemy_factories:
            factory = enemy_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x-1:x+2, y-1:y+2] = -1
            factory_ice[x-1:x+2, y-1:y+2] = - factory["cargo"]["ice"] / 500
            factory_water[x-1:x+2, y-1:y+2] = - factory["cargo"]["water"] / 500
            factory_metal[x-1:x+2, y-1:y+2] = - factory["cargo"]["metal"] / 500
            factory_ore[x-1:x+2, y-1:y+2] = - factory["cargo"]["ore"] / 500

            factory_has_ice[x-1:x+2, y-1:y+2] = -1 if factory["cargo"]["ice"] > 0 else 0
            factory_has_ore[x-1:x+2, y-1:y+2] = -1 if factory["cargo"]["ore"] > 0 else 0
            factory_will_survive[x-1:x+2, y-1:y+2] = -1 if factory["cargo"]["water"] > 15 else 0


        # 30 channels of information!
        obs = np.stack([
            is_day_,
            rubble, 
            ore, 
            ice,
            lichen, 
            is_resource,
            light_units,
            heavy_units,
            unit_ice,
            unit_ore,
            unit_power,
            unit_on_factories,
            unit_has_ice,
            unit_has_ore,
            factories,
            factory_ice,
            factory_metal,
            factory_ore,
            factory_water,
            factory_has_ice,
            factory_has_ore,
            factory_will_survive
        ]).astype("float64")

        # transpose for channel last
        return obs.transpose(1,2,0)

    # turn the raw json observations to actual model observations (uses only 1s and 0s)
    def simple_obs_(self, obs, player):
        obs = obs[player]

        opponent = ""
        if player == "player_0":
            opponent = "player_1"
        else:
            opponent = "player_0"

        env_steps = obs["real_env_steps"] + (obs["board"]["factories_per_team"] * 2 + 1)
        is_day_ = np.tile(int(is_day(self.env_.env_cfg, env_steps)), (48,48))
        
        rubble = np.array(obs["board"]["rubble"], dtype=bool) * 1
        ore = np.array(obs["board"]["ore"], dtype=bool) * 1
        ice = np.array(obs["board"]["ice"], dtype=bool) * 1
        lichen = np.array(obs["board"]["lichen"], dtype=bool) * 1

        light_units = np.zeros((48,48), dtype=float)
        heavy_units = np.zeros((48,48), dtype=float)
        unit_high_power = np.zeros((48,48), dtype=float)
        unit_low_power = np.zeros((48,48), dtype=float)
        unit_ice = np.zeros((48,48), dtype=float)
        unit_ore = np.zeros((48,48), dtype=float)
        unit_water = np.zeros((48,48), dtype=float)
        unit_metal = np.zeros((48,48), dtype=float)
        unit_on_factories = np.zeros((48,48), dtype=float)

        my_units = obs["units"][player]
        enemy_units = obs["units"][opponent]

        for unit_id in my_units:
            unit = my_units[unit_id]
            x, y = unit["pos"][0], unit["pos"][1]
            factory_there = self.env_.state.board.factory_occupancy_map[x,y]
            
            if unit["unit_type"] == 'LIGHT':
                light_units[x][y] = 1
                unit_high_power[x][y] = 1 if unit["power"] >= (150 * 0.75) else 0
                unit_low_power[x][y] = 1 if unit["power"] <= (150 * 0.75) else 0
            else:
                heavy_units[x][y] = 1
                unit_high_power[x][y] = 1 if unit["power"] >= (3000 * 0.75) else 0
                unit_low_power[x][y] = 1 if unit["power"] <= (3000 * 0.75) else 0

            unit_ice[x][y] = 1 if unit["cargo"]["ice"] > 0 else 0
            unit_water[x][y] = 1 if unit["cargo"]["water"] > 0 else 0
            unit_metal[x][y] = 1 if unit["cargo"]["metal"] > 0 else 0
            unit_ore[x][y] = 1 if unit["cargo"]["ore"] > 0 else 0
            unit_on_factories[x][y] = 1 if (factory_there != -1 and factory_there in self.env_.state.teams[player].factory_strains) else 0

        for unit_id in enemy_units:
            unit = enemy_units[unit_id]
            x, y = unit["pos"][0], unit["pos"][1]
            factory_there = self.env_.state.board.factory_occupancy_map[x,y]

            if unit["unit_type"] == 'LIGHT':
                light_units[x][y] = -1
                unit_high_power[x][y] = -1 if unit["power"] >= (150 * 0.75) else 0
                unit_low_power[x][y] = -1 if unit["power"] <= (150 * 0.75) else 0
            else:
                heavy_units[x][y] = -1
                unit_high_power[x][y] = -1 if unit["power"] >= (3000 * 0.75) else 0
                unit_low_power[x][y] = -1 if unit["power"] <= (3000 * 0.75) else 0

                unit_ice[x][y] = -1 if unit["cargo"]["ice"] > 0 else 0
                unit_water[x][y] = -1 if unit["cargo"]["water"] > 0 else 0
                unit_metal[x][y] = -1 if unit["cargo"]["metal"] > 0 else 0
                unit_ore[x][y] = -1 if unit["cargo"]["ore"] > 0 else 0
                unit_on_factories[x][y] = -1 if (factory_there != -1 and factory_there in self.env_.state.teams[opponent].factory_strains) else 0


        # get factories data
        factories = np.zeros((48,48), dtype=float)
        factory_ice = np.zeros((48,48), dtype=float)
        factory_ore = np.zeros((48,48), dtype=float)
        factory_water = np.zeros((48,48), dtype=float)
        factory_metal = np.zeros((48,48), dtype=float)

        my_factories = obs["factories"][player]
        enemy_factories = obs["factories"][opponent]

        for factory_id in my_factories:
            factory = my_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x][y] = 1
            factory_ice[x][y] = 1 if factory["cargo"]["ice"] > 0 else 0
            factory_water[x][y] = 1 if factory["cargo"]["water"] > 10 else 0
            factory_metal[x][y] = 1 if factory["cargo"]["metal"] > 0 else 0
            factory_ore[x][y] = 1 if factory["cargo"]["ore"] > 0 else 0

        for factory_id in enemy_factories:
            factory = enemy_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x][y] = -1
            factory_ice[x][y] = -1 if factory["cargo"]["ice"] > 0 else 0
            factory_water[x][y] = -1 if factory["cargo"]["water"] > 10 else 0
            factory_metal[x][y] = -1 if factory["cargo"]["metal"] > 0 else 0
            factory_ore[x][y] = -1 if factory["cargo"]["ore"] > 0 else 0

        # 30 channels of information!
        obs = np.stack([
            is_day_,
            rubble, 
            ore, 
            ice,
            lichen, 
            light_units,
            heavy_units,
            unit_ice,
            unit_metal,
            unit_water,
            unit_ore,
            unit_high_power,
            unit_low_power,
            unit_on_factories,
            factories,
            factory_ice,
            factory_metal,
            factory_ore,
            factory_water,
        ]).astype("float64")

        # transpose for channel last
        return obs.transpose(1,2,0)

    # simulates an opponent decision by querying the opponent model
    # the opponent is not learning
    # follows the same logic as the learning agent in ppo_res_gridnet_multigpu.py
    def enemy_step(self):
        with torch.no_grad():
            obs = torch.Tensor(self.current_enemy_obs).unsqueeze(0).to(self.device)
            robot_action, factory_action, logproba, _, robot_invalid_action_masks, factory_invalid_action_masks = self.enemy_agent.get_action(obs, envs_=self, player="player_1")

            robot_real_action = torch.cat([
            torch.stack(
                [torch.arange(0, 48*48, device=self.device) for i in range(1)
            ]).unsqueeze(2), robot_action], 2)
            
            robot_real_action = robot_real_action.cpu().numpy()
            robot_valid_actions = robot_real_action[robot_invalid_action_masks[:,:,0].bool().cpu().numpy()]
            robot_valid_actions_counts = robot_invalid_action_masks[:,:,0].sum(1).long().cpu().numpy()
            robot_java_valid_actions = []
            robot_valid_action_idx = 0
            for env_idx, robot_valid_action_count in enumerate(robot_valid_actions_counts):
                robot_java_valid_action = []
                for c in range(robot_valid_action_count):
                    robot_java_valid_action += [JArray(JInt)(robot_valid_actions[robot_valid_action_idx])]
                    robot_valid_action_idx += 1
                robot_java_valid_actions += [JArray(JArray(JInt))(robot_java_valid_action)]
            robot_java_valid_actions = JArray(JArray(JArray(JInt)))(robot_java_valid_actions)

            factory_real_action = torch.cat([
                torch.stack(
                    [torch.arange(0, 48*48, device=self.device) for i in range(1)
            ]).unsqueeze(2), factory_action], 2)
            
            factory_real_action = factory_real_action.cpu().numpy()
            factory_valid_actions = factory_real_action[factory_invalid_action_masks[:,:,0].bool().cpu().numpy()]
            factory_valid_actions_counts = factory_invalid_action_masks[:,:,0].sum(1).long().cpu().numpy()
            factory_java_valid_actions = []
            factory_valid_action_idx = 0
            for env_idx, factory_valid_action_count in enumerate(factory_valid_actions_counts):
                factory_java_valid_action = []
                for c in range(factory_valid_action_count):
                    factory_java_valid_action += [JArray(JInt)(factory_valid_actions[factory_valid_action_idx])]
                    factory_valid_action_idx += 1
                factory_java_valid_actions += [JArray(JArray(JInt))(factory_java_valid_action)]
            factory_java_valid_actions = JArray(JArray(JArray(JInt)))(factory_java_valid_actions)
            
            robot_valid_actions = np.array(robot_java_valid_actions, dtype=object)
            factory_valid_actions = np.array([np.array(xi) for xi in factory_java_valid_actions], dtype=object)

            actions =  {
                "factories" : factory_valid_actions[0],
                "robots" : robot_valid_actions[0]
            }

        return actions

    # simulates the opponent actions using a bad heurisitc (if doesnt want to use self play)
    def enemy_heuristic_step(self):
        actions = dict()

        game_state = self.env_.state

        factories = game_state.factories["player_1"]
        factory_tiles, factory_units = [], []

        for unit_id, factory in factories.items():
            if factory.power >= self.env_.env_cfg.ROBOTS["HEAVY"].POWER_COST and factory.cargo.metal >= self.env_.env_cfg.ROBOTS["HEAVY"].METAL_COST:
                actions[unit_id] = 1
            if water_cost(factory, self.env_) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = 2
            factory_tiles += [np.array([factory.pos.x, factory.pos.y])]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units["player_1"]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - np.array([unit.pos.x, unit.pos.y])) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = np.mean((closest_factory_tile - np.array([unit.pos.x, unit.pos.y])) ** 2) == 0

                # previous ice mining code
                if unit.cargo.ice < 40:
                    ice_tile_distances = np.mean((ice_tile_locations - np.array([unit.pos.x, unit.pos.y])) ** 2, 1)
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(game_state) + action_queue_cost(unit, self.env_.env_cfg):
                            actions[unit_id] = [np.array([3, 0, 0, 0, 0, 1])]
                    else:
                        direction = direction_to(np.array([unit.pos.x, unit.pos.y]), closest_ice_tile)
                        move_cost = compute_move_cost(unit, game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + action_queue_cost(unit, self.env_.env_cfg):
                            actions[unit_id] = [np.array([0, direction, 0, 0, 0, 1])]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= 40:
                    direction = direction_to(np.array([unit.pos.x, unit.pos.y]), closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= action_queue_cost(unit, self.env_.env_cfg):
                            actions[unit_id] = [np.array([1, direction, 0, unit.cargo.ice, 0, 1])]
                    else:
                        move_cost = compute_move_cost(unit, game_state, direction)
                        if move_cost is not None and unit.power >= move_cost + action_queue_cost(unit, self.env_.env_cfg):
                            actions[unit_id] = [np.array([0, direction, 0, 0, 0, 1])]
        
        return {"player_1" : actions}

    # called only once at the begining of training
    def set_enemy_agent(self, agent):
        self.enemy_agent = agent
    
    # needed for self-play with TrueSkill only
    def set_enemy_idx(self, idx):
        self.enemy_idx = idx
    
    # change reward to sparse
    def set_sparse_reward(self):
        self.is_sparse_reward = True

    # called at each env.reset(). loads a new enemy from the pool
    def update_enemy_agent(self):
        try:
            self.enemy_idx = self.enemy_agent.load_checkpoint(self.PATH_AGENT_CHECKPOINTS)
            self.enemy_agent.freeze_params()
        except:
            print(self.PATH_AGENT_CHECKPOINTS)