
import gym
import numpy as np
import luxai_s2.env
import torch
import math
import copy

#from luxai_s2.state import StatsStateDict
from IPython import embed

from jpype.types import JArray, JInt
from gym import spaces
from luxai_s2.env import LuxAI_S2

from jux.env import JuxEnv
from jux.config import JuxBufferConfig, EnvConfig
import jax.numpy as jnp
import rich

from .env_utils import *

"""
This custom env handles self-play and training of the bidder and placer.
1) Handles enemy sampling from a pool of checkpoints and its actions at each turn (so that from outside, env.step() only requires the agent action, needed for VectorizedEnvironments)
2) Every reset() call makes it to the normal game phase directly (queries the bidder and placer before resetting and trains them with a DQN-like algo and replay buffer)
"""

class CustomJuxEnv(gym.Env):
    def __init__(
            self, 
            self_play = False, 
            sparse_reward = True,
            simple_obs = True,
            seed = 0,
            env_cfg = None, 
            device = "cuda:0", 
            PATH_AGENT_CHECKPOINTS = "checkpoints"
        ):


        # the true env
        self.env_ = JuxEnv(
            env_cfg=EnvConfig(),
            buf_cfg=JuxBufferConfig(MAX_N_UNITS=200),
        )

        self.seed = seed

        self.self_play = self_play
        self.is_sparse_reward = sparse_reward
        self.is_survival_reward = True
        
        self.simple_obs = simple_obs

        self.prev_step_metrics = None
        self.enemy_agent = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.device = "cpu"
        self.PATH_AGENT_CHECKPOINTS = PATH_AGENT_CHECKPOINTS
        
        # observation space
        if self.simple_obs:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(48, 48, 19), dtype=np.float64)
        else:
            self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(48, 48, 18), dtype=np.float64)

        # ROBOTS
        # dim 0: position in map -- LENGTH 48 * 48 (e.g. pos (3,2) is (3 + 2*48) )
        # dim 1: action type [NOOP, move, transfer, pickup, dig, self-destruct] -- LENGTH 6
        # dim 2: move direction [up, right, down, left] -- LENGTH 4
        # dim 3: transfer direction [center, up, right, down, left] -- LENGTH 5
        # dim 4: transfer amount [25%, 50%, 75%, 95%] -- LENGTH 4
        # dim 5: transfer material [power, ore, metal, ice , water] --LENGTH 5
        # dim 6: pickup material [power, ore, metal, ice , water] --LENGTH 5
        
        # TODO
        # dim 9: recycle [0,1] -- LENGTH 2
        # dim 10: N [1,3,6] (action repeat parameter) -- LENGTH 3
        # TODO: when recycled, we also need to set the repeat value (right now we are recycling with repeat=1 which is a bit useless)

        # FACTORIES
        # dim 0: position in map -- LENGTH 48 * 48 (e.g. pos (3,2) is (3 + 2*48) )
        # dim 1: factory action [NOOP, build light robot, build heavy robot, grow lichen] -- LENGTH 4
        self.action_space = gym.spaces.Dict({
            'robots': gym.spaces.MultiDiscrete(
                [(48 * 48) - 1, 6, 4, 5, 4, 5, 5]
            ),
            'factories': gym.spaces.MultiDiscrete(
                [(48 * 48) - 1, 4]
            ),
        })

        # reward definition
        self.num_factories = 0
        self.num_units = 0

        # keeps updated versions of the player and enemy observations (preprocessed observations)
        self.current_player_obs = None
        self.current_enemy_obs = None
        # keep the current game observation
        self.current_state = None

    # TODO: figure out raw model output action shape
    def step(self, action):
        # turn the raw player model outputs to game actions
        player_action = self.act_(action, "player_0")  # returs {"player_0" : actions}

        if self.self_play:
            # get enemy action (raw output model)
            enemy_action = self.enemy_step()
            # turn the raw enemy model outputs to game actions
            enemy_action = self.act_(enemy_action, "player_1") # returs {"player_1" : actions}

        if self.self_play is False:
            enemy_action = self.enemy_heuristic_step()

        actions = {**player_action, **enemy_action} # final dict of actions from both players to send to the game

        # step actions to true env
        observations, reward, done, info = self.env_.step(actions)

        if reward["player_0"] == -1000:
            if self.is_sparse_reward:
                reward_now = -1
            else:
                reward_now = 0

            info["player_0"]["result"] = -1
        
        elif reward["player_1"] == -1000:
            if self.is_sparse_reward:
                reward_now = 1
            else:
                reward_now = 0

            info["player_0"]["result"] = 1
        
        else:
            if self.is_sparse_reward:
                reward_now = 0
            else:
                stats: StatsStateDict = self.env_.state.stats["player_0"]
                #info = dict()
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

                # pickup water and metal (-)
                metrics["pickup_water"] = stats["pickup"]["water"]
                metrics["pickup_metal"] = stats["pickup"]["metal"]

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
                    # survival reward only motivates getting ice, ore and creating robots
                    if self.is_survival_reward:
                        ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
                        water_produced_this_step = (metrics["water_produced"] - self.prev_step_metrics["water_produced"])
                        reward_now += (ice_dug_this_step / 100) + water_produced_this_step

                        ore_dug_this_step = metrics["ore_dug"] - self.prev_step_metrics["ore_dug"]
                        metal_produced_this_step = (metrics["metal_produced"] - self.prev_step_metrics["metal_produced"])
                        reward_now += ((ore_dug_this_step / 100) + metal_produced_this_step) / 2

                        new_pickedup_water = metrics["pickup_water"] - self.prev_step_metrics["pickup_water"]
                        reward_now -= new_pickedup_water / 100
                        new_pickedup_metal = metrics["pickup_metal"] - self.prev_step_metrics["pickup_metal"]
                        reward_now -= new_pickedup_metal / 100
                        new_pickedup_power = metrics["pickup_power"] - self.prev_step_metrics["pickup_power"]
                        reward_now += new_pickedup_power / 300
                        new_consumed_water = metrics["consumed_water"] - self.prev_step_metrics["consumed_water"]
                        reward_now -= new_consumed_water / 10

                        new_lights = metrics["count_lights"] - self.prev_step_metrics["count_lights"]
                        reward_now += (new_lights / 100)
                        new_heavies = metrics["count_heavies"] - self.prev_step_metrics["count_heavies"]
                        reward_now += (new_heavies / 100)
                        destroyed_lights = metrics["destroyed_lights"] - self.prev_step_metrics["destroyed_lights"]
                        reward_now -= (destroyed_lights / 100)
                        destroyed_heavies = metrics["destroyed_heavies"] - self.prev_step_metrics["destroyed_heavies"]
                        reward_now -= (destroyed_heavies * 2) / 100

                    else:
                        ice_dug_this_step = metrics["ice_dug"] - self.prev_step_metrics["ice_dug"]
                        water_produced_this_step = (metrics["water_produced"] - self.prev_step_metrics["water_produced"])
                        reward_now += (ice_dug_this_step / 100) + water_produced_this_step

                        ore_dug_this_step = metrics["ore_dug"] - self.prev_step_metrics["ore_dug"]
                        metal_produced_this_step = (metrics["metal_produced"] - self.prev_step_metrics["metal_produced"])
                        reward_now += ((ore_dug_this_step / 100) + metal_produced_this_step) / 2

                        new_lichen = metrics["lichen"] - self.prev_step_metrics["lichen"]
                        reward_now += new_lichen / 10

                        new_pickedup_water = metrics["pickup_water"] - self.prev_step_metrics["pickup_water"]
                        reward_now -= new_pickedup_water / 100
                        
                        new_pickedup_metal = metrics["pickup_metal"] - self.prev_step_metrics["pickup_metal"]
                        reward_now -= new_pickedup_metal / 100

                        new_pickedup_power = metrics["pickup_power"] - self.prev_step_metrics["pickup_power"]
                        reward_now += new_pickedup_power / 300

                        new_consumed_water = metrics["consumed_water"] - self.prev_step_metrics["consumed_water"]
                        reward_now -= new_consumed_water / 10

                        new_lights = metrics["count_lights"] - self.prev_step_metrics["count_lights"]
                        reward_now += new_lights / 100

                        new_heavies = metrics["count_heavies"] - self.prev_step_metrics["count_heavies"]
                        reward_now += (new_heavies * 2) / 100

                        destroyed_lights = metrics["destroyed_lights"] - self.prev_step_metrics["destroyed_lights"]
                        reward_now -= destroyed_lights / 100

                        destroyed_heavies = metrics["destroyed_heavies"] - self.prev_step_metrics["destroyed_heavies"]
                        reward_now -= (destroyed_heavies * 2) / 100

                        destroyed_factories = metrics["destroyed_factories"] - self.prev_step_metrics["destroyed_factories"]
                        reward_now -= destroyed_factories * 10

                        power_used = metrics["power_used"] - self.prev_step_metrics["power_used"]
                        reward_now -= power_used / 500

                self.prev_step_metrics = copy.deepcopy(metrics)

        done =  done["player_0"]

        # important: using sparse reward
        # important: that both agents get -1000 at the same time happens more often than expected when they are random. 
        # how to deal with this?

        # keep the current game state updated
        self.current_state = observations
    
        # update the current observations for both players and preprocess them into model input format
        self.current_player_obs = self.simple_obs_(observations, "player_0") if self.simple_obs == True else self.obs_(observations, "player_0")
        self.current_enemy_obs = self.simple_obs_(observations, "player_1")  if self.simple_obs == True else self.obs_(observations, "player_1")

        # important: only return from the point of view of player! enemy is handled as part of the environment
        return self.current_player_obs, reward_now, done, info["player_0"]



    def placement_heuristic_pro(self, observation, agent):
        embed()
        ice = observation[agent]["board"]["ice"]
        ore = observation[agent]["board"]["ore"]
        rubble = observation[agent]["board"]["rubble"]
        
        ice_distances = [manhattan_dist_to_nth_closest(ice, i) for i in range(1,5)]
        ore_distances = [manhattan_dist_to_nth_closest(ore, i) for i in range(1,5)]
        
        ICE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25]) 
        weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

        ORE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
        weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

        ICE_PREFERENCE = 5 # if you want to make ore more important, change to 0.3 for example

        combined_resource_score = (weigthed_ice_dist * ICE_PREFERENCE + weigthed_ore_dist)
        combined_resource_score = (np.max(combined_resource_score) - combined_resource_score) * observation[agent]["board"]["valid_spawns_mask"]

        """
        low_rubble = (rubble<25)
        low_rubble_scores = np.zeros_like(low_rubble, dtype=float)

        for i in range(low_rubble.shape[0]):
            for j in range(low_rubble.shape[1]):
                low_rubble_scores[i,j] = count_region_cells(low_rubble, (i,j), min_dist=0, max_dist=8, exponent=0.9)

        overall_score = (low_rubble_scores*2 + combined_resource_score ) * observation[agent]["board"]["valid_spawns_mask"]
        """
        overall_score = (combined_resource_score) * observation[agent]["board"]["valid_spawns_mask"]

        best_loc = np.argmax(overall_score)
        
        return np.unravel_index(best_loc, (48, 48))


    def placement_heuristic(self, observations, agent):
        area = 47
        # Used to store the values computed by the heuristic of the cells 
        values_array = np.zeros((48,48))
        resources_array = observations[agent]["board"]["ice"] #+ observations[agent]["board"]["ore"]
        # 2d locations of the resources
        resources_location = np.array(list(zip(*np.where(resources_array == 1))))
        for i in resources_location:
            for j in range(area):
                values_array[max(0, i[0]-(area-j)):min(47, i[0]+(area-j)), max(0, i[1]-(area-j)):min(47, i[1]+(area-j))] += (1/(area-j))
        valid_spawns = observations[agent]["board"]["valid_spawns_mask"]
        valid_grid = values_array * valid_spawns
        # Flattened index of the valid cell with the highest value
        spawn_loc = np.argmax(valid_grid)
        # 2d index
        spawn_index = np.unravel_index(spawn_loc, (48,48))
        
        return spawn_index

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

    def reset(self, load_new_enemy=True):
        # we now sample a new opponent at each game
        if self.self_play and load_new_enemy==True:
            self.update_enemy_agent()
        
        self.current_state = self.env_.reset(self.seed)
        self.num_factories = 0
        self.num_units = 0
        self.prev_step_metrics = None

        # TODO: update the current observations for both players and preprocess them into BIDDER input format
        # TODO: get action from bidder model (preprocess into placer format)

        #Total resources that are available to bid
        resources_amount = self.current_state.board.factories_per_team.item()*150
        
        # TODO: use a heurisitc
        #bids = self.bidding_heuristic(resources_amount)
        bids = [0,0]
        factions = [0,0]

        bid_action = jnp.array(np.hstack([bids[0], bids[1]]), dtype=jnp.int32)
        faction_action = jnp.array(np.hstack([factions[0], factions[1]]), dtype=jnp.int8)

        # step into factory placement phase
        self.current_state, (observations, _, _, _) = self.env_.step_bid(self.current_state, bid_action, faction_action)

        """
        factory_amount = {"player_0" : 0, "player_1" : 0}
        for player in factory_amount.keys():
            factory_amount[player] = observations["player_0"]["teams"][player]["water"]/observations["player_0"]["teams"][player]["factories_to_place"]
        """

        # handle all the factory placement phase
        while self.current_state.real_env_steps.item() < 0:
            spawn_action = jnp.zeros([2,2], dtype=jnp.int8)
            water_action = jnp.zeros([2,2], dtype=jnp.int32)
            metal_action = jnp.zeros([2,2], dtype=jnp.int32)

            
            for agent in ["player_0", "player_1"]:
                embed()
                if my_turn_to_place_factory(observations[agent].place_first, self.current_state.env_steps.item()):
                    # TODO: get action from placer model
                    spawn_loc = self.placement_heuristic_pro(observations, agent)
                    agent_idx = int(agent.split("_")[1])
                    
                    spawn_action[:, agent_idx, :] = jnp.array(spawn_loc, dtype=jnp.int8)
                    water_action[:, agent_idx] = jnp.array(150, dtype=jnp.int8)
                    metal_action[:, agent_idx] = jnp.array(150, dtype=jnp.int8)
                
            self.current_state, (observations, rewards, dones, infos) = self.env_.step_factory_placement(self.current_state, spawn_action, water_action, metal_action)

        # start total_water and total_metal used in to compute rewards
        factories_water = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["water"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
        self.total_water = factories_water
        factories_metal = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["metal"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
        self.total_metal = factories_metal
        self.num_factories = observations["player_0"]["board"]["factories_per_team"]

        # finally reset into the normal game
        self.current_player_obs = self.simple_obs_(observations, "player_0") if self.simple_obs == True else self.obs_(observations, "player_0")
        self.current_enemy_obs = self.simple_obs_(observations, "player_1") if self.simple_obs == True else self.obs_(observations, "player_1")
        
        # important: return only from players perspective! Enemy is handled as part of the environment
        return self.current_player_obs

    def render(self, mode):
        return self.env_.render(mode)

    def act_(self, action, player):
        """
        Transform the raw output actions into game actions that can be passed to the real env.
        """
        factory_actions = action["factories"]
        robot_actions = action["robots"]

        # we will fill this with the game actions
        commited_actions = {player : {}}

        game_state = self.env_.state

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

        for action in robot_actions:
            x = action[0] // 48
            y = action[0] % 48
            pos = Position(np.array([x,y]))

            robot = game_state.board.get_units_at(pos)[0]

            crafted_action = np.zeros(6)

            # IMPORTANT: see \luxai_s2\spaces\act_space.py

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
                crafted_action[2] = action[5]

                # transfer amount = 0.25
                if action[4] == 0:
                    if action[5] == 4:
                        amount = robot.power * 0.25
                    else:
                        resource = RESOURCE_MAPPING[action[5]]
                        amount = robot.cargo.state_dict()[resource] * 0.25
                
                # transfer amount = 0.5
                elif action[4] == 1:
                    if action[5] == 4:
                        amount = robot.power * 0.5
                    else:
                        resource = RESOURCE_MAPPING[action[5]]
                        amount = robot.cargo.state_dict()[resource] * 0.5
                
                # transfer amount = 0.75
                elif action[4] == 2:
                    if action[5] == 4:
                        amount = robot.power * 0.75
                    else:
                        resource = RESOURCE_MAPPING[action[5]]
                        amount = robot.cargo.state_dict()[resource] * 0.75
                
                # transfer amount = 0.95
                elif action[4] == 3:
                    if action[5] == 4:
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
                # pickup resource
                crafted_action[2] = action[6]
                # fix pickup amount to 25% of available cargo
                free_power_capacity = robot.battery_capacity - robot.power
                power_amount = free_power_capacity * 0.25
                
                current_cargo = robot.cargo.ice + robot.cargo.water +robot.cargo.metal + robot.cargo.ore 
                free_resource_capacity = robot.cargo_space - current_cargo
                resource_amount = free_resource_capacity * 0.25

                if action[6] == 4:
                    amount = power_amount
                else:
                    resource = RESOURCE_MAPPING[action[6]]
                    amount = resource_amount

                crafted_action[3] = amount

                """
                # pickup amount = 0.25
                if action[6] == 0:
                    if action[7] == 4:
                        free_power_capacity = robot.battery_capacity - robot.power
                        amount = free_power_capacity * 0.25
                    else:
                        current_cargo = robot.cargo.ice + robot.cargo.water +robot.cargo.metal + robot.cargo.ore 
                        free_resource_capacity = robot.cargo_space - current_cargo
                        resource = RESOURCE_MAPPING[action[7]]
                        amount = free_resource_capacity * 0.25
                
                # pickup amount = 0.5
                elif action[6] == 1:
                    if action[7] == 4:
                        free_power_capacity = robot.battery_capacity - robot.power
                        amount = free_power_capacity * 0.5
                    else:
                        current_cargo = robot.cargo.ice + robot.cargo.water +robot.cargo.metal + robot.cargo.ore 
                        free_resource_capacity = robot.cargo_space - current_cargo
                        resource = RESOURCE_MAPPING[action[7]]
                        amount = free_resource_capacity * 0.5

                elif action[6] == 2:
                    if action[7] == 4:
                        free_power_capacity = robot.battery_capacity - robot.power
                        amount = free_power_capacity * 0.75
                    else:
                        current_cargo = robot.cargo.ice + robot.cargo.water +robot.cargo.metal + robot.cargo.ore 
                        free_resource_capacity = robot.cargo_space - current_cargo
                        resource = RESOURCE_MAPPING[action[7]]
                        amount = free_resource_capacity * 0.75
                
                elif action[6] == 3:
                    if action[7] == 4:
                        free_power_capacity = robot.battery_capacity - robot.power
                        amount = free_power_capacity * 0.95
                    else:
                        current_cargo = robot.cargo.ice + robot.cargo.water +robot.cargo.metal + robot.cargo.ore 
                        free_resource_capacity = robot.cargo_space - current_cargo
                        resource = RESOURCE_MAPPING[action[7]]
                        amount = free_resource_capacity * 0.95
                else:
                    print("Invalid pickup amount action")

                crafted_action[3] = amount
                """

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
        ore = np.array(obs["board"]["ore"], dtype=float) / 10
        ice = np.array(obs["board"]["ice"], dtype=float) / 10
        lichen = np.array(obs["board"]["lichen"], dtype=float) / 100

        light_units = np.zeros((48,48), dtype=float)
        heavy_units = np.zeros((48,48), dtype=float)
        unit_power = np.zeros((48,48), dtype=float)
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

            unit_ice[x][y] = unit["cargo"]["ice"] 
            unit_water[x][y] = unit["cargo"]["water"] 
            unit_metal[x][y] = unit["cargo"]["metal"]
            unit_ore[x][y] = unit["cargo"]["ore"] 
            unit_on_factories[x][y] = 1 if (factory_there != -1 and factory_there in self.env_.state.teams[player].factory_strains) else 0
            
            if unit["unit_type"] == 'LIGHT':
                light_units[x][y] = 1
                unit_power[x][y] /= 150
                unit_ice[x][y] /= 100
                unit_water[x][y] /= 100
                unit_metal[x][y] /= 100
                unit_ore[x][y] /= 100
            else:
                heavy_units[x][y] = 1
                unit_power[x][y] /= 3000
                unit_ice[x][y] /= 1000
                unit_water[x][y] /= 1000
                unit_metal[x][y] /= 1000
                unit_ore[x][y] /= 1000

        for unit_id in enemy_units:
            unit = enemy_units[unit_id]
            x, y = unit["pos"][0], unit["pos"][1]
            factory_there = self.env_.state.board.factory_occupancy_map[x,y]

            unit_ice[x][y] = - unit["cargo"]["ice"] 
            unit_water[x][y] = - unit["cargo"]["water"] 
            unit_metal[x][y] = - unit["cargo"]["metal"]
            unit_ore[x][y] = - unit["cargo"]["ore"] 
            unit_on_factories[x][y] = -1 if (factory_there != -1 and factory_there in self.env_.state.teams[opponent].factory_strains) else 0
            
            if unit["unit_type"] == 'LIGHT':
                light_units[x][y] = -1
                unit_power[x][y] /= 150
                unit_ice[x][y] /= 100
                unit_water[x][y] /= 100
                unit_metal[x][y] /= 100
                unit_ore[x][y] /= 100
            else:
                heavy_units[x][y] = -1
                unit_power[x][y] /= 3000
                unit_ice[x][y] /= 1000
                unit_water[x][y] /= 1000
                unit_metal[x][y] /= 1000
                unit_ore[x][y] /= 1000

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
            factory_ice[x][y] = factory["cargo"]["ice"] / 100
            factory_water[x][y] = factory["cargo"]["water"] / 100
            factory_metal[x][y] = factory["cargo"]["metal"] / 100
            factory_ore[x][y] = factory["cargo"]["ore"] / 100

        for factory_id in enemy_factories:
            factory = enemy_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x][y] = -1
            factory_ice[x][y] = - factory["cargo"]["ice"] / 100
            factory_water[x][y] = - factory["cargo"]["water"] / 100
            factory_metal[x][y] = - factory["cargo"]["metal"] / 100
            factory_ore[x][y] = - factory["cargo"]["ore"] / 100

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
            unit_power,
            unit_on_factories,
            factories,
            factory_ice,
            factory_metal,
            factory_ore,
            factory_water,
        ]).astype("float64")

        # transpose for channel last
        return obs.transpose(1,2,0)


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

    def set_enemy_agent(self, agent):
        self.enemy_agent = agent
    
    def set_sparse_reward(self):
        self.is_sparse_reward = True

    def update_enemy_agent(self):
        try:
            self.enemy_agent.load_checkpoint(self.PATH_AGENT_CHECKPOINTS)
            self.enemy_agent.freeze_params()
        except:
            print(self.PATH_AGENT_CHECKPOINTS)


def make_env(seed):
    def thunk():
        env = CustomLuxEnv()
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
