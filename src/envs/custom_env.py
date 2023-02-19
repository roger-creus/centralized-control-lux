
import gym
import numpy as np
import luxai_s2.env
import torch
import math


from IPython import embed

from jpype.types import JArray, JInt
from gym import spaces
from luxai_s2.env import LuxAI_S2
from luxai_s2.utils.utils import my_turn_to_place_factory, is_day
from luxai_s2.map.position import Position

from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree


from luxai_s2.spaces.act_space import ActionsQueue

"""
SOME UTILITIES FOR THE HEURISTIC ENEMY
"""
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


"""
This custom env handles self-play and training of the bidder and placer.
1) Handles enemy sampling from a pool of checkpoints and its actions at each turn (so that from outside, env.step() only requires the agent action, needed for VectorizedEnvironments)
2) Every reset() call makes it to the normal game phase directly (queries the bidder and placer before resetting and trains them with a DQN-like algo and replay buffer)
"""

class CustomLuxEnv(gym.Env):
    def __init__(self, self_play=False, env_cfg = None, device = "cuda:0", PATH_AGENT_CHECKPOINTS = "agent_checkpoints"):
        # the tru env
        self.env_ = LuxAI_S2(env_cfg, verbose=False)
        
        self.self_play = self_play

        self.device = device
        self.PATH_AGENT_CHECKPOINTS = "/home/mila/r/roger.creus-castanyer/lux-ai-rl/src/checkpoints_dense"
        
        # observation space
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(48, 48, 24), dtype=np.float64)
        
        # keep running channel-wise statistics for normalisation each feature map independently

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
                [(48 * 48), 6, 4, 5, 4, 5, 5]
            ),
            'factories': gym.spaces.MultiDiscrete(
                [(48 * 48), 4]
            ),
        })

        # enemy agent is PyTorch model
        self.enemy_agent = None
        # this env handles the training of these 2
        self.bidder = None
        self.placer = None
        
        # reward definition
        self.is_sparse_reward = False
        self.prev_lichen = 0
        self.num_factories = 0
        self.num_units = 0
        self.total_water = 0
        self.total_metal = 0

        self.test_step_count = 0

        # keeps updated versions of the player and enemy observations (preprocessed observations)
        self.current_player_obs = None
        self.current_enemy_obs = None
        # keep the current game observation
        self.current_state = None

        self.max_lichen_and_rubble = 100

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
                reward_now = -10

            info["player_0"]["result"] = -1
        
        elif reward["player_1"] == -1000:
            if self.is_sparse_reward:
                reward_now = 1
            else:
                reward_now = 10
            info["player_0"]["result"] = 1
        
        else:
            if self.is_sparse_reward:
                reward_now = 0
            else:
                # dim 0: lichen reward (+)
                # dim 1: factories with no water (-)
                # dim 2: factories that died (-)
                # dim 3: units with no power (-)
                # dim 4: units that died (-)
                # dim 5: total resource gain (+/-)
                rewards = np.zeros(6)
                weights = np.array([0.25, 1, 5, 0.01, 0.1, 0.01])

                lichen_reward = (reward["player_0"] - self.prev_lichen) / 1000
                self.prev_lichen = reward["player_0"]
                rewards[0] = lichen_reward

                # Water threshold penalty
                water_threshold = 15
                num_factories = len(observations["player_0"]["factories"]["player_0"].keys())
                factories_water_shortage = -sum([water_threshold-observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["water"] for factory in observations["player_0"]["factories"]["player_0"].keys() if observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["water"] < water_threshold])
                rewards[1] = factories_water_shortage / (water_threshold-1)

                # Losing factory penalty
                if num_factories - self.num_factories < 0:
                    factories_lost = num_factories - self.num_factories
                    rewards[2] = factories_lost #/self.num_factories
                self.num_factories = num_factories

                # Power threshold penalty
                power_threshold_heavy = 450
                power_threshold_light = 22.5
                num_units = len(observations["player_0"]["units"]["player_0"].keys())
                if num_units != 0:
                    heavy_power = -len([observations["player_0"]["units"]["player_0"][f"{unit}"] for unit in observations["player_0"]["units"]["player_0"].keys() if observations["player_0"]["units"]["player_0"][f"{unit}"]["unit_type"]=="HEAVY" and observations["player_0"]["units"]["player_0"][f"{unit}"]["power"] < power_threshold_heavy])
                    light_power = -len([observations["player_0"]["units"]["player_0"][f"{unit}"] for unit in observations["player_0"]["units"]["player_0"].keys() if observations["player_0"]["units"]["player_0"][f"{unit}"]["unit_type"]=="LIGHT" and observations["player_0"]["units"]["player_0"][f"{unit}"]["power"] < power_threshold_light])
                    # unit_power_shortage = -sum([power_threshold-observations["player_0"]["units"]["player_0"][f"{unit}"]["power"] for unit in observations["player_0"]["units"]["player_0"].keys() if observations["player_0"]["units"]["player_0"][f"{unit}"]["power"] < power_threshold])
                    units_power = heavy_power + light_power
                    rewards[3] = units_power #(num_units)

                # Robot loss penalty
                if num_units - self.num_units < 0:
                    units_lost = num_units - self.num_units
                    rewards[4] = units_lost #/self.num_units
                self.num_units = num_units

                # Resources increase reward
                # Water
                lichen_water_cost = math.ceil(reward["player_0"]/10)
                
                factories_water = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["water"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
                factories_ice = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["ice"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
                factories_water_total = factories_water + math.ceil(factories_ice/4) + num_factories
                
                # Total ice in cargo for each unit (Divided by a constant so that cargo in robots is less important than cargo in factories)
                if num_units != 0:
                    unit_ice = sum([observations["player_0"]["units"]["player_0"][f"{unit}"]["cargo"]["ice"] for unit in observations["player_0"]["units"]["player_0"].keys()])
                else:
                    unit_ice = 0

                total_water = lichen_water_cost + factories_water_total + math.ceil(unit_ice/4)/5
                
                # Metal
                factories_metal = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["metal"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
                factories_ore = sum([observations["player_0"]["factories"]["player_0"][f"{factory}"]["cargo"]["ore"] for factory in observations["player_0"]["factories"]["player_0"].keys()])
                factories_metal_total = factories_metal + math.ceil(factories_ore/5)
                # Metal Value of robots
                if num_units != 0:
                    heavy_metal = len([observations["player_0"]["units"]["player_0"][f"{unit}"] for unit in observations["player_0"]["units"]["player_0"].keys() if observations["player_0"]["units"]["player_0"][f"{unit}"]["unit_type"]=="HEAVY"])*100
                    light_metal = len([observations["player_0"]["units"]["player_0"][f"{unit}"] for unit in observations["player_0"]["units"]["player_0"].keys() if observations["player_0"]["units"]["player_0"][f"{unit}"]["unit_type"]=="LIGHT"])*10
                    robots_metal_total = heavy_metal + light_metal
                else:
                    robots_metal_total = 0
                
                # Total ore in cargo for each unit (Divided by a constant so that cargo in robots is less important than cargo in factories)
                if num_units != 0:
                    unit_ore = sum([observations["player_0"]["units"]["player_0"][f"{unit}"]["cargo"]["ore"] for unit in observations["player_0"]["units"]["player_0"].keys()])
                else:
                    unit_ore = 0

                total_metal = factories_metal_total + robots_metal_total + math.ceil(unit_ore/5)/5

                # if this is the first step, initialize the total metal and total water of the player.
                if observations["player_0"]["real_env_steps"] == 1:
                    self.total_metal = total_metal
                    self.total_water = total_water

                metal_change = total_metal - self.total_metal
                water_change = total_water - self.total_water
                self.total_metal = total_metal
                self.total_water = total_water

                # Resources weights, we might want to weigh them differently. (Must sum to 1).
                resources_weights = np.array([0.5, 0.5])
                
                #print("Total Rewards: ", rewards, "    Total Metal:", total_metal, "Unit Ore: ", math.ceil(unit_ore/5)/5, "Total Water: ", total_water, "Unit Ice", math.ceil(unit_ice/4)/5)

                resources = np.array([metal_change, water_change])
                rewards[5] = np.sum(resources * resources_weights)

                reward_now = weights * rewards
                reward_now = np.sum(reward_now)

        done =  done["player_0"]

        # important: using sparse reward
        # important: that both agents get -1000 at the same time happens more often than expected when they are random. 
        # how to deal with this?

        # keep the current game state updated
        self.current_state = observations
    
        # update the current observations for both players and preprocess them into model input format
        self.current_player_obs = self.obs_(observations, "player_0")
        self.current_enemy_obs = self.obs_(observations, "player_1")

        # important: only return from the point of view of player! enemy is handled as part of the environment
        return self.current_player_obs, reward_now, done, info["player_0"]


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

        ICE_PREFERENCE = 3 # if you want to make ore more important, change to 0.3 for example

        combined_resource_score = (weigthed_ice_dist * ICE_PREFERENCE + weigthed_ore_dist)
        combined_resource_score = (np.max(combined_resource_score) - combined_resource_score) * observation[agent]["board"]["valid_spawns_mask"]

        low_rubble = (rubble<25)
        low_rubble_scores = np.zeros_like(low_rubble, dtype=float)

        for i in range(low_rubble.shape[0]):
            for j in range(low_rubble.shape[1]):
                low_rubble_scores[i,j] = count_region_cells(low_rubble, (i,j), min_dist=0, max_dist=8, exponent=0.9)

        overall_score = (low_rubble_scores*2 + combined_resource_score ) * observation[agent]["board"]["valid_spawns_mask"]

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

    def reset(self):
        # we now sample a new opponent at each game
        self.update_enemy_agent()
        
        observations = self.env_.reset()
        self.prev_lichen = 0
        self.num_factories = 0
        self.num_units = 0
        self.total_water = 0
        self.total_metal = 0

        # TODO: handle enemy sampling

        # TODO: update the current observations for both players and preprocess them into BIDDER input format
        
        # self.current_player_obs = self.obs_bid_phase_(observations, "player_0")
        #self.current_enemy_obs = self.obs_bid_phase_(observations, "player_1")

        # TODO: get action from bidder model (preprocess into placer format)

        #Total resources that are available to bid
        resources_amount = observations["player_0"]["board"]["factories_per_team"]*150
        
        bids = self.bidding_heuristic(resources_amount)
        
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

        # keep the current game state updated
        self.current_state = observations

        # finally reset into the normal game
        self.current_player_obs = self.obs_(observations, "player_0")
        self.current_enemy_obs = self.obs_(observations, "player_1")

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
        # TODO: add feature maps indicating current and future action queues
        # TODO: add feature maps with forward sim to show how the future will be
        # TODO: efficient normalization of features

        # obs is a dict with keys "player_0" and "player_1". then  obs["player_0"]["units"] is again a dict with keys "player_0" and "player_1". weird
        obs = obs[player]

        opponent = ""
        if player == "player_0":
            opponent = "player_1"
        else:
            opponent = "player_0"

        env_steps = obs["real_env_steps"] + (obs["board"]["factories_per_team"] * 2 + 1)
        # get game stats
        turn = np.tile(env_steps, (48,48)) #/ 999.       # max turns is 1000
        turn_in_cycle = np.tile((env_steps % 50), (48,48)) #/ 49.
        is_day_ = np.tile(int(is_day(self.env_.env_cfg, env_steps)), (48,48))

        # get map data
        rubble = np.array(obs["board"]["rubble"], dtype=float)
        ore = np.array(obs["board"]["ore"], dtype=float)
        ice = np.array(obs["board"]["ice"], dtype=float)
        lichen = np.array(obs["board"]["lichen"], dtype=float)
        #lichen_strains = np.array(obs["board"]["lichen_strains"], dtype=float)
        valid_spawns = np.array(obs["board"]["valid_spawns_mask"], dtype=float)

        #    normalization between 0 and 1 (we will try to keep negative values to indicate only enemy)
        #rubble = rubble / self.max_lichen_and_rubble
        #lichen = lichen / self.max_lichen_and_rubble
        #ore = ore / self.max_lichen_and_rubble
        #ice = ice / self.max_lichen_and_rubble
        #lichen_strains = lichen_strains / 10    # important: 10 is a heuristic

        # get units data
        light_units = np.zeros((48,48), dtype=float)
        heavy_units = np.zeros((48,48), dtype=float)
        unit_power = np.zeros((48,48), dtype=float)
        unit_ice = np.zeros((48,48), dtype=float)
        unit_ore = np.zeros((48,48), dtype=float)
        unit_water = np.zeros((48,48), dtype=float)
        unit_metal = np.zeros((48,48), dtype=float)
        unit_no_battery = np.zeros((48,48), dtype=float)
        unit_full_cargo = np.zeros((48,48), dtype=float)
        unit_on_factories = np.zeros((48,48), dtype=float)


        my_units = obs["units"][player]
        enemy_units = obs["units"][opponent]

        # add friendly light and heavy units to 2 different maps
        # important: normalization constants (different for light and heavies, or same for all?)
        for unit_id in my_units:
            unit = my_units[unit_id]


            if unit["unit_type"] == 'LIGHT':
                x, y = unit["pos"][0], unit["pos"][1]
                factory_there = self.env_.state.board.factory_occupancy_map[x,y]
                
                light_units[x][y] = 1
                unit_power[x][y] = unit["power"] #/ 150.
                unit_ice[x][y] = unit["cargo"]["ice"] #/ 100.
                unit_water[x][y] = unit["cargo"]["water"] #/ 100.
                unit_metal[x][y] = unit["cargo"]["metal"] #/ 100.
                unit_ore[x][y] = unit["cargo"]["ore"] #/ 100.
                unit_no_battery[x][y] = 1 if unit["power"] < int(1) else 0
                unit_full_cargo[x][y] = 1 if unit["cargo"] == int(100) else 0
                unit_on_factories[x][y] = 1 if (factory_there != -1 and factory_there in self.env_.state.teams[player].factory_strains) else 0
                

            if unit["unit_type"] == 'HEAVY':
                x, y = unit["pos"][0], unit["pos"][1]
                factory_there = self.env_.state.board.factory_occupancy_map[x,y]


                heavy_units[x][y] = 1
                unit_power[x][y] = unit["power"] #/ 3000.
                unit_ice[x][y] = unit["cargo"]["ice"] #/ 1000.
                unit_water[x][y] = unit["cargo"]["water"] #/ 1000.
                unit_metal[x][y] = unit["cargo"]["metal"] #/ 1000.
                unit_ore[x][y] = unit["cargo"]["ore"] #/ 1000.
                unit_no_battery[x][y] = 1 if unit["power"] < int(10) else 0
                unit_full_cargo[x][y] = 1 if unit["cargo"] == int(1000) else 0
                unit_on_factories[x][y] = 1 if (factory_there != -1 and factory_there in self.env_.state.teams[player].factory_strains) else 0

        # add enemy light and heavy units to 2 different maps
        # important, since the presence of an enemy unit is -1 in its map, its power and cargo should also be negative?
        for unit_id in enemy_units:
            unit = enemy_units[unit_id]
            
            if unit["unit_type"] == 'LIGHT':
                x, y = unit["pos"][0], unit["pos"][1]
                factory_there = self.env_.state.board.factory_occupancy_map[x,y]


                light_units[x][y] = -1

                unit_power[x][y] = - unit["power"] #/ 150.
                unit_ice[x][y] = - unit["cargo"]["ice"] #/ 100.
                unit_water[x][y] = - unit["cargo"]["water"] #/ 100.
                unit_metal[x][y] = - unit["cargo"]["metal"] #/ 100.
                unit_ore[x][y] = - unit["cargo"]["ore"] #/ 100.
                unit_no_battery[x][y] = -1 if unit["power"] < int(1) else 0
                unit_full_cargo[x][y] = -1 if unit["cargo"] == int(100) else 0
                unit_on_factories[x][y] = -1 if (factory_there != -1 and factory_there in self.env_.state.teams[opponent].factory_strains) else 0


            if unit["unit_type"] == 'HEAVY':
                x, y = unit["pos"][0], unit["pos"][1]
                factory_there = self.env_.state.board.factory_occupancy_map[x,y]

                heavy_units[x][y] = -1

                unit_power[x][y] = - unit["power"] #/ 3000.
                unit_ice[x][y] = - unit["cargo"]["ice"] #/ 1000.
                unit_water[x][y] = - unit["cargo"]["water"] #/ 1000.
                unit_metal[x][y] = - unit["cargo"]["metal"] #/ 1000.
                unit_ore[x][y] = - unit["cargo"]["ore"] #/ 1000.
                unit_no_battery[x][y] = -1 if unit["power"] < int(10) else 0
                unit_full_cargo[x][y] = -1 if unit["cargo"] == int(1000) else 0
                unit_on_factories[x][y] = -1 if (factory_there != -1 and factory_there in self.env_.state.teams[opponent].factory_strains) else 0
                
        # get factories data
        factories = np.zeros((48,48), dtype=float)
        factory_power = np.zeros((48,48), dtype=float)
        factory_ice = np.zeros((48,48), dtype=float)
        factory_ore = np.zeros((48,48), dtype=float)
        factory_water = np.zeros((48,48), dtype=float)
        factory_metal = np.zeros((48,48), dtype=float)

        my_factories = obs["factories"][player]
        enemy_factories = obs["factories"][opponent]

        # add factory positions and cargo. positive for my factories and negative otherwise
        # important: as with units, do I need to make all enemy things negative? if presence map is already -1 where there is an enemy factory
        # important: as capacity in factories is infinite, what normalization should I use? right now using same constant as for heavy robots
        for factory_id in my_factories:
            factory = my_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x][y] = 1
            factory_power[x][y] = factory["power"] #/ 1000.
            factory_ice[x][y] = factory["cargo"]["ice"] #/ 500.
            factory_water[x][y] = factory["cargo"]["water"] #/ 500.
            factory_metal[x][y] = factory["cargo"]["metal"] #/ 500.
            factory_ore[x][y] = factory["cargo"]["ore"] #/ 500.

        for factory_id in enemy_factories:
            factory = enemy_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x][y] = - 1
            factory_power[x][y] = - factory["power"] #/ 1000.
            factory_ice[x][y] = - factory["cargo"]["ice"] #/ 500.
            factory_water[x][y] = - factory["cargo"]["water"] #/ 500.
            factory_metal[x][y] = - factory["cargo"]["metal"] #/ 500.
            factory_ore[x][y] = - factory["cargo"]["ore"] #/ 500.

        # 30 channels of information!
        obs = np.stack([
            turn,
            turn_in_cycle,
            is_day_,
            rubble, 
            ore, 
            ice,
            lichen, 
            #lichen_strains, 
            valid_spawns,
            light_units,
            heavy_units,
            unit_ice,
            unit_metal,
            unit_water,
            unit_ore,
            unit_power,
            unit_no_battery,
            unit_full_cargo,
            unit_on_factories,
            factories,
            factory_ice,
            factory_metal,
            factory_ore,
            factory_water,
            factory_power
        ]).astype("float64")

        # transpose for channel last
        return obs.transpose(1,2,0)


    def enemy_step(self):
        with torch.no_grad():
            obs = torch.Tensor(self.current_enemy_obs).unsqueeze(0).to(self.device)
            robot_action_type, robot_action_params, factory_action, _, _, robot_invalid_action_masks_action_types, robot_invalid_action_masks_action_params, factory_invalid_action_masks = self.enemy_agent.get_action(obs, envs=self, player="player_1")

            robot_action = torch.cat([robot_action_type, robot_action_params], dim = 2)

            # the robot real action adds the source units
            robot_real_action = torch.cat([
                torch.stack(
                    [torch.arange(0, 48 * 48, device= self.device) for i in range(1)
            ]).unsqueeze(2), robot_action], 2)

            # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
            # so as to predict an action for each cell in the map; this obviously include a 
            # lot of invalid actions at cells for which no source units exist, so the rest of 
            # the code removes these invalid actions to speed things up
            robot_real_action = robot_real_action.cpu().numpy()
            robot_valid_actions = robot_real_action[robot_invalid_action_masks_action_types[:,:,0].bool().cpu().numpy()]
            robot_valid_actions_counts = robot_invalid_action_masks_action_types[:,:,0].sum(1).long().cpu().numpy()
            robot_java_valid_actions = []
            robot_valid_action_idx = 0
            for env_idx, robot_valid_action_count in enumerate(robot_valid_actions_counts):
                robot_java_valid_action = []
                for c in range(robot_valid_action_count):
                    robot_java_valid_action += [JArray(JInt)(robot_valid_actions[robot_valid_action_idx])]
                    robot_valid_action_idx += 1
                robot_java_valid_actions += [JArray(JArray(JInt))(robot_java_valid_action)]
            robot_java_valid_actions = JArray(JArray(JArray(JInt)))(robot_java_valid_actions)

            # the robot real action adds the source units
            factory_real_action = torch.cat([
                torch.stack(
                    [torch.arange(0, 48 * 48, device=self.device) for i in range(1)
            ]).unsqueeze(2), factory_action], 2)

            # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
            # so as to predict an action for each cell in the map; this obviously include a 
            # lot of invalid actions at cells for which no source units exist, so the rest of 
            # the code removes these invalid actions to speed things up
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

if __name__ == "__main__":
    envs = gym.vector.SyncVectorEnv(
        [make_env(3 + i) for i in range(8)]
    )

    obs = envs.reset()

    action = dict()
    action["factories"] = [10, 2]
    action["robots"] = [15, 2, 1, 1, 1, 1, 1, 1, 1, 0]