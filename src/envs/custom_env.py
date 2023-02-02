
import gym
import numpy as np
import luxai_s2.env

from IPython import embed
from gym import spaces
from luxai_s2.env import LuxAI_S2
from luxai_s2.utils.utils import my_turn_to_place_factory, is_day


"""
This custom env handles self-play and training of the bidder and placer.
1) Handles enemy sampling from a pool of checkpoints and its actions at each turn (so that from outside, env.step() only requires the agent action, needed for VectorizedEnvironments)
2) Every reset() call makes it to the normal game phase directly (queries the bidder and placer before resetting and trains them with a DQN-like algo and replay buffer)
"""
class CustomLuxEnv(gym.Env):
    def __init__(self, enemy_agent=None, env_cfg = None):
        self.env_ = LuxAI_S2(env_cfg, verbose=False)
        
        self.player = "player_0"

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(48, 48, 22), dtype=np.float64)
        
        # dim 1: action type [NOOP, move, transfer, pickup, dig, self-destruct, recharge-x] -- LENGTH 7
        # dim 2: move direction [up, right, down, left] -- LENGTH 4
        # dim 3: transfer direction [up, right, down, left] -- LENGTH 4
        # dim 4: transfer amount [25%, 50%, 75%, 100%] -- LENGTH 4
        # dim 5: transfer material [power, ore, metal, ice , water] --LENGTH 5
        # dim 6: pickup amount [25%, 50%, 75%, 100%] -- LENGTH 4
        # dim 7: pickup material [power, ore, metal, ice , water] --LENGTH 5
        # dim 8: recharge parameter [25%, 50%, 75%] -- LENGTH 3
        # dim 9: repeat [0,1] -- LENGTH 2
        # TODO: dim 10: N (action repeat parameter)
        self.action_space = gym.spaces.Dict({
            'robots': gym.spaces.MultiDiscrete(
                [(48 * 48) - 1, 7, 4, 4, 4, 5, 4, 5, 3, 2]
            ),
            'factories': gym.spaces.MultiDiscrete(
                [(48 * 48) - 1, 3]
            ),
        })

        # enemy agent is PyTorch model
        #self.enemy_agent = enemy_agent 
        #self.enemy_agent = self.enemy_agent.eval()

        # keeps updated versions of the player and enemy observations (preprocessed observations)
        self.current_player_obs = None
        self.current_enemy_obs = None

        # keep the current game observation
        self.current_state = None

        # this env handles the training of these 2
        self.bidder = None
        self.placer = None

        self.max_queue = 1
        self.max_lichen_and_rubble = 100

    # TODO: figure out raw model output action shape
    def step(self, action):

        # turn the raw player model outputs to game actions
        player_action = self.act_(action, "player_0")  # returs {"player_0" : actions}

        # get enemy action (raw output model)
        enemy_action = self.enemy_step(self.current_enemy_obs)
        # turn the raw enemy model outputs to game actions
        enemy_action = self.act_(enemy_action, "player_1") # returs {"player_1" : actions}

        actions = {**player_action, **enemy_action} # final dict of actions from both players to send to the game

        # step actions to true env
        observations, reward, done, info = self.env_.step(action)

        # keep the current game state updated
        self.current_state = observations
    
        # update the current observations for both players and preprocess them into model input format
        self.current_player_obs = self.obs_(observations, "player_0")
        self.current_enemy_obs = self.obs_(observations, "player_1")

        # important: only return from the point of view of player! enemy is handled as part of the environment
        return self.current_player_obs, reward["player_0"], done["player_0"], info["player_0"]

    def reset(self):
        observations = self.env_.reset()

        # TODO: handle enemy sampling

        # TODO: update the current observations for both players and preprocess them into BIDDER input format
        
        # self.current_player_obs = self.obs_bid_phase_(observations, "player_0")
        #self.current_enemy_obs = self.obs_bid_phase_(observations, "player_1")

        # TODO: get action from bidder model (preprocess into placer format)
        actions = {"player_0" : {"bid" : 0, "faction" : "AlphaStrike"}, "player_1" : {"bid" : 0, "faction" : "AlphaStrike"}}

        # step into factory placement phase
        observations, _, _, _ = self.env_.step(actions)

        # handle all the factory placement phase
        while self.env_.state.real_env_steps < 0:
            action = dict()
            
            for agent in self.env_.agents:
                if my_turn_to_place_factory(observations[agent]["teams"][agent]["place_first"], self.env_.state.env_steps):
                    # TODO: get action from placer model
                    potential_spawns = np.array(list(zip(*np.where(observations[agent]["board"]["valid_spawns_mask"] == 1))))
                    spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                    action[agent] = dict(spawn=spawn_loc, metal=150, water=150)
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
        return {player : None}

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
        turn = np.tile(env_steps, (48,48)) / 999.       # max turns is 1000
        turn_in_cycle = np.tile((env_steps % 50), (48,48)) / 49.
        is_day_ = np.tile(int(is_day(self.env_.env_cfg, env_steps)), (48,48))

        # get map data
        rubble = np.array(obs["board"]["rubble"], dtype=float)
        ore = np.array(obs["board"]["ore"], dtype=float)
        ice = np.array(obs["board"]["ice"], dtype=float)
        lichen = np.array(obs["board"]["lichen"], dtype=float)
        lichen_strains = np.array(obs["board"]["lichen_strains"], dtype=float)
        valid_spawns = np.array(obs["board"]["valid_spawns_mask"], dtype=float)

        # normalization between 0 and 1 (we will try to keep negative values to indicate only enemy)
        rubble = rubble / self.max_lichen_and_rubble
        lichen = lichen / self.max_lichen_and_rubble
        ore = ore / self.max_lichen_and_rubble
        ice = ice / self.max_lichen_and_rubble
        lichen_strains = lichen_strains / 10    # important: 10 is a heuristic

        # get units data
        light_units = np.zeros((48,48), dtype=float)
        heavy_units = np.zeros((48,48), dtype=float)
        unit_power = np.zeros((48,48), dtype=float)
        unit_ice = np.zeros((48,48), dtype=float)
        unit_ore = np.zeros((48,48), dtype=float)
        unit_water = np.zeros((48,48), dtype=float)
        unit_metal = np.zeros((48,48), dtype=float)

        my_units = obs["units"][player]
        enemy_units = obs["units"][opponent]

        # add friendly light and heavy units to 2 different maps
        # important: normalization constants (different for light and heavies, or same for all?)
        for unit_id in my_units:
            unit = my_units[unit_id]

            if unit["unit_type"] == 'LIGHT':
                x, y = unit["pos"][0], unit["pos"][1]
                
                light_units[x][y] = 1
                unit_power[x][y] = unit["power"] / 3000.
                unit_ice[x][y] = unit["cargo"]["ice"] / 1000.
                unit_water[x][y] = unit["cargo"]["water"] / 1000.
                unit_metal[x][y] = unit["cargo"]["metal"] / 1000.
                unit_ore[x][y] = unit["cargo"]["ore"] / 1000.

            if unit["unit_type"] == 'HEAVY':
                x, y = unit["pos"][0], unit["pos"][1]

                heavy_units[x][y] = 1
                unit_power[x][y] = unit["power"] / 3000.
                unit_ice[x][y] = unit["cargo"]["ice"] / 1000.
                unit_water[x][y] = unit["cargo"]["water"] / 1000.
                unit_metal[x][y] = unit["cargo"]["metal"] / 1000.
                unit_ore[x][y] = unit["cargo"]["ore"] / 1000.

        # add enemy light and heavy units to 2 different maps
        # important, since the presence of an enemy unit is -1 in its map, its power and cargo should also be negative?
        for unit_id in enemy_units:
            unit = enemy_units[unit_id]
            
            if unit["unit_type"] == 'LIGHT':
                x, y = unit["pos"][0], unit["pos"][1]

                light_units[x][y] = -1

                unit_power[x][y] = - unit["power"] / 3000.
                unit_ice[x][y] = - unit["cargo"]["ice"] / 1000.
                unit_water[x][y] = - unit["cargo"]["water"] / 1000.
                unit_metal[x][y] = - unit["cargo"]["metal"] / 1000.
                unit_ore[x][y] = - unit["cargo"]["ore"] / 1000.

            if unit["unit_type"] == 'HEAVY':
                x, y = unit["pos"][0], unit["pos"][1]

                heavy_units[x][y] = -1

                unit_power[x][y] = - unit["power"] / 3000.
                unit_ice[x][y] = - unit["cargo"]["ice"] / 1000.
                unit_water[x][y] = - unit["cargo"]["water"] / 1000.
                unit_metal[x][y] = - unit["cargo"]["metal"] / 1000.
                unit_ore[x][y] = - unit["cargo"]["ore"] / 1000.
                
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
            factory_power[x][y] = factory["power"] / 3000.
            factory_ice[x][y] = factory["cargo"]["ice"] / 1000.
            factory_water[x][y] = factory["cargo"]["water"] / 1000.
            factory_metal[x][y] = factory["cargo"]["metal"] / 1000.
            factory_ore[x][y] = factory["cargo"]["ore"] / 1000.

        for factory_id in enemy_factories:
            factory = enemy_factories[factory_id]
            x, y = factory["pos"][0], factory["pos"][1]

            factories[x][y] = - 1
            factory_power[x][y] = - factory["power"] / 3000.
            factory_ice[x][y] = - factory["cargo"]["ice"] / 1000.
            factory_water[x][y] = - factory["cargo"]["water"] / 1000.
            factory_metal[x][y] = - factory["cargo"]["metal"] / 1000.
            factory_ore[x][y] = - factory["cargo"]["ore"] / 1000.

        # 30 channels of information!
        obs = np.stack([
            turn,
            turn_in_cycle,
            is_day_,
            rubble, 
            ore, 
            ice,
            lichen, 
            lichen_strains, 
            valid_spawns,
            light_units,
            heavy_units,
            unit_ice,
            unit_metal,
            unit_water,
            unit_ore,
            unit_power,
            factories,
            factory_ice,
            factory_metal,
            factory_ore,
            factory_water,
            factory_power
        ]).astype("float64")

        # transpose for channel last
        return obs.transpose(1,2,0)


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

    embed()
