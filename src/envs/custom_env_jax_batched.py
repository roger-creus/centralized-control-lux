import gym
import numpy as np
import jax.numpy as jnp
import jux.utils
import numpy as np

from gym import spaces
from IPython import embed

from jux.config import JuxBufferConfig, EnvConfig
from jux.env import JuxEnvBatch

FACTIONS = [
    "AlphaStrike",
    "Mars"
]

class CustomLuxEnv(gym.Env):
    def __init__(self, seeds, env_cfg=None):
        if env_cfg is None:
            env_cfg = EnvConfig()

        self.env_ = JuxEnvBatch(buf_cfg=JuxBufferConfig(MAX_N_UNITS=200), env_cfg=env_cfg)
        self.seeds = seeds
        self.num_envs = len(self.seeds)

        # stores the current state of the game
        self.current_state = None

        # to know if we are in the bid phase
        self.first = True


    def reset(self):
        self.first = True
        self.current_state = self.env_.reset(self.seeds)

        return self.preprocess_obs(self.current_state)

    
    # action is a dict {"player_0" : x, "player_1" : y}
    def step(self, action):
        # BID PHASE
        if self.first:
            # action is {"player_0" : {"bid" : (batch_size, 1) , "faction : (batch_size, 1)}, "player_1" : {"bid" : (batch_size, 1), "faction : (batch_size, 1)}}
            
            bid_0 = action["player_0"]["bid"] # array of bids (batch_size, 1)
            bid_1 = action["player_0"]["bid"] # array of bids (batch_size, 1)

            faction_0 = action["player_0"]["faction"] # array of factions (batch_size, 1)
            faction_1 = action["player_1"]["faction"] # array of factions (batch_size, 1)

            bid_action = jnp.array(np.hstack([bid_0, bid_1]), dtype=jnp.int32)
            faction_action = jnp.array(np.hstack([faction_0, faction_1]), dtype=jnp.int8)

            self.current_state, (observations, rewards, dones, infos) = self.env_.step_bid(self.current_state, bid_action, faction_action)
            self.first = False

        # PLACEMENT PHASE
        elif self.current_state.real_env_steps < 0:
            # action is {"player_X" : {"spawn" : np.array([batch_size, x,y]), "water" :  np.array([batch_size, x]), "metal" :  np.array([batch_size, x])}}
            spawn_action = jnp.zeros([self.num_envs, 2,2], dtype=jnp.int8)
            water_action = jnp.zeros([self.num_envs, 2,2], dtype=jnp.int32)
            metal_action = jnp.zeros([self.num_envs, 2,2], dtype=jnp.int32)

            # get the player taking the action
            only_key = list(action.keys())[0]
            player_taking_action = int(only_key.split("_")[1])

            assert player_taking_action == self.current_state.next_player

            spawn_action[:, player_taking_action, :] = jnp.array(action[only_key]["spawn"], dtype=jnp.int8)
            water_action[:, player_taking_action] = jnp.array(action[only_key]["water"], dtype=jnp.int8)
            metal_action[:, player_taking_action] = jnp.array(action[only_key]["metal"], dtype=jnp.int8)

            self.current_state, (observations, rewards, dones, infos) = self.env_.step_factory_placement(self.current_state, spawn_action, water_action, metal_action)
            
        # NORMAL GAME    
        else:
            pass    
        return 0

    def render(self):
        return 0

    def preprocess_obs(self, obs):
        return obs


if __name__ == "__main__":
    batch_size = 8
    seeds = jnp.arange(batch_size)
    env = CustomLuxEnv(seeds)

    obs = env.reset()

    # craft bid action
    bid_0 = np.ones((batch_size, 1)) * 10
    bid_1 = np.ones((batch_size, 1)) * 5
    faction_0 = np.array([FACTIONS.index("AlphaStrike") for i in range(8)])[..., np.newaxis]
    faction_1 = np.array([FACTIONS.index("AlphaStrike") for i in range(8)])[..., np.newaxis]

    bid_action = {
        "player_0" : {"bid" : bid_0, "faction" : faction_0},
        "player_1" : {"bid" : bid_1, "faction" : faction_1}
    }

    env.step(bid_action)

    # craft placement action
    place_action = {
        "player_0" : {"spawn" : np.ones((batch_size, 1, 1)), "water" : np.ones((batch_size, 1)), "metal" : np.ones((batch_size, 1))}
    }

    env.step(place_action)
