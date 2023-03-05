import dataclasses
import json
import numpy as np
from luxai2022.env import LuxAI2022
from luxai_runner.utils import to_json
from IPython import get_ipython
from IPython.display import display, HTML

def run_agents(agent1, agent2):
    env = LuxAI2022()

    # This code is partially based on the luxai2022 CLI:
    # https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/luxai_runner/episode.py

    obs = env.reset()
    state_obs = env.state.get_compressed_obs()

    agents = {
        "player_0": agent1("player_0", env.state.env_cfg),
        "player_1": agent2("player_1", env.state.env_cfg)
    }

    game_done = False
    rewards, dones, infos = {}, {}, {}

    for agent_id in agents:
        rewards[agent_id] = 0
        dones[agent_id] = 0
        infos[agent_id] = {
            "env_cfg": dataclasses.asdict(env.state.env_cfg)
        }

    replay = {
        "observations": [state_obs],
        "actions": [{}]
    }

    i = 0
    while not game_done:
        i += 1

        actions = {}
        for agent_id, agent in agents.items():
            agent_obs = obs[agent_id]

            if env.state.real_env_steps < 0:
                agent_actions = agent.early_setup(env.env_steps, agent_obs)
            else:
                agent_actions = agent.act(env.env_steps, agent_obs)

            for key, value in agent_actions.items():
                if isinstance(value, list):
                    agent_actions[key] = np.array(value)

            actions[agent_id] = agent_actions

        new_state_obs, rewards, dones, infos = env.step(actions)

        change_obs = env.state.get_change_obs(state_obs)
        state_obs = new_state_obs["player_0"]
        obs = new_state_obs

        replay["observations"].append(change_obs)
        replay["actions"].append(actions)

        players_left = len(dones)
        for key in dones:
            if dones[key]:
                players_left -= 1

        if players_left < 2:
            game_done = True

    execution_id = get_ipython().execution_count

    html = f"""
<iframe
    src="https://jmerle.github.io/lux-eye-2022/kaggle"
    width="1040"
    height="560"
    id="luxEye2022IFrame{execution_id}"
    frameBorder="0"
></iframe>

<script>
document.querySelector('#luxEye2022IFrame{execution_id}').addEventListener('load', () => {{
    document.querySelector('#luxEye2022IFrame{execution_id}').contentWindow.postMessage({json.dumps(to_json(replay))}, 'https://jmerle.github.io');
}});
</script>
    """

    display(HTML(html))

# from luxai_s2 import obs_to_game_state, GameState, EnvConfig
# from luxai_s2.utils import direction_to, my_turn_to_place_factory
# import numpy as np
# import sys
# class Agent():
#     def __init__(self, player: str, env_cfg: EnvConfig) -> None:
#         self.player = player
#         self.opp_player = "player_1" if self.player == "player_0" else "player_0"
#         np.random.seed(0)
#         self.env_cfg: EnvConfig = env_cfg

#     def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
#         if step == 0:
#             # bid 0 to not waste resources bidding and declare as the default faction
#             return dict(faction="AlphaStrike", bid=0)
#         else:
#             game_state = obs_to_game_state(step, self.env_cfg, obs)
#             # factory placement period

#             # how much water and metal you have in your starting pool to give to new factories
#             water_left = game_state.teams[self.player].water
#             metal_left = game_state.teams[self.player].metal

#             # how many factories you have left to place
#             factories_to_place = game_state.teams[self.player].factories_to_place
#             # whether it is your turn to place a factory
#             my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
#             if factories_to_place > 0 and my_turn_to_place:
#                 # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
#                 potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
#                 spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
#                 return dict(spawn=spawn_loc, metal=150, water=150)
#             return dict()

#     def act(self, step: int, obs, remainingOverageTime: int = 60):
#         actions = dict()
#         game_state = obs_to_game_state(step, self.env_cfg, obs)
#         factories = game_state.factories[self.player]
#         game_state.teams[self.player].place_first
#         factory_tiles, factory_units = [], []
#         for unit_id, factory in factories.items():
#             if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
#             factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST:
#                 actions[unit_id] = factory.build_heavy()
#             if self.env_cfg.max_episode_length - game_state.real_env_steps < 50:
#                 if factory.water_cost(game_state) <= factory.cargo.water:
#                     actions[unit_id] = factory.water()
#             factory_tiles += [factory.pos]
#             factory_units += [factory]
#         factory_tiles = np.array(factory_tiles)

#         units = game_state.units[self.player]
#         ice_map = game_state.board.ice
#         ice_tile_locations = np.argwhere(ice_map == 1)
#         for unit_id, unit in units.items():
#             if unit.power < unit.action_queue_cost(game_state):
#                 continue

#             # track the closest factory
#             closest_factory = None
#             adjacent_to_factory = False
#             if len(factory_tiles) > 0:
#                 actions[unit_id] = [unit.move(0, repeat=0)]
#                 factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
#                 closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
#                 closest_factory = factory_units[np.argmin(factory_distances)]
#                 adjacent_to_factory = np.mean((closest_factory_tile - unit.pos) ** 2) == 0

#                 # previous ice mining code
#                 if unit.cargo.ice < 40:
#                     ice_tile_distances = np.mean((ice_tile_locations - unit.pos) ** 2, 1)
#                     closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
#                     if np.all(closest_ice_tile == unit.pos):
#                         if unit.power >= unit.dig_cost(game_state) + unit.action_queue_cost(game_state):
#                             actions[unit_id] = [unit.dig(repeat=0)]
#                     else:
#                         direction = direction_to(unit.pos, closest_ice_tile)
#                         move_cost = unit.move_cost(game_state, direction)
#                         if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
#                             actions[unit_id] = [unit.move(direction, repeat=0)]
#                 # else if we have enough ice, we go back to the factory and dump it.
#                 elif unit.cargo.ice >= 40:
#                     direction = direction_to(unit.pos, closest_factory_tile)
#                     if adjacent_to_factory:
#                         if unit.power >= unit.action_queue_cost(game_state):
#                             actions[unit_id] = [unit.transfer(direction, 0, unit.cargo.ice, repeat=0)]
#                     else:
#                         move_cost = unit.move_cost(game_state, direction)
#                         if move_cost is not None and unit.power >= move_cost + unit.action_queue_cost(game_state):
#                             actions[unit_id] = [unit.move(direction, repeat=0)]
#         return actions
    
run_agents(Agent, Agent)