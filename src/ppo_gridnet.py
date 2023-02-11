# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb

from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import jpype
import copy

from IPython import embed

from envs.custom_env import CustomLuxEnv

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from invalid_action_masks import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="Lux",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="lux",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default="rogercreus",
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    parser.add_argument('--self-play', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                            help="train by selfplay")
    parser.add_argument('--save-every', type=int, default=100,
                            help="how many train updates between saving a new checkpoint and loading a new enemy")
    parser.add_argument('--load-every', type=int, default=15,
                            help="how many train updates between saving a new checkpoint and loading a new enemy")
    parser.add_argument('--pool-size', type=int, default=10,
                            help="how many checkpoints to keep")                        

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

if args.self_play:
    PATH_AGENT_CHECKPOINTS = "agent_checkpoints"
    if not os.path.exists(PATH_AGENT_CHECKPOINTS):
        os.makedirs(PATH_AGENT_CHECKPOINTS)

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

# Utility for JArrays
def init_jvm(jvmpath=None):
    if jpype.isJVMStarted():
        return
    jpype.startJVM(jpype.getDefaultJVMPath())
    
# WandB Logging
if args.prod_mode:
    import wandb
    run = wandb.init(
        project=args.wandb_project_name, entity=args.wandb_entity,
        config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)

    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 50

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

# utility to create Vectorized Env
def make_env(seed, idx, self_play):
    def thunk():
        env = CustomLuxEnv(self_play=self_play)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

# Define a Masked Categorical distribution to sample legal actions only 
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[], sw=None):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.bool()
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8, device=device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

# Utility used in the CNNs of the Agent
class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)

# Utility to initialize the weights of the agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

#### The Agent Neural Network ####
class Agent(nn.Module):
    def __init__(self, mapsize=48*48):
        super(Agent, self).__init__()
        
        self.mapsize = mapsize

        # Feature extractor: stack of input feature maps -> hidden representation (vector) 
        # this can be a CNN or a ResNet
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(22, 16, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*3*3, 256)),
            nn.ReLU()
        )

        # we have a different actor for robots and factories with different action spaces
        self.actor_robots = layer_init(nn.Linear(256, self.mapsize * envs.single_action_space["robots"].nvec[1:].sum()), std=0.01)

        self.actor_factories = layer_init(nn.Linear(256, self.mapsize * envs.single_action_space["factories"].nvec[1:].sum()), std=0.01)
        
        # we only have one critic that outputs the value of a state
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def save_checkpoint(self, idx, pool_size, path):
        ckpts = os.listdir(path)
        ckpts.sort(key=lambda f: int(f.split(".")[0]))
        count = len(ckpts)
        torch.save(self.state_dict(), os.path.join(path, str(idx) + ".pt"))

        if count >= pool_size:
            os.remove(os.path.join(path, ckpts[0]))

    def load_checkpoint(self, path):
        ckpts = os.listdir(path)
        ckpts.sort(key=lambda f: int(f.split(".")[0]))

        import random
        r = random.uniform(0, 1)

        # with 50% chance load most recent ckpt
        if r <= 0.5 or len(ckpts) == 1:
            weights = torch.load(os.path.join(path, ckpts[-1]))
        else:
            weights = torch.load(os.path.join(path, np.random.choice(ckpts[:-1])))

        self.load_state_dict(weights)
        self.freeze_params()

    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

    def get_action(self, x, robot_action=None, factory_action=None, robot_invalid_action_masks=None, factory_invalid_action_masks=None, envs=None, player="player_0"):
        x = self.forward(x)

        # player_0 operates in VecEnv so needs to access the single_obs_space (and action)
        if player == "player_0":
            robots_nvec = envs.single_action_space["robots"].nvec
            robots_nvec_sum = envs.single_action_space["robots"].nvec[1:].sum()
            robots_nvec_tolist = envs.single_action_space["robots"].nvec[1:].tolist()
            factories_nvec = envs.single_action_space["factories"].nvec
            factories_nvec_sum = envs.single_action_space["factories"].nvec[1:].sum()
            factories_nvec_tolist = envs.single_action_space["factories"].nvec[1:].tolist()
        
        # player_1 operates in a single custom env so there is no such thing as single_obs_space
        if player == "player_1":
            robots_nvec = envs.action_space["robots"].nvec
            robots_nvec_sum = envs.action_space["robots"].nvec[1:].sum()
            robots_nvec_tolist = envs.action_space["robots"].nvec[1:].tolist()
            factories_nvec = envs.action_space["factories"].nvec
            factories_nvec_sum = envs.action_space["factories"].nvec[1:].sum()
            factories_nvec_tolist = envs.action_space["factories"].nvec[1:].tolist()

        robot_logits = self.actor_robots(x)
        robot_grid_logits = robot_logits.view(-1, robots_nvec_sum)
        robot_split_logits = torch.split(robot_grid_logits, robots_nvec_tolist , dim=1)

        factory_logits = self.actor_factories(x)
        factory_grid_logits = factory_logits.view(-1, factories_nvec_sum)
        factory_split_logits = torch.split(factory_grid_logits, factories_nvec_tolist, dim=1)
        
        if robot_action is None and factory_action is None:
            # get robot valid actions
            robot_invalid_action_masks = torch.tensor(get_robot_invalid_action_masks(envs, player)).to(device)
            robot_invalid_action_masks = robot_invalid_action_masks.view(-1, robot_invalid_action_masks.shape[-1])
            robot_split_invalid_action_masks = torch.split(robot_invalid_action_masks[:,1:], robots_nvec_tolist, dim=1)
            robot_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(robot_split_logits, robot_split_invalid_action_masks)]
            robot_action = torch.stack([categorical.sample() for categorical in robot_multi_categoricals])

            # get factory valid actions
            factory_invalid_action_masks = torch.tensor(get_factory_invalid_action_masks(envs, player)).to(device)
            factory_invalid_action_masks = factory_invalid_action_masks.view(-1, factory_invalid_action_masks.shape[-1])
            factory_split_invalid_action_masks = torch.split(factory_invalid_action_masks[:,1:], factories_nvec_tolist, dim=1)
            factory_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(factory_split_logits, factory_split_invalid_action_masks)]
            factory_action = torch.stack([categorical.sample() for categorical in factory_multi_categoricals])
        else:
            robot_invalid_action_masks = robot_invalid_action_masks.view(-1, robot_invalid_action_masks.shape[-1])
            robot_action = robot_action.view(-1, robot_action.shape[-1]).T
            robot_split_invalid_action_masks = torch.split(robot_invalid_action_masks[:,1:], robots_nvec_tolist, dim=1)
            robot_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(robot_split_logits, robot_split_invalid_action_masks)]

            factory_invalid_action_masks = factory_invalid_action_masks.view(-1, factory_invalid_action_masks.shape[-1])
            factory_action = factory_action.view(-1, factory_action.shape[-1]).T
            factory_split_invalid_action_masks = torch.split(factory_invalid_action_masks[:,1:], factories_nvec_tolist, dim=1)
            factory_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(factory_split_logits, factory_split_invalid_action_masks)]
        
        robot_logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(robot_action, robot_multi_categoricals)])
        robot_entropy = torch.stack([categorical.entropy() for categorical in robot_multi_categoricals])
        robot_num_predicted_parameters = len(robots_nvec) - 1
        robot_logprob = robot_logprob.T.view(-1, 48 * 48, robot_num_predicted_parameters)
        robot_entropy = robot_entropy.T.view(-1, 48 * 48, robot_num_predicted_parameters)
        robot_action = robot_action.T.view(-1, 48 * 48, robot_num_predicted_parameters)
        robot_invalid_action_masks = robot_invalid_action_masks.view(-1,  48 * 48, robots_nvec_sum + 1)

        factory_logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(factory_action, factory_multi_categoricals)])
        factory_entropy = torch.stack([categorical.entropy() for categorical in factory_multi_categoricals])
        factory_num_predicted_parameters = len(factories_nvec) - 1
        factory_logprob = factory_logprob.T.view(-1, 48 * 48, factory_num_predicted_parameters)
        factory_entropy = factory_entropy.T.view(-1, 48 * 48, factory_num_predicted_parameters)
        factory_action = factory_action.T.view(-1, 48 * 48, factory_num_predicted_parameters)
        factory_invalid_action_masks = factory_invalid_action_masks.view(-1,  48 * 48, factories_nvec_sum + 1)

        robot_logprob = robot_logprob.sum(1).sum(1)
        factory_logprob = factory_logprob.sum(1).sum(1)
        logprob = robot_logprob + factory_logprob

        robot_entropy = robot_entropy.sum(1).sum(1)
        factory_entropy = factory_entropy.sum(1).sum(1)
        entropy = robot_entropy + factory_entropy

        return robot_action, factory_action, logprob, entropy, robot_invalid_action_masks, factory_invalid_action_masks

    def get_value(self, x):
        return self.critic(self.forward(x))

init_jvm()

envs = gym.vector.SyncVectorEnv([make_env(i + args.seed, i, args.self_play) for i in range(args.num_envs)])

agent = Agent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

num_models_saved = 0
agent.save_checkpoint(num_models_saved, args.pool_size, PATH_AGENT_CHECKPOINTS)
num_models_saved += 1

for i in range(len(envs.envs)):
    enemy_agent = Agent().to(device)
    enemy_agent.freeze_params()
    envs.envs[i].set_enemy_agent(enemy_agent)

if args.anneal_lr:
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
mapsize = 48*48

# TODO: why is there +1 in invalid action shape?
robot_action_space_shape = (mapsize, envs.single_action_space["robots"].shape[0] - 1)
robot_invalid_action_shape = (mapsize, envs.single_action_space["robots"].nvec[1:].sum() + 1)

factory_action_space_shape = (mapsize, envs.single_action_space["factories"].shape[0] - 1)
factory_invalid_action_shape = (mapsize, envs.single_action_space["factories"].nvec[1:].sum() + 1)

robot_actions = torch.zeros((args.num_steps, args.num_envs) + robot_action_space_shape).to(device)
robot_invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + robot_invalid_action_shape).to(device)

factory_actions = torch.zeros((args.num_steps, args.num_envs) + factory_action_space_shape).to(device)
factory_invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + factory_invalid_action_shape).to(device)

obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()

next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

## CRASH AND RESUME LOGIC:
starting_update = 1
from jpype.types import JArray, JInt
if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1
    global_step = starting_update * args.batch_size
    api = wandb.Api()
    run = api.run(f"{run.entity}/{run.project}/{run.id}")
    model = run.file('agent.pt')
    model.download(f"models/{experiment_name}/")
    agent.load_state_dict(torch.load(f"models/{experiment_name}/agent.pt", map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")


for update in range(starting_update, num_updates+1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()
            robot_action, factory_action, logproba, _, robot_invalid_action_masks[step], factory_invalid_action_masks[step] = agent.get_action(obs[step], envs=envs)

        robot_actions[step] = robot_action
        factory_actions[step] = factory_action

        logprobs[step] = logproba


        # TRY NOT TO MODIFY: execute the game and log data.
        
        # the robot real action adds the source units
        robot_real_action = torch.cat([
            torch.stack(
                [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
        ]).unsqueeze(2), robot_action], 2)
        
        # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
        # so as to predict an action for each cell in the map; this obviously include a 
        # lot of invalid actions at cells for which no source units exist, so the rest of 
        # the code removes these invalid actions to speed things up
        robot_real_action = robot_real_action.cpu().numpy()
        robot_valid_actions = robot_real_action[robot_invalid_action_masks[step][:,:,0].bool().cpu().numpy()]
        robot_valid_actions_counts = robot_invalid_action_masks[step][:,:,0].sum(1).long().cpu().numpy()
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
                [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
        ]).unsqueeze(2), factory_action], 2)
        
        # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
        # so as to predict an action for each cell in the map; this obviously include a 
        # lot of invalid actions at cells for which no source units exist, so the rest of 
        # the code removes these invalid actions to speed things up
        factory_real_action = factory_real_action.cpu().numpy()
        factory_valid_actions = factory_real_action[factory_invalid_action_masks[step][:,:,0].bool().cpu().numpy()]
        factory_valid_actions_counts = factory_invalid_action_masks[step][:,:,0].sum(1).long().cpu().numpy()
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

        actions = []
        for i in range(args.num_envs):
            actions.append({
                "factories" : factory_valid_actions[i],
                "robots" : robot_valid_actions[i]
            })

        try:
            next_obs, rs, ds, infos = envs.step(actions)
            next_obs = torch.Tensor(next_obs).to(device)
        except Exception as e:
            e.printStackTrace()
            raise

        rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(device)

        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}, episode_winner={'player_0' if info['result'] == 1 else 'player_1'}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                writer.add_scalar("charts/episode_length", info['episode']['l'], global_step)
                writer.add_scalar("charts/episode_winner", info['result'], global_step)
                break

    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_robot_actions = robot_actions.reshape((-1,) + robot_action_space_shape)
    b_factory_actions = factory_actions.reshape((-1,) + factory_action_space_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_robot_invalid_action_masks = robot_invalid_action_masks.reshape((-1,) + robot_invalid_action_shape)
    b_factory_invalid_action_masks = factory_invalid_action_masks.reshape((-1,) + factory_invalid_action_shape)

    # Optimizaing the policy and value network
    inds = np.arange(args.batch_size,)
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            # raise

            _, _, newlogproba, entropy, _, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_robot_actions.long()[minibatch_ind],
                b_factory_actions.long()[minibatch_ind],
                b_robot_invalid_action_masks[minibatch_ind],
                b_factory_invalid_action_masks[minibatch_ind],
                envs
            )
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            # Stats
            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            # Value loss
            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 *((new_values - b_returns[minibatch_ind]) ** 2)

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

    ## CRASH AND RESUME LOGIC:
    if args.prod_mode:
        if not os.path.exists(f"models/{experiment_name}"):
            os.makedirs(f"models/{experiment_name}")
            torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")
            wandb.save(f"agent.pt")
        else:
            if update % CHECKPOINT_FREQUENCY == 0:
                torch.save(agent.state_dict(), f"{wandb.run.dir}/agent.pt")

    if update % args.save_every:
        agent.save_checkpoint(num_models_saved, args.pool_size, PATH_AGENT_CHECKPOINTS)
        num_models_saved += 1

    if update % args.load_every:
        for i in range(len(envs.envs)):
            envs.envs[i].update_enemy_agent()

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))

envs.close()
writer.close()