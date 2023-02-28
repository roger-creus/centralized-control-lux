# http://proceedings.mlr.press/v97/han19a/han19a.pdf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import wandb


import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor, NormalizeObservation
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import jpype
import copy

from IPython import embed

from envs.custom_env_cen_dec import CustomLuxEnvCenDec

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from invalid_action_masks import *

parser = argparse.ArgumentParser(description='PPO agent')
# Common arguments
parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                    help='the name of this experiment')
parser.add_argument('--gym-id', type=str, default="MicrortsDefeatCoacAIShaped-v3",
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

parser.add_argument('--save-path', type=str, default="outputs",
                        help="how many train updates between saving a new checkpoint and loading a new enemy")
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
args.save_path = "checkpoints"

# Utility for JArrays
def init_jvm(jvmpath=None):
    if jpype.isJVMStarted():
        return
    jpype.startJVM(jpype.getDefaultJVMPath())

# utility to create Vectorized Env
def make_env(seed, idx, self_play, device):
    def thunk():
        env = CustomLuxEnvCenDec(self_play=self_play, device = device, PATH_AGENT_CHECKPOINTS = args.save_path)
        #env = NormalizeObservation(env)
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
    

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

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

# env setup
init_jvm()
#envs = gym.vector.SyncVectorEnv([make_env(i + args.seed, i, args.self_play, device) for i in range(args.num_envs)])
envs = make_env(args.seed, 0, args.self_play, device)()


class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class Agent(nn.Module):
    def __init__(self, mapsize=48*48):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        c, h, w = envs.observation_space.shape
        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [16, 32, 32]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

        self.actor = layer_init(nn.Linear(256, envs.action_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def forward(self, x):
        return self.network(x) # "bhwc" -> "bchw"

    def get_action(self, x, action=None, envs=None):
        logits = self.actor(self.forward(x))
        grid_logits = logits.view(-1, envs.action_space.nvec.sum())
        split_logits = torch.split(grid_logits, envs.action_space.nvec.tolist(), dim=-1)
        
        if action is None:
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            action = action.view(-1, action.shape[-1]).T
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        
        return action, logprob.sum(0), entropy.sum(0)

    def get_value(self, x):
        return self.critic(self(x))

agent = Agent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
mapsize = 48*48
action_space_shape = envs.action_space.shape[0]
invalid_action_shape = envs.action_space.nvec.sum()

# obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
obs = [None] * args.num_steps
actions = [None] * args.num_steps
logprobs = [None] * args.num_steps
rewards = [None] * args.num_steps
dones = [None] * args.num_steps
values = [None] * args.num_steps
invalid_action_masks = [None] * args.num_steps

# actions = torch.zeros((args.num_steps, args.num_envs) + (action_space_shape,)).to(device)
# logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
# rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
# dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
# values = torch.zeros((args.num_steps, args.num_envs)).to(device)
# invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60

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

        # TODO: only needed if not VecEnv
        if next_done[0].item() == 1:
            next_obs = torch.Tensor(envs.reset()).to(device)

        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()
            action, logproba, _ = agent.get_action(obs[step], envs=envs)

        actions[step] = action
        logprobs[step] = logproba

        try:
            next_obs = None
            next_obs, rs, ds, infos = envs.step(actions[step].cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
        
        except Exception as e:
            e.printStackTrace()
            raise

        rewards[step], next_done = torch.Tensor([rs]).to(device), torch.Tensor([ds]).to(device)

        for info in [infos]:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                break

    """
    # bootstrap reward if not done. reached the batch limit
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
        if args.gae:
            #advantages = torch.zeros_like(rewards).to(device)
            advantages = [None] * args.num_steps
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                embed() 
                # nextvalues can be of different shape of values because one new unit
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = [None] * args.num_steps

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return

            embed()
            advantages = returns - values
    """

    embed()
    advantages = None
    returns = None

    # flatten the batch
    b_obs = torch.cat(obs)
    #b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = torch.cat(logprobs)
    #b_logprobs = logprobs.reshape(-1)
    b_actions = torch.cat(actions, dim=1)
    #b_actions = actions.reshape((-1,)+action_space_shape)
    b_advantages = torch.cat(advantages)
    #b_advantages = advantages.reshape(-1)
    b_returns = torch.cat(returns)
    #b_returns = returns.reshape(-1)
    b_values = torch.cat(values)
    #b_values = values.reshape(-1)

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
            _, newlogproba, entropy, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_actions.long()[minibatch_ind],
                envs)
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