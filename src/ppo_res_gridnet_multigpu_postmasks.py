import argparse
import os
import random
import time
import warnings
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from gym.wrappers import TimeLimit, Monitor, NormalizeObservation
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import jpype
from jpype.types import JArray, JInt

import copy

from IPython import embed
import torch.nn.functional as F

from envs_folder.custom_env import CustomLuxEnv

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from invalid_action_masks import *
from utils import *


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="lux",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="rogercreus",
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Lux-Multigpu",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100000000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument('--gae', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True,
        help='Use GAE for advantage computation')
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--device-ids", nargs="+", default=[],
        help="the device ids that subprocess workers will use")
    parser.add_argument("--backend", type=str, default="gloo", choices=["gloo", "nccl", "mpi"],
        help="the id of the environment")

    # self-play specs
    parser.add_argument('--save-path', type=str, default="outputs",
                            help="how many train updates between saving a new checkpoint and loading a new enemy")
    parser.add_argument('--self-play', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                            help="train by selfplay")
    parser.add_argument('--save-every', type=int, default=25,
                            help="how many train updates between saving a new checkpoint and loading a new enemy")
    parser.add_argument('--pool-size', type=int, default=5,
                            help="how many checkpoints to keep")
    parser.add_argument('--sparse-reward', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                            help="weather to use sparse reward")
    parser.add_argument('--eval-interval', type=int, default=10, 
                            help="how many updates between eval")
    parser.add_argument('--simple-obs', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                            help="how many updates between eval")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

# utility for JArrays
def init_jvm(jvmpath=None):
    if jpype.isJVMStarted():
        return
    jpype.startJVM(jpype.getDefaultJVMPath())

# utility to create Vectorized env
def make_env(seed, self_play, sparse_reward, simple_obs, device):
    def thunk():
        env = CustomLuxEnv(self_play=self_play, sparse_reward = sparse_reward, simple_obs = simple_obs, device=device, PATH_AGENT_CHECKPOINTS = PATH_AGENT_CHECKPOINTS)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# define the learning agent and utilities
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

class Transpose(nn.Module):
    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SqEx(nn.Module):
    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()
        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')
        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):
        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y

class ResBlockSqEx(nn.Module):
    def __init__(self, n_features):
        super(ResBlockSqEx, self).__init__()
        # convolutions
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False)

        # squeeze and excitation
        self.sqex  = SqEx(n_features)

    def forward(self, x):
        # convolutions
        y = self.conv1(self.relu1(x))
        y = self.conv2(self.relu2(y))
        # squeeze and excitation
        y = self.sqex(y)
        # add residuals
        y = torch.add(x, y)
        return y

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.res_block0 = ResBlockSqEx(self._out_channels)
        self.res_block1 = ResBlockSqEx(self._out_channels)

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


class Decoder(nn.Module):
    def __init__(self, output_channels):
        super().__init__()

        self.deconv = nn.Sequential(
            layer_init(nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            layer_init(nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1)),
            Transpose((0, 2, 3, 1)),
        )

        self.fc_dec = nn.Sequential(
            nn.Linear(256, 128 * 3 * 3),
        )

    def forward(self, x):
        x = self.fc_dec(x)
        x = x.view(-1, 128, 3, 3)
        return self.deconv(x)


class Agent(nn.Module):
    def __init__(self, mapsize=48 * 48):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        h, w, c = envs.single_observation_space.shape

        shape = (c, h, w)
        conv_seqs = []
        for out_channels in [32, 64, 128, 256]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
            nn.ReLU(),
        ]

        self.encoder = nn.Sequential(*conv_seqs)

        self.actor_robots = Decoder(22)
        self.actor_factories = Decoder(4)

        self.critic = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(256, 128), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),
        )
    
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
    
    def unfreeze_params(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x.permute(0,3,1,2))  # "bhwc" -> "bchw"

    def get_action(
            self,
            obs,
            robot_action=None,
            factory_action=None,
            robot_invalid_action_masks=None,
            factory_invalid_action_masks=None,
            envs_=None,
            player="player_0"
        ):

        if type(envs_) == gym.vector.sync_vector_env.SyncVectorEnv:
            robots_nvec = envs_.single_action_space["robots"].nvec
            robots_nvec_sum = envs_.single_action_space["robots"].nvec[1:].sum()
            robots_nvec_tolist = envs_.single_action_space["robots"].nvec[1:].tolist()
            factories_nvec = envs_.single_action_space["factories"].nvec
            factories_nvec_sum = envs_.single_action_space["factories"].nvec[1:].sum()
            factories_nvec_tolist = envs_.single_action_space["factories"].nvec[1:].tolist()
        
        # player_1 operates in a single custom env so there is no such thing as single_obs_space
        elif player == "player_1" or type(envs_) == VideoWrapper:
            robots_nvec = envs_.action_space["robots"].nvec
            robots_nvec_sum = envs_.action_space["robots"].nvec[1:].sum()
            robots_nvec_tolist = envs_.action_space["robots"].nvec[1:].tolist()
            factories_nvec = envs_.action_space["factories"].nvec
            factories_nvec_sum = envs_.action_space["factories"].nvec[1:].sum()
            factories_nvec_tolist = envs_.action_space["factories"].nvec[1:].tolist()
        else:
            print(type(envs_))

        x = self(obs)
        
        robot_logits = self.actor_robots(x)
        robot_grid_logits = robot_logits.reshape(-1, robots_nvec_sum)
        robot_split_logits = torch.split(robot_grid_logits, robots_nvec_tolist, dim=1)

        factory_logits = self.actor_factories(x)
        factory_grid_logits = factory_logits.reshape(-1, factories_nvec_sum)
        factory_split_logits = torch.split(factory_grid_logits, factories_nvec_tolist, dim=1)

        if robot_action is None and factory_action is None:
            factory_invalid_action_masks = torch.tensor(get_factory_invalid_action_masks(envs_, player)).to(device)
            factory_invalid_action_masks = factory_invalid_action_masks.view(-1, factory_invalid_action_masks.shape[-1])
            factory_split_invalid_action_masks = torch.split(factory_invalid_action_masks[:, 1:], factories_nvec_tolist ,dim=1)
            factory_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(factory_split_logits, factory_split_invalid_action_masks)]
            factory_action = torch.stack([categorical.sample() for categorical in factory_multi_categoricals])

            robot_invalid_action_masks = torch.tensor(get_robot_invalid_action_masks(envs_, player)).to(device)
            robot_invalid_action_masks = robot_invalid_action_masks.view(-1, robot_invalid_action_masks.shape[-1])
            robot_split_invalid_action_masks = torch.split(robot_invalid_action_masks[:, 1:], robots_nvec_tolist, dim=1)
            robot_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(robot_split_logits, robot_split_invalid_action_masks)]
            robot_action = torch.stack([categorical.sample() for categorical in robot_multi_categoricals])

            # during ROLLOUTs I can modify the logits and logprobs because they are no_grad()
            # post mask on logits and logprobs of actions that werent ilegal but are not gonna be executed
            action_types = robot_action[0]
            robot_multi_categoricals = post_categoricals(robot_multi_categoricals, action_types, robot_invalid_action_masks)
            # during TRAIN I cannot modify logits and logprobs because require grad, but if masks loaded are the POST masks
            # then the logprobs and logits should be the same as when collecting ROLLOUTS. Hence, update post_masks!
            robot_invalid_action_masks = post_masks(robot_invalid_action_masks, action_types)
            
        else:
            factory_invalid_action_masks = factory_invalid_action_masks.view(-1, factory_invalid_action_masks.shape[-1])
            factory_action = factory_action.view(-1, factory_action.shape[-1]).T
            factory_split_invalid_action_masks = torch.split(factory_invalid_action_masks[:, 1:], factories_nvec_tolist,dim=1)
            factory_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(factory_split_logits, factory_split_invalid_action_masks)]

            robot_invalid_action_masks = robot_invalid_action_masks.view(-1, robot_invalid_action_masks.shape[-1])
            robot_action = robot_action.view(-1, robot_action.shape[-1]).T
            robot_split_invalid_action_masks = torch.split(robot_invalid_action_masks[:, 1:], robots_nvec_tolist,dim=1)
            robot_multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(robot_split_logits, robot_split_invalid_action_masks)]


        robot_logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(robot_action, robot_multi_categoricals)])
        factory_logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(factory_action, factory_multi_categoricals)])

        robot_entropy = torch.stack([categorical.entropy() for categorical in robot_multi_categoricals])
        factory_entropy = torch.stack([categorical.entropy() for categorical in factory_multi_categoricals])

        robot_num_predicted_parameters = len(robots_nvec) - 1
        factory_num_predicted_parameters = len(factories_nvec) - 1
        
        robot_logprob = robot_logprob.T.view(-1, 48 * 48, robot_num_predicted_parameters)
        robot_entropy = robot_entropy.T.view(-1, 48 * 48, robot_num_predicted_parameters)
        robot_action = robot_action.T.view(-1, 48 * 48, robot_num_predicted_parameters)
        robot_invalid_action_masks = robot_invalid_action_masks.view(-1, 48 * 48, robots_nvec_sum + 1)

        factory_logprob = factory_logprob.T.view(-1, 48 * 48, factory_num_predicted_parameters)
        factory_entropy = factory_entropy.T.view(-1, 48 * 48, factory_num_predicted_parameters)
        factory_action = factory_action.T.view(-1, 48 * 48, factory_num_predicted_parameters)
        factory_invalid_action_masks = factory_invalid_action_masks.view(-1, 48 * 48, factories_nvec_sum + 1)

        robot_logprob = robot_logprob.sum(1).sum(1)
        factory_logprob = factory_logprob.sum(1).sum(1)
        logprob = factory_logprob + robot_logprob

        robot_entropy = robot_entropy.sum(1).sum(1)
        factory_entropy = factory_entropy.sum(1).sum(1)
        entropy = factory_entropy + robot_entropy

        return robot_action, factory_action, logprob, entropy, robot_invalid_action_masks, factory_invalid_action_masks

    def get_value(self, x):
        return self.critic(self.forward(x))


if __name__ == "__main__":
    # torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py
    # taken from https://pytorch.org/docs/stable/elastic/run.html
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    args = parse_args()
    args.world_size = world_size
    args.num_envs = int(args.num_envs / world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    if world_size > 1:
        # set the port number
        os.environ['MASTER_PORT'] = '29401'
        dist.init_process_group(args.backend, rank=local_rank, world_size=world_size)
    else:
        warnings.warn(
            """
Not using distributed mode!
If you want to use distributed mode, please execute this script with 'torchrun'.
E.g., `torchrun --standalone --nnodes=1 --nproc_per_node=2 ppo_atari_multigpu.py`
        """
        )
    print(f"================================")
    print(f"args.num_envs: {args.num_envs}, args.batch_size: {args.batch_size}, args.minibatch_size: {args.minibatch_size}")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = None

    PATH_AGENT_CHECKPOINTS = "/home/mila/r/roger.creus-castanyer/lux-ai-rl/src/checkpoints_gridnet_post_long"

    if local_rank == 0:
        if args.track:
            import wandb

            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        if not os.path.exists(PATH_AGENT_CHECKPOINTS):
            os.makedirs(PATH_AGENT_CHECKPOINTS)

    # TRY NOT TO MODIFY: seeding
    # CRUCIAL: note that we needed to pass a different seed for each data parallelism worker
    args.seed += local_rank
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed - local_rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False

    if len(args.device_ids) > 0:
        device = torch.device(f"cuda:{args.device_ids[local_rank]}" if torch.cuda.is_available() and args.cuda else "cpu")
    else:
        device_count = torch.cuda.device_count()
        if device_count < world_size:
            device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        else:
            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    init_jvm()

    print("local thread", local_rank, "has device", device, "Java is started:", jpype.isJVMStarted())

    envs = gym.vector.SyncVectorEnv([make_env(i + args.seed, args.self_play, args.sparse_reward, args.simple_obs, device) for i in range(args.num_envs)])

    agent = Agent(envs).to(device)

    if local_rank == 0:

        # start self-play
        num_models_saved = 0
        agent.save_checkpoint(num_models_saved, args.pool_size, PATH_AGENT_CHECKPOINTS)
        num_models_saved += 1

    torch.manual_seed(args.seed)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # all threads will wait for jvm to be initialized and initial checkpoint to exist
    if world_size > 1:
        dist.barrier()

    for i in range(len(envs.envs)):
        enemy_agent = Agent().to(device)
        enemy_agent.freeze_params()
        envs.envs[i].set_enemy_agent(enemy_agent)


    # ALGO Logic: Storage for epoch data
    mapsize = 48 * 48

    robot_action_space_shape = (mapsize, envs.single_action_space["robots"].shape[0] - 1)
    robot_invalid_action_shape = (mapsize, envs.single_action_space["robots"].nvec[1:].sum() + 1)

    factory_action_space_shape = (mapsize, envs.single_action_space["factories"].shape[0] - 1)
    factory_invalid_action_shape = (mapsize, envs.single_action_space["factories"].nvec[1:].sum() + 1)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    robot_actions = torch.zeros((args.num_steps, args.num_envs) + robot_action_space_shape).to(device)
    factory_actions = torch.zeros((args.num_steps, args.num_envs) + factory_action_space_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    robot_invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + robot_invalid_action_shape).to(device)
    factory_invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + factory_invalid_action_shape).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // (args.batch_size * world_size)

    if local_rank == 0:
        print("Model's state_dict:")
        for param_tensor in agent.state_dict():
            print(param_tensor, "\t", agent.state_dict()[param_tensor].size())
        total_params = sum([param.nelement() for param in agent.parameters()])
        print("Model's total parameters:", total_params)

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs * world_size
            
            if global_step > 1000000000:
                for i in range(len(envs.envs)):
                    envs.envs[i].set_sparse_reward()
                    args.gamma = 1
                    args.gae_lambda = 1

            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                values[step] = agent.get_value(obs[step]).flatten()
                robot_action, factory_action, logproba, _, robot_invalid_action_masks[step], factory_invalid_action_masks[step] = agent.get_action(obs[step], envs_=envs)
            
            robot_actions[step] = robot_action
            factory_actions[step] = factory_action
            logprobs[step] = logproba

            robot_real_action = torch.cat([
            torch.stack(
                    [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
            ]).unsqueeze(2), robot_action], 2)
            
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

            factory_real_action = torch.cat([
                torch.stack(
                    [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
            ]).unsqueeze(2), factory_action], 2)
            
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

            # TRY NOT TO MODIFY: execute the game and log data.
            try:
                next_obs, rs, ds, infos = envs.step(actions)
                next_obs = torch.Tensor(next_obs).to(device)
            except Exception as e:
                e.printStackTrace()
                raise

            rewards[step], next_done = torch.Tensor(rs).to(device), torch.Tensor(ds).to(device)

            for info in infos:
                if "episode" in info.keys() and local_rank == 0:
                    print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                    writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                    writer.add_scalar("charts/episode_length", info['episode']['l'], global_step)
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
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
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
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
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

        # Optimizing the policy and value network
        inds = np.arange(args.batch_size, )
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = inds[start:end]

                _, _, newlogproba, entropy, _, _ = agent.get_action(
                    b_obs[mb_inds],
                    b_robot_actions.long()[mb_inds],
                    b_factory_actions.long()[mb_inds],
                    b_robot_invalid_action_masks.long()[mb_inds],
                    b_factory_invalid_action_masks.long()[mb_inds],
                    envs
                )

                ratio = (newlogproba - b_logprobs[mb_inds]).exp()

                with torch.no_grad():
                    approx_kl = (b_logprobs[mb_inds] - newlogproba).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_values = agent.get_value(b_obs[mb_inds]).view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = ((new_values - b_returns[mb_inds]) ** 2)
                    v_clipped = b_values[mb_inds] + torch.clamp(new_values - b_values[mb_inds], -args.clip_coef,
                                                                    args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_values - b_returns[mb_inds]) ** 2)

                entropy_loss = entropy.mean()

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()

                if world_size > 1:
                    # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
                    all_grads_list = []
                    for param in agent.parameters():
                        if param.grad is not None:
                            all_grads_list.append(param.grad.view(-1))
                    all_grads = torch.cat(all_grads_list)
                    dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
                    offset = 0
                    for param in agent.parameters():
                        if param.grad is not None:
                            param.grad.data.copy_(
                                all_grads[offset : offset + param.numel()].view_as(param.grad.data) / world_size
                            )
                            offset += param.numel()

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        if local_rank == 0:
            if update % args.save_every == 0:
                agent.save_checkpoint(num_models_saved, args.pool_size, PATH_AGENT_CHECKPOINTS)
                num_models_saved += 1

            # Evaluation!
            if update % args.eval_interval == 0:
                agent.freeze_params()
                envs_test = make_eval_env(np.random.randint(1000) + update, args.self_play, args.sparse_reward, args.simple_obs, device, PATH_AGENT_CHECKPOINTS)
                envs_test.set_enemy_agent(agent)
                envs_test = VideoWrapper(envs_test, update_freq=1)
                mean_reward = []

                for ep in range(2):
                    eval_obs = envs_test.reset(load_new_enemy=False)
                    eval_done = False
                    total_eval_reward = 0

                    while not eval_done:
                        with torch.no_grad():
                            eval_robot_action, eval_factory_action, _, _, eval_robot_invalid_action_masks, eval_factory_invalid_action_masks = agent.get_action(torch.Tensor(np.array(eval_obs)).unsqueeze(0).to(device), envs_=envs_test) 
                            
                            eval_robot_real_action = torch.cat([
                            torch.stack(
                                [torch.arange(0, 48*48, device=device) for i in range(1)
                            ]).unsqueeze(2), eval_robot_action], 2)
                            
                            eval_robot_real_action = eval_robot_real_action.cpu().numpy()
                            eval_robot_valid_actions = eval_robot_real_action[eval_robot_invalid_action_masks[:,:,0].bool().cpu().numpy()]
                            eval_robot_valid_actions_counts = eval_robot_invalid_action_masks[:,:,0].sum(1).long().cpu().numpy()
                            eval_robot_java_valid_actions = []
                            eval_robot_valid_action_idx = 0
                            for env_idx, eval_robot_valid_action_count in enumerate(eval_robot_valid_actions_counts):
                                eval_robot_java_valid_action = []
                                for c in range(eval_robot_valid_action_count):
                                    eval_robot_java_valid_action += [JArray(JInt)(eval_robot_valid_actions[eval_robot_valid_action_idx])]
                                    eval_robot_valid_action_idx += 1
                                eval_robot_java_valid_actions += [JArray(JArray(JInt))(eval_robot_java_valid_action)]
                            eval_robot_java_valid_actions = JArray(JArray(JArray(JInt)))(eval_robot_java_valid_actions)

                            eval_factory_real_action = torch.cat([
                                torch.stack(
                                    [torch.arange(0, 48*48, device=device) for i in range(1)
                            ]).unsqueeze(2), eval_factory_action], 2)
                            
                            eval_factory_real_action = eval_factory_real_action.cpu().numpy()
                            eval_factory_valid_actions = eval_factory_real_action[eval_factory_invalid_action_masks[:,:,0].bool().cpu().numpy()]
                            eval_factory_valid_actions_counts = eval_factory_invalid_action_masks[:,:,0].sum(1).long().cpu().numpy()
                            eval_factory_java_valid_actions = []
                            eval_factory_valid_action_idx = 0
                            for env_idx, eval_factory_valid_action_count in enumerate(eval_factory_valid_actions_counts):
                                eval_factory_java_valid_action = []
                                for c in range(eval_factory_valid_action_count):
                                    eval_factory_java_valid_action += [JArray(JInt)(eval_factory_valid_actions[eval_factory_valid_action_idx])]
                                    eval_factory_valid_action_idx += 1
                                eval_factory_java_valid_actions += [JArray(JArray(JInt))(eval_factory_java_valid_action)]
                            eval_factory_java_valid_actions = JArray(JArray(JArray(JInt)))(eval_factory_java_valid_actions)
                            
                            eval_robot_valid_actions = np.array(eval_robot_java_valid_actions, dtype=object)
                            eval_factory_valid_actions = np.array([np.array(xi) for xi in eval_factory_java_valid_actions], dtype=object)

                            eval_actions =  {
                                "factories" : eval_factory_valid_actions[0],
                                "robots" : eval_robot_valid_actions[0]
                            }
                            
                            eval_obs, eval_reward, eval_done, _ = envs_test.step(eval_actions)
                            total_eval_reward += eval_reward
                
                    print("Evaluated the agent and got reward: " + str(total_eval_reward))
                    mean_reward.append(total_eval_reward)

                wandb.log({
                    "eval/reward" : np.mean(mean_reward)
                })

                envs_test.send_wandb_video()
                envs_test.close()
                agent.unfreeze_params()    

        if world_size > 1:
            dist.barrier()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if local_rank == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    if local_rank == 0:
        writer.close()
        if args.track:
            wandb.finish()