import torch
import itertools
from parallelEnv import parallelEnv, VecEnv
from copy import deepcopy
from torch import nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from torch.distributions.normal import Normal
from SimpleReplay import ReplayBuffer

class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[64, 64], activation=nn.ReLU,
                 log_std_limits=(-20, 2), action_scale:float = 1.0):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_limits
        self.action_scale = action_scale

        dims = [obs_dim] + hidden_sizes
        self.shared_layers = nn.ModuleList([nn.Linear(in_features, out_features) for (in_features, out_features) in zip(dims[:-1], dims[1:])])
        self.mu_FC = nn.Linear(dims[-1], action_dim)
        self.log_std_FC = nn.Linear(dims[-1], action_dim)
        self.activation = activation()

    def forward(self, x:torch.Tensor, determininstic:bool = False):
        '''
        INPUT: observations (torch.Tensor(batch_size, obs_dim))
        OUTPUT: 
            actions (torch.Tensor(batch_size, action_dim))
            log_probs (torch.Tensor(batch_size, action_dim))
        '''
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        mu = self.mu_FC(x)
        log_std = self.log_std_FC(x)
        info = {"mu" : mu, "log_std": log_std}

        if determininstic: #no sampling
            return mu, None, info
        
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        pi_dist = Normal(mu, std)
        
        action = pi_dist.rsample()
        # refer to original paper for this eq
        log_prob = pi_dist.log_prob(action).sum(axis=-1)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2*action))).sum(axis=1)

        action = torch.tanh(action)
        action *= self.action_scale

        return action, log_prob, info

class ModularSoftGatedSquashedGaussianActor(nn.Module):
    def __init__(self, 
                 obs_dim, 
                 em_input_dim,
                 em_hidden_sizes,
                 base_hidden_sizes,
                 module_hidden_size:int,
                 num_layers,
                 num_modules,
                 gating_hidden_size:int,
                 action_dim,  
                 activation = nn.ReLU,
                 log_std_limits=(-20, 2), 
                 action_scale:float = 1.0):
        super().__init__()
        self.log_std_min, self.log_std_max = log_std_limits
        self.action_scale = action_scale

        #others
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.activation = activation()
        
        assert(base_hidden_sizes[-1] == em_hidden_sizes[-1], "last dimension of base network and embedding network must match")
        feature_dim = base_hidden_sizes[-1]
        
        # base network
        dims = [obs_dim] + base_hidden_sizes
        self.base_network = nn.ModuleList([nn.Linear(in_features, out_features) for (in_features, out_features) in zip(dims[:-1], dims[1:])])
        
        # embedding network
        dims = [em_input_dim] + em_hidden_sizes
        self.embedding = nn.ModuleList([nn.Linear(in_features, out_features) for (in_features, out_features) in zip (dims[:-1], dims[1:])])
        
        # layered structure
        self.layers = []
        for i in range(num_layers):
            modules = []
            for j in range(num_modules):
                FC = nn.Linear(feature_dim, module_hidden_size)
                modules.append(FC)
                self.__setattr__("module_{}_{}".format(i,j), FC)
            self.layers.append(modules)
            feature_dim = module_hidden_size
            
        # output network
        self.output_layer = nn.Linear(feature_dim, action_dim * 2) #mu and std

        # gating network
        feature_dim = em_hidden_sizes[-1]
        self.gating_fcs = []
        for i in range(num_layers):
            gating_FC = nn.Linear(feature_dim, gating_hidden_size)
            self.gating_fcs.append(gating_FC)
            self.__setattr__("gating_fc_{}".format(i), gating_FC)
            feature_dim = gating_hidden_size

        # gating probability(weight) network
        # outputs weight(logit) p(i,j) from layer l to l+1
        self.gating_weight_fcs = []
        for l in range(num_layers-2):
            gating_weight_FC = nn.Linear(feature_dim, num_modules*num_modules)
            self.gating_weight_fcs.append(gating_weight_FC)
            self.__setattr__("gating_weight_fc_{}".format(l+1), gating_weight_FC)

        self.gating_weight_last = nn.Linear(feature_dim, num_modules)
        
    def forward(self, obs, embedding):
        
        #should return action(torch.Tensor), lob_prob, info(mu, std)
        pass

class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=[64, 64], activation=nn.ReLU):
        super().__init__()
        dims = [obs_dim+action_dim] + hidden_sizes + [1]
        self.layers = nn.ModuleList([nn.Linear(in_features, out_features) for (in_features, out_features) in zip(dims[:-1], dims[1:])])
        self.activation = activation()

    def forward(self, obs:torch.Tensor, action:torch.Tensor):
        x = torch.cat([obs, action], dim = -1)
        for layer in self.layers:
            x = self.activation(layer(x))
        # last dim in shape [1] so we want to unpack
        return x.squeeze(dim=-1)

class SAC(nn.Module):

    # refer to SpinningUp docs for the default numbers
    def __init__(self, obs_dim, action_dim, hidden_sizes, task_fn = None, actor=SquashedGaussianActor, critic=DoubleQCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        learning_starts=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, no_gpu=True):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env = task_fn()
        self.is_parallel_env = isinstance(self.env, VecEnv)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.replay = ReplayBuffer([obs_dim], [action_dim], replay_size)
        self.max_ep_len = max_ep_len

        #actor
        self.actor = actor(obs_dim, action_dim, hidden_sizes)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        # critic
        self.q1 = critic(obs_dim, action_dim, hidden_sizes)
        self.q2 = critic(obs_dim, action_dim, hidden_sizes)
        self.q_optim = optim.Adam(itertools.chain(self.q1.parameters(), self.q2.parameters()), lr=lr)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self.polyak = polyak
        for q in [self.q1_target, self.q2_target]:
            for p in q.parameters():
                p.requires_grad = False

        # training parameters
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.learning_starts = learning_starts
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not no_gpu else "cpu")


    def act(self, obs, deterministic = False):
        with torch.no_grad():
            actions, log_prob, _ = self.actor(obs, deterministic)
            return actions.numpy()
        
    def collect_rollout(self):
        obs, _ = self.env.reset()
        if not self.is_parallel_env:
            obs = [obs]
        for t in range(self.max_ep_len):
            with torch.no_grad():
                obs = torch.Tensor(obs, device=self.device) #torch.Tensor(n_envs, obs_dim)
                actions = self.act(obs)
                if not self.is_parallel_env:
                    actions = actions[0]
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            if not self.is_parallel_env:
                next_obs = [next_obs]; rewards = [rewards]; terminated = [terminated]
                truncated = [truncated]; info = [info]
            for obs, act, rew, nobs, ter in zip(obs, actions, rewards, next_obs, terminated):
                self.replay.store(obs, act, rew, nobs, ter)
            obs = next_obs
        if self.is_parallel_env:
            return self.max_ep_len * self.env.nenvs
        else:
            return self.max_ep_len
    
    def calculate_qloss(self, data):
        # y target
        q1 = self.q1(data["obs"], data["actions"])
        q2 = self.q2(data["obs"], data["actions"])
        #backups
        with torch.no_grad():
            # q_backup = r * (1-d)*gamma*Q(s', a*) where a* is from curruent policy
            actions, log_probs, info = self.actor(data["obs"]) #actions: torch.Tensor(batch_size, action_dim)
            q1_targets = self.q1_target(data["obs2"], actions)
            q2_targets = self.q2_target(data["obs2"], actions)
            q_targets = torch.min(q1_targets, q2_targets, dim=-1)
            ys = data["rewards"] + (1-data["dones"]) * self.gamma *(q_targets - self.alpha*log_probs)
        
        # loss calculation
        loss_q1 = (0.5 * (q1 - ys)**2).mean()
        loss_q2 = (0.5 * (q2 - ys)**2).mean()
        qloss = loss_q1 + loss_q2

        return qloss

    def calculate_actor_loss(self, data):
        actions, log_probs, info = self.actor(data["obs"])
        q1 = self.q1(data["obs"], actions)
        q2 = self.q2(data["obs"], actions)
        q = torch.min(q1, q2)
        # loss calculation
        # actor loss = Q(s, a*) + alpha*H(pi(*|s)) = Q(s, a*)-alpha*log(pi(a*|s))

        
        actor_loss = (self.alpha * log_probs - q).mean #gradient ascent
        return actor_loss

    def single_update(self, data):
        self.q_optim.zero_grad()
        q_loss = self.calculate_qloss(data)
        q_loss.backward()
        q_norm = torch.nn.utils.clip_grad(itertools.chain(self.q1.parameters(), self.q2.parameters()), 10)
        self.q_optim.step()

        self.actor_optim.zero_grad()
        actor_loss = self.calculate_actor_loss(data)
        actor_loss.backward()
        actor_norm = torch.nn.utils.clip_grad(self.actor.parameters(), 10)
        self.actor_optim.step()       

        #update target q networks
        with torch.no_grad():
            currs = [self.q1.parameters(), self.q2.parameters()]
            targets = [self.q1_target.parameters(), self.q2_target.parameters()]
            for curr, target in zip(currs, targets):
                #copy-less update.. or.. copy_ should be faster?
                target.data._mul(self.polyak)
                target.data._add(curr.data * (1-self.polyak))
                    
    def train(self):
        ep, total_timesteps = 0
        while ep < self.epochs:
            # collect trajs
            total_timesteps += self.collect_rollout()
            
            # do gradient updates
            if total_timesteps >= self.learning_starts and total_timesteps % self.update_every == 0:
                print("updating - total_timesteps: ", total_timesteps)
                data = self.replay.sample_batch(self.batch_size)
                for step in range(self.steps_per_epoch):
                    self.single_update(data)
                
            # log epoch
            ep += 1
            print("epoch ", ep)
    
    def evaluate(self):
        pass