import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from networks import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class a2c_agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(a2c_agent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = 64
        self.eps = 0.5
        self.shared_network = SharedNetwork(state_dim)
        self.critic_network = CriticNetwork(self.feature_dim)
        self.actor_network = ActorNetwork(self.feature_dim, self.action_dim)
        
        self.shared_params = list(self.shared_network.parameters())
        self.critic_params = list(self.critic_network.parameters())
        self.actor_params = list(self.actor_network.parameters())

        self.nets = [self.shared_network, self.critic_network, self.actor_network]
        self.to(device)

    def forward(self, state: torch.Tensor):
        '''
        receives state and outputs actions(A) and value predictions(V hats)
        
        INPUT:
            states = torch.Tensor(batch first)
        OUTPUT:
            dict(actions, log_probs, entropy, values)
        '''
        
        # pass through the nets.
        features = self.shared_network(state)
        value_pred = self.critic_network(features)
        logits_pred = self.actor_network(features)

        result = {}
        # Sampling action
        dist = torch.distributions.Categorical(logits=logits_pred)
        result["actions"] = dist.sample()
        result["log_probs"] = dist.log_prob(result["actions"])
        result["entropy"] = dist.entropy()
        result["values"] = value_pred.squeeze()
        return result

    def save(self, filename):
        torch.save(self.state_dict(), '%s.model' % (filename))

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)
