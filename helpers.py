import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

def sample_trajectories(envs, agent, tmax, initial_random_steps = 5, skip_frames = 0, with_images=False):
    n=len(envs.ps)
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    states=[]
    rewards=[]
    actions=[]
    probs=[]

    '''
    input : 
    envs : parallelEnv instance
    agent : must implement 'act'
    tmax : maximum trajectory length
    '''
    envs.reset()
    for _ in range(initial_random_steps):
        s, r, done, _ = envs.step(np.random.choice(np.arange(envs.action_space.n), n))
        for skip in range(skip_frames): s, r, done, _ = envs.step([0]* n) #wait a frame

    for t in range(tmax):
        # convert observations to torch
        prev_s = s
        s = np.asarray(s)
        s = torch.from_numpy(s).float().to(device)

        # get an action
        a = agent.act(s)
        
        # step env
        s, r_t, done,  _ = envs.step(a)
        r = r_t
        for skip in range(skip_frames): #skip frames, do nothing
            if done.any(): break
            s, r_t, done, _ = envs.step([0] * n)
            r += r_t

        states.append(prev_s)
        actions.append(a)
        rewards.append(r)
        #probs.append(p)
        
        if done.any():
            break #the max length will be the shortest one

    return states, actions, rewards
    
