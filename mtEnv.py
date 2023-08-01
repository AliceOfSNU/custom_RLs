
import sys
sys.path.append("..")

import gymnasium
import metaworld
import random
import numpy as np
import torch
import metaworld.envs.mujoco.env_dict as env_dict
from gymnasium.wrappers import TimeLimit
from gymnasium import spaces

# wrapper class around MtEnv
class MtEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    '''
    Box([-0.525   0.348  -0.0525 -1.        -inf    -inf    -inf    -inf    -inf
        -inf    -inf    -inf    -inf    -inf    -inf    -inf    -inf    -inf
        -0.525   0.348  -0.0525 -1.        -inf    -inf    -inf    -inf    -inf
            -inf    -inf    -inf    -inf    -inf    -inf    -inf    -inf    -inf
        0.      0.      0.    ], 
        [0.525 1.025 0.7   1.      inf   inf   inf   inf   inf   inf   inf   inf
        inf   inf   inf   inf   inf   inf 0.525 1.025 0.7   1.      inf   inf
        inf   inf   inf   inf   inf   inf   inf   inf   inf   inf   inf   inf
        0.    0.    0.   ], (39,), float64)
    Box(-1.0, 1.0, (4,), float64)
    '''
    def __init__(self, env_name:str, seed:int = 0):
        super(MtEnv, self).__init__()
        mt = metaworld.MT1(env_name, seed = seed)
        base_env = mt.train_classes[env_name]()
        task = random.choice(mt.train_tasks)
        #print(base_env._get_pos_goal()) # debugging.. if we have differet seeds
        base_env.set_task(task)
        self.base_env = base_env
        self.mt = mt
        self.action_space = spaces.Box(base_env.action_space.low, base_env.action_space.high, base_env.action_space.shape, base_env.action_space.dtype)
        self.observation_space = spaces.Box(base_env.observation_space.low, base_env.observation_space.high, base_env.observation_space.shape, base_env.observation_space.dtype)
        self.max_steps_mujoco = 500
        self.n_steps = 0


    def step(self, action):
        state, reward, done, _, info = self.base_env.step(action)
        #self.n_steps += 1
        truncated = False
        #if self.n_steps >= self.max_steps_mujoco:
        #    truncated = True
        #    self.n_steps = 0

        info["is_success"] = info["success"]
        # Should I be doing this?: NO!
        #if info["success"] > 0.1 or done:
        #    done = True
        return (state, reward, done, truncated, info)
    
    def seed(self, seed):
        self.base_env.seed(seed)

    def reset(self, seed= None, options=None):
        # resettinng will change the goal position
        # task = random.choice(self.mt.train_tasks)
        # self.base_env.set_task(task)
        # self.n_steps = 0
        if seed is not None: self.seed(seed)
        return self.base_env.reset()
    
    def render(self, mode):
        return self.base_env.render(mode)

    def close(self):
        self.base_env.close()

    
class ClsOneHotWrapper(gymnasium.ObservationWrapper):
    def __init__(self, mtenv:gymnasium.Env, cls_id, total_num_envs):
        super().__init__(mtenv)
        self.observation_space = spaces.Box(np.concatenate([mtenv.observation_space.low, [0.]*total_num_envs]), np.concatenate([mtenv.observation_space.high,[1.]*total_num_envs]))
        self.cls_id = cls_id
        self.cls_onehot = np.zeros(total_num_envs)
        self.cls_onehot[cls_id] = 1.0

    def observation(self, obs):
        return np.concatenate([obs, self.cls_onehot])

    

    
class GymWrapper(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def __init__(self, mtenv, seed:int = 0, **kwargs):
        super().__init__()
        self.env:metaworld.MetaWorldEnv = mtenv
        self.action_space = spaces.Box(mtenv.action_space.low, mtenv.action_space.high, dtype= mtenv.action_space.dtype)
        self.observation_space = spaces.Box(mtenv.observation_space.low, mtenv.observation_space.high, dtype=  mtenv.observation_space.dtype)
        
    def step(self, action):
        obs, reward, terminated, info =  self.env.step(action)
        return obs, reward, terminated, False, info
    
    def seed(self, seed):
        self.env.seed(seed)

    def reset(self, **kwargs):
        # resettinng will change the goal position
        # task = random.choice(self.mt.train_tasks)
        # self.base_env.set_task(task)
        # self.n_steps = 0
        if "seed" in kwargs.keys() and kwargs["seed"] is not None:
            self.seed(kwargs["seed"])
        obs = self.env.reset()
        return (obs, None)
    
    def render(self, mode):
        return self.env.render(mode)

    def close(self):
        self.env.close()
    
