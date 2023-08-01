# taken from openai/baseline
# with minor edits
# see https://github.com/openai/baselines/baselines/common/vec_env/subproc_vec_env.py
# 


import numpy as np
import gym
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from mtEnv import GymWrapper
from mtEnv import ClsOneHotWrapper
import metaworld
import random
from gymnasium.wrappers import TimeLimit

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        #logger.warn('Render not defined for %s' % self)
        pass
        
    @property
    def unwrapped(self):
        return self


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, truncated, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, truncated, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

# collects experiences.. puts into a replay buffer
class ParallelCollector(VecEnv):
    def __init__(self, mt:metaworld.Benchmark, env_names, workers_per_env=4):
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_names)
        self.workers_per_env = workers_per_env
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs * self.workers_per_env)])
        
        envs = []
        mt_cls_cnt = len(env_names)
        for i, env_name in enumerate(env_names):
            env_cls = mt.train_classes[env_name]
            task_list = [task for task in mt.train_tasks if task.env_name == env_name]
            for _ in range(workers_per_env): 
                env = env_cls()      
                task = random.choice(task_list)
                env.set_task(task)
                env = GymWrapper(env, {"env_name": env_name})
                env = TimeLimit(env, 500)
                env = ClsOneHotWrapper(env, i, mt_cls_cnt)
                envs.append(env)
            
        assert(len(envs) == self.nenvs * self.workers_per_env)
        
        #create 'workers_per_env' number of workers per env.
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env)))
            for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        print("started processes: ", [p.pid for p in self.ps])
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(envs), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        try:
            results = []
            for remote in self.remotes:
                x = remote.recv()
                results.append(x)
        except EOFError:
            self.exit_error()
            raise Exception("subprocess logic error")
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        if np.stack(dones).any():
            obs = [ob[0] if dones[i] else ob for i, ob in enumerate(obs)]
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def exit_error(self):
        print("FAIL: kill all processes")
        for remote in self.remotes:
            remote.close()
        for p in self.ps: p.kill()
        self.waiting = False
        self.closed = True
        
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def collect_trajectories(actor, replay_buffer, episodes_per_env:int):
        # for each of the tasks, collect episodes and add to replay buffer
        pass

class parallelEnv(VecEnv):
    def __init__(self, env_name=None, env_fn=None,
                 n=4, seed=None,
                 spaces=None):
        
        if env_name is not None: #create from registry
            env_fns = [ gym.make(env_name) for _ in range(n) ]
        elif env_fn is not None: #create from function
            env_fns = [ env_fn() for _ in range(n)]
        if seed is not None:
            for i,e in enumerate(env_fns):
                e.reset(seed=i+seed)
        
        """
        envs: list of gym environments to run in subprocesses
        adopted from openai baseline
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        print("started processes: ", [p.pid for p in self.ps])
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        try:
            results = []
            for remote in self.remotes:
                x = remote.recv()
                results.append(x)
        except EOFError:
            self.exit_error()
            raise Exception("subprocess logic error")
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        if np.stack(dones).any():
            obs = [ob[0] if dones[i] else ob for i, ob in enumerate(obs)]
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def exit_error(self):
        print("FAIL: kill all processes")
        for remote in self.remotes:
            remote.close()
        for p in self.ps: p.kill()
        self.waiting = False
        self.closed = True
        
    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True