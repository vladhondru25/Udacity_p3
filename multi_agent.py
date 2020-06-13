import numpy as np
import random
import copy
from collections import namedtuple, deque

from models import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from ddpg_agent import Agent


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MultiAgent():
    """The pseudo-agent that controls the other agents."""

    def __init__(self, state_size, action_size, num_agents, random_seed, config):
        """Initialize the MultAgent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.agents = [Agent(state_size, action_size, num_agents, random_seed, config) for _ in range(num_agents)]
        
        self.num_agents  = num_agents
        self.action_size = action_size
        
        # Replay memory
        self.batch_size = config["batch_size"]
        self.memory = ReplayBuffer(action_size, config["buffer_size"], self.batch_size, random_seed, num_agents)
        
     
    def act(self, states):
        """Returns actions for given state as per current policy."""
        actions = np.zeros((self.num_agents,self.action_size))
        
        for agent_idx in range(self.num_agents):
            actions[agent_idx,:] = self.agents[agent_idx].act(states[agent_idx])
        
        return actions

    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)
        
        if len(self.memory) > self.batch_size:
            for agent in self.agents:
                agent.step(self.memory.sample())


    def reset(self):
        for agent in self.agents:
            agent.reset()


    def learn(self, experiences, gamma, agent_number):
        pass
        
        
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
           
        
    def save_weights(self):
        for agent_idx in range(self.num_agents):
            torch.save(self.agents[agent_idx].critic_local.state_dict(),  'critic{}.pth'.format(agent_idx))
            torch.save(self.agents[agent_idx].critic_target.state_dict(), 'critic_target{}.pth'.format(agent_idx))
            torch.save(self.agents[agent_idx].actor_local.state_dict(),   'actor{}.pth'.format(agent_idx))
            torch.save(self.agents[agent_idx].actor_target.state_dict(),  'actor_target{}.pth'.format(agent_idx))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, num_agents):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.num_agents = num_agents

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = [torch.from_numpy(np.vstack([e.state[agent] for e in experiences if e is not None])).float().to(device) for agent in range(self.num_agents)]
        actions = [torch.from_numpy(np.vstack([e.action[agent] for e in experiences if e is not None])).float().to(device) for agent in range(self.num_agents)]
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = [torch.from_numpy(np.vstack([e.next_state[agent] for e in experiences if e is not None])).float().to(device) for agent in range(self.num_agents)]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
