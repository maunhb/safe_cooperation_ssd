import numpy as np
import random 
from ray.rllib.env import MultiAgentEnv
from social_dilemmas.envs.onehot_state_matrix_agent import MatrixAgent

class CoopMatrixEnv(MultiAgentEnv):
    def __init__(self, game, cooperation_level=0.5):
        ''' Cooperation level x in [0,1]. '''
        self.num_agents = 4
        self.agents = {}
        self.game = game
        self.setup_agents()
        self.epsilon = random.randint(0,10)
        self.cooperation_level = cooperation_level
        self.last_reward = np.ones(10) 
        self.discount = [0.9**i for i in range(10)]
        self.timestep = 0

    @property
    def observation_space(self):
        agents = list(self.agents.values())
        return agents[0].observation_space

    @property
    def action_space(self):
        agents = list(self.agents.values())
        return agents[0].action_space

    def setup_agents(self):
        """Construct all the agents for the environment"""
        for i in range(self.num_agents):
            agent_id = 'agent-' + str(i)
            agent = MatrixAgent(agent_id, self.game)
            self.agents[agent_id] = agent

    def step(self, actions):
        """Takes in a dict of actions and performs a stage game.

        Parameters
        ----------
        actions: dict {agent-id: float}
            dict of actions, keyed by agent-id that are passed to the agent. 

        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """
        self.timestep += 1 

        observations = {"agent-1": 0, 
                        "agent-2": 0,
                        "agent-3": 0}
        observations["agent-0"] = self.epsilon
        
        rewards = {}
        dones = {}
        info = {}
        agent_0, agent_1, agent_2, agent_3 = self.agents.values()
        played_actions = {}
        p = float(actions["agent-0"])
        played_actions["agent-0"] = np.random.choice([0,1], p=[1.0-p, p])
        
        if random.random() > self.epsilon/10 :
            # playing adversarial opponent 
            p = float(actions["agent-1"])
            played_actions["agent-1"] = np.random.choice([0,1], p=[1.0-p, p])

            r_0 = agent_0.compute_reward(played_actions)
            r_1 = agent_1.compute_reward(played_actions)
            rewards["agent-0"] = r_0
            rewards["agent-1"] = - r_0
            rewards["agent-2"] = 0
            rewards["agent-3"] = 0

        else:
            above_value = np.where(self.last_reward>self.agents["agent-1"].value(), 
                                                                                1, 0)
            weighted_num_above_value = np.average(above_value, weights=self.discount)
            if weighted_num_above_value < self.cooperation_level:
                ## be selfish if they are selfish
                p = float(actions["agent-2"])
                played_actions["agent-1"] = np.random.choice([0,1], p=[1.0-p, p])

                r_0  = agent_0.compute_reward(played_actions)
                r_1  = agent_1.compute_reward(played_actions)

                rewards["agent-0"] = r_0
                rewards["agent-1"] = 0
                rewards["agent-2"] = r_1 # -r_0 # agent-2 is selfish (D) opponent
                rewards["agent-3"] = 0
            else:
                ## be cooperative if they are cooperative 
                p = float(actions["agent-3"])
                played_actions["agent-1"] = np.random.choice([0,1], p=[1.0-p, p])
  
                r_0  = agent_0.compute_reward(played_actions)
                r_1  = agent_1.compute_reward(played_actions)

                rewards["agent-0"] = r_0
                rewards["agent-1"] = 0 
                rewards["agent-2"] = 0 
                rewards["agent-3"] = r_0 + r_1 ## agent-3 is (C) opponent 

            self.last_reward = np.concatenate(([r_1], self.last_reward[:-1]))

        dones["agent-0"] = agent_0.get_done()
        dones["agent-1"] = agent_1.get_done()
        dones["agent-2"] = agent_2.get_done()
        dones["agent-3"] = agent_3.get_done()
        dones["__all__"] = np.any(list(dones.values()))

        info["agent-0"] = {}
        info["agent-0"][0.0] = self.epsilon

        return observations, rewards, dones, info

    def reset(self):
        """Reset the environment.

        This method is performed in between rollouts. 
        Returns
        -------
        observation: dict of numpy ndarray
            the initial observation of the space. The initial reward is assumed
            to be zero.
        """
        self.agents = {}
        self.setup_agents()
        self.last_reward = np.ones(10)
        self.epsilon = random.randint(0,10)
        self.timestep = 0
        observations = {}
        for agent in self.agents.values():
            observations[agent.agent_id] = 0 

        observations["agent-0"] = self.epsilon
        return observations
