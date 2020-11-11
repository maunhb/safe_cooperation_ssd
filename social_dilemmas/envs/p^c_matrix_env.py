import numpy as np
from ray.rllib.env import MultiAgentEnv
from social_dilemmas.envs.matrix_agent import MatrixAgent

class CoopMatrixEnv(MultiAgentEnv):

    def __init__(self, game, cooperation_level=0.5):
        self.num_agents = 3

        self.agents = {}
        self.game = game
        self.setup_agents()

        self.last_cooperation = cooperation_level
        self.cooperation_level = cooperation_level

        self.timestep = 0
        self.last_reward = np.ones(10)

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
        self.timestep += 1
        played_actions = {}
        p = float(actions["agent-0"])
        played_actions["agent-0"] = np.random.choice([0,1], p=[1.0-p, p])

        above_value = np.where(self.last_reward>self.agents["agent-1"].value(),
                                                                          1, 0)
        weighted_num_above_value = np.average(above_value, 
                                              weights=[10,9,8,7,6,5,4,3,2,1])

        if weighted_num_above_value < self.cooperation_level:
            ## be selfish if they are selfish
            p = float(actions["agent-1"])
            played_actions["agent-1"] = np.random.choice([0,1], p=[1.0-p, p])
            observations = {"agent-0": 0, "agent-1": 0, "agent-2": 0}
            rewards = {}; dones = {}; info = {}
            agent_0, agent_1, agent_2 = self.agents.values()
            r_0  = agent_0.compute_reward(played_actions)
            r_1  = agent_1.compute_reward(played_actions)

            rewards["agent-0"] = r_0
            rewards["agent-1"] = r_1 ## agent-1 is selfish (D) opponent
            rewards["agent-2"] = 0
        else:
            ## be cooperative if they are cooperative 
            p = float(actions["agent-2"])
            played_actions["agent-1"] = np.random.choice([0,1], p=[1.0-p, p])
            observations = {"agent-0": 0, "agent-1": 0, "agent-2": 0}
            rewards = {}; dones = {}; info = {}
            agent_0, agent_1, agent_2 = self.agents.values()
            r_0  = agent_0.compute_reward(played_actions)
            r_1  = agent_1.compute_reward(played_actions)

            rewards["agent-0"] = r_0
            rewards["agent-1"] = 0 
            rewards["agent-2"] = r_0 + r_1 ## agent-2 is (C) opponent 

        self.last_reward = np.concatenate(([r_1], self.last_reward[:-1]))

        dones["agent-0"] = agent_0.get_done()
        dones["agent-1"] = agent_1.get_done()
        dones["agent-2"] = agent_2.get_done()
        dones["__all__"] = np.any(list(dones.values()))
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

        observations = {}
        for agent in self.agents.values():
            observations[agent.agent_id] = 0
        return observations
