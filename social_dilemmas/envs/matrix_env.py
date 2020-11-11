import numpy as np
from ray.rllib.env import MultiAgentEnv
from social_dilemmas.envs.matrix_agent import MatrixAgent

class MatrixEnv(MultiAgentEnv):

    def __init__(self, game):
        """
        Parameters
        ----------
        game: string
            Name of matrix game (prisoners, staghunt or routechoice).
        """
        self.num_agents = 2
        self.game = game
        self.agents = {}
        self.setup_agents()

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
        played_actions = {}
        for agent in actions.keys():
            p = float(actions[agent])
            p = np.clip(p, 0, 1)
            played_actions[agent] = np.random.choice([0,1], p=[1.0-p, p])
        observations = {}
        rewards = {}
        dones = {}
        info = {}
        player_num = 0
        for agent in self.agents.values():
            observations[agent.agent_id] = 0
            rewards[agent.agent_id] = agent.compute_reward(played_actions)
            dones[agent.agent_id] = agent.get_done()
            player_num += 1
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

        observations = {}
        for agent in self.agents.values():
            observations[agent.agent_id] = 0
        return observations
