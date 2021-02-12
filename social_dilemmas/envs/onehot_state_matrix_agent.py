"""Base class for an agent that defines the possible actions. """

from gym.spaces import Box
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

class Agent(object):

    def __init__(self, agent_id):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        """
        self.agent_id = agent_id

    @property
    def action_space(self):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError


class MatrixAgent(Agent):
    def __init__(self, agent_id, game):
        super().__init__(agent_id)
        self.game = game 

    @property
    def action_space(self):
        return Box(low=0, high=1, shape=(1,))

    @property
    def observation_space(self):
        return Discrete(11)

    def get_done(self):
        return False
    
    def compute_reward(self, actions_dict):
        agents = actions_dict.keys()
        for agent in agents:
            if agent == self.agent_id:
                my_action = actions_dict[self.agent_id]
            else:
                their_action = actions_dict[agent]
        reward = self.payoff_matrix()[int(my_action)][int(their_action)]

        return reward
    
    def compute_expected_reward(self, my_strategy, their_action):

        reward = (1-my_strategy)*self.payoff_matrix()[0][int(their_action)] \
                   + my_strategy*self.payoff_matrix()[1][int(their_action)]

        return reward

    def payoff_matrix(self):
        ## prisoner's dilemma 
        if self.game == "prisoners":
            return np.array([[0.75,0],[1,0.25]])
        elif self.game == "staghunt":
            return np.array([[1,0],[0.75,0.25]])
        elif self.game == "routechoice":
            return np.array([[0,0.25],[1,0.5]])
        else:
            raise "Illegal matrix game type"
    
    def value(self):
        ## prisoner's dilemma 
        if self.game == "prisoners":
            return 0.25
        elif self.game == "staghunt":
            return 0.25
        elif self.game == "routechoice":
            return 0.5
        else:
            raise "Illegal matrix game type"


