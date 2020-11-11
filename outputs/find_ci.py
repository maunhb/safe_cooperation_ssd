import numpy as np
import scipy.stats
import pandas as pd 

matrix_game = "prisoners"
agent_1 = "arctic"
agent_2 = "adv"

data = pd.read_csv('{}_{}_vs_{}_rewards.csv'.format(matrix_game,
                                                    agent_1,
                                                    agent_2), sep=',') 

reward_0 = data["rewards_0"]
reward_1 = data["rewards_1"]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

m, low, high = mean_confidence_interval(reward_0)
print("Agent 0")
print("Low ", low, "Mean ", m, "High ", high)
m, low, high = mean_confidence_interval(reward_1)
print("Agent 1")
print("Low ", low, "Mean ", m, "High ", high)
