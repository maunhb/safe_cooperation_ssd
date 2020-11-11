import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.stats
import numpy as np

matrix_game = "prisoners"

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h
time = np.arange(1,101)

plt.figure()

labels = ["ARCTIC","$p^C$","Adv","Baseline"]
i = 0 
def plot_data(data,label):
    cooperation = []
    cooperation_low = []
    cooperation_high = []
    for i in range(1,101):
        coop_vec = data["cooperation_{}".format(i)]
        coop_av, coop_low, coop_high = mean_confidence_interval(coop_vec)
        cooperation.append(coop_av)
        cooperation_low.append(coop_low)
        cooperation_high.append(coop_high)

    plt.plot(time, cooperation,'-', label="ARCTIC")
    plt.fill_between(time, cooperation_low, cooperation_high,
                    color='blue', alpha=0.2)

data = pd.read_csv("./tournaments/{}_arctic_vs_arctic_cooperation.csv".format(
                                                               matrix_game), sep=',')
plot_data(data,labels[i]) 
i += 1
data = pd.read_csv("./tournaments/{}_arctic_vs_coop_cooperation.csv".format(
                                                              matrix_game), sep=',')
plot_data(data,labels[i]) 
i += 1
data = pd.read_csv("./tournaments/{}_arctic_vs_adv_cooperation.csv".format(
                                                             matrix_game), sep=',')
plot_data(data,labels[i]) 
i += 1
data = pd.read_csv("./tournaments/{}_arctic_vs_baseline_cooperation.csv".format(
                                                                 matrix_game), sep=',')
plot_data(data,labels[i]) 

plt.ylabel("Cooperation")
plt.ylim([-0.05,1.05])
plt.xlabel("Round")
plt.legend(bbox_to_anchor=(1.25, 1.0))

plt.show()

