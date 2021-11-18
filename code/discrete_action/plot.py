import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import numpy.ma as ma
from itertools import zip_longest

GAME_NAME = "Ant"
LOAD_PATH = "../"+str(GAME_NAME)+"/Good"

folders = os.listdir(LOAD_PATH)
rewards = []
uncertainities = []

BASELINE_VALUE  = 3500
def average_plot(l,margin=100):
    avg_list = []
    for i in range(l.shape[0] - margin):
        temp = 0
        for j in range(margin):
            temp = temp + l[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    return avg_list


base_reward = np.load("../"+str(GAME_NAME)+"/Normal/reward.npy")
base_reward =base_reward[:1800]
base_reward = average_plot(base_reward)
lengths = []
for folder in folders:

    if(".png" not in folder):
        path = os.path.join(LOAD_PATH,folder)
        reward = np.load(os.path.join(path,"reward.npy"))
        reward = average_plot(reward)
        #reward = reward[:5300]
        lengths.append(len(reward))
        rewards.append(reward)
        plt.figure()
        plt.ylabel('reward')
        plt.plot(reward, 'b') # reward_1
        plt.plot(base_reward, 'r') # reward_2
        plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
        plt.legend(["Agent with help from Baseline ","Normal Agent","Baseline Value"], loc ="best")
        plt.savefig(os.path.join(path,"reward.png"))
        plt.close()

avg_reward = np.nanmean(np.array(list(zip_longest(*rewards)),dtype=float),axis=1)
plt.figure()
plt.ylabel('reward')
plt.plot(avg_reward, 'b') # reward_1
plt.plot(base_reward, 'r') # reward_2
plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
plt.legend(["Agent with baseline","Normal Agent","Baseline Value"], loc ="best")
plt.savefig(os.path.join(LOAD_PATH,"avg_reward.png"))
plt.close()



base_uncertainity = np.load("../Walker2D/Normal/uncertainity.npy",allow_pickle=True)
for i in range(len(base_uncertainity)):
    base_uncertainity[i] = np.mean(base_uncertainity[i])

base_uncertainity = average_plot(base_uncertainity,margin=20)
lengths = []
for folder in folders:

    if (".png" not in folder):
        path = os.path.join(LOAD_PATH,folder)
        uncertainity = np.load(os.path.join(path,"uncertainity.npy"),allow_pickle=True)
        for i in range(len(uncertainity)):
            uncertainity[i] = np.mean(uncertainity[i])
        uncertainity = average_plot(uncertainity,margin=5)
        lengths.append(len(uncertainity))
        uncertainities.append(uncertainity)
        plt.figure()
        plt.ylabel('uncertainity')
        plt.plot(uncertainity, 'b') # uncertainity_1
        #plt.plot(base_uncertainity, 'r') # uncertainity_2
        plt.legend(["Uncertainity"], loc ="best")
        plt.savefig(os.path.join(path,"uncertainity.png"))
        plt.close()

avg_uncertainity = 0
length = min(lengths)
for uncertainity in uncertainities:
    avg_uncertainity += uncertainity[0:length]

avg_uncertainity = avg_uncertainity/len(uncertainities)

plt.figure()
plt.ylabel('uncertainity')
plt.plot(avg_uncertainity, 'b') # uncertainity_1
#plt.plot(base_uncertainity, 'r') # uncertainity_2
#plt.legend(["Average performance of agent with help of baseline","Normal Agent"], loc ="best")
plt.legend(["Uncertainity"], loc="best")

plt.savefig(os.path.join(LOAD_PATH,"avg_uncertainity.png"))
plt.close()
