import numpy as np
import matplotlib.pyplot as plt
import os

GAME_NAME = "Qbert"
LOAD_PATH = "../"+str(GAME_NAME)+"/Good_10975"

folders = os.listdir(LOAD_PATH)
rewards = []
uncertainities = []

BASELINE_VALUE  = 10975
def average_plot(list,margin=500):
    avg_list = []
    for i in range(list.shape[0] - margin):
        temp = 0
        for j in range(margin):
            temp = temp + list[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    return avg_list


base_reward = np.load("../"+str(GAME_NAME)+"/Normal_DQN/reward.npy")
base_reward = average_plot(base_reward)
lengths = []
for folder in folders:

    if(".png" not in folder):
        path = os.path.join(LOAD_PATH,folder)
        reward = np.load(os.path.join(path,"reward.npy"))

        reward = average_plot(reward)
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

avg_reward = 0
length = min(lengths)
for reward in rewards:
    avg_reward += reward[0:length]

avg_reward = avg_reward/len(rewards)

plt.figure()
plt.ylabel('reward')
plt.plot(avg_reward, 'b') # reward_1
plt.plot(base_reward, 'r') # reward_2
plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
plt.legend(["Average performance of agent with help of baseline","Normal Agent","Baseline Value"], loc ="best")
plt.savefig(os.path.join(LOAD_PATH,"avg_reward.png"))
plt.close()




#base_uncertainity = np.load("../Pong/Normal_DQN/uncertainity.npy")
#base_uncertainity = average_plot(base_uncertainity)
lengths = []
for folder in folders:

    if (".png" not in folder):
        path = os.path.join(LOAD_PATH,folder)
        uncertainity = np.load(os.path.join(path,"uncertainity.npy"))

        uncertainity = average_plot(uncertainity)
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
