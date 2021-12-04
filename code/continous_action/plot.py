import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import numpy.ma as ma
from itertools import zip_longest

GAME_NAME = "Hopper"
LOAD_PATH = "../../"+str(GAME_NAME)+"/Good"

folders = os.listdir(LOAD_PATH)


BASELINE_VALUE  =  2300
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


steps = []

uncertainities = []

for folder in folders:

    if (".png" not in folder):
        seed_step = []
        j = 0
        path = os.path.join(LOAD_PATH,folder)
        uncertainity = np.load(os.path.join(path,"uncertainity.npy"),allow_pickle=True)
        uncertainity = np.asarray(uncertainity)
        main_list = []
        print(j+1)
        for i in range(len(uncertainity)):
            j += len(uncertainity[i])
            seed_step.append(j)
            uncertainity[i] = np.asarray(uncertainity[i])
            main_list.extend(uncertainity[i])
        main_list = average_plot(np.asarray(main_list),200)
        plt.figure()
        plt.ylabel('uncertainity')
        plt.plot(main_list, 'b')  # uncertainity_1
        # plt.plot(base_uncertainity, 'r') # uncertainity_2
        # plt.legend(["Average performance of agent with help of baseline","Normal Agent"], loc ="best")
        plt.legend(["Uncertainity"], loc="best")

        plt.savefig(os.path.join(path, "uncertainity.png"))
        plt.close()
        steps.append(seed_step)
        uncertainities.append(main_list)

base_reward = np.load("../../"+str(GAME_NAME)+"/Normal/reward.npy")
base_steps = []
base_uncertainity = np.load("../../"+str(GAME_NAME)+"/Normal/uncertainity.npy",allow_pickle=True)
j = 0
for i in range(len(base_uncertainity)):
    j += len(base_uncertainity[i])
    base_steps.append(j)
    base_uncertainity[i] = np.asarray(base_uncertainity[i])


base_reward = average_plot(base_reward,200)
h2 = min(base_reward.shape[0], len(base_steps))
j = 0
rewards = []
index = 0
l = 0
for folder in folders:
    if(".png" not in folder):
        path = os.path.join(LOAD_PATH,folder)
        reward = np.load(os.path.join(path,"reward.npy"))
        reward = average_plot(reward,margin=200)
        rewards.append(reward)
        h1 = min(reward.shape[0],len(steps[j]))
        if(reward.shape[0] > l ):
            l = reward.shape[0]
            index = j
        plt.figure()
        plt.ylabel('reward')
        plt.plot(steps[j][0:h1],reward,color = 'b')
        plt.plot(base_steps[0:h2],base_reward,color = 'r')
        plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
        plt.legend(["Agent with help from Baseline ","Normal Agent","Baseline Value"], loc ="best")
        plt.savefig(os.path.join(path,"reward.png"))
        plt.close()
        j = j + 1

rewards = np.asarray(rewards)
avg_reward = np.nanmean(np.array(list(zip_longest(*rewards)),dtype=float),axis=1)
std_reward = np.nanstd(np.array(list(zip_longest(*rewards)),dtype=float),axis=1)

h = min(rewards[index].shape[0],len(steps[index]))

plt.figure()
plt.ylabel('reward')
plt.plot(steps[index][0:h], avg_reward, color='b')
plt.fill_between(steps[index][0:h], avg_reward+std_reward, avg_reward - std_reward, alpha=.5)

plt.plot(base_steps[0:h2],base_reward, 'r') # reward_2
plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
plt.legend(["Agent with baseline","Normal Agent","Baseline Value"], loc ="best")
plt.savefig(os.path.join(LOAD_PATH,"avg_reward.png"))
plt.close()




uncertainities = np.asarray(uncertainities)
avg_uncertainity = np.nanmean(np.array(list(zip_longest(*uncertainities)),dtype=float),axis=1)
plt.figure()
plt.ylabel('uncertainity')
plt.plot(avg_uncertainity, 'b') # uncertainity_1
plt.legend(["Uncertainity"], loc="best")

plt.savefig(os.path.join(LOAD_PATH,"avg_uncertainity.png"))
plt.close()


## Hopper Medium = 1700
## Hopper Good = 2300
## Ant good = 5100
