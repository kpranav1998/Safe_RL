import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import numpy.ma as ma
from itertools import zip_longest

GAME_NAME = "Qbert"
LOAD_PATH = "../../"+str(GAME_NAME)+"/Good_10975"

folders = os.listdir(LOAD_PATH)


BASELINE_VALUE  =  10975
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


uncertainities = []

base_reward = np.load("../../"+str(GAME_NAME)+"/Normal_DQN/reward.npy")
base_reward = average_plot(base_reward,200)
base_steps = []

i = 0
while(i < int(2.5e7)):
    i += int(2.5e7)/base_reward.shape[0]
    base_steps.append(i)
    i = i + 1



j = 0
rewards = []
index = 0
l = int(1e8)


for folder in folders:
    if(".png" not in folder and ".npy" not in folder):
        path = os.path.join(LOAD_PATH,folder)
        reward = np.load(os.path.join(path,"reward.npy"))
        reward = average_plot(reward,margin=200)

        rewards.append(reward)

        steps = []

        i = 0
        while (i < int(3e7)):
            i += int(3e7) / reward.shape[0]
            steps.append(i)
            i = i + 1

        plt.figure()
        plt.ylabel('reward')
        x = min(reward.shape[0],len(steps))
        y = min(base_reward.shape[0],len(base_steps))
        if (x < l):
            l = x
        plt.plot(steps[0:x],reward[0:x],color = 'b')
        plt.plot(base_steps[0:y],base_reward[0:y],color = 'r')
        plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
        plt.legend(["Agent with help from Baseline ","Normal Agent","Baseline Value"], loc ="best")
        plt.savefig(os.path.join(path,"reward.png"))
        plt.close()

        uncertainity = np.load(os.path.join(path, "uncertainity.npy"), allow_pickle=True)
        uncertainity = np.asarray(uncertainity)
        uncertainity = average_plot(uncertainity, 200)
        uncertainities.append(uncertainity)
        plt.figure()
        plt.ylabel('uncertainity')
        plt.plot(steps[0:x],uncertainity[0:x], 'b')  # uncertainity_1
        plt.legend(["Uncertainity"], loc="best")

        plt.savefig(os.path.join(path, "uncertainity.png"))
        plt.close()

        j = j + 1

rewards = np.asarray(rewards)
avg_reward = np.nanmean(np.array(list(zip_longest(*rewards)),dtype=float),axis=1)
std_reward = np.nanstd(np.array(list(zip_longest(*rewards)),dtype=float),axis=1)

net_steps = []

i = 0
while (i < int(3e7)):
            i += int(3e7) / avg_reward.shape[0]
            net_steps.append(i)
            i = i + 1

x = min(avg_reward.shape[0],len(net_steps))

plt.figure()
plt.ylabel('reward')
plt.plot(net_steps[0:x], avg_reward[0:x], color='b')
plt.fill_between(net_steps[0:x], avg_reward[0:x]+std_reward[0:x], avg_reward[0:x] - std_reward[0:x], alpha=.5)

#plt.plot(base_steps[0:h2],base_reward, 'r') # reward_2
plt.plot(base_steps[0:y],base_reward[0:y], 'r') # reward_2

plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
plt.legend(["Agent with baseline","Normal Agent","Baseline Value"], loc ="best")
plt.savefig(os.path.join(LOAD_PATH,"avg_reward.png"))
plt.close()




uncertainities = np.asarray(uncertainities)
avg_uncertainity = np.nanmean(np.array(list(zip_longest(*uncertainities)),dtype=float),axis=1)
plt.figure()
plt.ylabel('uncertainity')
plt.plot(net_steps[0:x],avg_uncertainity[0:x], 'b') # uncertainity_1
plt.legend(["Uncertainity"], loc="best")

plt.savefig(os.path.join(LOAD_PATH,"avg_uncertainity.png"))
plt.close()


## Hopper Medium = 1700
## Hopper Good =
## Ant good = 5100
