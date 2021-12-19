import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import numpy.ma as ma
from itertools import zip_longest
import seaborn as sns
sns.set()
sns.color_palette("bright", 10)

episode = 100
GAME_NAME = "Ant"
LOAD_PATH = "../../"+str(GAME_NAME)+"/Good"
BASE_LOAD_PATH = "../../"+str(GAME_NAME)+"/Normal"

folders = os.listdir(LOAD_PATH)

base_folders = os.listdir(BASE_LOAD_PATH)
print(base_folders)
base_rewards = []
base_steps = []
steps = []
rewards = []

BASELINE_VALUE  =  5100
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

length = int(1e9)
for folder in base_folders:
    if(".png" not in folder and ".npy" not in folder):
        path = os.path.join(BASE_LOAD_PATH,folder)
        print(path)
        base_reward = np.load(os.path.join(path,"reward.npy"))
        if(base_reward.shape[0] < length):
            length = base_reward.shape[0]
        base_rewards.append(base_reward)


base_arr = np.array([base_rewards[0][0:length],base_rewards[1][0:length],base_rewards[2][0:length]])#,base_rewards[2][0:length]])
base_avg_reward = np.nanmean(np.array(list(zip_longest(*base_arr)),dtype=float),axis=1)
base_std_reward = np.nanstd(np.array(list(zip_longest(*base_arr)),dtype=float),axis=1)

i = 0
while(i < int(1.5e6)):
    i += int(1.5e6)/length
    base_steps.append(i)
    i = i + 1





for folder in folders:
    if(".png" not in folder and ".npy" not in folder):
        path = os.path.join(LOAD_PATH,folder)
        reward = np.load(os.path.join(path,"reward.npy"))

        rewards.append(reward)

        uncertainity = np.load(os.path.join(path,"uncertainity.npy"),allow_pickle=True)
        u = []
        for row in uncertainity:
            row = np.asarray(row)
            u.append(np.mean(row))
        u = np.asarray(u)
        uncertainities.append(u)




length_safe = min(len(rewards[0]),len(rewards[1]),len(rewards[2]))
arr = np.array([rewards[0][0:length_safe],rewards[1][0:length_safe],rewards[2][0:length_safe]])

rewards = np.asarray(rewards)
avg_reward = np.nanmean(np.array(list(zip_longest(*arr)),dtype=float),axis=1)
std_reward = np.nanstd(np.array(list(zip_longest(*arr)),dtype=float),axis=1)

avg_reward = average_plot(avg_reward,episode)
base_avg_reward = average_plot(base_avg_reward,episode)



i = 0
while(i < int(1.5e6)):
    i += int(1.5e6)/length_safe
    steps.append(i)
    i = i + 1


x = min(avg_reward.shape[0],len(steps))
y = min(base_avg_reward.shape[0],len(base_steps))



a =x
b = y
y = int(y * 0.95)

x = int(x * 0.95)
plt.xlabel('steps')
plt.ylabel('reward')

plt.plot(steps[0:x], avg_reward[0:x], color='b',lw=3)
plt.fill_between(steps[0:x], avg_reward[0:x] + std_reward[0:x], avg_reward[0:x] -std_reward[0:x], alpha=0.2,color='b')

plt.plot(base_steps[0:y],base_avg_reward[0:y], 'r',lw=3) # reward_2
plt.fill_between(base_steps[0:y], base_avg_reward[0:y] + base_std_reward[0:y], base_avg_reward[0:y] - base_std_reward[0:y], alpha= 0.2,color='r')

plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-',lw=3)  ## baseline
#plt.legend(["Agent with baseline","Normal Agent","Baseline Value"], loc ="best")
plt.savefig(os.path.join(LOAD_PATH,"avg_reward.png"))
plt.show()



x = a

y = b
uncertainities = np.asarray(uncertainities)
print(uncertainities.shape)
avg_uncertainity = np.nanmean(np.array(list(zip_longest(*uncertainities)),dtype=float),axis=1)
plt.figure()
plt.xlabel('steps')
plt.ylabel('uncertainity')
plt.plot(steps[0:x],avg_uncertainity[0:x], 'b') # uncertainity_1
#plt.legend(["Uncertainity"], loc="best")

plt.savefig(os.path.join(LOAD_PATH,"avg_uncertainity.png"))
plt.close()


## Hopper Medium = 1700
## Hopper Good =
## Ant good = 5100
