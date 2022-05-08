import numpy as np
import random
import os
import matplotlib.pyplot as plt

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


reward = average_plot(np.load("./PacMAN/Good/seed_2/reward.npy"),100)
reward_safe = average_plot(np.load("./PacMAN/Normal/seed_1/reward.npy"),100)

steps = np.load("./PacMAN/Good/seed_2/steps.npy").tolist()[0:len(reward)]
steps_safe = np.load("./PacMAN/Normal/seed_1/steps.npy").tolist()[0:len(reward_safe)]

i = steps[len(steps)-1]
print(steps[-1])

reward = reward[0:i]
steps = steps[0:i]


plt.figure()
plt.plot(steps,reward)
plt.plot(steps_safe,reward_safe)

plt.show()








 ssh username@gpu1.cse.iitk.ac.in
 cd /data/vatsalpj21
 source ./anaconda3/bin/activate
 conda create -m envname python=3.7
 conda activate envname
 nvidia-smi
