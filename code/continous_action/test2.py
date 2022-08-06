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


reward = np.load("./Action_size.npy")
'''
steps = np.load("./steps.npy")

reward_baseline = np.load("./reward_baseline.npy")

steps_baseline = np.load("./steps_baseline.npy")
#uncertainity = np.load("./uncertainity.npy",allow_pickle=True).tolist()

reward = average_plot(reward,200)
steps = average_plot(steps,200)


reward_baseline = average_plot(reward_baseline,200)
steps_baseline = average_plot(steps_baseline,200)
'''
'''i = 0
while(steps[i] < int(2e6)):
    i = i + 1

reward = reward[0:i]
steps = steps[0:i]
uncertainity = uncertainity[0:i]

np.save("./reward.npy",reward)
np.save("./steps.npy",steps)
np.save("./uncertainity.npy",uncertainity)'''

reward = average_plot(reward,1000)

plt.figure()
plt.plot(reward)
#plt.plot(steps_baseline,reward_baseline)

#plt.axhline(y = 0, color ="green", linestyle ="--")
#plt.fill_between(steps,2331+1416,2331-1416, alpha=0.2, color='b')

plt.show()








