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


reward = average_plot(np.load("./reward.npy"),20)
reward_safe = average_plot(np.load("./reward_safe.npy"),20)

steps = np.load("./steps.npy").tolist()

plt.figure()
plt.plot(reward)
plt.plot(reward_safe)

plt.show()








