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


reward = np.load("./reward.npy")

steps = np.load("./steps.npy")
uncertainity = np.load("./uncertainity.npy",allow_pickle=True).tolist()

reward = average_plot(reward,100)
steps = average_plot(steps,100)

'''i = 0
while(steps[i] < int(2e6)):
    i = i + 1

reward = reward[0:i]
steps = steps[0:i]
uncertainity = uncertainity[0:i]

np.save("./reward.npy",reward)
np.save("./steps.npy",steps)
np.save("./uncertainity.npy",uncertainity)'''

plt.figure()
plt.plot(steps,reward)

plt.show()








