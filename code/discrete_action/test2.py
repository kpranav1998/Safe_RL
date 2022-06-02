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



reward = average_plot(np.load("./reward.npy"),50)

reward_safe = average_plot(np.load("./reward_safe.npy"),1)

steps = np.load("./steps.npy").tolist()[0:len(reward)]
steps_safe = np.load("./steps_safe.npy").tolist()[0:len(reward_safe)]
'''
i = 0
while(steps[i] < int(12e6) and i < len(steps)-1):
    i = i + 1

#reward = reward[0:i]
#steps = steps[0:i]

i = 0
while(steps_safe[i] < int(12e6) and len(steps_safe)-1):
    i = i + 1

reward_safe = reward_safe[0:i]
steps_safe = steps_safe[0:i]

'''
x= [70]


plt.figure()
plt.plot(steps,reward)
#plt.axhline(y=10000, color='r', linestyle='-')
plt.axhline(y=0, color='r', linestyle='-')

plt.fill_between(steps,2700 + 250, 2700 - 250,
                 alpha=0.2, color='g')

#plt.plot(steps_safe,reward_safe)

plt.show()








