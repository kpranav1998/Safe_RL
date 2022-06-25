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



reward1 = average_plot(np.load("./reward.npy"),50)
steps1 = np.load("./steps.npy").tolist()[0:len(reward1)]
'''
reward2 = average_plot(np.load("./Good/seed_2/reward.npy"),50)
steps2 = np.load("./Good/seed_2/steps.npy").tolist()[0:len(reward2)]

reward3 = average_plot(np.load("./Good/seed_3/reward.npy"),50)
steps3 = np.load("./Good/seed_3/steps.npy").tolist()[0:len(reward3)]
#steps_safe = np.load("./steps (1).npy").tolist()[0:len(reward_safe)]

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
plt.plot(steps1,reward1)
#plt.plot(steps2,reward2)
#plt.plot(steps3,reward3)

plt.axhline(y=21, color='r', linestyle='-')
plt.axhline(y=-21, color='r', linestyle='-')
plt.fill_between(steps1, 15, 11, alpha= 0.2,color='r')

#
#plt.plot(steps_safe,reward_safe)

plt.show()








