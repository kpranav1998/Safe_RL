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


reward = average_plot(np.load("./reward.npy"),100)
#reward_safe = average_plot(np.load("./PacMAN/Normal/seed_1/reward.npy"),100)

steps = np.load("./steps.npy").tolist()[0:len(reward)]
#steps_safe = np.load("./PacMAN/Normal/seed_1/steps.npy").tolist()[0:len(reward_safe)]

i = steps[len(steps)-1]
print(steps[-1])

reward = reward[0:i]
steps = steps[0:i]
x= [70]


plt.figure()
plt.plot(steps,reward)
plt.axhline(y=70, color='r', linestyle='-')
plt.fill_between(steps, 70 + 9, 70 - 9,
                 alpha=0.2, color='g')

#plt.plot(steps_safe,reward_safe)

plt.show()








