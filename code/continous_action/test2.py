import numpy as np
import random
import os
import matplotlib.pyplot as plt


def average_plot(l, margin=100):
    avg_list = []
    for i in range(len(l) - margin):
        temp = 0
        for j in range(margin):
            temp = temp + l[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    return avg_list


def short_list(l):
    i = 0
    new_list = []
    while (i < len(l) - 1):
        new_list.append(l[i])
        i = i + 10

    return new_list

steps = []
while (i < int(3e6) - 1):
    steps.append(i)
    i = i + 10


steps = np.asarray(steps)
print(steps.shape[0])
average_size = 1000
Action_size_1 = short_list(np.load("/content/Safe_RL/Value_state_method/Humanoid/Medium/seed_1/Action_size.npy"))

Action_size_2 = short_list(np.load("/content/Safe_RL/Value_state_method/Humanoid/Medium/seed_2/Action_size.npy"))
Action_size_3 = short_list(np.load("/content/Safe_RL/Value_state_method/Humanoid/Medium/seed_3/Action_size.npy"))



length = min(len(Action_size_1), len(Action_size_2), len(Action_size_3))
Action_size_1 = np.asarray(Action_size_1)
Action_size_2 = np.asarray(Action_size_2)
Action_size_3 = np.asarray(Action_size_3)

Action_size_mean = (Action_size_1[0:length] + Action_size_2[0:length] + Action_size_3[0:length]) / 3

average_size = 50
Action_size_1 = average_plot(Action_size_1, average_size)
Action_size_2 = average_plot(Action_size_2, average_size)
Action_size_3 = average_plot(Action_size_3, average_size)

Action_size_mean = average_plot(Action_size_mean, 1000)

plt.figure()
plt.plot(steps[0:Action_size_mean.shape[0]],Action_size_mean, alpha=1, color='b')
plt.plot(steps[0:Action_size_1.shape[0]],Action_size_1, alpha=0.2, color='b')
plt.plot(steps[0:Action_size_2.shape[0]],Action_size_2, alpha=0.2, color='b')
plt.plot(steps[0:Action_size_3.shape[0]],Action_size_3, alpha=0.2, color='b')
plt.title("Action Size")

# plt.plot(steps_baseline,reward_baseline)

# plt.axhline(y = 0, color ="green", linestyle ="--")
# plt.fill_between(steps,2331+1416,2331-1416, alpha=0.2, color='b')
plt.legend(["Mean", "Seed 1", "Seed 2", "Seed 3"], loc="best")

# plt.show()
plt.savefig("./Action_size.jpg")









