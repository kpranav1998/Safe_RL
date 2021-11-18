import numpy as np
import matplotlib.pyplot as plt
import os


BASELINE_VALUE  = 1711
def average_plot(list,margin=50):
    avg_list = []
    for i in range(list.shape[0] - margin):
        temp = 0
        for j in range(margin):
            temp = temp + list[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    return avg_list


base_reward = np.load("./results/Hopper-v2_2/reward.npy")
base_reward = average_plot(base_reward)
if(True):
        reward = np.load("./results/Hopper-v2_safe_22/reward.npy")

        reward = average_plot(reward)
        plt.figure()
        plt.ylabel('reward')
        plt.plot(reward, 'b') # reward_1
        plt.plot(base_reward, 'r') # reward_2
        plt.axhline(y=BASELINE_VALUE, color='y', linestyle='-')  ## baseline
        plt.legend(["Agent with help from Baseline","Normal Agent","Baseline Value"], loc ="best")
        plt.savefig("./reward_Hopper.png")
        plt.close()


