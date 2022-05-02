import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import numpy.ma as ma
from itertools import zip_longest
import seaborn as sns
sns.set()
sns.color_palette("bright", 10)


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

'''for i in range(1,4):
    reward_agent = np.load("./Normal/seed_"+str(i)+"/reward.npy")
    steps_agent = np.load("./Normal/seed_"+str(i)+"/steps.npy")

    mean = 8828
    std = 824
    factor = 0.5
    reward_baseline = np.load("../reward.npy")
    steps_baseline = np.load("../steps.npy")
    
    reward_agent = average_plot(reward_agent)
    steps_agent = average_plot(steps_agent)
    
    reward_baseline = average_plot(reward_baseline)
    steps_baseline = average_plot(steps_baseline)

    plt.title(str(i))
    plt.plot(steps_agent,reward_agent, 'r',lw=2) # reward_2
    #plt.plot(steps_baseline,reward_baseline, 'b',lw=2) # reward_2
    
    plt.fill_between(steps_baseline, mean + factor*std, mean-factor*std, alpha=0.5,color='g')
    

    plt.legend(["Agent with baseline","Normal Agent","Baseline Value"], loc ="best")


    #plt.savefig("./Ant_4399.jpg")
    plt.show()
'''

    #### agent ANT 5127 mean = 4466, std =1138
    #### agent ANT 4399 mean = 3288, std =1567
    #### agent Walker 4207 mean = 4055, std =40
    #### agent HalfCheetah mean =8828, std = 824
    #### agent HalfCheetah 6198 mean = 5733 , std = 327
    #### agent Hopper 2920 mean = 2826 , std = 205


reward_agent = np.load("./reward.npy")
steps_agent = np.load("./steps.npy")


reward_agent = average_plot(reward_agent,200)
steps_agent = steps_agent[0:len(reward_agent)]

reward_safe = np.load("./reward (4).npy")
stesp_safe = np.load("./steps (2).npy")

reward_safe = average_plot(reward_safe,200)
stesp_safe = steps_agent[0:len(reward_safe)]

plt.plot(steps_agent,reward_agent)
plt.plot(stesp_safe,reward_safe)

#plt.plot(reward_safe)

plt.show()

