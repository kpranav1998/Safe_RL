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





reward1 = average_plot(np.load("./reward.npy"),200
                       )
steps1 = np.load("./steps.npy").tolist()[0:len(reward1)]
'''
reward2 = average_plot(np.load("./InvertedPendulum/Good/seed_2/reward.npy"),20
                       )
steps2 = np.load("./InvertedPendulum/Good/seed_2/steps.npy").tolist()[0:len(reward2)]

reward3 = average_plot(np.load("./InvertedPendulum/Good/seed_3/reward.npy"),20
                       )
steps3= np.load("./InvertedPendulum/Good/seed_3/steps.npy").tolist()[0:len(reward3)]




'''



plt.figure()
plt.title("InvertedPendulum Good")
plt.plot(steps1,reward1)
'''plt.plot(steps2,reward2)
plt.plot(steps3,reward3)
'''
#plt.plot(real_steps1,real_reward1)
#plt.plot(real_steps2,real_reward2)


#plt.axhline(y=1500, color='r', linestyle='-')
#plt.fill_between(steps1, 15, 11, alpha= 0.2,color='r')
plt.legend(["seed_1","seed_2","seed_3"], loc ="best")

#
#plt.plot(steps,rewards)

plt.show()
#plt.savefig("./InvertedPendulum_Good.png")








