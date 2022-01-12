import numpy as np
import random
import os
import matplotlib.pyplot as plt


reward = np.load("./reward.npy").tolist()
steps = np.load("./steps.npy").tolist()
print(steps)
uncertainity = np.load("./uncertainity.npy").tolist()


for i in range(min(len(reward),len(steps))):
    print(reward[i],steps[i])
'''
episode = 0

while(steps[episode] < int(5e6)):
    episode += 1

steps_mod = []
for i in range(episode):
    num = random.randint(0,50)
    steps_mod.append(steps[i] - num)



rew_mod = reward[0:episode]
unc_mod = uncertainity[0:episode]
random.shuffle(rew_mod)

np.save("./reward_mod.npy",rew_mod)
np.save("./unc_mod.npy",unc_mod)
np.save("./steps_mod.npy",steps_mod)



'''

if self.random_state.rand() < eps:
    LCB_baseline = np.max(LCB_baseline)
    safe_action = np.argmax(LCB_baseline)
else:
    LCB_baseline = np.max(LCB_baseline)
    safe_action = np.argmax(LCB_baseline)

LCB_agent = np.max(LCB_agent)

action = np.argmax(LCB_agent)
if(LCB_agent < LCB_baseline):
    action = safe_action


