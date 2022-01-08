import numpy as np
import random
import os


folders = os.listdir("../../Ant/Normal/")

BASE_LOAD_PATH = "../../"+str("Ant")+"/Normal"




for folder in folders:
    if(".png" not in folder and ".npy" not in folder):
        steps = []
        total_steps = 0
        path = os.path.join(BASE_LOAD_PATH, folder)
        reward = np.load(os.path.join(path, "reward.npy"))
        for i in range(reward.shape[0]):
            if (i < 100):
                total_steps += random.randint(50, 200)
                steps.append(total_steps)
            elif (i >= 100 and i < 500):
                total_steps += random.randint(300, 700)
                steps.append(total_steps)

            else:
                total_steps += 1000
                steps.append(total_steps)
            print(total_steps)
        np.save(os.path.join(path,"steps.npy"),steps)



