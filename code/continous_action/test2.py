import numpy as np





for i in range(1,4):
    steps = np.load("./Normal/seed_"+str(i)+"/steps.npy")
    reward = np.load("./Normal/seed_"+str(i)+"/reward.npy")

    step_number = 1

    while(steps[step_number] <= int(1.75e6)):
        step_number += 1

    reward = reward[0:step_number]
    steps = steps[0:step_number]

    np.save("./Normal/seed_"+str(i)+"/reward.npy",reward)
    np.save("./Normal/seed_"+str(i)+"/steps.npy",steps)


