import gym
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential, load_model
import cv2
import random
import numpy as np
import tensorflow.keras
import os
import tensorflow as tf
from my_grid_world import GridWorld
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import Counter
import gym

memory = []
memory_length = 1000
training_episodes = 15000
training_start = 32
learning_rate = 0.00001
ensemble_size = 5

num_of_actions = 4
batch_size = 32
ATARI_SHAPE = [4, 4]
alpha = 0.1
gamma = 0.99
beta = 3
render = True

reward = []
steps = []
episode = []

TARGET_UPDATE = 1000


def plot_reward(reward_list):
    plt.figure()
    reward_temp = np.asarray(reward)
    steps_temp = np.asarray(steps)
    plt.ylabel('reward')
    plt.plot(steps_temp)
    plt.savefig("./" + 'reward.jpg')
    plt.show()


def predict_q_values(models, state):
    q_values = []
    for model in models:
        q = model.predict(state.reshape(1, 16))
        q_values.append(q[0, :])

    return q_values


def get_action(present_state):
    if (len(memory) < training_start):
        return random.randint(0,num_of_actions-1)
    else:

        q_values = predict_q_values(model_ensemble, present_state)
        q_values = np.asarray(q_values)

        temp = model_ensemble[random.randrange(ensemble_size)].predict(present_state.reshape(1, 16))
        #print("temp:",temp)
        action = np.argmax(temp)

        return action


def get_randomized_prior_nn():
    ##trainable##
    net_input = Input(shape=(16,), name='input')

    trainable = Dense(164, input_shape=(16,))(net_input)
    trainable = Dense(150, input_shape=(16,))(trainable)
    trainable_output = Dense(num_of_actions, input_shape=(16,))(trainable)

    ##prior##
    prior = Dense(164, kernel_initializer='glorot_normal', trainable=False, input_shape=(16,))(net_input)
    prior = Dense(150, kernel_initializer='glorot_normal', trainable=False, input_shape=(16,))(prior)
    prior = Dense(num_of_actions, kernel_initializer='glorot_normal', trainable=False, input_shape=(16,))(prior)

    prior_output = Lambda(lambda x: x * 3)(prior)

    output = Add()([trainable_output, prior_output])
    model = Model(inputs=net_input, outputs=output)
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer, loss=tf.keras.losses.Huber(delta=100.0))

    return model


def ensemble_network(size):
    models = []
    for i in range(size):
        model = get_randomized_prior_nn()
        models.append(model)
    return models


def target_train():
    for i in range(ensemble_size):
        model_weights = model_ensemble[i].get_weights()

        target_ensemble[i].set_weights(model_weights)


def train():
    if (len(memory) < training_start):
        return 0

    for model_number in range(ensemble_size):
        mini_batch = random.sample(memory, batch_size)
        S_t_copy = np.zeros((batch_size, 16))
        S_t_1_copy = np.zeros((batch_size, 16))
        reward_copy = np.zeros((batch_size))
        A_t = np.zeros((batch_size, num_of_actions), np.int32)
        done_copy = []
        target = np.zeros((batch_size, num_of_actions))
        y = np.zeros((batch_size))

        for i in range(batch_size):
            S_t_copy[i] = mini_batch[i][0]
            reward_copy[i] = mini_batch[i][1]
            S_t_1_copy[i] = mini_batch[i][2]
            A_t[i] = mini_batch[i][3]
            done_copy.append(mini_batch[i][4])

        Q_t_1 = target_ensemble[model_number].predict(S_t_1_copy.reshape(-1, 16))

        for i in range(batch_size):
            if (done_copy[i] == True):
                target[i][A_t[i]] = reward_copy[i]
            else:
                target[i][A_t[i]] = reward_copy[i] + gamma * np.max(Q_t_1[i])

        h = model_ensemble[model_number].fit(S_t_copy, target, epochs=1, batch_size=batch_size, verbose=0)


model_ensemble = ensemble_network(ensemble_size)
target_ensemble = ensemble_network(ensemble_size)
'''i = 0
ensemble_models = []
for i in range(ensemble_size):
    model = load_model(ensemeble_path[i], compile=False)
    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(optimizer, loss=tf.keras.losses.Huber(delta=100.0))
    ensemble_models.append(model)
    i = i + 1

'''
def testAlgo():
    grid = GridWorld()
    action_names = ["up", "down", "left", "right"]
    grid.set()
    episode_reward = 0
    observation = grid.display()
    S_t = observation
    print(S_t)
    done = False
    action_list = []
    moves = 0

    while (done == False):
        action = get_action_test(S_t)

        action_list.append(action_names[action])
        moves = moves + 1
        new_state = grid.next(action)
        reward = grid.getReward()
        S_t_1 = new_state
        if reward != -1:
            done = True
        episode_reward += reward
        S_t = S_t_1
    print("reward:", episode_reward)




def main():
    episode_number = 0
    step_number = 0
    action_names = ["up", "down", "left", "right"]
    while (step_number < int(1e5)):
        grid = GridWorld()

        episode_reward = 0
        uncertainity = 0
        observation = grid.display()

        S_t = observation
        done = False
        observation_list = []
        observation_list.append((S_t, None))
        action_list = []
        episode_steps = 0

        while (done == False):
            #print(grid.display())
            action = get_action(S_t)
            #print("action:", action)

            action_list.append(action_names[action])
            new_state = grid.next(action)

            S_t_1 = new_state

            step_number += 1
            episode_steps += 1
            temp_reward = grid.getReward()
            episode_reward += temp_reward

            if(episode_steps >= int(150)):
                done = True
            if temp_reward != -1:
                done = True
            if (len(memory) == memory_length):
                memory.pop(0)

            if (step_number % 4 == 0):
                train()
            if (step_number % TARGET_UPDATE == 0):
                target_train()
            episode_reward += temp_reward
            memory.append((S_t.reshape(1, 16), episode_reward, S_t_1.reshape(-1, 16), action, done))
            S_t = S_t_1
            #print(episode_reward)

        episode_number += 1
        episode.append(episode_number)
        reward.append(episode_reward)
        steps.append(step_number)
        np.save(os.path.join("./", 'reward.npy'),reward)
        np.save(os.path.join("./", 'step.npy'),steps)

        print("episode_reward", episode_reward, "episode no:", episode_number,"step number: ", step_number)
        if (episode_number % int(50) == 0):
            j = 0
            for model in model_ensemble:
                model.save(os.path.join("./", 'safe_gridworld_' + str(j) + str("_") + str(episode_number)))
                j = j + 1


plot_reward(reward)

main()
