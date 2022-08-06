import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import json


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Humanoid-v2",
                    help='Mujoco Gym environment (default: Walker-2D)')
parser.add_argument('--lcb', default=0.1,type=float,
                    help='LCB constant value')
parser.add_argument('--safe_path',type=str,default="/home/deep8/pranav/Safe_RL_Technion/pytorch-soft-actor-critic/results/HalfCheetah-v2_0/model_10005.67001717289.pkl",
                    help='path to baseline')
parser.add_argument('--n_ensemble', default=3,type=int,
                    help='number of ensemble members')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=random.randint(1,100000) , metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=int(5.1e6), metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()
arg_dict = vars(args)

run_num = 0
model_base_filedir  = os.path.join("./results", args.env_name + "_" + str(run_num))
while os.path.exists(model_base_filedir):
            run_num += 1
            model_base_filedir = os.path.join("./results", args.env_name + "_" + str(run_num))
os.makedirs(model_base_filedir)
print("starting NEW project: %s" % model_base_filedir)
json.dump( arg_dict, open( os.path.join(model_base_filedir, "args.json"), 'w' ) )


# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
writer = SummaryWriter(model_base_filedir+'/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
run_num = 0


reward_list = []
uncertainity_list = []
steps = []

def average_plot(list,y_label,save_path, margin=3):
    list = np.asarray(list)
    avg_list = []
    for i in range(list.shape[0] - margin):
        temp = 0
        for j in range(margin):
            temp = temp + list[i + j]
        temp = np.float32(temp) / margin
        avg_list.append(temp)
    avg_list = np.asarray(avg_list, dtype=np.float32)
    plt.figure()
    plt.ylabel(y_label)
    plt.plot(avg_list, color='r')
    plt.savefig(save_path + '.png')

if True:
        avg_reward = 0.
        episodes = 1
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes



        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")



for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    episode_uncertainity = []
    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_loss,value_loss, policy_loss, ent_loss, alpha,uncertainity= agent.update_parameters(memory, args.batch_size, updates)
                episode_uncertainity.append(uncertainity.item())
                writer.add_scalar('uncertainity/model_uncertainity', uncertainity, updates)
                #writer.add_scalar('uncertainity/safe_model_uncertainity', safe_uncertainity, updates)
                writer.add_scalar('loss/critic', critic_loss, updates)
                writer.add_scalar('loss/value', value_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps +=1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break
    steps.append(total_numsteps)
    print(total_numsteps)
    uncertainity_list.append(episode_uncertainity)
    reward_list.append(episode_reward)
    np.save(os.path.join(model_base_filedir,'uncertainity.npy'), uncertainity_list)
    np.save(os.path.join(model_base_filedir, 'reward.npy'), reward_list)
    np.save(os.path.join(model_base_filedir,'steps.npy'), steps)

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 150 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes


        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

    if(i_episode % 20 == 0):
        agent.save_checkpoint(args.env_name,ckpt_path=os.path.join(model_base_filedir,"model_"+str(episode_reward)+".pkl"))

env.close()