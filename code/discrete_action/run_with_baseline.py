from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import Counter
import torch

torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datetime
import time
from dqn_model import EnsembleNet, NetWithPrior
from dqn_utils import seed_everything, write_info_file, generate_gif, save_checkpoint
from env import Environment
from replay import ReplayMemory
import config
import random


reward_save = []
steps_save = []



def rolling_average(a, n=5):
    if n == 0:
        return a
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_dict_losses(plot_dict, name='loss_example.png', rolling_length=4, plot_title=''):
    f, ax = plt.subplots(1, 1, figsize=(6, 6))
    for n in plot_dict.keys():
        print('plotting', n)
        ax.plot(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), lw=1)
        ax.scatter(rolling_average(plot_dict[n]['index']), rolling_average(plot_dict[n]['val']), label=n, s=3)
    ax.legend()
    if plot_title != '':
        plt.title(plot_title)
    plt.savefig(name)
    plt.close()


def matplotlib_plot_all(p):
    epoch_num = len(p['steps'])
    epochs = np.arange(epoch_num)
    steps = p['steps']
    print(model_base_filedir)
    plot_dict_losses({' steps': {'index': epochs, 'val': p['episode_step']}},
                     name=os.path.join(model_base_filedir, 'episode_step.png'), rolling_length=0)
    plot_dict_losses({'episode steps': {'index': epochs, 'val': p['episode_relative_times']}},
                     name=os.path.join(model_base_filedir, 'episode_relative_times.png'), rolling_length=10)
    plot_dict_losses({'episode head': {'index': epochs, 'val': p['episode_head']}},
                     name=os.path.join(model_base_filedir, 'episode_head.png'), rolling_length=0)
    plot_dict_losses({'steps loss': {'index': steps, 'val': p['episode_loss']}},
                     name=os.path.join(model_base_filedir, 'steps_loss.png'))

    plot_dict_losses({'steps reward': {'index': steps, 'val': p['episode_reward']}},
                     name=os.path.join(model_base_filedir, 'steps_reward.png'), rolling_length=0)
    plot_dict_losses({'episode reward': {'index': epochs, 'val': p['episode_reward']}},
                     name=os.path.join(model_base_filedir, 'episode_reward.png'), rolling_length=0)
    plot_dict_losses({'episode times': {'index': epochs, 'val': p['episode_times']}},
                     name=os.path.join(model_base_filedir, 'episode_times.png'), rolling_length=5)
    plot_dict_losses({'steps avg reward': {'index': steps, 'val': p['avg_rewards']}},
                     name=os.path.join(model_base_filedir, 'steps_avg_reward.png'), rolling_length=0)
    plot_dict_losses({'eval rewards': {'index': p['eval_steps'], 'val': p['eval_rewards']}},
                     name=os.path.join(model_base_filedir, 'eval_rewards_steps.png'), rolling_length=0)


def handle_checkpoint(last_save, cnt):
    if (cnt - last_save) >= info['CHECKPOINT_EVERY_STEPS']:
        st = time.time()
        print("beginning checkpoint", st)
        last_save = cnt
        state = {'info': info,
                 'optimizer': opt.state_dict(),
                 'cnt': cnt,
                 'policy_net_state_dict': policy_net.state_dict(),
                 'target_net_state_dict': target_net.state_dict(),
                 'perf': perf,
                 }
        filename = os.path.abspath(model_base_filepath + "_%010dq.pkl" % cnt)
        save_checkpoint(state, filename)
        # npz will be added
        buff_filename = os.path.abspath(model_base_filepath + "_%010dq_train_buffer" % cnt)
        if(info["SAVE_MEMORY_BUFFER"] == True):
            replay_memory.save_buffer(buff_filename)
        print("finished checkpoint", time.time() - st)
        return last_save
    else:
        return last_save


class ActionGetter:
    """Determines an action according to an epsilon greedy strategy with annealing epsilon"""

    def __init__(self, n_actions, replay_memory_start_size=50000, max_steps=25000000, random_seed=122):

        self.n_actions = n_actions
        self.replay_memory_start_size = replay_memory_start_size
        self.max_steps = max_steps
        self.random_state = np.random.RandomState(random_seed)

    def pt_get_action(self, step_number, state, evaluation=False,baseline_evaluation=False):

        state = torch.Tensor(state.astype(np.float) / info['NORM_BY'])[None, :].to(info['DEVICE'])

        if (evaluation == True):
            '''
            For evaluating the agent without baseline I using voting to decide the appropriate action
            '''
            vals = policy_net(state, None)
            acts = [torch.argmax(vals[h], dim=1).item() for h in range(info['N_ENSEMBLE'])]
            data = Counter(acts)
            action = data.most_common(1)[0][0]
            return action

        elif (baseline_evaluation == True):
            safe_vals = safe_net(state, None)
            safe_vals = torch.cat(safe_vals, 0)
            safe_mean_val = torch.mean(safe_vals, axis=0)
            safe_std_val = torch.std(safe_vals, axis=0)
            safe_LCB = safe_mean_val - info["LCB_constant"] * safe_std_val
            safe_action = torch.argmax(safe_LCB).item()
            return safe_action

        elif(step_number < self.replay_memory_start_size):
            safe_vals = safe_net(state, None)
            safe_vals = torch.cat(safe_vals, 0)
            safe_mean_val = torch.mean(safe_vals, axis=0)
            safe_std_val = torch.std(safe_vals, axis=0)
            safe_LCB = safe_mean_val - info["LCB_constant"] * safe_std_val
            safe_action = torch.argmax(safe_LCB).item()
            return safe_action

        else:
            vals = policy_net(state, None)
            vals = torch.cat(vals, 0)
            mean_val = torch.mean(vals, axis=0)
            std_val = torch.std(vals, axis=0)
            LCB = mean_val - info["LCB_constant"] * std_val
            action = torch.argmax(LCB).item()
            LCB_value = torch.max(LCB).item()

            safe_vals = safe_net(state, None)
            safe_vals = torch.cat(safe_vals, 0)
            safe_mean_val = torch.mean(safe_vals, axis=0)
            safe_std_val = torch.std(safe_vals, axis=0)
            safe_LCB = safe_mean_val -info["LCB_constant"]  * safe_std_val
            safe_action = torch.argmax(safe_LCB).item()
            safe_LCB_value = torch.max(safe_LCB).item()

            if(LCB_value < safe_LCB_value):
                action = safe_action

            return action

def ptlearn(states, actions, rewards, next_states, terminal_flags, masks):
    states = torch.Tensor(states.astype(np.float) / info['NORM_BY']).to(info['DEVICE'])
    next_states = torch.Tensor(next_states.astype(np.float) / info['NORM_BY']).to(info['DEVICE'])
    rewards = torch.Tensor(rewards).to(info['DEVICE'])
    actions = torch.LongTensor(actions).to(info['DEVICE'])
    terminal_flags = torch.Tensor(terminal_flags.astype(np.int)).to(info['DEVICE'])
    masks = torch.FloatTensor(masks.astype(np.int)).to(info['DEVICE'])
    # min history to learn is 200,000 frames in dqn - 50000 steps
    losses = [0.0 for _ in range(info['N_ENSEMBLE'])]
    opt.zero_grad()
    q_policy_vals = policy_net(states, None)


    next_q_target_vals = target_net(next_states, None)
    next_q_policy_vals = policy_net(next_states, None)
    cnt_losses = []
    for k in range(info['N_ENSEMBLE']):
        # TODO finish masking
        total_used = torch.sum(masks[:, k])
        if total_used > 0.0:
            next_q_vals = next_q_target_vals[k].data
            if info['DOUBLE_DQN']:
                next_actions = next_q_policy_vals[k].data.max(1, True)[1]
                next_qs = next_q_vals.gather(1, next_actions).squeeze(1)
            else:
                next_qs = next_q_vals.max(1)[0]  # max returns a pair

            preds = q_policy_vals[k].gather(1, actions[:, None]).squeeze(1)
            targets = rewards + info['GAMMA'] * next_qs * (1 - terminal_flags)
            l1loss = F.smooth_l1_loss(preds, targets, reduction='mean')
            full_loss = masks[:, k] * l1loss
            loss = torch.sum(full_loss / total_used)
            cnt_losses.append(loss)
            losses[k] = loss.cpu().detach().item()

    loss = sum(cnt_losses) / info['N_ENSEMBLE']
    loss.backward()
    for param in policy_net.core_net.parameters():
        if param.grad is not None:
            # divide grads in core
            param.grad.data *= 1.0 / float(info['N_ENSEMBLE'])
    nn.utils.clip_grad_norm_(policy_net.parameters(), info['CLIP_GRAD'])
    opt.step()
    return np.mean(losses)


def train(step_number, last_save):
    """Contains the training and evaluation loops"""
    epoch_num = len(perf['steps'])
    while step_number < info['MAX_STEPS']:
        ########################
        ####### Training #######
        ########################
        epoch_frame = 0

        while epoch_frame < info['EVAL_FREQUENCY']:
            terminal = False
            life_lost = True
            state = env.reset()
            start_steps = step_number
            st = time.time()
            random_state.shuffle(heads)
            active_head = heads[0]
            epoch_num += 1
            ep_eps_list = []
            ptloss_list = []


            episode_reward_sum = 0
            episode_steps = 0
            episode_loss = []

            while not terminal:
                if life_lost:
                    action = 1
                else:
                    action = action_getter.pt_get_action(step_number,state=state)

                episode_steps = episode_steps + 1

                next_state, reward, life_lost, terminal = env.step(action)
                # Store transition in the replay memory
                replay_memory.add_experience(action=action,
                                             frame=next_state[-1],
                                             reward=np.sign(reward),  # TODO -maybe there should be +1 here
                                             terminal=life_lost)

                step_number += 1
                epoch_frame += 1
                episode_reward_sum += reward
                state = next_state


                if step_number % info['LEARN_EVERY_STEPS'] == 0 and (step_number) > info['MIN_HISTORY_TO_LEARN']:
                    _states, _actions, _rewards, _next_states, _terminal_flags, _masks = replay_memory.get_minibatch(
                        info['BATCH_SIZE'])

                    ptloss = ptlearn(_states, _actions, _rewards, _next_states,_terminal_flags, _masks)
                    episode_loss.append(ptloss)
                    ptloss_list.append(ptloss)
                if step_number % info['TARGET_UPDATE'] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                    print("++++++++++++++++++++++++++++++++++++++++++++++++")
                    print('updating target network at %s' % step_number)
                    target_net.load_state_dict(policy_net.state_dict())
                    np.save(os.path.join(model_base_filedir, 'reward.npy'), reward_save)
                    np.save(os.path.join(model_base_filedir, 'steps.npy'), steps_save)

            if (step_number > info["MIN_HISTORY_TO_LEARN"]):
                steps_save.append(step_number)
                reward_save.append(episode_reward_sum)

            #print(episode_reward_sum)
            et = time.time()
            ep_time = et - st
            perf['steps'].append(step_number)
            perf['episode_step'].append(step_number - start_steps)
            perf['episode_head'].append(active_head)
            perf['episode_loss'].append(np.mean(ptloss_list))
            perf['episode_reward'].append(episode_reward_sum)
            perf['episode_times'].append(ep_time)
            perf['episode_relative_times'].append(time.time() - info['START_TIME'])
            perf['avg_rewards'].append(np.mean(perf['episode_reward'][-info["PLOT_EVERY_EPISODES"]:]))
            last_save = handle_checkpoint(last_save, step_number)

            if epoch_num % info["PLOT_EVERY_EPISODES"] == 0 and step_number > info['MIN_HISTORY_TO_LEARN']:
                # TODO plot title
                print('avg reward', perf['avg_rewards'][-1])
                print('last rewards', perf['episode_reward'][-info["PLOT_EVERY_EPISODES"]:])

                matplotlib_plot_all(perf)
                with open('rewards.txt', 'a') as reward_file:
                    print(len(perf['episode_reward']), step_number, perf['avg_rewards'][-1], file=reward_file)
        avg_eval_reward = evaluate(step_number)
        perf['eval_rewards'].append(avg_eval_reward)
        perf['eval_steps'].append(step_number)
        matplotlib_plot_all(perf)


def evaluate(step_number):
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    evaluate_step_number = 0
    frames_for_gif = []
    results_for_eval = []
    # only run one
    for i in range(info['NUM_EVAL_EPISODES']):
        state = env.reset()
        episode_reward_sum = 0
        terminal = False
        life_lost = True
        episode_steps = 0
        while not terminal:
            if life_lost:
                action = 1
            else:
                action = action_getter.pt_get_action(step_number, state, evaluation=True)
            next_state, reward, life_lost, terminal = env.step(action)
            evaluate_step_number += 1
            episode_steps += 1
            episode_reward_sum += reward
            if not i:
                # only save first episode
                frames_for_gif.append(env.ale.getScreenRGB())
                results_for_eval.append("%s, %s, %s, %s" % (action, reward, life_lost, terminal))
            if not episode_steps % 100:
                print('eval', episode_steps, episode_reward_sum)
            state = next_state
        eval_rewards.append(episode_reward_sum)

    print("Evaluation score:\n", np.mean(eval_rewards))
    generate_gif(model_base_filedir, step_number, frames_for_gif, eval_rewards[0], name='test',
                 results=results_for_eval)

    # Show the evaluation score in tensorboard
    efile = os.path.join(model_base_filedir, 'eval_rewards.txt')
    with open(efile, 'a') as eval_reward_file:
        print(step_number, np.mean(eval_rewards), file=eval_reward_file)
    return np.mean(eval_rewards)


def baseline_evaluate():
    print("""
         #########################
         ####### Evaluation ######
         #########################
         """)
    eval_rewards = []
    frames_for_gif = []
    results_for_eval = []
    # only run one
    for i in range(50):
        seed = random.randint(1,100000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        evaluate_step_number = 0
        state = env.reset()
        episode_reward_sum = 0
        terminal = False
        life_lost = True
        steps = 0
        episode_steps = 0

        while not terminal:
            steps = steps + 1
            if life_lost:
                action = 1
            else:
                action = action_getter.pt_get_action(int(3e6), state=state, baseline_evaluation = True)
            next_state, reward, life_lost, terminal = env.step(action)

            evaluate_step_number += 1
            episode_steps += 1
            episode_reward_sum += reward

            state = next_state
        eval_rewards.append(episode_reward_sum)
        print(episode_reward_sum)

    print("Evaluation score:\n", np.mean(eval_rewards),np.std(eval_rewards))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-c', '--cuda', action='store_true', default=True)
    parser.add_argument('-l', '--model_loadpath', default='', help='.pkl model file full path')
    parser.add_argument('-s', '--safe_model_loadpath',
                        default='./breakout_rpf_0001205244q.pkl',
                        help='.pkl model file full path')

    parser.add_argument('-b', '--buffer_loadpath', default='', help='.npz replay buffer file full path')
    args = parser.parse_args()
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
    print("running on %s" % device)

    info = {
        # "GAME":'roms/breakout.bin', # gym prefix
        "GAME": './breakout.bin',  # gym prefix
        "DEVICE": device,  # cpu vs gpu set by argument
        "NAME": 'breakout_safe_v2_',  # start files with name
        "DUELING": True,  # use dueling dqn
        "DOUBLE_DQN": True,  # use double dqn
        "PRIOR": True,  # turn on to use randomized prior
        "PRIOR_SCALE": 3,  # what to scale prior by
        "N_ENSEMBLE": 5,  # number of bootstrap heads to use. when 1, this is a normal dqn
        "LEARN_EVERY_STEPS": 4,  # updates every 4 steps in osband
        "BERNOULLI_PROBABILITY": 0.9,# Probability of experience to go to each head - if 1, every experience goes to every head
        "TARGET_UPDATE": 120000,  # how often to update target network
        "MIN_HISTORY_TO_LEARN": 64,  # in environment frames
        "NORM_BY": 255.,  # divide the float(of uint) by this number to normalize - max val of data is 255
        "NUM_EVAL_EPISODES": 1,  # num examples to average in eval
        "BUFFER_SIZE": int(1e6),  # Buffer size for experience replay
        "CHECKPOINT_EVERY_STEPS": int(1e6),  # how often to write pkl of model and npz of data buffer
        "EVAL_FREQUENCY": int(1e6),  # how often to run evaluation episodes
        "ADAM_LEARNING_RATE": 6.25e-5,
        "HISTORY_SIZE": 4,  # how many past frames to use for state input
        "N_EPOCHS": 90000,  # Number of episodes to run
        "BATCH_SIZE": 64,  # Batch size to use for learning
        "GAMMA": .99,  # Gamma weight in Q update
        "PLOT_EVERY_EPISODES": 50,
        "CLIP_GRAD": 10,  # Gradient clipping setting
        "SEED": random.randint(1,100000),
        "RANDOM_HEAD": -1,  # just used in plotting as demarcation
        "NETWORK_INPUT_SIZE": (84, 84),
        "SAVE_MEMORY_BUFFER" : False,
        "START_TIME": time.time(),
        "MAX_STEPS": int(15e6),  # 50e6 steps is 200e6 frames
        "MAX_EPISODE_STEPS": 27000,  # Orig dqn give 18k steps, Rainbow seems to give 27k steps
        "FRAME_SKIP": 4,  # deterministic frame skips to match deepmind
        "MAX_NO_OP_FRAMES": 30,  # random number of noops applied to beginning of each episode
        "DEAD_AS_END": True,  # do you send finished=true to agent while training when it loses a life
        "LCB_constant": 0.1,
        "Baseline_Value": "NA"
    }

    np.random.seed(info["SEED"])
    torch.manual_seed(info["SEED"])

    info['FAKE_ACTS'] = [info['RANDOM_HEAD'] for x in range(info['N_ENSEMBLE'])]
    info['args'] = args
    info['load_time'] = datetime.date.today().ctime()
    info['NORM_BY'] = float(info['NORM_BY'])

    # create environment
    env = Environment(rom_file=info['GAME'], frame_skip=info['FRAME_SKIP'],
                      num_frames=info['HISTORY_SIZE'], no_op_start=info['MAX_NO_OP_FRAMES'], rand_seed=info['SEED'],
                      dead_as_end=info['DEAD_AS_END'], max_episode_steps=info['MAX_EPISODE_STEPS'])

    # create replay buffer
    replay_memory = ReplayMemory(size=info['BUFFER_SIZE'],
                                 frame_height=info['NETWORK_INPUT_SIZE'][0],
                                 frame_width=info['NETWORK_INPUT_SIZE'][1],
                                 agent_history_length=info['HISTORY_SIZE'],
                                 batch_size=info['BATCH_SIZE'],
                                 num_heads=info['N_ENSEMBLE'],
                                 bernoulli_probability=info['BERNOULLI_PROBABILITY'])

    random_state = np.random.RandomState(info["SEED"])
    action_getter = ActionGetter(n_actions=env.num_actions,
                                 replay_memory_start_size=info['MIN_HISTORY_TO_LEARN'],
                                 max_steps=info['MAX_STEPS'])

    if args.model_loadpath != '':
        # load data from loadpath - save model load for later. we need some of
        # these parameters to setup other things
        print('loading model from: %s' % args.model_loadpath)
        model_dict = torch.load(args.model_loadpath)
        info = model_dict['info']
        info['DEVICE'] = device
        # set a new random seed
        info["SEED"] = model_dict['cnt']
        #model_base_filedir = os.path.split(args.model_loadpath)[0]
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d' % run_num)
        while os.path.exists(model_base_filedir):
            run_num += 1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d' % run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s" % model_base_filedir)

        start_step_number = start_last_save = model_dict['cnt']
        info['loaded_from'] = args.model_loadpath
        perf = model_dict['perf']



    else:
        # create new project
        perf = {'steps': [],
                'avg_rewards': [],
                'episode_step': [],
                'episode_head': [],
                'episode_loss': [],
                'episode_reward': [],
                'episode_times': [],
                'episode_relative_times': [],
                'eval_rewards': [],
                'eval_steps': []}

        start_step_number = 0#steps_save[len(steps_save) -1]
        start_last_save = 0
        # make new directory for this run in the case that there is already a
        # project with this name
        run_num = 0
        model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d' % run_num)
        while os.path.exists(model_base_filedir):
            run_num += 1
            model_base_filedir = os.path.join(config.model_savedir, info['NAME'] + '%02d' % run_num)
        os.makedirs(model_base_filedir)
        print("----------------------------------------------")
        print("starting NEW project: %s" % model_base_filedir)

    model_base_filepath = os.path.join(model_base_filedir, info['NAME'])
    write_info_file(info, model_base_filepath, start_step_number)
    heads = list(range(info['N_ENSEMBLE']))
    seed_everything(info["SEED"])

    policy_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                             n_actions=env.num_actions,
                             network_output_size=info['NETWORK_INPUT_SIZE'][0],
                             num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    target_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                             n_actions=env.num_actions,
                             network_output_size=info['NETWORK_INPUT_SIZE'][0],
                             num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])
    #### safe model ####
    safe_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                           n_actions=env.num_actions,
                           network_output_size=info['NETWORK_INPUT_SIZE'][0],
                           num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

    if info['PRIOR']:
        prior_net = EnsembleNet(n_ensemble=info['N_ENSEMBLE'],
                                n_actions=env.num_actions,
                                network_output_size=info['NETWORK_INPUT_SIZE'][0],
                                num_channels=info['HISTORY_SIZE'], dueling=info['DUELING']).to(info['DEVICE'])

        print("using randomized prior")
        policy_net = NetWithPrior(policy_net, prior_net, info['PRIOR_SCALE'])
        target_net = NetWithPrior(target_net, prior_net, info['PRIOR_SCALE'])

        safe_net = NetWithPrior(safe_net, prior_net, info['PRIOR_SCALE'])

    opt = optim.Adam(policy_net.parameters(), lr=info['ADAM_LEARNING_RATE'])


    #### safe model ####
    print('safe model from: %s' % args.safe_model_loadpath)
    safe_model_dict = torch.load(args.safe_model_loadpath)
    safe_model_base_filedir = os.path.split(args.safe_model_loadpath)[0]
    safe_net.load_state_dict(safe_model_dict['policy_net_state_dict'])



    if args.model_loadpath is not '':
        # what about random states - they will be wrong now???
        # TODO - what about target net update cnt
        target_net.load_state_dict(model_dict['target_net_state_dict'])
        policy_net.load_state_dict(model_dict['policy_net_state_dict'])

        #opt.load_state_dict(model_dict['optimizer'])
        print("loaded model state_dicts")
        if args.buffer_loadpath == '':
            args.buffer_loadpath = args.model_loadpath.replace('.pkl', '_train_buffer.npz')
            print("auto loading buffer from:%s" % args.buffer_loadpath)
            try:
                replay_memory.load_buffer(args.buffer_loadpath)
            except Exception as e:
                print(e)
                print('not able to load from buffer: %s. exit() to continue with empty buffer' % args.buffer_loadpath)

    #baseline_evaluate()
    train(0, start_last_save)
