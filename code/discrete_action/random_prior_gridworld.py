import gym
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model,Sequential,load_model
import cv2
import random
import numpy as np
import keras
import os
import tensorflow as tf
from my_grid_world import GridWorld
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import Counter 



memory = []
memory_length = 1000
training_episodes = 15000
training_start = 32
learning_rate = 0.00001
ensemble_size = 5
safety_threshold = 10	


num_of_actions = 4
batch_size = 32
ATARI_SHAPE =[4,4]
alpha = 0.1
gamma = 0.99
beta = 3
render = True
SAFE_PATH = '/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/saved_models/unsafe_gridworld_4_15000'
SAVE_PATH = '/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/safe_models/'
FIG_PATH = "/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/figs/"

ensemeble_path = ['/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/saved_models/unsafe_gridworld_0_15000',
'/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/saved_models/unsafe_gridworld_1_15000',
'/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/saved_models/unsafe_gridworld_2_15000',
'/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/saved_models/unsafe_gridworld_3_15000',
'/home/kpranav1998/Desktop/iit/Israel_technion/Project/code/mycode/SAFE/saved_models/unsafe_gridworld_4_15000']


  
def most_frequent(List): 
    occurence_count = Counter(List) 
    return occurence_count.most_common(1)[0][0] 
    
 
def plot_reward(reward_list):
	plt.figure()
	temp = np.asarray(reward_list)
	plt.ylabel('reward')
	plt.plot(temp)
	plt.savefig(FIG_PATH+'safe_reward.jpg')
	plt.show()

def plot_uncertainity(uncertainity_list):
	plt.figure()
	temp = np.asarray(uncertainity_list)
	plt.ylabel('uncertainity')
	plt.plot(temp)
	plt.savefig(FIG_PATH+'safe_uncertainity.jpg')
	plt.show()

def plot_No_of_uncertain_situations(uncertainity_list):
	plt.figure()
	temp = np.asarray(uncertainity_list)
	plt.ylabel('No_of_uncertain_situations')
	plt.plot(temp)
	plt.savefig(FIG_PATH+'safe_No_of_uncertain_situations.jpg')
	plt.show()




def uncertainity(q_values):
	std = np.std(q_values,axis = 0)
	mean = np.abs(np.mean(q_values,axis=0))
	coefficient_of_uncertainity = np.mean(std/mean)
	return coefficient_of_uncertainity

def predict_q_values(models,state):
	q_values = []
	for model in models:
		q = model.predict(state.reshape(1,16))
		q_values.append(q[0,:])

	return q_values

def get_action_test(present_state):
	model_action_list = []
	temp_list = []
	action_names = ["up", "down", "left", "right"]

	for model in ensemble_models:
		qval = model.predict(present_state.reshape(1,16), batch_size=1)
		action = (np.argmax(qval))
		temp_list.append(action)
		model_action_list.append(action_names[action])
	#print("model_action_list:",model_action_list)
	return most_frequent(temp_list)

def get_action(present_state):
		
		if(len(memory) < training_start):
			return random.randrange(num_of_actions),0
		else:
			
			q_values = predict_q_values(model_ensemble,present_state)
			q_values = np.asarray(q_values)
			coefficient_of_uncertainity = uncertainity(q_values)
			'''
			if(coefficient_of_uncertainity > safety_threshold):
				
				qval = safe_model.predict(present_state.reshape(1,16), batch_size=1)
				action = (np.argmax(qval))

				return action, coefficient_of_uncertainity
			'''
		
			temp = model_ensemble[random.randrange(ensemble_size)].predict(present_state.reshape(1,16))
				
			action = np.argmax(temp)	
			return action,coefficient_of_uncertainity


def get_randomized_prior_nn():

    ##trainable##
	net_input = Input(shape=(16,), name='input')

	trainable = Dense(164,input_shape=(16,))(net_input)
	trainable = Dense(150, input_shape=(16,))(trainable)
	trainable_output = Dense(num_of_actions, input_shape=(16,))(trainable)
	

    ##prior##
	prior = Dense(164, kernel_initializer='glorot_normal',trainable=False, input_shape=(16,))(net_input)
	prior = Dense(150, kernel_initializer='glorot_normal',trainable=False, input_shape=(16,))(prior)
	prior = Dense(num_of_actions, kernel_initializer='glorot_normal',trainable=False, input_shape=(16,))(prior)
	
	
	prior_output = Lambda(lambda x: x *3)(prior)
	 

	output = Add()([trainable_output,prior_output])
	model = Model(inputs=net_input, outputs=output)
	optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(optimizer, loss= tf.keras.losses.Huber(delta=100.0))
	

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
	if(len(memory) < training_start):
		return 0
	
	for model_number in range(ensemble_size):
		mini_batch = random.sample(memory, batch_size)
		S_t_copy = np.zeros((batch_size,16))
		S_t_1_copy = np.zeros((batch_size,16))
		reward_copy = np.zeros((batch_size))
		A_t = np.zeros((batch_size,num_of_actions),np.int32)
		done_copy = []
		target = np.zeros((batch_size,num_of_actions))
		y = np.zeros((batch_size))

		
		for i in range(batch_size):
			S_t_copy[i] =	 mini_batch[i][0]
			reward_copy[i] = mini_batch[i][1]
			S_t_1_copy[i] = mini_batch[i][2]
			A_t[i] = mini_batch[i][3]
			done_copy.append(mini_batch[i][4])
		
		Q_t_1 = target_ensemble[model_number].predict(S_t_1_copy.reshape(-1,16))
		

		for i in range(batch_size):
			if(done_copy[i] == True):
				target[i][A_t[i]] = reward_copy[i]
			else:
				target[i][A_t[i]] = reward_copy[i] + gamma * np.max(Q_t_1[i])
		
		h = model_ensemble[model_number].fit(S_t_copy, target, epochs=1,batch_size=batch_size, verbose=0)
		#print("model_number:"+str(model_number)+" "+str(h.history['loss'][0]))




model_ensemble = ensemble_network(ensemble_size)
target_ensemble = ensemble_network(ensemble_size)
i = 0
ensemble_models = []
for i in range(ensemble_size):

	model = load_model(ensemeble_path[i],compile=False)
	optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(optimizer, loss= tf.keras.losses.Huber(delta=100.0))
	ensemble_models.append(model)
	i = i + 1


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

	while(done == False):
		action = get_action_test(S_t)
		action_list.append(action_names[action])
		moves = moves + 1
		new_state = grid.next(action)
		reward = grid.getReward()
		S_t_1  = new_state
		if reward != -1:
			done = True
		episode_reward += reward
		S_t = S_t_1
	print("reward:",episode_reward)

testAlgo()

def main():

	grid = GridWorld() 

	
	i = 0
	episode = 0
	episode_list = []
	reward_list = []
	uncertainity_graph = []
	no_of_uncertainity = []

	action_names = ["up", "down", "left", "right"]
	while(episode < training_episodes):
		grid.set()

		episode_reward = 0
		uncertainity = 0
		observation = grid.display()
		#print(observation)
		S_t = observation
		done = False
		status = False
		observation_list= []
		observation_list.append((S_t,None))
		action_list = []
		moves = 0
		while(done == False):
			
			action,coefficient_of_uncertainity = get_action(S_t)
			action_list.append(action_names[action])
			moves = moves + 1
			if(coefficient_of_uncertainity > safety_threshold):
				uncertainity = uncertainity + 1
			new_state = grid.next(action)
			#observation_list.append((new_state,action))


			
			reward = grid.getReward()
			S_t_1  = new_state
			
			print(S_t)
			print(action_names[action])
			print(S_t_1)
			
			if reward != -1:
				done = True
			if(len(memory) == memory_length):
				memory.pop(0)
			
			if(episode % 4 == 0):
				train()
				target_train()
			episode_reward += reward
			if(episode_reward == -200):
				done = True
			memory.append((S_t.reshape(1,16),reward,S_t_1.reshape(-1,16),action,done))
			S_t = S_t_1
		episode = episode + 1
		reward_list.append(episode_reward)
		uncertainity_graph.append(coefficient_of_uncertainity)
		no_of_uncertainity.append(uncertainity)

		if(uncertainity ==  200):
			status = True
		if(episode % 1 == 0):
			
			#if(episode_reward ==  -200):
				#for temp in observation_list:
				#	print(temp)
			#print(action_list)
			print("episode_reward",episode_reward,"uncertainity: ",coefficient_of_uncertainity,"episode no:",episode,"moves:",moves)
		if(episode % 1000 ==0 and episode > 0):
			j = 0
			for model in model_ensemble:
				model.save(os.path.join(SAVE_PATH,'safe_gridworld_'+str(j)+str("_")+str(episode)))
				j = j + 1
	plot_reward(reward_list)
	plot_uncertainity(uncertainity_graph)
	plot_No_of_uncertain_situations(no_of_uncertainity)
	

main()
