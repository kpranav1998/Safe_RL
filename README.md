# Safe_RL

<p> This repository contains my work on Safe-RL as a student. In reinforcement learning there is a longstanding question of how to combine exploration and exploitation, however, in the real world this paradigm may lead to undesired behavior. This work aims to bring a theoretically grounded paradigm into practice – KWIK (knows what it knows). The goal is to combine a learning agent with a pre-defined behavior policy (pre-trained baseline or even a human operator). The agent should then complement the behavior policy by only taking over when it feels it is better with high confidence and on the other hand passing back control to a baseline policy once it is no longer confident in its actions. We developed our Safe-RL framework for both discrete and continuous action based envs.</p>
  
<p>I have used two repositories to build my code upon. 
 
The <a href = "https://github.com/johannah/bootstrap_dqn">implementation of bootstrap DQN by Johanna</a> was taken as a reference to build the code for Atari based envs.</h2> and  <a href = "https://github.com/pranz24/pytorch-soft-actor-critic">Pytorch implementation of Soft Actor Critic by Pranz24</a> was used as a reference to build for Mujoco based envs </p>

<h4> All the code files are in the code folder while the results for each environment are under its own heading. </h4>



