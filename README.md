# Description
This repository contains my work on Safe-RL as a student. In reinforcement learning there is a longstanding question of how to combine exploration and exploitation, however, in the real world this paradigm may lead to undesired behavior. This work aims to bring a theoretically grounded paradigm into practice – KWIK (knows what it knows). The goal is to combine a learning agent with a pre-defined behavior policy (pre-trained baseline or even a human operator). The agent should then complement the behavior policy by only taking over when it feels it is better with high confidence and on the other hand passing back control to a baseline policy once it is no longer confident in its actions. We developed our Safe-RL framework for both discrete and continuous action based envs.

## Usage
Check the code section of the repository to know how to use the files.

