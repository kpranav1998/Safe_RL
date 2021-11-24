# Description
This section of the code is to use the Safe RL framework for continous action envs. I have used an exisiting implemenation of [Soft Actor Critic](https://github.com/pranz24/pytorch-soft-actor-critic) to build my Safe RL framework. 

## Requirements
*   [mujoco-py](https://github.com/openai/mujoco-py)
*   [PyTorch](http://pytorch.org/)

## Usage
The file main_safe.py is used to run an ensemble of RL agents within the Safe RL framework. To run the file pass the path of an appropriate baseline and set the appropriate hyperparameters in the file.

```bash
python3 main_safe.py --alpha 0.2
```
To run an normal ensemble of SAC agents use the following code
```bash
python3 main.py --alpha 0.2
```
### Arguments
------------
```
Safe RL Framework arguements

optional arguments:
  -h, --help              show this help message and exit
  --env-name ENV_NAME     Mujoco Gym environment
  --safe_path             Path to baseline
  --baseline_performance  Value of the baseline(score)
  --policy POLICY         Policy Type: Gaussian | Deterministic
  --eval EVAL             Evaluates a policy a policy every 10 episode
  --gamma G               discount factor for reward
  --tau G                 target smoothing coefficient(τ)
  --lr G                  learning rate
  --alpha G               Temperature parameter α determines the relative
                        importance of the entropy term against the reward                     
  --automatic_entropy_tuning  Automaically adjust α
  --seed N                random seed
  --batch_size N          batch size
  --num_steps N           maximum number of steps
  --hidden_size N         hidden size
  --updates_per_step N    model updates per simulator step
  --start_steps N         Steps sampling random actions
  --target_update_interval Value target update per no. of updates per step                        
  --replay_size N         size of replay buffe
  --cuda                  run on CUDA
```

