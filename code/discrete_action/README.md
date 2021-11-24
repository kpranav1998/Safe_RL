# Description
This section of the code is to use the Safe RL framework for discrete action envs. I have used an exisiting implemenation of [Bootstrap DQN by Johannah](https://github.com/johannah/bootstrap_dqn) to build my Safe RL framework for discrete envs. 

## Requirements
Check out the dependencies from the original repository [Bootstrap DQN by Johannah](https://github.com/johannah/bootstrap_dqn)

## Usage
The file run_with_baseline.py is used to run an ensemble of RL agents within the Safe RL framework. To run the file pass the path of an appropriate baseline and set the appropriate hyperparameters in the file.

To run an normal ensemble of SAC agents with the Safe RL framework use the following code
```bash
python3 run_with_baseline.py --alpha 0.2
```
To run an normal ensemble of SAC agents use the following code
```bash
python3 run_bootstrap.py
```
### Arguments
Check the file run_with_baseline.py to know the arguments. They are self explanatory in nature.
