# end-to-end-negotiator

Implementation of following paper in Tensorflow

Name: Deal or No Deal? End-to-End Learning for Negotiation Dialogues

Link: https://arxiv.org/abs/1706.05125

Original repository: https://github.com/facebookresearch/end-to-end-negotiator


- run_sv_versus_rl.py - for generating sample conversation between supervised trained model and model trained with reinforcement learning, optionally you may set USE_ROLLOUTS to True to make RL agent perform rollouts (see paper). Sample logs available at *.log files.

- sv_agent.py, rl_agent.py define Models and Agents that write and read the dialogue.

- train_sv.py trains model in supervised setting, from data.

- train_rl.py initialize sv_agent, rl_agent from pretrained model and continue training rl_agent with policy gradient

- model folder contains trained models

- end-to-end-negotiator/src/data/negotiate contains data from original repository


Results (higher the better, min 0, max 10):

SV versus RL normal (without rollouts): 4.93 versus 6.20

SV versus RL with rollouts: 4.57 versus 7.05 

