# snake.ai - Multiplayer Snake AI

<p align="center">
  <img src="https://sds-dubois.github.io/img/projects/snake_game.gif" alt="Game demo" />
</p>

Read the [blog post](https://sds-dubois.github.io/2017/01/03/Multiplayer-Snake-AI.html) to get an overview of this project, 
as well as some details on the reinforcement learning methods implemented (Q-learning with function approximation by neural networks).


## Visualizing individual games

It is possible to run a single game with the GUI through the command 
```
$ python controller.py [h]
```
If you do use the option `h`, this will add a 'human player': an agent you can control with the keyboard.



## The config file

The config file `config.py` lets you configure the different agents or the details of the experiments/simulations 
you would like to run.
Here is an example configuration:
```
agent             = "RL"
filename          = "rl-pg-linear-r6-1000"
game_hp           = HP(grid_size = 20, max_iter = 3000, discount = 0.9)
rl_hp             = RlHp(rl_type = "policy_gradients", radius = 6, filter_actions = False, lambda_ = None, q_type = "linear")
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
opponents         = [SmartGreedyAgent, OpportunistAgent, searchAgent("alphabeta", depth, evalFn)]
num_trials        = 1000
```

Setting `agent` to `RL` or `ES` will add the corresponding agent to the `opponents` list (after training if necessary).
Setting it to anything else will keep this list unchanged.



## Running simulations

Once you filled the config file, you can simply run 500 (without the GUI) to get some stats about how the AIs perform against each other:
```
$ python simulation.py 500 [load] 
```

If you do use the `load` parameter, this will load pre-trained weights for the RL agents, otherwise it will first run trial games 
to learn such weights. In the latter case, learned weights will be saved in the `data/` folder with the name provided in the config
file.
For example, `simple-ql-r6.p` and `simple-pg-r6.p` contain the weights of RL agents trained respectively via Q-learning 
and Policy Gradients on 1,000 trials.  
We recommend training agents against hard-coded strategies instead of search-based ones such as Minimax (at least at first) 
since it will be much faster.

Basic statistics will be printed in the terminal, but these (and more) will be saved in a file in `experiments/` with the name
set in the config file. Note that the snakes' id correspond to the strategy's index in the list `opponents`.


## File structure

- `strategies.py` implements hard-coded strategies, especially useful to train RL agents or as baselines
- `minimax.py` implements adversarial strategies that expore trees of possible moves
- `rl.py` provides the interface for RL-based algorithms
- `rl_interface.py` provides utilities to train and load RL agents 
- `policy_gradients.py` implements a simple Policy Gradients algorithm for reinforcement learning
- `qlearning.py` implements Q-learning for reinforcement learning and supports both a simple linear model or neural nets
- `es.py` implements an Evolutionary Strategy algorithm
- `features.py` implements a `FeatureExtractor` to derive useful features from any state and used by RL agents
- `interface.py`, `agent.py`, `snake.py`, `move.py`, `hp.py` contain the general code for the game


## Have fun!

