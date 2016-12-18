# snake.ai - Multiplayer Snake AI

## Running simulations
You can configure simulations by filling the file `config.py` and then running 1000 simulations through the command:
```
$ python simulation.py 1000 [load] 
```
If you do use `load`, this will load pre-trained weights for the RL agents, otherwise it will first run trials to learn
such weights. The file set by default corresponds to configuration 2b in the report. However this can be changed to any file
available in `data/`; information about the learning trials is available in `info/` (read 'Minimax' instead of 'getAction' in 
these files). 

## The config file
In the config file, the parameter `agent` determines which method is to be tested. Setting `agent == "RL"` means that an RL
agent will be add to the list of players (defined by `opponents`). Setting `agent == "AlphaBeta"` will add a Minimax agent instead
with depth and evaluation functions set in the config file. It is also possible to let the RL agent play against a Minimax agent by
adding (for example)
```
AlphaBetaAgent(depth=lambda s,a: survivorDfunc(s, a, 2, 0.5), evalFn=greedyEvaluationFunction).getAction
```
to the list of opponents.

Basic statistics will be printed in the terminal, but these (and more) will be saved in a file in `experiments/` with the name
set in the config file. Note that the snakes' id correspond to the strategy's index in the list `opponents`.

## Visualizing individual games
Finally it is possible to run a single game with the GUI through the command 
```
$ python controller.py
```
Note that you can play by simply adding `humanStrategy` in the list of strategies in the file `controller.py` (see the commented line).


## Have fun!