from hp import *
from constants import *
from minimax import searchAgent, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc

agent             = "RL"
# agent             = "ES"
filename          = "rl-pg-linear-r6-1000"
# filename          = "rl-ql-linear-r6-1000"
# filename          = "es-linear-r6-50"
game_hp           = HP(grid_size = 20, max_iter = 3000, discount = 0.9)
rl_hp             = RlHp(rl_type = "policy_gradients", radius = 6, filter_actions = False, lambda_ = None, q_type = "linear")
# rl_hp             = RlHp(rl_type = "qlearning", radius = 6, filter_actions = False, lambda_ = None, q_type = "linear")
es_hp             = EsHp(radius = 6)
depth             = lambda s,a : survivorDfunc(s, a , 2, 0.5)
evalFn            = greedyEvaluationFunction
num_trials        = 1000
opponents         = [SmartGreedyAgent, OpportunistAgent]
# opponents         = [SmartGreedyAgent, OpportunistAgent, searchAgent("alphabeta", depth, evalFn)]
comment           = ""