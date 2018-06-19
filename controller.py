import sys, pickle
import pygame
import gui
import move
from hp import load_from
from interface import Game,Snake
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy,humanStrategy
from minimax import searchAgent, cowardCenterDepthFunction, cowardDepthFunction, greedyEvaluationFunction, smartCowardDfunc, survivorDfunc
from rl_interface import rl_strategy, load_rl_strategy
from es import es_strategy, load_es_strategy
from features import FeatureExtractor
from pdb import set_trace as t
from constants import *

def controller(strategies, grid_size, candy_ratio = 1., max_iter = None, verbose = 0, gui_active = False, game_speed = None):
    # Pygame Init
    pygame.init()
    clock = pygame.time.Clock()
    if gui_active:    
        gui_options = gui.Options()
        win = gui.Window(grid_size,'Multiplayer Snake', gui_options)
        quit_game = False
    
    # Start Game
    game = Game(grid_size, len(strategies), candy_ratio = candy_ratio, max_iter = max_iter)
    # state = game.startState()
    state = game.start(strategies)
    prev_human_action = None
    game_over = False

    agent_names = [a.name for a in strategies]
    i_human = None
    if "human" in agent_names:
        i_human = agent_names.index("human")

    while not ((gui_active and quit_game) or ((not gui_active) and game_over)):
        # Print state
        if verbose > 0:
            state.printGrid(game.grid_size)
        # Get events
        if gui_active:
            events = pygame.event.get()         
            if pygame.QUIT in [ev.type for ev in events]:
                quit_game = True 
                continue   


        # Compute the actions for each player following its strategy (except human)
        actions = game.agentActions()

        # Compute human strategy if necessary
        human_action = None 
        if i_human is not None:
            speed = 2. if pygame.K_SPACE in [ev.key for ev in events if ev.type == pygame.KEYDOWN] else 1.
            arrow_key = False 
            for event in events:                               
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        human_action = move.Move((-1,0),speed)
                        arrow_key = True
                    if event.key == pygame.K_RIGHT:
                        human_action = move.Move((1,0),speed)
                        arrow_key = True
                    if event.key == pygame.K_UP:
                        human_action = move.Move((0,-1),speed)
                        arrow_key = True
                    if event.key == pygame.K_DOWN:
                        human_action = move.Move((0,1),speed)
                        arrow_key = True      
                    
            if not arrow_key and prev_human_action is None:
                human_action = move.Move((0,-1),speed)
            elif not arrow_key:
                human_action = prev_human_action
 
        # Assign human action
        if i_human is not None and i_human in actions.keys(): 
            actions[i_human] = human_action 
            prev_human_action = human_action

        if verbose > 1:
            print state
            print actions

        # Update the state
        if not game_over:
            state = game.succ(state, actions, copy = False)
        # Pause
        if game_speed:
            clock.tick(game_speed)

        # Check if game over
        game_over = game.isEnd(state)
        # if game_over:
           # win.print_message('GAME OVER')
        
        # Update gui   
        if gui_active:     
            win.updateSprites(state)
            win.refresh()
        
    if verbose > 0:
        state.printGrid(game.grid_size)

    return state

if __name__ ==  "__main__":

    human_player = False
    if len(sys.argv) > 1 and sys.argv[1] == "h":
        human_player = True

    max_iter = None
    minimax_agent = searchAgent("minimax", depth = lambda s,a : 2)
    alphabeta_agent = searchAgent("alphabeta", depth = lambda s,a : survivorDfunc(s, a, 4, 0.5), evalFn = greedyEvaluationFunction)
    expectimax_agent = searchAgent("expectimax", depth = lambda s,a : cowardCenterDepthFunction(s, a, 2), evalFn = greedyEvaluationFunction)
    
    # strategies = [SmartGreedyAgent, OpportunistAgent]
    strategies = [SmartGreedyAgent, OpportunistAgent, alphabeta_agent]

    # add an RL agent
    # rl_hp = load_from("rl-pg-linear-r6-1000.p")
    rl_hp = load_from("rl-ql-linear-r6-1000.p")
    # rl_hp = load_from("nn-r6-assisted.p")
    featureExtractor = FeatureExtractor(len(strategies), grid_size = 20, radius_ = rl_hp.radius)
    rlStrategy = load_rl_strategy(rl_hp, strategies, featureExtractor)
    # rlStrategy = load_rl_strategy(rl_hp, strategies, featureExtractor)

    # esStrategy = load_es_strategy("es-linear-r4-50.p", strategies, featureExtractor, discount = 0.9)

    # strategies.append(esStrategy)
    strategies.append(rlStrategy)

    if human_player:
        strategies.append(HumanAgent)

    controller(strategies, 20, max_iter = max_iter, gui_active = True, verbose = 0, game_speed = 10)