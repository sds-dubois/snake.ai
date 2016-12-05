import sys
import pygame
import gui
import move
from interface import Game,Snake
from strategies import randomStrategy, greedyStrategy, smartGreedyStrategy, opportunistStrategy,humanStrategy
from minimax import MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent, greedyEvaluationFunction, cowardDepthFunction, cowardCenterDepthFunction
from rl import rl_strategy, load_rl_strategy, simpleFeatureExtractor0, simpleFeatureExtractor1, simpleFeatureExtractor2
from pdb import set_trace as t

def controller(strategies, grid_size, candy_ratio = 1., max_iter = None, verbose = 0, gui_active = False, game_speed = None):
    # Pygame Init
    pygame.init()
    clock = pygame.time.Clock()
    if gui_active:    
        gui_options = gui.Options()
        win = gui.Window(grid_size,'Multiplayer Snake',gui_options)
        quit_game = False
    
    # Start Game
    game = Game(grid_size, len(strategies), candy_ratio = candy_ratio, max_iter = max_iter)
    state = game.startState()
    prev_action = None
    game_over = False
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

        # Compute human strategy if necessary
        human_action = None 
        i_human = None
        if humanStrategy in [strategies[i] for i in state.snakes.keys()]:
            i_human = strategies.index(humanStrategy)
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
                    
            if not arrow_key:
                human_action = prev_action
        # Compute the actions for each player following its strategy (except human)
        actions = {i: strategies[i](i, state) for i in state.snakes.keys() if i!=i_human}
 
        # Assign human action
        if human_action != None: 
            actions[i_human] = human_action 
            prev_action = human_action

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
    if len(sys.argv) > 1:
        max_iter = int(sys.argv[1])
    else:
        max_iter = None


    minimax_agent = MinimaxAgent(depth=lambda s,a: 2)
    alphabeta_agent = AlphaBetaAgent(depth=lambda s,a: cowardCenterDepthFunction(s, a, 2), evalFn=greedyEvaluationFunction)
    expectimax_agent = ExpectimaxAgent(depth=lambda s,a: cowardCenterDepthFunction(s, a, 2), evalFn=greedyEvaluationFunction)
    controller([expectimax_agent.getAction, alphabeta_agent.getAction],
               20, max_iter = max_iter, gui_active = True, verbose = 0, game_speed = 10)

    # rlStrategy = load_rl_strategy("d-weights1.p", [opportunistStrategy], simpleFeatureExtractor1)
    # rlStrategy = load_rl_strategy("d-weights5.p", [randomStrategy, smartGreedyStrategy, opportunistStrategy], simpleFeatureExtractor1)

    # strategies = [opportunistStrategy, rlStrategy]
    # strategies = [randomStrategy, smartGreedyStrategy, opportunistStrategy, rlStrategy]

    # controller(strategies, 20, max_iter = max_iter, gui_active = True, verbose = 0, game_speed = 10)