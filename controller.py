__author__ = 'Real_Seb'

from interface import Game

def controller(strategies, grid_size, candy_ratio=1., max_iter=None):
    game = Game(grid_size, len(strategies), candy_ratio=candy_ratio, max_iter=max_iter)
