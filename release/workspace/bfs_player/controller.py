from game import *
from game import player_board
from game.enums import Action
import numpy as np
from . import utils

from collections.abc import Callable
class PlayerController:
    # for the controller to read
    def __init__(self, time_left: Callable):
        self.usesGPU = False


    def bid(self, b:player_board.PlayerBoard, time_left: Callable):
        return 0

    def play(self, b:player_board.PlayerBoard, time_left:Callable):
        moves = []

        while(self.check_trap(b)):
            moves.append(Action.TRAP)
            b.apply_trap()
            
        my_move = utils.get_move_to_nearest_apple(b)
        if (my_move is None):
            my_move = utils.get_best_move_spaces(b)

        moves.append(my_move)
        b.apply_move(my_move)
            
        while(self.check_trap(b)):
            moves.append(Action.TRAP)
            b.apply_trap()

        return moves

    def check_trap(self, b:player_board.PlayerBoard):
        return b.is_valid_trap() and self.get_dist(b.get_tail_location(), b.get_head_location(enemy=True)) <= 2
            
    def get_dist(self, loc_1, loc_2):
        return np.max(np.abs(loc_1 - loc_2))