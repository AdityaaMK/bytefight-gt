from game import *
from game import player_board
from game.enums import Action
import numpy as np
# from .utils import self.get_dist
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
            
        nearest_apple = self.get_nearest_apple(b)
        if (nearest_apple is None):
            my_move = self.get_open_move(b)
            moves.append(my_move)
            b.apply_move(my_move)
        else:
            my_move = self.get_move_to_nearest_apple(b, nearest_apple)
            moves.append(my_move)
            b.apply_move(my_move)
            

        while(self.check_trap(b)):
            moves.append(Action.TRAP)
            b.apply_trap()

        print(moves)

        return moves

    def check_trap(self, b:player_board.PlayerBoard):
        return b.is_valid_trap() and self.get_dist(b.get_tail_location(), b.get_head_location(enemy=True)) <= 2

    def get_spaces_around_vectorized(self, b:player_board.PlayerBoard):
        head_x, head_y = b.get_head_location()

        index_y = slice(max(0, head_y-2), min(b.get_dim_y(), head_y+2))
        index_x = slice(max(0, head_x-2), min(b.get_dim_x(), head_x+2))
        
        mask = b.player_cells[index_y, index_x] + b.enemy_cells[index_y, index_x] + b.game_board.map.cells_walls[index_y, index_x]
        return 25 - np.sum(mask)


    def get_spaces_around(self, b:player_board.PlayerBoard):
        head_x, head_y = b.get_head_location()

        total = 0
        for x in range (max(0, head_x-2), min(b.get_dim_x(), head_x+2)):
            for y in range (max(0, head_y-2), min(b.get_dim_y(), head_y+2)):
                if(not b.is_occupied(x, y)):
                    total += 1
        return total

    def get_open_move(self, b: player_board.PlayerBoard):
        best_spaces = 0
        best_move = Action.FF
        for move in b.get_possible_directions():
            if(b.is_valid_move(move)):
                new_board, ok = b.forecast_move(move)
                spaces = self.get_spaces_around(new_board)

                if(spaces > best_spaces):
                    best_spaces = spaces
                    best_move = move

        return best_move
    
    def get_move_to_nearest_apple(self, b: player_board.PlayerBoard, apple):
        player_head = b.get_head_location()

        best_move = None
        best_dist = float("infinity")

        for move in b.get_possible_directions():
            if(b.is_valid_move(move)):
                next_head_loc = b.get_loc_after_move(move)
                
                # manhattan distance
                new_dist = self.get_dist(next_head_loc, apple)

                if (new_dist  < best_dist):
                    best_move = move
                    best_dist = new_dist


        if(best_move is None):
            return self.get_open_move(b)
        return best_move


    def get_nearest_apple(self, b: player_board.PlayerBoard):
        player_head = b.get_head_location()
        apples = b.get_current_apples()

        if len(apples) < 1:
            return None

        min_apple = (float("infinity"), float("infinity"))
        min_apple_dist = float("infinity")

        for apple in apples:
            if(not b.is_occupied(apple[0], apple[1])):
                dist = self.get_dist(apple, player_head)
                if (dist < min_apple_dist):
                    min_apple_dist = dist
                    min_apple = apple

        return min_apple

    def get_dist_vectorized(self, loc_1, loc_2):
        return np.sum(np.abs(loc_1 - loc_2))

    def get_dist(self, loc_1, loc_2):
        return abs(loc_1[0]-loc_2[0]) + abs(loc_1[1]-loc_2[1]) 
            