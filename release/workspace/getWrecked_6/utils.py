from game import player_board
from game.game_queue import Queue
import numpy as np
from game.enums import Action, Result
# from collections import deque


INFTY = float("inf")
NEG_INFTY = -float("inf")


def get_move_to_nearest_apple(b: player_board.PlayerBoard):
    # format: x, y, snake current direction, initial move leading to this cell
    q = Queue(dim=4)
    print("b possible actions")
    for initial_move in b.get_possible_directions():  # enqueue all possible cells you can move to
        if (b.is_valid_move(initial_move)):
            loc_x, loc_y = b.get_loc_after_move(initial_move)
            q.push((loc_x, loc_y, initial_move, initial_move))

    # visited array (x by y by direction)
    visited = np.zeros([b.get_dim_y(), b.get_dim_x(), 8])

    while (not q.is_empty()):  # start bfs from each move
        x, y, snake_dir, initial_move = q.pop()

        if (b.has_apple(x, y)):  # return initial move that ends up yielding a path to this apple
            return initial_move

        # if no apple, enqueue all cells reachable from (x, y, dir)
        for next_move in b.player_snake.get_valid_directions(direction=snake_dir):
            x_new, y_new = b.player_snake.get_next_loc(
                next_move, head_loc=(x, y))

            if (b.game_board.is_valid_cell((x_new, y_new)) and visited[y_new, x_new, next_move] == 0):
                # make sure to pass along the inital move that started this path
                q.push((x_new, y_new, next_move, initial_move))
                visited[y_new, x_new, next_move] = 1

    return None



def get_best_move_spaces(b: player_board.PlayerBoard):
    best_move = None
    best_open = -1

    start_x, start_y = b.get_head_location()

    # visited initalization inside for loop because each move needs seperate occupancy grid
    for initial_move in b.get_possible_directions():
        visited = np.zeros([b.get_dim_y(), b.get_dim_x(), 8])

        # if inital move is valid, check the number of squares we can reach using bfs
        if (b.is_valid_move(initial_move)):

            # set current head cell to visited
            visited[start_y, start_x, :] = 1

            q = Queue(dim=3)  # format: x, y, snake current direction
            loc_x, loc_y = b.get_loc_after_move(initial_move)

            q.push(np.array([loc_x, loc_y, int(initial_move)]))

            while (not q.is_empty()):  # start bfs
                x, y, snake_dir = q.pop()

                for next_move in b.player_snake.get_valid_directions(direction=snake_dir):
                    x_new, y_new = b.player_snake.get_next_loc(
                        next_move, head_loc=(x, y))

                    if (b.game_board.is_valid_cell((x_new, y_new)) and visited[y_new, x_new, next_move] == 0):
                        q.push((x_new, y_new, next_move))
                        visited[y_new, x_new, next_move] = 1

        # in the end, total the number of cells we reached using this initial move
        # (don't double-count visiting the cell in multiple different directions)
        total_open = np.count_nonzero(np.sum(visited, axis=2))

        # maximize open squares
        if (total_open > best_open):
            best_open = total_open
            best_move = initial_move

    return best_move


def safe_after_apple(b: player_board.PlayerBoard, move: Action) -> bool:
    # Create a copy of the board to simulate the move without affecting the actual game
    board_copy = b.get_copy()
    # Apply the move that would consume the apple
    board_copy.apply_move(move)

    # Check possible directions after the move
    for next_move in board_copy.get_possible_directions():
        if board_copy.is_valid_move(next_move):
            return True
    return False


class MinimaxPlayer():
    def __init__(self):
        self.search_depth = 5

    def calculate_heuristic(self, b: player_board.PlayerBoard):
        # Get the number of possible moves for both players
        player_moves = len([move for move in b.get_possible_directions(
            is_enemy=False) if b.is_valid_move(move)])
        opponent_moves = len([move for move in b.get_possible_directions(
            is_enemy=True) if b.is_valid_move(move)])

        # Add a significant bonus if opponent has no moves (we win)
        if opponent_moves == 0:
            return 1000

        # Add a significant penalty if we have no moves (we lose)
        if player_moves == 0:
            return -1000

        # Simply return the difference in available moves
        # Positive means we have more moves than the opponent
        return player_moves - opponent_moves

    def minimax(self, b: player_board.PlayerBoard, depth, is_maximizing_player):
        if depth == 0 or b.is_game_over():
            return self.calculate_heuristic(b), None

        possible_moves = b.get_possible_directions(
            is_enemy=not is_maximizing_player)
        if is_maximizing_player:
            best_score = NEG_INFTY
            best_move = None
            for move in possible_moves:
                next_board, success = b.forecast_turn(move)
                if success:
                    score, _ = self.minimax(next_board, depth - 1, False)
                    if score > best_score:
                        best_score = score
                        best_move = move
            if best_move is None:
                return NEG_INFTY, None
            return best_score, best_move
        else:
            best_score = INFTY
            best_move = None
            for move in possible_moves:
                next_board, success = b.forecast_turn(move)
                if success:
                    score, _ = self.minimax(next_board, depth - 1, True)
                    if score < best_score:
                        best_score = score
                        best_move = move
            if best_move is None:
                return INFTY, None
            return best_score, best_move
