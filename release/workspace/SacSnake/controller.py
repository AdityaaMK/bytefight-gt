from game import *
from game import player_board
from game.enums import Action
import numpy as np
from . import utils


from collections.abc import Callable


class PlayerController:
    """
    Controller that implements a trapping strategy when possible, or falls back to BFS.
    Prioritizes trapping the opponent's snake head when close enough and with sufficient length.
    """

    def __init__(self, time_left: Callable):
        """Initialize the player controller."""
        self.usesGPU = False
        self.search_depth = 5

    def bid(self, b: player_board.PlayerBoard, time_left: Callable):
        """Bid function - currently just returns 0."""
        return 0

    def play(self, b: player_board.PlayerBoard, time_left: Callable):
        """
        Main play function that decides between trapping strategy and BFS strategy.

        Args:
            b: The current game board state
            time_left: Function to get remaining time

        Returns:
            List of moves to execute
        """
        # Calculate length needed to reach and trap the opponent
        head_dist = self.get_dist(
            b.get_head_location(), b.get_head_location(enemy=True))
        length_needed = self.calculate_length_needed(head_dist)

        # Choose strategy based on available length
        if length_needed <= b.get_length() - b.get_min_player_size():
            print("Trap Case")
            moves = self.find_trap_sequence(b)
            if moves is not None and len(moves) > 0:
                return moves
            
            print("Trap sequence failed, using BFS fallback")
            move = self.find_bfs_move(b)
            if move is not None and len(move) > 0:
                return move
            print("No valid moves found, forfeiting")
            return [Action.FF]
        else: 
            # Fall back to BFS if trap sequence failed
            print("BFS Case")
            move = self.find_bfs_move(b)
            if move is not None and len(move) > 0:
                return move
            print("No valid moves found, forfeiting")
            return [Action.FF]


    def calculate_length_needed(self, distance):
        """
        Calculate the length needed to reach and trap an opponent at given distance.

        Args:
            distance: Distance to the target

        Returns:
            Total length needed
        """
        length_needed = 0
        cost = 0
        for i in range(distance):
            length_needed += cost
            cost += 2
        return length_needed

    def find_trap_sequence(self, b: player_board.PlayerBoard):
        """
        Find a sequence of moves to trap the opponent's head.

        Args:
            b: The current game board state

        Returns:
            List of moves that trap the opponent, or None if not possible
        """
        moves = []
        board_copy = b.get_copy()
        
        current_cost = 0
        total_cost = 0
        # Check if opponent is already trapped
        enemy_moves = self.count_valid_moves(board_copy, enemy=True)

        # Keep making moves until we trap the opponent or run out of length
        while total_cost <= b.get_length() - b.get_min_player_size():
            if enemy_moves == 0:
                print("Successfully trapped opponent!")
                return moves

            # Find moves that reduce opponent's available moves
            best_moves = []
            best_reduction = 0

            for move in board_copy.get_possible_directions():
                if board_copy.is_valid_move(move):
                    # Simulate the move
                    next_board, success = board_copy.forecast_action(move)
                    if success:
                        # Count opponent's moves after this move
                        enemy_moves_after = self.count_valid_moves(
                            next_board, enemy=True)
                        reduction = enemy_moves - enemy_moves_after

                        if reduction > best_reduction:
                            best_reduction = reduction
                            best_moves = [move]
                        elif reduction == best_reduction:
                            best_moves.append(move)

            if not best_moves:
                break

            # Try each best move and see which one leads to the best outcome
            best_move = None
            best_future_reduction = -1

            for move in best_moves:
                # Simulate this move
                next_board, success = board_copy.forecast_action(move)
                if success:
                    # Look ahead one more step to see which move leads to better future reductions
                    future_reduction = 0
                    for next_move in next_board.get_possible_directions():
                        if next_board.is_valid_move(next_move):
                            next_next_board, next_success = next_board.forecast_action(
                                next_move)
                            if next_success:
                                future_enemy_moves = self.count_valid_moves(
                                    next_next_board, enemy=True)
                                future_reduction = max(future_reduction,
                                                       self.count_valid_moves(next_board, enemy=True) - future_enemy_moves)

                    if future_reduction > best_future_reduction:
                        best_future_reduction = future_reduction
                        best_move = move

            # Apply the best move
            moves.append(best_move)
            board_copy.apply_move(best_move)
            current_cost += 2  # Cost increases by 2 for each move
            total_cost += current_cost
            # Update enemy moves for next iteration
            enemy_moves = self.count_valid_moves(board_copy, enemy=True)

        # If we couldn't trap the opponent, return None
        if enemy_moves != 0:
            print("Could not completely trap opponent")
            return None

        print("Trap moves: ", moves)
        return moves

    def count_valid_moves(self, board, enemy=False):
        """
        Count the number of valid moves for a player.

        Args:
            board: The game board state
            enemy: Whether to count for the enemy (True) or player (False)

        Returns:
            Number of valid moves
        """
        return len([move for move in board.get_possible_directions(enemy=enemy)
                    if board.is_valid_move(move, enemy=enemy)])

    def find_bfs_move(self, b: player_board.PlayerBoard):
        """
        Find a move using BFS strategy.

        Args:
            b: The current game board state

        Returns:
            List of moves using BFS strategy
        """
        moves = []

        # First try to find a path to the nearest apple
        my_move = utils.get_move_to_nearest_apple(b)

        # If no path to apple or not safe, find move that maximizes space
        if (my_move is None or utils.safe_after_apple(b, my_move) is False):
            my_move = utils.get_best_move_spaces(b)

        print("got move:", my_move)

        # Apply the move
        moves.append(my_move)
        b.apply_move(my_move)

        # Check if we can place traps after moving
        while (self.check_trap(b)):
            moves.append(Action.TRAP)
            b.apply_trap()

        print("BFS move: ", moves)
        return moves

    def check_trap(self, b: player_board.PlayerBoard):
        """
        Check if placing a trap is valid and useful.

        Args:
            b: The current game board state

        Returns:
            Boolean indicating if a trap should be placed
        """
        return b.is_valid_trap() and self.get_dist(b.get_tail_location(), b.get_head_location(enemy=True)) <= 2

    def get_dist(self, loc_1, loc_2):
        """
        Calculate the Chebyshev distance between two locations.

        Args:
            loc_1: First location
            loc_2: Second location

        Returns:
            Chebyshev distance between locations
        """
        return np.max(np.abs(loc_1 - loc_2))
