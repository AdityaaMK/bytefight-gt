from game import player_board
from game.game_queue import Queue
import numpy as np
from game.enums import Action, Result

INFTY = float("inf")
NEG_INFTY = -float("inf")

# collecting info, to tune search_depth parameter
safe_apple_count = 0
unsafe_apple_count = 0


def print_board_info(b: player_board.PlayerBoard):
    print("=== Board Information ===")
    print("Dimensions: {} x {}".format(b.get_dim_x(), b.get_dim_y()))
    print("Turn Count: {}".format(b.get_turn_count()))
    print("Is my turn?: {}".format(b.is_my_turn()))
    print()

    # Snake positions and lengths
    print("Player Snake:")
    print("  Head Location: {}".format(b.get_head_location()))
    print("  Tail Location: {}".format(b.get_tail_location()))
    # print("  All Locations: {}".format(b.get_all_locations()))
    print("  Physical Length: {}".format(b.get_unqueued_length()))
    print("  Total Length (including queued): {}".format(b.get_length()))
    # print("  Apples Eaten: {}".format(b.get_apples_eaten()))
    print()

    print("Opponent Snake:")
    print("  Head Location: {}".format(b.get_head_location(enemy=True)))
    print("  Tail Location: {}".format(b.get_tail_location(enemy=True)))
    # print("  All Locations: {}".format(b.get_all_locations(enemy=True)))
    print("  Physical Length: {}".format(b.get_unqueued_length(enemy=True)))
    print("  Total Length (including queued): {}".format(b.get_length(enemy=True)))
    # print("  Apples Eaten: {}".format(b.get_apples_eaten(enemy=True)))
    print()

    # # Board masks and additional information
    # print("Wall Mask:")
    # print(b.get_wall_mask())
    # print("Apple Mask:")
    # print(b.get_apple_mask())
    # print("Trap Mask (Player - Enemy):")
    # print(b.get_trap_mask())
    # print("Portal Mask:")
    # print(b.get_portal_mask(descriptive=True))
    # print()

    # # Future apples and decay information
    # print("Future Apples:")
    # print(b.get_future_apples())
    # print("Current Decay Interval: {}".format(b.get_current_decay_interval()))
    # print("Future Decay Intervals:")
    # print(b.get_future_decay_intervals())
    # print("==========================")


def calculate_length_needed(distance):
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


def find_trap_sequence(b: player_board.PlayerBoard):
    """
    Find a sequence of moves to trap the opponent's head.

    Args:
        b: The current game board state

    Returns:
        List of moves that trap the opponent, or None if not possible
    """
    print("finding trap sequence: ")

    moves = []
    board_copy = b.get_copy()
    print_board_info(board_copy)
    current_cost = 0
    total_cost = 0
    # Check if opponent is already trapped
    enemy_moves = count_valid_moves(board_copy, enemy=True)

    # Keep making moves until we trap the opponent or run out of length
    while total_cost <= b.get_length() - b.get_min_player_size():
        if enemy_moves == 0:
            print("Successfully trapped opponent!")
            return moves

        # Find moves that reduce opponent's available moves
        best_moves = []
        best_reduction = 0
        # cost 0, 2
        for move in board_copy.get_possible_directions():
            print(f"trap move candidate: {move}")
            if board_copy.is_valid_move(move):
                # Simulate the move
                next_board, success = board_copy.forecast_action(move)
                if success:
                    print('reached successful simulation state')
                    # Count opponent's moves after this move
                    enemy_moves_after = count_valid_moves(
                        next_board, enemy=True)
                    reduction = enemy_moves - enemy_moves_after
                    if reduction == enemy_moves:
                        moves.append(move)
                        return moves
                    elif reduction > best_reduction:
                        best_reduction = reduction
                        best_moves = [move]
                    elif reduction == best_reduction:
                        best_moves.append(move)

        if not best_moves:
            break

        current_cost += 1
        total_cost += current_cost
        if total_cost > b.get_length() - b.get_min_player_size():
            break

        # Try each best move and see which one leads to the best outcome
        best_move = None
        best_future_reduction = -1
        # cost 1, 3
        for move in best_moves:
            # Simulate this move
            next_board, success = board_copy.forecast_action(move)
            if success:
                # Look ahead one more step to see which move leads to better future reductions
                future_reduction = 0
                for next_move in next_board.get_possible_directions():
                    if next_board.is_valid_move(next_move):
                        print("got to valid trap moves")
                        next_next_board, next_success = next_board.forecast_action(
                            next_move)
                        if next_success:
                            future_enemy_moves = count_valid_moves(
                                next_next_board, enemy=True)
                            future_reduction = max(future_reduction,
                                                   count_valid_moves(next_board, enemy=True) - future_enemy_moves)

                if future_reduction > best_future_reduction:
                    best_future_reduction = future_reduction
                    best_move = move

        # Apply the best move
        moves.append(best_move)
        print("candidate moves: ", moves)
        board_copy.apply_move(best_move)
        current_cost += 1  # Cost increases by 2 for each move
        total_cost += current_cost
        # Update enemy moves for next iteration
        enemy_moves = count_valid_moves(board_copy, enemy=True)
        print("remaining enemy moves, ", enemy_moves)

    # If we couldn't trap the opponent, return None
    if enemy_moves != 0:
        print("Could not completely trap opponent")
        return None


def count_valid_moves(board, enemy=False):
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


"""
    Check if applying my_move would allow the enemy snake to trap us.
    
    We simulate the move on a board copy and then reverse the board's perspective.
    If a sequence exists, it means that after my_move, the enemy has a strategy to trap us.
"""


def can_enemy_trap_me(b: player_board.PlayerBoard, my_move):

    print("Simulating our move, can enemy trap me")
    board_copy = b.get_copy()  # initial state
    print_board_info(board_copy)
    # apply the move
    if not board_copy.apply_move(my_move, check_validity=True):
        print("invalid move")
        return []
    # now it is the enemy's turn
    board_copy.reverse_perspective()
    print_board_info(board_copy)

    trap_seq = find_trap_sequence(board_copy)
    print("Trap Seq", trap_seq)
    return trap_seq

# Simulate moving from (x, y) continuously in the given direction
# count how many before hitting invalid cell or colliding with obstacle


def dfs_depth(b: player_board.PlayerBoard, x, y, max_depth, visited=None):
    if max_depth == 0:
        return 0
    if visited is None:
        visited = set()

    best_depth = 0
    for move in b.get_possible_directions():
        nx, ny = b.player_snake.get_next_loc(move, head_loc=(x, y))
        # Skip if the next cell is invalid or occupied.
        if not b.game_board.is_valid_cell((nx, ny)):
            continue
        if b.is_occupied(nx, ny):
            continue
        if (nx, ny) in visited:
            continue

        visited.add((nx, ny))
        depth = 1 + dfs_depth(b, nx, ny, max_depth - 1, visited)
        best_depth = max(best_depth, depth)
        visited.remove((nx, ny))
    return best_depth


# The apple is on a target square, run dfs from the target square
# ensure that we can go a certain safe depth in at least one direction from the square
# for now making safe_depth the size of the board in the direction that we are going - starting position
# for each path, we simulate the snakes movements, the point being that we also want to see if we collide with the tail
def get_square_safety(b: player_board.PlayerBoard, x, y):

    x_size, y_size = b.get_dim_x(), b.get_dim_y()
    # set the maximum search depth as the sum of the board's dimensions (an upper bound).
    safe_depth_threshold = max(x_size // 2, y_size // 2)
    depth = dfs_depth(b, x, y, safe_depth_threshold)
    # consider the square safe if the depth is at least half of the board’s dimension.
    is_safe = depth >= 4
    print("SQUARE SAFETY: ", depth, is_safe)
    return depth, is_safe


# def get_square_safety(b: player_board.PlayerBoard, x, y):
#     b_cpy = b.get_copy()
#     x_size,y_size = b_cpy.get_dim_x(),b_cpy.get_dim_y()
#     safe_depth_threshold = x_size + y_size
#     max_depth_found = 0
#     for move in b_cpy.get_possible_directions():
#         depth = dfs_depth(b_cpy, x, y, move, safe_depth_threshold)
#         if depth > max_depth_found:
#             max_depth_found = depth
#     # print("SQUARE SAFETY: ", max_depth_found, max_depth_found >= max(x_size // 2, y_size // 2))
#     return max_depth_found, max_depth_found >= 4 # max(x_size // 2, y_size // 2)

def get_move_to_nearest_apple(b: player_board.PlayerBoard):
    # format: x, y, snake current direction, initial move leading to this cell
    q = Queue(dim=4)
    for initial_move in b.get_possible_directions():  # enqueue all possible cells you can move to
        # do not queue a move that causes our snake to step into danger of entrapment.
        if can_enemy_trap_me(b, initial_move) is not None:
            print("Detected trap")
            continue
        # if find_trap_sequence_enemy(b,initial_move) is not None:
        #     continue
        if (b.is_valid_move(initial_move)):
            loc_x, loc_y = b.get_loc_after_move(initial_move)
            q.push((loc_x, loc_y, initial_move, initial_move))

    # visited array (x by y by direction)
    visited = np.zeros([b.get_dim_y(), b.get_dim_x(), 8])

    while (not q.is_empty()):  # start bfs from each move
        x, y, snake_dir, initial_move = q.pop()

        if (b.has_apple(x, y)):  # return initial move that ends up yielding a path to this apple
            print("APPLE SQUARE: ", (x, y))
            _, safe_square = get_square_safety(b, x, y)
            # b_cpy = b.get_copy()

            if safe_square:
                # if can_enemy_trap_me(b, initial_move) is not None:
                #     print("Detected trap")
                return initial_move
            else:
                print(f"apple at ({x},{y}), rejected, not sufficient depth")

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
            if can_enemy_trap_me(b, initial_move) is not None:
                print("Detected trap")
                continue

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
