from game import player_board
from game.game_queue import Queue
import numpy as np

def get_move_to_nearest_apple(b:player_board.PlayerBoard):
    q = Queue(dim=4) # format: x, y, snake current direction, initial move leading to this cell
    for initial_move in b.get_possible_directions(): # enqueue all possible cells you can move to
        if(b.is_valid_move(initial_move)):
            loc_x, loc_y = b.get_loc_after_move(initial_move)
            q.push((loc_x, loc_y, initial_move, initial_move))

    visited = np.zeros([b.get_dim_y(), b.get_dim_x(), 8]) # visited array (x by y by direction)

    while (not q.is_empty()): # start bfs from each move
        x, y, snake_dir, initial_move = q.pop()

        if(b.has_apple(x, y)): # return initial move that ends up yielding a path to this apple
            return initial_move
        

        # if no apple, enqueue all cells reachable from (x, y, dir)
        for next_move in b.player_snake.get_valid_directions(direction=snake_dir):
            x_new, y_new = b.player_snake.get_next_loc(next_move, head_loc=(x, y))
            
            if(b.game_board.is_valid_cell((x_new, y_new)) and visited[y_new, x_new, next_move] == 0):
                # make sure to pass along the inital move that started this path
                q.push((x_new, y_new, next_move, initial_move)) 
                visited[y_new, x_new, next_move] = 1
    
    return None
                    

def get_best_move_spaces(b:player_board.PlayerBoard):
    best_move = None
    best_open = -1

    start_x, start_y = b.get_head_location()

    # visited initalization inside for loop because each move needs seperate occupancy grid
    for initial_move in b.get_possible_directions():
        visited = np.zeros([b.get_dim_y(), b.get_dim_x(), 8])

        # if inital move is valid, check the number of squares we can reach using bfs
        if(b.is_valid_move(initial_move)):

            # set current head cell to visited
            visited[start_y, start_x, :] = 1

            q = Queue(dim=3) # format: x, y, snake current direction
            loc_x, loc_y = b.get_loc_after_move(initial_move)

            q.push(np.array([loc_x, loc_y, int(initial_move)]))

            while (not q.is_empty()): # start bfs
                x, y, snake_dir = q.pop()

                for next_move in b.player_snake.get_valid_directions(direction=snake_dir):
                    x_new, y_new = b.player_snake.get_next_loc(next_move, head_loc=(x, y))

                    if(b.game_board.is_valid_cell((x_new, y_new)) and visited[y_new, x_new, next_move] == 0):
                        q.push((x_new, y_new, next_move))
                        visited[y_new, x_new, next_move] = 1

        # in the end, total the number of cells we reached using this initial move
        # (don't double-count visiting the cell in multiple different directions)
        total_open = np.count_nonzero(np.sum(visited,axis=2))

        # maximize open squares
        if(total_open > best_open):
            best_open = total_open
            best_move = initial_move


    return best_move
    