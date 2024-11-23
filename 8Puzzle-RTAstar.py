import heapq
from collections import deque
from distanceGenerator import *
import os
import numpy as np
import generatePuzzles
import time

# CUTOFF = float("inf")
CUTOFF = 181440/4
GOAL_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 0]
INDEX_TO_COORDINATES = {0: (0, 2), 1: (1, 2), 2: (2, 2),
                        3: (0, 1), 4: (1, 1), 5: (2, 1),
                        6: (0, 0), 7: (0, 1), 8: (0, 2)}


class PuzzleState:
    def __init__(self, board, distances, heuristic, status, cutoff, nodeCount, g=1, parent=None, sigma=2.5, c=2/3):
        self.board = board
        self.distances = distances
        self.g = 1
        self.parent = parent
        self.heuristic = heuristic
        self.sigma = sigma
        self.c = c
        self.count = 0

        if status == "Basic" or nodeCount >= cutoff:
            self.h = self.calculate_heuristic()
        elif status == "Optimistic":
            self.h = self.optimistic_heuristic()
        elif status == "Pessimistic":
            self.h = self.pessimistic_heuristic()

    def f(self):
        return self.g + self.h

    def hstar(self):
        puzzle_tuple = tuple(self.board)

        if puzzle_tuple in self.distances:
            distance_to_goal = self.distances[puzzle_tuple]
            return distance_to_goal
        else:
            raise ValueError(
                "This puzzle configuration is not in the precomputed distances.")

    def manhattan_distance(self):
        total_distance = 0
        for tile in self.board:
            if tile != 0:
                self_index = self.board.index(tile)
                goal_index = GOAL_STATE.index(tile)
                if self_index != goal_index:

                    tile_coordinate = INDEX_TO_COORDINATES[self_index]
                    goal_coordinate = INDEX_TO_COORDINATES[goal_index]
                    menhatten_distance = (
                        tile_coordinate[0] - goal_coordinate[0])**2 + (tile_coordinate[1] - goal_coordinate[1])**2

                    total_distance += menhatten_distance

        return total_distance

    def linear_conflict(self):
        rows = [self.board[i:i+3] for i in range(0, 9, 3)]
        columns = [self.board[i::3] for i in range(3)]
        manhattan_distance = self.manhattan_distance()
        counter = []

        for i, rows in enumerate(rows):
            # get all pairs of tiles in the same row
            pairs = [(a, b) for a in rows for b in rows if a !=
                     0 and b != 0 and a != b]
            for pair in pairs:
                t_j, t_k = pair

                # check if t_j is on the right of t_k
                if rows.index(t_j) > rows.index(t_k):
                    arr = [i*3+1, i*3+2, i*3+3]

                    # check if t_j, t_k in the right row
                    if t_j in arr and t_k in arr:

                        if t_j < t_k:
                            counter.append((t_j, t_k))
                            # remove reverse pair to avoid double counting, if still in the list
                            if (t_k, t_j) in pairs:
                                pairs.remove((t_k, t_j))

        # same for columns
        for i, column in enumerate(columns):
            pairs = [(a, b) for a in column for b in column if a !=
                     0 and b != 0 and a != b]
            for pair in pairs:
                t_j, t_k = pair
                if column.index(t_j) > column.index(t_k):
                    arr = [i+1, i+4, i+7]
                    if t_j in arr and t_k in arr:
                        if t_j < t_k:
                            counter.append((t_j, t_k))
                            if (t_k, t_j) in pairs:
                                pairs.remove((t_k, t_j))

        return manhattan_distance + 2 * len(counter)

    def misplaced_tiles(self):
        return sum(1 for i, tile in enumerate(self.board) if tile != 0 and tile != GOAL_STATE[i])

    def Gaschnig_relaxed_adjacency(self):
        board_copy = self.board[:]
        moves = 0
        while board_copy != GOAL_STATE:
            empty_index = board_copy.index(0)
            if board_copy[empty_index] != GOAL_STATE[empty_index]:
                # Find the target tile to swap with the empty tile
                target_tile = GOAL_STATE[empty_index]
                target_index = board_copy.index(target_tile)
                # Swap
                board_copy[empty_index], board_copy[target_index] = board_copy[target_index], board_copy[empty_index]
            else:
                # Swap the empty tile with any misplaced tile
                for i, tile in enumerate(board_copy):
                    if tile != 0 and tile != GOAL_STATE[i]:
                        board_copy[empty_index], board_copy[i] = board_copy[i], board_copy[empty_index]
                        break
            moves += 1
        return moves

    def optimistic_heuristic(self):
        heuristic_value = self.calculate_heuristic()
        noise = np.random.normal(heuristic_value, self.sigma)
        return self.c*(heuristic_value + noise)

    def pessimistic_heuristic(self):
        heuristic_value = self.calculate_heuristic()
        noise = np.random.normal(heuristic_value, self.sigma)
        return (1/self.c)*(heuristic_value + noise)

    def calculate_heuristic(self):
        if self.heuristic == "hstar":
            return self.hstar()
        elif self.heuristic == "manhattan_distance":
            return self.manhattan_distance()
        elif self.heuristic == "linear_conflict":
            return self.linear_conflict()
        elif self.heuristic == "misplaced_tiles":
            return self.misplaced_tiles()
        elif self.heuristic == "Gaschnig_relaxed_adjacency":
            return self.Gaschnig_relaxed_adjacency()
        else:
            raise ValueError(f"Invalid heuristic specified: {self.heuristic}")

    def is_goal(self):
        return self.board == GOAL_STATE

# RTA* algorithm


def rta_star(start_state, status, cutoff):
    move_generated_time = 0
    current_state = start_state
    current_state.g = 0

    path = [current_state.board]
    nodeCount = 0

    visited = {tuple(current_state.board): current_state}

    while not current_state.is_goal():
        # print(path)
        # input()
        neighbors = []
        empty_index = current_state.board.index(0)
        moves = {
            'up': empty_index - 3,
            'down': empty_index + 3,
            'left': empty_index - 1 if empty_index % 3 != 0 else -1,
            'right': empty_index + 1 if (empty_index + 1) % 3 != 0 else -1
        }
        for direction, new_index in moves.items():
            if 0 <= new_index < 9:
                new_board = current_state.board[:]
                new_board[empty_index], new_board[new_index] = new_board[new_index], new_board[empty_index]

                if tuple(new_board) in visited:
                    neighbors.append(visited[tuple(new_board)])
                else:
                    neighbors.append(PuzzleState(
                        new_board, current_state.distances, current_state.heuristic, status, cutoff, nodeCount))

        nodeCount += len(neighbors)
        neighbors.sort(key=lambda x: (x.f(), x.count))

        best_neighbor = neighbors[0]
        second_best_neighbor = neighbors[1]

        try:
            cell_index = path.index(best_neighbor.board)
            path = path[:cell_index]
        except ValueError:
            pass

        best_neighbor.parent = current_state
        current_state.h = second_best_neighbor.f()

        current_state.count = move_generated_time
        move_generated_time += 1

        visited[tuple(current_state.board)] = current_state
        visited[tuple(best_neighbor.board)] = best_neighbor

        path.append(best_neighbor.board)

        current_state = best_neighbor

    number_of_right_decisions = -1

    for i in range(0, len(path)-1):
        current_hstar = visited[tuple(path[i])].hstar()
        parent_hstar = visited[tuple(path[i+1])].hstar()

        if current_hstar > parent_hstar:
            number_of_right_decisions += 1

    return path, nodeCount, (number_of_right_decisions + 2) / len(path)


def solution(start_state, status, cutoff, i):
    start_time = time.time()

    path, nodes, percentage_of_right_decision = rta_star(
        start_state, status, cutoff)

    if percentage_of_right_decision != -1:
        elapsed_time = time.time() - start_time

        if CUTOFF == float("inf"):
            name = "RTA_star"
        else:
            name = f"RTA_star_cutoff_{CUTOFF}"

        str_puzzle = "".join(str(i) for i in start_state.board)

        with open(f"results_{name}.csv", "a") as f:
            if os.stat(f"results_{name}.csv").st_size == 0:
                f.write(
                    "Algorithm,Heuristic,Status,NodesExpanded,rightDecisions, Solutionlength, Time, puzzle\n")
            f.write(
                f"rta_Star,{start_state.heuristic},{status},{nodes},{percentage_of_right_decision},{len(path)} ,{elapsed_time},{str_puzzle}\n")
    else:
        # save i in error file
        with open("errors_RTA.txt", "a") as f:
            f.write(f"{i}\n")


def get_random_puzzles(ALl_puzzles, num, seed):
    np.random.seed(seed)
    starting_puzzles = ALl_puzzles[:len(ALl_puzzles) // 3]
    middle_puzzles = ALl_puzzles[len(
        ALl_puzzles) // 3: 2 * len(ALl_puzzles) // 3]
    ending_puzzles = ALl_puzzles[2 * len(ALl_puzzles) // 3:]

    starting_puzzles_indices = np.random.choice(
        len(starting_puzzles), num // 3, replace=False)
    middle_puzzles_indices = np.random.choice(
        len(middle_puzzles), num // 3, replace=False)
    ending_puzzles_indices = np.random.choice(
        len(ending_puzzles), num // 3, replace=False)

    starting_puzzles = [starting_puzzles[i] for i in starting_puzzles_indices]
    middle_puzzles = [middle_puzzles[i] for i in middle_puzzles_indices]
    ending_puzzles = [ending_puzzles[i] for i in ending_puzzles_indices]

    return starting_puzzles, middle_puzzles, ending_puzzles, starting_puzzles_indices, middle_puzzles_indices, ending_puzzles_indices


if __name__ == "__main__":
    All_puzzles = generatePuzzles.generate_solvable_8_puzzles()
    starting_puzzles, middle_puzzles, ending_puzzles, starting_puzzles_indices, middle_puzzles_indices, ending_puzzles_indices = get_random_puzzles(
        All_puzzles, 30000, 42)

    if not os.path.exists("distances.json"):
        distances = bfs_shortest_distances(tuple(GOAL_STATE))
        save_distances(distances)
    else:
        distances = load_distances()

    heuristics = [
        "hstar",
        "manhattan_distance",
        "linear_conflict",
        "misplaced_tiles",
        "Gaschnig_relaxed_adjacency"
    ]
    statuses = ["Basic", "Optimistic", "Pessimistic"]

    i = 0

    # this is the cutoff for the optimistic \ pessimistic heuristic. When we reached this number of nodes expanded, we switch to the basic heuristic, which we know will solve the puzzle.
    print("Starting RTA*")
    for heuristic in heuristics:
        for status in statuses:
            for puzzle in starting_puzzles:
                i += 1
                if i % 1000 == 0:
                    print(f"i: {i}")
                try:
                    start_state = PuzzleState(
                        puzzle, distances, heuristic=heuristic, status=status, cutoff=CUTOFF, nodeCount=0)
                    solution(start_state, status, CUTOFF, i)
                except ValueError as e:
                    print(e)
                    continue
    print("middle")
    for heuristic in heuristics:
        for status in statuses:
            for puzzle in middle_puzzles:
                i += 1
                if i % 1000 == 0:
                    print(f"i: {i}")
                try:
                    start_state = PuzzleState(
                        puzzle, distances, heuristic=heuristic, status=status, cutoff=CUTOFF, nodeCount=0)
                    solution(start_state, status, CUTOFF, i)
                except ValueError as e:
                    print(e)
                    continue
    print("ending")
    for heuristic in heuristics:
        for status in statuses:
            for puzzle in ending_puzzles:
                i += 1
                if i % 1000 == 0:
                    print(f"i: {i}")
                try:
                    start_state = PuzzleState(
                        puzzle, distances, heuristic=heuristic, status=status, cutoff=CUTOFF, nodeCount=0)
                    solution(start_state, status, CUTOFF, i)
                except ValueError as e:
                    print(e)
                    continue
