import json
from collections import deque

# Define the goal state
GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)
GOAL_INDEX = {val: idx for idx, val in enumerate(
    GOAL_STATE)}  # To optimize movement lookup

# Possible moves for the blank (0) tile in a 3x3 grid
MOVES = {
    0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
    3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    6: [3, 7], 7: [4, 6, 8], 8: [5, 7]
}


def bfs_shortest_distances(goal_state):
    queue = deque([(goal_state, 0)])  # (state, distance)
    distances = {goal_state: 0}       # Map state to distance from goal

    while queue:
        current_state, dist = queue.popleft()

        # Find the blank (0) position
        blank_pos = current_state.index(0)

        # Generate all possible moves from the current blank position
        for move in MOVES[blank_pos]:
            # Create the new state by swapping the blank with the target position
            new_state = list(current_state)
            new_state[blank_pos], new_state[move] = new_state[move], new_state[blank_pos]
            new_state_tuple = tuple(new_state)

            # If this state hasn't been visited, record its distance and add to queue
            if new_state_tuple not in distances:
                distances[new_state_tuple] = dist + 1
                queue.append((new_state_tuple, dist + 1))

    return distances


def save_distances(distances, filename="distances.json"):
    # Convert each tuple key to a string
    serializable_distances = {
        str(key): value for key, value in distances.items()}
    with open(filename, 'w') as file:
        json.dump(serializable_distances, file)


def load_distances(filename="distances.json"):
    with open(filename, 'r') as file:
        # Load and convert keys back to tuples
        serialized_distances = json.load(file)
        distances = {tuple(map(int, key.strip("()").split(", "))): value
                     for key, value in serialized_distances.items()}
    return distances
