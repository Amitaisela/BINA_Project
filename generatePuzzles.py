from itertools import permutations


def count_inversions(state):
    inv_count = 0
    array = [tile for tile in state if tile != 0]
    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            if array[i] > array[j]:
                inv_count += 1
    return inv_count


def is_solvable(state):
    inv_count = count_inversions(state)
    return inv_count % 2 == 0


def generate_solvable_8_puzzles():
    target_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    all_states = []
    for perm in permutations(target_state):
        if is_solvable(perm):
            all_states.append(list(perm))

    return all_states
