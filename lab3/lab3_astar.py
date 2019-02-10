import random
import time
from copy import deepcopy
from operator import itemgetter

import numpy as np


# ==============================================================================
# Goal: Solve the 8-puzzle using A*Search
# ==============================================================================

# ===============================================================================
# Method: method_timing
#  Purpose: A timing function that wraps the called method with timing code.
#     Uses: time.time(), used to determine the time before an after a call to
#            func, and then returns the difference.
def method_timing(func):
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        print('%s took %0.3f ms for %s' % (func.__name__, (t2 - t1) * 1000.0, arg[1]))
        return [res, (t2 - t1) * 1000.0]

    return wrapper


# ===============================================================================
# Class: PriorityQueue
#  Purpose: A simplified PriorityQueue
class PriorityQueue:

    def __init__(self):
        self.queue = []

    def put(self, item, priority):
        node = [item, priority]
        self.queue.append(node)
        self.queue.sort(key=itemgetter(1))

    def get(self):
        if len(self.queue) == 0:
            return None
        node = self.queue.pop(0)
        return node[0]

    def empty(self):
        return len(self.queue) == 0


# ===============================================================================
# Class: Board
#  Purpose: Represents an 8-puzzle game Board
class Board:

    def __init__(self, board=None):
        '''
            Default Constructor
        '''
        if board:
            self.board = board
            if not self.solvable():
                raise Exception("This Board is not Solvable.")
            self.board = board
        else:
            self.board = random.sample(range(0, 9), 9)
            while not self.solvable():
                self.board = random.sample(range(0, 9), 9)

    def solvable(self):
        '''
            Returns True if the board is solvable (inversions must be even)
        '''
        return self.inversions() % 2 != 1

    def inversions(self):
        '''
            Count the inversions in the board
        '''
        inv = 0
        numbers = deepcopy(self.board)
        numbers.remove(0)

        inv = len(
            [(i, j) for i in range(len(numbers)) for j in range(i + 1, len(numbers)) if (numbers[j] > numbers[i])])

        return inv

    def to_s(self, state=None):
        '''
            Returns a string representation of the state passed
        '''
        shaped = np.reshape(np.array(state or self.board), (3, 3))
        string = '\n'.join(' '.join(map(str, x)) for x in shaped)
        return string.replace("0", " ")


# ===============================================================================
# Class: Solver
#  Purpose: Solves an 8-puzzle Board
class Solver:

    def __init__(self, board):
        '''
            Constructor: accepts a game board
        '''
        self.board = board
        self.solution = [1, 2, 3, 4, 5, 6, 7, 8, 0]

        self.rules = []
        self.rules.append([1, 3])
        self.rules.append([0, 2, 4])
        self.rules.append([1, 5])
        self.rules.append([0, 4, 6])
        self.rules.append([1, 3, 5, 7])
        self.rules.append([2, 4, 8])
        self.rules.append([3, 7])
        self.rules.append([4, 6, 8])
        self.rules.append([5, 7])

    def hamming(self, state):
        '''
            Hamming heuristic: Determine the wrong positions for a given state
        '''
        errors = len([x for x in range(9) if state[x] != self.solution[x]])
        return errors

    def manhattan(self, state):
        '''
            Manhattan Heuristic: Determine the Sum of the Manhattan distances for a given state
        '''
        man_dist = 0
        for i, item in enumerate(state):
            curr_row, curr_col = int(i / 3), i % 3
            goal_row, goal_col = int(item / 3), item % 3
            man_dist += abs(curr_row - goal_row) + abs(curr_col - goal_col)

        return man_dist

    def swap(self, state, a, b):
        '''
            Swap state a and state b
        '''
        state[a], state[b] = state[b], state[a]
        return state

    def get_neighbors(self, state):
        '''
            Returns a list of valid neighbors for the given position
              State is a list of nine numbers.
        '''
        neighbors = []

        for loc in range(9):
            if state[loc] == 0:
                break

        for swap_index in self.rules[loc]:
            neighbor = deepcopy(state)
            neighbor = self.swap(neighbor, loc, swap_index)
            neighbors.append(neighbor)

        return neighbors

    @method_timing
    def astar_search(self, heuristic_type="hamming"):
        '''
            Performs the A* search for the current board
        '''
        steps = 0  # Initialize the step counter
        board = self.board.board  # Copy of the board

        frontier = PriorityQueue()  # Create the frontier as a PriorityQueue
        frontier.put(board, 0)  # - And push the initial state to the frontier

        came_from = {}  # Use a dictionary to track the path
        cost_so_far = {}  # Use a dictionary to track the path cost by state

        came_from[str(board)] = None  # Push the initial state and set it's parent as None
        cost_so_far[str(board)] = 0  # The cost so far from the initial state is 0.
        max_nodes_explored = 0

        while frontier:
            current_item = frontier.get()
            if current_item == self.solution:
                break
            for num_swaps, neigbor in enumerate(self.get_neighbors(current_item)):
                str_neigbor = str(neigbor)
                path_cost = cost_so_far[str(current_item)] + 1
                if str_neigbor not in cost_so_far or path_cost < cost_so_far[str_neigbor]:
                    cost_so_far[str_neigbor] = path_cost
                    frontier.put(neigbor, path_cost+ getattr(self, heuristic_type)(neigbor))
                    came_from[str_neigbor] = current_item
                    max_nodes_explored = max(max_nodes_explored, len(frontier.queue))

        node = self.solution

        # -- Recreate the solution by walking back from the goal to the initial state
        solution = []

        while node != None:
            solution.append(node)
            parent = came_from[str(node)]
            node = parent

        # -- Reverse the solution so you move forward
        solution.reverse()

        # for state in solution: print(self.board.to_s(state), "\n")

        print("\nTotal Moves:", str(len(solution) - 1))
        print("nodes explored {}, {} distance from initial state:{} to goal state:{} is {} ".format(max_nodes_explored,heuristic_type,board, self.solution,getattr(self, heuristic_type)(board)))



# =====================
# Main Algorithm

tester1 = [[8, 1, 3, 4, 0, 2, 7, 6, 5], [1, 8, 0, 4, 3, 2, 5, 7, 6], None]
# tester1 = [ [0,1, 2, 3, 4, 5, 6, 8, 7]]
tester1 = [ None]
# tester2 = [1, 8, 0, 4, 3, 2, 5, 7, 6]
for n, test in enumerate(tester1):
    print("{indent} running for {data} {indent}".format(indent="*"*10, data=test))
    board = Board(test)
    solver = Solver(board)
    solver.astar_search("hamming")
    solver.astar_search("manhattan")
