import copy
import sys
import time
from collections import deque

import numpy as np


class Map:

    def __init__(self, mapfile):
        """
            Constructor - reads Mapfile
        """
        infile = open(mapfile, 'r')
        self.start = None
        self.goal = None
        self.map = [list(row) for row in infile.read().splitlines()]
        self.search = copy.deepcopy(self.map)
        infile.close()
        self.set_start_goal()

    def set_start_goal(self):
        """
             Searches the Map for a start location and a goal location
         """
        r = 0
        for row in self.map:
            c = 0
            for val in row:
                if val == 'S':
                    self.start = [r, c]
                elif val == 'G':
                    self.goal = [r, c]
                c = c + 1
            r = r + 1

    def get_neighbors(self, location):
        """
            Returns a list of all navigable (reachable) neighbors from the
             provided location.
        """

        neighbors =  []

        rows = np.arange(location[0] - 1, location[0] + 2)
        rows = rows[rows >= 0]
        rows = rows[rows <= 20]
        cols = np.arange(location[1] - 1, location[1] + 2)
        cols = cols[cols >= 0]
        cols = cols[cols <= 60]

        all_neighbors = np.transpose([np.tile(rows, len(cols)), np.repeat(cols, len(rows))])

        for loc in all_neighbors:
            if self.map[loc[0]][loc[1]] == "G" or self.map[loc[0]][loc[1]] == " ":
                neighbors.append([loc[0], loc[1]])
        if location in neighbors:
            neighbors.remove(location)
        return neighbors

    @staticmethod
    def get_distance(location1, location2):
        """
            Returns the Diagonal Distance between location1 and location2
        """
        'Gets the diagonal distance between location1 and location2, both formatted as [row,col]'
        d = d2 = 1
        y_dist = abs(location1[0] - location2[0])
        x_dist = abs(location1[1] - location2[1])
        return d * (x_dist + y_dist) + (d2 - 2 * d) * min(x_dist, y_dist)

    def to_s(self):
        """
             Returns a string containing the Map and start/end locations.
         """
        out = ""
        for row in self.map:
            for item in row:
                out = out + item
            out = out + "\n"
        out = out + "START LOCATION: " + str(self.start) + "\n"
        out = out + "GOAL LOCATION : " + str(self.goal) + "\n"
        out = out + "DIAGONAL DISTANCE BETWEEN START AND GOAL: " + str(self.get_distance(self.start, self.goal)) + "\n"
        return out

    def to_s_search(self):
        """
              Returns a string containing the Map as it looks during/after the search.
          """
        out = ""
        for row in self.search:
            for item in row:
                out = out + item
            out = out + "\n"
        return out

    def reset_search(self):
        """
               Re-copies the Map into search, resetting any previous changes
           """
        self.search = copy.deepcopy(self.map)

    def backtrack(self, explored, node):
        """
               Places the final path into the search variable.
               Requires explored to contain the list of explored locations and node
                to be the location at the end of the path.
           """
        path_cost = 0
        location = node[0]
        parent = node[1]
        step_cost = node[2]
        path_cost = path_cost + step_cost

        while location != self.start:
            self.search[location[0]][location[1]] = 'O'
            newnode = [x for x in explored if x[0] == parent[0]][0]
            location = newnode[0]
            parent = newnode[1]
            step_cost = newnode[2]
            path_cost = path_cost + step_cost
        self.search[location[0]][location[1]] = 'O'
        return path_cost + 1

    def breadth_first_search(self):
        """  Breadth-First Search:
            This algorithm implements a breadth-first search algorithm from
             page 82 in the textbook.
             """
        node = [self.start, [],  0]  # The initial node has the [row,col] coordinate followed by its parent node and its step-cost
        current_node = None
        dq = deque()  # The double-ended queue structure representing the frontier
        dq.append(node)  # Add the start location to the dq list.
        visted = []  # Create the visted list.

        while dq:
            current_node = dq.popleft()
            if current_node[0] not in [v[0] for v in visted]:
                if current_node[0] == self.goal:
                    break
                else:
                    for node in self.get_neighbors(current_node[0]):
                        dq.append([node, current_node, self.get_distance(node, current_node[0])])
                visted.append(current_node)

        return self.backtrack(visted, current_node)  # Return the path from the current node to the start node

    def depth_first_search(self):

        """
        Depth-First Search:
            This algorithm implements a depth-first search, a modified form of the
             breadth-first search algorithm on page 82 in the textbook.
        """
        node = [self.start, [],
                0]  # The initial node has the [row,col] coordinate followed by its parent node and its step-cost
        current_node = None
        dq = deque()  # The double-ended queue structure representing the frontier
        dq.append(node)  # Add the start location to the dq list.
        visted = []  # Create the visted list.

        while dq:
            current_node = dq.pop()
            if current_node[0] not in [v[0] for v in visted]:
                if current_node[0] == self.goal:
                    break
                else:
                    for node in self.get_neighbors(current_node[0]):
                        dq.append([node, current_node, self.get_distance(node, current_node[0])])
                visted.append(current_node)

        return self.backtrack(visted, current_node)  # Return the path from the current node to the start node


def main():
    """
    main method
        Creates the Map object by reading from the file specified on the command line.
        Runs the graph search, breadth-first search, depth-first search, and a-star search methods on the Map
         and displays timing statistics for each.
    """
    filename = sys.argv[1]  # The file path is the first argument.
    map = Map(filename)
    print("Reading from file: %s\n" % filename)
    print(map.to_s())

    # Breadth-First Search
    bfs_start = time.process_time()
    path_cost = map.breadth_first_search()
    bfs_end = time.process_time()
    bfs_time = bfs_end - bfs_start
    print("\n\nBREADTH-FIRST SOLUTION")
    print(map.to_s_search())
    print(" Breadth-First Path Cost: " + str(path_cost))
    map.reset_search()

    # Depth-First Search
    dfs_start = time.process_time()
    path_cost = map.depth_first_search()
    dfs_end = time.process_time()
    dfs_time = dfs_end - dfs_start
    print("\n\nDEPTH-FIRST SOLUTION")
    print(map.to_s_search())
    print(" Depth-First Path Cost: " + str(path_cost))
    map.reset_search()

    print("\n\nTIMING STATISTICS")
    print("=========================================")
    print(" Breadth-First Search : %f" % bfs_time)
    print(" Depth-First Search   : %f" % dfs_time)


if __name__ == "__main__":
    main()
