import copy
import sys
import time
from collections import deque

import numpy as np


class Map:
	'''
		Constructor - reads Mapfile
	'''
	def __init__(self,Mapfile):
		infile = open(Mapfile,'r')
		self.map = [list(row) for row in infile.read().splitlines()]
		self.search = copy.deepcopy(self.map)
		infile.close()
		self.set_start_goal()

	'''
		Searches the Map for a start location and a goal location
	'''
	def set_start_goal(self):
		r = 0
		c = 0
		for row in self.map:
			c = 0
			for val in row:
				if (val == 'S'):
					self.start = [r,c]
				elif (val == 'G'):
					self.goal = [r,c]
				c = c + 1
			r = r + 1

	'''
		Returns a list of all navigable (reachable) neighbors from the
		 provided location.
	'''
	def get_neighbors(self,location):

		neighbors = []

		rows = np.arange(location[0]-1,location[0]+2)
		rows = rows[rows >= 0]
		rows = rows[rows <= 20]
		cols = np.arange(location[1]-1,location[1]+2)
		cols = cols[cols >= 0]
		cols = cols[cols <= 60]

		all_neighbors = np.transpose([np.tile(rows, len(cols)), np.repeat(cols, len(rows))])

		for loc in all_neighbors:
			if self.map[loc[0]][loc[1]] == "G" or self.map[loc[0]][loc[1]] == " ":
				neighbors.append([loc[0],loc[1]])
		if location in neighbors:
			neighbors.remove(location)
		return (neighbors)

	'''
		Returns the Diagonal Distance between location1 and location2
	'''
	def get_distance(self,location1,location2):
		'Gets the diagonal distance between location1 and location2, both formatted as [row,col]'
		D = D2 = 1
		y_dist = abs(location1[0] - location2[0])
		x_dist = abs(location1[1] - location2[1])
		return D * (x_dist + y_dist) + (D2 - 2 * D) * min(x_dist, y_dist)

	'''
		Returns a string containing the Map and start/end locations.
	'''
	def to_s(self):
		out = ""
		for row in self.map:
			for item in row:
				out = out + item
			out = out + "\n"
		out = out + "START LOCATION: " + str(self.start) + "\n"
		out = out + "GOAL LOCATION : " + str(self.goal) + "\n"
		out = out + "DIAGONAL DISTANCE BETWEEN START AND GOAL: " + str(self.get_distance(self.start,self.goal)) + "\n"
		return out

	'''
		Returns a string containing the Map as it looks during/after the search.
	'''
	def to_s_search(self):
		out = ""
		for row in self.search:
			for item in row:
				out = out + item
			out = out + "\n"
		return out

	'''
		Re-copies the Map into search, resetting any previous changes
	'''
	def reset_search(self):
		self.search = copy.deepcopy(self.map)

	'''
		Places the final path into the search variable.
		Requires explored to contain the list of explored locations and node
		 to be the location at the end of the path.
	'''
	def backtrack(self,explored,node):
		path_cost = 0
		location = node[0]
		parent = node[1]
		step_cost = node[2]
		path_cost = path_cost + step_cost

		while (location != self.start):
			self.search[location[0]][location[1]] = 'O'
			newnode = [x for x in explored if x[0] == parent][0]
			location = newnode[0]
			parent = newnode[1]
			step_cost = newnode[2]
			path_cost = path_cost + step_cost
		self.search[location[0]][location[1]] = 'O'
		return path_cost+1

	'''
	Breadth-First Search:
		This algorithm implements a breadth-first search algorithm from
		 page 82 in the textbook.
	'''
	def breadth_first_search(self):
		node = [self.start,[],0]	# The initial node has the [row,col] coordinate followed by its parent node and its step-cost
		open = deque()				# The double-ended queue structure representing the frontier
		open.append(node)			# Add the start location to the open list.
		closed = []					# Create the closed (visited) list.

		#TODO: Code BFS here.

		return self.backtrack(closed,node)	# Return the path from the current node to the start node

	'''
	Depth-First Search:
		This algorithm implements a depth-first search, a modified form of the
		 breadth-first search algorithm on page 82 in the textbook.
	'''
	def depth_first_search(self):
		node = [self.start,[],0]	# The initial node has the [row,col] coordinate followed by its parent node and its step-cost
		open = deque()				# The double-ended queue structure representing the frontier
		open.append(node)			# Add the start location to the open list.
		closed = []					# Create the closed (visited) list.

		#TODO: Code DFS here.

		return self.backtrack(closed,node)	# Return the path from the current node to the start node



'''
main method
	Creates the Map object by reading from the file specified on the command line.

	Runs the graph search, breadth-first search, depth-first search, and a-star search methods on the Map
	 and displays timing statistics for each.
'''
def main():
	filename = sys.argv[1]						# The file path is the first argument.
	map = Map(filename)
	print("Reading from file: %s\n" % filename)
	print(map.to_s())

	### Breadth-First Search
	bfs_start = time.process_time()
	path_cost = map.breadth_first_search()
	bfs_end = time.process_time()
	bfs_time = bfs_end - bfs_start
	print ("\n\nBREADTH-FIRST SOLUTION")
	print (map.to_s_search())
	print (" Breadth-First Path Cost: " + str(path_cost))
	map.reset_search()

	### Depth-First Search
	dfs_start = time.process_time()
	path_cost = map.depth_first_search()
	dfs_end = time.process_time()
	dfs_time = dfs_end - dfs_start
	print ("\n\nDEPTH-FIRST SOLUTION")
	print (map.to_s_search())
	print (" Depth-First Path Cost: " + str(path_cost))
	map.reset_search()


	print("\n\nTIMING STATISTICS")
	print("=========================================")
	print(" Breadth-First Search : %f" % bfs_time)
	print(" Depth-First Search   : %f" % dfs_time)


if __name__ == "__main__":
	main()
