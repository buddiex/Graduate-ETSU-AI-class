import os
import time
from collections import deque

import numpy

# ==============================================================================
# Goal: Traverse a directory structure using Depth-First and Breadth-First
#        Search Algorithms
# ==============================================================================

# ---------------------------------------
# The Path to Search
#  You may need to alter this based on
#   your file location and OS.
path = "treepath"
BASE_PATH = r"C:\Users\OMIGIEO\Google Drive\etsu\weekend course\algo\ai\treepath"
# ---------------------------------------
# The Goal Filename
#  This is the name of the files you are
#   attempting to find.


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
        # print ('%s took %0.3f ms' % (func, (t2-t1)*1000.0))
        return [res, (t2 - t1) * 1000.0]
    return wrapper


# ===============================================================================
# Method: expand
#  Purpose: Returns the child nodes of the current node in a list
#     Uses: os.listdir, which returns a Python list of children--directories
#           as well as files.
def expand(path):
    return (os.listdir(path))

# ===============================================================================
# Method: breadthFirst
#  Purpose: Conducts a Breadth-First search of the file structure
#  Returns: The location of the file if it was found, an empty string otherwise.
#     Uses: Wrapped by method_timing method
@method_timing
def breadthFirst(path, goal):
    dq = deque([os.path.join(BASE_PATH, path)])
    visited = []
    while dq:
        current_item = dq.pop()
        if current_item not in visited:
            for i in expand(current_item):
                if i == goal:
                    return current_item
                if os.path.isdir(os.path.join(current_item, i)):
                    dq.appendleft(os.path.join(current_item, i))
            visited.append(current_item)
    return ""

# ===============================================================================
# Method: depthFirst
#  Purpose: Conducts a Depth-First search of the file structure
#  Returns: The location of the file if it was found, an empty string otherwise.
#     Uses: Wrapped by method_timing method
@method_timing
def depthFirst(path, goal):
    dq = deque([os.path.join(BASE_PATH, path)])
    visited = []
    while dq:
        current_item = dq.popleft()
        if current_item not in visited:
            for directory in expand(current_item):
                if directory == goal:
                    return current_item
                if os.path.isdir(os.path.join(current_item, directory)):
                    dq.appendleft(os.path.join(current_item, directory))
            visited.append(current_item)
    return ""


# ===============================================================================
# Method: serach
#  Purpose: compbines the two searches with an option to which to use BFS OR DFS
#  Returns: The location of the file if it was found, an empty string otherwise.
#     Uses: Wrapped by method_timing method
@method_timing
def search(path, goal, search_type = 'BFS'):
    dq = deque([os.path.join(BASE_PATH, path)])
    visited = []
    while dq:
        current_item = dq.popleft() if (search_type.upper()=='BFS') else dq.pop()
        if current_item not in visited:
            for i in expand(current_item):
                if i == goal:
                    return current_item
                if os.path.isdir(os.path.join(current_item, i)):
                    dq.appendleft(os.path.join(current_item, i))
            visited.append(current_item)
    return ""
# =====================
#
#  Completing the code above will allow this code to run. Comment or uncomment
#   as necessary, but the final submission should be appear as the original.

goal = "YAjrbqc.bin"
goal = "xhtj8.bin"
goal = "XUB.bin"

bfs = numpy.empty((10))
for x in range(0, 10):
    filelocation = search(path, goal, 'BFS')
    if filelocation[0] != "":
        # print ("BREADTH-FIRST: Found %s in %0.3f ms" % (goal,filelocation[1]))
        bfs[x] = filelocation[1]

dfs = numpy.empty((10))
for x in range(0, 10):
    filelocation = search(path, goal, 'DFS')
    if filelocation[0] != "":
        # print ("  DEPTH-FIRST: Found %s in %0.3f ms" % (goal,filelocation[1]))
        dfs[x] = filelocation[1]

print("FULL PATH: %s" % filelocation[0])
print ("BREADTH-FIRST SEARCH AVERAGE TIME: %0.3f ms" % bfs.mean())
print ("DEPTH-FIRST SEARCH AVERAGE TIME: %0.3f ms" % dfs.mean())