from maxcut_williams import maxcut_williams
import numpy as np


# Adjacency matrix of some graph
graph = np.asarray([
    [0, 1, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 1, 0, 1, 0]
])

maxcut = maxcut_williams(graph)

print(*maxcut)  # -> 0 1 3
