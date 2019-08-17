# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Rahul Kunji (rahulsk2@illinois.edu) on 01/16/2019

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import heapq

# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

# Authors: Megan Walsh (meganew2), Victoria Colthurst (vrc2), Kathryn Fejer (fejer2)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
        "extra_credit": extra_credit,
    }.get(searchMethod)(maze)


def bfs(maze):
    # return path, num_states_explored
    start = maze.getStart()
    end = maze.getObjectives()

    paths, num_explored = bfs_helper(maze, start, end[0])

    return paths, num_explored



def dfs(maze):
    # return path, num_states_explored

    paths = []
    paths.append(maze.getStart())
    explored = []
    current = maze.getStart()
    destinations = maze.getObjectives()

    predecessors = {}
    finalPath = []

    while paths:
        current = paths.pop()

        if current == destinations[0]:
            finalPath.append(current)
            while ( current != maze.getStart() ):
                finalPath.append( predecessors[current] )
                current = predecessors[current]
            finalPath.reverse()
            return finalPath, len(explored)

        explored.append((current))
        neighbors = maze.getNeighbors(current[0], current[1])

        for x in neighbors:
            if (not x in explored) and (not x in paths):
                paths.append(x)
                predecessors[x] = current;


    return finalPath, 0


def greedy(maze):

    end = (maze.getObjectives())[0]
    start = maze.getStart()
    frontier = []
    visited = []
    path = []
    num_explored = 0

    # Manhattan distance
    dist = abs(start[0]-end[0]) + abs(start[1]-end[1])
    # Initialize frontier to start
    heapq.heappush(frontier, (dist, start, path[:]))

    while len(frontier) > 0:
        curr = heapq.heappop(frontier)
        position = curr[1]
        path = curr[2][:]
        path.append(position)
        num_explored += 1

        if position == end:
            break;

        neighbors = maze.getNeighbors(position[0], position[1])
        for x in neighbors:
            if maze.isValidMove(x[0], x[1]) and x not in path and x not in visited:
                dist = abs(x[0] - end[0]) + abs(x[1] - end[1])
                heapq.heappush(frontier, (dist, x, path[:]))
                visited.append(position)

    return path, num_explored


def astar(maze):
    # initialize search
    goal = maze.getObjectives() #ignored

    if len(goal) > 1:
        return alternate_astar(maze)

    frontier = []
    visited = {}

    # initialize return values
    path = []
    num_explored = 0

    # for g in goal:
    manhattan = abs(maze.getStart()[0]-goal[0][0]) + abs(maze.getStart()[1]-goal[0][1])
    heapq.heappush(frontier, (manhattan, maze.getStart(), path[:]))
    visited[maze.getStart()] = len(path)

    while len(frontier) > 0:

        current = heapq.heappop(frontier)
        current_position = current[1]
        path = current[2][:]
        path.append(current_position)
        num_explored = num_explored + 1

        if current_position == goal[0]: # found goal
            break;

        neighbors = maze.getNeighbors(current_position[0], current_position[1])
        for n in neighbors:
            # if neighbor is not a wall add it to the frontier
            if maze.isValidMove(n[0], n[1]) and n not in path and (n not in visited.keys() or visited[n] > len(path)): #and n not in deadends:
                manhattan = abs(n[0] - goal[0][0]) + abs(n[1] - goal[0][1])
                heapq.heappush(frontier, (manhattan + len(path), n, path[:])) # save hueristic, node, path
                visited[n] = len(path)

    return path, num_explored


def alternate_astar(maze):
    goals = maze.getObjectives()
    start = maze.getStart()

    # obtain graph representation of problem
    # nodes of graph = goals of maze
    # edges of graph = minimum distance between nodes through graph
    # returns dictionary of nodes with associated edges, edge costs, and paths
    graph = build_a_graph(maze, goals, start)

    path = []
    mst_nodes = goals[:]
    num_explored = 0
    frontier = []

    # ( heuristic + cost, node, current path, mst_nodes list, path cost )
    heapq.heappush(frontier, (0, start, path[:], mst_nodes[:], 0))
    while len(frontier) > 0:
        current = heapq.heappop(frontier)
        current_position = current[1]
        path = current[2]
        path.append(current_position)
        num_explored = num_explored + 1
        mst_nodes = current[3]
        mst_nodes.remove(current_position)

        if len(path) == len(goals):
            # found all dots / goals including start
            break

        # returns [((x,y), cost, path)]
        neighbors = graph[current_position]
        h = mst(graph, mst_nodes)
        for n in neighbors:
            if n[0] not in path:
                cost = current[4] + n[1]
                heapq.heappush(frontier, (h + cost, n[0], path[:], mst_nodes[:], cost))

    # make path
    final_path = []
    for p in range(0, len(path)-1):
        neighbors = graph[path[p]]
        for n in neighbors:
            if n[0] is path[p+1]:
                final_path.extend(n[2])
        if p != len(path)-2:
            final_path.pop(len(final_path)-1)

    return final_path, num_explored

def mst(d, nodes):

    total_cost = 0
    visited = []

    # Choose arbitrary start
    visited.append(nodes[0])

    while len(visited) != len(nodes):
        # Examine all vertices reachable from node
        neighbors = []
        for v in visited:
            neighbors.extend(d[v])

        # Choose the smallest edge that connects to the node
        ascending_cost = sorted(neighbors, key=lambda x: x[1])
        for n in ascending_cost:
            if n[0] not in visited and n[0] in nodes:
                visited.append(n[0])
                # Update the minimum cost
                total_cost += n[1]
                break

    return total_cost

def bfs_helper(maze, start, end):
    count = 1
    paths = []
    paths.append(start)
    explored = set()
    current = start

    finalPath = []
    predecessors = {}
    while paths:
        current = paths[0]
        del paths[0]
        if current == end:
            # Start traversing from the end
            finalPath.append(current)
            while ( current != start ):
                finalPath.append( predecessors[current] )
                current = predecessors[current]
            # Started from end --> list is backwards
            finalPath.reverse()
            return finalPath, len(explored)

        explored.add(current)
        neighbors = maze.getNeighbors(current[0], current[1])

        for x in neighbors:
            if (not x in explored) and (not x in paths):
                paths.append(x)
                predecessors[x] = current
                count+=1

    return finalPath, 0

def build_a_graph(maze, list, start):
    # dict has three parts [((x,y), cost, path)]
    list.append(start)

    dict = {}
    for x in list:
        elem = []

        for y in list:
            if not (x[0] == y[0] and x[1] == y[1]):
                path, cost = bfs_helper(maze, x, y)
                elem.append((y, len(path), path))

        dict[x] = elem[:]

    return dict

def extra_credit(maze):
    goals = maze.getObjectives()
    start = maze.getStart()
    graph = build_a_graph(maze, goals, start)
    visited = []
    neighbors = []

    mst = {}

    # Choose arbitrary start
    visited.append(goals[0])

    while len(visited) != len(goals):
        # Examine all vertices reachable from node
        for v in visited:
            neighbors.extend(graph[v])

        # Choose the smallest edge that connects to the node
        ascending_cost = sorted(neighbors, key=lambda x: x[1])
        for n in ascending_cost:
            if n[0] not in visited and n[0] in goals:
                visited.append(n[0])
                start_node = n[2][0]
                if start_node in mst.keys():
                    mst[start_node].append((n[0], n[1], n[2]))
                else:
                    mst[start_node] = [(n[0], n[1], n[2])]
                nof2 = n[2][::-1]
                # nof2.reverse()
                if n[0] in mst.keys():
                    mst[n[0]].append((start_node, n[1], nof2))
                else:
                    mst[n[0]] = [(start_node, n[1], nof2)]
                break

    stack = []
    stack.append(start)
    explored = set()
    answer = []
    count = 0

    while len(stack)>0:
        count+=1
        current = stack.pop(len(stack)-1)
        explored.add(current)
        answer.append(current)
        neighbors = [x for x in mst[current] if x[0] not in explored]
        neighbors = sorted(neighbors, key=lambda x: x[1])
        for x in neighbors[::-1]:
            stack.append(x[0])


    # make path
    final_path = []
    for p in range(0, len(answer)-1):
        neighbors = graph[answer[p]]
        for n in neighbors:
            if n[0] is answer[p+1]:
                final_path.extend(n[2])
        if p != len(answer)-2:
            final_path.pop(len(final_path)-1)

    return final_path, count
