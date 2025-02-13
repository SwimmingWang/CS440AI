# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
import heapq

# Search should return the path and the number of states explored.
# The path should be a list of MazeState objects that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (astar)
# You may need to slight change your previous search functions in MP2 since this is 3-d maze


def search(maze, searchMethod):
    return {
        "astar": astar,
    }.get(searchMethod, [])(maze)


# TODO: VI
def astar(maze):
    start_state = maze.get_start()
    start_state.dist_from_start = 0
    visited_states = {start_state: (None, 0)}
    frontier = []
    heapq.heappush(frontier, start_state)
    closed_set = set()

    while frontier:
        current_state = heapq.heappop(frontier)

        if current_state in closed_set:
            continue

        if current_state.is_goal():
            return backtrack(visited_states, current_state)

        closed_set.add(current_state)

        for neighbor_state in current_state.get_neighbors():
            if neighbor_state in closed_set:
                continue 

            tentative_g_score = current_state.dist_from_start + current_state.get_cost(neighbor_state)

            if neighbor_state not in visited_states or tentative_g_score < visited_states[neighbor_state][1]:
                neighbor_state.dist_from_start = tentative_g_score
                visited_states[neighbor_state] = (current_state, tentative_g_score)
                heapq.heappush(frontier, neighbor_state)


# Go backwards through the pointers in visited_states until you reach the starting state
# NOTE: the parent of the starting state is None
# TODO: VI
def backtrack(visited_states, current_state):
    path = []
    if current_state not in visited_states:
        return path
    path.append(current_state)
    parent_node, distant = visited_states[current_state]
    while distant != 0:
        path.append(parent_node)
        parent_node, distant = visited_states[parent_node]  
    path = path[::-1]
    return path
