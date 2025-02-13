import heapq

def best_first_search(starting_state):
    # TODO(III): You should copy your code from MP3 here
    visited_states = {starting_state: (None, 0)}

    # The frontier is a priority queue
    # You can pop from the queue using "heapq.heappop(frontier)"
    # You can push onto the queue using "heapq.heappush(frontier, state)"
    # NOTE: states are ordered because the __lt__ method of AbstractState is implemented
    frontier = []
    heapq.heappush(frontier, starting_state)
    
    # TODO(III): implement the rest of the best first search algorithm
    # HINTS:
    #   - add new states to the frontier by calling state.get_neighbors()
    #   - check whether you've finished the search by calling state.is_goal()
    #       - then call backtrack(visited_states, state)...
    # Your code here ---------------
    while frontier:
        visit_node = heapq.heappop(frontier)
        if visit_node.is_goal() == True:
            return backtrack(visited_states,visit_node)
        for neighbor in visit_node.get_neighbors():
            if neighbor not in visited_states or visited_states[neighbor][1] > neighbor.dist_from_start:
                visited_states[neighbor] = (visit_node, neighbor.dist_from_start)
                heapq.heappush(frontier, neighbor)
    return []

def backtrack(visited_states, goal_state):
    # TODO(III): You should copy your code from MP3 here
    path = []
    path.append(goal_state)
    # Your code here ---------------
    parent_node, distant = visited_states[goal_state]
    while distant != 0:
        path.append(parent_node)
        parent_node, distant = visited_states[parent_node]  
    path.reverse()
    # ------------------------------
    return path