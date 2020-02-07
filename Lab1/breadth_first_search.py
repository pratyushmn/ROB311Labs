from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    #My Code:
    head = Node(0, problem.init_state, (0, 0), 0) # creating a node for the initial state
    num_nodes_expanded += 1

    # Using a dict to store the parent node that was used to get to each state
    parents = dict() 
    parents[head.state] = None

    # The frontier is the list of nodes to be explored next
    frontier = deque()
    frontier.append(head)

    # Using another dict to store the level (ie. distance) each state is away from the initial state
    levels = dict()
    levels[head.state] = 0

    # keep exploring the graph until all nodes have been explored (or unless the goal state is found)
    while len(frontier) != 0:

        # update the max frontier size
        if len(frontier) > max_frontier_size:
            max_frontier_size = len(frontier)

        # extract a new state to expand
        newState = frontier.popleft()          
        num_nodes_expanded += 1

        # get all the neighbors of the current state
        for action in problem.get_actions(newState.state):
            child = problem.get_child_node(newState, action)

            # only do anything with the child node if it hasn't already been visited
            if child.state not in levels:
                parents[child.state] = newState
                levels[child.state] = levels[parents[child.state].state] + 1

                # if the state is the goal state, we need to compute the path used to get there and then return
                if child.state == problem.goal_states[0]:
                    # backtracking to find path by using the parents dictionary
                    currentState = child
                    path.append(currentState.state)
                    while currentState != head:
                        currentState = parents[currentState.state]
                        path.append(currentState.state)
                    path.reverse()

                    return path, num_nodes_expanded, max_frontier_size

                # if it wasn't the goal state, add the node to the frontier to explore further
                frontier.append(child)

    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('C:/Users/praty/GoogleDrive/university/schoolstuff/Year3/ROB311/Lab1/rob311_winter_2020_project_01/datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = [0]
    problem = GraphSearchProblem(goal_states, init_state[0], V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)