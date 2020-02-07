import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem

def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []

    # My code:

    # Creating a node for the initial state
    head = Node(0, problem.init_state, (0, 0), 0)
    num_nodes_expanded += 1

    # Creating a priority queue for the frontier and adding the initial state to it
    frontier = queue.PriorityQueue()
    frontier.put((0 + problem.heuristic(head.state), head))

    # Using a set to keep track of already visited nodes
    visited = set()
    visited.add(head.state)

    # keep exploring the graph until there are no more nodes to explore (or unless the goal state is found)
    while not frontier.empty():

        # update the maximum frontier size
        if frontier.qsize() > max_frontier_size:
            max_frontier_size = frontier.qsize()

        # extract the next node to explore
        newState = frontier.get()[1]

        # if the new state is the goal state
        if newState.state == problem.goal_states[0]:
            # compute path by backtracking from the current node to the initial state
            currentState = newState
            while currentState != head:
                path.append(currentState.state)
                currentState = currentState.parent
            path.append(head.state)
            path.reverse()

            # return the path
            return path, num_nodes_expanded, max_frontier_size
        # if the new state wasn't the goal state, get all of its neighbors which haven't already been explored and put them in the queue to be explored
        else:
            num_nodes_expanded += 1 # update the number of nodes expanded

            for action in problem.get_actions(newState.state):
                child = problem.get_child_node(newState, action)
                if child.state not in visited:
                    visited.add(child.state)
                    frontier.put((child.path_cost + problem.heuristic(child.state), child))
    
    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3
    transition_end_probability = 0.5
    peak_nodes_expanded_probability = 0.35
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 20
    N = 20  
    problem = get_random_grid_problem(p_occ, M, N)

    # # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS
    
