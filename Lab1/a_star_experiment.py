import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem
from matplotlib import pyplot as plt

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

    head = Node(0, problem.init_state, (0, 0), 0)
    num_nodes_expanded += 1
    frontier = queue.PriorityQueue()
    frontier.put((0 + problem.heuristic(head.state), head))

    visited = set()
    visited.add(head.state)

    while not frontier.empty():
        if frontier.qsize() > max_frontier_size:
            max_frontier_size = frontier.qsize()

        newState = frontier.get()[1]

        if newState.state == problem.goal_states[0]:
            # backtracking to find path
            currentState = newState
            while currentState != head:
                path.append(currentState.state)
                currentState = currentState.parent
            path.append(head.state)
            path.reverse()

            return path, num_nodes_expanded, max_frontier_size
        else:
            num_nodes_expanded += 1

            for action in problem.get_actions(newState.state):
                child = problem.get_child_node(newState, action)
                if child.state not in visited:
                    visited.add(child.state)
                    frontier.put((child.path_cost + problem.heuristic(child.state), child))

    return path, num_nodes_expanded, max_frontier_size

def plot_graph(probabilities, success_rates, size, toggle):
    fig = plt.figure(size + toggle)
    if toggle == 0:
        plt.plot(probabilities, success_rates, 'k-')
        plt.title("Probability That Graph Search Problem (N = {}) Can Be Solved vs Probability That a Grid Cell is Occupied".format(size))
        plt.xlabel("Probability That a Grid Cell is Occupied (Pocc)")
        plt.ylabel("Probability That the Problem is Solvable")
    else:
        plt.plot(probabilities, success_rates, 'k-')
        plt.title("Average Shortest Path Length (N = {}) vs Probability That a Grid Cell is Occupied".format(size))
        plt.xlabel("Probability That a Grid Cell is Occupied (Pocc)")
        plt.ylabel("Average Shortest Path Length")
    return fig

if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    # p_occ = 0.3
    # M = 20
    # N = 20  
    # problem = get_random_grid_problem(p_occ, M, N)

    # # Solve it
    # path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # # Check the result
    # correct = problem.check_solution(path)
    # print("Solution is correct: {:}".format(correct))
    # # Plot the result
    # problem.plot_solution(path)

    # Experiment and compare with BFS

    sizes = [20, 100, 500]
    probabilities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    for N in sizes:
        success_rates = []
        avg_path_lengths = []
        print("Hi")
        for p_occ in probabilities:
            successes = 0
            path_length = 0
            for i in range(100):
                problem = get_random_grid_problem(p_occ, N, N)
                path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
                if len(path) != 0 and problem.check_solution(path):
                    successes += 1
                    path_length += len(path)
                    if not problem.check_solution(path):
                        print("Error - wrong solution. N = {}, P = {}".format(N, p_occ))
                        exit()
            success_rates.append(successes/100)
            avg_path_lengths.append(path_length/100)
        plot_graph(probabilities, success_rates, N, 0)
        plot_graph(probabilities, avg_path_lengths, N, 1)
        print("Bye")
    plt.show()

    
