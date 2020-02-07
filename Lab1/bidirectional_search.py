from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem
from breadth_first_search import breadth_first_search

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

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
    paths = []

    # My code:

    # Creating nodes for the initial and goal state
    head = Node(0, problem.init_state, (0, 0), 0)
    goal = Node(0, problem.goal_states[0], (0, 0), 0)

    num_nodes_expanded += 2

    # Creating deques which stored lists of nodes that were all at the same level away from either the initial/goal state
    headFrontierLayers = deque()
    headFrontierLayers.append(head)

    tailFrontierLayers = deque()
    tailFrontierLayers.append(goal)

    # creating sets to keep track of nodes which were visited by the frontier from either the initial/goal state
    headVisited = set()
    headVisited.add(head.state)

    tailVisited = set()
    tailVisited.add(goal.state)


    # keep going until either all nodes have been explored or at least one path at the most recent level has been found
    while len(headFrontierLayers) != 0 and len(tailFrontierLayers) != 0 and len(paths) == 0:

        # take all the nodes at the current level and put them into new deques coming from the initial and goal state
        headFrontier = deque()
        tailFrontier = deque()

        while len(headFrontierLayers) != 0:
            headFrontier.append(headFrontierLayers.pop())

        while len(tailFrontierLayers) != 0:
            tailFrontier.append(tailFrontierLayers.pop())

        # as long as there exist nodes at the same level away from the initial and goal state, keep exploring them
        while len(headFrontier) != 0 and len(tailFrontier) != 0:

            # update the max frontier size as necessary
            if len(headFrontier) + len(tailFrontier) > max_frontier_size:
                max_frontier_size = len(headFrontier) + len(tailFrontier) + len(tailFrontierLayers) + len(headFrontierLayers)

            # extract nodes from both frontiers and explore them by generating all of their children nodes
            newState1 = headFrontier.popleft()
            newState2 = tailFrontier.popleft()

            num_nodes_expanded += 2

            for action in problem.get_actions(newState1.state):
                child = problem.get_child_node(newState1, action)

                # if the current state was already visited by the other frontier, then it is part of a possible path
                if child.state in tailVisited:
                    # compute path by backtracking from the current node to the initial state, and also the current node to the goal state
                    newPath = []
                    
                    for vertex in tailFrontierLayers:
                        if vertex.state == child.state:
                            while vertex != goal:
                                newPath.append(vertex.parent.state)
                                vertex = vertex.parent

                    if len(newPath) == 0:
                        for vertex in tailFrontier:
                            if vertex.state == child.state:
                                while vertex != goal:
                                    newPath.append(vertex.parent.state)
                                    vertex = vertex.parent

                    currentState = child
                    while currentState != head:
                        newPath.insert(0, currentState.state)
                        currentState = currentState.parent
                    newPath.insert(0, head.state)       

                    # add the new path to the array of possible paths
                    paths.append([innerState for innerState in newPath])

                # if the state hasn't been visited by either frontier, add it to be explored layer and mark that it was visited
                elif child.state not in headVisited:
                    headVisited.add(child.state)
                    headFrontierLayers.append(child)

            for action in problem.get_actions(newState2.state):
                child = problem.get_child_node(newState2, action)

                # if the current state was already visited by the other frontier, then it is part of a possible path
                if child.state in headVisited:
                    # compute path by backtracking from the current node to the initial state, and also the current node to the goal state
                    newPath = []
                    currentState = child
                    
                    for vertex in headFrontierLayers:
                        if vertex.state == currentState.state:
                            while vertex != head:
                                newPath.append(vertex.state)
                                vertex = vertex.parent
                            newPath.append(head.state)
                            newPath.reverse()
                            break
                    
                    if len(newPath) == 0:
                        for vertex in headFrontier:
                            if vertex.state == currentState.state:
                                while vertex != head:
                                    newPath.append(vertex.state)
                                    vertex = vertex.parent
                                newPath.append(head.state)
                                newPath.reverse()
                                break
                    
                    while currentState != goal:
                        newPath.append(currentState.parent.state)
                        currentState = currentState.parent

                    # add the new path to the array of possible paths
                    paths.append([innerState for innerState in newPath])
                
                # if the state hasn't been visited by either frontier, add it to be explored layer and mark that it was visited
                elif child.state not in tailVisited:
                    tailVisited.add(child.state)
                    tailFrontierLayers.append(child)

    # find path with minimum length and return that, or if there weren't any paths found, return the empty array which was originally created
    if len(paths) > 0:
        return min(paths, key=len), num_nodes_expanded, max_frontier_size
    else:
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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('C:/Users/praty/GoogleDrive/university/schoolstuff/Year3/ROB311/Lab1/rob311_winter_2020_project_01/datasets/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = [0]
    problem = GraphSearchProblem(goal_states, init_state[0], V, E)
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("BDS:")
    print("Solution is correct: {:}".format(correct))
    print(path)

    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("BFS:")
    print("Solution is correct: {:}".format(correct))
    print(path)


    # Be sure to compare with breadth_first_search!