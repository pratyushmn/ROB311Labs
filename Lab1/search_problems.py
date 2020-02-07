from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt


class Node:

    def __init__(self, parent, state, action, path_cost):
        self.parent = parent
        self.state = state
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def __eq__(self, other):
        return self.state == other.state and self.path_cost == other.path_cost


class SearchProblem(ABC):
    """
    Abstract class for search problems.
    """
    def __init__(self, goal_states, init_state):
        self.goal_states = goal_states
        self.init_state = init_state
        super().__init__()

    @abstractmethod
    def get_child_node(self, parent_node, action):
        pass

    @abstractmethod
    def transition(self, state, action):
        pass

    @abstractmethod
    def goal_test(self, state):
        pass

    @abstractmethod
    def action_cost(self, start_state, action, end_state):
        pass

    @abstractmethod
    def get_actions(self, state):
        pass

    def trace_path(self, node, target_state=None):
        if target_state == None:
            target_state = self.init_state
        path = [node.state]
        node_current = node
        while node_current.state != target_state:
            node_current = node_current.parent
            path.append(node_current.state)
        path.reverse()
        return path


class SimpleSearchProblem(SearchProblem):
    """
        Abstract class for search problems where actions are described as a tuple between states named by numbers, and
        actions are all uniform cost.
        Specific problems only need to implement the abstract method get_actions().
    """
    def __init__(self, goal_states, init_state):
        super(SimpleSearchProblem, self).__init__(goal_states, init_state)

    def get_child_node(self, parent_node, action):
        child_state = self.transition(parent_node.state, action)
        child_path_cost = parent_node.path_cost + self.action_cost(parent_node.state, action, child_state)
        return Node(parent_node, child_state, action, child_path_cost)

    def transition(self, state, action):
        assert action.__contains__(state)
        if action[0] == state:
            return action[1]
        elif action[1] == state:
            return action[0]

    def goal_test(self, state):
        return self.goal_states.__contains__(state)

    def action_cost(self, start_state, action, end_state):
        return 1

    @abstractmethod
    def get_actions(self, state):
        pass


class GraphSearchProblem(SimpleSearchProblem):
    """
    Search problems given over an explicit graph with named vertices and unit edge costs.
    """
    def __init__(self, goal_states, init_state, V, E):
        super(GraphSearchProblem, self).__init__(goal_states, init_state)
        # Store graph vertices (should be array of unique identifiers, e.g. integers)
        self.V = V
        # Store graph edges and weights (uniform unit weights
        if E.shape[1] < 3:
            self.E = E
        else:
            self.E = E[:, 0:2]
        # Construct neighbour dictionary
        self.neighbours = {}
        for vertex in self.V:
            self.neighbours[vertex] = []
        for idx in range(self.E.shape[0]):
            edge = self.E[idx]
            self.neighbours[edge[0]].append(edge[1])
            self.neighbours[edge[1]].append(edge[0])

    def get_actions(self, state):
        return [(state, edge) for edge in self.neighbours[state]]

    def is_neighbour(self, state1, state2):
        return self.neighbours[state1].__contains__(state2)

    def check_graph_solution(self, path):
        for idx, state in enumerate(path):
            if idx == 0:
                if state != self.init_state:
                    return False
            if idx == len(path)-1:
                if not self.goal_states.__contains__(state):
                    return False
            else:
                if not self.is_neighbour(state, path[idx+1]):
                    return False
        return True


class GridSearchProblem(SimpleSearchProblem):

    def __init__(self, goal_states, init_state, M, N, grid_map):
        super(GridSearchProblem, self).__init__(goal_states, init_state)
        # Store graph vertices (should be array of unique identifiers, e.g. integers)
        self.M = M
        self.N = N
        self.grid_map = grid_map
        # Zero the inital and goal states
        x, y = self.get_position(init_state)
        self.grid_map[int(x), int(y)] = False
        for state in goal_states:
            x, y = self.get_position(state)
            self.grid_map[int(x), int(y)] = False

    def get_actions(self, state):
        x, y = self.get_position(state)
        x = int(x)
        y = int(y)
        assert(not self.grid_map[x, y])
        action_list = []
        if x + 1 < self.M and not self.grid_map[x+1, y]:
            action_list.append((state, self.get_state(x+1, y)))
        if x-1 >= 0 and not self.grid_map[x-1, y]:
            action_list.append((state, self.get_state(x-1, y)))
        if y+1 < self.N and not self.grid_map[x, y+1]:
            action_list.append((state, self.get_state(x, y+1)))
        if y-1 >= 0 and not self.grid_map[x, y-1]:
            action_list.append((state, self.get_state(x, y-1)))
        return action_list

    def get_position(self, state):
        assert (state < self.M * self.N)
        x = np.mod(state, self.M)
        y = np.floor(state/ self.M)
        return x, y

    def get_state(self, x, y):
        return y*self.M + x

    def heuristic(self, state):
        return self.manhattan_heuristic(state, self.goal_states[0])

    def manhattan_heuristic(self, state1, state2):
        x1, y1 = self.get_position(state1)
        x2, y2 = self.get_position(state2)
        return abs(x1 - x2) + abs(y1 - y2)

    def plot_solution(self, trajectory):
        fig = plt.figure()
        plt.imshow(1 - self.grid_map.T, cmap='gray')
        x = np.zeros(len(trajectory))
        y = np.zeros(x.shape)
        for idx in range(len(trajectory)):
            [xi, yi] = self.get_position(trajectory[idx])
            x[idx] = xi
            y[idx] = yi
        plt.plot(x, y, 'b-')

        #### Added by me for better visibility
        x, y = self.get_position(self.init_state)
        plt.plot(x, y, 'r.')
        x, y = self.get_position(self.goal_states[0])
        plt.plot(x, y, 'r.')
        #####

        plt.plot()
        plt.title('Grid Search Solution')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        return fig

    def check_solution(self, path):
        for idx, state in enumerate(path):
            x, y = self.get_position(state)
            if self.grid_map[int(x), int(y)]:
                return False
            if idx < len(path)-1:
                action_list = self.get_actions(state)
                neighbours = list(zip(*action_list))[1]
                if not neighbours.__contains__(path[idx+1]):
                    return False
            if idx == len(path) - 1:
                if state != self.goal_states[0]:
                    return False
            if idx == 0:
                if state != self.init_state:
                    return False
        return True


def get_random_grid_problem(p_occ, M, N):
    """
    Makes a random grid problem of size MxN where each cell has probability 0 <= p_occ <= 1 of being occupied.
    :param p_occ: probability of a cell being occupied (must be within [0.0, 1.0])
    :param M: width (x-dimension) in integer number of cells
    :param N: height (y-dimension) in integer number of cells
    :return: instance of GridSearchProblem
    """
    grid_map = np.random.rand(M, N) <= p_occ
    init_state = np.random.randint(M*N)
    goal_state = np.random.randint(M*N)
    problem = GridSearchProblem([goal_state], init_state, M, N, grid_map)
    return problem


if __name__ == '__main__':
    pass
