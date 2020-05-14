from matplotlib import pyplot as plt
import numpy as np

class definite_clause:
    """
        Defined on page 256 of AIMA book (4 ed.)

        Attributes
        --------------

            body: a list of symbol(s) (Have to be integers), "body" of a definite clause
            conclusion: a single integer symbol, also called head
    """

    def __init__(self, body : list = [], conclusion : int = None):
        self.body = body
        self.conclusion = conclusion

    def set_body(self, body : list):
        self.body = body

    def set_conclusion(self, conclusion : int):
        self.conclusion = conclusion


def plot_n_queens_solution(assignment):
    """
    Helper function that plots a (rather ugly looking) assignment of N-queens.
    :param assignment:
    :return:
    """
    N = len(assignment)
    board = np.zeros((N, N))
    board[assignment, np.arange(N)] = 1
    plt.figure()
    plt.imshow(board)
    plt.grid()
    plt.show()
