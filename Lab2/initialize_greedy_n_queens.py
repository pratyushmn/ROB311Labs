import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N)
    
    ### YOUR CODE GOES HERE
    # First queen goes in a random spot
    initial_pos = np.random.randint(0, N)
    greedy_init[0] = initial_pos

    upward_diagonals = np.zeros(2*N) # indexed by using row_num + col_num which remains constant for each upward diagonal
    downward_diagonals = np.zeros(2*N) # indexed by using (row_num - col_num + N) which remains constant for each downward diagonal
    rows = np.zeros(N)

    # arrays used to count the number of pieces which have been placed in such a way as to conflict with any new piece placed in a given row or diagonal (from either direction)
    rows[initial_pos] += 1
    downward_diagonals[initial_pos - 0 + N] += 1
    upward_diagonals[initial_pos + 0] += 1

    for i in range(N - 1): # go through all possible columns in the chessboard -  i + 1 corresponds to the column
        # store all row choices which produce a minimum conflict
        min_conflict_rows = []
        min_conflict = -1

        for j in range(N): # go through each row in the column
            conflicts = 0 # find number of conflicts if queen placed in this row

            conflicts += rows[j]
            conflicts += upward_diagonals[j + i + 1]
            conflicts += downward_diagonals[j - (i + 1) + N]
            
            # update min_conflict variables if this is the first row or if this row is a minimum conflict row
            if min_conflict_rows == []:
                min_conflict_rows.append(j)
                min_conflict = conflicts
            elif conflicts == min_conflict:
                min_conflict_rows.append(j)
            elif conflicts < min_conflict:
                min_conflict_rows.clear()
                min_conflict_rows.append(j)
                min_conflict = conflicts
        
        # pick a random row out of the minimum conflict rows, and mark that as the greedy initialization for that column
        best_pos = min_conflict_rows[np.random.randint(0, len(min_conflict_rows))]
        greedy_init[i + 1] = best_pos

        # update the conflict tracker arrays which will be needed when initializing the next column
        rows[best_pos] += 1
        downward_diagonals[best_pos - (i + 1) + N] += 1
        upward_diagonals[best_pos + i + 1] += 1

    return greedy_init.astype(int)


if __name__ == '__main__':
    # You can test your code here
    print(initialize_greedy_n_queens(4))
