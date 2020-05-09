import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    ### YOUR CODE GOES HERE
    upward_diagonals = np.zeros(2*N) # indexed by using row_num + col_num which remains constant
    downward_diagonals = np.zeros(2*N) # indexed by using (row_num - col_num + N) which remains constant
    rows = np.zeros(N)

    # setting up data structures to remember how many conflicts exist in each row, or diagonal
    for col in range(len(initialization)):
        row = initialization[col]
        rows[row] += 1
        upward_diagonals[row + col] += 1
        downward_diagonals[row - col + N] += 1

    for idx in range(max_steps):
        if solves_problem(rows, upward_diagonals, downward_diagonals): # return the current config if it solves the problem
            return solution, idx
        else: # if not, find the optimal position for a random queen 

            conflicted_queen = np.random.randint(0, len(solution)) # conflicted_queen is a column number (the queen in that column)
            
            # makes sure that a conflict actually exists in that column
            while (not verify_conflicts(solution[conflicted_queen], conflicted_queen, rows, upward_diagonals, downward_diagonals, N)):
                conflicted_queen = np.random.randint(0, len(solution))

            # find the optimal position for the queen, and update the data structures which kept track of the number of conflicts
            old_pos = solution[conflicted_queen] 
            new_pos = minimize(N, conflicted_queen, rows, upward_diagonals, downward_diagonals)

            rows[old_pos] -= 1
            upward_diagonals[old_pos + conflicted_queen] -= 1
            downward_diagonals[old_pos - conflicted_queen + N] -= 1

            solution[conflicted_queen]  = new_pos
            rows[new_pos] += 1
            upward_diagonals[new_pos + conflicted_queen] += 1
            downward_diagonals[new_pos - conflicted_queen + N] += 1
            
            num_steps += 1

    # if a solution wasn't found within the max iterations
    return [], -1


def minimize(N: int, idx: int, rows: list, upward_diagonals: list, downward_diagonals: list) -> int:
    """
    Returns the best position (ie. row) for the queen in the idx column which will minimize the number of conflicts it has
    """
    min_conflict_rows = []
    min_conflict = -1

    # go through each row and see which row minimizes the conflicts at that column based on the data structures
    for i in range(N): # i is a row
        conflicts = 0

        conflicts += rows[i]
        conflicts += upward_diagonals[i + idx]
        conflicts += downward_diagonals[i - idx + N]
                
        if min_conflict_rows == []:
            min_conflict_rows.append(i)
            min_conflict = conflicts
        elif conflicts == min_conflict:
            min_conflict_rows.append(i)
        elif conflicts < min_conflict:
            min_conflict_rows.clear()
            min_conflict_rows.append(i)
            min_conflict = conflicts

    return min_conflict_rows[np.random.randint(0, len(min_conflict_rows))]


def verify_conflicts(row: int, col: int, row_conflicts: list, upward_diagonal_conflicts: list, downward_diagonal_conflicts: list, N: int) -> bool:
    """
    Checks if the queen at (row, col) on the chessboard has any conflicts with the rest of the board. Returns True if at least one
    conflict exists, and false otherwise.
    """

    if (row_conflicts[row] <= 1 and upward_diagonal_conflicts[row + col] <= 1 and downward_diagonal_conflicts[row - col + N]  <= 1):
        return False
    else:
        return True


def solves_problem(row_conflicts: list, upward_diagonal_conflicts: list, downward_diagonal_conflicts: list) -> bool:
    """
    Checks if the current configuration of queens on the chess board produces zero conflicts. Returns True if no conflicts are found
    in rows or diagonals, and False otherwise.
    """

    # check to see that there doesn't exist more than one conflict in any row/diagonal
    # one is fine since it means that one queen exists in that row/diagonal
    for conflict_num in row_conflicts:
        if conflict_num > 1:
            return False
    
    for conflict_num in upward_diagonal_conflicts:
        if conflict_num > 1:
            return False

    for conflict_num in downward_diagonal_conflicts:
        if conflict_num > 1:
            return False
        
    return True


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 2000 

    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)

    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)
    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)

    # Plot the solution produced by your algorithm
    print(n_steps)
    plot_n_queens_solution(assignment_solved)


