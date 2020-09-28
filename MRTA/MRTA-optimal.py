from scipy import optimize
import numpy as np


def robot_task_assignment(num_tasks, num_robots, quality_matrix, cost_matrix):
    # Check the input matrices are same shape
    assert quality_matrix.shape == cost_matrix.shape, "Error: Input matrices are different shapes"
    # Check that the input matrices' dimensions match the number of tasks and robots, and are oriented correctly
    assert num_tasks == quality_matrix.shape[0], f'Error: num_tasks is {num_tasks} but input matrices have ' \
                                                 f'{quality_matrix.shape[0]} rows'
    assert num_robots == quality_matrix.shape[1], f'Error: num_robots is {num_robots} but input matrices have '\
                                                  f'{quality_matrix.shape[1]} columns'

    # Calculate the utility matrix
    utility_matrix = quality_matrix - cost_matrix

    # Use scipy to find the row and column indices corresponding to the optimal assignment
    # First we need to get the opposite of the utility matrix, a sort of "adjusted cost matrix" for the scipy function
    # Note that simply using the regular cost matrix wouldn't account for the different quality that assignments have
    adjusted_cost_matrix = cost_matrix - quality_matrix
    row_idx, col_idx = optimize.linear_sum_assignment(adjusted_cost_matrix)

    # Do a quick double check that row_idx and col_idx arrays are same size:
    assert row_idx.shape == col_idx.shape, "Error: Column and row arrays are different size"
    solutions = []

    # Print the solutions in readable format and return them
    for assignment in range(len(row_idx)):
        robot = col_idx[assignment]
        task = row_idx[assignment]
        print(f'Assignment #{assignment+1}:\tr = {robot}, n = {task}, q - c = {utility_matrix[task, robot]}')
        solutions.append((robot, task, utility_matrix[task, robot]))

    return solutions


def print_summary(N, R, Q, C):
    print(f"Number of tasks (N) = {N}")
    print(f"Number of robots (R) = {R}")
    print(f"Quality matrix is:\n{Q}")
    print(f"Cost matrix is:\n{C}")
    print(f"Calculated utility matrix is:\n{Q-C}")
    print("Solution is:")

def test1():
    N = 2
    R = 5
    Q = np.array([[1, 3, 5, 7, 9], [1, 2, 3, 4, 6]])
    C = np.array([[1, 3, 4, 5, 1], [1, 3, 4, 1, 3]])
    print_summary(N, R, Q, C)
    return robot_task_assignment(N, R, Q, C)


def test2():
    N = 5
    R = 3
    Q = np.array([[4, 4, 99], [-23, 1, 0], [100, 3, 0], [53, 7, 0], [-4, 6, 10]])
    C = np.array([[4, 55, 9], [-23, 0, 0], [200, 0, 3], [5, 0, 12], [10, 0, 0]])
    print_summary(N, R, Q, C)
    return robot_task_assignment(N, R, Q, C)


def test3():
    N = 4
    R = 5
    Q = np.array([[3.3, 5, 5, 7, 9], [2, 3, 2, 4, 9], [9, 1, 3, 4, 6], [1, 6, 3, 4.8, 6]])
    C = np.array([[1, 4, 3, 4, 1], [7.9, 2, 5, 4, 6], [0, 2, 3, 4, 2], [0, 2.2, 3, 4, 6]])
    print_summary(N, R, Q, C)
    return robot_task_assignment(N, R, Q, C)


def test4():
    N = 2
    R = 2
    Q = np.array([[1, 0], [1_000_000, 0]])
    C = np.array([[0, 0], [0, 0]])
    print_summary(N, R, Q, C)
    return robot_task_assignment(N, R, Q, C)


def test5():
    N = 2
    R = 2
    Q = np.array([[1, -1000], [10, 0]])
    C = np.array([[0, 0], [0, 0]])
    print_summary(N, R, Q, C)
    return robot_task_assignment(N, R, Q, C)

def main():
    print("################################")
    print("Test 1:")
    test1()
    print("\n")
    print("################################")
    print("Test 2:")
    test2()
    print("\n")
    print("################################")
    print("Test 3:")
    test3()
    print("\n")
    print("################################")
    print("Test 4:")
    test4()
    print("\n")
    print("################################")
    print("Test 5:")
    test5()
    return


if __name__ == '__main__':
    main()










