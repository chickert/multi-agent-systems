import numpy as np


def murdoch(num_tasks, num_robots, quality_matrix, cost_matrix):
    '''
    Implements the popular Murdoch algorithm for online multi-robot task allocation
    '''
    # Check the input matrices are same shape
    assert quality_matrix.shape == cost_matrix.shape, "Error: Input matrices are different shapes"
    # Check that the input matrices' dimensions match the number of tasks and robots, and are oriented correctly
    assert num_tasks == quality_matrix.shape[0], f'Error: num_tasks is {num_tasks} but input matrices have ' \
                                                 f'{quality_matrix.shape[0]} rows'
    assert num_robots == quality_matrix.shape[1], f'Error: num_robots is {num_robots} but input matrices have '\
                                                  f'{quality_matrix.shape[1]} columns'

    # Calculate utility matrix and convert to float for later
    utility_matrix = quality_matrix - cost_matrix
    utility_matrix = utility_matrix.astype(float)

    solutions = []

    # Present tasks to robots in the order they occur (no lookahead), and since robots can do
    # only one task at most, remove assigned robots from consideration for future tasks
    for task in range(utility_matrix.shape[0]):
        # Find robot-task pair with highest utility
        robot = np.argmax(utility_matrix[task, :])

        utility = utility_matrix[task, robot]
        print(f'Assignment #{task+1}:\tr = {robot}, n = {task}, q - c = {utility}')
        solutions.append((robot, task, utility))

        # Remove the robot from future consideration by setting its utility to negative infinity for all tasks in the
        # utility matrix
        utility_matrix[:, robot] = -np.inf

        # If all robots have been assigned (when utility_matrix is all -np.inf), quit
        if ~np.isfinite(utility_matrix).any():
            print("Process complete: All robots have been allocated")
            return solutions

    print("Process complete: All tasks have been allocated")
    return solutions


def print_summary(N, R, Q, C):
    '''
    Helper function for the tests below
    '''
    print(f"Number of tasks (N) = {N}")
    print(f"Number of robots (R) = {R}")
    print(f"Quality matrix is:\n{Q}")
    print(f"Cost matrix is:\n{C}")
    print(f"Calculated utility matrix is:\n{Q-C}")
    print("Solution is:")


'''
The tests below demonstrate the Murdoch algorithm in practice, and suggest a few cases in which
it demonstrates suboptimal performance.
'''


def test1():
    N = 2
    R = 4
    Q = np.array([[1, 3, 5, 4], [3, 2, 6, 4]])
    C = np.array([[0, 1, 0, 2], [1, 0, 1, 2]])
    print_summary(N, R, Q, C)
    return murdoch(N, R, Q, C)


def test2():
    N = 5
    R = 3
    Q = np.array([[4, 4, 99], [-23, 1, 0], [100, 3, 0], [53, 7, 0], [-4, 6, 10]])
    C = np.array([[4, 55, 9], [-23, 0, 0], [200, 0, 3], [5, 0, 12], [10, 0, 0]])
    print_summary(N, R, Q, C)
    return murdoch(N, R, Q, C)


def test3():
    N = 4
    R = 5
    Q = np.array([[3.3, 5, 5, 7, 9], [2, 3, 2, 4, 9], [9, 1, 3, 4, 6], [1, 6, 3, 4.8, 6]])
    C = np.array([[1, 4, 3, 4, 1], [7.9, 2, 5, 4, 6], [0, 2, 3, 4, 2], [0, 2.2, 3, 4, 6]])
    print_summary(N, R, Q, C)
    return murdoch(N, R, Q, C)


def test4():
    N = 2
    R = 2
    Q = np.array([[1, 0], [1_000_000, 0]])
    C = np.array([[0, 0], [0, 0]])
    print_summary(N, R, Q, C)
    return murdoch(N, R, Q, C)


def test5():
    N = 2
    R = 2
    Q = np.array([[1, -1000], [10, 0]])
    C = np.array([[0, 0], [0, 0]])
    print_summary(N, R, Q, C)
    return murdoch(N, R, Q, C)


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



