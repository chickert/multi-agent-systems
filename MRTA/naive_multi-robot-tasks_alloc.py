import simpy
import numpy as np


def calculate_avg_original_time(num_robots, task_runtime_list, env):
    '''
    Calculates the time required assuming all tasks are single-robot tasks.
    '''
    num_tasks = len(task_runtime_list)
    total_time = 0
    # First consider the case when there are at least as many robots as tasks.
    # Every task has a robot to work on it from the beginning, without needing to
    # wait in a queue while robots finish other tasks.
    if num_robots >= num_tasks:
        longest_task_runtime = np.max(task_runtime_list)
        longest_task = simpy.Container(env, init=longest_task_runtime, capacity=longest_task_runtime)
        while longest_task.level > 0:
            longest_task.get(1)
            total_time += 1

    # Next consider the case when there are more tasks than robots
    if num_robots < num_tasks:
        # Create a dict of all the values as tasks (using simpy containers) and keys as unique task ids
        unassigned_task_dict = dict()

        # Sort the list so the longest tasks are assigned first (which mitigates possibility of a case in which
        # towards the end of the process, a single robot is hacking away at a large task while all others sit idle
        # for an extended period of time)
        task_runtime_list.sort(reverse=True)
        task_id = 0
        for runtime in task_runtime_list:
            task = simpy.Container(env, init=runtime, capacity=runtime)
            unassigned_task_dict[task_id] = task
            task_id += 1

        inprogress_task_dict = dict()
        # Now allocate the homogenous robots to the tasks.
        # Since they are homogenous, the assignment order does not matter
        for bot in range(num_robots):
            # While a task is in unassigned_task_dict, it's available for a robot to be assigned to it
            # Once a bot is assigned to it, it moves to the inprogress_task_dict
            # Once a a task's level is 0, then it is removed from the inprogress_task_dict
            # All .get(1) methods are called at the same time across the robots, and a tally is added to
            # total_time for each call (which simulates a minute passing)
            inprogress_task_dict[bot] = None
        while len(unassigned_task_dict) > 0 or any(inprogress_task_dict.values()):
            # Move tasks from the unassigned_task_dict to the inprogress_task_dict as space allows
            for bot in inprogress_task_dict.keys():
                if inprogress_task_dict[bot] is None:
                    if len(unassigned_task_dict) > 0:
                        inprogress_task_dict[bot] = unassigned_task_dict.popitem()

            # Let the bots each do one minute of work on the tasks they have, and increment the total time accordingly
            for bot in inprogress_task_dict.keys():
                if inprogress_task_dict[bot] is not None:
                    inprogress_task = inprogress_task_dict[bot][1]
                    inprogress_task.get(1)
            total_time += 1

            # If a task is finished, remove it from the inprogress_task_dict so the robot that completed it can
            # begin work on another task.
            for bot in inprogress_task_dict.keys():
                if inprogress_task_dict[bot] is not None:
                    inprogress_task = inprogress_task_dict[bot][1]
                    if inprogress_task.level <= 0:
                        inprogress_task_dict[bot] = None

    avg_original_time = total_time

    # print(f"Average original time: {avg_original_time}")
    return avg_original_time


def calculate_multi_robot_task_time(num_robots, task_runtime_list, env):
    '''
    A naive method for assigning groups of robots to multi-robot tasks
    '''
    # Sort the task list so that the most time-consuming task is first
    num_tasks = len(task_runtime_list)
    task_runtime_list.sort(reverse=True)
    task_list = []

    # Build the tasks
    for runtime in task_runtime_list:
        task = simpy.Container(env, init=runtime, capacity=runtime)
        task_list.append(task)

    # Allocate the tasks, counting each round of get() calls (one get() call per robot) as one timestep passed
    # Keep track of all the timesteps used to get a measure of total time
    multi_robot_task_time = 0
    leftover = 0
    for task in task_list:
        # This simply keeps track of 'leftovers' in the case that there are more robots assigned to a task
        # than the task itself can accommodate, so the 'leftover' robots can be assigned to the next task in the queue.
        while leftover > 0 and task.level > 0:
            if leftover > task.level:
                decrement_to_leftover = task.level
                task.get(task.level)
                leftover -= decrement_to_leftover
            if leftover <= task.level:
                task.get(leftover)
                leftover = 0

        # The robots swarm to complete the tasks, with some leftover if the number is too great
        while task.level > 0:
            multi_robot_task_time += 1
            for robot in range(num_robots):
                task.get(1)
                if task.level <= 0:
                    leftover += 1

    # print(f'Multi-robot task time: {multi_robot_task_time}')

    return multi_robot_task_time


def st_mr_ia(num_robots, task_runtime_list):

    env = simpy.Environment()

    # Find the cumulative time of all tasks (if a single robot was responsible for completing all tasks on its own)
    cumulative_task_time = np.sum(task_runtime_list)
    # print(f"Cumulative task time (for example, if only had 1 robot to work on them): {cumulative_task_time}")

    # Find the average original time (the time without allowing robots to collaborate on a given task)
    avg_original_time = calculate_avg_original_time(num_robots, task_runtime_list, env)

    # Find the time required when we allow for multi-robot tasks
    multi_robot_task_time = calculate_multi_robot_task_time(num_robots, task_runtime_list, env)

    return cumulative_task_time, avg_original_time, multi_robot_task_time


def run(num_robots, num_task):

    cumulative_task_list = []
    avg_orig_time_list = []
    multi_robot_time_list = []

    for _ in range(50):
        task_runtime_array = np.random.randint(low=1, high=30, size=(num_task,), dtype=int)
        task_runtime_list = task_runtime_array.tolist()
        cumulative_task_time, avg_orig_time, multi_robot_time = st_mr_ia(num_robots=num_robots,
                                                                         task_runtime_list=task_runtime_list)
        cumulative_task_list.append(cumulative_task_time)
        avg_orig_time_list.append(avg_orig_time)
        multi_robot_time_list.append(multi_robot_time)

    final_cumulative_task_time = np.mean(cumulative_task_list)
    final_avg_orig_time = np.mean(avg_orig_time_list)
    final_multi_robot_mean_time = np.mean(multi_robot_time_list)
    final_multi_robot_std = np.std(multi_robot_time_list)

    print(f'{num_robots}    \t{num_task}    \t{final_cumulative_task_time}    \t{final_avg_orig_time}   \t\t{final_multi_robot_mean_time}   \t{final_multi_robot_std}', flush=True)


def make_table():
    np.random.seed(1)
    print('Robots', '\tTasks', '\tCumul.Time', '\tAvg.Orig.Time',  '\tMu',  '\t\tSigma', flush=True)
    run(11, 4)
    run(8, 3)
    run(6, 4)
    run(4, 12)
    run(8, 8)


def main():
    make_table()


if __name__ == '__main__':
    main()
