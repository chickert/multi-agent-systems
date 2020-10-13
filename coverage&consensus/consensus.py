import numpy as np
import matplotlib.pyplot as plt


# nodes for storing information
class Node(object):

    def __init__(self, init_state):

        self._prev_state = init_state
        self._next_state = init_state

    # store the state update
    def update(self, update):
        self._next_state += update

    # push the state update
    def step(self):
        self._prev_state = self._next_state

    @property
    def state(self):
        return self._prev_state


# adversarial node
class AdversaryNode(object):

    def __init__(self, init_state, target, epsilon_adv = 0.01):

        self._prev_state = init_state
        self._next_state = init_state
        
        self._target = target
        self._epsilon = epsilon_adv

    # store the state update
    def update(self, update):
        if self._prev_state < self._target:
            self._next_state += self._epsilon
        if self._prev_state > self._target:
            self._next_state -= self._epsilon
        if self._prev_state == self._target:
            self._next_state += 0

    # push the state update
    def step(self):
        self._prev_state = self._next_state

    @property
    def state(self):
        return self._prev_state


# Graph for connecting nodes
class Graph(object):

    def __init__(self, node_list, adj_matrix, epsilon = 0.2, threshold = 0., sigma = 0.1):

        self.node_list = node_list
        self.adj_matrix = adj_matrix


        self._epsilon = epsilon

        self._finished = False      # bool determining when we've reached a threshold
        self._threshold = threshold
        self._sigma = sigma

    # updates the graph
    def update_graph(self):

        # First, find and store the update value for each node based on Eqn 15 from Olfati-Saber et al.
        # We do not update on this first loop so as to ensure all nodes can calculate their update
        # before any have moved in this 'round' of iteration
        for i, node_i in enumerate(self.node_list):
            list_to_sum = []
            for j, node_j in enumerate(self.node_list):
                x_j = node_j.state
                # The code below adds stochasticity to the system, with mean x_j and std dev sigma
                # x_j = np.random.normal(loc=x_j, scale=self._sigma)
                target = self.adj_matrix[i][j] * (x_j - node_i.state)
                list_to_sum.append(target)

            update_val = self._epsilon * np.sum(list_to_sum)
            node_i.update(update_val)

        # Now that all nodes have their updates, they can take a step to execute those update
        for node in self.node_list:
            node.step()

    # returns the state of the nodes currently - you can disable print here
    def node_states(self):
        string = ""
        out = []
        for node in self.node_list:
            # string = string + node.state.astype('str') + "\t"
            string = string + str(node.state) + "\t"
            out.append(node.state)
        # print(string)

        return out

    # Checks if the graph has reached consensus somehow, even if there are adversaries
    def is_finished(self):
        global_mean = np.mean([node.state for node in self.node_list])
        total_error = np.sum([np.abs(node.state - global_mean) for node in self.node_list])
        if total_error > self._threshold:
            self._finished = False
        if total_error <= self._threshold:
            self._finished = True


    @property
    def finished(self):
        return self._finished
        # return self.is_finished()


# return a random adjacency matrix
def rand_adj(orig_adj_mat, p):
    # First check that the probability given is valid
    assert 0 <= p <= 1, "p must be between 0 and 1"
    # Next make a copy of the original graph, so we can leave the original graph intact for future iterations
    rand_adj_mat = orig_adj_mat.copy()

    # Now reconstruct the adjacency matrix, but with a probability p of 'dropout'
    # Note that while existing nodes can dropout, it is not the case that a dead node can ever `come to life'
    for i in range(rand_adj_mat.shape[0]):
        for j in range(rand_adj_mat.shape[1]):
            if i == j:
                rand_adj_mat[i][j] = 0
            else:
                probabilistic_boolean_mask = np.random.choice([0, 1], p=[1 - p, p])
                rand_adj_mat[i][j] = orig_adj_mat[i][j] * probabilistic_boolean_mask
    return rand_adj_mat


# return the Fiedler value to show strong connection of the array
def fiedler(adj_mat):

    # First, construct the graph Laplacian
    degree_mat = np.zeros(shape=adj_mat.shape)
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if i == j:
                degree_mat[i][j] = np.sum(adj_mat[i])
    graph_laplacian = degree_mat - adj_mat

    # Find the Eigenvalues of the graph laplacian
    eigvals = np.linalg.eigvals(graph_laplacian)
    sorted_eigvals = np.sort(eigvals)

    # The Fiedler value is the second smallest eigenvalue, so grab that
    fiedler_val = sorted_eigvals[1]
    return fiedler_val


# plots the development of node values in the consensus problem over time
def plot_states(node_states):

    steps = np.arange(len(node_states))

    _, ax = plt.subplots()
    for i in range(node_states.shape[1]):
        line, = ax.plot(steps, node_states[:, i])
        line.set_label(i)
    plt.xlabel("Steps")
    plt.ylabel("Node Values")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    node_list = [Node(4.0), Node(2.0), Node(-1.0), Node(3.0), Node(0.0)]
    # Uncomment the below to create lists with an adversarial node
    # adv_node_list = [Node(4.0), AdversaryNode(init_state=2, target=5.0, epsilon_adv=0.01), Node(-1.0), Node(3.0), Node(0.0)]
    # adv_node_list = [Node(4.0), AdversaryNode(init_state=4.9, target=5.0, epsilon_adv=0.01), Node(-1.0), Node(3.0), Node(0.0)]

    # Uncomment the adjacency matrix that you want to test.
    # linear formation
    adj_matrix = np.array([[0, 1, 0, 0, 0],
                            [1, 0, 1, 0, 0],
                            [0, 1, 0, 1, 0],
                            [0, 0, 1, 0, 1],
                            [0, 0, 0, 1, 0]])
    # print(f'Fiedler Value for linear formation: {fiedler(adj_matrix)}')

    # circular formation
    # adj_matrix = np.array([[0, 1, 0, 0, 1],
    #                         [1, 0, 1, 0, 0],
    #                         [0, 1, 0, 1, 0],
    #                         [0, 0, 1, 0, 1],
    #                         [1, 0, 0, 1, 0]])
    # print(f'Fiedler Value for circular formation: {fiedler(adj_matrix)}')

    # fully connected
    # adj_matrix = np.array([[0, 1, 1, 1, 1],
    #                         [1, 0, 1, 1, 1],
    #                         [1, 1, 0, 1, 1],
    #                         [1, 1, 1, 0, 1],
    #                         [1, 1, 1, 1, 0]])
    # print(f'Fiedler Value for fully connected formation: {fiedler(adj_matrix)}')

    # linear formation test case that weights directly-connected nodes twice as strong
    # as other connections
    # adj_matrix = np.array([[0, 2, 0, 0, 0],
    #                         [2, 0, 2, 0, 0],
    #                         [0, 2, 0, 2, 0],
    #                         [0, 0, 2, 0, 2],
    #                         [0, 0, 0, 2, 0]])
    # print(f'Fiedler Value for weighted linear formation: {fiedler(adj_matrix)}')

    # circular formation test case that weights directly-connected nodes twice as strong
    # as other connections
    # adj_matrix = np.array([[0, 2, 0, 0, 2],
    #                         [2, 0, 2, 0, 0],
    #                         [0, 2, 0, 2, 0],
    #                         [0, 0, 2, 0, 2],
    #                         [2, 0, 0, 2, 0]])
    # print(f'Fiedler Value for weighted circular formation: {fiedler(adj_matrix)}')

    # fully connected test case that weights directly-connected nodes twice as strong
    # as other connections
    # adj_matrix = np.array([[0, 2, 1, 1, 2],
    #                         [2, 0, 2, 1, 1],
    #                         [1, 2, 0, 2, 1],
    #                         [1, 1, 2, 0, 2],
    #                         [2, 1, 1, 2, 0]])
    # print(f'Fiedler Value for weighted fully connected formation: {fiedler(adj_matrix)}')

    # linear formation with gap in middle
    # adj_matrix = np.array([[0, 1, 0, 0, 0],
    #                         [1, 0, 1, 0, 0],
    #                         [0, 1, 0, 0, 0],
    #                         [0, 0, 0, 0, 1],
    #                         [0, 0, 0, 1, 0]])
    # print(f'Fiedler Value for broken linear formation: {fiedler(adj_matrix)}')


    # graph = Graph(node_list, adj_matrix)
    graph = Graph(node_list, adj_matrix, threshold=0.05)

    # Use this line to use the adversarial node list from above
    # (also ensure that the appropriate node list is uncommented above)
    # graph = Graph(adv_node_list, adj_matrix, threshold=0.05)

    node_states = []
    # I added in this line so the plots also show the initial states
    node_states.append(graph.node_states())
    for _ in range(200):
        # The following two lines implement dropout nodes (an unreliable graph)
        # rand_adj_mat = rand_adj(orig_adj_mat=adj_matrix, p=0.5)
        # graph.adj_matrix = rand_adj_mat

        # If you want stochasticity, uncomment the appropriate line in the update_graph() function above
        graph.update_graph()
        node_states.append(graph.node_states())
        # This checks the error and update the graph.finished attribute if necessary
        graph.is_finished()
        if graph.finished:
            break

    node_states = np.array(node_states)
    plot_states(node_states)
    


