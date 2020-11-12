import numpy as np
import matplotlib.pyplot as plt


# this returns a bunch of weights - sets a mean alpha level, then randomly generates alphas around the means
class AlphaWeights(object):

    def __init__(self, mean_alpha, std = 0.1):

        self.std = std
        self.iter = 1

        # assuming fully connected - this is an adjancency matrix
        # assume the vector is [legitimate, spoofed], assume it's diagraph
        self.means = mean_alpha
        self.betas = self.means + np.random.randn(self.means.shape[0], self.means.shape[1]) * self.std

    # Calculate the updates fo the betas used in updating W
    # be sure to cap alpha at -0.5 and 0.5 at all times
    # this function should generate a random set of alphas based on the means and then update beta accordingly
    def update_betas(self):

        alpha = self.means + np.random.randn(self.means.shape[0], self.means.shape[1]) * self.std

        # cap alphas at 0.5 because that's the way it is in the paper
        thresh = alpha > 0.5
        alpha[thresh] = 0.5
        thresh = alpha < -0.5
        alpha[thresh] = -0.5

        # update beta because it's a running sum of alphas
        self.betas += alpha
        
        self.iter += 1


# define the simulation environment
class Environment(object):

    def __init__(self, leg, spoof, Theta, Omega):

        self.leg = leg         # legitimate nodes n_l
        self.spoof = spoof     # spoofed nodes n_s

        self.leg_len = leg.shape[0]
        self.full_len = leg.shape[0] + spoof.shape[0]

        self.Theta = Theta              # transition for spoofed based on leg - n_s x n_l   (row, col)
        self.Omega = Omega              # transition for spoofed based on spoofed - n_s x n_s

        # transition for legitimate based on spoof and leg - n_l x (n_l + n_s)
        # first n_l columns are the legit part W_L
        self.W = np.zeros((self.leg_len, self.full_len))

        self.iter = 1

    # updates according to the system dynamics given
    def update(self):

        self.leg = np.matmul(self.W, np.concatenate((self.leg, self.spoof), axis=0))
        self.spoof = np.matmul(self.Theta, self.leg) + np.matmul(self.Omega, self.spoof)

        # Uncomment the below for periodic spoof node values
        # period = 49
        # frequency = 1 / period
        # w = 2 * np.pi * frequency
        # timestep = self.iter
        # new_spoof_val = np.sin(w * timestep) + 4.
        # self.spoof.fill(new_spoof_val)
        # self.iter += 1


    # set the transitions of W
    # call alphaweights to get an updated beta, then use that to update W.
    # the code will call update to progress the simulation
    def transition_W(self, alphaweights):
        # get updated beta
        alphaweights.update_betas()

        # handle the first two cases in Eqn. 3 from Gil, Baykal, Rus
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                if i == j:
                    pass
                elif alphaweights.betas[i][j] >= 0:
                    self.W[i][j] = (1 / self.leg_len) * (1 - np.exp(-alphaweights.betas[i][j] / 2))
                elif alphaweights.betas[i][j] < 0:
                    self.W[i][j] = (1 / (2 * self.leg_len)) * np.exp(alphaweights.betas[i][j])

        # handle the last case in Eqn. 3 from Gil, Baykal, Rus
        for i in range(self.W.shape[0]):
            j = i
            # Here we add back in the W[i][j] term since we want the sum of the row excluding the diagonal element
            self.W[i][j] = 1 - (np.sum(self.W[i]) - self.W[i][j])


# it plots the states - basically the same function from HW 2
def plot_states(node_states, spoof_states):

    steps = np.arange(len(node_states))

    _, ax = plt.subplots()

    # Plot the legitimate node values
    for i in range(node_states.shape[1]):
        line, = ax.plot(steps, node_states[:, i], 'g')
    line.set_label(f'$x_L$ ({node_states.shape[1]} legitimate node values)')

    # plot the average
    initial_avg = np.average(node_states[0, :])
    print(f"Legitimate nodes' initial average state: {initial_avg}")
    line, = ax.plot(steps, np.ones(len(steps)) * initial_avg, 'b.')
    line.set_label("Initial average $x_L$")

    # Plot the spoofed nodes
    for i in range(spoof_states.shape[1]):
        line, = ax.plot(steps, spoof_states[:, i], 'r')
    line.set_label(f'$x_S$ ({spoof_states.shape[1]} spoof node values)')

    # Find the value to which the legitimate nodes converged
    final_avg = np.average(node_states[-1, :])
    print(f"Legitimate nodes' final average state: {final_avg}")

    plt.legend()
    plt.title("Plot of nodes' states over time")
    plt.show()


if __name__ == "__main__":

    # Set seed
    np.random.seed(0)

    # assume everything is in 1-D and fully connected
    leg = np.array([1, 2, 1.1, 1.9, 1.4, 2.3, 0.7, 2,1,2,1,2,1,0.5,0.8,1.5,1,2,1,2])

    # Generate 20 random legitimate node values with mean 1.5 for each run,
    # but with standard deviations of 0.1, 1, 3, and 5 (be sure to uncomment the correct std_dev)
    # mean = 1.5
    # std_dev = 0.1
    # std_dev = 1.
    # std_dev = 3.
    # std_dev = 5.
    # leg = mean + std_dev * np.random.randn(20)

    # Uncomment for the spoofed node value of your choosing
    # spoof = np.array([0., 0., 0., 0.])
    # spoof = np.array([2., 2., 2., 2.])
    spoof = np.array([4., 4., 4., 4.])
    # spoof = np.array([6., 6., 6., 6.])

    # Uncomment for varying numbers of spoofed nodes
    # spoof = np.array([4.])
    # spoof = np.array([4., 4., 4., 4.])
    # spoof = np.array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])
    # spoof = np.array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4.])

    # Setting up a new matrix of inputs for the dynamics
    alphas = np.ones((leg.shape[0] + spoof.shape[0], leg.shape[0] + spoof.shape[0]))
    alphas = 0.4 * alphas
    alphas[:, -spoof.shape[0]:] = -1 * alphas[:, -spoof.shape[0]:] 

    alphas = AlphaWeights(alphas)

    theta = np.zeros((spoof.shape[0], leg.shape[0]))
    # In the below case the spoof node values remain constant
    omega = np.eye(spoof.shape[0])
    # Uncomment the below to run the case in which spoof node values increase exponentially at the given rate
    # omega = np.eye(spoof.shape[0]) * 1.01

    # Define the environment
    env = Environment(leg, spoof, theta, omega)

    iter = 20

    # Run the simulation and plot
    leg_states = []
    spoof_states = []
    leg_states.append(env.leg)
    spoof_states.append(np.array(env.spoof.tolist()))
    for _ in range(iter):
        env.transition_W(alphas)        # update W at every iteration
        env.update()

        leg_states.append(env.leg)
        spoof_states.append(np.array(env.spoof.tolist()))

    plot_states(np.array(leg_states), np.array(spoof_states))

    print("out")