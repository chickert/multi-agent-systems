import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.stats import multivariate_normal
import math

class Robot(object):

    def __init__(self, state, k=0.1):
        self._state = state     # 2-vec
        self._stoch_state = self._state + np.random.randn(2)
        self.input = [0,0]      # movement vector later

        # self.k = k * 0.001
        # Changed this to speed things up
        self.k = k * 1

    def update(self):     # update the robot state
        self._state += self.k * np.array(self.input)
        self._stoch_state = self._state + np.random.randn(2)
        self.input = [0, 0]

    @property
    def state(self):
        return np.array(self._state)

    @property
    def stoch_state(self):
        return np.array(self._stoch_state)


class Environment(object):

    def __init__(self, width, height, res, robots, alpha = -10, sigma = 0, cov = 5, target = []):       # width and height are in pixels, so actual dimensions are width * res in meters
        self.width = width
        self.height = height
        self.res = res

        # bottom left corner is 0, 0 both in pixels and in meters
        self.robots = robots        # initialized w/ array of Robot objects
        self.meas_func = np.zeros((len(robots)))
        self.dist = np.zeros((2, len(robots)))

        # define the points you're iterating over
        self.pointsx = np.arange(0, width, res)
        self.pointsy = np.arange(0, height, res)

        self.alpha = alpha
        self.sigma = sigma
        self.cov = cov

        self.target = target
        self.global_k = None

    def update_meas_func_and_dist(self, botstate_array, point):
        # Although we only need meas_func [f(p_i, q)] to calculate g, we also take advantage
        # of the values here to find and store the dist [(q - p_i)] for later use

        _point_asarray = np.empty(shape=(2, len(botstate_array)))
        _point_asarray[0, :] = point[0]
        _point_asarray[1, :] = point[1]

        # We find the simple "distance" here as a difference, and store as an array filled with a 2-dim value for each robot
        self.dist = _point_asarray - np.transpose(botstate_array)
        # We find the f(p_i, q) value, and store as an array filled with a scalar value for each robot
        for i in range(len(self.meas_func)):
            self.meas_func[i] = 0.5 * np.linalg.norm(self.dist[:, i])

    def calc_g_alpha(self, point):
        """
        This calculates g_alpha in line with Eqn. 2 in Schwager et al.
        """
        botstate_list = []
        k_list = []
        for bot in self.robots:
            # Here the second line adds stochasticity to the system by changing what the robot reports
            # The first line eliminates this stochasticity
            botstate_list.append(bot.state)
            # botstate_list.append(bot.stoch_state)
            k_list.append(bot.k)
        botstate_array = np.array(botstate_list)

        # We assume that all bots have the same k value (i.e., all move at same speed)
        # The below code sets the global_k and does a rough check on that assumption
        self.global_k = k_list[0]
        assert np.mean(k_list) == self.global_k, "Robots have different k values"

        # We calculate our f(p_i, q) meas_func and dist, and store for use again later
        self.update_meas_func_and_dist(botstate_array=botstate_array, point=point)

        # The boolean mask handles div by 0
        boolean_mask = self.meas_func.astype(bool)
        value_to_sum = np.power(self.meas_func, self.alpha, out=np.zeros(len(botstate_array)), where=boolean_mask)
        sum_result = np.sum(value_to_sum)

        # Handle the div by 0
        if sum_result == 0 and self.alpha < 0:
            return 0
        return sum_result ** (1 / self.alpha)

    # calc the mixing function for the function aka g_alpha, also record f(p, q) and dist, point is np.array([x,y])
    def mix_func(self, point, value=1):
        """
        This function implements eqn. 4 from Schwager et al., which also requires implementing eqn. 2
        """
        # Note that the "value" parameter here corresponds to the Phi(q) function in the Schwager et al. paper

        # First we find g_alpha
        # In the process of doing so, we calculate our f(p_i, q) meas_func and dist,
        # which we store for later use to save computation
        g_alpha = self.calc_g_alpha(point=point)
        # Now find the complete integrand for each robot, and update that robot accordingly

        # We do this to handle the case in which g_alpha = 0, which would cause a div/0 error
        if g_alpha == 0:
            fs_over_g = self.meas_func * 0
        else:
            fs_over_g = self.meas_func / g_alpha

        # Similarly, the boolean mask handles div by 0
        boolean_mask = fs_over_g.astype(bool)
        parentheses_term = np.power(fs_over_g, (self.alpha - 1), out=np.zeros(len(fs_over_g)), where=boolean_mask)

        # Now we find the pdot_i for each bot.
        # Note that we don't have to do the integral in Eqn. 4 explicitly here; since mix_func
        # is called for every point below, and simply adds its pdot_i contribution via a moving sum to each
        # bot's input value, it implicitly calculates the integral as-is.
        for i, bot in enumerate(self.robots):
            integrand = parentheses_term[i] * self.dist[:, i] * value
            pdot_i = self.global_k * integrand
            bot.input += pdot_i

    def update_gradient(self, iter=0):
        # Uncomment the below if/else statements and rv (along with the second 'value' line in the nested for loops)
        # to run the simulation with a target point (note that the target can be stationary or moving)
        # rv = None
        # if(type(self.target) is np.ndarray):
        #     rv = multivariate_normal(mean = self.target[:, iter], cov = self.cov)
        # else:
        #     rv = multivariate_normal(mean = self.target, cov = self.cov)

        for x in self.pointsx:
            for y in self.pointsy:
                value = 1
                # Uncomment the below to run with a target point
                # value = rv.pdf((x,y)) * (np.sqrt(np.power((2 * np.pi * rv.cov[0,0]), 2)))
                self.mix_func(np.array([x, y]), value)

    def moves(self):
        for bot in self.robots:
            bot.update()


# function to run the simulation
def run_grid(env, iter):
    x = []
    y = []

    # initialize state
    for i, bot in enumerate(env.robots):

        x.append([bot.state[0]])
        y.append([bot.state[1]])

    # run environment for iterations
    for iteration in range(iter):
        env.update_gradient(iteration)
        env.moves()

        for i, bot in enumerate(env.robots):

            x[i].append(bot.state[0])
            y[i].append(bot.state[1])

        if (iteration % 5 == 0):
            print(f'Iteration is {iteration}')

    # set up the plot
    fig, ax = plt.subplots()
    points = []

    # plt the robot points
    plt.axes(ax)
    for i in range(len(env.robots)):
        plt.scatter(x[i], y[i], alpha=(i+1)/len(env.robots))
        points.append([x[i][-1], y[i][-1]])

    # if there is a target setup plot it
    if type(env.target) is tuple:
        plt.scatter(env.target[0], env.target[1])
    if(type(env.target) is np.ndarray):
        for i in range(iter):
            plt.scatter(env.target[0, i], env.target[1, i], alpha=(i+1)/iter)

    # set polygon bounds
    bounds = Polygon([(0,0), (10,0), (10,10), (0, 10)])
    b_x, b_y = bounds.exterior.xy
    plt.plot(b_x, b_y)        

    # set Voronoi
    print("Setting Voronoi")
    vor = Voronoi(np.array(points))
    voronoi_plot_2d(vor, ax=ax)

    ax.set_xlim((-1, 11))
    ax.set_ylim((-1, 11))

    plt.show()
    plt.close()

# The below code generates an array with (x,y) points that together trace a circle of radius r around the center
# The number of points in the array is given by the period parameter
def target(center=(5, 5), r=3, period=800):
    pi = math.pi
    full_circle = []
    for point in range(period):
        circle_point = (center[0] + math.cos(2 * pi / period * point) * r,
                        center[1] + math.sin(2 * pi / period * point) * r)
        full_circle.append(circle_point)
    full_circle_array = np.array(full_circle)
    full_circle_array = full_circle_array.transpose()
    return full_circle_array


if __name__ == "__main__":

    rob1 = Robot([4, 1])
    rob2 = Robot([2, 2])
    rob3 = Robot([5, 6])
    rob4 = Robot([3, 4])

    robots = [rob1, rob2, rob3, rob4]

    # env = Environment(10, 10, 0.1, robots)
    # env = Environment(10, 10, 0.1, robots, target=(5,5))
    # run_grid(env, 5)

    # This code runs the basic case with no target point
    # To run the stochastic case, uncomment the bot.stoch_state line in calc_g_alpha() above
    env = Environment(10, 10, 1, robots)
    run_grid(env, 200)

    # This code runs the case in which there is a stationary target point
    # MAKE SURE to also uncomment the appropriate section in update_gradient()
    # env = Environment(10, 10, 0.5, robots, target=(5,5))
    # run_grid(env, 200)

    # Uncomment the below (AND ALSO the necessary parts in update_gradient())
    # to run with the target point moving in a quarter-circle
    # period = 800
    # env = Environment(10, 10, 0.5, robots, target=target(center=(5,5), r=3, period=period))
    # run_grid(env, int(period * 0.25))


