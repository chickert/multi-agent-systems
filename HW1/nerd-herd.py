import random
import numpy as np


class Robot:
    def __init__(self, x, y):
        # index for row position in grid env
        self.x = x
        # index for column position in grid env
        self.y = y


class Env:
    def __init__(self, bots, size=10):
        # number of rows and columns in square grid
        self.size = size 
        # list of Robot objects
        self.bots = bots 
        # number of Robots in this Env
        self.num_bots = len(bots)
        # 2D list containing sets, empty or otherwise, of Robot objects at each coordinate location
        self.grid = self.update_grid()

    def update_grid(self):
        grid = [[set() for i in range(self.size)] for i in range(self.size)]
        for b in self.bots:
            grid[b.x][b.y].add(b)
        return grid

    def safe_wander(self, flock=False):
        '''
        Move each bot one step. 
        If flock, all bots in a flock move in same random direction.
        Otherwise, each bot moves in its own random direction
        '''
        if flock:
            x_move = random.randint(-1, 1)
            y_move = random.randint(-1, 1)
            # Since bots must move in some direction, ensure that (x_move, y_move) != (0,0)
            while (x_move, y_move) == (0, 0):
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)
            for bot in self.bots:
                self.move_bot(bot, (x_move, y_move))

        if not flock:
            for bot in self.bots:
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)
                # Since bot must move in some direction, ensure that (x_move, y_move) != (0,0)
                while (x_move, y_move) == (0, 0):
                    x_move = random.randint(-1, 1)
                    y_move = random.randint(-1, 1)
                self.move_bot(bot, (x_move, y_move))

        self.grid = self.update_grid()
            
    def aggregate(self, loc):
        '''
        Move all bots to grid coordinate loc (tuple).
        After this method is called, all aggregation should be complete (each bot will have taken all
        steps, likely more than one, to completely aggregate.)
        '''
        x_goal, y_goal = loc
        # Run a couple checks to ensure the loc is a point to which the bots can converge
        assert type(x_goal) is int, "Error: goal x-coord must be an integer"
        assert type(y_goal) is int, "Error: goal y-coord must be an integer"
        assert 0 <= x_goal < self.size, "Error: goal x-coord is off the grid"
        assert 0 <= y_goal < self.size, "Error: goal y-coord is off the grid"

        for bot in self.bots:
            while bot.x != x_goal:
                if bot.x < x_goal:
                    self.move_bot(bot, (1, 0))
                if bot.x > x_goal:
                    self.move_bot(bot, (-1, 0))
            while bot.y != y_goal:
                if bot.y < y_goal:
                    self.move_bot(bot, (0, 1))
                if bot.y > y_goal:
                    self.move_bot(bot, (0, -1))
        self.grid = self.update_grid()

    def is_not_valid_disperse_move(self, bot, centroid_x, centroid_y, x_move, y_move):
        '''
        Checks that a move is viable for the disperse method
        '''
        # Since bot must move in some direction, ensure that (x_move, y_move) != (0,0).
        # Then check that the direction actually moves the bot away from the centroid, rather than towards it.
        # True 'dispersion' here requires that the move NOT reduce distance in EITHER x- OR y-directions
        # (though it can have 0 change in one of the two directions).
        new_x = bot.x + x_move
        new_y = bot.y + y_move
        old_x_dist_to_centroid = np.abs(centroid_x - bot.x)
        old_y_dist_to_centroid = np.abs(centroid_y - bot.y)
        new_x_dist_to_centroid = np.abs(centroid_x - new_x)
        new_y_dist_to_centroid = np.abs(centroid_y - new_y)
        if (x_move, y_move) == (0, 0):
            return True
        elif new_x_dist_to_centroid < old_x_dist_to_centroid:
            return True
        elif new_y_dist_to_centroid < old_y_dist_to_centroid:
            return True
        # These last checks just ensure that if the bot is against a wall, it won't try to push into the wall,
        # which would not disperse as we intend
        elif new_x < 0:
            return True
        elif new_y < 0:
            return True
        elif new_x >= self.size:
            return True
        elif new_y >= self.size:
            return True
        else:
            return False

    def disperse(self):
        '''
        Move all bots away from centroid, each in a random direction, for 3 steps.
        '''

        # First find the global centroid
        x_list = []
        y_list = []
        for bot in self.bots:
            x_list.append(bot.x)
            y_list.append(bot.y)
        centroid_x = np.mean(x_list)
        centroid_y = np.mean(y_list)

        # Next, for each bot pick a random direction
        for bot in self.bots:
            x_move = random.randint(-1, 1)
            y_move = random.randint(-1, 1)
            # Check if the move is valid (i.e., actually moves bot away from centroid)
            # If not, reselect a random direction until a valid pick is achieved
            while self.is_not_valid_disperse_move(bot, centroid_x, centroid_y, x_move, y_move):
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)

            # Now take 3 steps in the random direction that has been defined
            for _ in range(3):
                self.move_bot(bot, (x_move, y_move))

        self.grid = self.update_grid()

    def flock(self, loc, t=5):
        '''
        Aggregate all bots to grid coordinate loc (tuple)
        Then have the flock safe wander for t (int) steps.
        Afterwards, disperse. 
        Display the grid after each of these steps, including after each safe wander interation.
        '''
        self.aggregate(loc)
        print(f"Aggregate to {loc}")
        self.display_grid()

        for timestep in range(t):
            self.safe_wander(flock=True)
            print(f'Safe wander {timestep + 1}')
            self.display_grid()

        self.disperse()
        print("Disperse")
        self.display_grid()

    def safe_wander_sense(self, flock=False):
        '''
        Move each bot one step. 
        If flock, all bots in a flock move in same random direction.
        Otherwise, each bot moves in its own random direction
        '''
        ##############################################
        '''
        E said Kathryn in piazza said we don't have to change our above safe_wander?
        So I pasted from above
        '''
        # self.safe_wander(flock)
        ###################################################
        if flock:
            x_move = random.randint(-1, 1)
            y_move = random.randint(-1, 1)
            # Since bots must move in some direction, ensure that (x_move, y_move) != (0,0)
            while (x_move, y_move) == (0, 0):
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)
            for bot in self.bots:
                self.move_bot(bot, (x_move, y_move))

        if not flock:
            for bot in self.bots:
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)
                # Since bot must move in some direction, ensure that (x_move, y_move) != (0,0)
                while (x_move, y_move) == (0, 0):
                    x_move = random.randint(-1, 1)
                    y_move = random.randint(-1, 1)
                self.move_bot(bot, (x_move, y_move))

        self.grid = self.update_grid()

    def aggregate_sense(self, sense_r):
        '''
        Aggregate bots into one or more flocks, each using sensing radius of sense_r (int).
        '''
        # Have each bot find all bots within sensing radius
        assert type(sense_r) is int, "Error: sensing radius must be non-negative integer"
        assert sense_r >= 0, "Error: sensing radius must be non-negative"

        # Start a loop that continues the aggregating process until all robots have stopped moving.
        # If aggregation takes too long, you can alter the 'while' terminating condition to a value arbitrarily above 0.
        total_bot_moves_this_round = np.inf
        while total_bot_moves_this_round > 0:
            total_bot_moves_this_round = 0

            for bot in self.bots:
                x_list = []
                y_list = []
                # Identify the bot's sensing radius and then scan over it.
                x_range_min = bot.x - sense_r
                x_range_max = bot.x + sense_r
                y_range_min = bot.y - sense_r
                y_range_max = bot.y + sense_r

                # Handle the cases in which the resulting indices would be out of range
                if x_range_min < 0:
                    x_range_min = 0
                if x_range_max >= self.size:
                    x_range_max = self.size - 1
                if y_range_min < 0:
                    y_range_min = 0
                if y_range_max >= self.size:
                    y_range_max = self.size - 1

                for x in range(x_range_min, x_range_max + 1):
                    for y in range(y_range_min, y_range_max + 1):
                        # Find all robots in sensing radius (including itself)
                        # and grab the x- and y-coords of them in order to calculate the local centroid.
                        # Note that since the robot will always register at least itself, the averaging function below
                        # will not fail (though we double-check with the assert statement below).
                        if self.grid[x][y]:
                            x_list.append(x)
                            y_list.append(y)

                # Calculate local centroid for the given bot.
                # To ensure it's an achievable spot (and to prevent equivocation between spots for intermediate values)
                # round the centroid float values to an int
                if not x_list or not y_list:
                    raise ValueError("Error: robot did not sense any robots (incl. itself),"
                                     "so cannot calculate local centroid")
                centroid_x = np.mean(x_list)
                centroid_y = np.mean(y_list)
                centroid_x = int(centroid_x)
                centroid_y = int(centroid_y)

                # Move the bot until it reaches its local centroid
                while bot.x != centroid_x:
                    if bot.x < centroid_x:
                        self.move_bot(bot, (1, 0))
                        total_bot_moves_this_round += 1
                    if bot.x > centroid_x:
                        self.move_bot(bot, (-1, 0))
                        total_bot_moves_this_round += 1
                while bot.y != centroid_y:
                    if bot.y < centroid_y:
                        self.move_bot(bot, (0, 1))
                        total_bot_moves_this_round += 1
                    if bot.y > centroid_y:
                        self.move_bot(bot, (0, -1))
                        total_bot_moves_this_round += 1

            # Now that each bot has moved to its local centroid based on the bot configuration
            # at the beginning of the round, update the grid and repeat the process.
            self.grid = self.update_grid()
            print(f'Total moves this round: {total_bot_moves_this_round}')

        # Finally, repeat the process as necessary via the while loop

    def disperse_sense(self, sense_r=1):
        '''
        Move all bots away from their respective flock's centroid.
        '''

        ########## FIRST DO MY ANSWER TRY #################
        ##### HAD TO ADD IN sense_r AS PARAM TO DO THIS ########
        # Move in that direction...but for how many steps???

        # Added a sense_r for generality but set to 1 since generally this is called after aggregate_sense is called,
        # so all are in their local flocks

        # First find the local centroid for each bot
        # Generally, this will just be the point it's at, since it will just have aggregated
        for bot in self.bots:
            x_list = []
            y_list = []
            # Identify the bot's sensing radius and then scan over it.
            x_range_min = bot.x - sense_r
            x_range_max = bot.x + sense_r
            y_range_min = bot.y - sense_r
            y_range_max = bot.y + sense_r

            # Handle the cases in which the resulting indices would be out of range
            if x_range_min < 0:
                x_range_min = 0
            if x_range_max >= self.size:
                x_range_max = self.size - 1
            if y_range_min < 0:
                y_range_min = 0
            if y_range_max >= self.size:
                y_range_max = self.size - 1

            for x in range(x_range_min, x_range_max + 1):
                for y in range(y_range_min, y_range_max + 1):
                    # Find all robots in sensing radius (including itself)
                    # and grab the x- and y-coords of them in order to calculate the local centroid.
                    # Note that since the robot will always register at least itself, the averaging function below
                    # will not fail (though we double-check with the assert statement below).
                    if self.grid[x][y]:
                        x_list.append(x)
                        y_list.append(y)

            # Calculate local centroid for the given bot.
            if not x_list or not y_list:
                raise ValueError("Error: robot did not sense any robots (incl. itself),"
                                 "so cannot calculate local centroid")
            centroid_x = np.mean(x_list)
            centroid_y = np.mean(y_list)

            # Next, for each bot pick a random direction
            x_move = random.randint(-1, 1)
            y_move = random.randint(-1, 1)
            # Check if the move is valid (i.e., actually moves bot away from centroid)
            # If not, reselect a random direction until a valid pick is achieved
            while self.is_not_valid_disperse_move(bot, centroid_x, centroid_y, x_move, y_move):
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)

            # Now take 1 step in the random direction that has been defined
            self.move_bot(bot, (x_move, y_move))

        # Update grid after all bots have moved
        self.grid = self.update_grid()
        ######################################################

        ###### NOW DO based on E comments: said Kathryn in Piazza said we don't have to change our disperse_sense?? ###
        # self.disperse()
        #############################################

    def flock_sense(self, sense_r, t=5):
        '''
        Aggregate all bots to grid coordinate loc (tuple) using sensing radius sense_r.
        Then have the flock(s) safe wander for t (int) steps.
        Afterwards, disperse flock/s beyond aggregation centroid/s. 
        Display the grid after each of these steps, including after each safe wander interation.
        '''
        self.aggregate_sense(sense_r=sense_r)
        print(f'Aggregate using sensing radius {sense_r}')
        env.display_grid()

        for timestep in range(t):
            self.safe_wander_sense(flock=True)
            print(f'Safe wander sense {timestep+1}')
            self.display_grid()

        self.disperse_sense()
        print(f'Disperse')
        self.display_grid()

    def move_bot(self, bot, move_cmd):
        '''
        Update position of bot (Robot obj) using move_cmd (tuple).
        Note that move_cmd = (x,y) where x and y are each integers between -1 and 1, inclusive.
        '''
        bot.x += move_cmd[0]
        bot.y += move_cmd[1]
        if bot.x >= self.size:
            bot.x = self.size-1
        if bot.x < 0:
            bot.x = 0
        if bot.y >= self.size:
            bot.y = self.size-1
        if bot.y < 0:
            bot.y = 0

    def display_grid(self):
        # print grid with number of bots in each coordinate location

        print("Grid["+("%d" % self.size)+"]["+("%d" % self.size)+"]")
        for i in range(0, self.size):
            for j in range(0, self.size):
                print(len(self.grid[i][j]), end="  ")
            print()
        print()


if __name__ == "__main__":
    # bot1 = Robot(1,2)
    # bot2 = Robot(0,0)
    # bot3 = Robot(3,0)
    # bot4 = Robot(3,0)
    #
    # bots = [bot1, bot2, bot3, bot4]
    #
    # env = Env(bots, 4)

    #### CH test ####
    bot1 = Robot(1,2)
    bot2 = Robot(8,8)
    bot3 = Robot(8,8)
    bot4 = Robot(4,4)
    bot5 = Robot(2, 6)
    bot6 = Robot(0, 7)
    bot7 = Robot(7,0)
    bot8 = Robot(8,0)
    bot9 = Robot(1,8)
    bots = [bot1, bot2, bot3, bot4, bot5, bot6, bot7, bot8, bot9]
    env = Env(bots, 9)
    ##################

    env.display_grid()

    import ipdb; ipdb.set_trace()

    # env.flock((2,2))

    # env.flock_sense(1)

    # Write additional test cases here, and print some lines to document each visual.