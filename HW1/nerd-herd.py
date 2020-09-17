import random 


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
                env.move_bot(bot, (x_move, y_move))

        if not flock:
            for bot in self.bots:
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)
                # Since bot must move in some direction, ensure that (x_move, y_move) != (0,0)
                while (x_move, y_move) == (0, 0):
                    x_move = random.randint(-1, 1)
                    y_move = random.randint(-1, 1)
                env.move_bot(bot, (x_move, y_move))

        self.grid = self.update_grid()
            
    def aggregate(self, loc):
        '''
        Move all bots to grid coordinate loc (tuple).
        After this method is called, all aggregation should be complete (each bot will have taken all
        steps, likely more than one, to completely aggregate.)
        '''
        x_goal, y_goal = loc
        assert type(x_goal) is int, "Error: goal x-coord must be an integer"
        assert type(y_goal) is int, "Error: goal y-coord must be an integer"
        assert 0 <= x_goal < self.size, "Error: goal x-coord is off the grid grid"
        assert 0 <= y_goal < self.size, "Error: goal y-coord is off the grid"
        for bot in self.bots:
            while bot.x != x_goal:
                if bot.x < x_goal:
                    env.move_bot(bot, (1, 0))
                if bot.x > x_goal:
                    env.move_bot(bot, (-1, 0))
            while bot.y != y_goal:
                if bot.y < y_goal:
                    env.move_bot(bot, (0, 1))
                if bot.y > y_goal:
                    env.move_bot(bot, (0, -1))
        self.grid = self.update_grid()

    def disperse(self):
        '''
        Move all bots away from centroid, each in a random direction, for 3 steps.
        FROM OH: Assume disperse only happens once aggregated.
        '''
        x = 4
        assert x == 1, "NEED TO UPDATE disperse() method based on Piazza post!"
        # Disperse can only called after all bots have aggregated to same location,
        # since it is impossible for all robots to move in a random direction AND away from a centroid unless
        # all robots are already at that centroid.
        # So check that they are indeed all in the same location
        for bot in self.bots:
            assert (bot.x, bot.y) == (self.bots[0].x, self.bots[0].y), "Error: Attempting to call disperse before " \
                                                                       "all robots have aggregated to single location"
        for bot in self.bots:
            x_move = random.randint(-1, 1)
            y_move = random.randint(-1, 1)
            # Since bot must move in some direction, ensure that (x_move, y_move) != (0,0)
            while (x_move, y_move) == (0, 0):
                x_move = random.randint(-1, 1)
                y_move = random.randint(-1, 1)
            # Now take 3 steps in the random direction that has now been defined
            for _ in range(3):
                env.move_bot(bot, (x_move, y_move))

        self.grid = self.update_grid()

    def flock(self, loc, t=5):
        '''
        Aggregate all bots to grid coordinate loc (tuple)
        Then have the flock safe wander for t (int) steps.
        Afterwards, disperse. 
        Display the grid after each of these steps, including after each safe wander interation.
        '''
        self.aggregate(loc)
        self.display_grid()

        x = 4
        assert x == 2, "DO WE WANT THIS TO WANDER AS A FLOCK? I DON'T THINK SO?"
        for _ in range(t):
            self.safe_wander()
            self.display_grid()

        self.disperse()
        self.display_grid()

    def safe_wander_sense(self, flock=False):
        '''
        Move each bot one step. 
        If flock, all bots in a flock move in same random direction.
        Otherwise, each bot moves in its own random direction
        '''

        x = 4
        assert x == 2, "How to move in same random direction if that's not coordinated, esp. if no sensing radius?"
        self.safe_wander()

    def aggregate_sense(self, sense_r):
        '''
        Aggregate bots into one or more flocks, each using sensing radius of sense_r (int).
        '''
        
        pass

    def disperse_sense(self):
        '''
        Move all bots away from their respective flock's centroid.
        '''
        pass

    def flock_sense(self, sense_r, t=5):
        '''
        Aggregate all bots to grid coordinate loc (tuple) using sensing radius sense_r.
        Then have the flock(s) safe wander for t (int) steps.
        Afterwards, disperse flock/s beyond aggregation centroid/s. 
        Display the grid after each of these steps, including after each safe wander interation.
        '''
        pass

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
    bot1 = Robot(1,2)
    bot2 = Robot(0,0)
    bot3 = Robot(3,0)
    bot4 = Robot(3,0)

    bots = [bot1, bot2, bot3, bot4]

    env = Env(bots, 4)

    env.display_grid()

    import ipdb; ipdb.set_trace()

    # env.flock((2,2))

    # env.flock_sense(1)

    # Write additional test cases here, and print some lines to document each visual.