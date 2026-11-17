from pool_simulation import *

class ball():
    def __init__(self, val, x, y):
        self.radius = BALL_RADIUS
        self.val = val
        self.x = x
        self.y = y

    def get_coordinates(self):
        coor = (self.x, self.y)
        return coor

    def set_x(self, x):
        self.x = x

    def set_y(self, y):
        self.y = y

    def set_val(self, val):
        self.val = val

    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y

    def get_val(self):
        return self.val


class table():

    def __init__(self):
        self.balls = []
     
    def __init__(self, balls:list[ball]):
        self.balls_list = balls
        self.balls_dict = self.balls_list2dict()

    def balls_list2dict(self):  
        d = {}
        for ball in self.balls:
            d[ball.get_val()] = ball

        return d




def hit_ball_check(start_coor, end_coor, balls:list[ball]):
    x0 = start_coor[0]
    y0 = start_coor[1]
    x1 = end_coor[0]
    y1= end_coor[1]
    try:
        m = (y1-y0)/(x1-x0)
    except:
        m = 1000
    planned_line = lambda x: m* (x - x0) + y0

    dir = abs(x1-x0) / (x1-x0)
        