import math
import random
from typing import List,Union


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

TABLE_WIDTH = 255  # (cm)
TABLE_HEIGHT = 129  # (cm)

BALL_DIAMETER = 6  # Ball diameter (cm)
BALL_RADIUS = BALL_DIAMETER / 2  # Ball radius (cm)
SAFETY_DISTANCE = 5  #distance to stay away from the balls (cm)

ROBOT_WIDTH = 24 # cm
ROBOT_LENGTH = 20 # cm


# ============================================================================
# Input
# ============================================================================

input_positions = [
    (random.randint(BALL_RADIUS, int(TABLE_WIDTH - BALL_RADIUS)), random.randint(BALL_RADIUS, int(TABLE_HEIGHT - BALL_RADIUS)))
    for _ in range(16)
]

# If a ball is off the table his position is (0, 0)


# ============================================================================
# Utilities
# ============================================================================
class Point:
    def __init__ (self, x: float, y: float):
        self.x = x
        self.y = y

class Line:
    def __init__(self, p1 : Point, p2 : Point):
        self.p1 = p1
        self.p2 = p2

class Ball:
    def __init__(self, pos : Point):
        self.pos = pos

    def __str__(self):
        return f"X {self.pos.x}, Y {self.pos.y}"

class Circle:
    def __init__(self, center : Point, radius: float):
        self.center = center
        self.radius = radius

class Pocket:
    def __init__(self, pos: Point):
        self.pos = pos

    def __str__(self):
        return f"X {self.pos.x}, Y {self.pos.y}"

class Table:
    def __init__(self, corners : List[Point], pockets : List[Pocket],
                 cue_ball : Ball, my_balls : List[Ball],
                 enemy_balls : List[Ball], black_ball : Ball,
                 off_table_balls : int):
        self.corners = corners
        self.pockets = pockets
        self.cue_ball = cue_ball
        self.my_balls = my_balls
        self.enemy_balls = enemy_balls
        self.black_ball = black_ball
        self.off_table_balls = off_table_balls

def initialise_table(balls_pos: List[tuple[float, float]]) -> Table:
    corners = [Point(0, 0),
               Point(0, TABLE_HEIGHT),
               Point(TABLE_WIDTH, TABLE_HEIGHT),
               Point(TABLE_WIDTH, 0)]

    pockets = [Pocket(Point(0, 0)),
               Pocket(Point(TABLE_WIDTH / 2, 0)),
               Pocket(Point(TABLE_WIDTH, 0)),
               Pocket(Point(0, TABLE_HEIGHT)),
               Pocket(Point(TABLE_WIDTH / 2, TABLE_HEIGHT)),
               Pocket(Point(TABLE_WIDTH, TABLE_HEIGHT))]

    cue_ball = Ball(Point(balls_pos[0][0], balls_pos[0][1]))
    my_balls = []
    off_the_table = 0

    for i in range(1, 8):
        if balls_pos[i] == (0, 0):
            off_the_table += 1
        else:
            my_balls.append(Ball(Point(balls_pos[i][0], balls_pos[i][1])))

    black_ball = Ball(Point(balls_pos[8][0], balls_pos[8][1]))

    enemy_balls = []
    for i in range(9, 16):
        if balls_pos[i] == (0, 0):
            off_the_table += 1
        else:
            enemy_balls.append(Ball(Point(balls_pos[i][0], balls_pos[i][1])))

    return Table(corners, pockets, cue_ball, my_balls, enemy_balls, black_ball, off_the_table)

# ============================================================================
# Choose ball algorithm
# ============================================================================

def find_forbidden_areas(table : Table) -> List[Union[Circle, Line]]:
    forbidden_areas = []
    for ball in table.my_balls + table.enemy_balls + [table.black_ball]:
        forbidden_areas.append(Circle(ball.pos, BALL_RADIUS + SAFETY_DISTANCE))

    forbidden_areas.append(Line(table.corners[0], table.corners[1]))
    forbidden_areas.append(Line(table.corners[1], table.corners[2]))
    forbidden_areas.append(Line(table.corners[2], table.corners[3]))
    forbidden_areas.append(Line(table.corners[3], table.corners[0]))

    return forbidden_areas

def calc_difficulty(d1: float, d2: float, theta : float) -> float:
    if theta >= 90:
        return math.inf
    return d1 * d2 /  (2 * math.cos(math.radians(theta)))

def calc_distance(point1: Point, point2: Point) -> float:
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calc_theta(cue_pos: Point, ball_pos : Point, pocket_pos : Point) -> float:
    ab = calc_distance(cue_pos, ball_pos)
    bc = calc_distance(ball_pos, pocket_pos)
    ac = calc_distance(cue_pos, pocket_pos)
    theta = 180 - math.degrees(math.acos((ab**2 + bc**2 - ac**2) / (2*ab*bc)))
    return theta

def find_intersections_line_with_circle(center : Point, r : float, m : float, b : float) -> List[Point]:
    A = 1 + m**2
    B = 2 * (m * (b - center.y) - center.x)
    C = center.x**2 + (b - center.y)**2 - r**2

    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        return []
    elif discriminant == 0:
        x = -B / (2 * A)
        y = m * x + b
        return [Point(x, y)]
    else:
        sqrt_discriminant = math.sqrt(discriminant)
        x1 = (-B + sqrt_discriminant) / (2 * A)
        x2 = (-B - sqrt_discriminant) / (2 * A)
        y1 = m * x1 + b
        y2 = m * x2 + b
        return [Point(x1, y1), Point(x2, y2)]

def find_intersections_vertical_line_circle(center : Point, radius : float, line_x : float) -> List[Point]:
    dx = line_x - center.x
    discriminant = radius**2 - dx**2
    if discriminant < 0:
        return []

    if math.isclose(discriminant, 0, abs_tol=1e-9):
        return [Point(line_x, center.y)]

    dy = math.sqrt(discriminant)
    y1 = center.y + dy
    y2 = center.y - dy

    return [Point(line_x, y1), Point(line_x, y2)]

def find_intersections_two_lines(point1 : Point, point2 : Point, point3 : Point, point4  :Point) -> List[Point]:
    a1 = point2.y - point1.y
    b1 = point1.x - point2.x
    c1 = a1 * point1.x + b1 * point1.y

    a2 = point4.y - point3.y
    b2 = point3.x - point4.x
    c2 = a2 * point3.x + b2 * point3.y

    determinant = a1*b2 - a2*b1

    if determinant == 0:
        return []
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return [Point(x, y)]

def is_between_two_points(point : Point, p1 : Point, p2 : Point) -> bool:
    if p2.x >= point.x >= p1.x or p1.x >= point.x >= p2.x:
        if p2.y >= point.y >= p1.y or p1.y >= point.y >= p2.y:
            return True
    return False

def is_free_way(point1 : Point, point2 : Point, forbidden_areas : List[Union[Circle, Line]]) -> bool:
    if (point1.x - point2.x) == 0:
        for area in forbidden_areas:
            if isinstance(area, Circle):
                if area.center == point2 or area.center == point1:  # if the forbidden area is the ball itself skip
                    continue

                intersections = find_intersections_vertical_line_circle(area.center, area.radius, point1.x)
                if len(intersections) == 1:
                    if is_between_two_points(intersections[0], point1, point2):
                        return False

                elif len(intersections) == 2:
                    intersection_point_1 = intersections[0]
                    intersection_point_2 = intersections[1]
                    if is_between_two_points(intersection_point_1, point1, point2):
                        return False
                    if is_between_two_points(intersection_point_2, point1, point2):
                        return False
        return True

    m = (point1.y - point2.y) / (point1.x - point2.x)
    b = point1.y - m * point1.x

    for area in forbidden_areas:
        if isinstance(area, Circle):
            if area.center == point2 or area.center == point1: # if the forbidden area is the ball itself skip
                continue

            intersections = find_intersections_line_with_circle(area.center, area.radius, m, b)
            if len(intersections) == 1:
                if is_between_two_points(intersections[0], point1, point2):
                    return False

            elif len(intersections) == 2:
                intersection_point_1 = intersections[0]
                intersection_point_2 = intersections[1]
                if is_between_two_points(intersection_point_1, point1, point2):
                    return False
                if is_between_two_points(intersection_point_2, point1, point2):
                    return False
    return True

def choose_ball_and_pocket_to_heat(table : Table) -> List[Union[Ball, Pocket]]:
    chosen_ball = ''
    chosen_pocket  = ''
    best_difficulty = math.inf
    cue_ball_pos = table.cue_ball.pos

    forbidden_areas_list = find_forbidden_areas(table)

    for ball in table.my_balls:
        ball_pos = ball.pos
        if is_free_way(cue_ball_pos, ball_pos, forbidden_areas_list):
            d1 = calc_distance(cue_ball_pos, ball_pos)
            for pocket in table.pockets:
                if is_free_way(ball_pos, pocket.pos, forbidden_areas_list):
                    d2 = calc_distance(ball_pos, pocket.pos)
                    theta = calc_theta(cue_ball_pos, ball_pos, pocket.pos)
                    diff = abs(calc_difficulty(d1, d2, theta))
                    if ball.pos.x > TABLE_WIDTH / 2:
                        print(f"BALL {ball} \n POCKET {pocket} \n D1 {d1} \n D2 {d2} \n theta {theta} \n diff {diff}")
                        print("------------------------------------------------")
                    if diff < best_difficulty:
                        best_difficulty = diff
                        chosen_ball = ball
                        chosen_pocket = pocket

    if chosen_ball == '':
        return []
    return [chosen_ball, chosen_pocket]

# ============================================================================
# Main
# ============================================================================

def main():
    print(input_positions)
    table = initialise_table(input_positions)
    chosen = choose_ball_and_pocket_to_heat(table)
    if chosen:
        print(f"chosen ball pos: {chosen[0]}")
        print(f"chosen pocket pos: {chosen[1]}")
        return [input_positions, [(chosen[0].pos.x,chosen[0].pos.y) , (chosen[1].pos.x, chosen[1].pos.y)]]
    else:
        return [input_positions, []]

# ============================================================================
# Run
# ============================================================================
if __name__ == "__main__":
    main()
