"""
Pool Simulation — A 2D physics-based pool simulation to plan the robot route and heat on each round.

Uses Pygame for graphics rendering and Pymunk as a 2D rigid-body physics engine.

Libraries:
    - pygame: Handles graphics rendering, event processing, and the game loop.
    - pymunk: Simulates 2D physics (collisions, friction, and elasticity).
"""

import math
import sys
import pygame



# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# --- Window & Interface ---
TITLE = "Pool Game"
SCREEN_WIDTH = 1124 #change to cm
SCREEN_HEIGHT = 608
TABLE_WIDTH = 281 # cm
TABLE_HEIGHT = 152 # cm
BORDER_WIDTH = 13 # cm
BOTTOM_PANEL = 50
BACKGROUND_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)

# --- Ball Parameters ---
BALL_COUNT = 16  # Total number of balls (15 colored balls + 1 cue ball)
BALL_MASS = 5  # Mass of each ball (Pymunk units)
BALL_ELASTICITY = 0.8  # Coefficient of restitution (elasticity) during collisions
BALL_DIAMETER = 6  # Ball diameter (cm)
BALL_RADIUS = BALL_DIAMETER / 2  # Ball radius (cm)
SAFETY_DISTANCE = 3



# --- Off-screen position (for balls that have fallen into pockets) ---
OFFSCREEN_POS = (620, 620)

# --- Bottom panel: display for potted balls ---
POTTED_BALL_X_START = 10  # Starting X-coordinate for drawing potted balls
POTTED_BALL_SPACING = 50  # Spacing between potted balls
POTTED_BALL_Y_OFFSET = 10  # Vertical distance from the panel boundary line

# --- Robot size and position for 
ROBOT_WIDTH = 24 # cm
ROBOT_LENGTH = 20 # cm
ROBOT_COLOR = (100, 50, 40)

# --- Frame rate ---
FPS = 120

# ============================================================================
# Input 
# ============================================================================

# --- Ball position ---
input_positions = [(100, 50), (20, 20), (30, 30), (110, 110), 
                  (40, 40), (50, 50), (60, 60), (70, 70), 
                  (80, 80), (90, 90), (100, 100), (10, 20), 
                  (10, 30), (10, 40), (10, 50), (10, 60)]
# If a ball is off the table his position is (0, 0)


# ============================================================================
# Utilites 
# ============================================================================

class Point():
    def __init__ (self, x, y):
        self.x = x
        self.y = y

class Ball():
    def __init__(self, pos):
        self.pos = pos

class Circle():
    def __init__(self, center, radius):
        self.center = center
        self.radiuse = radius

class Table():
    def __init__(self, corners, pockets, cue_ball, my_balls, enemy_balls, black_ball, off_table_balls):
        self.corners = corners
        self.pockets = pockets
        self.cue_ball = cue_ball
        self.my_balls = my_balls
        self.enemy_balls = enemy_balls
        self.black_ball = black_ball
        self.off_table_balls = off_table_balls

class Pocket():
    def __init__(self, pos):
        self.pos = pos

def initiallies_table(balls_pos):
    corners = [Point(0, 0),
                Point(0, TABLE_HEIGHT),
                Point(TABLE_WIDTH, TABLE_HEIGHT),
                Point(TABLE_WIDTH, 0)]
    
    pockets = [Pocket(Point(0, 0)), Pocket(Point(TABLE_WIDTH / 2, 0)), Pocket(Point(TABLE_WIDTH, 0)), 
                Pocket(Point(0, TABLE_HEIGHT)), Pocket(Point(TABLE_WIDTH / 2, TABLE_HEIGHT)), Pocket(Point(TABLE_WIDTH, TABLE_HEIGHT))]

    cue_ball = Ball(Point(balls_pos[0][0], balls_pos[0][1]))
    my_balls = []
    off_the_table = 0

    for i in range(1, 8):
        if balls_pos[i] == (0,0):
            off_the_table += 1
        else:
            my_balls.append(Ball(Point(balls_pos[i][0], balls_pos[i][1])))

    black_ball = Ball(Point(balls_pos[8][0], balls_pos[8][1]))

    enemy_balls = []
    for i in range(9, 16):
        if balls_pos[i] == (0,0):
            off_the_table += 1
        else:
            enemy_balls.append(Ball(Point(balls_pos[i][0], balls_pos[i][1])))

    return Table(corners, pockets, cue_ball, my_balls, enemy_balls, black_ball, off_the_table)


# ============================================================================
# Choose ball algoritem & plan route
# ============================================================================

def pixel_to_cm(pixel_value):
    return (pixel_value / 4)  - 13

def cm_to_pixel(cm_value):
    return (cm_value) * 4

def cm_to_pixel_pos(cm_value):
    return (cm_value + 13) * 4

def forbiten_areas(balls):
    forbiten_areas = [] 
    for i, ball in enumerate(balls):
        forbiten_areas.append(Circle(ball, BALL_RADIUS + SAFETY_DISTANCE))
    return forbiten_areas 

def clac_difficulty(d1, d2, teta):
    return (d1 * d2 / math.cos(teta))

def clac_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def calc_teta(point1, point2, point3):
    m1 = (point1.y - point2.y) / (point1.x - point2.x)
    m2 = (point2.y - point3.y) / (point2.x - point3.x)
    return math.degrees(math.atan(abs((m1 - m2) / (1 + m1*m2))))
 
def find_intersections_line_with_circle(center, r, m, b):
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

def find_intersections_vertical_line_circle(center, radius, line_x):
    dx = line_x - center.x
    discriminant = radius**2 - dx**2
    if discriminant < 0:
        return []

    if math.isclose(discriminant, 0, abs_tol=1e-9):
        return [(line_x, center.y)]

    dy = math.sqrt(discriminant)
    y1 = center.y + dy
    y2 = center.y - dy

    return [(line_x, y1), (line_x, y2)]

def find_intersections_two_lines(point1, point2, point3, point4):
    a1 = point2.y - point1.y
    b1 = point1.x - point2.x
    c1 = a1*(point1.x) + b1*(point1.y)

    a2 = point4.y - point3.y
    b2 = point3.x - point4.x
    c2 = a2*(point3.x) + b2*(point3.y)

    determinant = a1*b2 - a2*b1

    if (determinant == 0):
        return []
    else:
        x = (b2*c1 - b1*c2)/determinant
        y = (a1*c2 - a2*c1)/determinant
        return [Point(x, y)]

def is_between_two_points(point, p1, p2):
    if p2.x >= point.x >= p1.x or p1.x >= point.x >= p2.x:
        if p2.y >= point.y >= p1.y or p1.y >= point.y >= p2.y:
            return True
    return False

def is_free_way(point1, point2, forbiten_areas):
    if (point1.x - point2.x) == 0:
        
        for area in forbiten_areas:
            intersections = find_intersections_vertical_line_circle(area.center, area.radiuse, point1.x)
            if area.center == point2 or area.center == point1:
                continue
            if len(intersections) == 1:
                if(is_between_two_points(intersections[0], point1, point2)):
                    return False

            elif len(intersections) == 2:
                intersection_point_1 = intersections[0]
                intersection_point_2 = intersections[1]
                if(is_between_two_points(intersection_point_1, point1, point2)):
                        return False
                if(is_between_two_points(intersection_point_2, point1, point2)):
                        return False
        
        return True

    m = (point1.y - point2.y) / (point1.x - point2.x)
    b = point1.y - m * point1.x

    for area in forbiten_areas:
        if area.center == point2 or area.center == point1:
            continue
        intersections = find_intersections_line_with_circle(area.center, area.radiuse, m, b)
        if len(intersections) == 1:
            if(is_between_two_points(intersections[0], point1, point2)):
                return False

        elif len(intersections) == 2:
            intersection_point_1 = intersections[0]
            intersection_point_2 = intersections[1]
            if(is_between_two_points(intersection_point_1, point1, point2)):
                    return False
            if(is_between_two_points(intersection_point_2, point1, point2)):
                    return False

    return True

def choose_ball_and_pocket_to_heat(table):
    choosen_ball = ''
    best_pocket  = ''
    best_difficulty = math.inf
    cue_ball_pos = table.cue_ball.pos

    forbiten_areas_list = forbiten_areas([ball.pos for ball in table.my_balls] + [ball.pos for ball in table.enemy_balls] + [table.black_ball.pos])

    for ball in table.my_balls:
        ball_pos = ball.pos
        if(is_free_way(cue_ball_pos, ball_pos, forbiten_areas_list)):
            best_pocket_for_ball  = ''
            d2 = math.inf
            for pocket in table.pockets:
                if((is_free_way(ball_pos, pocket.pos, forbiten_areas_list))):
                    d = clac_distance(ball_pos, pocket.pos)
                    if (d < d2):
                        d2 = d
                        best_pocket_for_ball = pocket
            if not best_pocket_for_ball:
                continue
            d1 = clac_distance(cue_ball_pos, ball_pos)
            teta = calc_teta(cue_ball_pos, ball_pos, pocket.pos)
            print(d1, d2, teta)
            print(ball.pos.x, ball.pos.y)
            print(pocket.pos.x, pocket.pos.y)
            print('-----')
            diff = clac_difficulty(d1, d2, teta)
            if diff < best_difficulty:
                print('--chosen---')
                best_difficulty = diff
                choosen_ball = ball
                best_pocket = pocket

    if choosen_ball == '':
        return []
    return [choosen_ball, best_pocket]


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main() -> None:
    """Main function — initializes the game and runs the main loop.

    Execution flow:
        1. Initialize Pygame and Pymunk; load assets.
        2. Create the balls and table cushions.
        3. Choose ball to heat
        4. Plan route and draw it.
        5. Exit when the player closes the window.
    """
    # --- Initialize Pygame ---
    pygame.init()
    clock = pygame.time.Clock()

    # --- Create the game window ---
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + BOTTOM_PANEL))
    pygame.display.set_caption(TITLE)

    # --- Load images ---
    table_image = pygame.image.load("assets/images/table.png").convert_alpha()
    table_image = pygame.transform.scale(table_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    ball_images = []
    for i in range(1, BALL_COUNT + 1):
        ball_image = pygame.image.load(f"assets/images/ball_{i}.png").convert_alpha()
        ball_image = pygame.transform.scale(ball_image, (cm_to_pixel(BALL_DIAMETER), cm_to_pixel(BALL_DIAMETER)))
        ball_images.append(ball_image)


    robot_image = pygame.image.load("assets/images/robot.png").convert_alpha()
    robot_image = pygame.transform.scale(robot_image, (cm_to_pixel(ROBOT_WIDTH), cm_to_pixel(ROBOT_LENGTH)))

    table = initiallies_table(input_positions)

    forbiten_areas_list = forbiten_areas([ball.pos for ball in table.my_balls] + [ball.pos for ball in table.enemy_balls] + [table.black_ball.pos])


    # ========================================================================
    # MAIN LOOP
    # ========================================================================
    game_on = True

    while game_on:
        clock.tick(FPS)

        # --- Draw the background and the pool table ---
        screen.fill(BACKGROUND_COLOR)
        screen.blit(table_image, (0, 0))
        # --- Draw the border of the table without the border ---
        pygame.draw.line(screen, (0, 0, 0), (cm_to_pixel(BORDER_WIDTH), cm_to_pixel(BORDER_WIDTH)), (cm_to_pixel(BORDER_WIDTH), SCREEN_HEIGHT - cm_to_pixel(BORDER_WIDTH)))
        pygame.draw.line(screen, (0, 0, 0), (cm_to_pixel(BORDER_WIDTH), SCREEN_HEIGHT - cm_to_pixel(BORDER_WIDTH)), (SCREEN_WIDTH - cm_to_pixel(BORDER_WIDTH), SCREEN_HEIGHT - cm_to_pixel(BORDER_WIDTH)))
        pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH - cm_to_pixel(BORDER_WIDTH), SCREEN_HEIGHT - cm_to_pixel(BORDER_WIDTH)), (SCREEN_WIDTH - cm_to_pixel(BORDER_WIDTH), cm_to_pixel(BORDER_WIDTH)))
        pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH -  cm_to_pixel(BORDER_WIDTH), cm_to_pixel(BORDER_WIDTH)), (cm_to_pixel(BORDER_WIDTH), cm_to_pixel(BORDER_WIDTH)))

        for area in forbiten_areas_list:
            pygame.draw.circle(screen, (255, 0, 0), [cm_to_pixel_pos(area.center.x), cm_to_pixel_pos(area.center.y)], cm_to_pixel(area.radiuse), 2)

        # --- Draw the bottom panel ---
        pygame.draw.rect(
            screen, BACKGROUND_COLOR, (0, SCREEN_HEIGHT, SCREEN_WIDTH - cm_to_pixel(BORDER_WIDTH), BOTTOM_PANEL)
        )

        for i, ball in enumerate(table.my_balls):
            screen.blit(
                    ball_images[0],
                    (cm_to_pixel_pos(ball.pos.x -  BALL_RADIUS), cm_to_pixel_pos(ball.pos.y - BALL_RADIUS)),
                )

        for i, ball in enumerate(table.enemy_balls):
            screen.blit(
                    ball_images[9],
                    (cm_to_pixel_pos(ball.pos.x -  BALL_RADIUS), cm_to_pixel_pos(ball.pos.y - BALL_RADIUS)),
                )

        screen.blit(
                    ball_images[7],
                    (cm_to_pixel_pos(table.black_ball.pos.x -  BALL_RADIUS), cm_to_pixel_pos(table.black_ball.pos.y - BALL_RADIUS)),
                )

        screen.blit(
                    ball_images[15],
                    (cm_to_pixel_pos(table.cue_ball.pos.x -  BALL_RADIUS), cm_to_pixel_pos(table.cue_ball.pos.y - BALL_RADIUS)),
                )

        screen.blit(robot_image, ((SCREEN_WIDTH - cm_to_pixel(BORDER_WIDTH) - cm_to_pixel(ROBOT_WIDTH)), cm_to_pixel(BORDER_WIDTH)))


        # --- Handle events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_on = False

        pygame.display.update()

    pygame.quit()
    sys.exit()


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    main()

