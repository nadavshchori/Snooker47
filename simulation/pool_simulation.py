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
SCREEN_WIDTH = 1124
SCREEN_HEIGHT = 608
BOTTOM_PANEL = 50
BACKGROUND_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)

# --- Ball Parameters ---
BALL_COUNT = 16  # Total number of balls (15 colored balls + 1 cue ball)
BALL_MASS = 5  # Mass of each ball (Pymunk units)
BALL_ELASTICITY = 0.8  # Coefficient of restitution (elasticity) during collisions
BALL_DIAMETER = 24  # Ball diameter (pixels)
BALL_RADIUS = BALL_DIAMETER / 2  # Ball radius (pixels)
SAFETY_DISTANCE = 3

BORDER_WIDTH = 52

# --- Off-screen position (for balls that have fallen into pockets) ---
OFFSCREEN_POS = (620, 620)

# --- Bottom panel: display for potted balls ---
POTTED_BALL_X_START = 10  # Starting X-coordinate for drawing potted balls
POTTED_BALL_SPACING = 50  # Spacing between potted balls
POTTED_BALL_Y_OFFSET = 10  # Vertical distance from the panel boundary line

# --- Robot size and position for 
ROBOT_WIDTH = 24 * 4
ROBOT_LENGTH = 20 * 4
ROBOT_COLOR = (100, 50, 40)

# --- Coordinates of the 6 table pockets (x, y) ---
POCKETS = {
    1: (BORDER_WIDTH, BORDER_WIDTH), 2: (SCREEN_WIDTH /  2, BORDER_WIDTH), 3: (SCREEN_WIDTH -  BORDER_WIDTH, BORDER_WIDTH),  # 3 top pockets
    4: (BORDER_WIDTH, SCREEN_HEIGHT - BORDER_WIDTH), 5: (SCREEN_WIDTH /  2, SCREEN_HEIGHT - BORDER_WIDTH), 6: (SCREEN_WIDTH -  BORDER_WIDTH, SCREEN_HEIGHT - BORDER_WIDTH),  # 3 bottom pockets
}

# --- Frame rate ---
FPS = 120


# ============================================================================
# Input 
# ============================================================================

# --- Ball position ---
balls_position = [(550, 400), (150, 340), (110, 110), (120, 480), 
                  (130, 500), (140, 140), (400, 150), (160, 320), 
                  (170, 400), (0, 0), (190, 250), (900, 400), 
                  (800, 400), (700, 300), (500, 250), (600, 400)]
# If a ball is off the table his position is (0, 0)


# ============================================================================
# Utilites 
# ============================================================================

class Circle():
    def __init__(self, center, radius):
        self.center = center
        self.radiuse = radius

# ============================================================================
# Choose ball algoritem & plan route
# ============================================================================

def forbiten_areas(ball_position):
    forbiten_areas = [] 
    for i, ball in enumerate(balls_position):
        forbiten_areas.append(Circle(ball, BALL_RADIUS + SAFETY_DISTANCE))
    return forbiten_areas 

def clac_difficulty(d1, d2, teta):
    return (d1 * d2 / math.cos(teta))

def clac_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_intersections(center_x, center_y, r, m, b):
    A = 1 + m**2
    B = 2 * (m * (b - k) - center_x)
    C = center_x**2 + (b - center_y)**2 - r**2

    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        return []
    elif discriminant == 0:
        x = -B / (2 * A)
        y = m * x + b
        return [(x, y)]
    else:
        sqrt_discriminant = math.sqrt(discriminant)
        x1 = (-B + sqrt_discriminant) / (2 * A)
        x2 = (-B - sqrt_discriminant) / (2 * A)
        y1 = m * x1 + b
        y2 = m * x2 + b
        return [(x1, y1), (x2, y2)]

def is_free_way(point1, point2, forbiten_areas):
    if (point1[0] - point2[0]) == 0:
        pass

    m = (point1[1] - point2[1]) / (point1[0] - point2[0])
    b = point1[1] - m * point1[0]

    for area in forbiten_areas:
        intersections = find_intersections(area.circle[0], area.circle[1], area.radius, m, b)

    return True


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
        ball_image = pygame.transform.scale(ball_image, (BALL_DIAMETER, BALL_DIAMETER))
        ball_images.append(ball_image)


    robot_image = pygame.image.load("assets/images/robot.png").convert_alpha()
    robot_image = pygame.transform.scale(robot_image, (ROBOT_WIDTH, ROBOT_LENGTH))

    # --- Create the balls on the table ---
    balls = []
    # Arrange the 15 colored balls in a triangle (5 columns, decreasing rows)
    for i in range(1, len(balls_position)):
            pos = (
                balls_position[i][0], balls_position[i][1])
            balls.append(pos)

    # Cue ball (always at the end of the balls list)
    cue_ball_pos = (balls_position[0][0], balls_position[0][1])
    balls.append(cue_ball_pos)

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
        pygame.draw.line(screen, (0, 0, 0), (BORDER_WIDTH, BORDER_WIDTH), (BORDER_WIDTH, SCREEN_HEIGHT - BORDER_WIDTH))
        pygame.draw.line(screen, (0, 0, 0), (BORDER_WIDTH, SCREEN_HEIGHT - BORDER_WIDTH), (SCREEN_WIDTH - BORDER_WIDTH, SCREEN_HEIGHT - BORDER_WIDTH))
        pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH - BORDER_WIDTH, SCREEN_HEIGHT - BORDER_WIDTH), (SCREEN_WIDTH - BORDER_WIDTH, BORDER_WIDTH))
        pygame.draw.line(screen, (0, 0, 0), (SCREEN_WIDTH -  BORDER_WIDTH, BORDER_WIDTH), (BORDER_WIDTH, BORDER_WIDTH))

        # --- Draw the bottom panel ---
        pygame.draw.rect(
            screen, BACKGROUND_COLOR, (0, SCREEN_HEIGHT, SCREEN_WIDTH - BORDER_WIDTH, BOTTOM_PANEL)
        )
        
        count_potted_ball = 0
        # --- Draw the balls on the table ---
        for i, ball in enumerate(balls):
            if ball[0] == 0:
                screen.blit(ball_images[i],
                (POTTED_BALL_X_START + count_potted_ball * POTTED_BALL_SPACING, SCREEN_HEIGHT),
                )
                count_potted_ball += 1
            else:
                screen.blit(
                    ball_images[i],
                    (ball[0] -  BALL_RADIUS, ball[1] - BALL_RADIUS),
                )

        screen.blit(robot_image, ((SCREEN_WIDTH - BORDER_WIDTH - ROBOT_WIDTH), BORDER_WIDTH))


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



# TO DO
# להתאים פורפרוציה של כדור למגרש
# לעשות קלט של מצב נתון שממנו מופיע מצב המגרש
# להוסיף ציור של הרובוט ומיקום התחלתי שלו במידות פרופורציונליות
# לבחור מסלול לכדור הלבן 
