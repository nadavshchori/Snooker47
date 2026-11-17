import math
import sys
import pygame
import random
import ball_selector

def pixel_to_cm(pixel_value):
    return pixel_value / 4

def cm_to_pixel(cm_value):
    return cm_value * 4

# --- Window & Interface ---
TITLE = "Ball Selector"
SCREEN_WIDTH =  cm_to_pixel(ball_selector.TABLE_WIDTH)
SCREEN_HEIGHT = cm_to_pixel(ball_selector.TABLE_HEIGHT)
BALL_DIAMETER = cm_to_pixel(ball_selector.BALL_DIAMETER)
BALL_RADIUS = BALL_DIAMETER / 2

BACKGROUND_COLOR = (50, 50, 50)

ROBOT_WIDTH = cm_to_pixel(ball_selector.ROBOT_WIDTH)
ROBOT_LENGTH = cm_to_pixel(ball_selector.ROBOT_LENGTH)

BALL_COUNT = 16

# ============================================================================
# Input
# ============================================================================


def main(input_positions, chosen, path):
    vertix, adj = path

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(TITLE)

    table_image = pygame.image.load("simulation/assets/images/table.png").convert_alpha()
    table_image = pygame.transform.scale(table_image, (SCREEN_WIDTH, SCREEN_HEIGHT))

    ball_images = []
    for i in range(1, BALL_COUNT + 1):
        ball_image = pygame.image.load(f"simulation/assets/images/ball_{i}.png").convert_alpha()
        ball_image = pygame.transform.scale(ball_image, (BALL_DIAMETER, BALL_DIAMETER))
        ball_images.append(ball_image)

    robot_image = pygame.image.load("simulation/assets/images/robot.png").convert_alpha()
    robot_image = pygame.transform.scale(robot_image, (ROBOT_WIDTH, ROBOT_LENGTH))

    game_on = True

    while game_on:
        screen.fill(BACKGROUND_COLOR)
        screen.blit(table_image, (0, 0))

        for i, ball in enumerate(input_positions[1:8]):
            screen.blit(
                ball_images[0],
                (cm_to_pixel(ball[0]) - BALL_RADIUS, cm_to_pixel(ball[1]) - BALL_RADIUS),
            )

        for i, ball in enumerate(input_positions[9:16]):
            screen.blit(
                ball_images[9],
                (cm_to_pixel(ball[0]) - BALL_RADIUS, cm_to_pixel(ball[1]) - BALL_RADIUS),
            )

        screen.blit(
            ball_images[7],
            (cm_to_pixel(input_positions[8][0]) - BALL_RADIUS,
             cm_to_pixel(input_positions[8][1]) - BALL_RADIUS),
        )

        screen.blit(
            ball_images[15],
            (cm_to_pixel(input_positions[0][0]) - BALL_RADIUS,
             cm_to_pixel(input_positions[0][1]) - BALL_RADIUS),
        )

        screen.blit(robot_image,(SCREEN_WIDTH- ROBOT_WIDTH, 0))

        if chosen == []:
            pygame.draw.line(screen, (255, 0, 0), (0,0), (SCREEN_WIDTH, SCREEN_HEIGHT))
        else:
            chosen_ball = chosen[0]
            chosen_pocket = chosen[1]
            pygame.draw.line(screen, (255, 0, 0),
                             (cm_to_pixel(input_positions[0][0]), cm_to_pixel(input_positions[0][1])),
                             (cm_to_pixel(chosen_ball[0]), cm_to_pixel(chosen_ball[1])))
            pygame.draw.line(screen, (255, 0, 0),
                             (cm_to_pixel(chosen_ball[0]), cm_to_pixel(chosen_ball[1])),
                              (cm_to_pixel(chosen_pocket[0]), cm_to_pixel(chosen_pocket[1])))

        print(adj)
        print(len(vertix))
        for i,v in enumerate(vertix):
            for u in adj[i]:
                    pygame.draw.line(screen, (0, 0, 255),
                                                (cm_to_pixel(v.x), cm_to_pixel(v.y)),
                                                (cm_to_pixel(u[0].x), cm_to_pixel(u[0].y)))

        # --- Handle events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_on = False

        pygame.display.update()

    pygame.quit()
    sys.exit()




if __name__ == '__main__':
        input_positions, chosen, path = ball_selector.main()
        main(input_positions, chosen, path)



