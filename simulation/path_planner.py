from ball_selector import *

input_positions = [
    (random.randint(0, int(TABLE_WIDTH - BALL_RADIUS)), random.randint(0, int(TABLE_HEIGHT - BALL_RADIUS)))
    for _ in range(16)
]

print(input_positions)


def is_edge_legit(v1, v2, threshold):
    dir = np.array([v2[0] - v1[0], v2[1] - v1[1]])
    for ball_pos in input_positions: #changed
        #ball_pos = ball.get_pos()
        AP = np.array([ball_pos[0] - v1[0], ball_pos[1] - v1[1]])

        dist = np.linalg.norm(np.cross(AP, dir)) / np.linalg.norm(dir)
        if dist < threshold:
            return False

    return True


def find_shortest_path():
    HEIGHT_DELTA = 30 #cm
    WIDTH_DELTA = 40 #cm
    ROBOT_RADIUS = 12 #cm
    
    initial = (0, 0)
    end = (200, 100)

    
    adj = [[], []]
    vertex_list = [initial, end]
    for x in range(ROBOT_RADIUS, TABLE_WIDTH - ROBOT_RADIUS, WIDTH_DELTA):
        for y in range(ROBOT_RADIUS, TABLE_HEIGHT - ROBOT_RADIUS, HEIGHT_DELTA):
            #check for collision with balls
            
            vertex_index = 0
            for v in vertex_list:
                if is_edge_legit(v, (x, y), ROBOT_RADIUS):
                    adj[vertex_index].append([(x, y), two_point_dist((x, y), v)])
                vertex_index += 1
            
            vertex_list.append((x, y))
            adj.append([])


    print(adj)

def two_point_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

find_shortest_path()



#table = initialise_table(input_positions)
