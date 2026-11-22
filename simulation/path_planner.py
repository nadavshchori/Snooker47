from ball_selector import *

from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra


class Point:
    def __init__ (self, x: float, y: float):
        self.x = x
        self.y = y

def two_point_dist_squared(p1:Point, p2:Point):
    return (p1.x - p2.x)**2 + (p1.y - p2.y)**2


class Node():
     def __init__(self, point: Point, id: int):
          self.point = point
          self.id = id

class Edge():
     def __init__(self, v1: Node, v2: Node, w: float):
          self.v1 = v1
          self.v2 = v2
          self.w = w

class Graph():
    def __init__(self):
        self.matric = []
        self.nodes = []
        self.edges = []
        self.adj_list = []
        self.curr_id = 0
    
    def add_edge(self, edge: Edge):
        self.edges.append(edge)

        self.matric[edge.v1.id][edge.v2.id] = edge.w
        self.matric[edge.v2.id][edge.v1.id] = edge.w

    def add_node(self, point: Point):
        self.matric.append([0] * (len(self.matric))) 
        for i in range(len(self.matric)):
            self.matric[i].append(0)
        curr_node = Node(point, self.curr_id)
        self.nodes.append(curr_node)
        self.curr_id += 1
        return curr_node
        
    def get_node_from_id(self, id: int):
         return self.nodes[id]
        
    def get_id_from_point(self, point: Point):
        i = 0 
        for node in self.nodes:
            if node.point.x == point.x and node.point.x == point.y:
                return i
            i += 1

    def dijkstra(self):
        graph = csr_array(self.matric)
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)
        return predecessors

'''
p1 = Point(1, 1)
   
p2 = Point(2, 5)

p3 = Point(18, 10)

p4 = Point(10, 2)

p5 = Point(7, 7)

G = Graph()

G.add_node(p1)
print(G.matric)
G.add_node(p2)
print(G.matric)
G.add_node(p3)
print(G.matric)
G.add_node(p4)
print(G.matric)
G.add_node(p5)
print(G.matric)

graph = csr_array(G.matric)
dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)
print(predecessors)
'''


'''
HEIGHT_DELTA = 4 #cm
        WIDTH_DELTA = 4 #cm
        ROBOT_RADIUS = max(ROBOT_LENGTH, ROBOT_WIDTH)//2 #cm
        
        initial = Point(self.robot.get_pos()[0], self.robot.get_pos()[1])
        end = self.cue_ball.get_pos()
       
        
        forbidden_areas = find_forbidden_areas(self, ROBOT_RADIUS + BALL_DIAMETER/2)
        
        self.adj = [[], []]
        self.vertex_list = [initial, end]
        if self.is_edge_valid(initial, end, forbidden_areas, ROBOT_RADIUS + BALL_DIAMETER/2):
            self.adj[0].append([end, two_point_dist(initial, end), 1])
            self.adj[1].append([initial, two_point_dist(initial, end), 0])
        index = 2
        for x in range(ROBOT_RADIUS, TABLE_WIDTH - ROBOT_RADIUS, WIDTH_DELTA):
            for y in range(ROBOT_RADIUS, TABLE_HEIGHT - ROBOT_RADIUS, HEIGHT_DELTA):
                #check for collision with balls
                self.adj.append([])
                vertex_index = 0
                for v in self.vertex_list:
                    if self.is_edge_valid(v, Point(x, y), forbidden_areas, ROBOT_RADIUS + BALL_DIAMETER/2):
                        
                        self.adj[vertex_index].append([Point(x, y), two_point_dist(Point(x, y), v), index])
                        self.adj[index].append([v, two_point_dist(Point(x, y), v), vertex_index])
                    vertex_index += 1
                
                self.vertex_list.append(Point(x, y))
                index += 1
        
        
        dist, predecessors = dijkstraF(self.adj, 0)

        path = retrace(predecessors)
        point_path = []
        for index in path:
            point_path.append(self.vertex_list[index])

        coor_path = []
        for point in point_path:
            coor_path.append((point.x, point.y))

        print(coor_path)
        
        return coor_path

'''