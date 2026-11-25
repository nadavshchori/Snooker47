from simulation.find_path import *
from scipy.sparse import csr_array
from scipy.sparse.csgraph import dijkstra


class Point:
    def __init__ (self, x: float, y: float):
        self.x = x
        self.y = y

def two_point_dist_squared(p1:Point, p2:Point):
    return (p1.x - p2.x)**2 + (p1.y - p2.y)**2


class Node:
     def __init__(self, point: Point, id: int):
          self.point = point
          self.id = id

class Edge:
     def __init__(self, v1: Node, v2: Node, w: float):
          self.v1 = v1
          self.v2 = v2
          self.w = w

class Graph:
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
        return None

    def dijkstra(self):
        graph = csr_array(self.matric)
        dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, indices=0, return_predecessors=True)
        return predecessors
