# global configures
# never changed once they're set
import numpy as np
import math



class obstacle(object):
    def __init__(self,center,r,V,b):
        self.center = np.array(center)
        self.r = r
        self.V = V
        self.b = b
class destination(object):
    def __init__(self,center,vertices,dest_range,prob=0.05,exit=False):
        self.center = np.array(center)
        self.vertices = np.array(vertices)
        self.dest_range = np.array(dest_range)
        self.exit = exit
        self.prob = prob
class destination_rect(object):
    """
    range_object = (left,right,up,low)
    range_dest = (left,right,up,low)
    vertices = [lower left, upper left, upper right, lower right]
    """
    def __init__(self,center,range_object,range_dest,prob=0.03,exit=False):
        self._ro = range_object
        self._rd = range_dest
        self.prob = prob
        self.exit = exit
        self.center = np.array(center)
        self.vertices = np.array([[self.center[0]-self._ro[0],self.center[1]-self._ro[3]],
                                  [self.center[0]-self._ro[0],self.center[1]+self._ro[2]],
                                  [self.center[0]+self._ro[1],self.center[1]+self._ro[2]],
                                  [self.center[0]+self._ro[1],self.center[1]-self._ro[3]]])
        self.dest_range = np.array([[self.center[0] - self._rd[0], self.center[1] - self._rd[3]],
                                    [self.center[0] - self._rd[0], self.center[1] + self._rd[2]],
                                    [self.center[0] + self._rd[1], self.center[1] + self._rd[2]],
                                    [self.center[0] + self._rd[1], self.center[1] - self._rd[3]]])

class contour(object):
    def __init__(self,vertices,invalid_side=[]):
        """
        :param vertices : should be clockwise and convex
                np.array(((x1,y1),(x2,y2),(x3,y3),(x4,y4)))
        :param invalid_side : doesn't affect pedestrians(list)
                np.array(((x1,y1),(x2,y2)))
        """
        self.vertices = np.array(vertices)
        self.invalid_side = invalid_side

gc = {
    "num_peds" : 50,
    "bucket_scale" : 10,
    "iteration" : 800,
    "walls" : [],
    "obstacles":[],
    "destinations":[],
    "entrance":destination_rect([0,13],[0,1,2,7],[0,1,2,7])
}

# contour
gc["walls"].append(contour(((0,0),(0,15),(12,7.5),(3,0)),
                           [((0,15),(12,7.5))]))
gc["walls"].append(contour(((0,15),(15,15),(15,7.5),(12,7.5)),
                           [((12,7.5),(0,15)),((15,15),(15,7.5))]))
gc["walls"].append(contour(((15,7.5),(15,15),(30,15),(30,7.5)),
                           [((15,7.5),(15,15)),((30,7.5),(15,7.5))]))
gc["walls"].append(contour(((15,0),(15,7.5),(30,7.5),(30,0)),
                          [((15,7.5),(30,7.5))]))
# geometry
gc["destinations"].append(destination_rect((6.5,15), [1.5,1.5,0,1], [2,2,0,2]))
gc["destinations"].append(destination_rect((13,12), [0.5,0.5,0.5,0.5], [1,1,1,1]))
gc["destinations"].append(destination_rect((22.5,15),[4.5,4.5,0,1],[5.5,5.5,0,2]))
gc["destinations"].append(destination_rect((30,7.5),[1,0,2.5,2.5],[2,0,3.5,3.5]))
gc["destinations"].append(destination_rect((27,0),[1.5,1.5,1,0],[2,2,1.5,0]))
gc["destinations"].append(destination_rect((21,0),[1.5,1.5,1,0],[2,2,1.5,0]))
gc["destinations"].append(destination_rect((22.5,7.5),[3.5,3.5,1,1],[4.5,4.5,2,2]))
gc["destinations"].append(destination_rect((5,10), [0.5,0.5,0.5,0.5], [1,1,1,1]))
gc["destinations"].append(destination((7.5,3.8),
                                      ((3,0),(2,1.2),(11,8.7),(12,7.5)),
                                      ((3,0),(2,1.2),(11,8.7),(12,7.5))))
gc["destinations"].append(destination_rect((0,8),(0,1,7,2),(0,1,7,2),exit=True))
# obstacles
V = 1
b = 0.5
gc["obstacles"].append(obstacle(np.array((13,12)),0.5,V,b))
gc["obstacles"].append(obstacle(np.array((6.5,15)),1,V,b))
gc["obstacles"].append(obstacle(np.array((20,15)),1,V,b))
gc["obstacles"].append(obstacle(np.array((25,15)),1,V,b))
gc["obstacles"].append(obstacle(np.array((30,6)),1,V,b))
gc["obstacles"].append(obstacle(np.array((30,9)),1,V,b))
gc["obstacles"].append(obstacle(np.array((27,0)),1,V,b))
gc["obstacles"].append(obstacle(np.array((21,0)),1,V,b))
gc["obstacles"].append(obstacle(np.array((20,7.5)),1,V,b))
gc["obstacles"].append(obstacle(np.array((21.25,7.5)),1,V,b))
gc["obstacles"].append(obstacle(np.array((22.5,7.5)),1,V,b))
gc["obstacles"].append(obstacle(np.array((23.75,7.5)),1,V,b))
gc["obstacles"].append(obstacle(np.array((25,7.5)),1,V,b))
gc["obstacles"].append(obstacle(np.array((5,10)),0.5,V,b))
gc["obstacles"].append(obstacle(np.array((5.4,2)),1,V,b))
gc["obstacles"].append(obstacle(np.array((7.5,3.75)),1,V,b))
gc["obstacles"].append(obstacle(np.array((9.6,5.5)),1,V,b))


gc["max_xy"] = np.max(gc["walls"][0].vertices, axis=0)
gc["min_xy"] = np.min(gc["walls"][0].vertices, axis=0)
for wall in gc["walls"]:
    gc["max_xy"] = np.max((gc["max_xy"],np.max(wall.vertices,axis=0)),axis=0)
    gc["min_xy"] = np.min((gc["min_xy"],np.min(wall.vertices,axis=0)),axis=0)
gc["max_entr_xy"] = np.max(gc["entrance"].vertices,axis=0)
gc["min_entr_xy"] = np.min(gc["entrance"].vertices,axis=0)
gc["width"] = gc["max_xy"][0]-gc["min_xy"][0]
gc["height"] = gc["max_xy"][1]-gc["min_xy"][1]
gc["xmesh"] = int(math.ceil(gc["width"]*1.0/gc["bucket_scale"]))
gc["ymesh"] = int(math.ceil(gc["height"]*1.0/gc["bucket_scale"]))


# Plot test
if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    walls = gc["walls"]
    obstacles = gc["obstacles"]
    destinations = gc["destinations"]
    fig = plt.figure(figsize=matplotlib.figure.figaspect(1))
    plt.gca().set_aspect('equal', adjustable='box')
    ax = fig.add_subplot(111)
    # contour
    for wall in walls:
        wall = plt.Polygon(wall.vertices, color="red", alpha=0.2)
        ax.add_patch(wall)
    # obstacles
    for obstacle in obstacles:
        obst = plt.Circle(obstacle.center, obstacle.r, color="green", alpha=0.5)
        ax.add_patch(obst)
    # destinations
    for dest in destinations:
        dest_range = plt.Polygon(dest.dest_range, color="black", alpha=0.5)
        dest = plt.Polygon(dest.vertices, color="black", alpha=0.2)
        ax.add_patch(dest_range)
        ax.add_patch(dest)

    plt.xlim(gc["min_xy"][0], gc["max_xy"][0])
    plt.ylim(gc["min_xy"][1], gc["max_xy"][1])
    plt.show()
