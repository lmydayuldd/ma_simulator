import numpy as np
from global_config import gc

def update_bucket(peds):
    """
    speed up simulation by bucket algorithm
    """
    xmesh = gc["xmesh"]
    ymesh = gc["ymesh"]
    bucket = [[] for col in range(xmesh * ymesh)]
    for ped in peds:
        x = int(ped.x[0] / gc["bucket_scale"])
        y = int(ped.x[1] / gc["bucket_scale"])
        bucket_num = x+y*xmesh
        if bucket_num >= xmesh*ymesh:
            continue
        bucket[x + y * xmesh].append(ped)
    return bucket

class Forces(object):
    def __init__(self,props,ped):
        self.f = np.zeros(2)
        self.props = props
        self.ped = ped
        self.order = self.ped.destorder
    def f_destination(self):
        """
        force to destination
        """
        now = self.ped.destnow
        r = self.ped.dests[self.order[now]].center - self.ped.x
        r_norm = np.linalg.norm(r)
        e = r/r_norm
        return (self.ped.desired_v*e - self.ped.v)/self.props.tau
    def f_rep(self,peds):
        """
        repulsive force between pedestrians
        """
        xmesh = gc["xmesh"]
        ymesh = gc["ymesh"]
        f_r = np.array([0.0, 0.0])
        # update_bucket
        bucket = update_bucket(peds)
        # get own bucket positions
        x = int(self.ped.x[0] / gc["bucket_scale"])
        y = int(self.ped.x[1] / gc["bucket_scale"])
        # list of neighbor bucket
        neighbors = [(x - 1 + (y - 1) * xmesh), (x + (y - 1) * xmesh), (x + 1 + (y - 1) * xmesh),
                     (x - 1 + y * xmesh), (x + y * xmesh), (x + 1 + y * xmesh),
                     (x - 1 + (y + 1) * xmesh), (x + (y + 1) * xmesh), (x + 1 + (y + 1) * xmesh)]
        for neighbor in neighbors:
            # neighbor bucket
            if 0 <= neighbor < xmesh * ymesh:
                # bucket[neighbor] has information of peds.
                for ped_other in bucket[neighbor]:
                    if (self.ped != ped_other and ped_other.isvalid):
                        r = self.ped.x - ped_other.x
                        r_norm = np.linalg.norm(r)
                        _f =self.props.V_to_ped * \
                             np.exp(-r_norm / self.props.b_to_ped)
                        # TODO set view angle
                        v_norm = np.linalg.norm(self.ped.v)
                        if v_norm == 0 or (np.dot(r, self.ped.v)) / (r_norm * v_norm) > 0:
                            f_r += (_f / r_norm) * r
                        f_r += (_f / r_norm) * r
        return f_r

    def f_wall(self):
        """
        repulsive force from wall
        """
        f_w = np.zeros(2)
        for wall in gc["walls"]:
            vertices = wall.vertices
            invalid_side = wall.invalid_side
            if self.ped.is_inside(vertices,self.ped.x):
                for i in range(len(vertices)):
                    if len(invalid_side)!=0 and \
                        (tuple(vertices[i-1]),tuple(vertices[i])) in invalid_side:
                        continue
                    line = vertices[i]-vertices[i-1]
                    vec  = self.ped.x - vertices[i-1]
                    # distance between point and line
                    distance = abs(np.cross(line,vec)/np.linalg.norm(line))
                    n_tmp = np.array([line[1],-line[0]])
                    # normal vector
                    n = n_tmp/np.linalg.norm(n_tmp)
                    f_w += n*self.props.V_to_obst * np.exp(-distance / self.props.b_to_obst)
        return f_w

    def f_obstacle(self):
        """
        repulsive force from obstacles
        """
        f_o = np.zeros(2)
        obstacles = gc["obstacles"]
        for obs in obstacles:
            vec = self.ped.x - obs.center
            vec_normal = vec/np.linalg.norm(vec)
            distance = np.abs(np.linalg.norm(vec)-obs.r)
            f_o += vec_normal * \
                   obs.V * np.exp(-distance / obs.b)

        return f_o

    def f_sum(self,peds):
        f_dest = self.f_destination()
        f_rep = self.f_rep(peds)
        f_wall = self.f_wall()
        f_obst = self.f_obstacle()
        f_rand = np.random.rand(2)*0.5 # heuristic
        self.f = f_obst + f_wall + f_dest + f_rep + f_rand
        return self.f

