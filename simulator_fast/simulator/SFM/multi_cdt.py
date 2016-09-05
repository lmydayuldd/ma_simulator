# -*- coding: utf-8 -*-
import numpy as np
from pandas import DataFrame
import util,force_helbing
from global_config import gc
import copy
import os, time

class Props:
    """
    Pedestrians' properties
    :param initialized->OD, parameters
    :param updated->x,v,f
    """
    def __init__(self,
                 ave_v = 1.34,var_v = 0.26,to_max_v = 1.3,
                 tau = 1.1,
                 V_to_ped = 1.0, b_to_ped = 1.0,
                 V_to_obst = 1.0, b_to_obst = 1.0,
                 view_ang = (180.0/180.0)*np.pi,
                 repulsive_r=0.5
                 ):
        self.ave_v = ave_v   # ave. of desired velocity
        self.var_v = var_v   # var. of desired velocity
        self.to_max_v = to_max_v  # max_v / ave_desired_v
        self.tau = tau       # aspiration to destination
        self.V_to_ped = V_to_ped # intensity of interaction to others
        self.b_to_ped = b_to_ped  # diminishing rate of interaction
        self.V_to_obst = V_to_obst # intensity of interaction to obstacle
        self.b_to_obst = b_to_obst # diminishing rate
        self.view_ang = view_ang * np.pi # view angle
        self.repulsive_r = repulsive_r
        self.rotate_thetas = np.array([-np.pi/2,-np.pi/3,-np.pi/6,
                                        np.pi/2, np.pi/3, np.pi/6
                                       ])

class Pedestrians(object):
    def __init__(self,props,id,destorder,origin="enterance",kind="normal"):
        self.props = props   #partcle propaties (props class)
        self.id = id
        self.kind = kind
        self.isvalid = True
        self.kind = kind
        self.destorder = destorder
        self.destnow = 0
        self.dests = gc["destinations"]
        self.desired_v = np.random.normal(props.ave_v, props.var_v)
        self.maximum_v = self.desired_v * props.to_max_v
        ### vector[0] : x, vector[1] : y
        self.x = np.array([0.0, 0.0])    # initialize x (vector)
        self.v = np.array([0.0, 0.0])   # initialize v (vector)
        self.f = np.array([0.0, 0.0])   # initialize f (vector)
        # initialize x randomly
        if origin == "random":
            self.x = np.array([np.random.uniform(gc["max_xy"][0], gc["min_xy"][0]),
                               np.random.uniform(gc["max_xy"][1], gc["min_xy"][1])])
            while not self.check_position(self.x):
                self.x = np.array([np.random.uniform(gc["max_xy"][0], gc["min_xy"][0]),
                                   np.random.uniform(gc["max_xy"][1], gc["min_xy"][1])])
        # initialize x from entrance
        elif origin == "entrance":
            self.x = np.array([np.random.uniform(gc["max_entr_xy"][0], gc["min_entr_xy"][0]),
                               np.random.uniform(gc["max_entr_xy"][1], gc["min_entr_xy"][1])])

            while not self.check_position(self.x):
                self.x = np.array([np.random.uniform(gc["max_entr_xy"][0], gc["min_entr_xy"][0]),
                                   np.random.uniform(gc["max_entr_xy"][1], gc["min_entr_xy"][1])])
        print "ID {} enter inside".format(self.id)

    def show_properties(self):
        print "ID:{0} x:{1} v:{2} f:{3}"\
            .format(self.id,self.x,self.v,self.f)

    def xv_update(self,peds,timescale):
        ### update v
        self.v += self.f
        norm_v = np.linalg.norm(self.v)
        #boundary
        if norm_v >= self.maximum_v:
            self.v = self.v * self.maximum_v/norm_v
        v_tmp = self.v
        ### update x
        x_tmp = self.x + self.v*timescale
        # present destination
        dest = self.dests[self.destorder[self.destnow]]
        #boundary condition
        if self.is_inside(gc["destinations"][-1].vertices,x_tmp) and dest.exit:
            print "ID {} exits".format(self.id)
            self.isvalid = False
        ### CDT
        iter = 0
        collisions = self.check_collisions(peds,x_tmp)
        while not (self.check_position(x_tmp) and len(collisions)==0):
            #この中にcollision check?
            thetas = np.random.permutation(self.props.rotate_thetas)
            R_theta = np.array([
                [np.cos(thetas[iter]),np.sin(thetas[iter])],
                [-np.sin(thetas[iter]),np.cos(thetas[iter])]
            ])
            v_tmp = np.dot(R_theta,self.v)
            x_tmp = self.x+v_tmp
            collisions = self.check_collisions(peds,x_tmp)
            iter += 1
            if iter == len(self.props.rotate_thetas):
                print "particle {} cannot move! position:{}, collisions:{}"\
                    .format(self.id, self.check_position(x_tmp),len(collisions))
                return self.x
        self.x = x_tmp
        self.v = v_tmp

    def f_update(self,force,peds):
        self.f = force.f_sum(peds)

    def dest_update(self):
        """
        decide if change the destination
        :param p: transition rate
        """
        # destination
        now = self.dests[self.destorder[self.destnow]]
        # check if dest is exit or not
        if not now.exit:
            # check if ped is inside the dest_range
            if self.is_inside(now.dest_range,self.x):
                _p = np.random.rand()
                # decide transit destination or not
                if _p < now.prob and self.destnow<len(self.destorder):
                    self.destnow += 1


    def is_inside(self, vertices, point,is_convex=False):
        """
        check if point is inside or outside
        (Polygon)
        """
        if is_convex:
            # TODO 内部にあるときは、なす角360度
            return True
        elif not is_convex:
            count = 0
            for i in range(len(vertices)):
                if ((vertices[i - 1][0] <= point[0] and vertices[i][0] > point[0]) or
                        (vertices[i - 1][0] > point[0] and vertices[i][0] <= point[0])):
                    vt = (point[0] - vertices[i - 1][0]) / (vertices[i][0] - vertices[i - 1][0])
                    if (point[1] < (vertices[i - 1][1] + vt * (vertices[i][1] - vertices[i - 1][1]))):
                        # print "chosen vertices 2 : ({},{}) line : {}" \
                        #     .format(vertices[i - 1], vertices[i], vertices[i] - vertices[i - 1])
                        count += 1
            return count % 2 == 1

    def is_inside_circle(self,center,r,point):
        """
        check if point is inside or outside
        (circle)
        :param center: center of circle
        :param r: radius of circle
        :param point: point
        :return:
        """
        distance = np.linalg.norm(point-center)
        return distance < r

    def check_position(self,point):
        """
        check if the particle's position is appropriate
        :param point: particles' position
        :return: True (appropriate)
        """
        # inside the contour
        in_contour = False
        for wall in gc["walls"]:
            in_contour = in_contour or self.is_inside(wall.vertices,point)
        # particle must be located outside the destination vertices
        in_dests = True
        for dest in gc["destinations"]:
            in_dests = in_dests and not(self.is_inside(dest.vertices,point))
        in_obst = True
        for obst in gc["obstacles"]:
            in_obst = in_obst and not(self.is_inside_circle(obst.center,obst.r,point))
        return in_contour and in_obst #and in_dests

    def check_collisions(self,peds,update_x):
        collisions = []
        for ped in peds:
            if self != ped and ped.isvalid:
                if np.linalg.norm(update_x-ped.x) < 2*self.props.repulsive_r:
                    collisions.append(ped)
        return collisions

def update(props,peds,timescale):
    """
    update force, velocity and position of each pedestrian
    :param peds: list of pedestrians
    :param force: force to pedestrians(class)
    """
    peds_tmp = copy.deepcopy(peds)
    peds_new = []
    for i,ped in enumerate(peds_tmp):
        force = force_helbing.Forces(props, ped)
        if ped.isvalid:
            peds[i].f_update(force, peds_tmp)
            peds[i].xv_update(peds,timescale)
            peds[i].dest_update()
            peds_new.append(peds[i])
    return peds_new

def test(save_csv=False,save_anime=False):
    peds = []
    props = Props()
    destorder = [0,7,1,2,3,4,5,6,1,8,-1]
    # initialize pedestrians
    for i in range(gc["num_peds"]):
        peds.append(Pedestrians(props,i,destorder,origin="random",kind=0))
    # initialize dataframe
    cols = ["STEP","ID","x","y","v_x","v_y","f_x","f_y","KIND"]
    mat = []
    flags = True
    step = 0
    # execute simulation
    while flags:
        print "step:{0},num:{1}".format(step, len(peds))
        step += 1
        peds = update(props,peds,0.3)
        if len(peds) == 0 or step > gc["iteration"]:
            flags = False
        util.record_data(step,peds,mat)
    frame = DataFrame(mat, columns=cols)
    # create animation
    if save_anime:
        anime_file = "{}_{}_{}.mp4"\
            .format(props.tau,props.V_to_ped,props.b_to_ped)
        util.anime(frame,anime_file,save_anime)
    if save_csv:
        file_name = "{}_{}_{}.csv" \
            .format(props.tau, props.V_to_ped, props.b_to_ped)
        frame.to_csv(file_name)
    return frame

if __name__ == '__main__':
    test(save_anime=True)