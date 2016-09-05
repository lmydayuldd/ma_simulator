# -*- coding: utf-8 -*-
# ref:"centrifugal force model for pedestrian dynamics"
# W.J. Yu, L. Chen, et. al., PRE(2005)
# CDTの実装(collisionすると向きを変える)

import pygame, os, math, time, random
from pygame.locals import * #イベント
import numpy as np
import pandas as pd
from pandas import Series,DataFrame


config = {
    "width" : 300,                  #corridorの幅
    "height" : 100,                 #corridorの高さ
    "ave_desired_v" : 2.34,         #粒子の望ましい速度の平均
    "var_desired_v" : 0.26,         #粒子の望ましい速度の分散
    "desire_to_max_v" : 1.3,        #最大速度を決定
    "tau" : 0.5,                    #出口へのaspiration的なやつ
    "view_angle" : (90/180)*np.pi,  #視野角
    "repulsive_r" : 5.0,            #collision detectionの半径
    "rotate_thetas" :                #衝突時の回転する方向
        np.array([-np.pi/2,-np.pi/3,-np.pi/6,
                  np.pi/2,np.pi/3,np.pi/6]),
    "r_paint": 5,                   #描画の半径
    "eta" : 0.0,                    #式(10)でのeta
    "r_cutoff" : 30,                #式(1)でのcut off r
    "out_prob" : 0.2,               #流出確率
    "ped_num" : 100,               #粒子の数

}
print np.random.permutation(config["rotate_thetas"])

#pygameのスクリーンを設定しておく
pygame.init()
screen = pygame.display.set_mode([config["width"],config["height"]])

#動く粒子の特性を記述する
"""
# Goalをlistで持たせる
"""
class Pedestrians(object):
    def __init__(self,id,start,goal,kind):
        self.id = id
        self.kind = kind
        self.isvalid = True
        self.kind = kind
        self.x = start
        self.goal = goal
        self.v = np.array([0.0,0.0])
        self.desired_v = np.random.normal(config["ave_desired_v"], config["var_desired_v"])
        self.maximum_v = self.desired_v * config["desire_to_max_v"]

    #速度に応じて粒子の大きさを変える
    def rad_particle(self):
        return np.sqrt(self.vx*self.vx + self.vy*self.vy)+1
    #自分と他の粒子との距離を測る
    def distance(self,ped_other):
        return np.sqrt((self.x-ped_other.x)**2+(self.y-ped_other.y)**2)
    #退出するかどうか
    def isexit(self):
        _p = np.random.rand()
        return _p < config["out_prob"]
    #退出処理
    def exit(self):
        if np.abs(self.x[0]-self.goal[0])<config["repulsive_r"]:
            if(self.isexit()):
                self.isvalid = False

    #速度の更新(Forcesで出した力をぶち込む)
    def v_update(self,force):
        self.v += force
        _v = np.linalg.norm(self.v)
        if _v >= self.maximum_v:
            self.v = self.v * self.maximum_v/_v

    #衝突のチェック→衝突してるやつだけ返してくれる
    def collision_check(self,neighbors,update_x):
        collisions = []
        for neighbor in neighbors:
            if self != neighbor and neighbor.isvalid:
                if np.linalg.norm(update_x-neighbor.x) < 2*config["repulsive_r"]:
                    collisions.append(neighbor)
        return collisions

    #位置の更新
    def x_update(self,peds):
        #とりあえず更新
        x_tmp = self.x + self.v
        #collisionが見つかった場合
        collisions = self.collision_check(peds,x_tmp)
        iter = 0
        while len(collisions) > 0:
            thetas = np.random.permutation(config["rotate_thetas"])
            #回転行列
            R_theta = np.array([
                [np.cos(thetas[iter]),np.sin(thetas[iter])],
                [-np.sin(thetas[iter]),np.cos(thetas[iter])]
            ])
            v_tmp = np.dot(R_theta,self.v)
            x_tmp = self.x + v_tmp
            collisions = self.collision_check(peds,x_tmp)
            iter += 1
            #回す選択肢がなくなると終了
            if iter == len(config["rotate_thetas"]):
                # return self.x
                collisions = []
        #壁
        if 0 < x_tmp[1] < config["height"]:
            self.x = x_tmp
        """
        周期境界条件のお話
        """
        if self.kind == 0 and np.abs(self.x[0]-config["width"])<1:
            self.x[0] = 0
        if self.kind == 1 and np.abs(self.x[0]) < 1:
            self.x[0] = config["width"]




    #startとgoalの設定
    def get_start(self):
        return self.start
    def get_goal(self):
        return self.goal

#力を決定する
#TODO コンストラクタに値をもたせたほうがいいと思う。
class Forces(object):
    def __init__(self):
        self.f = np.array([0.0,0.0])
    # 目的地に向かう力
    def f_destination(self,ped):
        r = ped.goal - ped.x
        # x方向だけに。
        e = np.array([r[0]/np.abs(r[0]),0])
        return (ped.desired_v*e - ped.v)/config["tau"]
    # 人との相互作用(反発)
    def f_repulsive(self,ped,peds):
        return np.array([0,0])
    ## 障害物からの影響
    def f_wall(self,ped):
        f_iw = np.array([0, 0])
        #壁からの距離と法線ベクトル(距離,priority,法線)
        """
        numpyの比較は
        ValueError: The truth value of an array with more than one element is ambiguous.
        と出るので、priorityで回避する
        """
        dist_to_walls = [
            (ped.x[1],0,np.array([0,1])),                  #壁上端 ↓
            (config["height"]-ped.x[1],1,np.array([0,-1])),#壁下端 ↑
        #    (ped.x[0],2,np.array([1,0])),                  #壁左端 →
        #    (config["width"]-ped.x[0],3,np.array([-1,0])), #壁右端 ←
        ]
        min_to_wall = min(dist_to_walls) #最小距離とそのときの法線ベクトル
        v_iw = max(np.dot(ped.v,min_to_wall[2]),0)
        if np.linalg.norm(ped.v) != 0:
            k_iw = max(np.dot(ped.v,min_to_wall[2]),0)/np.linalg.norm(ped.v)
        else: k_iw = 0
        f_iw = -k_iw *(config["eta"]*ped.desired_v + np.linalg.norm(v_iw)**2)\
                /np.abs(np.linalg.norm(min_to_wall[0])-2*config["repulsive_r"]) * min_to_wall[2]
        return f_iw
    def f_sum(self,ped,peds):
        self.f = self.f_destination(ped)+self.f_wall(ped)+self.f_repulsive(ped, peds)#+np.array([0,np.random.normal(0,1)])
        return self.f

#粒子のアップデートと描画の仕方を記述する
def update(peds):
    force = Forces()
    peds_new = []
    for ped in peds:
        if ped.isvalid:
            ped.x_update(peds)
            ped.v_update(force.f_sum(ped,peds))
            if ped.kind == 0:
                pygame.draw.circle(screen, (0, 100, 255),
                                   (int(ped.x[0]), int(ped.x[1])),
                                   config["r_paint"])#int(config["repulsive_r"]))#int(ped.rad_particle()))
            elif ped.kind == 1:
                pygame.draw.circle(screen, (255, 100, 0),
                                   (int(ped.x[0]), int(ped.x[1])),
                                   config["r_paint"])  # int(config["repulsive_r"]))#int(ped.rad_particle()))
            peds_new.append(ped)
    return peds_new

#データを記録する
def record_data(step,peds,frame):
    cols = ["STEP", "ID", "x", "y", "v_x", "v_y", "f_x", "f_y", "Kind"]
    for ped in peds:
        records = Series([
            step,
            ped.id,
            ped.x[0],
            ped.x[1],
            ped.v[0],
            ped.v[1],
            0,
            0,
            ped.kind
        ],index=cols)
        frame = frame.append(records,ignore_index=True)
    return frame
#実行する


def run():
    # 初期化：それぞれプロパティを持った粒子をばらまく
    peds = []
    for i in range(config["ped_num"]):
        if i < 50:
            ped = Pedestrians(id,
                              np.array([0, np.random.uniform(0, config["height"])]),
                              np.array([config["width"], np.random.uniform(0, config["height"])]),
                              0)
            # 被らないようにランダム生成し直す
            while len(ped.collision_check(peds, ped.x)) > 0:
                ped.x = np.array([np.random.uniform(0, config["width"]), np.random.uniform(0, config["height"])])
            peds.append(ped)
        else:
            ped = Pedestrians(id,
                              np.array([config["width"], np.random.uniform(0, config["height"])]),
                              np.array([0, np.random.uniform(0, config["height"])]),
                              1)
            # 被らないようにランダム生成し直す
            while len(ped.collision_check(peds, ped.x)) > 0:
                ped.x = np.array([np.random.uniform(0, config["width"]), np.random.uniform(0, config["height"])])
            peds.append(ped)
    # 初期化その2:dataframeを用意する(pandas)
    cols = ["STEP", "ID", "x", "y", "v_x", "v_y", "f_x", "f_y", "Kind"]
    frame = DataFrame(columns=cols)

    ####実行
    flags = True
    step = 0
    while(flags and step < 3000):
        screen.fill((0,0,0))
        peds = update(peds)
        time.sleep(0.01)
        pygame.display.flip()
        ### イベントハンドラ
        for event in pygame.event.get():
            # 終了条件
            if event.type == MOUSEBUTTONDOWN:
                flags = False
                print "This simulation is finished with buttondown."
        step += 1
        print step
    print ("finish")

if __name__=='__main__':
    run()