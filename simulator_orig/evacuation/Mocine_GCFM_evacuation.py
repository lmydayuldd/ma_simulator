# -*- coding: utf-8 -*-
# ref:"Generalized Centrifugal Force Model for Pedestrian Dynamics"
# Mocine Chraibi et.al. 2010
# 力学機構でうまくoverlappingとoscillationのバランスをとろうとしている。

import pygame, os, math, time, random
from pygame.locals import * #イベント
import numpy as np
config = {
    "width" : 500,             #画面の幅
    "height" : 250,            #画面の高さ
    "num_peds" : 50,            #粒子の数
    "ave_desired_v" : 1.34,     #粒子の望ましい速度の平均
    "var_desired_v" : 0.26,     #粒子の望ましい速度の分散
    "desire_to_max_v" : 1.3,    #最大速度を決定
    "ex_pr" : {"rect":pygame.Rect(220,220,60,30),
               "prob" : 0.0},      #出口のプロパティ(x,y,幅,高,流出確率)
    "tau" : 0.5,                    #出口へのaspiration的なやつ
    "view_angle" : (90/180)*np.pi,   #視野角
    "repulsive_r" : 5.0,             #式(9)でのr
    "eta" : 0.5,                     #式(10)でのeta
    "r_cutoff" : 50,                 #式(1)でのcut off r
}
# 目的地の座標
dest_x = config["ex_pr"]["rect"].x + config["ex_pr"]["rect"].width / 2
dest_y = config["ex_pr"]["rect"].y + config["ex_pr"]["rect"].height
dest = np.array([dest_x, dest_y])

#pygameのスクリーンを設定しておく
pygame.init()
screen = pygame.display.set_mode([500,500])

#動く粒子の特性を記述する
class Pedestrians(object):
    def __init__(self):
        self.isvalid = True
        self.kind = 0
        self.x = np.array([np.random.uniform(0, config["width"]), np.random.uniform(0, config["height"])])
        self.v = np.array([0.0,0.0])
        self.desired_v = np.random.normal(config["ave_desired_v"], config["var_desired_v"])
        # self.desired_v = 1.0
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
        return _p <config["ex_pr"]["prob"]
    #退出処理
    def exit(self):
        if(config["ex_pr"]["rect"].x<self.x[0]<config["ex_pr"]["rect"].x+config["ex_pr"]["rect"].width
           and config["ex_pr"]["rect"].y<self.x[1]<config["ex_pr"]["rect"].y+config["ex_pr"]["rect"].height):
            if(self.isexit()):
                self.isvalid = False

    #速度の更新(Forcesで出した力をぶち込む)
    def v_update(self,force):
        self.v += force
        _v = np.linalg.norm(self.v)
        if _v >= self.maximum_v:
            self.v = self.v * self.maximum_v/_v
    #位置の更新
    def x_update(self):
        #壁はすり抜けちゃダメ
        tmp = self.x + self.v
        if 0<tmp[0]<config["width"] and 0<tmp[1]<config["height"]:
            self.x = tmp


#力を決定する
#TODO コンストラクタに値をもたせたほうがいいと思う。
class Forces(object):
    def __init__(self):
        self.f = np.array([0.0,0.0])

    # 目的地に向かう力
    def f_destination(self,ped):
        r = dest - ped.x
        r_norm = np.linalg.norm(r)
        e = r/r_norm
        return (ped.desired_v*e - ped.v)/config["tau"]
    # 人との相互作用(反発)
    def f_repulsive(self,ped,peds):
        f_i = np.array([0,0])
        for ped_other in peds:
            if(ped != ped_other and ped_other.isvalid):
                r_ij = ped_other.x - ped.x     #R_ij 式(5)
                #cut off
                if np.linalg.norm(r_ij) < config["r_cutoff"]:
                    e_ij = r_ij/np.linalg.norm(r_ij)     #e_ij 式(5)
                    v_ij = max(np.dot((ped.v - ped_other.v),e_ij),0) #式(7)
                    #式(8)
                    if np.linalg.norm(ped.v) != 0:
                        k_ij = max(np.dot(ped.v,e_ij)/np.linalg.norm(ped.v),0)
                    else:
                        k_ij = 0
                    f_ij = -k_ij * (config["eta"]*ped.desired_v+np.linalg.norm(v_ij))**2\
                           / np.abs(np.linalg.norm(r_ij)-2*config["repulsive_r"]) * e_ij #式(10)の分母に絶対値
                    f_i += f_ij.astype(f_i.dtype)
        return f_i
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
            (ped.x[0],2,np.array([1,0])),                  #壁左端 →
            (config["width"]-ped.x[0],3,np.array([-1,0])), #壁右端 ←
        ]
        min_to_wall = min(dist_to_walls) #最小距離とそのときの法線ベクトル
        #print min_wall #(最小のインデックス, (距離, 法線ベクトル))のセット
        v_iw = max(np.dot(ped.v,min_to_wall[2]),0)
        if np.linalg.norm(ped.v) != 0:
            k_iw = max(np.dot(ped.v,min_to_wall[2]),0)/np.linalg.norm(ped.v)
        else: k_iw = 0
        f_iw = -k_iw *(config["eta"]*ped.desired_v + np.linalg.norm(v_iw)**2)\
                /(np.linalg.norm(min_to_wall[0])-2*config["repulsive_r"]) * min_to_wall[2]
        return f_iw
    def f_attractive(self):
        return f_attractive
    def f_sum(self,ped,peds):
        self.f = self.f_destination(ped)+self.f_wall(ped)+self.f_repulsive(ped, peds)
        return self.f
#粒子のアップデートと描画の仕方を記述する
def update(peds):
    pygame.draw.rect(screen,
                     (255, 255, 255),
                     config["ex_pr"]["rect"])
    force = Forces()
    for ped in peds:
        if ped.isvalid:
            ped.x_update()
            ped.v_update(force.f_sum(ped,peds))

            pygame.draw.circle(screen, (0, 100, 255),
                               (int(ped.x[0]), int(ped.x[1])),
                               5)#int(config["repulsive_r"]))#int(ped.rad_particle()))
            ped.exit()
    return peds


def run():
    #初期化：それぞれプロパティを持った粒子をばらまく
    peds = []
    for i in range(config["num_peds"]):
        peds.append(Pedestrians())
    # 更新
    flags = True
    # iter = 0
    while(flags):
        # print iter
        # iter += 1
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
    print ("finish")

if __name__=='__main__':
    run()