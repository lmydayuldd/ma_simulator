# -*- coding: utf-8 -*-
# social force model を pygameｓ で 実装してみた
# 参考:"Social Force Model for Pedestrian Dynamics"
# Dirk Helbing et.al. 1995
import pygame, os, math, time, random
from pygame.locals import * #イベント
import numpy as np
#TODO:座標をnumpyにする
#全体の形/障害物の配置とか流入流出量とかを決定する。
config = {
    "width" : 1000,             #画面の幅
    "height" : 1000,            #画面の高さ
    "num_peds" : 50,            #粒子の数
    "ave_desired_v" : 1.34,     #粒子の望ましい速度の平均
    "var_desired_v" : 0.26,     #粒子の望ましい速度の分散
    "desire_to_max_v" : 1.3,    #最大速度を決定
    "ex_pr" : {"rect":pygame.Rect(470,970,60,30),
               "prob" : 0.01},      #出口のプロパティ(x,y,幅,高,流出確率)
    "tau" : 0.5,                    #出口へのaspiration的なやつ
    "interaction_to_others" : 1,    #他粒子との相互作用(強さ)
    "range_of_interaction" : 5,     #相互作用の減衰具合(高い→減衰しにくい)
    "view_angle" : (90/180)*np.pi   #視野角
}
#pygameのスクリーンを設定しておく
pygame.init()
screen = pygame.display.set_mode([config["width"],config["height"]])

#動く粒子の特性を記述する
class Pedestrians(object):
    def __init__(self):
        self.isvalid = True
        self.kind = 0
        self.x = np.array([np.random.uniform(0, config["width"]), np.random.uniform(0, config["height"])])
        self.v = np.array([0,0])
        self.vx = 0
        self.vy = 0
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
        return _p <config["ex_pr"]["prob"]
    #退出処理
    def exit(self):
        if(config["ex_pr"]["rect"].x<self.x[0]<config["ex_pr"]["rect"].x+config["ex_pr"]["rect"].width
           and config["ex_pr"]["rect"].y<self.x[1]<config["ex_pr"]["rect"].y+config["ex_pr"]["rect"].height):
            if(self.isexit()):
                self.isvalid = False
    #速度の更新(Forcesで出した力をぶち込む)
    def v_update(self,force):
        self.v += (force).astype(self.v.dtype)
        _v = np.linalg.norm(self.v)
        if _v >= self.maximum_v:
            self.v = self.v * self.maximum_v/_v
    #位置の更新
    def x_update(self):
        self.x += self.v


#力を決定する
class Forces(object):
    def __init__(self):
        # 目的地に向かう力
        self.f = np.array([0,0])

    def f_destination(self,ped):
        dest_x = config["ex_pr"]["rect"].x+config["ex_pr"]["rect"].width/2
        dest_y = config["ex_pr"]["rect"].y+config["ex_pr"]["rect"].height
        dest = np.array([dest_x,dest_y])
        r = dest - ped.x
        r_norm = np.linalg.norm(r)
        e = r/r_norm
        return (ped.desired_v*e - ped.v)/config["tau"]
    # 人との相互作用(反発)
    def f_repulsive(self,ped,peds):
        f_r = np.array([0,0])
        for ped_other in peds:
            if(ped != ped_other and ped_other.isvalid):
                r = ped.x - ped_other.x
                r_norm = np.linalg.norm(r)
                # f_ij = Aexp(-r_ij/B) r_ij/||r_ij|| としておく
                # ココらへんはあとで変えたい
                _f = config["interaction_to_others"]*np.exp(-r_norm**2/config["range_of_interaction"])
                # 視野角(angle)
                v_norm = np.linalg.norm(ped.v)
                if (np.dot(-r,ped.v))/(r_norm*v_norm) > np.cos(config["view_angle"]):
                    f_r += (_f / r_norm) * r
        return f_r
    ## 障害物からの影響
    def f_obstacle(self):
        return f_obstacle
    def f_sum(self,ped,peds):
        self.f = self.f_destination(ped) + self.f_repulsive(ped, peds)
        return self.f


    def f_attractive(self):
        return f_attractive

#粒子のアップデートと描画の仕方を記述する
def update(peds):
    pygame.draw.rect(screen,
                     (255, 255, 255),
                     config["ex_pr"]["rect"])
    force = Forces()
    for ped in peds:
        if ped.isvalid:
            ped.v_update(force.f_sum(ped,peds))
            ped.x_update()
            pygame.draw.circle(screen, (0, 100, 255),
                               (int(ped.x[0]), int(ped.x[1])),
                               5)#int(ped.rad_particle()))
            ped.exit()
    return peds


def run():
    #初期化：それぞれプロパティを持った粒子をばらまく
    peds = []
    for i in range(config["num_peds"]):
        peds.append(Pedestrians())
    # 更新
    flags = True
    while(flags):
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