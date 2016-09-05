# -*- coding: utf-8 -*-
import pygame, os, math, time, random
import copy
from pygame.locals import * #イベント
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

#CAによる人流シミュレーション
conf = {
    "width" : 25,        #幅
    "height":25,         #高さ
    "particle_num":100,  #初期粒子の数
    "exit_prob" : 0.2,  #退出確率
    "k":2           #SFFのexpの肩に掛かる係数
}
#出口情報を予め作っておこう
exit_info = [np.array([12,25]),
             np.array([13,25]),
             np.array([14,25])]


#粒子たちの箱
particles = []


#モノ
class Cell(object):
    def __init__(self,c_kind):
        #empty/obstacle/wall/pedes/exit
        self.c_kind = c_kind
        if self.c_kind == "wall" or self.c_kind == "obstacle":
            self.SFF = 10000
        elif self.c_kind == "exit":
            self.SFF = 0
        else:
            self.SFF = 0
    def setkind(self,c_kind):
        self.c_kind = c_kind


#粒子(動く)
class Particle(Cell):
    def __init__(self,p_kind,id,y,x):
        self.p_kind = p_kind  #種類(男女など)
        self.id = id        #id番号
        self.x = x
        self.y = y
        self.valid = True
    def get_xy(self):
        return [self.y,self.x]
    def right(self):
        self.x += 1
    def left(self):
        self.x -= 1
    def up(self):
        self.y -= 1
    def down(self):
        self.y += 1


#初期化メソッド
def initialize_cell():
    #configから今回使うフィールドを作成
    #ダメな例# cell = [[Cell("empty")]*(2+conf["width"])]*(2+conf["height"])
    #参考:http://sonickun.hatenablog.com/entry/2014/06/13/132821
    cell = [[Cell("empty") for i in range(2+conf["width"])] for j in range(2+conf["height"])]
    #初期設定(壁と場)
    for i in range(conf["width"]+2):
        for j in range(conf["height"]+2):
            #壁
            if i == 0 or i == conf["width"]+1 or j == 0 or j == conf["height"]+1:
                cell[j][i] = Cell("wall")
            else:
                cell[j][i] = Cell("empty")
    #出口
    for exit in exit_info:
        cell[exit[0]][exit[1]] = Cell("exit")
    #障害物
    cell[11][18] = Cell("obstacle")
    cell[12][18] = Cell("obstacle")
    cell[13][16] = Cell("obstacle")
    cell[14][18] = Cell("obstacle")
    cell[15][18] = Cell("obstacle")
    # cell[12][20] = Cell("obstacle")
    # cell[12][21] = Cell("obstacle")
    # cell[12][22] = Cell("obstacle")

    #Particleの配置
    id = 0
    while(id < conf["particle_num"]):
        x = random.randint(1,conf["width"])
        y = random.randint(1,conf["height"])
        if cell[y][x].c_kind == "empty":
            cell[y][x].c_kind ="pedes"
            par = Particle("normal",id,y,x)
            id += 1
            particles.append(par)
    return cell

#可視化
def visualize(cell):
    for i in range(conf["width"] + 2):
        for j in range(conf["height"] + 2):
            if cell[j][i].c_kind == "wall":
                print "*",
            elif cell[j][i].c_kind == "empty":
                print " ",
            elif cell[j][i].c_kind == "obstacle":
                print "x",
            elif cell[j][i].c_kind == "exit":
                print "!",
            elif cell[j][i].c_kind == "pedes":
                print "o",
        print ""

#可視化(pygame)
def visualize_m(screen,cell,m):
    for i in range(conf["width"] + 2):
        for j in range(conf["height"] + 2):
            rect = pygame.Rect(j*m,i*m,m,m)
            if cell[j][i].c_kind == "wall":
                screen.fill((255,255,255),rect)
            elif cell[j][i].c_kind == "empty":
                screen.fill((0,0,0),rect)
            elif cell[j][i].c_kind == "obstacle":
                screen.fill((0,0,255),rect)
            elif cell[j][i].c_kind == "exit":
                screen.fill((255,255,255),rect)
            elif cell[j][i].c_kind == "pedes":
                screen.fill((0,255,0),rect)



#SFFを出す(euclid距離で)
def SFF_Euclid(cell):
    SFF = np.zeros([conf["height"]+2,conf["width"]+2])
    for i in range(conf["width"] + 2):
        for j in range(conf["height"] + 2):
            #壁と障害物には絶対侵入させない
            if cell[j][i].c_kind == "wall" or cell[j][i].c_kind == "obstacle":
                SFF[j][i] = 10000
            #出口に向かって進むようなField
            else:
                tmp = []
                x = np.array([j,i])
                for exit in exit_info:
                    #euclidでSFFを定義
                    n = np.linalg.norm(exit-x)
                    tmp.append(n)
                SFF[j][i] = min(tmp)
    return SFF

#particleのアップデート(neumann近傍)
def update(SFF,cell):
    k = conf["k"]
    mv_candidate = []
    particles_tmp = copy.deepcopy(particles)
    for particle in particles_tmp:
        if particle.valid:
            #SFFから移動確率を出す
            e_right = np.exp(-k*SFF[particle.y][particle.x+1])
            e_left = np.exp(-k * SFF[particle.y][particle.x-1])
            e_up = np.exp(-k * SFF[particle.y-1][particle.x])
            e_down = np.exp(-k * SFF[particle.y+1][particle.x])
            e_sum = e_right+e_left+e_down+e_up
            p_right = e_right/e_sum
            p_left  = e_left /e_sum
            p_up    = e_up / e_sum
            p_down = e_down / e_sum

            #一つ選択(空いてれば)
            rnd = random.random()
            if rnd < p_right and (cell[particle.y][particle.x+1].c_kind == "empty"
                                  or cell[particle.y][particle.x+1].c_kind == "exit"):
                particle.right()
                mv_candidate.append(particle)
            if p_right < rnd < p_right+p_left and (cell[particle.y][particle.x-1].c_kind == "empty"
                                                   or cell[particle.y][particle.x-1].c_kind == "exit"):
                particle.left()
                mv_candidate.append(particle)
            if p_right+p_left < rnd < p_right+p_left+p_up\
                    and (cell[particle.y-1][particle.x].c_kind == "empty"
                         or cell[particle.y-1][particle.x].c_kind == "exit"):
                particle.up()
                mv_candidate.append(particle)
            if p_right+p_left+p_up < rnd < 1 and \
                    (cell[particle.y+1][particle.x].c_kind == "empty"
                     or cell[particle.y+1][particle.x].c_kind == "exit"):
                particle.down()
                mv_candidate.append(particle)
    #particle_tmpにより、粒子を動かした
    #あとはconflictを避ける
    mv_ids = avoid_conflict(mv_candidate)
    for i in range(len(particles_tmp)):
        if particles_tmp[i].id in mv_ids:
            # print (particles_tmp[i].y,particles_tmp[i].x),(particles[i].y,particles[i].x)
            cell[particles_tmp[i].y][particles_tmp[i].x].setkind("pedes")
            cell[particles[i].y][particles[i].x].setkind("empty")
            particles[i] = particles_tmp[i]
    ##退出するかどうか
    for particle in particles:
        for ex in exit_info:
            if list(ex) == [particle.y,particle.x]:
                rnd = random.random()
                if rnd<conf["exit_prob"]:
                    cell[particle.y][particle.x].setkind("exit")
                    particle.valid = False
                    particles.remove(particle)
    return cell

#衝突を避けたい
"""
todo 避けるアルゴリズム(ゆずりあいなどはここで入れられそう)
動かす候補のリストを受けて
動かしていい粒子のidリストを返す
"""
def avoid_conflict(candidates):
    move_id = [] #動かしていい粒子のid
    while len(candidates) > 0:
        # かぶってるかどうかを判定するリスト
        conflict_list = [candidates[0]]
        #リストから消去
        candidates.remove(candidates[0])
        for particle in candidates:
                if conflict_list[0].get_xy() == particle.get_xy():
                    conflict_list.append(particle)
                    candidates.remove(particle)
        #conflictの数を見る
        if len(conflict_list) == 1:
            move_id.append(conflict_list[0].id)
        if len(conflict_list) == 2:
            rnd = random.random()
            if rnd < 0.5:
                sample = random.choice(conflict_list)
                move_id.append(sample.id)
        if len(conflict_list) == 3:
            rnd = random.random()
            if rnd < 0.25:
                sample = random.choice(conflict_list)
                move_id.append(sample.id)
        if len(conflict_list) == 4:
            rnd = random.random()
            if rnd < 0.125:
                sample = random.choice(conflict_list)
                move_id.append(sample.id)
    # print move_id
    return move_id

#でータの保管
def record_data(step, particles, mat):
    for par in particles:
        record = [
            step,
            par.id,
            par.x,
            par.y,
            par.p_kind
        ]
        mat.append(record)


def run():
    #初期化
    # pygameのスクリーンを設定しておく
    # pygame.init()
    m = 10
    screen = pygame.display.set_mode([(conf["width"]+2)*m, (conf["height"]+2)*m])
    #初期値の可視化
    # visualize(cell)
    steps = []
    for i in range(1):
        step = 0
        mat = []
        # セルの初期化とSFFの設定
        cell = initialize_cell()
        SFF = SFF_Euclid(cell)
        cols = ["STEP", "ID", "x", "y", "Kind"]
        while len(particles) != 0:
            cell = update(SFF, cell)
            # time.sleep(0.01)
            # visualize_m(screen,cell,m)
            record_data(step,particles,mat)
            # pygame.display.flip()
            # print "step : "+str(step)+", particles : " + str(len(particles))
            step += 1
            # 初期化:dataframeを用意する(pandas)
        frame = DataFrame(mat, columns=cols)
        file_name = "hoge.csv"
        frame.to_csv(file_name)
        print i,step
        steps.append(step)
    visualize(cell)
    st = np.array(steps)
    print st.mean(),

import time
if __name__ == '__main__':
    start = time.time()
    for i in range(100):
        run()
        print i
    end = time.time()
    print end-start