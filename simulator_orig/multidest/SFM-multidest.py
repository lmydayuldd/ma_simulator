# -*- coding: utf-8 -*-
# social force model を pygameｓ で 実装してみた
# 参考:"Social Force Model for Pedestrian Dynamics"
# Dirk Helbing et.al. 1995

import numpy as np
import json,time,matplotlib
from pandas import Series,DataFrame
import copy
import matplotlib.animation as animation
import matplotlib.pyplot as plt

#全体の形/障害物の配置とか流入流出量とかを決定する。
static = {
    "width" : 250,
    "height" : 250,
    "num_peds" : 50,
    "bucket_scale" : 25,
    "iteration" : 500,
}

config = {
    "ave_desired_v" : 1.34,     #粒子の望ましい速度の平均
    "var_desired_v" : 0.26,     #粒子の望ましい速度の分散
    "desire_to_max_v" : 1.3,    #最大速度を決定
    "tau" : 0.5,                    #出口へのaspiration的なやつ
    "interaction_to_ped" : 3.0,    #他粒子との相互作用(強さ)
    "range_of_interaction_ped" : 10.0,     #相互作用の減衰具合(高い→減衰しにくい)
    "interaction_to_wall" : 3.0,
    "range_of_interaction_wall" : 1.0,
    "view_angle" : (90/180)*np.pi,   #視野角
}

#出口の設定
class exit(object):
    def __init__(self):
        self.x = 124
        self.y = 251
        self.width = 3
        self.height = 1
        self.prob = 0.05
#はじめに出口をconfigより作っておく
exit = exit()

##バケットのメッシュ
if static["width"] % static["bucket_scale"] == 0:
    xmesh = static["width"] / static["bucket_scale"]
else:
    xmesh = static["width"] / static["bucket_scale"] + 1
if static["height"] % static["bucket_scale"] == 0:
    ymesh = static["height"] / static["bucket_scale"]
else:
    ymesh = static["height"] / static["bucket_scale"] + 1



#動く粒子の特性を記述する
class Pedestrians(object):
    def __init__(self,id):
        self.id = id
        self.isvalid = True
        self.kind = 0
        self.x = np.array([np.random.uniform(0, static["width"]), np.random.uniform(0, static["height"])])
        self.v = np.array([0.0,0.0])
        self.desired_v = np.random.normal(config["ave_desired_v"], config["var_desired_v"])
        self.maximum_v = self.desired_v * config["desire_to_max_v"]
        #入るバケットを記憶(はじめは-1(フラグ))
        # self.bucket_num = -1.0
        # print "generate",self.id,self.x

    #速度に応じて粒子の大きさを変える
    def rad_particle(self):
        return np.linalg.norm(v)+1
    #自分と他の粒子との距離を測る
    def distance(self,ped_other):
        return np.sqrt((self.x-ped_other.x)**2+(self.y-ped_other.y)**2)

    #速度の更新(Forcesで出した力をぶち込む)
    def v_update(self,force):
        self.v += (force).astype(self.v.dtype)
        _v = np.linalg.norm(self.v)
        if _v >= self.maximum_v:
            self.v = self.v * self.maximum_v/_v

    #位置の更新->退出判定も入れる
    def x_update(self):
        x_tmp = self.x+self.v
        a = 0
        #x,yともにはみ出す時
        if (0 > x_tmp[0] or x_tmp[0] > static["width"]) \
                and (0 > x_tmp[1] or x_tmp[1] > static["height"]):
            self.x = self.x
            a+=1
        #y座標がはみ出している時その1
        elif 0 > x_tmp[1]:
            self.x[0] = x_tmp[0]
            a+=1
        #y座標がはみ出している時+出口の考慮
        elif x_tmp[1] > static["height"]:
            if exit.x < x_tmp[0] < exit.x + exit.width:
                #退出処理
                self.isvalid = False
                a+=1
            else:
                self.x[0] = x_tmp[0]
                a+=1
        #x座標がはみ出している時
        elif 0 > x_tmp[0] or x_tmp[0] > static["width"]:
            self.x[1] = x_tmp[1]
            a+=1
        #すべての座標が収まっている時
        elif 0<x_tmp[0]<static["width"] and 0<x_tmp[1]<static["height"]:
            self.x = x_tmp
            a+=1
        if a != 1:
            print self.id,self.x,x_tmp,self.v,"yabai"
            print config["tau"],config["interaction_to_ped"],config["range_of_interaction_ped"]
        return self.x

# speed up by bucket method
def update_bucket(peds):
    bucket = [[] for col in range(xmesh * ymesh)]
    for ped in peds:
        x = int(ped.x[0] / static["bucket_scale"])
        y = int(ped.x[1] / static["bucket_scale"])
        bucket_num = x+y*xmesh
        if bucket_num >= xmesh*ymesh:
            print ped.id,ped.x,bucket_num
            continue
        bucket[x + y * xmesh].append(ped)
    return bucket

# 粒子をバケットにinする(工事中)
# def update_bucket(bucket,peds):
#     for ped in peds:
#         x = int(ped.x[0] / static["bucket_scale"])
#         y = int(ped.x[1] / static["bucket_scale"])
#         bucket_num = x+y*xmesh
#         print bucket_num
#         if bucket_num >= xmesh*ymesh or ped.bucket_num == x+y*xmesh:
#             pass
#         else:
#             if(-1 == ped.bucket_num):
#                 ped.bucket_num = bucket_num
#             else:
#                 bucket[ped.bucket_num].remove(ped)
#                 ped.bucket_num = bucket_num
#             bucket[bucket_num].append(ped)
#     return bucket


# caluculate the force
class Forces(object):
    def __init__(self):
       # 目的地に向かう力
       self.f = np.array([0,0])
    def f_destination(self,ped):
       dest_x = exit.x+exit.width/2
       dest_y = exit.y
       dest = np.array([dest_x,dest_y])
       r = dest - ped.x
       r_norm = np.linalg.norm(r)
       e = r/r_norm
       return (ped.desired_v*e - ped.v)/config["tau"]
   # 人との相互作用(反発)
    def f_repulsive(self,ped,peds):
       f_r = np.array([0.0,0.0])
       for ped_other in peds:
           if(ped != ped_other and ped_other.isvalid):
               r = ped.x - ped_other.x
               r_norm = np.linalg.norm(r)
               _f = config["interaction_to_ped"]*np.exp(-r_norm/config["range_of_interaction_ped"])
               #TODO 視野角(angle)
               v_norm = np.linalg.norm(ped.v)
               if v_norm == 0 or (np.dot(r,ped.v))/(r_norm*v_norm) > 0:
                   f_r += (_f / r_norm) * r
               f_r += (_f / r_norm) * r
       return f_r

   # 人との相互作用(反発)
    def f_repulsive_su(self, ped, peds):
       f_r = np.array([0.0,0.0])
       # バケットに入れていく
       bucket = update_bucket(peds)
       # 自分のバケットを見る
       x = int(ped.x[0] / static["bucket_scale"])
       y = int(ped.x[1] / static["bucket_scale"])
       # 隣接8バケットを抽出(ただのindexを保持)
       neighbors = [(x - 1 + (y - 1) * xmesh), (x + (y - 1) * xmesh), (x + 1 + (y - 1) * xmesh),
                    (x - 1 + y * xmesh), (x + y * xmesh), (x + 1 + y * xmesh),
                    (x - 1 + (y + 1) * xmesh), (x + (y + 1) * xmesh), (x + 1 + (y + 1) * xmesh)]
       for neighbor in neighbors:
           # バケットが範囲内
           if 0 <= neighbor < xmesh*ymesh:
               # bucket[neighbor]にはpedestriansの情報が入っている
               for ped_other in bucket[neighbor]:
                   if (ped != ped_other and ped_other.isvalid):
                       r = ped.x - ped_other.x
                       r_norm = np.linalg.norm(r)
                       _f = config["interaction_to_ped"] *\
                            np.exp(-r_norm / config["range_of_interaction_ped"])
                       # TODO 視野角(angle)
                       v_norm = np.linalg.norm(ped.v)
                       if v_norm == 0 or (np.dot(r, ped.v)) / (r_norm * v_norm) > 0:
                           f_r += (_f / r_norm) * r
                       f_r += (_f / r_norm) * r
       return f_r

   ## TODO 障害物からの影響
    def f_obstacle(self,ped):
       # f_iw = np.array([0, 0])
       # # 壁からの距離と法線ベクトル(距離,priority,法線)
       # """
        # numpyの比較は
        # ValueError: The truth value of an array with more than one element is ambiguous.
        # と出るので、priorityで回避する
        # """
        if exit.x-exit.width/2.0 <ped.x[0] < exit.x+exit.width/2.0 and exit.y > static["height"]-5:
            f_iw = 0
        else:
            dist_to_walls = [
                (ped.x[1], 0, np.array([0, 1])),  # 壁上端 ↓
                (static["height"] - ped.x[1], 1, np.array([0, -1])),  # 壁下端 ↑
                (ped.x[0], 2, np.array([1, 0])),  # 壁左端 →
                (static["width"] - ped.x[0], 3, np.array([-1, 0])),  # 壁右端 ←
            ]
            min_to_wall = min(dist_to_walls)  # 最小距離とそのときの法線ベクトル
            r_norm = min_to_wall[0]
            f_iw = config["interaction_to_wall"] * np.exp(-r_norm / config["range_of_interaction_wall"])*min_to_wall[2]

        return f_iw
    #TODO 展示物など？
    def f_attractive(self):
        return f_attractive
    def f_sum(self, ped, peds):
        self.f = self.f_repulsive_su(ped, peds)+self.f_destination(ped)
        # self.f = self.f_repulsive(ped, peds)+self.f_destination(ped)
        return self.f

def record_data(step, peds, mat):
    for ped in peds:
        record = [
            step,
            ped.id,
            ped.x[0],
            ped.x[1],
            ped.v[0],
            ped.v[1],
            0,
            0,
            ped.kind
        ]
        mat.append(record)

#粒子のアップデートと描画の仕方を記述する
def update(peds):
    force = Forces()
    peds_tmp = copy.deepcopy(peds) #一斉に更新しようとする。
    peds_new = [] #有効なpedestrian listを更新する
    for i,ped in enumerate(peds_tmp):
        if ped.isvalid:
            peds[i].v_update(force.f_sum(ped,peds_tmp))
            peds[i].x_update()
            peds_new.append(peds[i])
    return peds_new

#人流アニメーションづくり
def anime(df,filename,save=False):
    step_index = {}
    for step in df.STEP:
        step_index[step] = df.index[df.STEP == step]
    fig = plt.figure(figsize=matplotlib.figure.figaspect(1))
    ims = []
    for step in range(len(step_index)):
        im = plt.scatter(df.y[step_index[step+1]],
                         df.x[step_index[step+1]],
                         color="blue",
                         s = 80,
                         alpha = 0.5)
        ims.append([im])
        # print "painting",step,"is finished"
    ani = animation.ArtistAnimation(fig,ims,interval=10)
    plt.xlim(0,static["height"])
    plt.ylim(0,static["width"])
    if save:
        print("start saving animation")
        ani.save(filename)
        print("finish saving animation")
    plt.show()

##configのリスト
def change_config(confs):
    #confの値は正にする(定義域)
    # config["ave_desired_v"] = confs[0]
    # config["var_desired_v"] = confs[1]
    # config["desire_to_max_v"] = confs[2]
    config["tau"] = confs[0]
    config["interaction_to_ped"] = confs[1]
    config["range_of_interaction_ped"] = confs[2]
    # config["interaction_to_wall"] = confs[6]
    # config["range_of_interaction_wall"] = confs[7]
    # config["view_angle"] = confs[8]
    # print config

##running simulation
def run(confs):
    #configの変更(tau,interaction,range)
    change_config(confs)
    #初期化:pedestrianの追加
    peds = []
    for i in range(static["num_peds"]):
        peds.append(Pedestrians(i))
    # bucketを用意
    bucket = [[] for col in range(xmesh * ymesh)]
    # 初期化:dataframeを用意する(pandas)
    cols = ["STEP", "ID", "x", "y", "v_x", "v_y", "f_x", "f_y", "Kind"]
    mat = []
    flags = True
    step = 0
    # 実行文
    while (flags):
        # print step, len(peds),
        step += 1
        peds = update(peds)
        # 全員退出したら終了,または
        # iteration回数が500超えると終了
        if len(peds) == 0 or step > static["iteration"]:
            flags = False
        record_data(step, peds, mat)
    print step,
    #アニメーション
    frame = DataFrame(mat, columns=cols)
    # anime_file = str(confs[0])+".mp4"
    # anime(frame,anime_file,save=False)
    # データ吐き出し
    # file_name = ""
    # frame.to_csv(file_name)
    return frame

#make data with various parameters
def experiment():
    import os
    conf = [0.5,0,0]
    #iterationのリスト(実験用)
    int_to_peds = [1+i*0.2 for i in range(21)]#[3.0,5.0](0.1)
    range_of_ints = [i*1.0 for i in range(21)]#[0.0,20.0](1.0)
    ###ファイル出力
    ex_dir = "exp_result/original_data"
    os.makedirs(ex_dir)
    for int_to_ped in int_to_peds:
        for range_of_int in range_of_ints:
            conf[1] = int_to_ped
            conf[2] = range_of_int
            frame = run(conf)
            file_name = "/%s_%s_%s.csv" %(conf[1],conf[2],conf[0])
            frame.to_csv(ex_dir+file_name)


if __name__=='__main__':
    confs = [0.50,1,0]
    times = []
    for i in range(1):
        start = time.time()
        frame = run(confs)
        anime_file=str(confs[0])+"_"+str(confs[1])+"_"+str(confs[2])+".mp4"
        anime(frame,anime_file,True)
        exec_time = time.time()-start
        times.append(exec_time)
    times = np.array(times)
    print times.mean()