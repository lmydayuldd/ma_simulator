# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
from simulator_fast.simulator import SFM
from simulator_fast.metrics import compare_metrics
import os.path


orig_dir = "../../dataset/data_for_opt/"
file_orig = orig_dir+"orig50-3.0_10.0_0.5.csv"

class CompareHeat:
    def __init__(self,path,x_scale,y_scale,t_interval,iteration,metrics=("NAIVE","SSD")):
        # size of simulation space
        self.xsize = 250
        self.ysize = 250
        self.correct_params = np.array([0.5,3.0,10.0])
        # original data
        self.orig_path = path
        # space-time resolution
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.t_interval = t_interval
        # num of restarts of sim. for optimization
        self.iteration = iteration
        # make heatmap of original data
        df_orig = pd.read_csv(self.orig_path)
        self.heat_orig = self.heatmap(df_orig)
        self.metrics = metrics
        print "finish making heatmap of original data at"
        print "x_scale:%d t_interval:%d with (%s,%s)"\
              %(x_scale,t_interval,metrics[0],metrics[1])

    # Comparison with two heatmaps
    def compare(self, heat1):
        """
        :param heat_orig:
        :param heat1:
        :return:
        Implement state
        time series diff : naive, DTW
        heatmap diff : SAD, SSD, KL, JS, NCC, ZNCC
        """
        Met = compare_metrics.metrics(self.x_scale, self.t_interval)
        #Naive Time Series
        if(self.metrics[0] == 'NAIVE'):
            if(self.metrics[1] == 'SSD'):
                return Met.Sum_diff_with_time(Met.SSD, self.heat_orig, heat1)
            if (self.metrics[1] == 'SAD'):
                return Met.Sum_diff_with_time(Met.SAD, self.heat_orig, heat1)
            if (self.metrics[1] == 'KL'):
                return Met.Sum_diff_with_time(Met.KL, self.heat_orig, heat1)
            if (self.metrics[1] == 'JS'):
                return Met.Sum_diff_with_time(Met.JS, self.heat_orig, heat1)
            if (self.metrics[1] == 'NCC'):
                return Met.Sum_diff_with_time(Met.NCC, self.heat_orig, heat1)
            if (self.metrics[1] == 'ZNCC'):
                return Met.Sum_diff_with_time(Met.ZNCC, self.heat_orig, heat1)
        #Dynamic Time Warping
        if(self.metrics[0] == "DTW"):
            if(self.metrics[1] == 'SSD'):
                return Met.Sum_diff_with_DTW(Met.SSD, self.heat_orig, heat1)
            if (self.metrics[1] == 'SAD'):
                return Met.Sum_diff_with_DTW(Met.SAD, self.heat_orig, heat1)
            if (self.metrics[1] == 'KL'):
                return Met.Sum_diff_with_DTW(Met.KL, self.heat_orig, heat1)
            if (self.metrics[1] == 'JS'):
                return Met.Sum_diff_with_DTW(Met.JS, self.heat_orig, heat1)
            if (self.metrics[1] == 'NCC'):
                return Met.Sum_diff_with_DTW(Met.NCC, self.heat_orig, heat1)
            if (self.metrics[1] == 'ZNCC'):
                return Met.Sum_diff_with_DTW(Met.ZNCC, self.heat_orig, heat1)
        #exception
        else:
            print "¥n ERROR : No such mertic. metrics should be written as self.metric=(A,B),"
            print "A : NAIVE, DTW"
            print "B : SSD, SAD, KL, JS, NCC, ZNCC"



    def step_index(self, df):
        steps = {}
        for step in df.STEP:
            steps[step] = df.index[df.STEP == step]
        return steps

    #人流データからヒートマップの時系列情報を抽出
    def heatmap(self, df, normalize=False):
        heatmaps = [] #heat mapの時系列情報
        xmesh_num = self.xsize/self.x_scale
        ymesh_num = self.ysize/self.y_scale
        step_ind = self.step_index(df)
        step_num = len(step_ind)/self.t_interval+1

        for chunk in range(step_num):
            #時系列の時間スケールごとに区切ったやつ
            heatmap = np.zeros((xmesh_num,ymesh_num))
            for i in range(chunk*self.t_interval+1,(chunk+1)*self.t_interval+1):
                if i > len(step_ind):
                    break
                for step in step_ind[i]:
                    ###よくわからないエラーの応急処置
                    ### NaN check
                    if df.x[step]!=df.x[step] or df.y[step]!=df.y[step]:
                        continue
                    x = int(df.x[step]/self.x_scale)
                    y = int(df.y[step]/self.y_scale)
                    if (x >= 0 and x < xmesh_num and y >= 0 and y < ymesh_num):
                        heatmap[x][y] += 1.0
            if normalize:
                #正規化(各時間幅/空間幅での平均値)<-値が小さくなりすぎる
                heatmap /= self.t_interval*x_scale*y_scale
            heat1D = heatmap.reshape(xmesh_num*ymesh_num)
            heatmaps.append(heat1D)
        return heatmaps


    ##compare with oiriginal data
    def run(self, conf):
        #confに従ってdataframeを作成
        #5回の平均値にしてみる
        diff = 0
        for i in range(self.iteration):
            df = SFM.run(conf)
            heat = self.heatmap(df)
            #heatmapの比較
            diff += self.compare(heat)
            print diff,
        print "ave", diff/(self.iteration*1.0)
        return diff/(self.iteration*1.0),

if __name__=='__main__':
    confs = [0.5,3.0,10.0]
    # main(confs)
    ch = CompareHeat(file_orig,10,10,10,1,("DTW","NCC"))
    ch.run(confs)
    ch = CompareHeat(file_orig,4,4,4,3)
    ch.run(confs)