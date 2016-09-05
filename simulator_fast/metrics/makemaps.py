    # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import os

#stepだけを抽出
def step_index(df):
    steps = {}
    for step in df.STEP:
        steps[step] = df.index[df.STEP == step]
    return steps
 
#人流データからヒートマップの時系列情報を抽出
def heatmap(df,x_scale,y_scale,t_interval):
    heatmaps = [] #heat mapの時系列情報
    xmesh_num = 250/x_scale
    ymesh_num = 250/y_scale
    step_ind = step_index(df)
    step_num = len(step_ind)/t_interval+1
 
    for chunk in range(step_num):
        #時系列の時間スケールごとに区切ったやつ
        heatmap = np.zeros((xmesh_num,ymesh_num))
        for i in range(chunk*t_interval+1,(chunk+1)*t_interval+1):
            if i > len(step_ind):
                break
            for step in step_ind[i]:
                ###よくわからないエラーの応急処置
                ### NaN check
                if df.x[step]!=df.x[step] or df.y[step]!=df.y[step]:
                    continue
                x = int(df.x[step]/x_scale)
                y = int(df.y[step]/y_scale)
                if (x >= 0 and x < xmesh_num and y >= 0 and y < ymesh_num):
                    heatmap[x][y] += 1.0
        heat1D = heatmap.reshape(xmesh_num*ymesh_num)
        heatmaps.append(heat1D)
    return heatmaps

def makemaps():
    x_scales = [5, 10, 25, 50, 250]
    t_intervals = [5, 10, 25, 50, 250]
    # x_scales = [25]
    # t_intervals = [25]
    int_to_peds = [1 + i * 0.2 for i in range(21)]  # [1.0,5.0](0.2)
    range_of_ints = [i * 1.0 for i in range(21)]  # [0.0,20.0](1.0)
    orig_dir = "exp_result/original_data/"
    map_dir = "exp_result/heatmap_data/"
    for x_scale in x_scales:
        for t_interval in t_intervals:
            #make directory for heatmap data
            outdir = "t%dx%d"%(t_interval,x_scale)
            outpath = map_dir+outdir
            os.makedirs(outpath)
            for int_to_ped in int_to_peds:
                for range_of_int in range_of_ints:
                    #read original data (result of simulation)
                    file_orig = "%s_%s_0.5.csv"%(int_to_ped,range_of_int)
                    path_orig = orig_dir+file_orig
                    df = pd.read_csv(path_orig)
                    result_file = outpath+"/%s_%s_0.5.csv"%(int_to_ped,range_of_int)
                    with open(result_file, 'w'):
                        pass
                    heat_orig = heatmap(df, x_scale, x_scale, t_interval)
                    #ファイル保存
                    with open(result_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerows(heat_orig)
                    print "finish",result_file

if __name__=='__main__':
    makemaps()