# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import os.path
import matplotlib.pyplot as plt


"""
Sum diff with timeと
Sum diff with DTWの関数が
ヒートマップ感の類似度を表す(その中に画像館の比較をどうするかも含まれている)
最適化は,"maximize"する方向に働くので、それを踏まえて符号は調整
"""
class metrics:
    def __init__(self,x_scale,t_interval):
        self.diff = 0
        self.x_scale = x_scale
        self.y_scale = x_scale #aspect ratioが違うときはここを変える(引数一個追加)
        self.t_interval = t_interval
        self.xmesh_num = 250/self.x_scale
        self.ymesh_num = 250/self.y_scale

    #時系列にそって単純和を取る
    def Sum_diff_with_time(self,f_diff,heat1,heat2):
        self.diff = 0
        steps = np.array([len(heat1),len(heat2)])
        # ->0を挿入する
        if steps[0] > steps[1]:
            heat2 = np.append(heat2,np.zeros((steps[0]-steps[1],self.xmesh_num*self.ymesh_num)),axis=0)
        elif steps[1] > steps[0]:
            heat1 = np.append(heat1,np.zeros((steps[1]-steps[0],self.xmesh_num*self.ymesh_num)),axis=0)
        for i in range(steps.max()):
            delta = f_diff(heat1[i],heat2[i])
            self.diff += np.sum(delta)
        return self.diff

    #dynamic time warping(dtw)で時系列距離を出す
    def Sum_diff_with_DTW(self,f_diff,heat1, heat2):
        d = np.zeros([len(heat1) + 1, len(heat2) + 1])
        d[:] = -np.inf
        d[0, 0] = 0
        for i in range(1, d.shape[0]):
            for j in range(1, d.shape[1]):
                cost = f_diff(heat1[i-1],heat2[j-1])
                d[i, j] = cost + max(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
        return d[-1][-1]

    #Sum of Squared Distance
    def SSD(selfs,vec1,vec2):
        ssd = np.sum((vec1-vec2)**2)
        return -ssd

    #Sum of Absolute Distance
    def SAD(self, vec1,vec2):
        sad = np.sum(np.abs(vec1-vec2))
        return -sad

    #Normalized Cross Correlation
    def NCC(self,vec1,vec2):
        ncc =0
        numerator = np.dot(vec1,vec2)
        denominator = np.linalg.norm(vec1)*np.linalg.norm(vec2)
        if denominator != 0:
            ncc = numerator/denominator
        return ncc

    #Zero-mean Normalized Cross Correlation
    def ZNCC(self,vec1,vec2):
        zncc = 0
        numerator = self.xmesh_num*self.ymesh_num*np.dot(vec1,vec2) \
                    -np.sum(vec1)*np.sum(vec2)
        denominator = np.sqrt(self.xmesh_num*self.ymesh_num*np.sum(vec1**2)-np.sum(vec1)**2) \
                      *np.sqrt(self.xmesh_num*self.ymesh_num*np.sum(vec2**2)-np.sum(vec2)**2)
        if denominator != 0:
            zncc = numerator / denominator
        return zncc

    #Kullback-Leibler Divergnce
    def KL(self,vec1,vec2):
        kl = 0
        #smoothing
        tmp1 = vec1 + 0.01
        tmp2 = vec2 + 0.01
        #caluclation
        p1 = tmp1/np.sum(tmp1)
        p2 = tmp2/np.sum(tmp2)
        for i in range(len(p1)):
            kl+=p1[i]*np.log(p1[i]/p2[i])
        return -kl

    #Jensen-Shannon Divergence
    def JS(self,vec1,vec2):
        js = 0
        #smoothing
        tmp1 = vec1 + 0.01
        tmp2 = vec2 + 0.01
        #caluclation distribution
        p1 = tmp1/np.sum(tmp1)
        p2 = tmp2/np.sum(tmp2)
        r = (p1+p2)/2.0
        for i in range(len(p1)):
            js += (p1[i]*np.log(p1[i]/r[i])+p2[i]*np.log(p2[i]/r[i]))/2.0
        return -js

    # Earth Movers Distance(途中)
    def EMD(self,vec1,vec2):
        emd = 0
        return emd

    def PCA(self, heats):
        from sklearn import decomposition as dc
        X = np.zeros((len(heats), self.xmesh_num * self.ymesh_num))
        for i in range(len(X)):
            X[i] = heats[i]
        num_data, dim = X.shape
        # print X.shape
        pca = dc.PCA(n_components=3)
        pca.fit(X)
        print pca.explained_variance_ratio_
        print pca.components_
        X_pca = pca.transform(X)
        print X_pca.shape
        ####途中
        # Principle Component Analysis(途中)



def read_heatmap(x_scale,t_interval,int_to_ped,range_of_int):
    map_dir = "exp_result/heatmap_data/t%sx%s/" % (t_interval, x_scale)
    # map_dir = "t%sx%s/" % (t_interval, x_scale)
    file = map_dir + "%s_%s_0.5.csv" % (int_to_ped, range_of_int)
    heatmap_series = []
    with open(file,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            #string -> float(nparray)
            numrow = np.array(map(float,row))
            heatmap_series.append(numrow)
    return heatmap_series

def main():
    # x_scales = [5, 10, 25]
    # t_intervals = [5, 10, 25]
    int_to_peds = [1 + i * 0.2 for i in range(21)]  # [1.0,5.0](0.2)
    range_of_ints = [i * 1.0 for i in range(21)]  # [0.0,20.0](1.0)
    x_scales = [25]
    t_intervals = [25]
    # int_to_peds = [5.0]
    # range_of_ints = [20.0]
    for x_scale in x_scales:
        for t_interval in t_intervals:
            metrics = metrics(x_scale,t_interval)
            heat1 = read_heatmap(x_scale,t_interval,5.0,10.0)
            for int_to_ped in int_to_peds:
                for range_of_int in range_of_ints:
                    print "CONFIG   scale:%s interval:%s,intensity:%s range:%s" \
                          % (x_scale, t_interval, int_to_ped, range_of_int)
                    heat2 = read_heatmap(x_scale, t_interval, int_to_ped, range_of_int)
                    kl  = metrics.Sum_diff_with_time(metrics.KL,heat1,heat2)
                    js  = metrics.Sum_diff_with_time(metrics.JS,heat1,heat2)
                    ssd = metrics.Sum_diff_with_time(metrics.SSD,heat1,heat2)
                    ncc_dtw = metrics.Sum_diff_with_DTW(metrics.NCC,heat1,heat2)
                    sad = metrics.Sum_diff_with_time(metrics.SAD,heat1,heat2)
                    ncc= metrics.Sum_diff_with_time(metrics.NCC,heat1,heat2)
                    zncc = metrics.Sum_diff_with_time(metrics.ZNCC,heat1,heat2)
                    print "DISTANCE ssd %f, sad %f, ncc %f, zncc %f"%(ssd,sad,ncc,zncc)
                    print "naive dist:%d  dtw dist:%d  Delta:%d"%(ncc,ncc_dtw,(ncc-ncc_dtw))
                    print "kl %f   js %f    delta %f"%(kl,js,kl-js)

import time
def run(x_scale,t_interval,conf):
    metrics = metrics(x_scale,t_interval)
    heat1 = read_heatmap(x_scale,t_interval,3.0,10.0)
    import makemaps as mheat
    import SFM_speed as SFM
    df = SFM.run(conf)
    heat2 = mheat.heatmap(df,x_scale,x_scale,t_interval)
    #単純和
    start = time.time()
    kl = metrics.Sum_diff_with_time(metrics.KL, heat1, heat2)
    js = metrics.Sum_diff_with_time(metrics.JS, heat1, heat2)
    ssd = metrics.Sum_diff_with_time(metrics.SSD,heat1,heat2)
    sad = metrics.Sum_diff_with_time(metrics.SAD,heat1,heat2)
    ncc = metrics.Sum_diff_with_time(metrics.NCC,heat1,heat2)
    zncc = metrics.Sum_diff_with_time(metrics.ZNCC,heat1,heat2)
    middle = time.time()
    #dtw
    Dkl = metrics.Sum_diff_with_DTW(metrics.KL, heat1, heat2)
    Djs = metrics.Sum_diff_with_DTW(metrics.JS, heat1, heat2)
    Dssd = metrics.Sum_diff_with_DTW(metrics.SSD,heat1,heat2)
    Dsad = metrics.Sum_diff_with_DTW(metrics.SAD,heat1,heat2)
    Dncc = metrics.Sum_diff_with_DTW(metrics.NCC,heat1,heat2)
    Dzncc = metrics.Sum_diff_with_DTW(metrics.ZNCC,heat1,heat2)
    end = time.time()
    # print "NAIVE DISTANCE TIME %f kl %f, js %f, ssd %f, sad %f, ncc %f, zncc %f"%(middle-start,kl,js,ssd,sad,ncc,zncc)
    # print "DTW DISTANCE   TIME %f kl %f, js %f, ssd %f, sad %f, ncc %f, zncc %f"%(end - start,Dkl,Djs,Dssd,Dsad,Dncc,Dzncc)
    return kl,js,ssd,sad,ncc,zncc,Dkl,Djs,Dssd,Dsad,Dncc,Dzncc

def exp1(restart,x_scale,t_interval,conf):
    kls = np.zeros(restart)
    jss = np.zeros(restart)
    ssds = np.zeros(restart)
    sads = np.zeros(restart)
    nccs = np.zeros(restart)
    znccs = np.zeros(restart)
    dkls = np.zeros(restart)
    djss = np.zeros(restart)
    dssds = np.zeros(restart)
    dsads = np.zeros(restart)
    dnccs = np.zeros(restart)
    dznccs = np.zeros(restart)
    for i in range(restart):
        kl,js,ssd,sad,ncc,zncc,Dkl,Djs,Dssd,Dsad,Dncc,Dzncc = run(x_scale,t_interval,conf)
        kls[i] = kl
        jss[i] = js
        ssds[i] = ssd
        sads[i] = sad
        nccs[i] = ncc
        znccs[i] = zncc
        dkls[i] = Dkl
        djss[i] = Djs
        dssds[i] = Dssd
        dsads[i] = Dsad
        dnccs[i] = Dncc
        dznccs[i] = Dzncc
    #基準みたいなもの(同じパラメータでもどのくらいの誤差があるのか)
    #単純和編
    # print ""
    # print "*****************************************************"
    # print "naive sumation"
    # print "AVERAGE  KL:%f, JS:%f, SSD:%f, SAD:%f, NCC:%f, ZNCC:%f"\
    #       %(kls.mean(),jss.mean(), ssds.mean(),sads.mean(),nccs.mean(),znccs.mean())
    # print "VARIANCE KL:%f, JS:%f, SSD:%f, SAD:%f, NCC:%f, ZNCC:%f" \
    #        % (kls.std(), jss.std(), ssds.std(), sads.std(), nccs.std(), znccs.std())

    #DTWを用いた方編
    # print "*****************************************************"
    # print "DTW"
    # print "AVERAGE  KL:%f, JS:%f, SSD:%f, SAD:%f, NCC:%f, ZNCC:%f"\
    #       %(dkls.mean(),djss.mean(),dssds.mean(),dsads.mean(),dnccs.mean(),dznccs.mean())
    # print "VARIANCE KL:%f, JS:%f, SSD:%f, SAD:%f, NCC:%f, ZNCC:%f" \
    #        % (dkls.std(), djss.std(), dssds.std(), dsads.std(), dnccs.std(), dznccs.std())

    #データ吐き出し
    print kls.mean(),jss.mean(), ssds.mean(),sads.mean(),nccs.mean(),znccs.mean(),
    # print kls.std(), jss.std(), ssds.std(), sads.std(), nccs.std(), znccs.std()
    print dkls.mean(), djss.mean(), dssds.mean(), dsads.mean(), dnccs.mean(), dznccs.mean()
    # print dkls.std(), djss.std(), dssds.std(), dsads.std(), dnccs.std(), dznccs.std()
    return kls.mean(),jss.mean(), ssds.mean(),sads.mean(),nccs.mean(),znccs.mean(),\
           dkls.mean(), djss.mean(), dssds.mean(), dsads.mean(), dnccs.mean(), dznccs.mean(),\
           kls.std(), jss.std(), ssds.std(), sads.std(), nccs.std(), znccs.std(),\
           dkls.std(), djss.std(), dssds.std(), dsads.std(), dnccs.std(), dznccs.std()

#描画する(excelのちなみに②のやつ)
def exp2(x_space,t_interval,iteration):
    conf = [0.5, 3.0, 10.0]
    x,t,iteration = x_space,t_interval,iteration
    percentages = [-1.0 + 0.02 * (i+1) for i in range(100)]
    #percentages = [-1.0+0.4*(i+1) for i in range(5)]
    taus = np.zeros(len(percentages))
    interactions = np.zeros(len(percentages))
    ranges = np.zeros(len(percentages))

    #1.tauをいじる
    #平均
    kls_m = np.zeros(len(percentages))
    jss_m = np.zeros(len(percentages))
    ssds_m = np.zeros(len(percentages))
    sads_m = np.zeros(len(percentages))
    nccs_m = np.zeros(len(percentages))
    znccs_m = np.zeros(len(percentages))
    Dkls_m = np.zeros(len(percentages))
    Djss_m = np.zeros(len(percentages))
    Dssds_m = np.zeros(len(percentages))
    Dsads_m = np.zeros(len(percentages))
    Dnccs_m = np.zeros(len(percentages))
    Dznccs_m = np.zeros(len(percentages))
    #標準偏差
    kls_s = np.zeros(len(percentages))
    jss_s = np.zeros(len(percentages))
    ssds_s = np.zeros(len(percentages))
    sads_s = np.zeros(len(percentages))
    nccs_s = np.zeros(len(percentages))
    znccs_s = np.zeros(len(percentages))
    Dkls_s = np.zeros(len(percentages))
    Djss_s = np.zeros(len(percentages))
    Dssds_s = np.zeros(len(percentages))
    Dsads_s = np.zeros(len(percentages))
    Dnccs_s = np.zeros(len(percentages))
    Dznccs_s = np.zeros(len(percentages))
    for i,percentage in enumerate(percentages):
        conf_tmp = [conf[0]+conf[0]*percentage,conf[1],conf[2]]
        # conf_tmp = [conf[0], conf[1] + conf[1] * percentage, conf[2]]
        # conf_tmp = [conf[0],conf[1],conf[2]+conf[2]*percentage]
        taus[i] = conf_tmp[0]
        kls_m[i], jss_m[i], ssds_m[i], sads_m[i], nccs_m[i], znccs_m[i], \
        Dkls_m[i], Djss_m[i], Dssds_m[i], Dsads_m[i], Dnccs_m[i], Dznccs_m[i],\
        kls_s[i], jss_s[i], ssds_s[i], sads_s[i], nccs_s[i], znccs_s[i], \
        Dkls_s[i], Djss_s[i], Dssds_s[i], Dsads_s[i], Dnccs_s[i], Dznccs_s[i]=\
            exp1(iteration, x, t, conf_tmp)
    plt.figure(figsize=(12,18),dpi=300)
    #plt.xlim(taus[0],taus[-1])
    plt.subplot(3, 2, 1)
    plt.title("KL divergence")
    plt.plot(taus,kls_m,'b-',taus,Dkls_m,'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 2)
    plt.title("JS Divergence")
    plt.plot(taus,jss_m, 'b-', taus, Djss_m,'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 3)
    plt.title("SSD")
    plt.plot(taus, ssds_m, 'b-', taus, Dssds_m,'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 4)
    plt.title("SAD")
    plt.plot(taus, sads_m, 'b-', taus, Dsads_m,'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 5)
    plt.title("NCC")
    plt.plot(taus, nccs_m, 'b-', taus, Dnccs_m,'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 6)
    plt.title("ZNCC")
    plt.plot(taus, znccs_m, 'b-', taus, Dznccs_m,'r-')
    plt.legend(['naive', 'DTW'])
    filename = "tau_x%d_t%d.png"%(x,t)
    plt.savefig(filename)
    #2. intensityをいじる
    kls_m = np.zeros(len(percentages))
    jss_m = np.zeros(len(percentages))
    ssds_m = np.zeros(len(percentages))
    sads_m = np.zeros(len(percentages))
    nccs_m = np.zeros(len(percentages))
    znccs_m = np.zeros(len(percentages))
    Dkls_m = np.zeros(len(percentages))
    Djss_m = np.zeros(len(percentages))
    Dssds_m = np.zeros(len(percentages))
    Dsads_m = np.zeros(len(percentages))
    Dnccs_m = np.zeros(len(percentages))
    Dznccs_m = np.zeros(len(percentages))
    # 標準偏差
    kls_s = np.zeros(len(percentages))
    jss_s = np.zeros(len(percentages))
    ssds_s = np.zeros(len(percentages))
    sads_s = np.zeros(len(percentages))
    nccs_s = np.zeros(len(percentages))
    znccs_s = np.zeros(len(percentages))
    Dkls_s = np.zeros(len(percentages))
    Djss_s = np.zeros(len(percentages))
    Dssds_s = np.zeros(len(percentages))
    Dsads_s = np.zeros(len(percentages))
    Dnccs_s = np.zeros(len(percentages))
    Dznccs_s = np.zeros(len(percentages))
    for i, percentage in enumerate(percentages):
        conf_tmp = [conf[0], conf[1] + conf[1] * percentage, conf[2]]
        # conf_tmp = [conf[0],conf[1],conf[2]+conf[2]*percentage]
        interactions[i] = conf_tmp[1]
        kls_m[i], jss_m[i], ssds_m[i], sads_m[i], nccs_m[i], znccs_m[i], \
        Dkls_m[i], Djss_m[i], Dssds_m[i], Dsads_m[i], Dnccs_m[i], Dznccs_m[i], \
        kls_s[i], jss_s[i], ssds_s[i], sads_s[i], nccs_s[i], znccs_s[i], \
        Dkls_s[i], Djss_s[i], Dssds_s[i], Dsads_s[i], Dnccs_s[i], Dznccs_s[i] = \
            exp1(iteration, x, t, conf_tmp)
    plt.figure(figsize=(12, 18), dpi=300)
   # plt.xlim=(interactions[0],interactions[-1])
    plt.subplot(3, 2, 1)
    plt.title("KL divergence")
    plt.plot(interactions, kls_m, 'b-', interactions, Dkls_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 2)
    plt.title("JS Divergence")
    plt.plot(interactions, jss_m, 'b-', interactions, Djss_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 3)
    plt.title("SSD")
    plt.plot(interactions, ssds_m, 'b-', interactions, Dssds_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 4)
    plt.title("SAD")
    plt.plot(interactions, sads_m, 'b-', interactions, Dsads_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 5)
    plt.title("NCC")
    plt.plot(interactions, nccs_m, 'b-', interactions, Dnccs_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 6)
    plt.title("ZNCC")
    plt.plot(interactions, znccs_m, 'b-', interactions, Dznccs_m, 'r-')
    plt.legend(['naive', 'DTW'])
    filename = "interactions_x%d_t%d.png"%(x,t)
    plt.savefig(filename)

    # 3. rangesをいじる
    kls_m = np.zeros(len(percentages))
    jss_m = np.zeros(len(percentages))
    ssds_m = np.zeros(len(percentages))
    sads_m = np.zeros(len(percentages))
    nccs_m = np.zeros(len(percentages))
    znccs_m = np.zeros(len(percentages))
    Dkls_m = np.zeros(len(percentages))
    Djss_m = np.zeros(len(percentages))
    Dssds_m = np.zeros(len(percentages))
    Dsads_m = np.zeros(len(percentages))
    Dnccs_m = np.zeros(len(percentages))
    Dznccs_m = np.zeros(len(percentages))
    # 標準偏差
    kls_s = np.zeros(len(percentages))
    jss_s = np.zeros(len(percentages))
    ssds_s = np.zeros(len(percentages))
    sads_s = np.zeros(len(percentages))
    nccs_s = np.zeros(len(percentages))
    znccs_s = np.zeros(len(percentages))
    Dkls_s = np.zeros(len(percentages))
    Djss_s = np.zeros(len(percentages))
    Dssds_s = np.zeros(len(percentages))
    Dsads_s = np.zeros(len(percentages))
    Dnccs_s = np.zeros(len(percentages))
    Dznccs_s = np.zeros(len(percentages))
    for i, percentage in enumerate(percentages):
        conf_tmp = [conf[0],conf[1],conf[2]+conf[2]*percentage]
        ranges[i] = conf_tmp[2]
        kls_m[i], jss_m[i], ssds_m[i], sads_m[i], nccs_m[i], znccs_m[i], \
        Dkls_m[i], Djss_m[i], Dssds_m[i], Dsads_m[i], Dnccs_m[i], Dznccs_m[i], \
        kls_s[i], jss_s[i], ssds_s[i], sads_s[i], nccs_s[i], znccs_s[i], \
        Dkls_s[i], Djss_s[i], Dssds_s[i], Dsads_s[i], Dnccs_s[i], Dznccs_s[i] = \
            exp1(iteration, x, t, conf_tmp)
    plt.figure(figsize=(12, 18), dpi=300)
    #plt.xlim = (ranges[0], ranges[-1])
    plt.subplot(3, 2, 1)
    plt.title("KL divergence")
    plt.plot(ranges, kls_m, 'b-', ranges, Dkls_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 2)
    plt.title("JS Divergence")
    plt.plot(ranges, jss_m, 'b-', ranges, Djss_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 3)
    plt.title("SSD")
    plt.plot(ranges, ssds_m, 'b-', ranges, Dssds_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 4)
    plt.title("SAD")
    plt.plot(ranges, sads_m, 'b-', ranges, Dsads_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 5)
    plt.title("NCC")
    plt.plot(ranges, nccs_m, 'b-', ranges, Dnccs_m, 'r-')
    plt.legend(['naive', 'DTW'])
    plt.subplot(3, 2, 6)
    plt.title("ZNCC")
    plt.plot(ranges, znccs_m, 'b-', ranges, Dznccs_m, 'r-')
    plt.legend(['naive', 'DTW'])
    filename = "ranges_x%d_t%d.png"%(x,t)
    plt.savefig(filename)



if __name__ == '__main__':
    run(10,10,conf)