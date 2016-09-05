# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np
import numpy.linalg as la
import collections
import operator
import csv

import cma_es as cmaes
import compare_heat as ch

result_dir = "../../dataset/result_data/bo_ucb/"


# Bayesian Optimization using Upper Confidence Bound
class BayesOptUCB:
    def __init__(self, f, bounds):
        self.f = f
        self.bounds = collections.OrderedDict(bounds)
        self.dim = len(bounds.keys())

        normalized_bounds = collections.OrderedDict(bounds)
        lower = np.zeros(self.dim)      # 下限
        upper = np.zeros(self.dim)      # 上限
        normalize = np.zeros(self.dim)  # 正規化用倍率

        for i, key in enumerate(self.bounds):
            b = self.bounds[key]
            normalized_bounds[key] = (0.0,1.0)
            lower[i] = b[0]
            upper[i] = b[1]
            normalize[i] = (b[1]-b[0])*1.0
        self.normalized_bounds = normalized_bounds
        self.upper = upper
        self.lower = lower
        self.normalize = normalize
        print normalize
    # generate n return random vectors from lower to higher
    def init_random(self, num):
        X = np.zeros((num, self.dim))
        for i in range(num):
            for j in range(self.dim):
                X[i][j] = np.random.uniform(0,1)
        return X

    #make the kernel vectors and matrices
    def kernel(self, x1, x2, sigma0=5.0,sigma1=2.0):
        a = map(operator.sub, x1, x2)
        r = (la.norm(a) ** 2) / (sigma1 ** 2)
        #gauss kernel
        gauss = sigma0 * np.exp(-r / 2.0)
        ### materm52
        km52 = sigma0 * (1 + np.sqrt(5 * r) + 5 * r / 3) * np.exp(-np.sqrt(5 * r))
        return km52

    # Kernel Matrix
    def k_matrix(self,x):
        k_mat = []
        for xi in x:
            k_vec = []
            for xj in x:
                k = self.kernel(xi,xj)
                k_vec.append(k)
            k_mat.append(k_vec)
        return np.matrix(k_mat)

    # Kernel Vector
    def k_vector(self,x, x_new):
        k_vec = []
        for xi in x:
            k = self.kernel(xi, x_new)
            k_vec.append(k)
        return np.array(k_vec)

    #fitting with gaussian process
    def gp(self,data_x, data_y, new_x):
        # randomness
        s_ = 0.00001
        # calculate kernel
        k_vec = self.k_vector(data_x, new_x)
        k_mat = self.k_matrix(data_x)
        I = np.identity(len(data_x))
        mu = np.dot(np.dot(k_vec.transpose(),
                           la.inv(k_mat + I * s_ ** 2)),
                    data_y)
        sig = self.kernel(new_x, new_x) - \
              np.dot(np.dot(k_vec.transpose(),
                            la.inv(k_mat + I * s_ ** 2)),
                     k_vec)
        return float(mu),float(sig)

    def acquisition(self,new_x,data_x,data_y,beta):
        mu,sig2 = self.gp(data_x,data_y,new_x)
        return (mu + np.sqrt(beta*sig2)),

    # acquisition function (UCB)
    # 定義域内で# 新たにxを選ぶ(UCB)
    def update_with_cmaes(self,ch,data_x, data_y,beta,iteration):
        ###最大値を求める
        x_opt = cmaes.optimize_cmaes(lambda x:self.acquisition(x,data_x,data_y,beta),
                                     ch,
                                     self.normalized_bounds,
                                     5.0, 5,
                                     self.dim,iteration,
                                     record=False)
        return x_opt

    # Return arg max of Given Function
    def optimize(self, ch,init_num=5, iter_num=30, beta=1.0, record=True):
        """
        正規化したboundで更新していく
        X:0-1の範囲を取る(正規化した値)
        X_orig:元の範囲に戻したもの
        Y:f(X_orig):関数の時だけX_origをいれる
        self.X,self.Y = X,Y:BOの更新はすべて正規化された範囲で行う
        """
        if record:
            correct_paras = ch.correct_params
            print "xscale=%d, t_interval=%d, met=(%s,%s)" %\
                  (ch.x_scale, ch.t_interval,ch.metrics[0],ch.metrics[1])
            #ex) "opt_result_t10_x10_metNAIVE-SSD_repeat5.csv
            record_file = result_dir + "opt_result_t" + str(ch.t_interval) + \
                          "_x" + str(ch.x_scale) + \
                          "_met"+str(ch.metrics[0])+"-"+str(ch.metrics[1])+\
                          "_repeat"+str(ch.iteration)+".csv"
            # generation, tau, intensity, range, normalized error
            with open(record_file, 'w')as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow(["generation", "tau", "interaction_to_ped",
                                 "range_of_interaction", "compare_result", "normalized_error"])

        # Initialize by random vectors
        X = self.init_random(init_num)
        #正規化前の値に直す
        X_orig = np.zeros((init_num,self.dim))
        for i,x in enumerate(X):
            X_orig[i] = (x*self.normalize)+self.lower
        # Y
        Y = np.zeros(init_num)
        for i, x in enumerate(X_orig):

            Y[i] = self.f(x)[0]
        # 初期Y(max)
        y_max = max(Y)
        # 初期X
        ind_y_max = np.where(Y == y_max)[0][0]
        x_best = X[ind_y_max]

        # Optimization with Gaussian Process
        for i in range(iter_num):
            #正規化した値でcmaesを更新
            x_new = self.update_with_cmaes(ch,X,Y,1,50)
            #正規化前の値
            x_new_orig = x_new*self.normalize+self.lower
            y_new = self.f(x_new_orig)[0]

            if y_new > y_max:
                x_best = x_new
                y_max = y_new

            X = np.vstack((X, x_new))
            Y = np.hstack((Y, y_new))

            print i,":new " ,(x_new,y_new)
            print i,":best",(x_best,y_max)
            print ""
            if record:
                ###データの書き込み
                x_best_orig = x_best * self.normalize + self.lower
                normalized_error = error(correct_paras, x_best_orig)
                with open(record_file, 'a') as f:
                    writer = csv.writer(f, lineterminator='\n')
                    writer.writerow([i, x_best_orig[0], x_best_orig[1], x_best_orig[2],
                                     y_max, normalized_error])

        self.X = X
        self.Y = Y
        x_best_orig = x_best*self.normalize+self.lower
        print x_best_orig, y_max
        return (x_best_orig, y_max)

def error(v_orig,v):
    """
    正規化されたエラーにする
    error = Sum_x {(v_orig_x-v_x)/v_orig)}**2
    :param v_orig:
    :param v:
    :return:
    """
    v_diff = v_orig-v
    for (i,_v) in enumerate(v_orig):
        v_diff[i] /= _v
    e = np.linalg.norm(v_diff)
    return e

# optimizer
def optimize(ch,bounds,init_num=5,iter_num=50,beta=0.5,record=True):
    bo = BayesOptUCB(lambda x: ch.run(x), bounds)
    ret = bo.optimize(ch,init_num, iter_num, beta,record=record)


if __name__ == "__main__":
    orig_dir = "../../dataset/data_for_opt/"
    file_orig = orig_dir + "orig50-3.0_10.0_0.5.csv"
    ch = ch.CompareHeat(file_orig,25,25,10,1,("NAIVE","KL"))

    bounds = collections.OrderedDict()
    #探索空間の定義域を設定 tau/intensity/range
    bounds['tau'] = (0.45, 0.55)
    bounds['interaction'] = (0.0, 20.0)
    bounds['range'] = (0.0, 30.0)

    optimize(ch,bounds,init_num=1,iter_num=30)
