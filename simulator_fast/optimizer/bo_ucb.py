# -*- coding: utf-8 -*-
from math import sqrt
import numpy as np
import numpy.linalg as la
import collections
import operator

import cma_es as cmaes
import compare_heat as ch

# Bayesian Optimization using Upper Confidence Bound
class BayesOptUCB:
    def __init__(self, f, bounds):
        self.f = f
        self.bounds = collections.OrderedDict(bounds)
        self.dim = len(bounds.keys())

        lower = np.zeros(self.dim)      # 下限
        upper = np.zeros(self.dim)      # 上限

        for i, key in enumerate(self.bounds):
            b = self.bounds[key]
            lower[i] = b[0]
            upper[i] = b[1]
        self.upper = upper
        self.lower = lower

    # generate n return random vectors from lower to higher
    def init_random(self, num):
        X = np.zeros((num, self.dim))
        for i in range(num):
            for j in range(self.dim):
                X[i][j] = np.random.uniform(self.lower[j], self.upper[j])
        return X

    #make the kernel vectors and matrices
    def kernel(self, x1, x2, sigma0=2.0,sigma1=5.0):
        a = map(operator.sub, x1, x2)
        r = (la.norm(a) ** 2) / (sigma1 ** 2)
        #gauss kernel
        gauss = sigma0 * np.exp(-r / 2.0)
        ### materm52
        km52 = sigma0 * (1 + np.sqrt(5 * r) + 5 * r / 3) * np.exp(-np.sqrt(5 * r))
        return km52

    # Kernel Matrix
    # 今までのデータからカーネル行列を作る
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
    def update_with_cmaes(self,data_x, data_y,beta,iteration):
        ###最大値を求める
        x_opt = cmaes.optimize_cmaes(lambda x:self.acquisition(x,data_x,data_y,beta),
                                     self.bounds,
                                     self.dim,iteration,
                                     record=False
                                     )
        print x_opt,self.acquisition(x_opt,data_x,data_y,beta)
        return x_opt


    # Return arg max of Given Function
    def optimize(self, init_num=5, iter_num=30, beta=1.0):
        # Initialize by random vectors
        X = self.init_random(init_num)
        # Y
        Y = np.zeros(init_num)
        for i, x in enumerate(X):
            Y[i] = self.f(x)[0]
        # 初期Y(max)
        y_max = max(Y)
        # 初期X
        ind_y_max = np.where(Y == y_max)[0][0]
        x_best = X[ind_y_max]

        # Optimization with Gaussian Process
        for i in range(iter_num):

            x_new = self.update_with_cmaes(X,Y,1,50)
            y_new = self.f(x_new)


            if y_new > y_max:
                x_best = x_new
                y_max = y_new

            X = np.vstack((X, x_new))
            Y = np.hstack((Y, y_new))

            print i,":new " ,(x_new,y_new)
            print i,":best",(x_best,y_max)
            print ""
        self.X = X
        self.Y = Y
        print zip(X,Y)
        return (x_best, y_max)

# DEBUG
if __name__ == "__main__":
    #simと連携
    bounds = {'tau': (0.45, 0.55),'interaction':(0.0,20.0),'range':(0.0,30.0)}
    bo = BayesOptUCB(lambda x: ch.main(x), bounds)
    ret = bo.optimize(init_num=5, iter_num=50, beta=1.0)