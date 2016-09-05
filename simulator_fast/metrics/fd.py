# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import SFM_speed as sfm
"""
7/22
make fundamental diagram (FD) from pedestrian flow
"""

def load_data(int_to_ped,range_of_int):
    """
    load files from datasets
    :param int_to_ped,range_of_int (from simulation)
    :return dataframe
    """
    orig_dir = "exp_result/original_data/"
    file_orig = "%s_%s_0.5.csv" % (int_to_ped, range_of_int)
    file_path = orig_dir+file_orig
    df = pd.read_csv(file_path)
    return df

def step_index(df):
    """
    extract step index from dataframe
    """
    steps = {}
    for step in df.STEP:
        steps[step] = df.index[df.STEP == step]
    return steps

def agent_index(df):
    """
    tracking agents from dataframe
    """
    agents = {}
    for agent in df.ID:
        agents[agent] = df.index[df.ID == agent]
    return agents

def make_fd(df):
    ai = agent_index(df)
    si = step_index(df)
    num_agents = len(ai)
    #average speed
    v_ave = np.zeros(num_agents)
    #average density
    d_ave = np.zeros(num_agents)
    for id in range(len(ai)):
        _v = np.sqrt(df.v_x[ai[id]] ** 2 + df.v_y[ai[id]] ** 2)
        v_ave[id] = _v.mean()
        _d = 0
        for step in range(df.STEP[ai[id][0]], df.STEP[ai[id][-1]] + 1):
            _d += len(si[step])
            d_ave[id] = _d * 1.0 / (1 + df.STEP[ai[id][-1]] - df.STEP[ai[id][0]])
    return np.vstack((d_ave,v_ave))

#clustering scatter with k groups
def kmeans(scatter,k):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, random_state=10).fit(scatter.T)
    labels = kmeans.labels_
    plt.figure()
    #visualize
    result = np.vstack((labels,scatter))
    plt.scatter(result[1],result[2],c=result[0],alpha=0.8)
    plt.savefig("kmeans.png")

def gmm(scatter,k):
    from sklearn.mixture import GMM
    gmm = GMM(n_components=k)
    gmm.fit(scatter.T)
    # visualize
    plt.figure()
    plt.xlim(20,50)
    plt.ylim(0,2.5)
    X,Y = np.meshgrid(np.linspace(20,100),np.linspace(0,2.5))
    f = lambda x,y : np.exp(gmm.score(np.array([x,y])))
    Z = np.vectorize(f)(X,Y)
    plt.pcolor(X,Y,Z,alpha=0.2)
    plt.scatter(scatter[0],scatter[1],alpha=0.8)
    plt.savefig("gmm.png")

import time
if __name__ == '__main__':
    # load data
    # df = load_data(3.0,20.0)
    conf = [0.5,3.0,20.0]
    start = time.time()
    df = sfm.run(conf)
    end = time.time()
    print "simulation time is %f"%(end-start)

    # make fundamental diagram
    start = time.time()
    # density,velocity = make_fd(df)
    scatter = make_fd(df)
    plt.scatter(scatter[0],scatter[1])
    plt.savefig("test.png")
    end = time.time()
    print "mapping time is %f"%(end-start)

    start = time.time()
    kmeans(scatter,3)
    end = time.time()
    print "kmeans time is %f" %(end-start)

    start = time.time()
    gmm(scatter,3)
    end = time.time()
    print "gmms time is %f" %(end-start)