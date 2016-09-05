# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#取り出したいやつを取り出して整形するプログラム


ext_num = 210
file = "result-50-10.csv"

df = pd.read_csv(file,header=None)
da = df.as_matrix()
ext_result = []
for i in range(ext_num):
    ext_result.append(da[i][ext_num])
for j in range(ext_num,len(da)):
    ext_result.append(da[ext_num][j])
shape = np.reshape(ext_result,(21,20))


def writeresult(Z):
    frame = pd.DataFrame(shape)
    file_name = str(ext_num) + "reshape.csv"
    frame.to_csv(file_name)

def drawHeat(Z):
    im = plt.imshow(Z,aspect='auto',interpolation='nearest')
    plt.colorbar
    plt.show()

def draw3D(Z):#作成途中
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(1.0, 5.0, 0.2)
    Y = np.arange(1, 20, 1)
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(-2, 2)
    plt.xticks(())
    plt.yticks(())
    ax.set_zticks(())

drawHeat(shape)