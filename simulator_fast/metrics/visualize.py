# -*- coding: utf-8 -*-
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def visualize(t_interval,x_scale,int_to_ped = 3.0, range_of_int = 10.0):
    fig = plt.figure()
    ims = []
    map_dir = "exp_result/heatmap_data/t%sx%s/"%(t_interval,x_scale)
    # map_dir = "t%sx%s/"%(t_interval,x_scale)
    file = map_dir + "%s_%s_0.5.csv"%(int_to_ped,range_of_int)

    x_mesh = 250/x_scale #!!割り切れる時
    with open(file,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            for i,moji in enumerate(row):
                row[i] = float(moji)
            nprow = np.array(row)
            npmap = nprow.reshape(x_mesh,x_mesh)
            im = plt.imshow(npmap,aspect='auto',interpolation='nearest')
            ims.append([im])
    ani = animation.ArtistAnimation(fig,ims,interval=10)
    # ani.save("anime.mp4")
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    visualize(25,25)