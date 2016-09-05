# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from global_config import gc

def record_data(step, peds, mat):
    """
    record flow data with dataframe (pandas)
    :param step: number of steps in simulation
    :param peds: list of pedestrians(class)
    :param mat: dataframe
    """
    for ped in peds:
        record = [
            step,
            ped.id,
            ped.x[1],
            ped.x[0],
            ped.v[1],
            ped.v[0],
            ped.f[1],
            ped.f[0],
            ped.kind
        ]
        mat.append(record)

def anime(df,filename,save=False):
    """
    create animation from dataframe
    :param df: data frame of people flow
    :param filename: filename when saving animation
    :param save: decide if you save or not (default:False)
    :param dest:destination
    """
    walls = gc["walls"]
    obstacles = gc["obstacles"]
    destinations = gc["destinations"]

    step_index = {}
    for step in df.STEP:
        step_index[step] = df.index[df.STEP == step]

    fig = plt.figure(figsize=matplotlib.figure.figaspect(1))
    plt.gca().set_aspect('equal', adjustable='box')
    ax = fig.add_subplot(111)
    # contour
    for wall in walls:
        wall = plt.Polygon(wall.vertices,color="red",alpha=0.2)
        ax.add_patch(wall)
    # obstacles
    for obstacle in obstacles:
        obst = plt.Circle(obstacle.center,obstacle.r,color="green",alpha = 0.5)
        ax.add_patch(obst)
    # destinations
    for dest in destinations:
        dest_range = plt.Polygon(dest.dest_range,color="black",alpha=0.5)
        dest = plt.Polygon(dest.vertices,color="black",alpha=0.2)
        ax.add_patch(dest_range)
        ax.add_patch(dest)

    ims = []
    colors = ["red","blue"]
    for step in range(len(step_index)):
        im = plt.scatter(df.y[step_index[step + 1]],
                         df.x[step_index[step + 1]],
                         color=[colors[i] for i in df.KIND[step_index[step+1]]],
                         s=80,
                         alpha=0.5)
        ims.append([im])

    ani = animation.ArtistAnimation(fig,ims,interval=10)
    plt.xlim(gc["min_xy"][0],gc["max_xy"][0])
    plt.ylim(gc["min_xy"][1],gc["max_xy"][1])
    # if save:
    #     print("start saving animation")
    #     ani.save(filename)
    #     print("finish saving animation")
    plt.show()
