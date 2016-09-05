# -*- coding: utf-8 -*-

import operator
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base,benchmarks,creator,tools
import opt_obstacle

# objectを設定

# 最適化設定
# maximize the value of the fitness of out particles
# weight->多目的の時の重み。単目的ならば、maximize->1.0, minimize->-1.0

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))

# 粒子の設定
# listとしてスピードを保持, sminとsmaxはスピードの範囲
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
    smin=None, smax=None, best=None)


"""
Operators
PSOのアルゴリズムはinitializer/updater/evaluatorに分かれている。
"""
def generate(size, pmin, pmax, smin, smax):
    #particleの作成(位置だけ決める)
    part = creator.Particle(random.uniform(pmin,pmax) for _ in range(size))
    #初期スピードをリストとして記述(スカラー量)
    part.speed = [random.uniform(smin,smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

#粒子のアップデート by PSO
#map(operator.hoge,A,B)->AとBをhogeして返する
def updateParticle(part, best, phi1, phi2):
    #乱数発生
    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    #移動スピードの更新を行う
    #自分の過去のベストと比べる
    v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    #前ステップの全体のベストと比べる
    v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    #慣性項+(過去の自分との修正)+(全体を見ての修正) phiはどっちに重みをおくか。探索or探検
    part.speed = list(map(operator.add, part.speed,map(operator.add,v_u1,v_u2)))
    ##speedのキャッピング
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(operator.add,part,part.speed))

#関数を作る時→xはlist(tuple?)で渡されている
#評価関数はこれにて記述する。ここのparameterの数に応じて
#register.particleのsizeを変える
def function(x):
    # y = (x[0]-30)**2*(np.abs(np.sin(x[0]))+1)
    # y = x[0]**4 - 20*x[0]**3 + 400*x[0]- 5
    y = (x[0]-1)**2+(x[1]-4)**2+(x[2]-19)**2
    # tuple型で受けるように設定せねばならない(多目的最適化用にそうなっている)ので、,を忘れない
    return y,

#一次元の可視化
def graph1D(xmin,xmax,best):
    x = np.linspace(xmin,xmax,500)
    dammy = np.zeros(500)
    y = []
    x_ = zip(x,dammy)
    for i in range(500):
        y.append(function(x_[i]))
    plt.plot(x,y)
    plt.plot(best, function(best, )[0], 'bo')
    plt.show()

#準備したものをフレームワークにぶち込む
toolbox = base.Toolbox()
toolbox.register("Particle", generate, size=625, pmin=0, pmax=1.2, smin=-0.5, smax=0.5)
toolbox.register("population", tools.initRepeat,list,toolbox.Particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate",opt_obstacle.run)


def main():
    pop = toolbox.population(n=5)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    # stats.register("avg", np.mean)
    # stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    logbook = tools.Logbook()
    logbook.header = ["gen","evals"] + stats.fields
    GEN = 500
    best = None
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)
        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        # print(logbook.stream)
        print GEN,best
    return pop, logbook, best

main()