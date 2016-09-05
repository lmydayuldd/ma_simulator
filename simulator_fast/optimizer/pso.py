# -*- coding: utf-8 -*-

import operator
import random
import numpy as np
from deap import base,benchmarks,creator,tools
import csv
import compare_heat as ch
import collections


orig_dir = "../../dataset/data_for_opt/"
result_dir = "../../dataset/result_data/pso/"

# objectを設定
# 最適化設定
# maximize the value of the fitness of out particles
# weight->多目的の時の重み。単目的ならば、maximize->1.0, minimize->-1.0
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 粒子の設定 listとしてスピードを保持, sminとsmaxはスピードの範囲
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
    smin=None, smax=None, best=None)

def generate(size, pmin, pmax, smin, smax):
    #particleの作成(位置だけ決める)
    part = creator.Particle(random.uniform(pmin,pmax))
    #初期スピードをリストとして記述(スカラー量)
    part.speed = random.uniform(smin,smax)
    part.smin = smin
    part.smax = smax
    return part

#粒子のアップデート by PSO
#map(operator.hoge,A,B)->AとBをhogeして返する
def updateParticle(part, best,pmin,pmax, phi1, phi2):
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
    for i  in range(len(part.speed)):
        if part.speed[i] < part.smin[i]:
            part.speed[i] = part.smin[i]
        elif part.speed[i] > part.smax[i]:
            part.speed[i] = part.smax[i]
    part[:] = list(map(operator.add,part,part.speed))
    ##定義域で範囲のcapping
    for i in range(len(part)):
        if part[i] < pmin[i]:
            part[i] = pmin[i]
        elif part[i] > pmax[i]:
            part[i] = pmax[i]


# #準備したものをフレームワークにぶち込む
# #tau,相互作用の強度、範囲を最適化
# pmins = np.array([0.45, 1.0, 0.0])
# pmaxs = np.array([0.55, 20.0, 30.0])
# smaxs = (pmaxs-pmins)*0.2
# smins = -smaxs
#
# toolbox = base.Toolbox()
# toolbox.register("Particle", generate, size=5, pmin=pmins, pmax=pmaxs, smin=smins, smax=smaxs)
# toolbox.register("population", tools.initRepeat,list,toolbox.Particle)
# toolbox.register("update", updateParticle,pmin=pmins,pmax=pmaxs, phi1=2.0, phi2=2.0)
# toolbox.register("evaluate",ch.main)


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

def pso(ch,bounds,N,GEN,record=True):
    # 準備したものをフレームワークにぶち込む
    # tau,相互作用の強度、範囲を最適化
    pmins = np.zeros(len(bounds))
    pmaxs = np.zeros(len(bounds))
    for i,key in enumerate(bounds):
        pmins[i] = bounds[key][0]
        pmaxs[i] = bounds[key][1]
    smaxs = (pmaxs - pmins) * 0.2
    smins = -smaxs

    toolbox = base.Toolbox()
    toolbox.register("Particle", generate, size=N, pmin=pmins, pmax=pmaxs, smin=smins, smax=smaxs)
    toolbox.register("population", tools.initRepeat, list, toolbox.Particle)
    toolbox.register("update", updateParticle, pmin=pmins, pmax=pmaxs, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", ch.run)

    if record:
        print "xscale=%d, t_interval=%d, met=(%s,%s)" % \
              (ch.x_scale, ch.t_interval, ch.metrics[0], ch.metrics[1])
        # ex) "opt_result_t10_x10_metNAIVE-SSD_repeat5.csv
        record_file = result_dir + "opt_result_t" + str(ch.t_interval) + \
                      "_x" + str(ch.x_scale) + \
                      "_met" + str(ch.metrics[0]) + "-" + str(ch.metrics[1]) + \
                      "_repeat" + str(ch.iteration) + ".csv"
        # generation, tau, intensity, range, normalized error
        with open(record_file, 'w')as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(["generation", "tau", "interaction_to_ped",
                             "range_of_interaction", "compare_result", "normalized_error"])

    pop = toolbox.population(n=N)
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
        # logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        # print(logbook.stream)
        print g,
        normalized_error = error(ch.correct_params, best)
        ### recording result
        if record:
            with open(record_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')  # 改行コード（\n）を指定しておく
                writer.writerow([g, best[0],best[1],best[2],best.fitness.values[0],normalized_error])  # list（1次元配列）の場合
            print best,best.fitness.values[0]

def optimize(ch,bounds,N=5,GEN=50,record=True):
    pso(ch,bounds,N,GEN,record=record)

if __name__=='__main__':
    orig_dir = "../../dataset/data_for_opt/"
    file_orig = orig_dir + "orig50-3.0_10.0_0.5.csv"
    ch = ch.CompareHeat(file_orig, 10, 10, 10, 1)
    bounds = collections.OrderedDict()
    # 探索空間の定義域を設定 tau/intensity/range
    bounds['tau'] = (0.45, 0.55)
    bounds['interaction'] = (0.0, 20.0)
    bounds['range'] = (0.0, 30.0)

    optimize(ch,bounds,N=5,GEN=10)