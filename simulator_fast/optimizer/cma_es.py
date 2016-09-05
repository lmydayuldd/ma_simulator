# -*- coding: utf-8 -*-

import numpy as np
from deap import base,benchmarks,creator,tools,cma,algorithms
import random,csv
import compare_heat as ch
import collections

result_dir = "../../dataset/result_data/cma_es/"

creator.create("FitnessMin",base.Fitness, weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMin)
toolbox = base.Toolbox()


###定義域を考慮してgenerateを生成
def generate(ind_init,cma,bounds):
    """Generate a population of :math:`\lambda` individuals of type
    *ind_init* from the current strategy.
    :param ind_init: A function object that is able to initialize an
                     individual from a list.
    :returns: A list of individuals.
    """
    arz = np.random.standard_normal((cma.lambda_, cma.dim))
    arz = cma.centroid + cma.sigma * np.dot(arz, cma.BD.T)

    for n in range(len(arz)):
        for i,key in enumerate(bounds):
            b = bounds[key]
            if arz[n][i] < b[0]:
                arz[n][i] = b[0]
            elif arz[n][i] > b[1]:
                arz[n][i] = b[1]
    return map(ind_init, arz)

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


def optimize_cmaes(func, ch, bounds,sigma_,lambda_,dim,NGEN,record=True):
    # record optimization result if record == TRUE
    if record:
        # correct_paras = ch.correct_params
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

    ###########################################################
    # CMA-ESの中核:strategy#

    #
    # centroid:中心        sigma:大域ステップサイズ
    # lambda:評価用生成個数  strategy.C:共分散行列.初めは単位行列
    ###########################################################
    np.random.seed(64)
    init_params = [0 for i in range(dim)]
    strategy = cma.Strategy(init_params,sigma=sigma_,lambda_=lambda_)
    # 評価関数を登録する
    toolbox.register("evaluate", func)
    # 新しいパラメータ群を生成する
    toolbox.register("generate",generate,creator.Individual,strategy,bounds)
    # パラメータにしたがって評価する
    toolbox.register("update",strategy.update)
    # 一番良かった値を持っておく(引数は持っておく個数)
    hof = tools.HallOfFame(1)
    for gen in range(NGEN):
        # lambda分だけcmaの行列にしたがって点列を生成
        population = toolbox.generate()
        # 点列を評価(evaluateで登録した評価関数に従う)
        fitnesses = toolbox.map(toolbox.evaluate,population)
        # 評価結果を持った個体群を生成
        for ind,fit in zip(population,fitnesses):
            #populationの中の個体それぞれに評価値を与えてる
            ind.fitness.values = fit
            # print ind,fit
        # CMA-ESでアップデート
        toolbox.update(population)
        hof.update(population)
        if record:
            # generation, パラメータ, 結果 をここに。
            print gen, hof[0], hof[0].fitness.values
            ### data output
            normalized_error = error(ch.correct_params, hof[0])
            with open(record_file, 'a') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerow([gen, hof[0][0], hof[0][1], hof[0][2],
                                 hof[0].fitness.values[0], normalized_error])
    return hof[0]

# optimizer
def optimize(ch,bounds,_sigma=5.0,_lambda=5,iter=100,record=True):
    dim = len(bounds.keys())
    ##chとch.run(x)があるのは気持ち悪いかも
    optimize_cmaes(lambda x: ch.run(x),ch,bounds, _sigma, _lambda, dim, iter,record=record)

if __name__ == '__main__':
    orig_dir = "../../dataset/data_for_opt/"
    file_orig = orig_dir + "orig50-3.0_10.0_0.5.csv"
    ch = ch.CompareHeat(file_orig,10,10,10,1)

    bounds = collections.OrderedDict()
    #探索空間の定義域を設定 tau/intensity/range
    bounds['tau'] = (0.45, 0.55)
    bounds['interaction'] = (0.0,20.0)
    bounds['range'] = (0.0,30.0)

    optimize(ch,bounds,_sigma=5.0,_lambda=5,iter=100)

