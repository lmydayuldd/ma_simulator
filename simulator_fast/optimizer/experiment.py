# -*- coding: utf-8 -*-

import simulator_fast.optimizer.pso as PSO
import simulator_fast.optimizer.cma_es as CMA
import simulator_fast.optimizer.bo_ucb_normalize as BO
import compare_heat as CH
import collections
import argparse


parser = argparse.ArgumentParser(description='Select optimizer and recording')
parser.add_argument('--optimizer','-o', type=str,
                    help='select optimizer from PSO, CMA and BO')
parser.add_argument('--record','-r', type=bool, default=False,
                    help='select whether you will record optimization result or not')
args = parser.parse_args()


#settings
xscales = [250,50,25,10,5,1]
tintervals = [250,50,25,10,5,1]
met1 = ["NAIVE","DTW"]
met2 = ["SAD","SSD","KL","JS","NCC","ZNCC"]
orig_dir = "../../dataset/data_for_opt/"
file_orig = orig_dir+"orig50-3.0_10.0_0.5.csv"

bounds = collections.OrderedDict()
bounds['tau'] = (0.45, 0.55)
bounds['interaction'] = (0.0, 20.0)
bounds['range'] = (0.0, 30.0)


def experiment():
    if args.optimizer == "PSO":
        # script for expetiment
        # For checking the performance of metrics and optimizer
        for m1 in met1:
            for m2 in met2:
                for x in xscales:
                    for t in tintervals:
                        ch = CH.CompareHeat(file_orig,x,x,t,1,(m1,m2))
                        PSO.optimize(ch,bounds,N=5,GEN=50,record=args.record)
    if args.optimizer == "CMA":
        # script for expetiment
        # For checking the performance of metrics and optimizer
        for m1 in met1:
            for m2 in met2:
                for x in xscales:
                    for t in tintervals:
                        ch = CH.CompareHeat(file_orig,x,x,t,1,(m1,m2))
                        CMA.optimize(ch, bounds, _lambda=5, iter=50, record=args.record)
    if args.optimizer == "BO":
        # script for expetiment
        # For checking the performance of metrics and optimizer
        for m1 in met1:
            for m2 in met2:
                for x in xscales:
                    for t in tintervals:
                        ch = CH.CompareHeat(file_orig,x,x,t,1,(m1,m2))
                        BO.optimize(ch, bounds, init_num=5, iter_num=50,record=args.record)

if __name__ == '__main__':
    experiment()