import numpy as np 
from scipy import stats
import random
import matplotlib.pyplot as plt
import operator
import copy
from hospital_module import *
from outside_math_module import *
import time


if __name__ == '__main__':
    tic=time.perf_counter()
    # tic=time.time()
    print(tic)

    worse_prob=0.2
    better_prob=1-worse_prob
    better_mean=10
    better_std=5
    worse_mean=7
    worse_std=3
    days=20    # for days in range(19,20): #用for loop可以让医院一天一天跑，每天把结果输出。range最低1天，0天跑不了，但可以从多天起跑。下面氧气和呼吸机供应率的更新日期最大可以是days-1日， 而且理论上应该根据每日收治的病人情况来每日更新计算。
    
    # batch_population_list=[1000, 500, 200]
    # take_list_dict = {}
    # better_dict={}
    # worse_dict={}
    # better_cumu_dict={}
    # worse_cumu_dict={}
    # remain_dict={}

    # for day in range(2,days):
    #     for batch_counter in range(1,day):

    #         batch_population=batch_population_list[batch_counter] # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
    batch_population=1000
    batch_list=np.ones(days)*batch_population
    print("batch_list = ", batch_list)

    take_list=np.ones(days)*20
    print("take_list = ", take_list)

            #### 
    better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list=outside_batch(days, batch_list, take_list, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std)
            # better_dict[batch_counter]=better_list
            # worse_dict[batch_counter]=worse_list
            # better_cumu_dict[batch_counter]=better_cumu_list
            # worse_cumu_dict[batch_counter]=worse_cumu_list
            # remain_dict[batch_counter]=remain_list
            # print("better_dict: ", better_dict)
            # print("worse_dict: ", worse_dict)
            # print("better_cumu_dict: ", better_cumu_dict)
            # print("worse_cumu_dict: ", worse_cumu_dict)

    toc=time.perf_counter()
    # toc=time.time()
    print(toc)
    print(toc-tic)




