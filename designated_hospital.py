import numpy as np 
from scipy import stats
import random
import matplotlib.pyplot as plt
import operator
import copy
from hospital_module import *
from outside_math_module import *

#########  把queue里的排队病人列表，按病程顺序收到empty_bed 张床上，然后给出接收病人列表，拒绝病人列表，和医院剩余空床 #############
def allocate_bed(queue, empty_bed): 
    patients_rejected = copy.copy(queue)  # 保留一份原稿，最后用来减拒绝的列表，得到接收的列表。
    for i in range(len(queue)):
        if empty_bed <= int(patients_rejected[i]):
            patients_rejected[i] -= empty_bed
            empty_bed = 0
        else:
            empty_bed -= patients_rejected[i]
            patients_rejected[i] = 0
    # 排队列表 减 拒绝的列表，得到 接收的列表
    patients_accepted = list(map(operator.sub, queue, patients_rejected))
    return patients_accepted, patients_rejected, empty_bed 

if __name__ == '__main__':

    queue = []
    take_list = []
    for days in range(19,20): #用for loop可以让医院一天一天跑，每天把结果输出。range最低1天，0天跑不了，但可以从多天起跑。下面氧气和呼吸机供应率的更新日期最大可以是days-1日， 而且理论上应该根据每日收治的病人情况来每日更新计算。
        # days=40
        worse_prob=0.2
        better_prob=1-worse_prob
        better_mean=10
        better_std=5
        worse_mean=7
        worse_std=3
        batch_population=1000 # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
        batch_list=np.ones(days)*batch_population
        print("batch_list = ", batch_list)

        take_list=np.ones(days)*20
        print("take_list = ", take_list)

        # ###### update batch_list with take_list #######
        # for i in range(0, len(take_list)):
        #     print(i)
        #     for j in range(i, days):
        #         print(j)
        #         batch_list[j]-=take_list[i]
        # print("batch_list = ", batch_list)


        #### 医院抽走人数 测试用 ##########
        # take_list=np.zeros(days) #list(range(days))
        # take_tuple_list=[(2,10),(5,300),(6,50),(10,20),(15,50)] #第5天抽走300人，第10天抽走20人。
        # for take_date, take_number in take_tuple_list:
        #     take_list[take_date]=take_number   ### take list 在plot_curve里用来从remain里减去黄线。

        #### 
        better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list=outside_batch(days, batch_list, take_list, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std)

        # queue.append(days) # 先随便用days凑个数。应该要用院外各曲线集统计的结果。
        # empty_bed = 10
        # patients_accepted, patients_rejected, empty_bed  = allocate_bed(queue, empty_bed)
        # # print('queue = ', queue)
        # print('accepted = ', patients_accepted)
        # print('rejected =', patients_rejected)
        # print('empty_bed = ', empty_bed )

        # worse_prob=0.1
        # better_prob=1-worse_prob
        # better_mean=10
        # better_std=5
        # worse_mean=7
        # worse_std=3
        # # batch_population=100 # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
        # number_of_batches = days # 假设每天新增一批病人，
        # oxygen_change_list=[(0,0.3)] #(1,0.6), (9, 0.8) #tuple list, with first element = date the change starts, second element=updated oxygen supply rate. List order doesn't matter
        # breathing_machine_change_list=[] #(4,0.8)#tuple list, with first element = date the change starts, second element=updated machine supply rate. List order doesn't matter

        # ### 医院每天结算一次空床，每天收治一批病人 ### 目前的简化院内模型中，院内只有一种病人，要么出院，要么死亡，没有中间过渡病程。
        # daily_total_better, daily_total_worse=run_hospital(days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population, number_of_batches, oxygen_change_list, breathing_machine_change_list)
        # print("daily_total_better = ", daily_total_better)
        # print("daily_total_worse = ", daily_total_worse)
        # print("\n")
