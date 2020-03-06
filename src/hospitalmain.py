# -*- coding: utf-8 -*-
# 对原hospital.py代码的改造，使其更结构化，清晰可见。
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import operator
import copy

# 代表模型本身的一些参数
class HospitalModelParam():
    days=20
    worse_prob = 0.1
    better_prob = 1 - worse_prob
    better_mean = 10
    better_std = 5
    worse_mean = 7
    worse_std = 3
    batch_population = 100

# 模型本身在运行时，会改变的一些参数。
class HospitalModelRunParam():
    # 氧气变化的参数
    oxygen_change_list = [(0, 0.3), (5, 0.5), (9, 1)]
    # (1,0.6), (9, 0.8) #tuple list, with first element = date the change starts,
    # second element=updated oxygen supply rate. List order doesn't matter
    # 呼吸机变化的参数
    breathing_machine_change_list = [(7, 0.4), (10, 1), (15, 0.5)]
    # (4,0.8)#tuple list, with first element = date the change starts,
    #  second element=updated machine supply rate. List order doesn't matter

class HospitalBatch():
    #存放批次信息
    batch_list=[]
    #
    better_prob_list= []
    better_prob_list_original= []
    worse_prob_list= []
    worse_prob_list_original= []

    better_list = []  # daily increment, following normal distribution
    worse_list = []  # daily increment, following normal distribution
    # 对改善的列表的累计
    better_cumu_list = []  # cumulative record, following the shape of a cumulative density function
    # 对恶化的列表累计
    worse_cumu_list = []  # cumulative record, following the shape of a cumulative density function
    remain_list = []  # those who remain in the same medical status

    # 初始化工作，每次新建时设置一下。
    def __init__(self):
        self.better_list=[]
        self.worse_list=[]
        self.better_cumu_list=[]
        self.worse_cumu_list=[]
        self.remain_list = []
        return


class HospitalModel():


    def __init__(self,modelparam,runparam):
        #用于存储参数用的两个变量
        # 模型一些固定的参数
        self.modelparam=modelparam
        # 一些比较运行时改变的参数或一些经常发生变动的参数
        self.runparam=runparam
        return

    # 根据参数来生成状态改变概率
    ### oop中可以用来在概率之间掷骰子 ###
    def change_state(self,probability_dict: dict):  # probability_dict={'better': 0.2, 'worse': 0.3}
        dice = random.random()
        print("roll dice = ", dice)
        prob_better = probability_dict["better"]
        prob_worse = prob_better + probability_dict["worse"]
        if dice <= prob_better:
            state = "better"
        elif dice <= prob_worse:
            state = "worse"
        else:
            state = "remain"
        print("result state is ", state)
        return state

    ### 每日新增概率 ###
    def daily_prob(self,day, mu, std):
        # cdf函数为累积概率，第二个参数为平均值，第三个参数为标准偏移度。
        prob = stats.norm.cdf(day + 1, mu, std) - stats.norm.cdf(day, mu, std)
        return prob


    ### 生成默认概率曲线集 ####
    ### 累计概率 ###
    def daily_cumu(self,day, mu, std):
        prob = stats.norm.cdf(day + 1, mu, std)
        return prob


    ### 生成默认概率曲线集 ####
    #def batch_curves(self,days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std):
    def batch_curves(self,batchobj):
        days = self.modelparam.days
        dates = list(range(days))
        better_cumu = 0
        worse_cumu = 0
        remain = batchobj.batch_list[0]
        for t in dates:
            # 按照分布产生每日的病情改善和恶化分布
            better = self.daily_prob(t, self.modelparam.better_mean, self.modelparam.better_std) * batchobj.better_prob_list[t]
            worse = self.daily_prob(t, self.modelparam.worse_mean, self.modelparam.worse_std) * batchobj.worse_prob_list[t]
            better_cumu += better
            # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
            worse_cumu += worse
            # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
            # 病情没变化的，等于总的数量-改善-恶化
            remain = batchobj.batch_list[t] - better_cumu - worse_cumu
            batchobj.better_list.append(better)
            batchobj.worse_list.append(worse)
            batchobj.better_cumu_list.append(better_cumu)
            batchobj.worse_cumu_list.append(worse_cumu)
            batchobj.remain_list.append(remain)

        return dates,batchobj


    #### 画图 ####
    #def plot_curves(self,dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, cure_ability_list,
    #                batch_list):
    def plot_curves(self,batchobj,cure_ability_list):

        dates = list(range(self.modelparam.days))
        plt.plot(dates, cure_ability_list * batchobj.batch_list[0], color='green', marker='o', markersize=1, linestyle='-',
                 label='Medical_supply')
        plt.plot(dates, batchobj.better_list, color='grey', marker='o', markersize=3, linestyle='--',
                 label='Better_daily(mean=10,std=5)')
        plt.plot(dates, batchobj.worse_list, color='red', marker='o', markersize=3, linestyle='--',
                 label='Worse_daily(mean=7,std=3)')
        plt.plot(dates, batchobj.better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-',
                 label='Better_cumu(mean=10,std=5)')
        plt.plot(dates, batchobj.worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-',
                 label='Worse_cumu(mean=7,std=3)')
        plt.plot(dates, batchobj.remain_list, color='yellow', marker='o', markersize=3, linestyle='-', label='Remain population')
        plt.plot(dates, batchobj.batch_list, color='blue', marker='o', markersize=3, linestyle='-', label='Batch population')
        plt.xlabel('Days')
        plt.ylabel('probability')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()
        return


    # 医院介入，拿走*个病人
    #### 更新同期病人在院外的剩余人数 #####
    def intervention_take_out_patients(self,date, take_number, batch_list):
        for i in range(date, len(batch_list)):
            batch_list[i] -= take_number
        return batch_list


    # 医院介入后，改变的概率
    #### 院外模型中可以在oop中使用的，同期病人部分被医院收治后，每日好转和恶化的概率更新
    def intervention_take_out_patients_change_prob_lists(self,date, remain_before_date_prob, better_prob_list, worse_prob_list):
        for i in range(date, len(better_prob_list)):
            better_prob_list[i] *= remain_before_date_prob  # 条件概率的应用，P(A|B)*P(B)= P(B|A)*P(A) = P()
        for i in range(date, len(worse_prob_list)):
            worse_prob_list[i] *= remain_before_date_prob  # 条件概率的应用
        return better_prob_list, worse_prob_list


    # 题：如果医院第10天从这批100个院外轻症患者中收治了20 个人，那么剩下的80人在第10天之后，每天多少人转重症，多少人转自愈？
    # 设：A=第10天转， B=前9天没有转
    # P( 第10天转 | 前9天没有转 ) x P(前9天没有转) = P( 前9天没有转 | 第10天转 ) x P(第10天转)
    # 因为所有第10天转的人，前9天一定没有转，所以 P( 前9天没有转 | 第10天转 ) = 1 , 所以
    # P( 第10天转 | 前9天没有转 ) x P(前9天没有转) = 1 x P(第10天转) = 想要的答案

    # 医院内的设备参数对整个概率的改变，对改善概率和恶化概率的改变
    #### 针对一次医力变化，对之后的好转和恶化概率做list更新 ####
    #def intervention_hospital_supply_change_step(self,date, new_better_prob_ratio, new_worse_prob_ratio, better_prob_list,
    #                                             worse_prob_list, better_prob_list_original, worse_prob_list_original):
    def intervention_hospital_supply_change_step(self, date, new_better_prob_ratio, new_worse_prob_ratio,batchobj):
        for i in range(date, len(batchobj.better_prob_list)):
            batchobj.better_prob_list[i] = batchobj.better_prob_list_original[i] * new_better_prob_ratio
        for i in range(date, len(batchobj.worse_prob_list)):
            batchobj.worse_prob_list[i] = batchobj.worse_prob_list_original[i] * new_worse_prob_ratio
        return


    # 将呼吸机供应率和氧气供应率，融入到治疗率中，并返回需要变动的时间。
    #### 综合每日氧气供应率和每日呼吸机供应率list，计算每日医力，以及需要更新好转和恶化率的日期 ####
    def cal_cure_ability(self,breathing_machine_supply_list, oxygen_supply_list):
        # 参数：breathing_machine_supply_list:呼吸机改变
        # oxygen_change_list：氧气改变参数，为一个列表，每个元素中包括两个数据，第一个为改变的时间（天），第二个为改变的供应率
        # 范例：[(1,0.6), (9, 0.8)]
        # breathing_machine_change_list :类似氧气改变参数，为呼吸机供应改变参数。

        # 供氧参数
        for oxygen_change_date, oxygen_supply_rate in self.runparam.oxygen_change_list:
            for i in range(oxygen_change_date, len(oxygen_supply_list)):
                oxygen_supply_list[i] = oxygen_supply_rate
                # print("oxygen_supply_list", oxygen_supply_list)
            # 呼吸机参数
        for breathing_machine_change_date, breathing_machine_supply_rate in self.runparam.breathing_machine_change_list:
            for i in range(breathing_machine_change_date, len(breathing_machine_supply_list)):
                breathing_machine_supply_list[i] = breathing_machine_supply_rate
                # print("breathing_machine_supply_list", breathing_machine_supply_list)

        # 由于外部传入的氧气改变的日子，顺序不确定，所以需要整理一下，并重新排序一下
        change_dates = [x[0] for x in self.runparam.oxygen_change_list] + [x[0] for x in self.runparam.breathing_machine_change_list]
        change_dates.sort()
        change_dates = np.unique(change_dates)
        # print(change_dates)
        # 治疗率=氧气参数*呼吸机参数
        cure_ability_list = oxygen_supply_list * breathing_machine_supply_list
        # print(cure_ability_list)

        return cure_ability_list, change_dates


    # 根据改变的时间，来调整改善和恶化的概率分布。
    #### 模拟周期内，医院医力多次变化，计算更新后的每日好转与恶化的概率list，需要保证总概率=1，单项概率>=0 ####
    #def intervention_hospital_supply_change_repeated(self,cure_ability_list, change_dates, better_prob_list, worse_prob_list,
    #                                                 better_prob_list_original, worse_prob_list_original, better_mean,
    #                                                 better_std, worse_mean, worse_std):
    def intervention_hospital_supply_change_repeated(self,cure_ability_list, change_dates,batchobj):
        # 先获得初步的病情改善概率和恶化概率，
        # 介入前的改善概率=正态分布*改善概率，同理，介入前恶化概率=正态分布*恶化概率。
        better_bef_date_prob = (stats.norm.cdf(change_dates[0], self.modelparam.better_mean, self.modelparam.better_std)) * self.modelparam.better_prob
        worse_bef_date_prob = (stats.norm.cdf(change_dates[0], self.modelparam.worse_mean, self.modelparam.worse_std)) * self.modelparam.worse_prob
        # print("\n")
        # 针对需要进行概率变化的时间，进行改变，
        for index, t in enumerate(change_dates):
            # print("########## change", index, "on day", t, "to cure_ability =", cure_ability_list[t], "#############\n")
            # print()
            # 新的改善率=原有的改善率*新治疗率参数（=氧气*呼吸机）
            new_better_prob = self.modelparam.better_prob * cure_ability_list[t]
            # 还是有些疑问，需要公式来说明：剩下部分的概率=剩下部分*新的改善率？
            better_after_date_prob = (1 - stats.norm.cdf(t, self.modelparam.better_mean, self.modelparam.better_std)) * new_better_prob
            # print("better_bef_date_prob = ", better_bef_date_prob)
            # print("better_after_date_prob = " , better_after_date_prob)
            # print("better_total_prob = ",  better_after_date_prob+better_bef_date_prob)

            # print("\n")
            # 按改善的概率面积+恶化的概率面积=1来计算。
            worse_after_date_prob = 1 - better_bef_date_prob - worse_bef_date_prob - better_after_date_prob
            # print("worse_bef_date_prob = ", worse_bef_date_prob)
            # print("worse_after_date_prob = " , worse_after_date_prob)
            # print("worse_total_prob = ", worse_after_date_prob+worse_bef_date_prob)

            ### 万一后期医力突然上升，结果没有这么多人来让我医治了，为了避免恶化率变成负数，设恶化率为0，然后反推实际可以有的治愈率。###
            if worse_after_date_prob < 0:
                worse_after_date_prob = 0
                better_after_date_prob = 1 - better_bef_date_prob - worse_bef_date_prob
                new_better_prob = better_after_date_prob / (1 - stats.norm.cdf(t, self.modelparam.better_mean, self.modelparam.better_std))

            # print("\n")

            new_better_prob_ratio = new_better_prob / self.modelparam.better_prob
            # print("new_better_prob_ratio = ", new_better_prob_ratio)
            new_worse_prob = worse_after_date_prob / (1 - stats.norm.cdf(t, self.modelparam.worse_mean, self.modelparam.worse_std))
            new_worse_prob_ratio = new_worse_prob / self.modelparam.worse_prob
            # print("new_worse_prob_ratio = ", new_worse_prob_ratio)
            # print("new_better_prob = ", new_better_prob)
            # print("new_worse_prob = ", new_worse_prob)

            # 使用新的好转和恶化比例系数来更新好转和恶化概率,并保存到batchobj中。
            self.intervention_hospital_supply_change_step(t, new_better_prob_ratio,new_worse_prob_ratio,batchobj)
            # The ratio 1.4, 0.2 are arbitrary for now and will be calculated based on breathing machine and oxygen supplies.
            # print("better_prob_list = ", better_prob_list)
            # print("worse_prob_list = ", worse_prob_list)
            # 这边做什么作用的，不清楚，
            if index + 1 < len(change_dates):
                better_bef_date_prob += (stats.norm.cdf(change_dates[index + 1], self.modelparam.better_mean,self.modelparam.better_std) - stats.norm.cdf(t,self.modelparam.better_mean,self.modelparam.better_std)) * new_better_prob
                worse_bef_date_prob += (stats.norm.cdf(change_dates[index + 1], self.modelparam.worse_mean, self.modelparam.worse_std) - stats.norm.cdf(t,self.modelparam.worse_mean,self.modelparam.worse_std)) * new_worse_prob

                # print("\n")

        return


    ### 每日新增的同期病人自己开启一个新曲线集 ###
    def initialise_batch(self,days, batch_population):

        batchobj=HospitalBatch()
        batchobj.batch_list = np.ones(days) * batch_population
        # print("batch_list = ", batch_list)
        batchobj.better_prob_list = batchobj.batch_list * self.modelparam.better_prob
        batchobj.better_prob_list_original = copy.copy(batchobj.better_prob_list)
        # print("better_prob_list = ", better_prob_list)
        batchobj.worse_prob_list = batchobj.batch_list * self.modelparam.worse_prob
        batchobj.worse_prob_list_original = copy.copy(batchobj.worse_prob_list)
        # print("worse_prob_list = ", worse_prob_list)
        # cure_ability_list=np.ones(days) # dummy, so that the final plotting function doesn't complain.
        #return batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original
        return batchobj

    ### 让医力的变化反应在不同批次新增病人的正确病程日上 ###
    def shift_cure_ability_list(self,cure_ability_list_original, change_dates_original, batch_counter):
        change_dates = [x - batch_counter for x in change_dates_original]
        change_dates = [i for i in change_dates if i >= 0]  # keep only non-zero change_dates
        head = cure_ability_list_original[batch_counter:]  # 截取 batch counter 日期之后的医力list
        tail = [head[-1]] * batch_counter  # 为了让list跟原来一样长， 把最后一位数字复制几遍，补全尾巴
        cure_ability_list = np.concatenate((head, tail))
        if len(change_dates) == 0:
            change_dates = [0]
        return cure_ability_list, change_dates


    ### 代入原始参数，连续跑几天几批病人，输出为模拟最后一日各批病人好转和恶化的总和 ###
    #def run_hospital(self,days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population,
    #                 number_of_batches, oxygen_change_list, breathing_machine_change_list):
    def run_hospital(self):
        ### 准备医力变化总体曲线 ###
        days=self.modelparam.days
        cure_ability_list = np.ones(days)
        change_dates = [0]
        breathing_machine_supply_list = np.ones(days)
        oxygen_supply_list = np.ones(days)
        # 根据呼吸机供应和氧气供应参数，以及已经设置好的参数变化时间，计算出治疗率
        cure_ability_list, change_dates =self.cal_cure_ability(breathing_machine_supply_list, oxygen_supply_list)
        cure_ability_list_original = copy.copy(cure_ability_list)
        change_dates_original = copy.copy(change_dates)
        ### 准备 ###
        batch_counter = 0
        better_dict = {}
        worse_dict = {}
        better_cumu_dict = {}
        worse_cumu_dict = {}
        better_final_list = np.zeros(days)
        worse_final_list = np.zeros(days)
        better_final_cumu_list = np.zeros(days)
        worse_final_cumu_list = np.zeros(days)
        empty_bed = self.modelparam.batch_population
        daily_total_better = 0  # sum up total bed release for the day from all batches of patients
        daily_total_worse = 0

        ### 画图 ###
        fig, axs = plt.subplots(nrows=number_of_batches + 1, sharex=True)
        plt.xlim(0, days - 1)
        for ax in axs:
            # for batch_counter in range(number_of_batches): #如果后来决定不用每次都画图，可以用这一行来带for loop，然后下面turn off画图的if else部分。
            ### last row of subplot 计算多期病人曲线总和，画在最下面的一个画框里 ###
            if batch_counter == number_of_batches:

                better_final_list = np.zeros(days)
                worse_final_list = np.zeros(days)
                better_final_list = np.zeros(days)
                worse_final_list = np.zeros(days)
                for j in range(1, days):
                    daily_total_better = 0
                    daily_total_worse = 0
                    for i in range(0, batch_counter):  # range()出的list不带最后一个值，所以不用加1
                        daily_total_worse += worse_dict[i][j - i]
                        daily_total_better += better_dict[i][j - i]
                    better_final_list[j] = daily_total_better
                    worse_final_list[j] = daily_total_worse
                    better_final_cumu_list[j] = sum(better_final_list)
                    worse_final_cumu_list[j] = sum(worse_final_list)
                dates = list(range(days))
                ax.plot(dates, better_final_list, color='grey', marker='o', markersize=3, linestyle='--',
                        label='Better_sum_daily(mean=10,std=5)')
                ax.plot(dates, worse_final_list, color='red', marker='o', markersize=3, linestyle='--',
                        label='Worse_sum_daily(mean=7,std=3)')
                ax.plot(dates, better_final_cumu_list, color='grey', marker='o', markersize=3, linestyle='-',
                        label='Better_sum_cumu(mean=10,std=5)')
                ax.plot(dates, worse_final_cumu_list, color='red', marker='o', markersize=3, linestyle='-',
                        label='Worse_sum_cumu(mean=7,std=3)')
                ax.set_ylabel("Total")
                ax.set_xlabel("Dates")
            else:  ### all but the last row of subplot 每日病人曲线集，按天从上往下画。
                batch_population = empty_bed  # 目前假设医院每天都是满员运转。考虑不满员，就在这里对比一下排队人数，然后注意算对剩下的空床，明天跟新空的床一起让排队病人进来。
                ### 启动一批病人 ###
                batchobj=self.initialise_batch(days, batch_population)
                ### 让医力的变化在正确的同一天，一起影响到不同批次、已经进入到病程不同天的病人 ###
                cure_ability_list, change_dates = self.shift_cure_ability_list(cure_ability_list_original, change_dates_original,
                                                                          batch_counter)
                self.intervention_hospital_supply_change_repeated(cure_ability_list,change_dates,batchobj)
                self.batch_curves(batchobj)
                ### 用词典记录每个批次的数据 ###
                better_dict[batch_counter] = batchobj.better_list
                worse_dict[batch_counter] = batchobj.worse_list
                better_cumu_dict[batch_counter] = batchobj.better_cumu_list
                worse_cumu_dict[batch_counter] = batchobj.worse_cumu_list
                ### 计算不同批次病人在同一天好转和恶化的总数 ###
                daily_total_better = 0  # sum up total bed release for the day from all batches of patients
                daily_total_worse = 0
                for i in range(0, batch_counter + 1):  # range()出的list不带最后一个值，所以要加1
                    daily_total_worse += worse_dict[i][batch_counter - i]
                    daily_total_better += better_dict[i][batch_counter - i]
                empty_bed = daily_total_worse + daily_total_better
                # print("第", batch_counter, "天")
                # print("今日康复", daily_total_better)
                # print("今日死亡", daily_total_worse)


                better_final_list[batch_counter] = daily_total_better
                worse_final_list[batch_counter] = daily_total_worse
                better_final_cumu_list[batch_counter] = sum(better_final_list)
                worse_final_cumu_list[batch_counter] = sum(better_final_list)

                dates = list(range(self.modelparam.days))
                dates = [x + batch_counter for x in dates]
                ax.plot(dates, cure_ability_list * batch_population, color='green', marker='o', markersize=3, linestyle='-',
                        label='Medical_supply')
                ax.plot(dates, batchobj.better_list, color='grey', marker='o', markersize=3, linestyle='--',
                        label='Better_daily(mean=10,std=5)')
                ax.plot(dates, batchobj.worse_list, color='red', marker='o', markersize=3, linestyle='--',
                        label='Worse_daily(mean=7,std=3)')
                ax.plot(dates, batchobj.better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-',
                        label='Better_cumu(mean=10,std=5)')
                ax.plot(dates, batchobj.worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-',
                        label='Worse_cumu(mean=7,std=3)')
                ax.plot(dates, batchobj.remain_list, color='yellow', marker='o', markersize=3, linestyle='-',
                        label='Remain population')
                ax.plot(dates, batchobj.batch_list, color='blue', marker='o', markersize=3, linestyle='-', label='Batch population')
                ax.set_ylabel("Day {}".format(batch_counter))
                batch_counter += 1

        print("第", number_of_batches, "天")  # 这个天数比batch counter要大1，就是人看着舒服点，其实第一天是在说从第0到第1天这24个小时，在代码运算里是第0天。
        print("今日康复", daily_total_better)
        print("今日死亡", daily_total_worse)
        plt.show()
        plt.close()
        return daily_total_better, daily_total_worse


    ### 单批同期病人的病情曲线，可以有很多天 ###
    #def run_single_batch(self,days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population,
    #                     number_of_batches, oxygen_change_list, breathing_machine_change_list):
    def run_single_batch(self):
        days=self.modelparam.days
        breathing_machine_supply_list = np.ones(self.modelparam.days)  # 默认实力全开
        oxygen_supply_list = np.ones(days)  # 默认实力全开
        # batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original = self.initialise_batch(
        #     days, self.modelparam.batch_population)

        batchobj=self.initialise_batch(days, self.modelparam.batch_population)
        # 根据呼吸机供应和氧气供应参数，以及已经设置好的参数变化时间，计算出治疗率
        cure_ability_list, change_dates = self.cal_cure_ability(breathing_machine_supply_list, oxygen_supply_list)
        # 根据新的治疗率，去调整原来的每天的好转率及恶化率
        self.intervention_hospital_supply_change_repeated(cure_ability_list, change_dates,batchobj)
        # 根据调整完的好转率及恶化率数据，获得每天具体的数值。
        self.batch_curves(batchobj)
        # 绘出结果曲线
        self.plot_curves(batchobj,cure_ability_list)
        #self.plot_curves(dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, cure_ability_list,
        #            batch_list)
        return


# 主程序，可供测试的范例。
if __name__ == '__main__':


    modelparam=HospitalModelParam()
    runparam=HospitalModelRunParam()
    hospitalModel = HospitalModel(modelparam,runparam)

    #### 必要参数 #########
    # for days in range(1,6): #用for loop可以让医院一天一天跑，每天把结果输出。range最低1天，0天跑不了，但可以从多天起跑。下面氧气和呼吸机供应率的更新日期最大可以是days-1日， 而且理论上应该根据每日收治的病人情况来每日更新计算。

    # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
    number_of_batches = modelparam.days  # 假设每天新增一批病人，

    ### 单批同期病人的病情曲线，可以有很多天 ###
    #hospitalModel.run_single_batch(days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population,
    #                 number_of_batches, oxygen_change_list, breathing_machine_change_list)

    #一次的测试代码
    hospitalModel.run_single_batch()

    ### 医院每天结算一次空床，每天收治一批病人 ### 目前的简化院内模型中，院内只有一种病人，要么出院，要么死亡，没有中间过渡病程。
    # daily_total_better, daily_total_worse = hospitalModel.run_hospital(days, better_prob, worse_prob, better_mean, better_std,
    #                                                     worse_mean, worse_std, batch_population, number_of_batches,
    #                                                     oxygen_change_list, breathing_machine_change_list)
    daily_total_better, daily_total_worse=hospitalModel.run_hospital()
    print("daily_total_better = ", daily_total_better)
    print("daily_total_worse = ", daily_total_worse)
    print("\n")

    ### 如果用oop，可以用当日概率掷骰子
    # probability_dict={"better": 0.2, "worse": 0.3}
    # change_state(probability_dict)
