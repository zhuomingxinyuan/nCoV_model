# -*- coding: utf-8 -*-
# 对原hospital.py代码的改造，使其更结构化，清晰可见。
import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
# import operator
import copy


# 代表模型本身的一些参数
class HospitalModelParam:
    """存放模型参数的类，方便修改整体模型的参数
    attributes：
    days:需要运行模型的天数
    worse_prob：总体的恶化概率
    better_prob：总体的改善概率
    better_mean：好转的平均时间,以天为单位
    better_std：好转的方差
    worse_mean：恶化的平均时间，以天为单位
    worse_std:恶化的方差
    batch_population：一开始运行的在院病人数量
    """
    days=20
    worse_prob = 0.1
    better_prob = 1 - worse_prob
    better_mean = 10
    better_std = 5
    worse_mean = 7
    worse_std = 3
    batch_population = 100


# 模型本身在运行时，会改变的一些参数。
class HospitalModelRunParam:
    """用来存放运行时参数的类，相对于模型参数，这部分数据比较容易变化

    attributes:
    oxygen_change_list：氧气变化的参数
    breathing_machine_change_list：呼吸机变化的参数
    cure_ability_list_original: 治疗率的原始数据
    change_dates_original: 在运行时需要保存的病历变化的时间。
    """
    # 氧气变化的参数
    oxygen_change_list = [(0, 0.3), (5, 0.5), (9, 1)]
    # (1,0.6), (9, 0.8) #tuple list, with first element = date the change starts,
    # second element=updated oxygen supply rate. List order doesn't matter
    # 呼吸机变化的参数
    breathing_machine_change_list = [(7, 0.4), (10, 1), (15, 0.5)]
    # (4,0.8)#tuple list, with first element = date the change starts,
    #  second element=updated machine supply rate. List order doesn't matter
    # 在运行时需要保存的原始治疗率对象
    cure_ability_list_original= []
    # 在运行时需要保存的病历变化的时间。
    change_dates_original= []

class HospitalBatch:
    """用来存放各批次的数据的类
    attributes:

    """
    # 存放批次信息
    batch_list=[]
    # list对象代表的是每天的数据，
    better_prob_list = []
    better_prob_list_original= []
    worse_prob_list = []
    worse_prob_list_original= []

    better_list = []  # daily increment, following normal distribution
    worse_list = []  # daily increment, following normal distribution
    # 对改善的列表的累计
    better_cumu_list = []  # cumulative record, following the shape of a cumulative density function
    # 对恶化的列表累计
    # cumulative record, following the shape of a cumulative density function
    worse_cumu_list = []
    # those who remain in the same medical status
    remain_list = []
    # 用于存放修正后的治疗率数据
    cure_ability_list =[]
    # 保存其中变化的时间。
    change_dates =[]
    batch_population =0

    # 初始化工作，每次新建时设置一下。
    def __init__(self):
        self.better_list = []
        self.worse_list = []
        self.better_cumu_list = []
        self.worse_cumu_list = []
        self.remain_list = []
        self.cure_ability_list = []
        self.change_dates = []
        return


class ModelResult:
    """作为保存模型运行的结果类

    属性：

    """
    # 保存每天汇总的好转及恶化人员。
    better_final_list=[]
    worse_final_list=[]
    # 以下两个变量暂时不用，因为要用时可以计算得出来
    # better_final_cumu_list = []
    # worse_final_cumu_list = []
    better_dict = []
    worse_dict = []
    better_cumu_dict = []
    worse_cumu_dict = []
    daily_total_betters = []
    daily_total_worses = []
    # 保存每天的空床位数=恢复健康+恶化的人
    # 准备给新一批对象使用。
    empty_beds = []
    # 保存每批次信息的对象。
    batchobjs = []

    def __init__(self, days):
        self.better_final_list = np.zeros(days)
        self.worse_final_list = np.zeros(days)
        self.better_final_list = np.zeros(days)
        self.worse_final_list = np.zeros(days)
        self.better_final_cumu_list = np.zeros(days)
        self.worse_final_cumu_list = np.zeros(days)


class HospitalModel:
    """整体的模型控制类，控制院内模型的运行及规则

    """
    def __init__(self, modelparam, runparam):
        # 用于存储参数用的两个变量
        # 模型一些固定的参数
        self.modelparam=modelparam
        # 一些比较运行时改变的参数或一些经常发生变动的参数
        self.runparam=runparam
        days = self.modelparam.days

        # 保存模型结果的对象
        self.modelResult = ModelResult(days)
        return

    # 根据参数来生成状态改变概率
    # oop中可以用来在概率之间掷骰子 ###
    @staticmethod
    def change_state(probability_dict: dict):  # probability_dict={'better': 0.2, 'worse': 0.3}
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

    # 每日新增概率 ###
    @staticmethod
    def daily_prob(day, mu, std):
        # cdf函数为累积概率，第二个参数为平均值，第三个参数为标准偏移度。
        prob = stats.norm.cdf(day + 1, mu, std) - stats.norm.cdf(day, mu, std)
        return prob

    # 累计概率 ###
    @staticmethod
    def daily_cumu(day, mu, std):
        prob = stats.norm.cdf(day + 1, mu, std)
        return prob

    # 生成默认概率曲线集 ####
    def batch_curves(self, batchobj):
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
    # 画出单批次的数据
    def plot_curves(self, batchobj):

        dates = list(range(self.modelparam.days))
        plt.plot(dates, batchobj.cure_ability_list * batchobj.batch_list[0], color='green', marker='o', markersize=1,
                 linestyle='-', label='Medical_supply')
        plt.plot(dates, batchobj.better_list, color='grey', marker='o', markersize=3, linestyle='--',
                 label='Better_daily(mean=10,std=5)')
        plt.plot(dates, batchobj.worse_list, color='red', marker='o', markersize=3, linestyle='--',
                 label='Worse_daily(mean=7,std=3)')
        plt.plot(dates, batchobj.better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-',
                 label='Better_cumu(mean=10,std=5)')
        plt.plot(dates, batchobj.worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-',
                 label='Worse_cumu(mean=7,std=3)')
        plt.plot(dates, batchobj.remain_list, color='yellow', marker='o', markersize=3, linestyle='-',
                 label='Remain population')
        plt.plot(dates, batchobj.batch_list, color='blue', marker='o', markersize=3, linestyle='-',
                 label='Batch population')
        plt.xlabel('Days')
        plt.ylabel('probability')
        plt.legend(loc='upper right')
        plt.show()
        plt.close()
        return

    # 第二个绘图程序，主要是用于总图的许多子图的绘图程序，绘完后不需要关闭。
    def plot_curves2(self, batchobjs):
        # 初始化的时间序列
        dates_original = list(range(self.modelparam.days))
        # 考虑绘图需要，将日期相对批次向后移动，
        # 画图的一些设置 ###
        fig, axs = plt.subplots(nrows=self.modelparam.days + 1, sharex=True)
        plt.xlim(0, self.modelparam.days - 1)
        batch_counter = 0

        for batchobj in  batchobjs:
            ax = axs[batch_counter]
            # dates = list(range(self.modelparam.days))
            # 为方便展示效果，将各时间根据批次向后推，
            dates = [x + batch_counter for x in dates_original]
            ax.plot(dates, batchobj.cure_ability_list * batchobj.batch_population, color='green', marker='o',
                    markersize=3, linestyle='-',label='Medical_supply')
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
            ax.plot(dates, batchobj.batch_list, color='blue', marker='o', markersize=3, linestyle='-',
                    label='Batch population')
            ax.set_ylabel("Day {}".format(batch_counter))
            batch_counter += 1
        return

    # 第三个绘图程序，主要是用于综合性结果输出。
    # 在图上绘出每日好转、恶化的结果及累积的好转及恶化结果。
    def plot_curves3(self, modelResult):
        # 可以采用以下方法新建一张图
        # figure2 = plt.figure(2)

        dates = list(range(self.modelparam.days))
        better_final_cumu_list =np.cumsum(self.modelResult.better_final_list)
        worse_final_cumu_list = np.cumsum(self.modelResult.worse_final_list)

        plt.plot(dates, modelResult.better_final_list, color='grey', marker='o', markersize=3, linestyle='--',
                 label='Better_sum_daily(mean=10,std=5)')
        plt.plot(dates, modelResult.worse_final_list, color='red', marker='o', markersize=3, linestyle='--',
                 label='Worse_sum_daily(mean=7,std=3)')
        plt.plot(dates, better_final_cumu_list, color='grey', marker='o', markersize=3, linestyle='-',
                 label='Better_sum_cumu(mean=10,std=5)')
        plt.plot(dates, worse_final_cumu_list, color='red', marker='o', markersize=3, linestyle='-',
                 label='Worse_sum_cumu(mean=7,std=3)')
        plt.ylabel("Total")
        plt.xlabel("Dates")
        plt.show()
        plt.close()

        return

    # 医院介入，拿走*个病人
    # 更新同期病人在院外的剩余人数 #####
    def intervention_take_out_patients(self,date, take_number, batch_list):
        for i in range(date, len(batch_list)):
            batch_list[i] -= take_number
        return batch_list

    # 医院介入后，改变的概率
    # 院外模型中可以在oop中使用的，同期病人部分被医院收治后，每日好转和恶化的概率更新
    def intervention_take_out_patients_change_prob_lists(self,date, remain_before_date_prob, better_prob_list, worse_prob_list):
        for i in range(date, len(better_prob_list)):
            better_prob_list[i] *= remain_before_date_prob  # 条件概率的应用，P(A|B)*P(B)= P(B|A)*P(A) = P()
        for i in range(date, len(worse_prob_list)):
            worse_prob_list[i] *= remain_before_date_prob  # 条件概率的应用
        return better_prob_list, worse_prob_list

    # 题：如果医院第10天从这批100个院外轻症患者中收治了20 个人，
    # 那么剩下的80人在第10天之后，每天多少人转重症，多少人转自愈？
    # 设：A=第10天转， B=前9天没有转
    # P( 第10天转 | 前9天没有转 ) x P(前9天没有转) = P( 前9天没有转 | 第10天转 ) x P(第10天转)
    # 因为所有第10天转的人，前9天一定没有转，所以 P( 前9天没有转 | 第10天转 ) = 1 , 所以
    # P( 第10天转 | 前9天没有转 ) x P(前9天没有转) = 1 x P(第10天转) = 想要的答案

    # 医院内的设备参数对整个概率的改变，对改善概率和恶化概率的改变
    # 针对一次医力变化，对之后的好转和恶化概率做list更新 ####
    def intervention_hospital_supply_change_step(self, date, new_better_prob_ratio, new_worse_prob_ratio,batchobj):
        # 使用原始的original数据，是因为要根据最初的状态来进行变化。
        # 而不是用可能中间已经变化的数据来算。
        for i in range(date, len(batchobj.better_prob_list)):
            batchobj.better_prob_list[i] = batchobj.better_prob_list_original[i] * new_better_prob_ratio
        for i in range(date, len(batchobj.worse_prob_list)):
            batchobj.worse_prob_list[i] = batchobj.worse_prob_list_original[i] * new_worse_prob_ratio
        return

    # 将呼吸机供应率和氧气供应率，融入到治疗率中，并返回需要变动的时间。
    # 综合每日氧气供应率和每日呼吸机供应率list，计算每日医力，以及需要更新好转和恶化率的日期 ####
    def cal_cure_ability(self):
        # 参数：breathing_machine_supply_list:呼吸机改变
        # oxygen_change_list：氧气改变参数，为一个列表，每个元素中包括两个数据，第一个为改变的时间（天），第二个为改变的供应率
        # 范例：[(1,0.6), (9, 0.8)]
        # breathing_machine_change_list :类似氧气改变参数，为呼吸机供应改变参数。

        breathing_machine_supply_list = np.ones(self.modelparam.days)
        oxygen_supply_list = np.ones(self.modelparam.days)
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

        # 将产生的结果保存到运行参数模型中。结果一个是治疗率，一个是治疗率变化的时间。
        self.runparam.cure_ability_list_original = copy.copy(cure_ability_list)
        self.runparam.change_dates_original = copy.copy(change_dates)

        return

    # 根据改变的时间，来调整改善和恶化的概率分布。
    # 模拟周期内，医院医力多次变化，计算更新后的每日好转与恶化的概率list，需要保证总概率=1，单项概率>=0 ####
    def intervention_hospital_supply_change_repeated(self,batchobj):
        # 先获得初步的病情改善概率和恶化概率，
        # 介入前的改善概率=正态分布*改善概率，同理，介入前恶化概率=正态分布*恶化概率。
        better_bef_date_prob = (stats.norm.cdf(batchobj.change_dates[0], self.modelparam.better_mean,
                                               self.modelparam.better_std)) * self.modelparam.better_prob
        worse_bef_date_prob = (stats.norm.cdf(batchobj.change_dates[0], self.modelparam.worse_mean,
                                              self.modelparam.worse_std)) * self.modelparam.worse_prob
        # print("\n")
        # 针对需要进行概率变化的时间，进行改变，
        for index, t in enumerate(batchobj.change_dates):
            # print("########## change", index, "on day", t, "to cure_ability =", cure_ability_list[t],
            # "#############\n")
            # print()
            # 新的改善率=原有的改善率*新治疗率参数（=氧气*呼吸机）
            # 变量名称为*_prob的均为面积，
            new_better_prob = self.modelparam.better_prob * batchobj.cure_ability_list[t]
            # 还是有些疑问，需要公式来说明：剩下部分的概率=剩下部分*新的改善率？
            # (1 - stats.norm.cdf(t, self.modelparam.better_mean, self.modelparam.better_std))
            # 代表总的部分减去前面的部分，剩余后面的.
            better_after_date_prob = (1 - stats.norm.cdf(t, self.modelparam.better_mean,
                                                         self.modelparam.better_std)) * new_better_prob
            # print("better_bef_date_prob = ", better_bef_date_prob)
            # print("better_after_date_prob = " , better_after_date_prob)
            # print("better_total_prob = ",  better_after_date_prob+better_bef_date_prob)

            # print("\n")

            # 按改善的概率面积+恶化的概率面积=1来计算。
            # 由于总的面积应该等于1，那么变动后的恶化的面积应该是等于1-前面好转和恶化的面积后，再减去后面好转的面积。
            worse_after_date_prob = 1 - better_bef_date_prob - worse_bef_date_prob - better_after_date_prob
            # print("worse_bef_date_prob = ", worse_bef_date_prob)
            # print("worse_after_date_prob = " , worse_after_date_prob)
            # print("worse_total_prob = ", worse_after_date_prob+worse_bef_date_prob)

            # 万一后期医力突然上升，结果没有这么多人来让我医治了，为了避免恶化率变成负数，
            # 设恶化率为0，然后反推实际可以有的治愈率。###
            if worse_after_date_prob < 0:
                worse_after_date_prob = 0
                better_after_date_prob = 1 - better_bef_date_prob - worse_bef_date_prob
                new_better_prob = better_after_date_prob / (1 - stats.norm.cdf(t, self.modelparam.better_mean,
                                                                               self.modelparam.better_std))

            # print("\n")

            new_better_prob_ratio = new_better_prob / self.modelparam.better_prob
            # print("new_better_prob_ratio = ", new_better_prob_ratio)
            # 分母为尾巴应该有的面积，在恶化率=1时的面积，
            # 把一整个正态分布，worse_after_date_prob：代表后面需要恶化的人数，
            # new_worse_prob代表情况变化后，尾巴应该是多高，代表一整个拱桥的面积，（****！）
            new_worse_prob = worse_after_date_prob / (1 - stats.norm.cdf(t, self.modelparam.worse_mean,
                                                                         self.modelparam.worse_std))
            new_worse_prob_ratio = new_worse_prob / self.modelparam.worse_prob
            # print("new_worse_prob_ratio = ", new_worse_prob_ratio)
            # print("new_better_prob = ", new_better_prob)
            # print("new_worse_prob = ", new_worse_prob)

            # 使用新的好转和恶化比例系数来更新好转和恶化概率,并保存到batchobj中。
            self.intervention_hospital_supply_change_step(t, new_better_prob_ratio,new_worse_prob_ratio,batchobj)
            # The ratio 1.4, 0.2 are arbitrary for now and will be calculated based on breathing machine and oxygen supplies.
            # print("better_prob_list = ", better_prob_list)
            # print("worse_prob_list = ", worse_prob_list)

            # 如果没有到最后一次（还需要下一轮进行运行），则需要更新一下新的好转和恶化总面积，给下一轮运行提供参数。
            # 需要算出从这次干预到下次干预的变化的面积，好转或恶化的面积。
            # (stats.norm.cdf(change_dates[index + 1], self.modelparam.better_mean,self.modelparam.better_std)
            # 代表下一轮需要运行的面积
            if index + 1 < len(batchobj.change_dates):
                better_bef_date_prob += (stats.norm.cdf(batchobj.change_dates[index + 1], self.modelparam.better_mean,
                                                        self.modelparam.better_std) - stats.norm.cdf(t,self.modelparam.better_mean,self.modelparam.better_std)) * new_better_prob
                worse_bef_date_prob += (stats.norm.cdf(batchobj.change_dates[index + 1], self.modelparam.worse_mean,
                                                       self.modelparam.worse_std) - stats.norm.cdf(t,self.modelparam.worse_mean,self.modelparam.worse_std)) * new_worse_prob

                # print("\n")

        return

    # 每日新增的同期病人自己开启一个新曲线集 ###
    def initialise_batch(self, days, batch_population):
        # 根据人数及原安排的参数生成每一天的相应情况。
        batchobj=HospitalBatch()
        batchobj.batch_population=batch_population
        batchobj.batch_list = np.ones(days) * batch_population
        # print("batch_list = ", batch_list)
        batchobj.better_prob_list = batchobj.batch_list * self.modelparam.better_prob
        batchobj.better_prob_list_original = copy.copy(batchobj.better_prob_list)
        # print("better_prob_list = ", better_prob_list)
        batchobj.worse_prob_list = batchobj.batch_list * self.modelparam.worse_prob
        batchobj.worse_prob_list_original = copy.copy(batchobj.worse_prob_list)
        # print("worse_prob_list = ", worse_prob_list)
        # cure_ability_list=np.ones(days) # dummy, so that the final plotting function doesn't complain.
        # return batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original
        return batchobj

    # 让医力的变化反应在不同批次新增病人的正确病程日上 ###
    def shift_cure_ability_list(self,batch_counter, batchobj):
        # batch_counter 代表第几批次
        # 把改变的日期按照批次向后延伸，因为对于第1批次的第5天，相当于第2批次的第四天。
        change_dates = [x - batch_counter for x in self.runparam.change_dates_original]
        # 过滤出只>0的日期，（这是相对的），范例：在第1批次的第2天改变，对于第3批次是没有意义的，因为不相关。
        change_dates = [i for i in change_dates if i >= 0]  # keep only non-zero change_dates
        # 新的数据=原始数据的后面部分，
        # 原来新的变化过程的变动已经体现在cure_ability_list_original里面，因为原来这个已经把氧气和呼吸机整合进去。
        # 所以变化只需要将cure_ability_list_original,进行部分截取的变化，产生对于某批次的新的医力变化过程。

        head = self.runparam.cure_ability_list_original[batch_counter:]  # 截取 batch counter 日期之后的医力list
        tail = [head[-1]] * batch_counter  # 为了让list跟原来一样长， 把最后一位数字复制几遍，补全尾巴
        cure_ability_list = np.concatenate((head, tail))
        if len(change_dates) == 0:
            change_dates = [0]
        # 将改变的结果保存到批次对象中。
        batchobj.cure_ability_list=cure_ability_list
        batchobj.change_dates=change_dates

        return

    def acceptpatient(self, dayofsimulate):
        """接收从院外模型转过来的数据，并返回接收人员的数量
        :param dayofsimulate: 模拟的第几天
        :return: acceptPatientnum:接收几个病人
        """
        # TODO:准备写接收病人的逻辑，及调用后续模拟进程的内容
        # 参数：dayofSimulate:
        # 以下是后期准备用的参数。
        # perDaydata:某日的病人数据。
        # numofmildaccept= 0
        # numofsevereaccept= 0
        # TODO：还需要根据数量来决定
        # 如果是第一天，则接收病人为模型中的人员参数
        # 如果非第一天，则用前一天空出来的病床数作为接收病人数。
        if dayofsimulate == 0:
            acceptPatientnum=self.modelparam.batch_population
        else:
            acceptPatientnum=self.modelResult.empty_beds[dayofsimulate-1]

        return acceptPatientnum

    def start(self):
        """模型启动，进行一些变量初始化工作
        :return: 不返回数据
        """
        # 准备医力变化总体曲线 ###
        days = self.modelparam.days

        # 一些初始化工作
        # cure_ability_list = np.ones(days)
        # change_dates = [0]

        # 根据呼吸机供应和氧气供应参数，以及已经设置好的参数变化时间，计算出治疗率,
        # 这个治疗率是对于所有批次是一样的，即时间是相对于程序开始模拟的日期的，
        # 而对于不同批次，在批次中的日期可能是不同的。
        self.cal_cure_ability()

        ### 准备 ###
        empty_bed = self.modelparam.batch_population
        daily_total_better = 0  # sum up total bed release for the day from all batches of patients
        daily_total_worse = 0

        # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
        # number_of_batches = days  # 假设每天新增一批病人，总的批次=多少天数。

        # 保存批次对象信息的对象初始化。
        self.modelResult.batchobjs = []

        return

    def calbatch(self,batch_counter,acceptPatientnum):
        """根据接收到的病人及第几天，对第几批数据进行计算
        :param batch_counter: 第几天
        :param acceptPatientnum: 接收的病人数量
        :return: 不返回数据
        """
        modelResult=self.modelResult
        # TODO：这边还需要根据病人程度来进行分组计算。
        batch_population=acceptPatientnum
        # batch_population = empty_bed
        print("new batch number is :" + str(batch_population))
        # 启动一批新的病人 ###
        batchobj = self.initialise_batch(self.modelparam.days, batch_population)
        # 让医力的变化在正确的同一天，一起影响到不同批次、已经进入到病程不同天的病人 ###
        self.shift_cure_ability_list(batch_counter, batchobj)
        # 根据改变的时间，来调整改善和恶化的概率分布。
        self.intervention_hospital_supply_change_repeated(batchobj)
        # 根据调整后的概率分布，来产生具体的数值。
        self.batch_curves(batchobj)

        ### 用词典记录每个批次的数据 ###
        modelResult.better_dict.append(batchobj.better_list)
        modelResult.worse_dict.append(batchobj.worse_list)
        modelResult.better_cumu_dict.append(batchobj.better_cumu_list)
        modelResult.worse_cumu_dict.append(batchobj.worse_cumu_list)

        # 计算不同批次病人在同一天好转和恶化的总数 ###
        # sum up total bed release for the day from all batches of patients
        daily_total_better = 0
        daily_total_worse = 0
        # range()出的list不带最后一个值，所以要加1
        for i in range(0, batch_counter + 1):
            daily_total_worse += modelResult.worse_dict[i][batch_counter - i]
            daily_total_better += modelResult.better_dict[i][batch_counter - i]

        # 算出今日的床位（？）
        empty_bed = daily_total_worse + daily_total_better

        # 把今日结果保存到模型结果集对象中
        modelResult.daily_total_betters.append(daily_total_better)
        modelResult.daily_total_worses.append(daily_total_worse)
        modelResult.empty_beds.append(empty_bed)

        # print("第", batch_counter, "天")
        # print("今日康复", daily_total_better)
        # print("今日死亡", daily_total_worse)

        modelResult.better_final_list[batch_counter] = daily_total_better
        modelResult.worse_final_list[batch_counter] = daily_total_worse

        # 将本批次的结果对象batchobj保存到整个模型结果对象中。
        modelResult.batchobjs.append(batchobj)

        return

    # 输出模型的结果
    def outputResult(self):

        # 绘出每个批次的数据，绘在同个图上
        self.plot_curves2(self.modelResult.batchobjs)
        # 显示批次的图。
        plt.show()
        # 有需要时进行清除
        # plt.clf()

        # last row of subplot 计算多期病人曲线总和，画在最下面的一个画框里 ###
        # TODO：需要改造成更高效的写法，不一定要用循环
        # 获得结果中的批次对象的数量
        batchcount = len(self.modelResult.batchobjs)

        for j in range(1, self.modelparam.days):
            daily_total_better = 0
            daily_total_worse = 0
            for i in range(0, batchcount):  # range()出的list不带最后一个值，所以不用加1
                daily_total_worse += self.modelResult.worse_dict[i][j - i]
                daily_total_better += self.modelResult.better_dict[i][j - i]
            self.modelResult.better_final_list[j] = daily_total_better
            self.modelResult.worse_final_list[j] = daily_total_worse


        # 需要时调用绘图程序，绘出总的结果数据
        self.plot_curves3(self.modelResult)

        # 输出最后一天的数据。
        # 这个天数比batch counter要大1，就是人看着舒服点，其实第一天是在说从第0到第1天这24个小时，在代码运算里是第0天。
        # print("第", number_of_batches, "天")
        # print("今日康复", daily_total_better)
        # print("今日死亡", daily_total_worse)

        return self.modelResult

    # 代入原始参数，连续跑几天几批病人，输出为模拟最后一日各批病人好转和恶化的总和 ###
    def run_hospital(self):
        # TODO：需要重新改造，应该主要工作放在初始化数据上
        # TODO：然后再根据外界模型输入病人数再进行下一步工作。
        #for batch_counter in range(number_of_batches):

            #如果后来决定不用每次都画图，可以用这一行来带for loop，
            # 然后下面turn off画图的if else部分。

            ### all but the last row of subplot 每日病人曲线集，按天从上往下画。
            # 目前假设医院每天都是满员运转。考虑不满员，就在这里对比一下排队人数，
            # 然后注意算对剩下的空床，明天跟新空的床一起让排队病人进来。
            #
            # TODO: 这边需要改为从外界接收病人。
            # TODO: 暂时还是使用空病床数代表新增的人员。

        return


    # 单批同期病人的病情曲线，可以有很多天 ###
    def run_single_batch(self):
        days=self.modelparam.days
        # breathing_machine_supply_list = np.ones(self.modelparam.days)  # 默认实力全开
        # oxygen_supply_list = np.ones(days)  # 默认实力全开

        # 创建初始化的批次数据。
        batchobj=self.initialise_batch(days, self.modelparam.batch_population)
        # 根据呼吸机供应和氧气供应参数，以及已经设置好的参数变化时间，计算出治疗率
        self.cal_cure_ability()
        # 根据新的治疗率，去调整原来的每天的好转率及恶化率
        self.intervention_hospital_supply_change_repeated(batchobj)
        # 根据调整完的好转率及恶化率数据，获得每天具体的数值。
        self.batch_curves(batchobj)
        # 绘出结果曲线
        self.plot_curves(batchobj)

        return


# 此处开始已经在类的外部，写一些测试模型及主控函数
# 测试模型
def testmodel():
    # 测试模型，后期应转为放在院外模型调用。
    # 应该是先建立一个模型框架，然后模拟，不断从外面输入病人，来模拟结果。
    # 最后输出结果
    modelparam = HospitalModelParam()
    runparam = HospitalModelRunParam()
    hospitalModel = HospitalModel(modelparam, runparam)
    # 模型进行初始化
    hospitalModel.start()

    number_of_batches=modelparam.days
    # 模拟多期病人的处理，真实情形应该是从院外每天调用，
    # 开始进入接收病人的处理。
    for batch_counter in range(number_of_batches):
        # TODO：还需要针对院外模型进行接入。
        # 初步测试，还没有从院外模型接入参数，等各病程转换比较清楚后再接入
        acceptPatientnum=hospitalModel.acceptpatient(batch_counter)
        hospitalModel.calbatch(batch_counter,acceptPatientnum)

    # 输出模型结果，实际就是汇总及出图。
    hospitalModel.outputResult()
    # print ("test finished")

    return


def main():

    return

# 主程序，可供测试的范例。
if __name__ == '__main__':
    # 现改为使用测试模型来进行模拟。
    testmodel()

    #### 必要参数 #########
    # for days in range(1,6): #用for loop可以让医院一天一天跑，每天把结果输出。range最低1天，0天跑不了，但可以从多天起跑。下面氧气和呼吸机供应率的更新日期最大可以是days-1日， 而且理论上应该根据每日收治的病人情况来每日更新计算。


    ### 单批同期病人的病情曲线，可以有很多天 ###

    #只运行一批次的测试代码
    # hospitalModel.run_single_batch()

    ### 医院每天结算一次空床，每天收治一批病人
    ### 目前的简化院内模型中，院内只有一种病人，要么出院，要么死亡，没有中间过渡病程。

   #运行多批次的测试代码，暂时不考虑输入病人，而以每天空的病床数作为新的病人添加进来
    # modelResult=hospitalModel.run_hospital()

    # 测试性的输出结果。没有太大意义。
    # print("daily_total_better = ", modelResult.daily_total_betters[-1])
    # print("daily_total_worse = ", modelResult.daily_total_worses[-1])
    # print("\n")

    ### 如果用oop，可以用当日概率掷骰子
    # probability_dict={"better": 0.2, "worse": 0.3}
    # change_state(probability_dict)

    print("the program is FINISHED.")
