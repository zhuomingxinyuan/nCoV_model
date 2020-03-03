# -*- coding: utf-8 -*-
import numpy as np
#import random
import matplotlib.pyplot as plt
from enum import Enum
#从同目录中引入SIModel对象类。
from . import SIModel
#import src.SIModel as SIModel

#import pandas as pd
import numpy.matlib
from pandas import DataFrame
#import time

# 用于模拟疫情变化的文件
# 输入参数：
TotalPopulation = 1000000
# 开始重症人数
StartSevere = 100
# 开始轻症人数
StartMild = 900
# 传染率默认参数，Basic reproduction number
Ro = 0.6
DayofStimulate = 30

EpidemicStartParam = {
    "TotalPopulation": TotalPopulation,
    "StartSevere": StartSevere,
    "StartMild": StartMild,
    "Ro": Ro,
    "DayofStimulate": DayofStimulate
}
EpidemicOutputParam = {}


#代表每天保存的数据对象。
class PerDayData:

    # 总人数，表示人数，
    # 死亡人数包括在内。
    Population = 0
    # 未感染人群
    UnInfectMan = 0
    # 无症状人数
    AsymMan=0
    # 重症人数
    SevereMan = 0
    # 轻症人数
    MildMan = 0
    # 痊愈人数：
    RecoveryMan = 0
    # 院内痊愈人数：
    RecoveryHosMan = 0
    # 院外痊愈人数：
    # RecoveryOutsideMan = 0
    # 死亡人数：
    DeadMan = 0
    #暂时未使用的变量
    DeadHosMan = 0
    #DeadOutsideMan = 0
    # 医院内收走轻症和重症人数
    SevereHosMan = 0
    MildHosMan = 0

    # 计算未感染人口数
    def CalUnInfectMan(self, accumuDead, accumuRecovery,accumSevereHosMan,accuMildHosMan):
        # 传入累积的死亡数、恢复健康者，
        #未感染人为总人口，减去死亡人口，恢复人口，减去医院收治的轻，重症，
        # 再减去重症人口及轻症人口。
        self.UnInfectMan = self.Population - accumuDead - accumuRecovery -accumSevereHosMan-accuMildHosMan- self.SevereMan - self.MildMan-self.AsymMan
        return self.UnInfectMan

    # 增加新感染者，默认情况下均为轻症患者。
    def AddNewInfectMan(self, newInfectMan):
        #参数newInfectMan:表示新感染人的数目，
        #由于定义的规则，每天计算过全部批次人员的状态变化后，才计算新增加的患者。
        #新增加的患者，为创建新一批数据后，还需要更新本日的统计人数。因此使用此方法进行更新。
        # 轻症患者增加，未感染人减少。
        self.MildMan += newInfectMan
        self.UnInfectMan -= newInfectMan
        return

    # 根据医院数据，修改本日数据，暂时没有调用到，
    #还未测试功能。
    def ChangByHospital(self, deadHosMan, recoveryHosMan):
        self.DeadHosMan = deadHosMan
        self.RecoveryHosMan = recoveryHosMan
        # 先设置院外死亡人数为原死亡人数
        self.DeadOutsideMan = self.DeadMan
        # 死亡人数累加上医院死亡人数
        self.DeadMan += deadHosMan
        # 先设置院外恢复人数为原恢复人数
        self.RecoveryOutsideMan = self.RecoveryMan
        # 恢复人数累加上医院恢复人数
        self.RecoveryMan += recoveryHosMan


#医院每日接收对象
class HospitalAcceptObj:
    #重症对象，
    SevereMan=0
    #轻症对象
    MildMan=0
    def __init__(self, SevereMan,MildMan):
        self.SevereMan=SevereMan
        self.MildMan=MildMan

#代表状态
class IllManStatus(Enum):
    ASYM=1 #无症状
    MILD=2 #轻症
    SEVERE=3 #重症
    DEAD=4 #死亡
    RECOVERY=5 #痊愈
    HOSPITAL=6 #送进医院了。


# 每批次对象的
class PerBatchData:
    # 总人数
    Population = 0
    #需要模拟的天数
    simulateDays=0
    # 无症状人数
    #AsymptomMan = 0
    # 重症人数
    #SevereMan = 0
    # 轻症人数
    #MildMan = 0
    # 痊愈人数：
    #RecoveryMan = 0
    # 院内痊愈人数：

    # 病程系数对象列表,存放每日病程系数。
    # 依次存放：死亡、恢复、重症、轻症
    # 各项内容均为列表对象array.
    # 分别关键字为：Dead,Recovery,Severe,Mild,Asym
    DiseaseCourse = {}

    #病人对象数据存储对象
    IllManData=DataFrame()
    #
    # 痊愈系数
    # RecoveryCourse=[]
    # 转重症
    # 转死亡

    # 病程第几天，还没怎么使用，暂时停用
    # 随着其移动，获得每日的病程。
    NowDayIndex = 0
    # 病程有没有被更改
    Modify = False

    # 产生某一批次人员的病程,这种方法是一开始就创建整个病程。
    # 比较适用于没有医院介入的情况。
    def CreateDiseaseCourse(self,courseParam,DayofStimulate):
        # perBatchData:每一批次的初始化数据
        # courseseParam:病程参数,应可能根据需要进行调整

        # 模拟时间，考虑从无症，转轻症，再转重症，转死亡，病程延长，所以会把这个时间向后拓展，
        # 先假定一个比较大的时间段
        self.simulateDays = DayofStimulate + 30

        #timetest=[]

        # perBatchData.MildMan = perBatchData.population
        # ——实际计算过程---

        # 创建病程后，存放在病程列表中。
        # 根据需要，产生**个人，然后进行分布：
        # 假设一开始均为无症状人员，后面再部分变成轻症人员，然后再生成重症人员及恢复人员

        # 一直无症状或无症状转健康的人员的产生，根据参数比率来产生
        asymptomNum = int(self.Population * courseParam["AsymptomPercent"])
        # 整个期间内，需要从无症状转轻症的人员
        asym2mildNum = self.Population - asymptomNum

        # 定义：长久无症状最终将转为健康状态。
        # 产生无症状人员，转换为健康状态的分布。
        asym2RecoveryNormal = np.random.normal(courseParam["AverAsym2Recovery"],
                                               courseParam["varianceAsym2Recovery"],
                                               asymptomNum)
        asym2RecoveryCountPerDayNP = np.histogram(asym2RecoveryNormal, bins=range(self.simulateDays + 1))[0]

        # 产生无症状转轻症的在第几天数量的正态分布
        asym2mildNormal = np.random.normal(courseParam["AverAsym2Mild"], courseParam["varianceAsym2Mild"],
                                           asym2mildNum)
        # 统计直方图，获得第几天有几个无症状转轻症。
        asym2mildCountPerDayNP = np.histogram(asym2mildNormal, bins=range(self.simulateDays + 1))[0]
        # 总人口减去每天汇总的从无症状转轻症的人,以及无症状转健康的人汇总，得到每天无症状总人数
        #asymCountPerDayNP = perBatchData.Population - np.cumsum(asym2mildCountPerDayNP) - np.cumsum(asym2RecoveryCountPerDayNP)

        #存放轻症转重症的人的分布对象。
        mild2SevereNormal = np.array([])
        #存放由轻症转健康的人分布
        recoveryNormal = np.array([])
        #存放死亡分布。
        deadNormal = np.array([])

        # 产生存放病人状态的矩阵。
        # 默认使用1作为初始值，即表示无症状人员
        data = np.matlib.ones((self.Population, self.simulateDays))
        df = DataFrame(data)
        df['A2M']=0
        df['A2R']=0
        df['M2S']=0
        df['M2R']=0
        df["S2D"]=0
        df["id"]=df.index

        # i 代表整批病人病程的第几日
        i = 0

        for i in range(self.simulateDays):
            #获得当日无症转轻症的人数。
            tempasym2mildNum=asym2mildCountPerDayNP[i]
            # 如果某日人数>0,即有转为轻症的人，则计算，否则跳过
            if (tempasym2mildNum > 0):
                #print("start"+str(time.clock()))
                #从数据中无症状的人，且还没转化过轻症的人中随机取样。
                dfA2M=df[(df[i] == IllManStatus.ASYM.value) & (df['A2M']==0) ].sample(tempasym2mildNum)
                #iloc使用索引，loc使用列的名称来定位，两者不同。
                #将第i个位置开始后面的值都变成轻症的值。并在行最后字段定义表示已经转化过。
                df.iloc[dfA2M.index, i:self.simulateDays] = IllManStatus.MILD.value
                df.loc[dfA2M.index, 'A2M']=1

                #print("mild" + str(time.clock()))
                # 轻症转重症率：
                # 获得上面这批无症状转轻症的人，在以后转重症的数量
                mild2SevereNum = int(tempasym2mildNum * courseParam["Mild2SeverePercent"])
                # 产生轻症转重症的在接下来几天数量的正态分布，i表示向后推几天，病程是从今日开始向后推的。
                mild2SevereNormalTemps = i + np.random.normal(courseParam["AverMild2SevereDay"],
                                                              courseParam["varianceMild2SevereDay"],
                                                              mild2SevereNum)
                #从这批人中抽样出以后转重症的病人，
                dfM2S=dfA2M.sample(mild2SevereNum)

                #使用参数，产生这批转重症的病人何时死亡
                #直接将结果转为整数，获得哪一天。
                deadDayTemps = np.random.normal(courseParam["AverDied"], courseParam["varianceDied"], mild2SevereNum).astype(np.int)
                #获得其每个病人的索引号
                tempindexs = dfM2S.index
                tempindex=0

                #print(time.clock())

                for tempday in mild2SevereNormalTemps:
                    #设置转为重症的病人的值。
                    df.iloc[tempindexs[tempindex],int(tempday):self.simulateDays]=IllManStatus.SEVERE.value
                    #由于假设院外的病人重症后都会死亡。所以用这批转重症的人来产生死亡人员，
                    #每次只抽一个人，产生它死亡在第几日。

                    #deadNormal=np.append(deadNormal, deadDayTemps[tempindex])
                    #将某人在deadDayTemp设置为死亡状态。
                    df.iloc[tempindexs[tempindex],int(tempday)+deadDayTemps[tempindex]:self.simulateDays]=IllManStatus.DEAD.value

                    tempindex=tempindex+1

                df.loc[tempindexs, 'M2S'] = 1
                df.loc[tempindexs, 'S2D'] = 1

                #print(time.clock())
                #print("severe+dead:"+str(time.clock()))
                #timetest.append(time.clock())

                # 汇总到一个列表中。
                mild2SevereNormal = np.append(mild2SevereNormal, mild2SevereNormalTemps)
                # 计算轻症变健康的人数
                # 轻症变健康的人数应该=轻症人数-变重症人数
                recoveryNum = tempasym2mildNum - mild2SevereNum
                # 产生轻症转痊愈的在第几天数量的正态分布,i表示向后推几天
                recoveryNormalTemps = i + np.random.normal(courseParam["AverMild2RecoveryDay"],
                                                           courseParam["varianceMild2RecoveryDay"],
                                                           recoveryNum)

                # 从刚转为轻症的这批人中获得以后要转健康的人。即剩下没有转轻症的人。
                # 从轻症病人库中剔除掉会变成重症的那批人，剩下的即为转为健康的人。
                dfM2R= dfA2M.iloc[~dfA2M.index.isin(dfM2S.index)]
                #print("recoverystart:" + str(time.clock()))
                #临时变量，代表索引值，因为dfM2R.index代表其索引号，为一个列表，
                # 因此需要从中一个个取出其索引号。
                tempindexs = dfM2R.index
                tempindex=0
                #根据其变健康的日程安排，将病人在某日的状态改为健康。
                for tempday in recoveryNormalTemps:
                    df.iloc[tempindexs[tempindex], int(tempday):self.simulateDays ] = IllManStatus.RECOVERY.value
                    tempindex = tempindex + 1

                df.loc[tempindexs, 'M2R'] = 1
                    # 汇总到一个列表中。
                recoveryNormal = np.append(recoveryNormal, recoveryNormalTemps)
                #print("recovery:" + str(time.clock()))
            #计算出今日无症转健康的人，并安排到数据中。
            if asym2RecoveryCountPerDayNP[i]>0:
                tempasym2RecoveryNum=asym2RecoveryCountPerDayNP[i]
                # 从数据中无症状的人，且还没转化过轻症的人中随机取样。
                dfA2R = df[(df[i] == IllManStatus.ASYM.value) & (df['A2R'] == 0)].sample(tempasym2RecoveryNum)
                df.iloc[dfA2R.index, i:self.simulateDays ] = IllManStatus.RECOVERY.value
                df.loc[dfA2R.index, 'A2R'] = 1

            #print("recovery2:" + str(time.clock()))
            #i = i + 1

        self.IllManData = df
        #计算一下病程结果，产生数据汇总。
        diseaseCourse=self.CalDiseaseCourse()

        #有需要时，可画出当前病程结果。
        #self.DrawBatchData()



        return diseaseCourse

    #计算一下当前病程结果，需要从每天人数中进行统计
    def CalDiseaseCourse(self):
        # 创建结果输出对象。
        # 这部分可能还需要改进，如何更高效计算。
        i = 0
        diseaseCourse = {}
        diseaseCourse['ASYM'] = []
        diseaseCourse['MILD'] = []
        diseaseCourse['SEVERE'] = []
        diseaseCourse['RECOVERY'] = []
        diseaseCourse['DEAD'] = []
        diseaseCourse['HOSPITAL'] = []
        # timetest.append(time.clock())
        # 统计每日结果
        df=self.IllManData
        for i in range(self.simulateDays):
            diseaseCourse['ASYM'].append(len(df[df[i] == IllManStatus.ASYM.value]))
            diseaseCourse['MILD'].append(len(df[df[i] == IllManStatus.MILD.value]))
            diseaseCourse['SEVERE'].append(len(df[df[i] == IllManStatus.SEVERE.value]))
            diseaseCourse['RECOVERY'].append(len(df[df[i] == IllManStatus.RECOVERY.value]))
            diseaseCourse['DEAD'].append(len(df[df[i] == IllManStatus.DEAD.value]))
            diseaseCourse['HOSPITAL'].append(len(df[df[i] == IllManStatus.HOSPITAL.value]))
        self.DiseaseCourse=diseaseCourse
        return diseaseCourse

    # 从当前批次数据中移除重症病人
    def RemoveSevereMan(self,NumofSevere2Remove,dayofWant):
        #参数：NumofSevere2Remove：需要移除的重症病人人数
        severeManindex=self.GetIndexByDiseaseDays(dayofWant,IllManStatus.SEVERE.value)
        df = self.IllManData

        df.iloc[severeManindex.index[0:NumofSevere2Remove],dayofWant:self.simulateDays]=IllManStatus.HOSPITAL.value

        return

    # 从当前批次数据中移除轻症病人
    def RemoveMildMan(self, NumofMild2Remove,dayofWant):
        # 参数：NumofSevere2Remove：需要移除的重症病人人数
        mildManindex = self.GetIndexByDiseaseDays(dayofWant, IllManStatus.MILD.value)
        df = self.IllManData

        df.iloc[mildManindex.index[0:NumofMild2Remove], dayofWant:self.simulateDays] = IllManStatus.HOSPITAL.value
        return

    # 需要对当前病人按其病程进行排序，即按什么时候开始重症
    # 什么时候开始轻症进行排序，以方便根据程度推送医院
    # 根据病人在第几日，查某日其病人状态。
    def GetIndexByDiseaseDays(self,dayofWant,illManStatus):
        df=self.IllManData
        #先获得在某天为某种状态的病人子集
        tempdf=df.loc[df[dayofWant]==illManStatus]
        #如果已经没有符合要求的记录
        if len(tempdf)==0:
            return []
        #内部函数，准备给lambda用的，
        #根据输入的某行row，从第0天到第dayindex,计算值=value的次数
        def calcount(row, dayindex, value):
            count = 0
            for i in range(dayindex + 1):
                if row[i] == value:
                    count += 1
            # row['count']=count
            return count

        try:
            #tempdf['count']=None
            #先获得各病程的某种状态（轻或重）的天数
            tempdf.loc[:,'count'] = tempdf.apply(lambda row: calcount(row, dayofWant, illManStatus), axis=1)
            #把病程按从大到小进行排序，优先推送比较久的人。
            index=tempdf.sort_values(by='count', ascending=False)
        except Exception:
            print ("error in GetIndexByDiseaseDays")
            index=[]


        return index
    # 画图功能，画出某批次的数据。有需要时，可测试使用
    def DrawBatchData(self):

        plt.plot(self.DiseaseCourse['DEAD'], color='black', label='deadAccumu', marker='.')
        plt.plot(self.DiseaseCourse['SEVERE'], color='red', label='severe', marker='.')
        plt.plot(self.DiseaseCourse['MILD'], color='orange', label='mild', marker='.')
        plt.plot(self.DiseaseCourse['RECOVERY'], color='grey', label='recoveryAccumu', marker='.')
        plt.plot(self.DiseaseCourse['ASYM'], color='green', label='asym', marker='.')
        plt.plot(self.DiseaseCourse['HOSPITAL'], color='purple', label='hospital', marker='.')
        plt.legend()
        plt.show()


        # 画图功能，画出某批次的数据。有需要时，可测试使用。
        # plt.plot(deadCumsum, color='black', label='deadAccumu', marker='.')
        # plt.plot(severeCountPerDayNP, color='red', label='severe', marker='.')
        # plt.plot(mildCountPerDayNP, color='orange', label='mild', marker='.')
        # plt.plot(recoveryCumsum, color='grey', label='recoveryAccumu', marker='.')
        # plt.plot(asymCountPerDayNP, color='green', label='asym', marker='.')
        # plt.legend()
        # plt.show()

        #plt.clf()

        return

    #打印出病程的详细数据。
    def printResult(self):
        df = self.IllManData
        # 打印一下结果
        for i in range(len(df)):
            row = df.iloc[i].values  # 返回一个list
            print(row)
            pass
        return


# 核心的模拟对象，代表整个疾病模拟对象。
class Epidemic:
    # 开始的一些设置参数,默认为0，类初始化时由外界参数设置。
    # 输入参数：
    Population = 0
    # 开始重症人数
    StartSevere = 0
    # 开始轻症人数
    StartMild = 0
    # 开始无症人数：
    StartAsym = 900
    # 传染率默认参数，Basic reproduction number
    Ro = 0.0
    # 模拟天数
    DayofStimulate = 0

    # 生成病程的详细参数，由外界输入
    # 在生成每批次数据过程CreateBatchData中作为输入参数。
    DetailCourseParam={}



    #用于保存过程结果的一些变量
    # 累计死亡者
    AccumuDead = 0
    # 累计恢复健康者
    AccumuRecovery = 0
    # 累计医院拿走重症人数
    AccumuHosSevere = 0
    AccumuHosMild = 0

    # 批次的列表对象，存储每批次的数据
    batchsData = []
    # 保存每日的数据对象。
    daysData = []



    # 使用外界输入的参数进行一些参数设置。
    def __init__(self, param):
        self.Population = param["TotalPopulation"]
        self.StartSevere = param["StartSevere"]
        self.StartMild = param["StartMild"]
        self.StartAsym=param["StartAsym"]
        self.Ro = param["Ro"]
        self.DayofStimulate = param["DayofStimulate"]
        self.DetailCourseParam=param["DetailCourseParam"]


    # 进行模拟
    def Simulate(self):

        InputParam={}
        InputParam["Ro"]=self.Ro
        InputParam["gamma"]=1/14

        #模型对象，准备在下面使用的模拟模型。
        simpleSIModel=SIModel.SimpleSIModel(InputParam)


        for i in range(self.DayofStimulate):
            #打印输出：表示在模拟的第几天。
            print("do "+str(i)+" day simulate")

            # 每日模拟

            # 对每批次的人员进行计算,获得每日人员的动态，并得到病愈人口
            perDayData = PerDayData()
            perDayData.Population = self.Population

            # 其中要与医院进行数据交换
            # ？未完待写与医院交互数据

            # 得到医院返回对象
            # 如果是第一天，不与医院交互，即前一天没病人，先不送医院。
            if i != 0:

                # 获得前一天汇总数据
                preDayData = self.daysData[i - 1]
                #向医院发送请求，根据前一天的数据，让医院决定接收多少个人，并返回数据。
                hosAcceptObj = self.SendRequest2HosModel(i, preDayData)

                # 将医院带走的人数记录在主类中。
                # 为什么在这边就开始计算，因为后面随着每批次数据进行，
                # 医院接收病人要分到各批次数据去。
                self.AccumuHosMild += hosAcceptObj.MildMan
                self.AccumuHosSevere += hosAcceptObj.SevereMan
                # 总人数不打算减少，而是只记着医院拿走的人数以及进医院的状态。
                #self.Population -= (hosAcceptObj.MildMan + hosAcceptObj.SevereMan)

                # perDayData.ChangByHospital()
            else:
                # 第一天情况处理，因为没有批次数据,也设置医院先不接收

                # 先用传入参数中的无症人数来创建初始病人。
                perBatchData = self.CreateBatchData(self.StartAsym,self.DetailCourseParam)
                self.batchsData.append(perBatchData)
                #创建一个空的医院接收对象，表示没有接收。
                hosAcceptObj = HospitalAcceptObj(0,0)


            # 在每批次计算过程中汇总每日数据：
            for perBatchData in self.batchsData:

                # 获得各批次中相应人数，并累加到每日数据中。

                # 先移动各批次数据中的索引，表示到各批次的第几天。
                #因为第一天没变，所以i=0时，不需要变。
                if i != 0:
                    perBatchData.NowDayIndex += 1

                # 如果医院接收数都等于0，表示不接收。
                if not ((hosAcceptObj.SevereMan == 0) and (hosAcceptObj.MildMan== 0)):

                    # 根据医院数据，修改各批次人员。
                    self.ChangeBatchsByHosAcceptObj2(hosAcceptObj, perBatchData, perBatchData.NowDayIndex)


                # 将当前批次数据汇总到当日汇总数据中。
                perDayData.DeadMan += perBatchData.DiseaseCourse["DEAD"][perBatchData.NowDayIndex]
                perDayData.RecoveryMan += perBatchData.DiseaseCourse["RECOVERY"][perBatchData.NowDayIndex]
                perDayData.SevereMan += perBatchData.DiseaseCourse["SEVERE"][perBatchData.NowDayIndex]
                perDayData.MildMan += perBatchData.DiseaseCourse["MILD"][perBatchData.NowDayIndex]
                perDayData.AsymMan += perBatchData.DiseaseCourse["ASYM"][perBatchData.NowDayIndex]
            # 所有批次的数据更新后，则累积计算总死亡人口，总痊愈人口，进而计算未感染人数
            self.AccumuDead += perDayData.DeadMan
            self.AccumuRecovery += perDayData.RecoveryMan

            # 需要重新计算下未感染人数
            perDayData.CalUnInfectMan(self.AccumuDead, self.AccumuRecovery,self.AccumuHosSevere,self.AccumuHosMild)

            # 计算新感染人口。
            #newInfectMan = self.SIModel(self.Ro, perDayData)
            # 使用模型框架来计算新感染人口
            modelrunParam={}
            modelrunParam['perDayData']=perDayData
            modelResult=simpleSIModel.run(modelrunParam)
            newInfectMan=modelResult["newInfectMan"]

            # 更新当日数据
            perDayData.AddNewInfectMan(newInfectMan)
            # 保存今日数据到列表中。
            self.daysData.append(perDayData)

            # 创新新一批感染人口群，并附加到批次列表中。
            # 获得新感染人口后，需要针对新感染人口，启动新一批次人员
            newBatchData = self.CreateBatchData(newInfectMan,self.DetailCourseParam)
            self.batchsData.append(newBatchData)

        # 输出总的模型计算结果，汇总各日数据，并产生图表。
        self.OutputResult()

        return






    # 生成新的每批数据。
    # 根据多少个患者，生成每日批量数据
    def CreateBatchData(self, population,detailcourseParam):

        detailcourseParam=detailcourseParam
        # 根据病程参数，来准备生成病程

        # 默认情况下，不考虑重症输入，先全部假定为轻症患者。
        # 后期有需要时再改造。
        perBatchData = PerBatchData()
        perBatchData.Population = population
        perBatchData.CreateDiseaseCourse(detailcourseParam,self.DayofStimulate)

        return perBatchData




    # 调用院内模型，获得数据，应是每天运行的。
    def SendRequest2HosModel(self, dayofSimulation, preDayData):
        # dayofSimulation,表示模拟的第几天，好让医院根据实际情况返回数据。
        # preDayData:前一天的数据。

        # 向院内模型发送数据,需要将传进来的参数发送出去。
        # 根据院内模型返回结果来获得数据
        # 医院只决定接收多少个重症，多少个轻症。而与病程没有关系：
        # 因此决定送谁过去由院外来决定。
        # 医院返回的接收病人数不能大于发过去的病人总数。
        # 先模拟医院返回结果，每天只接收**重症，多少轻症。
        #以下为各种情况测试
        if dayofSimulation>5:
            hosAcceptObj=HospitalAcceptObj(0,50)
        else:
            hosAcceptObj = HospitalAcceptObj(0, 0)

        #一些可用的测试数据。
        hosAcceptObj = HospitalAcceptObj(0, 0)
        #hosAcceptObj=HospitalAcceptObj(random.randint(50,60),100)
        #hosAcceptObj = HospitalAcceptObj(2*dayofSimulation, 5*dayofSimulation)
        return hosAcceptObj

    # 根据医院可接收的数量，来进行每批次人员调整
    def ChangeBatchsByHosAcceptObj2(self, hosAcceptObj, perBatchData, dayofCourse):

        # 如果有减少重症病人，而且批次数据中重症病人不为0
        if (hosAcceptObj.SevereMan != 0) and (perBatchData.DiseaseCourse["SEVERE"][dayofCourse]!=0) :
            # 如果这批次现有重症<医院接受重症人数,则把医院的重症病人都给这批次病人
            if (perBatchData.DiseaseCourse["SEVERE"][dayofCourse] > hosAcceptObj.SevereMan):
                # 总人数不需要减少，使用在医院的状态表示已送到医院。
                perBatchData.RemoveSevereMan(hosAcceptObj.SevereMan, dayofCourse)
                hosAcceptObj.SevereMan = 0
            else:

                # 从医院接受重症病人中减去当前这批次重症人数，
                # 累积运行，表示只减去这批次的人，有剩余的病人留给其它批次减少。
                hosAcceptObj.SevereMan -= perBatchData.DiseaseCourse["SEVERE"][dayofCourse ]
                perBatchData.RemoveSevereMan(perBatchData.DiseaseCourse["SEVERE"][dayofCourse], dayofCourse)
                # 将今日开始重症人数全部被医院接收
                perBatchData.DiseaseCourse["SEVERE"][dayofCourse] = 0

        # 如果有减少轻症病人,而且批次数据中轻症病人不为0
        if (hosAcceptObj.MildMan != 0) and (perBatchData.DiseaseCourse["MILD"][dayofCourse]!=0):
            # 如果这批次现有轻症<医院接受轻症人数,则把医院的轻症病人都给这批次病人
            if (perBatchData.DiseaseCourse["MILD"][dayofCourse] > hosAcceptObj.MildMan):
                # 总人数不需要减少，使用在医院的状态表示已送到医院。
                perBatchData.RemoveMildMan(hosAcceptObj.MildMan, dayofCourse)
                hosAcceptObj.MildMan = 0
            else:

                # 从医院接受轻症病人中减去当前这批次轻症人数
                hosAcceptObj.MildMan -= perBatchData.DiseaseCourse["MILD"][dayofCourse]
                perBatchData.RemoveMildMan(perBatchData.DiseaseCourse["MILD"][dayofCourse], dayofCourse)

                # 将今日开始轻症人数全部被医院接收
                perBatchData.DiseaseCourse["MILD"][dayofCourse] = 0

                #SevereReductRatio=perBatchData.DiseaseCourse["Severe"]

        #在医院接收病人后，关于某批数据的统计信息，需要再重新计算一下。
        perBatchData.CalDiseaseCourse()

        return
    # 根据医院可接收的数量，来进行每批次人员调整。(此段不再调用，准备删除）
    def ChangeBatchsByHosAcceptObj(self, hosAcceptObj, perBatchData, dayofCourse):
        # hosAcceptObj:需要改变的人员数量
        # perBatchData：这批次数据
        # dayofSimulation:获得模拟日，类似索引

        # 如果有减少重症病人
        if hosAcceptObj.SevereMan != 0:
            # 如果这批次现有重症>昨天医院接受重症人数
            if (perBatchData.DiseaseCourse["Severe"][dayofCourse - 1] > hosAcceptObj.SevereMan):
                # 从库中减去多少个重症病人。
                # 今日开始时重症人数=昨日重症人数-医院接收人数
                perBatchData.DiseaseCourse["Severe"][dayofCourse] = perBatchData.DiseaseCourse["Severe"][
                                                                        dayofCourse - 1] - hosAcceptObj.SevereMan
                # 总人数要减少
                perBatchData.Population -= hosAcceptObj.SevereMan

                hosAcceptObj.SevereMan = 0

            else:
                # 从医院接受重症病人中减去当前这批次重症人数
                hosAcceptObj.SevereMan -= perBatchData.DiseaseCourse["Severe"][dayofCourse - 1]
                # 总人数要减少
                perBatchData.Population -= perBatchData.DiseaseCourse["Severe"][dayofCourse - 1]
                # 将今日开始重症人数全部被医院接收
                perBatchData.DiseaseCourse["Severe"][dayofCourse] = 0

        # 如果有减少轻症病人
        if hosAcceptObj.MildMan  != 0:
            # 如果这批次现有轻症>医院接受轻症人数
            if (perBatchData.DiseaseCourse["Mild"][dayofCourse - 1] > hosAcceptObj.MildMan ):
                # 从库中减去多少个轻症病人。
                perBatchData.DiseaseCourse["Mild"][dayofCourse] = perBatchData.DiseaseCourse["Mild"][dayofCourse - 1] - \
                                                                  hosAcceptObj.MildMan
                # 总人数要减少
                perBatchData.Population -= hosAcceptObj.MildMan

                hosAcceptObj.MildMan = 0
            else:
                # 从医院接受病人中减去当前这批次轻症人数
                hosAcceptObj.MildMan -= perBatchData.DiseaseCourse["Mild"][dayofCourse - 1]
                # 总人数要减少
                perBatchData.Population -= perBatchData.DiseaseCourse["Mild"][dayofCourse - 1]

                perBatchData.DiseaseCourse["Mild"][dayofCourse] = 0

        return

    # 使用比率的方法来计算新的病程，由于人数变化
    # 不再调用此程序，因为依赖新的算法，直接删除病人对象，不再修改病程。
    def ChangeDiseaseCoursse(self, perBatchData, dayofCourse):
        # 按新方式重新计算今日病程


        # 先计算死亡人数
        perBatchData.DiseaseCourse["Dead"][dayofCourse] = int(perBatchData.DiseaseCourse["Severe"][dayofCourse] * \
                                                              perBatchData.DiseaseCourse["DieDivideSevere"][
                                                                  dayofCourse])
        # 再计算新增重症人数,并与现有重症人数相加,再减去死亡人数。
        perBatchData.DiseaseCourse["Severe"][dayofCourse] = int(perBatchData.DiseaseCourse["Mild"][dayofCourse] * \
                                                                perBatchData.DiseaseCourse["Mild2SevereDivideMild"][
                                                                    dayofCourse]) + \
                                                            perBatchData.DiseaseCourse["Severe"][dayofCourse] - \
                                                            perBatchData.DiseaseCourse["Dead"][dayofCourse]

        # 再计算痊愈人数
        perBatchData.DiseaseCourse["Recovery"][dayofCourse] = int(perBatchData.DiseaseCourse["Mild"][dayofCourse] * \
                                                                  perBatchData.DiseaseCourse["RecoveryDivideMild"][
                                                                      dayofCourse])
        # 最后再重新计算轻症人数，为总人口-累积死亡人口-重症-累积痊愈人口
        perBatchData.DiseaseCourse["Mild"][dayofCourse] = perBatchData.Population - \
                                                          np.sum(perBatchData.DiseaseCourse["Dead"][0:dayofCourse+1]) - \
                                                          perBatchData.DiseaseCourse["Severe"][dayofCourse] - \
                                                          np.sum(perBatchData.DiseaseCourse["Recovery"][0:dayofCourse+1])
        perBatchData.Modify = True

    # 输出模拟结果
    def OutputResult(self):
        # 打算调用mapplot及pandas或numpy来完成输出。
        # 院内输出结果

        # 院外输出结果
        asymArray=[]
        deadArray = []
        severeArray = []
        mildArray = []
        recoveryArray = []
        unInfectArray = []
        hosSevereArray = []
        hosMildArray = []
        populationArray=[]
        unInfectArray=[]
        InfectArray=[]
        #对保存的每日数据，进行汇总。
        for everydayData in self.daysData:
            deadArray.append(everydayData.DeadMan)
            severeArray.append(everydayData.SevereMan)
            mildArray.append(everydayData.MildMan)
            recoveryArray.append(everydayData.RecoveryMan)
            hosSevereArray.append(everydayData.SevereHosMan)
            hosMildArray.append(everydayData.MildHosMan)
            asymArray.append(everydayData.AsymMan)
            unInfectArray.append(everydayData.UnInfectMan)
            populationArray.append(everydayData.Population)
            InfectArray.append(everydayData.MildMan+everydayData.SevereMan)
            unInfectArray.append(everydayData.UnInfectMan)

        #将上面的数据结果，使用matplotlib来进行绘图。
        #
        plt.plot(np.cumsum(deadArray), color='black', label='deadAccumu', marker='.')
        plt.plot(severeArray, color='red', label='severe', marker='.')
        plt.plot(mildArray, color='orange', label='mild', marker='.')
        plt.plot(np.cumsum(recoveryArray), color='grey', label='recoveryAccumu', marker='.')
        plt.plot(np.cumsum(hosSevereArray), color='purple', label='HosSevereAccumu', marker='.')
        plt.plot(np.cumsum(hosMildArray), color='violet', label='HosMildAccumu', marker='.')
        #使用log来画图。用于显示差别比较大的数据
        plt.semilogy(asymArray, color='green', label='asym', marker='.')
        #plt.plot(asymArray, color='green', label='asym', marker='.')
        #plt.plot(unInfectArray, color='blue', label='recovery', marker='.')
        #画出图的图例。
        plt.legend()
        plt.show()

        return
