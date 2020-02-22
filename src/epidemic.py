# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt

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
    # 要扣除医院拿走的人数，死亡人数包括在内。
    Population = 0
    # 未感染人群
    UnInfectMan = 0
    # 重症人数
    SevereMan = 0
    # 轻症人数
    MildMan = 0
    # 痊愈人数：
    RecoveryMan = 0
    # 院内痊愈人数：
    RecoveryHosMan = 0
    # 院外痊愈人数：
    RecoveryOutsideMan = 0
    # 死亡人数：
    DeadMan = 0
    DeadHosMan = 0
    DeadOutsideMan = 0
    # 医院内收走轻症和重症人数
    SevereHosMan = 0
    MildHosMan = 0

    # 计算未感染人口数
    def CalUnInfectMan(self, accumuDead, accumuRecovery):
        # 传入累积的死亡数、恢复健康者，
        #未感染人为总人口，减去死亡人口，恢复人口，再减去重症人口及轻症人口。
        self.UnInfectMan = self.Population - accumuDead - accumuRecovery - self.SevereMan - self.MildMan
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

    # 根据医院接收数据，存入本日数据
    def ChangByHospitalAccept(self, hosAcceptObj):
        #hosAcceptObj：医院本日接收对象
        self.SevereHosMan = hosAcceptObj.SevereMan
        self.MildHosMan = hosAcceptObj.MildMan
        #人数也更新，减少医院接收的病人数。
        self.Population -= (hosAcceptObj.MildMan + hosAcceptObj.SevereMan)

#医院每日接收对象
class HospitalAcceptObj:
    #重症对象，
    SevereMan=0
    #轻症对象
    MildMan=0
    def __init__(self, SevereMan,MildMan):
        self.SevereMan=SevereMan
        self.MildMan=MildMan



# 每批次对象的
class PerBatchData:
    # 总人数
    Population = 0
    # 重症人数
    SevereMan = 0
    # 轻症人数
    MildMan = 0
    # 痊愈人数：
    RecoveryMan = 0
    # 院内痊愈人数：

    # 病程系数对象列表,存放每日病程系数。
    # 依次存放：死亡、恢复、重症、轻症
    # 各项内容均为列表对象array.
    # 分别关键字为：dead,recovery,servere,mild

    DiseaseCourse = {}
    #
    # 痊愈系数
    # RecoveryCourse=[]
    # 转重症
    # 转死亡

    # 病程第几天，比较重要，
    # 随着其移动，获得每日的病程。
    NowDayIndex = 0
    # 病程有没有被更改
    Modify = False


# 核心的模拟对象
class Epidemic:
    # 开始的一些设置参数
    # 输入参数：
    Population = 0
    # 开始重症人数
    StartSevere = 0
    # 开始轻症人数
    StartMild = 0
    # 传染率默认参数，Basic reproduction number
    Ro = 0.0
    # 模拟天数
    DayofStimulate = 0
    # 重症率
    # 死亡率
    # 痊愈率

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

    # 需要设计一个结构，用于保存每日的结果

    # 使用外界输入的参数进行模拟。
    def __init__(self, param):
        self.Population = param["TotalPopulation"]
        self.StartSevere = param["StartSevere"]
        self.StartMild = param["StartMild"]
        self.Ro = param["Ro"]
        self.DayofStimulate = param["DayofStimulate"]

    # 进行模拟
    def Simulate(self):

        for i in range(self.DayofStimulate):
            # print("do sth simulate")

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
                self.AccumuHosMild += hosAcceptObj.MildMan
                self.AccumuHosSevere += hosAcceptObj.SevereMan
                self.Population -= (hosAcceptObj.MildMan + hosAcceptObj.SevereMan)

                # perDayData.ChangByHospital()
            else:
                # 第一天情况处理，因为没有批次数据,也设置医院先不接收

                # 先用传入参数中的轻症人数来创建初始轻症病人。
                perBatchData = self.CreateBatchData(self.StartMild)
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
                if (hosAcceptObj.SevereMan == 0) and (hosAcceptObj.MildMan== 0):

                    # 病程没有变的话。因为如果前几天医院有收过这批人，则病程每天都需要重新计算。
                    # 如果不需要改变，按原来方式进行,不需要修改，因为已经存在批次数据中。
                    if (perBatchData.Modify == True):

                        self.ChangeDiseaseCoursse(perBatchData, perBatchData.NowDayIndex)
                else:
                    # 根据医院数据，修改各批次人员。
                    self.ChangeBatchsByHosAcceptObj(hosAcceptObj, perBatchData, perBatchData.NowDayIndex)
                    # 改变当日病程
                    self.ChangeDiseaseCoursse(perBatchData, perBatchData.NowDayIndex)

                # 将当前批次数据汇总到当日汇总数据中。
                perDayData.DeadMan += perBatchData.DiseaseCourse["Dead"][perBatchData.NowDayIndex]
                perDayData.RecoveryMan += perBatchData.DiseaseCourse["Recovery"][perBatchData.NowDayIndex]
                perDayData.SevereMan += perBatchData.DiseaseCourse["Severe"][perBatchData.NowDayIndex]
                perDayData.MildMan += perBatchData.DiseaseCourse["Mild"][perBatchData.NowDayIndex]

            # 所有批次的数据更新后，则累积计算总死亡人口，总痊愈人口，进而计算未感染人数
            self.AccumuDead += perDayData.DeadMan
            self.AccumuRecovery += perDayData.RecoveryMan

            # 需要重新计算下未感染人数
            perDayData.CalUnInfectMan(self.AccumuDead, self.AccumuRecovery)

            # 计算新感染人口。
            newInfectMan = self.SIModel(self.Ro, perDayData)

            # 更新当日数据
            perDayData.AddNewInfectMan(newInfectMan)
            # 保存今日数据到列表中。
            self.daysData.append(perDayData)

            # 创新新一批感染人口群，并附加到批次列表中。
            # 获得新感染人口后，需要针对新感染人口，启动新一批次人员
            newBatchData = self.CreateBatchData(newInfectMan)
            self.batchsData.append(newBatchData)

        # 输出总的模型计算结果，汇总各日数据，并产生图表。
        self.OutputResult()

        return

    # 产生某一批次人员的病程,这种方法是一开始就创建整个病程。
    # 比较适用于没有医院介入的情况。
    def CreateDiseaseCourse(self, perBatchData, courseParam):
        # perBatchData:每一批次的初始化数据
        # courseseParam:病程参数,应可能根据需要进行调整

        simulateDays = self.DayofStimulate

        # ——实际计算过程---

        # 创建病程后，存放在病程列表中。
        # 根据需要，产生**个人，然后进行分布：

        # 轻症转重症率：
        # 获得总的轻症转重症的数量
        mild2SevereNum = int(perBatchData.MildMan * courseParam["Mild2SeverePercent"])
        # 产生轻症转重症的在第几天数量的正态分布
        mild2SevereNormal = np.random.normal(courseParam["AverMild2SevereDay"], courseParam["varianceMild2SevereDay"],
                                             mild2SevereNum)
        # 初步按30天来算。获得第几天有几个轻症转重症。暂时不考虑开始时就有重症的
        mild2SevereCountPerDayNP = np.histogram(mild2SevereNormal, bins=range(simulateDays + 1))[0]
        deadNormal=np.array([])
        #需要根据每天的轻症转重症人，生成接下来的重症转死亡的人的分布。
        for tempSevereNum in mild2SevereCountPerDayNP:
            if (tempSevereNum>0):
                deadTemps =np.random.normal(courseParam["AverDied"], courseParam["varianceDied"], tempSevereNum)
                deadNormal=np.append(deadNormal,deadTemps)

        # 初步按30天来算。获得第几天死几个。
        deadCountPerDayNP = np.histogram(deadNormal, bins=range(simulateDays + 1))[0]
        # 计算死亡人数，估计可能的数量的分布

        # 死亡累加人口
        deadCumsum = np.cumsum(deadCountPerDayNP)

        # 获得存活总数,暂时用不上。
        #surviceNum = perBatchData.Population - deadNum


        # 累加每天的轻症转重症人数，并减去每天死亡的人数，成为每天重症人数

        severeCountPerDayNP = np.cumsum(mild2SevereCountPerDayNP) - deadCumsum

        # 获得当时死亡人数/前一天重症人数的比率
        DieDividePredaySevere = deadCountPerDayNP / np.append([0], severeCountPerDayNP[0:-1])

        # 计算总治愈人数的分布
        # 计算轻症变健康的人数
        # 总人数应该=轻症人数-变重症人数
        recoveryNum = perBatchData.MildMan - mild2SevereNum
        # 产生轻症转痊愈的在第几天数量的正态分布
        recoveryNormal = np.random.normal(courseParam["AverMild2RecoveryDay"], courseParam["varianceMild2RecoveryDay"],
                                          recoveryNum)
        # 初步按30天来算。获得第几天有几个痊愈。
        recoveryCountPerDayNP = np.histogram(recoveryNormal, bins=range(simulateDays + 1))[0]

        recoveryCumsum = np.cumsum(recoveryCountPerDayNP)

        # 逻辑检测：问题。=》死亡人数从哪来。
        # 论理每天重症人数不能少于死亡人数
        # 还需要补上逻辑检测部分

        # 计算剩下的病人数的分数：
        # 获得轻症人口
        mildCountPerDayNP = perBatchData.Population - deadCumsum - severeCountPerDayNP - recoveryCumsum

        # 获得每天轻症转重症人数/前一天轻症人数的比率
        mild2SevereDividePredayMild = mild2SevereCountPerDayNP / np.append([0], mildCountPerDayNP[0:-1])
        # 获得每天痊愈人数/前一天轻症人数的比率
        recoveryDividePredayMild = recoveryCountPerDayNP / np.append([0], mildCountPerDayNP[0:-1])

        # 画图功能，画出某批次的数据。有需要时，可测试使用。
        #plt.plot(deadCumsum, color='black', label='deadAccumu', marker='.')
        #plt.plot(severeCountPerDayNP, color='red', label='severe', marker='.')
        #plt.plot(mildCountPerDayNP, color='yellow', label='mild', marker='.')
        #plt.plot(recoveryCumsum, color='grey', label='recoveryAccumu', marker='.')
        #plt.legend()
        #plt.show()

        # 创建返回对象。
        # 按照顺序，返回每天的死亡人数、痊愈人数、重症人数、轻症人数
        diseaseCourse = {}
        diseaseCourse["Dead"] = deadCountPerDayNP
        diseaseCourse["Recovery"] = recoveryCountPerDayNP
        diseaseCourse["Severe"] = severeCountPerDayNP
        diseaseCourse["Mild"] = mildCountPerDayNP
        # 获得两个比率数据。
        # 调用np.nan_to_num,把比率中的nan元素变为很小，将近0的数据。
        diseaseCourse["DieDivideSevere"] = np.nan_to_num(DieDividePredaySevere)
        diseaseCourse["Mild2SevereDivideMild"] = np.nan_to_num(mild2SevereDividePredayMild)
        diseaseCourse["RecoveryDivideMild"] = np.nan_to_num(recoveryDividePredayMild)

        return diseaseCourse

    # 产生每一天的病程，根据参数。
    # #这段代码暂时没启用。
    def CreateCurrentDayDiseaseCourse(self, perBatchData, courseParam):

        # 向下取整，尽量减少死亡数
        deadNum = int(perBatchData.Population * courseParam["Mortality"])
        # 产生死亡数的在第几天死亡的正态分布
        deadNormal = np.random.normal(courseParam["AverDied"], courseParam["varianceDied"], deadNum)

        return

    # 传入每天的数据，进行计算。
    # 要输出每日新感染的人数。
    def SIModel(self, Ro, perDayData):

        # 和人口总数论理应该没有关系，但还是需要传入一个总人口。

        # N为人群总数,获得现在总人口。
        N = perDayData.Population

        # 使用RO来计算感染人数
        # gamma为恢复率系数，为平均住院天数的倒数。
        gamma = 1 / 14
        # β为传染率系数
        beta = gamma * Ro

        # 旧感染人口为原来轻症人数和重症人数之和。
        infectOldData = perDayData.SevereMan + perDayData.MildMan
        print("infect old Data:" + str(infectOldData))
        # 未来计算时，这地方应该加上一个随机系数，现暂时确定用固定的。

        # 计算感染人员公式。=>未感染者的，还需要重新计算。
        newInfectMan = int(beta * infectOldData * perDayData.UnInfectMan / N - gamma * infectOldData)
        # 如果新感染人数>未感染人数，即已经无可感染人数，则将新感染人数设为未感染人数
        # 以下规则还有问题。
        # if newInfectMan>=perDayData.UnInfectMan:
        #    newInfectMan=perDayData.UnInfectMan
        if newInfectMan < 0:
            newInfectMan = 0

        #print("new infect man:" + str(newInfectMan))
        return newInfectMan

    # 此处各参数（p、Tg_1、Tg_2）取值参考其他博客（见文章最后博客链接）

    # 生成新的每批数据。
    # 根据多少个患者，生成每日批量数据
    def CreateBatchData(self, population):

        perBatchData = PerBatchData()
        perBatchData.Population = population
        perBatchData.MildMan = population

        # 根据病程参数，来准备生成病程
        courseParam = {}
        # 在这边可以进行参数设置，现暂时参数设置放在程序内部
        # 准备迁移过来。
        # ————参数赋值——
        # 死亡率参数
        courseParam["Mortality"] = 0.05
        # 平均死亡时间：轻症转重症，9天，重症死亡4天
        courseParam["AverDied"] = 13
        # 死亡时间方差
        courseParam["varianceDied"] = 3
        # 轻症转重症比率
        courseParam["Mild2SeverePercent"] = 0.10
        # 轻症转重症平均天数
        courseParam["AverMild2SevereDay"] = 5
        # 轻症转重症时间方差
        courseParam["varianceMild2SevereDay"] = 2
        # 轻症转痊愈平均天数
        courseParam["AverMild2RecoveryDay"] = 7
        # 轻症转痊愈时间方差
        courseParam["varianceMild2RecoveryDay"] = 2

        # 默认情况下，不考虑重症输入，先全部假定为轻症患者。
        # 后期有需要时再改造。
        # 生成病程
        perBatchData.DiseaseCourse = self.CreateDiseaseCourse(perBatchData, courseParam)

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

        #hosAcceptObj=HospitalAcceptObj(0,0)
        hosAcceptObj=HospitalAcceptObj(random.randint(50,60),100)
        return hosAcceptObj

    # 根据医院可接收的数量，来进行每批次人员调整。(按新的比率算法）
    def ChangeBatchsByHosAcceptObj2(self, hosAcceptObj, perBatchData, dayofCourse):

        return
    # 根据医院可接收的数量，来进行每批次人员调整。
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

        for everydayData in self.daysData:
            deadArray.append(everydayData.DeadMan)
            severeArray.append(everydayData.SevereMan)
            mildArray.append(everydayData.MildMan)
            recoveryArray.append(everydayData.RecoveryMan)
            hosSevereArray.append(everydayData.SevereHosMan)
            hosMildArray.append(everydayData.MildHosMan)
            unInfectArray.append(everydayData.UnInfectMan)
            populationArray.append(everydayData.Population)
            InfectArray.append(everydayData.MildMan+everydayData.SevereMan)
            unInfectArray.append(everydayData.UnInfectMan)

        #plt.subplot(1, 2, 1)
        # 画图功能，画出某批次的数据。有需要时，可测试使用。
        plt.plot(np.cumsum(deadArray), color='black', label='deadAccumu', marker='.')
        plt.plot(severeArray, color='red', label='severe', marker='.')
        plt.plot(mildArray, color='yellow', label='mild', marker='.')
        plt.plot(np.cumsum(recoveryArray), color='grey', label='recoveryAccumu', marker='.')
        # plt.plot(unInfectArray, color='blue', label='recovery', marker='.')
        plt.legend()

        #plt.subplot(1, 2, 2)
        # 画图功能，画出某批次的数据。有需要时，可测试使用。
        #plt.plot(np.cumsum(deadArray), color='black', label='dead', marker='.')
        #plt.plot(populationArray, color='blue', label='population', marker='.')
        #plt.plot(unInfectArray, color='yellow', label='unInfect', marker='.')
        #plt.plot(InfectArray, color='red', label='Infect', marker='.')

        #plt.legend()

        plt.show()

        return
