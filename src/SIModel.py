# -*- coding: utf-8 -*-
#本代码为了实现传染病模型框架，为了方便扩展。

# 核心的模型对象，代表整个疾病模拟模型。
class SIModel:
    #输入参数
    #一些比较固定的参数放在这
    InputParam={}

    #一些经常发生变化的参数：
    RunParam={}
    #输出结果
    OutValues={}
    def __init__(self,inputParam):
        self.InputParam=inputParam

        return

    def SetParam(self,paramName,paramValue):

        self.InputParam["paramName"]=paramValue

        return True

    # 使用模型核心功能进行计算。根据各模型实际进行拓展
    def run(self,runParam={}):
        #使用runParam作为运行参数来计算。
        self.RunParam=runParam

        return self.OutValues

# 简单的传染病模型，作为初步模型框架。
class SimpleSIModel(SIModel):
    #def __init__(self, InputParam):
    #    SIModel.__init__(InputParam)

    #模型的核心计算过程。
    # 不太好，就是与原来模型中的核心数据对象捆绑
    # 有需要时，后期再解耦。
    def run(self,runParam={}):

        self.RunParam=runParam

        # 先设置模型参数
        Ro=self.InputParam["Ro"]
        gamma=self.InputParam["gamma"]

        # 后设置运行参数
        perDayData = self.RunParam["perDayData"]

        #详细的计算过程。
        # 和人口总数论理应该没有关系，但还是需要传入一个总人口。
        # N为人群总数,获得现在总人口。
        N = perDayData.Population

        # 使用RO来计算感染人数
        # gamma为恢复率系数，为平均住院天数的倒数。
        #gamma = 1 / 14
        # β为传染率系数
        beta = gamma * Ro

        # 旧感染人口为原来轻症人数和重症人数之和。
        infectOldData = perDayData.SevereMan + perDayData.MildMan + perDayData.AsymMan
        # print("infect old Data:" + str(infectOldData))
        # 未来计算时，这地方应该加上一个随机系数，现暂时确定用固定的。

        # 计算感染人员公式。=>未感染者的，还需要重新计算。
        newInfectMan = int(beta * infectOldData * perDayData.UnInfectMan / N - gamma * infectOldData)
        # 如果新感染人数>未感染人数，即已经无可感染人数，则将新感染人数设为未感染人数
        # 以下规则还有问题。
        # if newInfectMan>=perDayData.UnInfectMan:
        #    newInfectMan=perDayData.UnInfectMan
        if newInfectMan < 0:
            newInfectMan = 0

        self.OutValues["newInfectMan"]=newInfectMan




        return self.OutValues
