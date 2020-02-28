# -*- coding: utf-8 -*-
#本代码为了实现nCov院外模拟逻辑，模拟院外病人扩散状态，并输出可视化结果。
#同时本模型需要与院内模型进行数据交互，根据院内模型反馈结果，进行模型模拟和运行。
#项目思路按【新冠建模】ppt思路中的模型来进行。

#可能调用的类库：
#调用画图函数库
#import matplotlib.pyplot as plt
#用于进行数据输出的工具
#import pandas as pd
from src.epidemic import Epidemic
#主程序，用于进行测试控制


# --总体模型控制参数：---
TotalPopulation = 100000
# 开始重症人数
StartSevere = 0
# 开始轻症人数
StartMild = 0
# 开始无症人数：
StartAsym=900
# 传染率默认参数，Basic reproduction number
Ro = 2.4
DayofStimulate = 30

# ———详细病程参数赋值——
detailcourseParam = {}
# 死亡率参数
detailcourseParam["Mortality"] = 0.05
# 平均死亡时间：轻症转重症，9天，重症死亡4天
detailcourseParam["AverDied"] = 13
# 死亡时间方差
detailcourseParam["varianceDied"] = 3
# 轻症转重症比率
detailcourseParam["Mild2SeverePercent"] = 0.10
# 轻症转重症平均天数
detailcourseParam["AverMild2SevereDay"] = 5
# 轻症转重症时间方差
detailcourseParam["varianceMild2SevereDay"] = 2
# 轻症转痊愈平均天数
detailcourseParam["AverMild2RecoveryDay"] = 7
# 轻症转痊愈时间方差
detailcourseParam["varianceMild2RecoveryDay"] = 2

#一直无症状人员占总人员的比重
detailcourseParam["AsymptomPercent"] = 0.30
# 无症状人员转轻症平均天数
detailcourseParam["AverAsym2Mild"] = 7
# 无症状人员转轻症时间方差
detailcourseParam["varianceAsym2Mild"] =3
# 无症状人员转痊愈平均天数
detailcourseParam["AverAsym2Recovery"] = 14
# 无症状人员转痊愈时间方差
detailcourseParam["varianceAsym2Recovery"] = 3

#将参数汇总起来。
epidemicStartParam={
    "TotalPopulation":TotalPopulation,
    "StartSevere":StartSevere,
    "StartMild":StartMild,
    "StartAsym":StartAsym,
    "Ro":Ro,
    "DayofStimulate":DayofStimulate,
    "DetailCourseParam":detailcourseParam
}
#启动病程模拟对象。
epidemic=Epidemic(epidemicStartParam)
epidemic.Simulate()

print("finish")