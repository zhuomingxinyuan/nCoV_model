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


# 输入参数：
TotalPopulation = 100000
# 开始重症人数
StartSevere = 0
# 开始轻症人数
StartMild = 900
# 传染率默认参数，Basic reproduction number
Ro = 2.4
DayofStimulate = 30

epidemicStartParam={
    "TotalPopulation":TotalPopulation,
    "StartSevere":StartSevere,
    "StartMild":StartMild,
    "Ro":Ro,
    "DayofStimulate":DayofStimulate
}
epidemic=Epidemic(epidemicStartParam)
epidemic.Simulate()

print("finish")