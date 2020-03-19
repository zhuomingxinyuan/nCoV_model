# -*- coding: utf-8 -*-
# 医院的管理者文件，用于管理各种医院模型并交换数据。

# -*- coding: utf-8 -*-
import numpy as np
# import random
import matplotlib.pyplot as plt
from enum import Enum
# 从同目录中引入SIModel对象类。
# from . import SIModel
from src import hospitalmain
from src import epidemic

# from src.epidemic import PerDayData
# from src.epidemic import  Epidemic
# import src.epidemic


# 代表病人状态
class HospitalType(Enum):
    MCDS_COMPOSITE = 1  # 方舱和定点医院都有,两者复合
    MCDS_INDEPENDENT = 2  # 方舱和定点医院都有,两者独立
    ONLY_DS = 3  # 只有重症的定点医院

# region 医院创建的类


class Hospital:
    hospitaltype = 0
    # 床位数
    bednum = 0

    # 医院接口类
    def __init__(self):

        return

    def acceptpatient(self, dayofsimulation: int, predaydata):
        raise NotImplementedError

    def outputresult(self):
        raise NotImplementedError

    def cal_empty_bed(self):
        # 计算空床位功能
        raise NotImplementedError

    def add_dshbed(self):
        # 增加重症的床位
        raise NotImplementedError


class McDsIndHospital(Hospital):
    # 医院接口类,方舱医院与定点医院独立

    mchospital = None
    dshospital = None

    mchbednum = 0
    dshbednum = 0

    def __init__(self):
        super().__init__()
        self.hospitaltype = HospitalType.MCDS_INDEPENDENT.value

        # 先建立方舱医院
        mchmodelparam = hospitalmain.HospitalModelParam()
        mchrunparam = hospitalmain.HospitalModelRunParam()
        # 假设方舱医院的氧气和呼吸机是满足的，即不需要改变病程。
        mchrunparam.oxygen_change_list = []
        mchrunparam.breathing_machine_change_list = []
        # 开始建立模型，并启动。
        self.mchospital = hospitalmain.HospitalModel(mchmodelparam, mchrunparam)
        self.mchospital.start()

        # 建立定点医院
        # 定点医院初始化
        dshmodelparam = hospitalmain.HospitalModelParam()
        dshrunparam = hospitalmain.HospitalModelRunParam()
        # 测试改变定点医院的床位数据
        dshrunparam.bedstartnum = 150
        # 假设定点医院的参数均按照原来的参数进行变化，即有病程变化
        self.dshospital = hospitalmain.HospitalModel(dshmodelparam, dshrunparam)
        self.dshospital.start()

        self.bednum = mchrunparam.bedstartnum + dshrunparam.bedstartnum

        return

    def acceptpatient(self, dayofsimulation: int, predaydata):
        # outmodelsevereaccept = 0
        # mchaccept_patientnum = 0

        # 如果是第一天, 则不接受病人
        if dayofsimulation == 0:
            mchaccept_patientnum = 0
            outmodelsevereaccept = 0
            dshaccept_patientnum = 0
            mch_better_num = 0
            mch_worse_num = 0
            self.mchospital.calbatch(dayofsimulation)
            self.mchospital.addnewbatch(dayofsimulation, mchaccept_patientnum)

            self.mch_cal_empty_bed(dayofsimulation, mch_better_num, mch_worse_num, mchaccept_patientnum)

            dsh_better_num = 0
            dsh_worse_num = 0
            self.dshospital.calbatch(dayofsimulation)
            self.dshospital.addnewbatch(dayofsimulation, dshaccept_patientnum)
            self.dsh_cal_empty_bed(dayofsimulation, dsh_better_num, dsh_worse_num, dshaccept_patientnum)

        else:
            # 方舱医院的新增病人为每日增加进来的轻症病人
            mchaccept_patientnum = self.mch_acceptpatient(dayofsimulation, predaydata.MildMan)
            print(u"今日方舱医院接收：" + str(mchaccept_patientnum), end=''),
            # 方舱医院开始今日的计算
            mch_better_num, mch_worse_num = self.mchospital.calbatch(dayofsimulation)
            self.mchospital.addnewbatch(dayofsimulation, mchaccept_patientnum)

            print(u" 昨日方舱医院病程恶化：" +
                  str(self.mchospital.modelResult.worse_final_list[dayofsimulation - 1]))

            self.mch_cal_empty_bed(dayofsimulation, mch_better_num, mch_worse_num, mchaccept_patientnum)

            # 定点医院的新增病人来源为：原来未被接收在等待的方舱医院病人+
            # 方舱医院前一天的重症病人+每日增加进来的重症病人
            [mchwaitaccept, mchworseaccept, outmodelsevereaccept] = self.dsh_acceptpatient(
                dayofsimulation, self.mchospital.modelResult.waitforaccept[dayofsimulation - 1],
                self.mchospital.modelResult.worse_final_list[dayofsimulation - 1], predaydata.SevereMan
            )
            print(u"定点医院接收 方舱等待接受人员 %d 人, "
                  "方舱今日重症 %d 人,  院外重症 %d 人 " % (mchwaitaccept, mchworseaccept, outmodelsevereaccept))

            # 根据接收数目再更新到方舱医院数据中。
            self.mchospital.refreshwaitpatient(dayofsimulation, mchwaitaccept, mchworseaccept)

            # 总的接收数量
            dshaccept_patientnum = mchwaitaccept + mchworseaccept + outmodelsevereaccept
            # 开始计算定点医院的每日过程
            dsh_better_num, dsh_worse_num = self.dshospital.calbatch(dayofsimulation)
            self.dshospital.addnewbatch(dayofsimulation, dshaccept_patientnum)
            self.dsh_cal_empty_bed(dayofsimulation, dsh_better_num, dsh_worse_num, dshaccept_patientnum)

        # 定义返回对象
        hosacceptobj = epidemic.HospitalAcceptObj(outmodelsevereaccept, mchaccept_patientnum)

        return hosacceptobj

    def mch_acceptpatient(self, dayofsimulate, patientwaitforaccept):
        """接收从院外模型转过来的数据，并返回接收人员的数量，
        比较简化的代码，现适用于方舱医院来调用。
        :param:
         dayofsimulate: 模拟的第几天
         patientwaitforaccept:等着接收的病人数
        :return: acceptPatientnum:接收几个病人
        """
        # 参数：dayofSimulate:
        # 以下是后期准备用的参数。
        # perDaydata:某日的病人数据。
        # numofmildaccept= 0
        # numofsevereaccept= 0
        # 如果是第一天，则接收病人为0
        # 如果非第一天，则用前一天空出来的病床数作为接收病人数。
        if dayofsimulate == 0:
            # acceptpatientnum = self.modelparam.batch_population
            acceptpatientnum = 0
        else:
            # 原临时测试使用前一天病床数
            # acceptPatientnum=self.modelResult.empty_beds[dayofsimulate-1]
            # 考虑床位限制，床位够则全部接收，不够则只接收床位数。
            if patientwaitforaccept < self.mchospital.modelResult.empty_beds[dayofsimulate - 1]:
                acceptpatientnum = patientwaitforaccept
            else:
                print(u"方舱床位不够用了")
                acceptpatientnum = self.mchospital.modelResult.empty_beds[dayofsimulate - 1]

        return int(acceptpatientnum)

    def dsh_acceptpatient(self, dayofsimulate: int, mchwait: int, mchworse: int, outmodelsevere: int):
        """应用于定点医院的接收病人逻辑
        :param dayofsimulate: 模拟的第几天
        :param mchworse: 方舱医院的恶化病人
        :param mchwait: 方舱医院的等待接收的病人（以前未接收的）
        :param outmodelsevere: 院外模型的重症患者
        :return: mchworseaccept ：接收的方舱医院的恶化病人数
        :return: mchwaitaccept：方舱医院的等待接收的病人接收数
        :return: outmodelsevereaccept：院外重症病人接收数
        """
        mchworseaccept = 0
        mchwaitaccept = 0
        outmodelsevereaccept = 0

        if dayofsimulate != 0:
            # 原临时测试使用前一天病床数
            # acceptPatientnum=self.modelResult.empty_beds[dayofsimulate-1]
            # 考虑床位限制，床位够则全部接收，不够则只接收床位数。
            # 优先顺序规则：优先接收等待的病人，然后再是方舱医院的病人，然后才是院外重症。
            # 先获得昨天的空床位，
            nowbed = int(self.dshospital.modelResult.empty_beds[dayofsimulate - 1])
            # 先接收等待的病人
            if mchwait < nowbed:
                mchwaitaccept = mchwait
                nowbed -= mchwait
            else:
                print(u"定点医院床位不够接收方舱滞留重症用了")
                mchwaitaccept = nowbed
                nowbed = 0
            # 再接收方舱医院病人
            if nowbed != 0:
                if mchworse < nowbed:
                    mchworseaccept = mchworse
                    nowbed -= mchworse
                else:
                    print(u"定点医院床位不够接收方舱滞留重症了")
                    mchworseaccept = nowbed
                    nowbed = 0
            else:
                print(u"定点医院床位不够接收方舱滞留重症了")
            # 再接收院外重症
            if nowbed != 0:
                if outmodelsevere < nowbed:
                    outmodelsevereaccept = outmodelsevere
                    nowbed -= outmodelsevere
                else:
                    print(u"定点医院床位不够接收院外重症了")
                    outmodelsevereaccept = nowbed
                    # nowbed = 0
            else:
                print(u"定点医院床位不够接收院外重症了")

        # 返回以上三个数据
        return [int(mchwaitaccept), int(mchworseaccept), int(outmodelsevereaccept)]

    def outputresult(self):
        # 画出院内模型方舱医院的结果
        plt.figure()
        plt.title('MC hospital Model result', fontsize='large', fontweight='bold')
        self.mchospital.outputResult()

        # 画出院内模型定点医院的结果
        plt.figure()
        plt.title('DE hospital Model result', fontsize='large', fontweight='bold')
        self.dshospital.outputResult()

        # plt.show()

        # 输出汇总的数据。\
        print("方舱医院：总人数：%d ,总康复：%d , 总恶化：%d" %
              (self.mchospital.modelResult.population, self.mchospital.modelResult.better_final_cumu_list[-1],
               self.mchospital.modelResult.worse_final_cumu_list[-1]))

        print("定点医院：总人数：%d ,总康复：%d , 总恶化：%d" %
              (self.dshospital.modelResult.population, self.dshospital.modelResult.better_final_cumu_list[-1],
               self.dshospital.modelResult.worse_final_cumu_list[-1]))

        # 总数为两个医院的接受病人数-方舱送到定点医院的人数
        patientsum = self.mchospital.modelResult.population + self.dshospital.modelResult.population - \
                     self.mchospital.modelResult.worse_final_cumu_list[-1]
        # 求出死亡率
        deadpercent = self.dshospital.modelResult.worse_final_cumu_list[-1] / patientsum

        print("院内总人数： %d, 死亡率： %f " % (patientsum, deadpercent))

        return patientsum, deadpercent

    def cal_empty_bed(self):
        # 计算空床位功能
        # self.dsh_cal_empty_bed()
        # self.mch
        return

    def dsh_cal_empty_bed(self, dayofsimulate: int, daily_total_better: int,
                          daily_total_worse: int, acceptpatient: int):
        """定点医院计算出今日的空余床位
        :param dayofsimulate: 模拟的第几天
        :param daily_total_better: 今日好转的人数
        :param daily_total_worse: 今日恶化的人数
        :param acceptpatient: 今日接收的新病人
        :return: 无
        """

        # 算出今日的床位（？）
        # =床位总数-（每日的恶化人员+好转人员）
        # empty_bed = self.runparam.bedstartnum-(daily_total_worse + daily_total_better)
        # 规则发现不对，不应是每日的，而有人还在占着床位的。需要用昨日的床位来计算
        if dayofsimulate != 0:
            # 空床位为昨天的空床位，再加上今日恶化的，及好转的，再加上接受院外的新病人
            empty_bed = self.dshospital.modelResult.empty_beds[dayofsimulate-1] + \
                        (daily_total_worse + daily_total_better) - acceptpatient
        else:
            # 如果是第一天，空床位为起始数量
            empty_bed = self.dshospital.runparam.bedstartnum
            # empty_bed = self.dshospital.modelResult.everybeds[dayofsimulate]
            # - (daily_total_worse + daily_total_better)

        # 如果有滞留人员
        # 如果不是第一天，则床位要减去滞留人员
        if dayofsimulate != 0:
            empty_bed = empty_bed - self.dshospital.modelResult.waitforaccept[dayofsimulate]
            # 如果滞留人员过多，则导致没有床位。
            if empty_bed < 0:
                empty_bed = 0

        self.dshospital.modelResult.empty_beds[dayofsimulate] = empty_bed

        return

    def mch_cal_empty_bed(self, dayofsimulate: int, daily_total_better: int,
                          daily_total_worse: int, acceptpatient: int):
        """方舱医院计算出今日的空余床位
       :param dayofsimulate: 模拟的第几天
       :param daily_total_better: 今日好转的人数
       :param daily_total_worse: 今日恶化的人数
       :param acceptpatient: 今日接收的新病人
       :return: 无
       """

        # 改为床位使用昨日的床位数来减去（每日的恶化人员+好转人员）
        # empty_bed = self.mchospital.modelResult.everybeds[dayofsimulate] - (daily_total_worse + daily_total_better)

        if dayofsimulate != 0:
            # 空床位为昨天的空床位
            empty_bed = self.mchospital.modelResult.empty_beds[dayofsimulate - 1] + \
                        (daily_total_worse + daily_total_better) - acceptpatient

        else:
            # 如果是第一天，空床位为起始数量
            empty_bed = self.mchospital.runparam.bedstartnum

        # 如果有滞留人员
        # 如果不是第一天，则床位要减去滞留人员
        if dayofsimulate != 0:
            empty_bed = empty_bed - self.mchospital.modelResult.waitforaccept[dayofsimulate]
            # 如果滞留人员过多，则导致没有床位。
            if empty_bed < 0:
                empty_bed = 0

        # 如果床位过多，（有可能昨天空床多，今天走的人也多）
        if empty_bed > self.mchospital.modelResult.everybeds[dayofsimulate]:
                    empty_bed = self.mchospital.modelResult.everybeds[dayofsimulate]

        # 将今日的空床位写到相应记录中
        self.mchospital.modelResult.empty_beds[dayofsimulate] = empty_bed

        return

    # 以下代码还有问题，暂时不使用
    # TODO: 关于如何添加新床位，又不影响原来床位计算逻辑上，还有一些问题。
    def add_dshbed(self, dayofsimulation: int, newbednum: int):
        """增加某日的床位，以实现医院的床位调整，
        考虑基本不会减床位，因此只实现床位增加
        :param dayofsimulation: 模拟的第几天
        :param newbednum: 增加的新床位
        :return:
        """
        # 从第几日开始的床位，都增加。
        # 先只针对定点医院
        self.dshospital.modelResult.everybeds[dayofsimulation:] += newbednum
        print("第" + str(dayofsimulation) + "日增加" + str(newbednum) + "床位")

    def add_mchbed(self, dayofsimulation: int, newbednum: int):
        """增加某日的床位，以实现医院的床位调整，
        考虑基本不会减床位，因此只实现床位增加
        :param dayofsimulation: 模拟的第几天
        :param newbednum: 增加的新床位
        :return:
        """
        # 从第几日开始的床位，都增加。
        # 先只针对定点医院
        self.mchospital.modelResult.everybeds[dayofsimulation:] += newbednum
        print("第" + str(dayofsimulation) + "日增加" + str(newbednum) + "床位")


class McDsComHospital(Hospital):
    # 医院接口类,方舱医院与定点医院独立

    mchospital = None
    dshospital = None

    mchbednum = 0
    dshbednum = 0

    empty_beds = []

    def __init__(self):

        super().__init__()
        self.hospitaltype = HospitalType.MCDS_COMPOSITE.value

        # 先建立方舱医院
        mchmodelparam = hospitalmain.HospitalModelParam()
        mchrunparam = hospitalmain.HospitalModelRunParam()
        # 假设方舱医院的氧气和呼吸机是满足的，即不需要改变病程。
        mchrunparam.oxygen_change_list = []
        mchrunparam.breathing_machine_change_list = []
        # 开始建立模型，并启动。
        self.mchospital = hospitalmain.HospitalModel(mchmodelparam, mchrunparam)
        self.mchospital.start()

        # 建立定点医院
        # 定点医院初始化
        dshmodelparam = hospitalmain.HospitalModelParam()
        dshrunparam = hospitalmain.HospitalModelRunParam()
        # 复合医院，两个医院的床位是一样的。
        # 如果需要修改医院床位，则只修改方舱的床位参数。
        dshrunparam.bedstartnum = mchrunparam.bedstartnum
        # 假设定点医院的参数均按照原来的参数进行变化，即有病程变化
        self.dshospital = hospitalmain.HospitalModel(dshmodelparam, dshrunparam)
        self.dshospital.start()
        # 床位的初始化
        self.bednum = mchrunparam.bedstartnum
        self.empty_beds = np.ones(self.mchospital.modelparam.days) * self.bednum

        return

    def acceptpatient(self, dayofsimulation: int, predaydata):
        # outmodelsevereaccept = 0
        # mchaccept_patientnum = 0
        # 方舱和定点合在一起的接收病人规则，应该和两者独立的状态是不一样的。


        # 如果是第一天, 则不接受病人
        if dayofsimulation == 0:
            mchaccept_patientnum = 0
            outmodelsevereaccept = 0
            dshaccept_patientnum = 0
            mch_better_num = 0
            # mch_worse_num = 0
            dsh_better_num = 0
            dsh_worse_num = 0
            self.mchospital.calbatch(dayofsimulation)
            self.mchospital.addnewbatch(dayofsimulation, mchaccept_patientnum)
            self.dshospital.calbatch(dayofsimulation)
            self.dshospital.addnewbatch(dayofsimulation, dshaccept_patientnum)

            self.cal_empty_bed2(dayofsimulation, mch_better_num, mchaccept_patientnum,
                                dsh_better_num, dsh_worse_num, dshaccept_patientnum)
            # self.mch_cal_empty_bed(dayofsimulation, mch_better_num, mch_worse_num, mchaccept_patientnum)
            # self.dsh_cal_empty_bed(dayofsimulation, dsh_better_num, dsh_worse_num, dshaccept_patientnum)

        else:
            # 流程：应该定点医院先接收重症病人，方舱后接收
            empty_bed = self.empty_beds[dayofsimulation - 1]

            # 先算出要接收院外的几个重症
            if predaydata.SevereMan < empty_bed:
                dshaccept_patientnum = predaydata.SevereMan
                empty_bed -= dshaccept_patientnum
            else:
                print(u"定点医院床位不够接收院外重症了")
                dshaccept_patientnum = empty_bed
                empty_bed = 0

            # 再算要接收院外的几个轻症
            if predaydata.MildMan < empty_bed:
                mchaccept_patientnum = predaydata.MildMan
                empty_bed -= mchaccept_patientnum
            else:
                print(u"方舱床位不够用了")
                mchaccept_patientnum = empty_bed
                empty_bed = 0

            print(u"今日定点医院接收：" + str(dshaccept_patientnum), end=''),
            print(u" 今日方舱医院接收：" + str(mchaccept_patientnum), end=''),

            # 方舱医院开始今日的计算
            mch_better_num, mch_worse_num = self.mchospital.calbatch(dayofsimulation)
            self.mchospital.addnewbatch(dayofsimulation, mchaccept_patientnum)

            # 定点医院的新增病人来源为：方舱医院前一天的重症病人+每日增加进来的重症病人
            print(u"定点医院接收 %d 方舱昨日重症, %d 院外重症 " % (
                self.mchospital.modelResult.worse_final_list[dayofsimulation - 1], dshaccept_patientnum))

            # 总的接收数量
            dshaccept_patientnumsum = self.mchospital.modelResult.worse_final_list[dayofsimulation - 1] + \
                                      dshaccept_patientnum
            # 开始计算定点医院的每日过程
            dsh_better_num, dsh_worse_num = self.dshospital.calbatch(dayofsimulation)
            self.dshospital.addnewbatch(dayofsimulation, dshaccept_patientnumsum)

            self.cal_empty_bed2(dayofsimulation, mch_better_num, mchaccept_patientnum,
                                dsh_better_num, dsh_worse_num, dshaccept_patientnum)

        # 定义返回对象
        hosacceptobj = epidemic.HospitalAcceptObj(dshaccept_patientnum, mchaccept_patientnum)

        return hosacceptobj

    def mch_acceptpatient(self, dayofsimulate, patientwaitforaccept):
        """接收从院外模型转过来的数据，并返回接收人员的数量，
        比较简化的代码，现适用于方舱医院来调用。
        :param:
         dayofsimulate: 模拟的第几天
         patientwaitforaccept:等着接收的病人数
        :return: acceptPatientnum:接收几个病人
        """
        # 参数：dayofSimulate:
        # 以下是后期准备用的参数。
        # perDaydata:某日的病人数据。
        # numofmildaccept= 0
        # numofsevereaccept= 0
        # 如果是第一天，则接收病人为0
        # 如果非第一天，则用前一天空出来的病床数作为接收病人数。
        if dayofsimulate == 0:
            # acceptpatientnum = self.modelparam.batch_population
            acceptpatientnum = 0
        else:
            # 原临时测试使用前一天病床数
            # acceptPatientnum=self.modelResult.empty_beds[dayofsimulate-1]
            # 考虑床位限制，床位够则全部接收，不够则只接收床位数。
            # TODO: 需要修改床位获得逻辑。
            if patientwaitforaccept < self.mchospital.modelResult.empty_beds[dayofsimulate - 1]:
                acceptpatientnum = patientwaitforaccept
            else:
                print(u"方舱床位不够用了")
                acceptpatientnum = self.mchospital.modelResult.empty_beds[dayofsimulate - 1]

        return int(acceptpatientnum)

    def dsh_acceptpatient(self, dayofsimulate: int, outmodelsevere: int):
        """应用于定点医院的接收病人逻辑
        :param dayofsimulate: 模拟的第几天
        :param mchworse: 方舱医院的恶化病人
        :param outmodelsevere: 院外模型的重症患者
        :return: mchworseaccept ：接收的方舱医院的恶化病人数
        :return: mchwaitaccept：方舱医院的等待接收的病人接收数
        :return: outmodelsevereaccept：院外重症病人接收数
        """
        mchworseaccept = 0
        # mchwaitaccept = 0
        outmodelsevereaccept = 0

        if dayofsimulate != 0:
            # 原临时测试使用前一天病床数
            # acceptPatientnum=self.modelResult.empty_beds[dayofsimulate-1]
            # 考虑床位限制，床位够则全部接收，不够则只接收床位数。
            # 优先顺序规则：优先接收等待的病人，然后再是方舱医院的病人，然后才是院外重症。
            # 先获得昨天的空床位，
            # TODO: 需要修改床位获得逻辑。
            nowbed = int(self.empty_beds[dayofsimulate - 1])

            # 接收院外重症
            if nowbed != 0:
                if outmodelsevere < nowbed:
                    outmodelsevereaccept = outmodelsevere
                    nowbed -= outmodelsevere
                else:
                    print(u"定点医院床位不够接收院外重症了")
                    outmodelsevereaccept = nowbed
                    # nowbed = 0
            else:
                print(u"定点医院床位不够接收院外重症了")

        # 返回以上三个数据
        return int(outmodelsevereaccept)

    def outputresult(self):
        # 画出院内模型方舱医院的结果
        plt.figure()
        plt.title('MC hospital Model result', fontsize='large', fontweight='bold')
        self.mchospital.outputResult()

        # 画出院内模型定点医院的结果
        plt.figure()
        plt.title('DE hospital Model result', fontsize='large', fontweight='bold')
        self.dshospital.outputResult()

        # plt.show()

        # 输出汇总的数据。\
        print("方舱医院：总人数：%d ,总康复：%d , 总恶化：%d" %
              (self.mchospital.modelResult.population, self.mchospital.modelResult.better_final_cumu_list[-1],
               self.mchospital.modelResult.worse_final_cumu_list[-1]))

        print("定点医院：总人数：%d ,总康复：%d , 总恶化：%d" %
              (self.dshospital.modelResult.population, self.dshospital.modelResult.better_final_cumu_list[-1],
               self.dshospital.modelResult.worse_final_cumu_list[-1]))

        # 总数为两个医院的接受病人数-方舱送到定点医院的人数
        patientsum = self.mchospital.modelResult.population + self.dshospital.modelResult.population - self.mchospital.modelResult.worse_final_cumu_list[-1]
        # 求出死亡率
        deadpercent = self.dshospital.modelResult.worse_final_cumu_list[-1] / patientsum

        print("院内总人数： %d, 死亡率： %f " % (patientsum, deadpercent))

        return patientsum, deadpercent

    def cal_empty_bed(self):
        # 计算空床位功能


        return

    #
    def cal_empty_bed2(self, dayofsimulate: int, mch_daily_total_better: int, mchacceptpatient: int,
                      dsh_daily_total_better: int , dsh_daily_total_worse: int, dshacceptpatient: int):
        """定点医院计算出今日的空余床位
        :param dayofsimulate: 模拟的第几天
        :param mch_daily_total_better: 今日好转的方舱人数
        :param mchacceptpatient: 今日接收的轻症症新病人
        :param dsh_daily_total_better: 今日好转的定点医院人数
        :param dsh_daily_total_worse: 今日恶化的定点医院人数
        :param dshacceptpatient: 今日接收的重症新病人
        :return: 无
        """
        # =床位总数-（每日的恶化人员+好转人员）
        # empty_bed = self.runparam.bedstartnum-(daily_total_worse + daily_total_better)
        # 规则发现不对，不应是每日的，而有人还在占着床位的。需要用昨日的床位来计算
        if dayofsimulate != 0:
            # 空床位为昨天的空床位，再加上今日恶化的，及好转的，再加上接受院外的新病人
            empty_bed = self.empty_beds[dayofsimulate - 1] + mch_daily_total_better + \
                        dsh_daily_total_better+dsh_daily_total_worse - mchacceptpatient - dshacceptpatient

            # empty_bed = self.dshospital.modelResult.empty_beds[dayofsimulate - 1] + \
            #             (daily_total_worse + daily_total_better) - acceptpatient
        else:
            # 如果是第一天，空床位为起始数量-原来方舱和定点医院的起始病人
            empty_bed = self.dshospital.runparam.bedstartnum-self.dshospital.modelparam.start_population - self.mchospital.modelparam.start_population
            # empty_bed = self.dshospital.modelResult.everybeds[dayofsimulate]
            # - (daily_total_worse + daily_total_better)

        # 如果有滞留人员
        # 如果不是第一天，则床位要减去滞留人员，=》现不需要。
        # if dayofsimulate != 0:
        #     empty_bed = empty_bed - self.dshospital.modelResult.waitforaccept[dayofsimulate]
        #     # 如果滞留人员过多，则导致没有床位。
        #     if empty_bed < 0:
        #         empty_bed = 0

        # 如果床位过多，（有可能昨天空床多，今天走的人也多,适用于开始模拟）
        if empty_bed > self.dshospital.runparam.bedstartnum:
            empty_bed = self.dshospital.runparam.bedstartnum

        self.empty_beds[dayofsimulate] = empty_bed

        print("空床位为" + str(empty_bed))
        return



    # 以下代码还有问题，暂时不使用
    # TODO: 关于如何添加新床位，又不影响原来床位计算逻辑上，还有一些问题。
    def add_dshbed(self, dayofsimulation: int, newbednum: int):
        """增加某日的床位，以实现医院的床位调整，
        考虑基本不会减床位，因此只实现床位增加
        :param dayofsimulation: 模拟的第几天
        :param newbednum: 增加的新床位
        :return:
        """
        # 从第几日开始的床位，都增加。
        # 先只针对定点医院
        self.dshospital.modelResult.everybeds[dayofsimulation:] += newbednum
        print("第" + str(dayofsimulation) + "日增加" + str(newbednum) + "床位")

    def add_mchbed(self, dayofsimulation: int, newbednum: int):
        """增加某日的床位，以实现医院的床位调整，
        考虑基本不会减床位，因此只实现床位增加
        :param dayofsimulation: 模拟的第几天
        :param newbednum: 增加的新床位
        :return:
        """
        # 从第几日开始的床位，都增加。
        # 先只针对定点医院
        self.mchospital.modelResult.everybeds[dayofsimulation:] += newbednum
        print("第" + str(dayofsimulation) + "日增加" + str(newbednum) + "床位")


class OnlyDsHospital(Hospital):
    # 只有定点医院，即只接受重症

    dshospital = None
    dshbednum = 0
    empty_beds = []

    def __init__(self):

        super().__init__()
        self.hospitaltype = HospitalType.ONLY_DS.value

        # 建立定点医院
        # 定点医院初始化
        dshmodelparam = hospitalmain.HospitalModelParam()
        dshrunparam = hospitalmain.HospitalModelRunParam()
        # 复合医院，两个医院的床位是一样的。
        # 如果需要修改医院床位，则只修改方舱的床位参数。
        dshrunparam.bedstartnum = dshrunparam.bedstartnum
        # 假设定点医院的参数均按照原来的参数进行变化，即有病程变化
        self.dshospital = hospitalmain.HospitalModel(dshmodelparam, dshrunparam)
        self.dshospital.start()
        # 床位的初始化
        self.bednum = dshrunparam.bedstartnum
        self.empty_beds = np.ones(self.dshospital.modelparam.days) * self.bednum

        return

    def acceptpatient(self, dayofsimulation: int, predaydata):

        # 只接收重症病人

        # 如果是第一天, 则不接受病人
        if dayofsimulation == 0:

            dshaccept_patientnum = 0

            dsh_better_num = 0
            dsh_worse_num = 0

            self.dshospital.calbatch(dayofsimulation)
            self.dshospital.addnewbatch(dayofsimulation, dshaccept_patientnum)

            self.cal_empty_bed2(dayofsimulation, dsh_better_num,
                                dsh_worse_num, dshaccept_patientnum)

        else:
            # 流程：应该定点医院先接收重症病人，方舱后接收
            empty_bed = self.empty_beds[dayofsimulation - 1]

            # 先算出要接收院外的几个重症
            if predaydata.SevereMan < empty_bed:
                dshaccept_patientnum = predaydata.SevereMan
                empty_bed -= dshaccept_patientnum
            else:
                print(u"定点医院床位不够接收院外重症了")
                dshaccept_patientnum = empty_bed
                empty_bed = 0

            print(u"今日定点医院接收：" + str(dshaccept_patientnum), end='')

            # 定点医院的新增病人来源为：每日增加进来的重症病人
            print(u"定点医院接收 %d 院外重症 " % dshaccept_patientnum)

            # 开始计算定点医院的每日过程
            dsh_better_num, dsh_worse_num = self.dshospital.calbatch(dayofsimulation)
            self.dshospital.addnewbatch(dayofsimulation, dshaccept_patientnum)

            self.cal_empty_bed2(dayofsimulation, dsh_better_num,
                                dsh_worse_num, dshaccept_patientnum)

        # 定义返回对象
        hosacceptobj = epidemic.HospitalAcceptObj(dshaccept_patientnum, 0)

        return hosacceptobj

    def dsh_acceptpatient(self, dayofsimulate: int, outmodelsevere: int):
        """应用于定点医院的接收病人逻辑
        :param dayofsimulate: 模拟的第几天
        :param mchworse: 方舱医院的恶化病人
        :param outmodelsevere: 院外模型的重症患者
        :return: mchworseaccept ：接收的方舱医院的恶化病人数
        :return: mchwaitaccept：方舱医院的等待接收的病人接收数
        :return: outmodelsevereaccept：院外重症病人接收数
        """
        outmodelsevereaccept = 0

        if dayofsimulate != 0:

            # 考虑床位限制，床位够则全部接收，不够则只接收床位数。
            # 先获得昨天的空床位，
            nowbed = int(self.empty_beds[dayofsimulate - 1])

            # 接收院外重症
            if nowbed != 0:
                if outmodelsevere < nowbed:
                    outmodelsevereaccept = outmodelsevere
                    nowbed -= outmodelsevere
                else:
                    print(u"定点医院床位不够接收院外重症了")
                    outmodelsevereaccept = nowbed
                    # nowbed = 0
            else:
                print(u"定点医院床位不够接收院外重症了")

        # 返回以上三个数据
        return int(outmodelsevereaccept)

    def outputresult(self):
        # 画出院内模型定点医院的结果
        plt.figure()
        plt.title('DE hospital Model result', fontsize='large', fontweight='bold')
        self.dshospital.outputResult()

        # plt.show()

        # 输出汇总的数据。\
        print("定点医院：总人数：%d ,总康复：%d , 总恶化：%d" %
              (self.dshospital.modelResult.population, self.dshospital.modelResult.better_final_cumu_list[-1],
               self.dshospital.modelResult.worse_final_cumu_list[-1]))

        # 总数为两个医院的接受病人数-方舱送到定点医院的人数
        patientsum = self.dshospital.modelResult.population
        # 求出死亡率
        deadpercent = self.dshospital.modelResult.worse_final_cumu_list[-1] / patientsum

        print("院内总人数： %d, 死亡率： %f " % (patientsum, deadpercent))

        return patientsum, deadpercent

    def cal_empty_bed(self):
        # 计算空床位功能


        return

    #
    def cal_empty_bed2(self, dayofsimulate: int, dsh_daily_total_better: int,
                       dsh_daily_total_worse: int, dshacceptpatient: int):
        """定点医院计算出今日的空余床位
        :param dayofsimulate: 模拟的第几天
        :param dsh_daily_total_better: 今日好转的定点医院人数
        :param dsh_daily_total_worse: 今日恶化的定点医院人数
        :param dshacceptpatient: 今日接收的重症新病人
        :return: 无
        """

        # 规则发现不对，不应是每日的，而有人还在占着床位的。需要用昨日的床位来计算
        if dayofsimulate != 0:
            # 空床位为昨天的空床位，再加上今日恶化的，及好转的，再加上接受院外的新病人
            empty_bed = self.empty_beds[dayofsimulate - 1] + dsh_daily_total_better + \
                        dsh_daily_total_worse - dshacceptpatient
        else:
            # 如果是第一天，空床位为起始数量
            empty_bed = self.dshospital.runparam.bedstartnum - self.dshospital.modelparam.start_population

        # 如果床位过多，（有可能昨天空床多，今天走的人也多,适用于开始模拟）
        if empty_bed > self.dshospital.runparam.bedstartnum:
            empty_bed = self.dshospital.runparam.bedstartnum

        self.empty_beds[dayofsimulate] = empty_bed
        print("空床位为" + str(empty_bed))
        return

    # 以下代码还有问题，暂时不使用
    # TODO: 关于如何添加新床位，又不影响原来床位计算逻辑上，还有一些问题。
    def add_dshbed(self, dayofsimulation: int, newbednum: int):
        """增加某日的床位，以实现医院的床位调整，
        考虑基本不会减床位，因此只实现床位增加
        :param dayofsimulation: 模拟的第几天
        :param newbednum: 增加的新床位
        :return:
        """
        # 从第几日开始的床位，都增加。
        # 先只针对定点医院
        self.dshospital.modelResult.everybeds[dayofsimulation:] += newbednum
        print("第" + str(dayofsimulation) + "日增加" + str(newbednum) + "床位")

        # endregion

# endregion


def testmodel():
    """测试模型，测试hospitalManager是否可以正常运行
    也可以用于不需要外面模型，单独院内模型运行。
    :return:
    """

    # 测试方舱和定点医院同一个的
    # hospitalobj = McDsComHospital()
    # 测试方舱和定点医院独立的
    # hospitalobj = McDsIndHospital()

    # 测试只有重症医院的
    hospitalmain.HospitalModelRunParam.oxygen_change_list = [(0, 0.7)]
    hospitalobj = OnlyDsHospital()
    preday_data = epidemic.PerDayData()
    preday_data.MildMan = 50
    preday_data.SevereMan = 50

    for dayofSimulation in range(30):
        # 与院内模型进行交互
        hosacceptobj = hospitalobj.acceptpatient(dayofSimulation, preday_data)

    # 调用院内模型，输出结果
    patientnum,deadercent = hospitalobj.outputresult()
    return

def testmodel2():

    # 测试方舱和定点医院独立的


    patientnums=[]
    deadpercents=[]
    # 测试只有重症医院的
    for i in range(10, 5, -1):
        hospitalmain.HospitalModelRunParam.oxygen_change_list = [(0, i*0.1)]
        # hospitalmain.HospitalModelRunParam.breathing_machine_change_list=[(0, i*0.1)]
        # hospitalobj = McDsComHospital()
        hospitalobj = McDsIndHospital()
        # hospitalobj = OnlyDsHospital()
        preday_data = epidemic.PerDayData()
        preday_data.MildMan = 50
        preday_data.SevereMan = 50

        for dayofSimulation in range(30):
            # 与院内模型进行交互
            hosacceptobj = hospitalobj.acceptpatient(dayofSimulation, preday_data)

        # 调用院内模型，输出结果
        patientnum, deadpercent = hospitalobj.outputresult()
        patientnums.append(patientnum)
        deadpercents.append(deadpercent)

    print("test finish")

def main():

    return

# 主程序，可供测试的范例。
if __name__ == '__main__':
    # main()
    # testmodel()
    testmodel2()
    print("finish")
