from scipy.stats import norm
import random
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #SimHei是黑体的意思


from classes import patient

patient_list=[]
num_cured_list=[]#list for number of cured patient each day
num_mild_list=[]#list for num of mild patient each day
num_severe_list=[]# list for num of severe patient each day
#insert a list of patients object
for i in np.arange(0,10000):
    patient_list.append(patient(1,0,10,5,7,3))

# now we simulate 10 days
# for each day, we can just call "evolve" for each patient
for i in np.arange(0,40):
    for j in patient_list:
        j.evolve()

    cured_patient=[i for i in patient_list if i.state==0]
    mild_patient=[i for i in patient_list if i.state==1]
    severe_patient=[i for i in patient_list if i.state==2]
    num_cured_list.append(len(cured_patient))
    num_mild_list.append(len(mild_patient))
    num_severe_list.append(len(severe_patient))




num_cured_increase=[num_cured_list[i+1]-num_cured_list[i] for i in np.arange(0,len(num_cured_list)-1)]
num_cured_increase.insert(0,0)
num_severe_increase=[num_severe_list[i+1]-num_severe_list[i] for i in np.arange(0,len(num_severe_list)-1)]
num_severe_increase.insert(0,0)
plt.plot(np.array(num_cured_increase)/10000,'--',label=u"每日新增康复")
plt.plot(np.array(num_severe_increase)/10000,'--',label=u"每日新增重症")
plt.plot(np.array(num_mild_list)/10000,label=u"轻症总人数占比")
plt.plot(np.array(num_cured_list)/10000,label=u"康复总人数占比(mean=10,std=5)")
plt.plot(np.array(num_severe_list)/10000,label=u"重症总人数占比(mean=7,std=3)")
plt.xlabel("天数（天）")
plt.ylabel("占比")
plt.legend()
plt.title(u"任意一批同期病人，病情好转和恶化的概率演化")
plt.show()




# 接下来模拟第10天抽走100人：
patient_list=[]
num_cured_list=[]#list for number of cured patient each day
num_mild_list=[]#list for num of mild patient each day
num_severe_list=[]# list for num of severe patient each day
num_total_list=[]
#insert a list of patients object
for i in np.arange(0,10000):
    patient_list.append(patient(1,0,10,5,7,3))

# now we simulate 10 days
# for each day, we can just call "evolve" for each patient
for i in np.arange(0,40):
    # 第五天抽走20%人：
    if i == 10:
        list_to_del = mild_patient[0:2000]
        for i in list_to_del:
            mild_patient.remove(i)
            patient_list.remove(i)
    for j in patient_list:
        j.evolve()


    cured_patient=[i for i in patient_list if i.state==0]
    mild_patient=[i for i in patient_list if i.state==1]
    severe_patient=[i for i in patient_list if i.state==2]
    num_cured_list.append(len(cured_patient))
    num_mild_list.append(len(mild_patient))
    num_severe_list.append(len(severe_patient))
    num_total_list.append(len(patient_list))


num_cured_increase=[num_cured_list[i+1]-num_cured_list[i] for i in np.arange(0,len(num_cured_list)-1)]
num_cured_increase.insert(0,0)
num_severe_increase=[num_severe_list[i+1]-num_severe_list[i] for i in np.arange(0,len(num_severe_list)-1)]
num_severe_increase.insert(0,0)

plt.plot(np.array(num_cured_increase)/10000,'--',label=u"每日新增康复")
plt.plot(np.array(num_mild_list)/10000,label=u"轻症总人数")
plt.plot(np.array(num_severe_increase)/10000,'--',label=u"每日新增重症")
plt.plot(np.array(num_cured_list)/10000,label=u"康复总人数占比(mean=10,std=5)")
plt.plot(np.array(num_severe_list)/10000,label=u"重症总人数占比(mean=7,std=3)")
plt.plot(np.array(num_total_list)/10000,label=u"院外总人数")
plt.xlabel("天数（天）")
plt.ylabel("占比")
plt.legend(loc="upper right")
plt.title(u"院外模型干预，医院第10天收治20人")
plt.show()






