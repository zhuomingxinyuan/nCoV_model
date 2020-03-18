import numpy as np 
from scipy import stats
import random
import matplotlib.pyplot as plt
import operator
import copy

### 每日新增概率 ###
def daily_prob(day, mu, std): 
	prob=stats.norm.cdf(day+1, mu, std) - stats.norm.cdf(day, mu, std)
	return prob

### 累计概率 ###
def daily_cumu(day, mu, std):
	prob=stats.norm.cdf(day+1, mu, std)
	return prob

### 生成默认概率曲线集 ####
def batch_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std):
	better_list=[] # daily increment, following normal distribution
	worse_list=[] # daily increment, following normal distribution
	better_cumu_list=[] # cumulative record, following the shape of a cumulative density function
	worse_cumu_list=[] # cumulative record, following the shape of a cumulative density function
	remain_list=[] # those who remain in the same medical status
	dates=list(range(days))
	better_cumu=0
	worse_cumu=0
	remain=batch_list[0]
	for t in dates:
		better=daily_prob(t, better_mean, better_std)*better_prob_list[t]
		worse=daily_prob(t, worse_mean, worse_std)*worse_prob_list[t]
		better_cumu+=better # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		worse_cumu+=worse # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		remain=batch_list[t]-better_cumu-worse_cumu
		better_list.append(better)
		worse_list.append(worse)
		better_cumu_list.append(better_cumu)
		worse_cumu_list.append(worse_cumu)
		remain_list.append(remain)

	return dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, batch_list

#### 画图 ####
def plot_curves(dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, cure_ability_list, batch_list):
	plt.plot(dates, cure_ability_list*batch_list[0], color='green', marker='o', markersize=1, linestyle='-', label='Medical_supply')
	plt.plot(dates, better_list, color='grey', marker='o', markersize=3, linestyle='--', label='Better_daily(mean=10,std=5)')
	plt.plot(dates, worse_list, color='red', marker='o', markersize=3, linestyle='--', label='Worse_daily(mean=7,std=3)')
	plt.plot(dates, better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-', label='Better_cumu(mean=10,std=5)')
	plt.plot(dates, worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-', label='Worse_cumu(mean=7,std=3)')
	plt.plot(dates, remain_list, color='yellow', marker='o', markersize=3, linestyle='-', label='Remain population')
	plt.plot(dates, batch_list, color='blue', marker='o', markersize=3, linestyle='-',label='Batch population')
	plt.xlabel('Days')
	plt.ylabel('probability')
	plt.legend(loc='upper right')
	plt.show()
	plt.close()
	return

#### 更新同期病人在院外的剩余人数 #####
def intervention_take_out_patients(date, take_number, batch_list):
	for i in range(date, len(batch_list)):
		batch_list[i]-=take_number
	return batch_list 

#### 院外模型中可以在oop中使用的，同期病人部分被医院收治后，每日好转和恶化的概率更新
def intervention_take_out_patients_change_prob_lists(date, remain_before_date_prob, better_prob_list, worse_prob_list):
	for i in range(date, len(better_prob_list)):
		better_prob_list[i]*=remain_before_date_prob # 条件概率的应用，P(A|B)*P(B)= P(B|A)*P(A) = P()
	for i in range(date, len(worse_prob_list)):
		worse_prob_list[i]*=remain_before_date_prob # 条件概率的应用
	return better_prob_list, worse_prob_list

# 题：如果医院第10天从这批100个院外轻症患者中收治了20 个人，那么剩下的80人在第10天之后，每天多少人转重症，多少人转自愈？
# 设：A=第10天转， B=前9天没有转
# P( 第10天转 | 前9天没有转 ) x P(前9天没有转) = P( 前9天没有转 | 第10天转 ) x P(第10天转)
# 因为所有第10天转的人，前9天一定没有转，所以 P( 前9天没有转 | 第10天转 ) = 1 , 所以
# P( 第10天转 | 前9天没有转 ) x P(前9天没有转) = 1 x P(第10天转) = 想要的答案


#### 针对一次医力变化，对之后的好转和恶化概率做list更新 ####
def intervention_hospital_supply_change_step(date, new_better_prob_ratio, new_worse_prob_ratio, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original):
	for i in range(date, len(better_prob_list)):
		better_prob_list[i]=better_prob_list_original[i]*new_better_prob_ratio
	for i in range(date, len(worse_prob_list)):
		worse_prob_list[i]=worse_prob_list_original[i]*new_worse_prob_ratio
	return better_prob_list, worse_prob_list

#### 综合每日氧气供应率和每日呼吸机供应率list，计算每日医力，以及需要更新好转和恶化率的日期 ####
def cal_cure_ability(breathing_machine_supply_list, oxygen_supply_list, oxygen_change_list, breathing_machine_change_list):

	for oxygen_change_date, oxygen_supply_rate in oxygen_change_list:
		for i in range(oxygen_change_date, len(oxygen_supply_list)):
			oxygen_supply_list[i]=oxygen_supply_rate
	# print("oxygen_supply_list", oxygen_supply_list)
	for breathing_machine_change_date, breathing_machine_supply_rate in breathing_machine_change_list:
		for i in range(breathing_machine_change_date, len(breathing_machine_supply_list)):
			breathing_machine_supply_list[i]=breathing_machine_supply_rate
	# print("breathing_machine_supply_list", breathing_machine_supply_list)

	change_dates=[x[0] for x in oxygen_change_list]+[x[0] for x in breathing_machine_change_list] 
	change_dates.sort()
	change_dates=np.unique(change_dates)
	# print(change_dates)

	cure_ability_list=oxygen_supply_list*breathing_machine_supply_list   
	# print(cure_ability_list)

	return cure_ability_list, change_dates

#### 模拟周期内，医院医力多次变化，计算更新后的每日好转与恶化的概率list，需要保证总概率=1，单项概率>=0 ####
def intervention_hospital_supply_change_repeated(cure_ability_list, change_dates, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original, better_mean, better_std, worse_mean, worse_std, better_prob, worse_prob):

	better_bef_date_prob=(stats.norm.cdf(change_dates[0], better_mean, better_std))*better_prob
	worse_bef_date_prob=(stats.norm.cdf(change_dates[0], worse_mean, worse_std))*worse_prob
	# print("\n")

	for index, t in enumerate(change_dates):
		# print("########## change", index, "on day", t, "to cure_ability =", cure_ability_list[t], "#############\n")
		# print()

		new_better_prob=better_prob*cure_ability_list[t]
		better_after_date_prob=(1-stats.norm.cdf(t, better_mean, better_std))*new_better_prob
		# print("better_bef_date_prob = ", better_bef_date_prob)
		# print("better_after_date_prob = " , better_after_date_prob)
		# print("better_total_prob = ",  better_after_date_prob+better_bef_date_prob)

		# print("\n")

		worse_after_date_prob=1-better_bef_date_prob-worse_bef_date_prob-better_after_date_prob
		# print("worse_bef_date_prob = ", worse_bef_date_prob)
		# print("worse_after_date_prob = " , worse_after_date_prob)
		# print("worse_total_prob = ", worse_after_date_prob+worse_bef_date_prob)

		### 万一后期医力突然上升，结果没有这么多人来让我医治了，为了避免恶化率变成负数，设恶化率为0，然后反推实际可以有的治愈率。###
		if worse_after_date_prob<0 : 
			worse_after_date_prob = 0
			better_after_date_prob = 1-better_bef_date_prob-worse_bef_date_prob
			new_better_prob = better_after_date_prob / (1-stats.norm.cdf(t, better_mean, better_std))

		# print("\n")

		new_better_prob_ratio=new_better_prob/better_prob
		# print("new_better_prob_ratio = ", new_better_prob_ratio)
		new_worse_prob=worse_after_date_prob/(1-stats.norm.cdf(t, worse_mean, worse_std))
		new_worse_prob_ratio=new_worse_prob/worse_prob
		# print("new_worse_prob_ratio = ", new_worse_prob_ratio)
		# print("new_better_prob = ", new_better_prob)
		# print("new_worse_prob = ", new_worse_prob)
		better_prob_list, worse_prob_list=intervention_hospital_supply_change_step(t, new_better_prob_ratio, new_worse_prob_ratio, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original) # The ratio 1.4, 0.2 are arbitrary for now and will be calculated based on breathing machine and oxygen supplies. 
		# print("better_prob_list = ", better_prob_list)
		# print("worse_prob_list = ", worse_prob_list)
		if index+1<len(change_dates):
			better_bef_date_prob+=(stats.norm.cdf(change_dates[index+1], better_mean, better_std)-stats.norm.cdf(t, better_mean, better_std))*new_better_prob
			worse_bef_date_prob+=(stats.norm.cdf(change_dates[index+1], worse_mean, worse_std)-stats.norm.cdf(t, worse_mean, worse_std))*new_worse_prob
		
		# print("\n")

	return better_prob_list, worse_prob_list

### 每日新增的同期病人自己开启一个新曲线集 ###
def initialise_batch(days, batch_population, better_prob, worse_prob):
	batch_list=np.ones(days)*batch_population
	#print("batch_list = ", batch_list)
	better_prob_list=batch_list*better_prob
	better_prob_list_original=copy.copy(better_prob_list)
	#print("better_prob_list = ", better_prob_list)
	worse_prob_list=batch_list*worse_prob
	worse_prob_list_original=copy.copy(worse_prob_list)
	#print("worse_prob_list = ", worse_prob_list)
	# cure_ability_list=np.ones(days) # dummy, so that the final plotting function doesn't complain.
	return batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original

### 让医力的变化反应在不同批次新增病人的正确病程日上 ###
def shift_cure_ability_list(cure_ability_list_original, change_dates_original, batch_counter):
	change_dates=[x-batch_counter for x in change_dates_original]
	change_dates = [i for i in change_dates if i>=0] # keep only non-zero change_dates
	head=cure_ability_list_original[batch_counter:] # 截取 batch counter 日期之后的医力list
	tail=[head[-1]] * batch_counter # 为了让list跟原来一样长， 把最后一位数字复制几遍，补全尾巴
	cure_ability_list=np.concatenate((head, tail))
	if len(change_dates)==0:
		change_dates=[0]
	return cure_ability_list, change_dates

### 代入原始参数，连续跑几天几批病人，输出为模拟最后一日各批病人好转和恶化的总和 ###
def run_hospital(days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population, number_of_batches, oxygen_change_list, breathing_machine_change_list):
	### 准备医力变化总体曲线 ###
	cure_ability_list=np.ones(days)
	change_dates=[0]
	breathing_machine_supply_list=np.ones(days)
	oxygen_supply_list=np.ones(days)
	cure_ability_list, change_dates = cal_cure_ability(breathing_machine_supply_list, oxygen_supply_list, oxygen_change_list, breathing_machine_change_list)
	cure_ability_list_original=copy.copy(cure_ability_list)
	change_dates_original=copy.copy(change_dates)	
	### 准备 ###
	batch_counter=0
	better_dict={}
	worse_dict={}
	better_cumu_dict={}
	worse_cumu_dict={}
	better_final_list=np.zeros(days)
	worse_final_list=np.zeros(days)
	better_final_cumu_list=np.zeros(days)
	worse_final_cumu_list=np.zeros(days)
	empty_bed=batch_population
	daily_total_better=0 # sum up total bed release for the day from all batches of patients
	daily_total_worse=0
	batch_population_list=[]

	### 画图 ###
	fig, axs = plt.subplots(nrows=number_of_batches+1, sharex=True)
	plt.xlim(0, days-1)
	for ax in axs: 
	# for batch_counter in range(number_of_batches): #如果后来决定不用每次都画图，可以用这一行来带for loop，然后下面turn off画图的if else部分。
		### last row of subplot 计算多期病人曲线总和，画在最下面的一个画框里 ### 
		if batch_counter==number_of_batches: 

			better_final_list=np.zeros(days)
			worse_final_list=np.zeros(days)
			better_final_list=np.zeros(days)
			worse_final_list=np.zeros(days)
			for j in range(1, days):
				daily_total_better=0
				daily_total_worse=0
				for i in range(0, batch_counter): # range()出的list不带最后一个值，所以不用加1
					daily_total_worse+=worse_dict[i][j-i]
					daily_total_better+=better_dict[i][j-i]
				better_final_list[j]=daily_total_better
				worse_final_list[j]=daily_total_worse
				better_final_cumu_list[j]=sum(better_final_list)
				worse_final_cumu_list[j]=sum(worse_final_list)
			dates=list(range(days))
			ax.plot(dates, better_final_list, color='grey', marker='o', markersize=3, linestyle='--', label='Better_sum_daily(mean=10,std=5)')
			ax.plot(dates, worse_final_list, color='red', marker='o', markersize=3, linestyle='--', label='Worse_sum_daily(mean=7,std=3)')
			ax.plot(dates, better_final_cumu_list, color='grey', marker='o', markersize=3, linestyle='-', label='Better_sum_cumu(mean=10,std=5)')
			ax.plot(dates, worse_final_cumu_list, color='red', marker='o', markersize=3, linestyle='-', label='Worse_sum_cumu(mean=7,std=3)')
			ax.set_ylabel("Total")
			ax.set_xlabel("Dates")
		else:	### all but the last row of subplot 每日病人曲线集，按天从上往下画。
			
			batch_population=empty_bed #目前假设医院每天都是满员运转。考虑不满员，就在这里对比一下排队人数，然后注意算对剩下的空床，明天跟新空的床一起让排队病人进来。
			batch_population_list.append(batch_population)
			### 启动一批病人 ###
			batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original = initialise_batch(days, batch_population, better_prob, worse_prob)
			### 让医力的变化在正确的同一天，一起影响到不同批次、已经进入到病程不同天的病人 ###
			cure_ability_list, change_dates = shift_cure_ability_list(cure_ability_list_original, change_dates_original, batch_counter)
			better_prob_list, worse_prob_list = intervention_hospital_supply_change_repeated(cure_ability_list, change_dates, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original, better_mean, better_std, worse_mean, worse_std, better_prob, worse_prob)
			dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, batch_list=batch_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std)
			### 用词典记录每个批次的数据 ###
			better_dict[batch_counter]=better_list
			worse_dict[batch_counter]=worse_list
			better_cumu_dict[batch_counter]=better_cumu_list
			worse_cumu_dict[batch_counter]=worse_cumu_list
			### 计算不同批次病人在同一天好转和恶化的总数 ###
			daily_total_better=0 # sum up total bed release for the day from all batches of patients
			daily_total_worse=0
			for i in range(0, batch_counter+1): # range()出的list不带最后一个值，所以要加1
				daily_total_worse+=worse_dict[i][batch_counter-i]
				daily_total_better+=better_dict[i][batch_counter-i]
			empty_bed=daily_total_worse+daily_total_better
			# print("第", batch_counter, "天")
			# print("今日康复", daily_total_better)
			# print("今日死亡", daily_total_worse)


			better_final_list[batch_counter]=daily_total_better
			worse_final_list[batch_counter]=daily_total_worse
			better_final_cumu_list[batch_counter]=sum(better_final_list)
			worse_final_cumu_list[batch_counter]=sum(better_final_list)

			dates=[x + batch_counter for x in dates] 
			ax.plot(dates, cure_ability_list*batch_population, color='green', marker='o', markersize=3, linestyle='-', label='Medical_supply')
			ax.plot(dates, better_list, color='grey', marker='o', markersize=3, linestyle='--', label='Better_daily(mean=10,std=5)')
			ax.plot(dates, worse_list, color='red', marker='o', markersize=3, linestyle='--', label='Worse_daily(mean=7,std=3)')
			ax.plot(dates, better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-', label='Better_cumu(mean=10,std=5)')
			ax.plot(dates, worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-', label='Worse_cumu(mean=7,std=3)')
			ax.plot(dates, remain_list, color='yellow', marker='o', markersize=3, linestyle='-', label='Remain population')
			ax.plot(dates, batch_list, color='blue', marker='o', markersize=3, linestyle='-',label='Batch population')
			ax.set_ylabel("Day {}".format(batch_counter))
			batch_counter+=1
	
	# print("第", number_of_batches, "天") #这个天数比batch counter要大1，就是人看着舒服点，其实第一天是在说从第0到第1天这24个小时，在代码运算里是第0天。
	# print("今日康复", daily_total_better)
	# print("今日死亡", daily_total_worse)
	# print("累计康复列表 = ", better_final_cumu_list)
	# print("累计死亡列表 = ", worse_final_cumu_list)
	# print("收治新增人数列表 = ", batch_population_list)

	# plt.show()
	# plt.close()
	return daily_total_better, daily_total_worse, better_final_cumu_list, worse_final_cumu_list, batch_population_list

### 单批同期病人的病情曲线，可以有很多天 ###
def run_single_batch(days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population, number_of_batches, oxygen_change_list, breathing_machine_change_list):
	breathing_machine_supply_list=np.ones(days) #默认实力全开
	oxygen_supply_list=np.ones(days) #默认实力全开
	batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original = initialise_batch(days,batch_population)
	cure_ability_list, change_dates = cal_cure_ability(breathing_machine_supply_list, oxygen_supply_list, oxygen_change_list, breathing_machine_change_list)
	better_prob_list, worse_prob_list = intervention_hospital_supply_change_repeated(cure_ability_list, change_dates, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original, better_mean, better_std, worse_mean, worse_std, better_prob, worse_prob)
	dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, batch_list=batch_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std)
	plot_curves(dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, cure_ability_list, batch_list)
	return

if __name__ == '__main__':
	#### 必要参数 #########
	date_list=[]
	better_perc=[]
	worse_perc=[]
	f = open("hospital_tests.txt","w") 
	f.write("模拟天数  参数治愈率  总收治人数  总治愈   总死亡 \n")
	# for days in range(1,51): #用for loop可以让医院一天一天跑，每天把结果输出。range最低1天，0天跑不了，但可以从多天起跑。下面氧气和呼吸机供应率的更新日期最大可以是days-1日， 而且理论上应该根据每日收治的病人情况来每日更新计算。
	days=50
	worse_rate_list=[]
	for worse_prob in np.arange(0.1, 1, 0.1):
		worse_rate_list.append(worse_prob)
		# date_list.append(days)
		# worse_prob=0.1
		better_prob=1-worse_prob
		better_mean=10
		better_std=5
		worse_mean=7
		worse_std=3
		batch_population=100 # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
		number_of_batches = days # 假设每天新增一批病人，
		oxygen_change_list=[] #(0,0.3), (5,0.5), (9,1) (1,0.6), (9, 0.8) #tuple list, with first element = date the change starts, second element=updated oxygen supply rate. List order doesn't matter
		breathing_machine_change_list=[] #(7,0.4), (10,1), (15,0.5)(4,0.8)#tuple list, with first element = date the change starts, second element=updated machine supply rate. List order doesn't matter
		       

		### 单批同期病人的病情曲线，可以有很多天 ###
		# run_single_batch(days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population, number_of_batches, oxygen_change_list, breathing_machine_change_list)
		
		### 医院每天结算一次空床，每天收治一批病人 ### 目前的简化院内模型中，院内只有一种病人，要么出院，要么死亡，没有中间过渡病程。
		daily_total_better, daily_total_worse, better_final_cumu_list, worse_final_cumu_list, batch_population_list=run_hospital(days, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std, batch_population, number_of_batches, oxygen_change_list, breathing_machine_change_list)
		# print("daily_total_better = ", daily_total_better)
		# print("daily_total_worse = ", daily_total_worse)
		print("模拟天数 = ", days)
		print("收治新增人数列表 = ", batch_population_list)
		print("治愈率 = ", better_prob)
		print("阶段治愈人数 = ", better_final_cumu_list[-1])
		print("阶段死亡人数 = ", worse_final_cumu_list[-1])
		print("收治总人数 = ", sum(batch_population_list))
		print("阶段治愈率 = ", better_final_cumu_list[-1]/sum(batch_population_list))
		print("阶段死亡率 = ", worse_final_cumu_list[-1]/sum(batch_population_list))
		print("\n")

		better_perc.append(better_final_cumu_list[-1]/sum(batch_population_list))
		worse_perc.append(worse_final_cumu_list[-1]/sum(batch_population_list))
		f.write("{:d} {:.1f} {:.1f} {:.1f} {:.1f} \n".format(days,  better_prob,  sum(batch_population_list),  better_final_cumu_list[-1],   worse_final_cumu_list[-1]))
	f.close()
	
	print("worse_rate_list = ", worse_rate_list)
	print("better_perc = ", better_perc)
	print("worse_perc = ", worse_perc)

	# plt.plot(days, better_perc, color='grey', marker='o', markersize=3, linestyle='-', label='better%') 
	# plt.plot(days, worse_perc, color='red', marker='o', markersize=3, linestyle='-', label='worse%')     
	# plt.xlabel('Days')
	# plt.ylabel('Probability') 
	# plt.legend(loc='upper right')
	# plt.title('Better_prob=0.1')
	# plt.show()  



    ### 如果用oop，可以用当日概率掷骰子
	# probability_dict={"better": 0.2, "worse": 0.3}
	# change_state(probability_dict)