import numpy as np 
from scipy import stats
import random
import matplotlib.pyplot as plt
import operator


def daily_prob(day, mu, std): 
	prob=stats.norm.cdf(day+1, mu, std) - stats.norm.cdf(day, mu, std)
	return prob

def daily_cumu(day, mu, std):
	prob=stats.norm.cdf(day, mu, std) - stats.norm.cdf(0, mu, std)
	return prob

def outside_batch(days, batch_list, take_list, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std):
	better_list=[] # daily increment, following normal distribution
	worse_list=[] # daily increment, following normal distribution
	better_cumu_list=[] # cumulative record, following the shape of a cumulative density function
	worse_cumu_list=[] # cumulative record, following the shape of a cumulative density function
	remain_list=[] # those who remain in the same medical status
	dates=list(range(days))
	better_cumu=0
	worse_cumu=0
	remain=batch_list[0]
	###### update batch_list with take_list #######
	for i in range(0, len(take_list)):
		# print(i)
		for j in range(i, days):
			# print(j)
			batch_list[j]-=take_list[i]
	print("batch_list = ", batch_list)
	###### calculate daily new better and new worse and cumulative and remaining ###########
	for t in dates:
		if remain-take_list[t]>0 :
			remain-=take_list[t]
			better=remain * daily_prob(t, better_mean, better_std) * better_prob / (1-daily_cumu(t, better_mean, better_std)*better_prob-daily_cumu(t, worse_mean, worse_std) * worse_prob)
			worse=remain * daily_prob(t, worse_mean, worse_std) * worse_prob / (1-daily_cumu(t, better_mean, better_std)*better_prob-daily_cumu(t, worse_mean, worse_std) * worse_prob)
		else:
			remain=0
			better=0
			worse=0
		better_cumu+=better # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		worse_cumu+=worse # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		remain-=better+worse
		better_list.append(better)
		worse_list.append(worse)
		better_cumu_list.append(better_cumu)
		worse_cumu_list.append(worse_cumu)
		remain_list.append(remain)
	plt.plot(dates, better_list, color='grey', marker='o', markersize=3, linestyle='--', label='Better_daily(mean=10,std=5)')
	plt.plot(dates, worse_list, color='red', marker='o', markersize=3, linestyle='--', label='Worse_daily(mean=7,std=3)')
	plt.plot(dates, better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-', label='Better_cumu(mean=10,std=5)')
	plt.plot(dates, worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-', label='Worse_cumu(mean=7,std=3)')
	plt.plot(dates, remain_list, color='yellow', marker='o', markersize=3, linestyle='-', label='Remain population')
	plt.plot(dates, batch_list, color='blue', marker='o', markersize=3, linestyle='-',label='Batch population')
	plt.xlabel('Days')
	plt.ylabel('Batch patient number')
	plt.legend(loc='upper right')
	plt.show()
	plt.close()
	return better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list


if __name__ == '__main__':
	days=40
	worse_prob=0.3
	better_prob=1-worse_prob
	better_mean=10
	better_std=5
	worse_mean=7
	worse_std=3
	batch_population=1000 # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
	batch_list=np.ones(days)*batch_population
	# print("batch_list = ", batch_list)

	#### 医院抽走人数 测试用 ########## 正式要靠院内模型 ##########
	take_list=np.zeros(days) #list(range(days))
	take_tuple_list=[(2,10),(5,300),(6,50),(10,20),(15,50)] #第5天抽走300人，第10天抽走20人。
	for take_date, take_number in take_tuple_list:
		take_list[take_date]=take_number   ### take list 在plot_curve里用来从remain里减去黄线。
    #### 
	better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list=outside_batch(days, batch_list, take_list, better_prob, worse_prob, better_mean, better_std, worse_mean, worse_std)
	print("remain_list = ", remain_list)
	print("better_cumu_list = ", better_cumu_list)
	print("worse_cumu_list = ", worse_cumu_list)
	probability_dict={"better": 0.2, "worse": 0.3}
	change_state(probability_dict)