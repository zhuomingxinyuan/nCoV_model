import numpy as np 
from scipy import stats
import random
import matplotlib.pyplot as plt
import operator

def change_state(probability_dict:dict): # probability_dict={'better': 0.2, 'worse': 0.3}
	dice=random.random()
	print("roll dice = ", dice)
	prob_better=probability_dict["better"]
	prob_worse=prob_better + probability_dict["worse"]
	if dice <= prob_better:
		state="better"
	elif dice <= prob_worse:
		state="worse"
	else:
		state="remain"
	print("result state is ", state)
	return state


def daily_prob(day, mu, std): 
	prob=stats.norm.cdf(day+1, mu, std) - stats.norm.cdf(day, mu, std)
	return prob

def daily_cumu(day, mu, std):
	prob=stats.norm.cdf(day+1, mu, std)
	return prob

def plot_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std):
	better_list=[]
	worse_list=[]
	better_cumu_list=[]
	worse_cumu_list=[]
	# remain_list=[]
	dates=list(range(days))
	better_cumu=0
	worse_cumu=0
	for t in dates:
		better=daily_prob(t, better_mean, better_std)*better_prob_list[t]
		worse=daily_prob(t, worse_mean, worse_std)*worse_prob_list[t]
		better_cumu+=better # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		worse_cumu+=worse # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		# remain=batch_list[t]-better_cumu-worse_cumu
		better_list.append(better)
		worse_list.append(worse)
		better_cumu_list.append(better_cumu)
		worse_cumu_list.append(worse_cumu)
		# remain_list.append(remain)
	# print("better_list = " , better_list)
	# print("worse_list = ", worse_list)
	out_list=list(map(operator.add, better_cumu_list, worse_cumu_list))
	remain_list=list(map(operator.sub, batch_list, out_list))
	plt.plot(dates, better_list, color='grey', marker='o', markersize=3, linestyle='--', label='Better_daily(mean=7,std=5)')
	plt.plot(dates, worse_list, color='red', marker='o', markersize=3, linestyle='--', label='Worse_daily(mean=7,std=3)')
	plt.plot(dates, better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-', label='Better_cumu(mean=10,std=5)')
	plt.plot(dates, worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-', label='Worse_cumu(mean=7,std=3)')
	plt.plot(dates, remain_list, color='yellow', marker='o', markersize=3, linestyle='-', label='Remain population')
	plt.plot(dates, batch_list, color='blue', marker='o', markersize=3, linestyle='-',label='Batch population')
	plt.xlabel('Days')
	plt.ylabel('Probability')
	plt.legend(loc='upper right')
	plt.show()
	plt.close()
	return better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list

def intervention_take_out_patients(date, take_number, batch_list):
	for i in range(date, len(batch_list)):
		# ratio=(batch_list[i]-take_number)/batch_list[i]
		batch_list[i]-=take_number
		# better_prob_list[i]*=ratio
		# worse_prob_list[i]*=ratio
	return batch_list #, better_prob_list, worse_prob_list


if __name__ == '__main__':
	days=40
	better_prob=0.7
	worse_prob=0.3
	better_mean=10
	better_std=5
	worse_mean=7
	worse_std=3
	batch_population=100
	batch_list=np.ones(days)*batch_population
	print("batch_list = ", batch_list)
	batch_list=intervention_take_out_patients(10, 20, batch_list)
	print("new batch_list = ", batch_list)
	better_prob_list=batch_list*better_prob
	print("better_prob_list = ", better_prob_list)
	worse_prob_list=batch_list*worse_prob
	print("worse_prob_list = ", worse_prob_list)
	better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list=plot_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std)
	print("remain_list = ", remain_list)
	# probability_dict={"better": 0.2, "worse": 0.3}
	# change_state(probability_dict)