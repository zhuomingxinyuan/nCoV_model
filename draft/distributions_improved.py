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
	prob=stats.norm.cdf(day, mu, std) - stats.norm.cdf(0, mu, std)
	return prob

def plot_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std):
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
		if remain>0 :
			better=daily_prob(t, better_mean, better_std)*better_prob_list[t]
			worse=daily_prob(t, worse_mean, worse_std)*worse_prob_list[t]
		else:
			better=0
			worse=0
		better_cumu+=better # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		worse_cumu+=worse # set as a counter instead of recalculating cumulative probability for easy implementation of interventions.
		remain=batch_list[t]-better_cumu-worse_cumu
		better_list.append(better)
		worse_list.append(worse)
		better_cumu_list.append(better_cumu)
		worse_cumu_list.append(worse_cumu)
		remain_list.append(remain)
	# print("better_list = " , better_list)
	# print("worse_list = ", worse_list)
	# out_list=list(map(operator.add, better_cumu_list, worse_cumu_list))
	# remain_list=list(map(operator.sub, batch_list, out_list))
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

def intervention_take_out_patients(date, take_number, batch_list):
	for i in range(date, len(batch_list)):
		batch_list[i]-=take_number
	return batch_list 

def intervention_take_out_patients_change_prob_lists(date, remain_before_date_prob, better_prob_list, worse_prob_list):
	for i in range(date, len(better_prob_list)):
		better_prob_list[i]*=remain_before_date_prob
	for i in range(date, len(worse_prob_list)):
		worse_prob_list[i]*=remain_before_date_prob
	return better_prob_list, worse_prob_list


# def intervention_hospital_supply_increase(date, new_better_prob_ratio, new_worse_prob_ratio, better_prob_list, worse_prob_list):
# 	for i in range(date, len(better_prob_list)):
# 		better_prob_list[i]*=new_better_prob_ratio
# 	for i in range(date, len(worse_prob_list)):
# 		worse_prob_list[i]*=new_worse_prob_ratio
# 	return better_prob_list, worse_prob_list


# def cal_new_better_worse_ratios(breathing_machine_supply_rate, oxygen_supply_rate, better_prob, worse_prob):
# 	new_better_prob=better_prob*breathing_machine_supply_rate*oxygen_supply_rate
# 	new_worse_prob=1-new_better_prob
# 	new_better_prob_ratio=new_better_prob/better_prob
# 	new_worse_prob_ratio=new_worse_prob/worse_prob
# 	return new_better_prob_ratio, new_worse_prob_ratio


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
	print("batch_list = ", batch_list)

	############ this block takes out patients (can toggle on and off, but has to be here, don't change order) #####################################
	take_out_date=2
	take_out_number=50
	batch_list=intervention_take_out_patients(take_out_date, take_out_number, batch_list) # how intervention happens once. Need to put in a loop to intervene everyday.
	print("new batch_list = ", batch_list)
	################# this block applies conditional probability after patients taken out (can toggle on and off, but has to be here, don't change order) ################################
	better_prob_list=batch_list*better_prob
	print("better_prob_list = ", better_prob_list)
	worse_prob_list=batch_list*worse_prob
	print("worse_prob_list = ", worse_prob_list)

	better_before_date_prob=daily_cumu(take_out_date-1, better_mean, better_std)*better_prob
	worse_before_date_prob=daily_cumu(take_out_date-1, worse_mean, worse_std)*worse_prob
	# better_before_date_prob=(stats.norm.cdf(take_out_date-1, better_mean, better_std)-stats.norm.cdf(0, better_mean, better_std))*better_prob
	# worse_before_date_prob=(stats.norm.cdf(take_out_date-1, worse_mean, worse_std)-stats.norm.cdf(0, worse_mean, worse_std))*worse_prob
	remain_before_date_prob=1-better_before_date_prob-worse_before_date_prob #, remain_before_date_prob
	better_prob_list, worse_prob_list=intervention_take_out_patients_change_prob_lists(take_out_date, remain_before_date_prob, better_prob_list, worse_prob_list)
	print("new better_prob_list = ", better_prob_list)
	print("new worse_prob_list = ", worse_prob_list)
	# ############ this block changes hospital capability  (can toggle on and off, but has to be here, don't change order) #################################
	# breathing_machine_supply_rate=0.2
	# oxygen_supply_rate=1
	# supply_change_date=10
	# new_better_prob=better_prob*breathing_machine_supply_rate*oxygen_supply_rate
	# ################# this block calculates the theoretical new worse prob by enforcing constant population #########################
	# better_before_date_prob=(stats.norm.cdf(supply_change_date, better_mean, better_std)-stats.norm.cdf(0, better_mean, better_std))*better_prob
	# worse_before_date_prob=(stats.norm.cdf(supply_change_date, worse_mean, worse_std)-stats.norm.cdf(0, worse_mean, worse_std))*worse_prob
	
	# better_after_date_prob=(better_prob-better_before_date_prob)*new_better_prob/better_prob
	# worse_after_date_prob=1-better_before_date_prob-worse_before_date_prob-better_after_date_prob

	# new_better_prob_ratio=breathing_machine_supply_rate*oxygen_supply_rate
	# new_worse_prob_ratio=worse_after_date_prob/(worse_prob-worse_before_date_prob)

	# better_prob_list, worse_prob_list=intervention_hospital_supply_increase(supply_change_date, new_better_prob_ratio, new_worse_prob_ratio, better_prob_list, worse_prob_list) # The ratio 1.4, 0.2 are arbitrary for now and will be calculated based on breathing machine and oxygen supplies. 
	# print("better_prob_list = ", better_prob_list)
	# print("worse_prob_list = ", worse_prob_list)
	################### standard lines here always ################################
	better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list=plot_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std)
	print("remain_list = ", remain_list)
	print("better_cumu_list = ", better_cumu_list)
	print("worse_cumu_list = ", worse_cumu_list)
	# probability_dict={"better": 0.2, "worse": 0.3}
	# change_state(probability_dict)