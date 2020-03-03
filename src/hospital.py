import numpy as np 
from scipy import stats
import random
import matplotlib.pyplot as plt
import operator
import copy

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

def plot_curves(dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, cure_ability_list, batch_list):
	plt.plot(dates, cure_ability_list, color='green', marker='o', markersize=1, linestyle='-', label='Medical_supply')
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
	# plt.close()
	return

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


def intervention_hospital_supply_change_step(date, new_better_prob_ratio, new_worse_prob_ratio, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_original):
	for i in range(date, len(better_prob_list)):
		better_prob_list[i]=better_prob_list_original[i]*new_better_prob_ratio
	for i in range(date, len(worse_prob_list)):
		worse_prob_list[i]=worse_prob_list_original[i]*new_worse_prob_ratio
	return better_prob_list, worse_prob_list


def cal_cure_ability(breathing_machine_supply_list, oxygen_supply_list, oxygen_change_list, breathing_machine_change_list):

	for oxygen_change_date, oxygen_supply_rate in oxygen_change_list:
		for i in range(oxygen_change_date, len(oxygen_supply_list)):
			oxygen_supply_list[i]=oxygen_supply_rate
	print("oxygen_supply_list", oxygen_supply_list)
	for breathing_machine_change_date, breathing_machine_supply_rate in breathing_machine_change_list:
		for i in range(breathing_machine_change_date, len(breathing_machine_supply_list)):
			breathing_machine_supply_list[i]=breathing_machine_supply_rate
	print("breathing_machine_supply_list", breathing_machine_supply_list)

	change_dates=[x[0] for x in oxygen_change_list]+[x[0] for x in breathing_machine_change_list] 
	change_dates.sort()
	change_dates=np.unique(change_dates)
	print(change_dates)

	cure_ability_list=oxygen_supply_list*breathing_machine_supply_list   
	print(cure_ability_list)

	return cure_ability_list, change_dates

def intervention_hospital_supply_change_repeated(cure_ability_list, change_dates, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original, better_mean, better_std, worse_mean, worse_std):

	better_bef_date_prob=(stats.norm.cdf(change_dates[0], better_mean, better_std))*better_prob
	worse_bef_date_prob=(stats.norm.cdf(change_dates[0], worse_mean, worse_std))*worse_prob
	print("\n")

	for index, t in enumerate(change_dates):
		print("########## change", index, "on day", t, "to cure_ability =", cure_ability_list[t], "#############\n")
		# print()

		new_better_prob=better_prob*cure_ability_list[t]
		better_after_date_prob=(1-stats.norm.cdf(t, better_mean, better_std))*new_better_prob
		print("better_bef_date_prob = ", better_bef_date_prob)
		print("better_after_date_prob = " , better_after_date_prob)
		print("better_total_prob = ",  better_after_date_prob+better_bef_date_prob)

		print("\n")

		worse_after_date_prob=1-better_bef_date_prob-worse_bef_date_prob-better_after_date_prob
		print("worse_bef_date_prob = ", worse_bef_date_prob)
		print("worse_after_date_prob = " , worse_after_date_prob)
		print("worse_total_prob = ", worse_after_date_prob+worse_bef_date_prob)

		print("\n")

		new_better_prob_ratio=cure_ability_list[t]
		print("new_better_prob_ratio = ", new_better_prob_ratio)
		new_worse_prob=worse_after_date_prob/(1-stats.norm.cdf(t, worse_mean, worse_std))
		new_worse_prob_ratio=new_worse_prob/worse_prob
		print("new_worse_prob_ratio = ", new_worse_prob_ratio)
		print("new_better_prob = ", new_better_prob)
		print("new_worse_prob = ", new_worse_prob)
		better_prob_list, worse_prob_list=intervention_hospital_supply_change_step(t, new_better_prob_ratio, new_worse_prob_ratio, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original) # The ratio 1.4, 0.2 are arbitrary for now and will be calculated based on breathing machine and oxygen supplies. 
		# print("better_prob_list = ", better_prob_list)
		# print("worse_prob_list = ", worse_prob_list)
		if index+1<len(change_dates):
			better_bef_date_prob+=(stats.norm.cdf(change_dates[index+1], better_mean, better_std)-stats.norm.cdf(t, better_mean, better_std))*new_better_prob
			worse_bef_date_prob+=(stats.norm.cdf(change_dates[index+1], worse_mean, worse_std)-stats.norm.cdf(t, worse_mean, worse_std))*new_worse_prob
		
		print("\n")

	return better_prob_list, worse_prob_list

def initialise_batch(days, batch_population):
	batch_list=np.ones(days)*batch_population
	print("batch_list = ", batch_list)
	better_prob_list=batch_list*better_prob
	better_prob_list_original=copy.copy(better_prob_list)
	print("better_prob_list = ", better_prob_list)
	worse_prob_list=batch_list*worse_prob
	worse_prob_list_original=copy.copy(worse_prob_list)
	print("worse_prob_list = ", worse_prob_list)
	# cure_ability_list=np.ones(days) # dummy, so that the final plotting function doesn't complain.
	return batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original

def shift_cure_ability_list(cure_ability_list_original, change_dates_original, batch_counter):
	change_dates=[x-batch_counter for x in change_dates_original]
	change_dates = [i for i in change_dates if i>=0] # keep only non-zero change_dates
	head=cure_ability_list_original[batch_counter:]
	tail=[head[-1]] * batch_counter
	cure_ability_list=np.concatenate((head, tail))
	if len(change_dates)==0:
		change_dates=[0]
	return cure_ability_list, change_dates


if __name__ == '__main__':
	#### parameters #########
	days=21
	worse_prob=0.1
	better_prob=1-worse_prob
	better_mean=10
	better_std=5
	worse_mean=7
	worse_std=3
	batch_population=100 # set to 1 if we want the probabilities instead of numbers of people in the batch as outputs.
	number_of_batches = 3
	cure_ability_list=np.ones(days)
	change_dates=[0]
	########### this block changes hospital capability  (can toggle on and off) #################################
	# breathing_machine_supply_list=np.ones(days)
	# oxygen_supply_list=np.ones(days)
	# oxygen_change_list=[(1,0.6), (9, 0.8)] #tuple list, with first element = date the change starts, second element=updated oxygen supply rate. List order doesn't matter
	# breathing_machine_change_list=[(4,0.8)] #tuple list, with first element = date the change starts, second element=updated machine supply rate. List order doesn't matter
	# cure_ability_list, change_dates = cal_cure_ability(breathing_machine_supply_list, oxygen_supply_list, oxygen_change_list, breathing_machine_change_list)

	############### standard lines ###########################
	cure_ability_list_original=copy.copy(cure_ability_list)
	change_dates_original=copy.copy(change_dates)	
	############### setting up figure plots #####################

	
	fig, axs = plt.subplots(nrows=number_of_batches+1, sharex=True)
	plt.xlim(0, days-1)

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
	for ax in axs: 

		if batch_counter==number_of_batches: # last row of subplot
			better_final_list=np.zeros(days)
			worse_final_list=np.zeros(days)
			better_final_list=np.zeros(days)
			worse_final_list=np.zeros(days)
			for j in range(0, days):
				daily_total_better=0
				daily_total_worse=0
				for i in range(0, batch_counter-1):
					daily_total_worse+=worse_dict[i][j-i]
					daily_total_better+=better_dict[i][j-i]
				better_final_list[j]=daily_total_better
				worse_final_list[j]=daily_total_worse
				better_final_cumu_list[j]=sum(better_final_list)
				worse_final_cumu_list[j]=sum(worse_final_list)
			dates=list(range(days))
			ax.plot(dates, better_final_list, color='grey', linestyle='--', label='Better_sum_daily(mean=10,std=5)')
			ax.plot(dates, worse_final_list, color='red', linestyle='--', label='Worse_sum_daily(mean=7,std=3)')
			ax.plot(dates, better_final_cumu_list, color='grey', linestyle='-', label='Better_sum_cumu(mean=10,std=5)')
			ax.plot(dates, worse_final_cumu_list, color='red', linestyle='-', label='Worse_sum_cumu(mean=7,std=3)')
			ax.set_ylabel("Total")
			ax.set_xlabel("Dates")
		else:	# all but the last row of subplot
			batch_population=empty_bed

			batch_list, better_prob_list, better_prob_list_original, worse_prob_list, worse_prob_list_original = initialise_batch(days,batch_population)

			cure_ability_list, change_dates = shift_cure_ability_list(cure_ability_list_original, change_dates_original, batch_counter)

			better_prob_list, worse_prob_list = intervention_hospital_supply_change_repeated(cure_ability_list, change_dates, better_prob_list, worse_prob_list, better_prob_list_original, worse_prob_list_original, better_mean, better_std, worse_mean, worse_std)
			dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, batch_list=batch_curves(days, batch_list, better_prob_list, worse_prob_list, better_mean, better_std, worse_mean, worse_std)
			better_dict[batch_counter]=better_list
			worse_dict[batch_counter]=worse_list
			better_cumu_dict[batch_counter]=better_cumu_list
			worse_cumu_dict[batch_counter]=worse_cumu_list

			daily_total_better=0 # sum up total bed release for the day from all batches of patients
			daily_total_worse=0
			for i in range(0, batch_counter):
				daily_total_worse+=worse_dict[i][batch_counter-i]
				daily_total_better+=better_dict[i][batch_counter-i]
			empty_bed=daily_total_worse+daily_total_better

			better_final_list[batch_counter]=daily_total_better
			worse_final_list[batch_counter]=daily_total_worse
			better_final_cumu_list[batch_counter]=sum(better_final_list)
			worse_final_cumu_list[batch_counter]=sum(better_final_list)

			dates=[x + batch_counter for x in dates] 
			ax.plot(dates, cure_ability_list*batch_population, color='green', marker='o', markersize=3, linestyle='-', label='Medical_supply')
			ax.plot(dates, better_list, color='grey', linestyle='--', label='Better_daily(mean=10,std=5)')
			ax.plot(dates, worse_list, color='red', linestyle='--', label='Worse_daily(mean=7,std=3)')
			ax.plot(dates, better_cumu_list, color='grey', linestyle='-', label='Better_cumu(mean=10,std=5)')
			ax.plot(dates, worse_cumu_list, color='red', linestyle='-', label='Worse_cumu(mean=7,std=3)')
			ax.plot(dates, remain_list, color='yellow', linestyle='-', label='Remain population')
			ax.plot(dates, batch_list, color='blue', linestyle='-',label='Batch population')
			batch_counter+=1
			ax.set_ylabel("Day {}".format(batch_counter))


			# ax.plot(dates, remain_list, color='yellow', linestyle='-', label='Remain population')
			# ax.plot(dates, batch_list, color='blue', linestyle='-',label='Batch population')




	# ax1.plot(dates, cure_ability_list, color='green', marker='o', markersize=1, linestyle='-', label='Medical_supply')
	# ax1.plot(dates, better_list, color='grey', marker='o', markersize=3, linestyle='--', label='Better_daily(mean=10,std=5)')
	# ax1.plot(dates, worse_list, color='red', marker='o', markersize=3, linestyle='--', label='Worse_daily(mean=7,std=3)')
	# ax1.plot(dates, better_cumu_list, color='grey', marker='o', markersize=3, linestyle='-', label='Better_cumu(mean=10,std=5)')
	# ax1.plot(dates, worse_cumu_list, color='red', marker='o', markersize=3, linestyle='-', label='Worse_cumu(mean=7,std=3)')
	# ax1.plot(dates, remain_list, color='yellow', marker='o', markersize=3, linestyle='-', label='Remain population')
	# ax1.plot(dates, batch_list, color='blue', marker='o', markersize=3, linestyle='-',label='Batch population')
	# # ax1.xlabel('Days')
	# ax1.ylabel('probability')
	# ax1.legend(loc='upper right')
	

	# ax2.plot(dates, cure_ability_list, color='green', marker='o', markersize=1, linestyle='-', label='Medical_supply')
	plt.show()
	plt.close()
	# ax2.fill_between(x, y1, 1)
	# ax2.set_ylabel('between y1 and 1')

	# ax3.fill_between(x, y1, y2)
	# ax3.set_ylabel('between y1 and y2')
	# ax3.set_xlabel('x')

	# plot_curves(dates, better_list, worse_list, better_cumu_list, worse_cumu_list, remain_list, cure_ability_list, batch_list)
	print("remain_list = ", remain_list)
	print("better_cumu_list = ", better_cumu_list)
	print("worse_cumu_list = ", worse_cumu_list)
	# probability_dict={"better": 0.2, "worse": 0.3}
	# change_state(probability_dict)