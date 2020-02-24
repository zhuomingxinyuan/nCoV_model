import numpy as np 
import scipy as scp 
import random


def change_state(probability_dict:dict): # probability_dict={'better': 0.2, 'worse': 0.3}
	dice=random()
	print("roll dice = ", dice)
	prob_better=probability_dict["better"]""]
	if dice < prob_better:
		state="better"
	elif dice < prob_worse:
		state="worse"
	else:
		state="remain"
	print("result state is ", state)
	return state



if __name__ == '__main__':
	probability_dict={"better": 0.2, "worse": 0.3}
	change_state(probability_dict)