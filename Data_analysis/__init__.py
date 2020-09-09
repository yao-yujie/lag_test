'''
the tool of data analysis
'''

from .Bayesian_duration import *
from .Baseline import *
from .Separate_source import *
from .Time_transform import *
from .overlap_tools import *


def get_energy_of_ch(time,e1,e2):
	'''
	
	:param time: time
	:param e1:
	:param e2:
	:return:
	'''
	numb = len(time)
	energy_random_arr = np.random.random_sample(numb)
	energy_array = e1 + (e2-e1)*energy_random_arr
	return energy_array

def ch_to_energy(time,ch,ch_n,e1,e2):
	'''
	
	:param time:
	:param ch:
	:param ch_n:
	:param e1:
	:param e2:
	:return:
	'''
	new_t = np.array([])
	new_energy = np.array([])
	for index,channel in enumerate(ch_n):
		ch_t_index = np.where(ch == channel)
		ch_t = time[ch_t_index]
		energy_array = get_energy_of_ch(ch_t,e1[index],e2[index])
		new_t = np.concatenate((new_t,ch_t))
		new_energy = np.concatenate((new_energy,energy_array))
	index_all = np.argsort(new_t)
	new_t = new_t[index_all]
	new_energy = new_energy[index_all]
	return new_t,new_energy



