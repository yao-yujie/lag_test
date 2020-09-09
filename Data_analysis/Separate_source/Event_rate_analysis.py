'''

'''
import numpy as np
from .WT_kernel import WT_kernel
from ..Baseline import TD_baseline

class Event_rate_analysis(object):
	
	def __init__(self,t,time_range = None):
		
		if time_range is None:
			self.time_start = t[0]
			self.time_stop = t[-1]
			self.t = t
		else:
			self.time_start = time_range[0]
			self.time_stop = time_range[-1]
			self.t = t[np.where((t>=self.time_start)&(t<=self.time_stop))]
	def get_BPS(self):
		edges = np.arange(self.time_start,self.time_stop+1,1)
		bin_n,bin_edges = np.histogram(self.t,bins = edges)
		bin_n = np.concatenate((bin_n[:1],bin_n[:-1],bin_n[-2:-1]))#Remove the last incomplete block
		bs = TD_baseline(bin_edges,bin_n)[2]
		BPS = np.interp(self.t,bin_edges,bs)
		#rand = np.random.normal(BPS.size)*np.sqrt(BPS)
		return BPS #+ rand
	
	def get_GPS(self,binsize = 1):
		'''
		Isochronous slice, GPS analysis, the result of such analysis has the problem of slice accuracy.
		:param binsize:
		:return:
		'''
		edges = np.arange(self.time_start,self.time_stop+binsize,binsize)
		bin_n,bin_edges = np.histogram(self.t,bins = edges)
		bin_size = bin_edges[1:]-bin_edges[:-1]
		bin_rate = bin_n/bin_size
		bin_c = (bin_edges[1:]+bin_edges[:-1])*0.5

		bin_c = np.concatenate(([self.time_start],bin_c,[self.time_stop]))
		bin_rate = np.concatenate(([bin_rate[0]],bin_rate,[bin_rate[-1]]))

		return np.interp(self.t,bin_c,bin_rate)
	def get_wt_GPS(self,dt = 1.024):
		'''
		Using WT to analyze GPS will require more computation. There is no problem of slice accuracy in the analysis results.
		:return:

		'''
		return WT_kernel(self.t,dt = dt).rate
