'''

'''
import numpy as np
from ..Baseline import WhittakerSmooth

class WT_kernel(object):

	def __init__(self,t,dt = 1.024):

		self.dt = dt
		self.t = np.array(t)
		self.wt = self.work_WT()
		rate = 1.0/self.wt
		w = np.ones(rate.size)
		self.rate = WhittakerSmooth(rate,w,2)
		

	def get_wt(self):
		return self.wt

	def get_rate(self):
		return self.rate

	def weight(self,x,x0,sigma):
		rr = np.exp(-(x-x0)**2*sigma)
		#if self.fast:
		#	rr = 1-sigma*(x-x0)**2
		#	rr[rr<1e-22] = 0
		return rr
		#return np.exp(-(x-x0)**2*sigma)

	def work_WT(self):
		#这里的卷积有些慢。
		dt = self.t[1:]-self.t[:-1]
		wt = np.zeros(len(self.t))
		#sigma_ = (1/self.dt)**2
		sigma = 0.5/(self.dt)**2
		for index,value in enumerate(self.t):
			ww = self.weight(self.t,value,sigma)  #
			dti = np.insert(dt,index,0)
			ww_sum = ww.sum()-1
			wt[index] = (dti*ww).sum()/ww_sum
			#wti = (dti*ww).sum()/ww_sum
			#wt.append(wti)
		#wt = np.array(wt)
		return wt