'''
Separation of background photon events is performed.

'''

from .MC_separate import *
from .Event_rate_analysis import *
import numpy as np

class Separate_source(object):
	'''

	'''
	def __init__(self,t,ch,ch_n,
		     time_range = None,
		     WT = True
	             ):

		if time_range is None:
			self.time_start = t[0]
			self.time_stop = t[-1]
			self.t = t
			self.ch = ch
		else:
			self.time_start = time_range[0]
			self.time_stop = time_range[-1]
			self.t = t[np.where((t>=self.time_start)&(t<=self.time_stop))]
			self.ch = ch[np.where((t>=self.time_start)&(t<=self.time_stop))]
		#self.t = t
		#self.ch = ch
		self.WT = WT
		self.ch_n = ch_n
		self.get_background()
		self.check_background()
		
	def separate_background_for_one_ch(self,t,ch,ch_n):
		t = t[np.where(ch == ch_n)]
		if len(t) == 0:
			return np.array([]),np.array([])
		Era = Event_rate_analysis(t,time_range=[self.time_start,self.time_stop])
		if self.WT:
			GPS = Era.get_wt_GPS()
		else:
			GPS = Era.get_GPS()
		BPS = Era.get_BPS()
		#print('GPS \n',GPS)
		#print('BPS \n',BPS)
		mc = MC_separate(t,GPS,BPS)
		return mc.S,mc.B
	def get_background(self):

		s = np.array([])
		b = np.array([])
		s_ch = np.array([])
		b_ch = np.array([])
		for i in self.ch_n:
			print('                                  ',end = '\r')
			print('inite channel ', i,end = '\r')
			S, B = self.separate_background_for_one_ch(self.t, self.ch, i)
			S_ch = np.zeros_like(S) + i
			B_ch = np.zeros_like(B) + i
			s = np.concatenate((s, S))
			s_ch = np.concatenate((s_ch, S_ch))
			b = np.concatenate((b, B))
			b_ch = np.concatenate((b_ch, B_ch))
		s_index = np.argsort(s)
		b_index = np.argsort(b)
		self.s_t = s[s_index]
		self.s_ch = s_ch[s_index]
		self.b_t = b[b_index]
		self.b_ch = b_ch[b_index]
	def check_background(self):
		'''
		Background check. First check the background one by one, then check the general background to prevent the background from crossing.
		:return:
		'''
		#-------------------------------------------------------------------------------------------------------
		b_s = np.array([])                                                          #Check the background part by energy path
		b_b = np.array([])
		b_s_ch = np.array([])
		b_b_ch = np.array([])
		for i in self.ch_n:
			print('                                  ',end = '\r')
			print('check channel ', i,end = '\r')
			b_S, b_B = self.separate_background_for_one_ch(self.b_t, self.b_ch, i)
			S_ch = np.zeros_like(b_S) + i
			B_ch = np.zeros_like(b_B) + i
			b_s = np.concatenate((b_s, b_S))
			b_s_ch = np.concatenate((b_s_ch, S_ch))
			b_b = np.concatenate((b_b, b_B))
			b_b_ch = np.concatenate((b_b_ch, B_ch))
		b_s_index = np.argsort(b_s)
		b_b_index = np.argsort(b_b)
		c_s = b_s[b_s_index]
		c_s_ch = b_s_ch[b_s_index]

		c_b = b_b[b_b_index]
		c_b_ch = b_b_ch[b_b_index]
		self.b_t = c_b
		self.b_ch = c_b_ch
		c_s = np.concatenate((c_s, self.s_t))
		c_s_ch = np.concatenate((c_s_ch, self.s_ch))
		sort_index = np.argsort(c_s)
		self.s_t = c_s[sort_index]
		self.s_ch = c_s_ch[sort_index]
	
	def get_S_t_and_ch(self):
		return self.s_t,self.s_ch
	def get_B_t_and_ch(self):
		return self.b_t,self.b_ch
	



