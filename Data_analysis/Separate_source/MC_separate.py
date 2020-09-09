
import numpy as np

class MC_separate(object):
	'''
	Markov separation, the separation of random events.
	'''
	def __init__(self,t,GPS,BPS,ch = None):
		'''

		:param t:  array or list . Samples that need to be separated
		:param GPS: array or list . Total probability standard
		:param BPS: array or list . Background probability standard
		:param ch: array or list .
		'''
		#data = np.asarray(data,dtype=float)
		#GPS = np.asarray(GPS)
		#BPS = np.asarray(BPS)
		self.size = len(t)
		self.t = np.array(t)
		self.GPS = np.array(GPS)
		self.BPS = np.array(BPS)
		self.ch = ch
		if self.ch is not None:
			self.ch = np.array(self.ch)
			if self.ch.size != self.t.size:
				print('ch and t parameters have different lengths! Do not return ch.')
				self.ch = None
		self.S_index,self.B_index = self.MC_kernel()

		if self.ch is not None:
			self.S = self.t[self.S_index],self.ch[self.S_index]
			self.B = self.t[self.B_index],self.ch[self.B_index]
		else:
			self.S = self.t[self.S_index]
			self.B = self.t[self.B_index]
			
	def MC_kernel(self):
		'''
		
		:return:
		'''
		S_index = []#
		B_index = []#

		MC_rand = np.random.rand(self.size)*self.GPS
		for index,value in enumerate(MC_rand):

			if value > self.BPS[index]:#
				S_index.append(index)
			else:
				B_index.append(index)
		return np.array(S_index,dtype=int),np.array(B_index,dtype=int)

	def get_B(self):
		return self.B
	def get_S(self):
		return self.S












