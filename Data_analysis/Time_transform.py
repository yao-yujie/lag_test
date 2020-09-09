'''
utc and instrument time conversion

'''

from astropy.time import Time
import numpy as np

class Clock(object):
	def __init__(self,time_origin = None):
		'''
		
		:param time_origin:
		'''
		leap_second_mjd = np.array([41499.0,41499.0,41864.0,42413.0,42778.0,43144.0,43509.0,
		                            43874.0,44239.0,44786.0,45151.0,45516.0,46247.0,47161.0,
		                            47892.0,48257.0,48804.0,49169.0,49534.0,50083.0,50630.0,
		                            51179.0,53736.0,54832.0,56109.0,57204.0,57754.0])
		
		if time_origin is None:
			self.time_origin = 51910 #+ 0.0007428703703
		elif isinstance(time_origin, str):
			self.time_origin = Time(time_origin).mjd
		else:
			self.time_origin = Time(time_origin,format = 'mjd',scale = 'utc').mjd
		leap_second_mjd = leap_second_mjd[leap_second_mjd>self.time_origin]
		add_second = np.arange(len(leap_second_mjd)) + 1
		index_r = np.argsort(-add_second)
		self.leap_second_mjd = leap_second_mjd[index_r]
		self.add_second = add_second[index_r]
		self.leap_second_met = (self.leap_second_mjd-self.time_origin)*86400+self.add_second
		
	def utc_to_met(self,utc,astropyTime = False):
		'''
		
		:param utc:
		:return:
		'''
		try:
			if isinstance(utc, str):
				mjd_ = Time(utc).mjd
				add_s = 0
				for index,value in enumerate(self.leap_second_mjd):
					if mjd_ >= value:
						add_s = self.add_second[index]
						break
				met = (mjd_ - self.time_origin)*86400 + add_s
				return met
			
			add_array = np.zeros(len(utc))
			if astropyTime:
				mjd_array = utc.mjd
			else:
				if isinstance(utc[0], str):
					mjd_array = Time(utc).mjd
				else:
					mjd_array = Time(utc,format = 'mjd',scale = 'utc').mjd
			for index1,mjd_ in enumerate(mjd_array):
				for index2,value2 in enumerate(self.leap_second_mjd):
					if mjd_ >= value2:
						add_array[index1] = self.add_second[index2]
						break
			met = (mjd_array - self.time_origin)*86400 + add_array
			return met
		except:
			if astropyTime:
				mjd_ = utc.mjd
			else:
				if isinstance(utc, str):
					mjd_ = Time(utc).mjd
				else:
					mjd_ = Time(utc,format = 'mjd',scale = 'utc').mjd
			add_s = 0
			for index,value in enumerate(self.leap_second_mjd):
				if mjd_ >= value:
					add_s = self.add_second[index]
					break
			met = (mjd_ - self.time_origin)*86400 + add_s
			return met
	
	def met_to_utc(self,met):
		'''
		
		:param met:
		:return:
		'''
		try:
			add_array = np.zeros(len(met))
			met_array = np.array(met)
			for index1,met in enumerate(met_array):
				for index2,value in enumerate(self.leap_second_met):
					if met >= value:
						add_array[index1] = self.add_second[index2]
						break
			mjd_array = (met_array-add_array)/86400 + self.time_origin
			return Time(mjd_array,format='mjd',scale = 'utc')
		except:
			add_s = 0
			for index,value in enumerate(self.leap_second_met):
				if met >= value:
					add_s = self.add_second[index]
					break
			mjd = (met-add_s)/86400 + self.time_origin
			return Time(mjd,format='mjd',scale = 'utc')

	def batch_utc_to_met(self,utc_list,astropyTime = False):
		'''
		
		:param utc_list:
		:return:
		'''
		add_array = np.zeros(len(utc_list))
		if astropyTime:
			mjd_array = utc_list.mjd
		else:
			if isinstance(utc_list[0], str):
				mjd_array = Time(utc_list).mjd
			else:
				mjd_array = Time(utc_list,format = 'mjd',scale = 'utc').mjd
		for index1,mjd_ in enumerate(mjd_array):
			for index2,value2 in enumerate(self.leap_second_mjd):
				if mjd_ >= value2:
					add_array[index1] = self.add_second[index2]
					break
		met = (mjd_array - self.time_origin)*86400 + add_array
		return met
	
	def batch_met_to_utc(self,met_array):
		'''
		
		:param met_array:
		:return:
		'''
		met_array = np.array(met_array)
		add_array = np.zeros(met_array.size)
		for index1,met in enumerate(met_array):
			for index2,value in enumerate(self.leap_second_met):
				if met >= value:
					add_array[index1] = self.add_second[index2]
					break
		mjd_array = (met_array-add_array)/86400 + self.time_origin
		return Time(mjd_array,format='mjd',scale = 'utc')
Time_transform = Clock