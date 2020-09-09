import os
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from Data_analysis import Clock
import Data_analysis.file as myfile
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def time_slic(time_start,time_stop,format = None,scale='utc',H = 3):
	
	if isinstance(time_start,Time) ==False:
		time_start = Time(time_start, format=format, scale=scale)
	if isinstance(time_stop,Time) ==False:
		time_stop =Time(time_stop,format=format,scale=scale)
	re = pd.date_range(time_start.fits,time_stop.fits,freq = str(H)+'H')
	re = pd.DataFrame({'time':re})
	stp = pd.date_range(time_stop.fits,freq = 'H',periods=1)
	stp = pd.DataFrame({'time':stp})
	
	t_ar = pd.concat([re,stp])
	t_ar.drop_duplicates('time','first',inplace=True,ignore_index=True)
	t_ar.sort_values(by = 'time',inplace=True,ignore_index=True)
	time_list = t_ar['time'].values
	t_ar1 = pd.DataFrame({'time_start':time_list[:-1],'time_stop':time_list[1:]})
	return t_ar1.values
	
	
class Database(object):
	
	def __init__(self,databasepath,detectors = None,clock = None):
		'''
		
		:param databasepath:
		:param clock:
		'''
		
		
		self.topdir = databasepath
		if detectors is not None:
			self.detector = detectors
		else:
			self.detector = ['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']
		if clock is None:
			self.clock = Clock()
		else:
			self.clock = clock
		
		
	def get_detector_data(self,time_start,time_stop,format = None,scale='utc'):
		'''
		
		:return: data {'n0':{'ch_E':pandas_table,            # columns CHANNEL,E_MIN,E_MAX
				'events':pandas_table},              #columns TIME, PHA
				...}
		'''
		if isinstance(time_start,Time) ==False:
			time_start = Time(time_start, format=format, scale=scale)
		if isinstance(time_stop,Time) ==False:
			time_stop =Time(time_stop,format=format,scale=scale)
		met_start = self.clock.utc_to_met(time_start,astropyTime = True)
		met_stop = self.clock.utc_to_met(time_stop,astropyTime = True)
		#mjd_in_h = np.arange(time_start.mjd,time_stop.mjd,1/24)
		#time_h_array = Time(mjd_in_h,format='mjd',scale='utc')
		date_time_arr = pd.date_range(time_start.fits,time_stop.fits,freq = 'H')
		print(date_time_arr)
		#date_time_arr = time_h_array.to_datetime()
		data = {}
		for deter in self.detector:#['n0','n1','n2','n3','n4','n5','n6','n7','n7','n8','n9','na','nb','b0','b1']
			data[deter] = {}
		
		for deter in self.detector:
			#print('                                                                   ',end = '\r')
			#print('get',deter,'data.',end='\r')
			#year,month,day
			data[deter]['ch_E'] = None
			data[deter]['events'] = None
			#time = np.array([])
			#pha = np.array([])
			
			for date_i in range(date_time_arr.shape[0]):
				date_t = date_time_arr[date_i]
				year = '%d' % date_t.year
				month = '%.2d' % date_t.month
				day = '%.2d' % date_t.day
				hour = '%.2d' % date_t.hour
				link = self.topdir + year + '/' + month + '/' + day + '/'
				name = 'glg_tte_'+deter+'_'+year[-2:]+month+day + '_'+hour+'z_v*'
				file = myfile.findfile(link,name)
				
				if len(file) > 0:
					
					file_name = file[0]
					#hl = fits.open(link + file_name)
					if data[deter]['ch_E'] is None:
						#data[deter]['ch_E'] = hl[1].data
						hl = Table.read(link + file_name, hdu=1)
						data[deter]['ch_E'] = hl.to_pandas()
					#time = np.concatenate((time,hl[2].data.field(0)))
					#pha = np.concatenate((pha,hl[2].data.field(1)))
					
					if data[deter]['events'] is None:
						#time = hl[2].data.field(0)
						#pha = hl[2].data.field(1)
						#data[deter]['events'] = [time,pha]
						hl = Table.read(link + file_name, hdu=2).to_pandas()
						#print(hl.shape)
						data[deter]['events'] = hl
					else:
						
						#time =np.concatenate((data[deter]['events'][0],hl[2].data.field(0)))
						#pha = np.concatenate((data[deter]['events'][1],hl[2].data.field(1)))
						#data[deter]['events'] = [time,pha]
						
						hl = Table.read(link + file_name, hdu=2).to_pandas()
						#print(hl.shape)
						data[deter]['events']=pd.concat([data[deter]['events'],hl])
						#data[deter]['events'].append(hl)
						
						#news = pd.concat([data[deter]['events'],hl])
						#news.drop_duplicates('TIME','first',inplace=True,ignore_index=True)
						#news = pd.merge(data[deter]['events'],hl,on = list(columns_)[0],how = 'outer')
						#data[deter]['events'] = news.sort_values(by = 'TIME')
					
				else:
					print('lost file:',name)
			#fl = pd.DataFrame({'TIME':time,'PHA':pha})
			#fl.drop_duplicates('TIME','first',inplace=True,ignore_index=True)
			#fl.sort_values(by = 'TIME',inplace=True,ignore_index=True)
			#index_ = (fl['TIME']>=met_start)&(fl['TIME']<=met_stop)
			#data[deter]['events'] = fl[index_]
			#print('all',data[deter]['events'].shape)
			if date_time_arr.shape[0]>1:
				data[deter]['events'].drop_duplicates(['TIME'],keep = 'first',inplace=True,ignore_index=True)
				data[deter]['events'].sort_values(by = 'TIME',inplace=True,ignore_index=True)
				#print('drop',data[deter]['events'].shape)
			index_ = (data[deter]['events']['TIME']>=met_start)&(data[deter]['events']['TIME']<=met_stop)
			data[deter]['events'] = data[deter]['events'][index_]
	
		return data
				
	def get_poshist_data(self,time_start,time_stop,format = None,scale='utc'):
		'''
		
		:return:
		'''
		if isinstance(time_start,Time) == False:
			time_start = Time(time_start, format=format, scale=scale)
		
		if isinstance(time_stop,Time) == False:
			time_stop =Time(time_stop,format=format,scale=scale)
			
		met_start = self.clock.utc_to_met(time_start,astropyTime = True)
		met_stop = self.clock.utc_to_met(time_stop,astropyTime = True)
		date_time_arr = pd.date_range(time_start.fits,time_stop.fits,freq = 'D')
		timelist = []
		data = None
		for t_i in range(date_time_arr.shape[0]):
			year = '%d' % date_time_arr[0].year
			month = '%.2d' % date_time_arr[0].month
			day = '%.2d' % date_time_arr[0].day
			link = self.topdir + year + '/' + month + '/' + day + '/'
			date = year+'-'+month+'-'+day
			timelist.append(date)
			name = 'glg_poshist_all_'+year[-2:]+month+day + '_v*'
			file = myfile.findfile(link,name)
			if len(file) > 0:
				file_name = file[0]
				if data is None:
					hl = Table.read(link + file_name,hdu = 1).to_pandas()
					data = hl
				else:
					hl = Table.read(link + file_name,hdu = 1).to_pandas()
					data = pd.concat([data,hl])
					#data.append(hl)
					#columns_ = hl.columns.values
					#new = pd.concat([data,hl])
					#new.drop_duplicates('SCLK_UTC','first',inplace=True,ignore_index=True)
					#new = pd.merge(data,hl,on=columns_[:-1],how = 'outer')
					#new.sort_values(by = 'SCLK_UTC')
					#data = new
			else:
				print('The poshist file is missing.')
				print(name)
				return None
		if data.shape[0]>1:
			data.drop_duplicates(['SCLK_UTC'],keep='first',inplace=True,ignore_index=True)
			#_,index_ = np.unique(data['SCLK_UTC'].values,return_index=True)
			#data = data[index_]
			data.sort_values(by = 'SCLK_UTC',inplace=True,ignore_index=True)
		index_ = (data['SCLK_UTC']>=met_start-60)&(data['SCLK_UTC']<=met_stop+60)
		print('poshish data shape :',data[index_].shape)
		return data[index_]
	


