
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec


class Plot_serch(object):
	
	def __init__(self,result,detector,geometry):
		
		self.result = result
		self.detector = detector
		self.geometry = geometry
		self.clock = geometry.Time_transition

	def get_bayesian_responses_size(self):
		trig_1 = self.result['trig_1'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		return trig_1.shape[0]
	def plot_bayesian_responses(self,index):
		
		lc_c = self.result['lc']
		trig_1 = self.result['trig_1'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		t_i = trig_1.iloc[index]
		overlap = t_i['overlap']
		n = len(overlap)
		plt.figure(constrained_layout=True,figsize=(10,4*n))
		for i in range(n):
			ni = lc_c[overlap[i]]
			lc_list = ni['lc']
			lc_bs_list = ni['lc_bs']
			sigma = ni['sigma']
			plt.subplot(n,1,i+1)
			plt.plot(0,0,color = 'k',label = overlap[i])
			min_ = 100000
			max_ = 0
			for indx,lc in enumerate(lc_list):
				lc_t,lc_rate = lc
				index1 = np.where((lc_t>=t_i['wind_start'])&(lc_t<=t_i['wind_stop']))[0]
				if len(index1)>2:
					lc_bs = lc_bs_list[indx]
					lc_t1 = lc_t[index1]
					lc_rate1 = lc_rate[index1]
					if lc_rate1.min()<min_:
						min_ = lc_rate1.min()
					if lc_rate1.max()>max_:
						max_ = lc_rate1.max()
					lc_bs1 = lc_bs[index1]
					plt.plot(lc_t1-t_i['wind_start'],lc_rate1,color = 'k')
					plt.plot(lc_t1-t_i['wind_start'],lc_bs1,color = '#ef4136')
					plt.plot(lc_t1-t_i['wind_start'],lc_bs1+3*sigma[indx],color = '#f47920')
			plt.axvline(x = t_i['start']-t_i['wind_start'],color = 'r')
			plt.axvline(x = t_i['stop']-t_i['wind_start'],color = 'g')
			plt.xlim(0,t_i['wind_stop']-t_i['wind_start'])
			mean_ = 0.5*(max_+min_)
			high_harf =0.5*(max_-min_)/0.9
			plt.ylim(mean_-high_harf,mean_+high_harf)
			plt.ylabel('Count rate')
			plt.legend(loc = 'upper left')
			if i != n-1:
				plt.xticks([])
		utc_start = self.clock.met_to_utc(t_i['wind_start']).fits
		plt.xlabel('Start at ' + str(utc_start)+' (s)')
	
	def get_threshold_responses_size(self):
		trig_0 = self.result['trig_0'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		return trig_0.shape[0]
	
	def plot_threshold_responses(self,index):
		lc_c = self.result['lc']
		trig_0 = self.result['trig_0'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		t_i = trig_0.iloc[index]
		overlap = t_i['overlap']
		n = len(overlap)
		plt.figure(constrained_layout=True,figsize=(10,4*n))
		for i in range(n):
			ni = lc_c[overlap[i]]
			lc_list = ni['lc']
			lc_bs_list = ni['lc_bs']
			sigma = ni['sigma']
			plt.subplot(n,1,i+1)
			plt.plot(0,0,color = 'k',label = overlap[i])
			min_ = 100000
			max_ = 0
			for indx,lc in enumerate(lc_list):
				lc_t,lc_rate = lc
				index1 = np.where((lc_t>=t_i['wind_start'])&(lc_t<=t_i['wind_stop']))[0]
				if len(index1)>2:
					lc_bs = lc_bs_list[indx]
					lc_t1 = lc_t[index1]
					lc_rate1 = lc_rate[index1]
					if lc_rate1.min()<min_:
						min_ = lc_rate1.min()
					if lc_rate1.max()>max_:
						max_ = lc_rate1.max()
					lc_bs1 = lc_bs[index1]
					plt.plot(lc_t1-t_i['wind_start'],lc_rate1,color = 'k')
					plt.plot(lc_t1-t_i['wind_start'],lc_bs1,color = '#ef4136')
					plt.plot(lc_t1-t_i['wind_start'],lc_bs1+3*sigma[indx],color = '#f47920')
			plt.axvline(x = t_i['start']-t_i['wind_start'],color = '#ea66a6')
			plt.axvline(x = t_i['stop']-t_i['wind_start'],color = '#009ad6')
			plt.xlim(0,t_i['wind_stop']-t_i['wind_start'])
			mean_ = 0.5*(max_+min_)
			high_harf =0.5*(max_-min_)/0.9
			plt.ylim(mean_-high_harf,mean_+high_harf)
			plt.ylabel('Count rate')
			plt.legend(loc = 'upper left')
			if i != n-1:
				plt.xticks([])
		utc_start = self.clock.met_to_utc(t_i['wind_start']).fits
		plt.xlabel('Start at ' + str(utc_start)+' (s)')
	
	def get_all_responses_size(self):
		return self.result['trig_all'].shape[0]
	def plot_all_responses(self,index):
		lc_c = self.result['lc']
		trig_all = self.result['trig_all']
		t_i = trig_all.iloc[index]
		detector= t_i['detector']
		ni = lc_c[detector]
		lc_list = ni['lc']
		lc_bs_list = ni['lc_bs']
		sigma = ni['sigma']
		plt.figure(constrained_layout=True)
		plt.subplot(1,1,1)
		plt.plot(0,0,color = 'k',label = detector)
		min_ = 100000
		max_ = 0
		for indx,lc in enumerate(lc_list):
			lc_t,lc_rate = lc
			index1 = np.where((lc_t>=t_i['wind_start'])&(lc_t<=t_i['wind_stop']))[0]
			if len(index1)>2:
				lc_bs = lc_bs_list[indx]
				lc_t1 = lc_t[index1]
				lc_rate1 = lc_rate[index1]
				if lc_rate1.min()<min_:
					min_ = lc_rate1.min()
				if lc_rate1.max()>max_:
					max_ = lc_rate1.max()
				lc_bs1 = lc_bs[index1]
				plt.plot(lc_t1-t_i['wind_start'],lc_rate1,color = 'k')
				plt.plot(lc_t1-t_i['wind_start'],lc_bs1,color = '#ef4136')
				plt.plot(lc_t1-t_i['wind_start'],lc_bs1+3*sigma[indx],color = '#f47920')
		if t_i['bayes'] == 0:
			plt.axvline(x = t_i['start']-t_i['wind_start'],color = '#ea66a6')
			plt.axvline(x = t_i['stop']-t_i['wind_start'],color = '#009ad6')
		else:
			plt.axvline(x = t_i['start']-t_i['wind_start'],color = 'r')
			plt.axvline(x = t_i['stop']-t_i['wind_start'],color = 'g')
		plt.xlim(0,t_i['wind_stop']-t_i['wind_start'])
		mean_ = 0.5*(max_+min_)
		high_harf = 0.5*(max_-min_)/0.9
		plt.ylim(mean_-high_harf,mean_+high_harf)
		plt.ylabel('Count rate')
		plt.legend(loc = 'upper left')
		utc_start = self.clock.met_to_utc(t_i['wind_start']).fits
		plt.xlabel('Start at ' + str(utc_start)+' (s)')
		
		
	
class Plot_track(object):
	
	def __init__(self,result,detector,geometry,sources=None):
		'''
		
		:param result:
		:param detector: detector name list
		:param geometry:
		'''
		
		self.sources=sources
		self.result = result
		self.detector = detector
		self.geometry = geometry
		self.clock = geometry.Time_transition
		self.t_r_cmap = ListedColormap(['k', '#d3d7d4'])
		self.t_b_cmap = ListedColormap(['#f47920', '#ffce7b'])
		
		
	def get_bayesian_responses_size(self,sn):
		sn_serch_result = self.result[sn]
		sn_trig_1 = sn_serch_result['trig_1'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		return sn_trig_1.shape[0]
		
	def plot_bayesian_responses(self,sn,index,sky_map = False):
		
		tirg_data = self.result['lc']
		sn_serch_result = self.result[sn]
		sn_trig_1 = sn_serch_result['trig_1'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		t_i = sn_trig_1.iloc[index]
		overlap = t_i['overlap']
		n = len(overlap)
		if sky_map:
			fig = plt.figure(constrained_layout=True,figsize=(10,2.5*(n+2)))
			gs = GridSpec(n+2,1)
		else:
			fig = plt.figure(constrained_layout=True,figsize=(10,4*n))
			gs = GridSpec(n,1)
		for i in range(n):
			n0_c = tirg_data[overlap[i]]
			lc_list = n0_c['lc']
			lc_bs_list = n0_c['lc_bs']
			sigma = n0_c['sigma']
			
			axi = fig.add_subplot(gs[i])
			axi.plot(0,0,color = 'k',label = overlap[i])
			#plt.plot(0,0,color = '#f47920',label = 'background')
			min_ = 100000
			max_ = 0
			for indx,lc in enumerate(lc_list):
				lc_t,lc_rate = lc
				index1 = np.where((lc_t>=t_i['wind_start'])&(lc_t<=t_i['wind_stop']))[0]
				if len(index1)>2:
					lc_bs = lc_bs_list[indx]
					lc_t1 = lc_t[index1]
					lc_rate1 = lc_rate[index1]
					if lc_rate1.min()<min_:
						min_ = lc_rate1.min()
					if lc_rate1.max()>max_:
						max_ = lc_rate1.max()
					lc_bs1 = lc_bs[index1]
					axi.plot(lc_t1-t_i['wind_start'],lc_rate1,color = 'k')
					axi.plot(lc_t1-t_i['wind_start'],lc_bs1,color = '#ef4136')
					axi.plot(lc_t1-t_i['wind_start'],lc_bs1+3*sigma[indx],color = '#f47920')
			axi.axvline(x = t_i['start']-t_i['wind_start'],color = 'r')
			axi.axvline(x = t_i['stop']-t_i['wind_start'],color = 'g')
			axi.set_xlim(0,t_i['wind_stop']-t_i['wind_start'])
			mean_ = 0.5*(max_+min_)
			high_harf =0.5*(max_-min_)/0.9
			axi.set_ylim(mean_-high_harf,mean_+high_harf)
			axi.set_ylabel('Count rate')
			axi.legend(loc = 'upper left')
			if i != n-1:
				axi.set_xticks([])
			else:
				utc_start = self.clock.met_to_utc(t_i['wind_start']).fits
				axi.set_xlabel('Start at ' + str(utc_start)+' (s)')
		if sky_map:
			ax_ = fig.add_subplot(gs[-2:],projection=ccrs.Mollweide(central_longitude=180),facecolor = '#f6f5ec')
			
			if self.sources is not None:
				source_name = self.sources.names
				index = np.where(np.array(source_name) == sn)
				pos = self.sources.positions[index]
			else:
				pos = None
			self.geometry.detector_plot(ax=ax_, time=t_i['start'], show_bodies=True, size=0.5,source = pos,highlight=overlap)
			
			
			
	def get_threshold_responses_size(self,sn):
		sn_serch_result = self.result[sn]
		sn_trig_0 = sn_serch_result['trig_0'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		return sn_trig_0.shape[0]
		
	def plot_threshold_responses(self,sn,index,sky_map = False):
		
		tirg_data = self.result['lc']
		sn_serch_result = self.result[sn]
		sn_trig_0 = sn_serch_result['trig_0'].drop_duplicates('start','first',inplace=False,ignore_index=True)
		t_i = sn_trig_0.iloc[index]
		overlap = t_i['overlap']
		n = len(overlap)
		if sky_map:
			fig = plt.figure(constrained_layout=True,figsize=(10,2.5*(n+2)))
			gs = GridSpec(n+2,1)
		else:
			fig = plt.figure(constrained_layout=True,figsize=(10,4*n))
			gs = GridSpec(n,1)
		#plt.figure(constrained_layout=True,figsize=(10,4*n))
		for i in range(n):
			n0_c = tirg_data[overlap[i]]
			lc_list = n0_c['lc']
			lc_bs_list = n0_c['lc_bs']
			sigma = n0_c['sigma']
			axi = fig.add_subplot(gs[i])
			#plt.subplot(n,1,i+1)
			axi.plot(0,0,color = 'k',label = overlap[i])
			#plt.plot(0,0,color = '#f47920',label = 'background')
			min_ = 100000
			max_ = 0
			for indx,lc in enumerate(lc_list):
				lc_t,lc_rate = lc
				index1 = np.where((lc_t>=t_i['wind_start'])&(lc_t<=t_i['wind_stop']))[0]
				if len(index1)>2:
					lc_bs = lc_bs_list[indx]
					lc_t1 = lc_t[index1]
					lc_rate1 = lc_rate[index1]
					if lc_rate1.min()<min_:
						min_ = lc_rate1.min()
					if lc_rate1.max()>max_:
						max_ = lc_rate1.max()
					lc_bs1 = lc_bs[index1]
					axi.plot(lc_t1-t_i['wind_start'],lc_rate1,color = 'k')
					axi.plot(lc_t1-t_i['wind_start'],lc_bs1,color = '#ef4136')
					axi.plot(lc_t1-t_i['wind_start'],lc_bs1+3*sigma[indx],color = '#f47920')
			axi.axvline(x = t_i['start']-t_i['wind_start'],color = '#ea66a6')
			axi.axvline(x = t_i['stop']-t_i['wind_start'],color = '#009ad6')
			axi.set_xlim(0,t_i['wind_stop']-t_i['wind_start'])
			mean_ = 0.5*(max_+min_)
			high_harf =0.5*(max_-min_)/0.9
			axi.set_ylim(mean_-high_harf,mean_+high_harf)
			axi.set_ylabel('Count rate')
			axi.legend(loc = 'upper left')
			if i != n-1:
				axi.set_xticks([])
			else:
				utc_start = self.clock.met_to_utc(t_i['wind_start']).fits
				plt.xlabel('Start at ' + str(utc_start)+' (s)')
		if sky_map:
			ax_ = fig.add_subplot(gs[-2:],projection=ccrs.Mollweide(central_longitude=180),facecolor = '#f6f5ec')
			if self.sources is not None:
				source_name = self.sources.names
				index = np.where(np.array(source_name) == sn)
				pos = self.sources.positions[index]
			else:
				pos = None
			self.geometry.detector_plot(ax=ax_, time=t_i['start'], show_bodies=True, size=0.5, source=pos,highlight=overlap)
		
		
	def get_all_responses_size(self,sn):
		sn_serch_result = self.result[sn]
		sn_trig_0 = sn_serch_result['trig_all']
		return sn_trig_0.shape[0]
	
	def plot_all_responses(self,sn,index):
		
		tirg_data = self.result['lc']
		sn_serch_result = self.result[sn]
		sn_trig_0 = sn_serch_result['trig_all']
		t_i = sn_trig_0.iloc[index]
		detector= t_i['detector']
		n0_c = tirg_data[detector]
		lc_list = n0_c['lc']
		lc_bs_list = n0_c['lc_bs']
		sigma = n0_c['sigma']
		plt.figure(constrained_layout=True)
		plt.subplot(1,1,1)
		plt.plot(0,0,color = 'k',label = detector)
		#plt.plot(0,0,color = '#f47920',label = 'background')
		min_ = 100000
		max_ = 0
		for indx,lc in enumerate(lc_list):
			lc_t,lc_rate = lc
			index1 = np.where((lc_t>=t_i['wind_start'])&(lc_t<=t_i['wind_stop']))[0]
			if len(index1)>2:
				lc_bs = lc_bs_list[indx]
				lc_t1 = lc_t[index1]
				lc_rate1 = lc_rate[index1]
				if lc_rate1.min()<min_:
					min_ = lc_rate1.min()
				if lc_rate1.max()>max_:
					max_ = lc_rate1.max()
				lc_bs1 = lc_bs[index1]
				plt.plot(lc_t1-t_i['wind_start'],lc_rate1,color = 'k')
				plt.plot(lc_t1-t_i['wind_start'],lc_bs1,color = '#ef4136')
				plt.plot(lc_t1-t_i['wind_start'],lc_bs1+3*sigma[indx],color = '#f47920')
		if t_i['bayes'] == 0:
			plt.axvline(x = t_i['start']-t_i['wind_start'],color = '#ea66a6')
			plt.axvline(x = t_i['stop']-t_i['wind_start'],color = '#009ad6')
		else:
			plt.axvline(x = t_i['start']-t_i['wind_start'],color = 'r')
			plt.axvline(x = t_i['stop']-t_i['wind_start'],color = 'g')
		plt.xlim(0,t_i['wind_stop']-t_i['wind_start'])
		mean_ = 0.5*(max_+min_)
		high_harf = 0.5*(max_-min_)/0.9
		plt.ylim(mean_-high_harf,mean_+high_harf)
		plt.ylabel('Count rate')
		plt.legend(loc = 'upper left')
		utc_start = self.clock.met_to_utc(t_i['wind_start']).fits
		plt.xlabel('Start at ' + str(utc_start)+' (s)')
	
		
	def plot_one_source(self,sn,positions,range_,axs):
		'''
		
		:param sn:
		:param positions:
		:param axs:
		:return:
		'''
		
		t_r_norm = BoundaryNorm([0, self.geometry.radius+range_,181+range_], self.t_r_cmap.N)
		t_b_norm = BoundaryNorm([0, self.geometry.radius+range_,181+range_], self.t_b_cmap.N)
		
		tirg_data = self.result['lc']
		sn_serch_result = self.result[sn]
		sn_trig_0 = sn_serch_result['trig_0']
		sn_trig_1 = sn_serch_result['trig_1']
		sn_trig_all = sn_serch_result['trig_all']
		for ind,deteri in enumerate(self.detector):
			n0_c = tirg_data[deteri]
			det_trig_0 = sn_trig_0[sn_trig_0['detector'] == deteri]
			det_trig_1 = sn_trig_1[sn_trig_1['detector'] == deteri]
			det_trig_all = sn_trig_all[sn_trig_all['detector'] == deteri]
			lc_list = n0_c['lc']
			lc_bs_list = n0_c['lc_bs']
			t_min = lc_list[0][0][0]
			utc_start = self.clock.met_to_utc(t_min).fits
			t_max = lc_list[-1][0][-1]
			start = det_trig_all['start'].values
			stop  = det_trig_all['stop'].values
			bayes = det_trig_all['bayes'].values
			for i in range(len(start)):
				if bayes[i] == 0:
					axs[ind * 2].axvline(x=start[i] -t_min, color='#4f5555', alpha=0.2)
					axs[ind * 2].axvline(x=stop[i] - t_min, color='#4f5555', alpha=0.2)
				else:
					axs[ind * 2].axvline(x=start[i] - t_min, color='#c37e00', alpha=0.2)
					axs[ind * 2].axvline(x=stop[i] - t_min, color='#c37e00', alpha=0.2)
			
			start = det_trig_0['start'].values
			stop = det_trig_0['stop'].values
			for i in range(len(start)):
				axs[ind * 2].axvline(x=start[i] - t_min, color='#ea66a6')
				axs[ind * 2].axvline(x=stop[i] - t_min, color='#009ad6')
			
			start = det_trig_1['start'].values
			stop = det_trig_1['stop'].values
			for i in range(len(start)):
				axs[ind * 2].axvline(x=start[i] - t_min, color='#aa2116')
				axs[ind * 2].axvline(x=stop[i] - t_min, color='#225a1f')

			
			axs[ind*2].plot(0,0,color = 'k',label = deteri+' lightcurve')
			axs[ind*2].plot(0,0,color = '#f47920',label = 'background')
			lc_ratemax = 0
			for indx,lc in enumerate(lc_list):
				lc_t,lc_rate = lc
				lcr_max = lc_rate.max()
				if lcr_max>lc_ratemax:
					lc_ratemax = lcr_max
					
				lc_bs = lc_bs_list[indx]
				
				lc_tc = 0.5*(lc_t[:-1]+lc_t[1:])
				
				t_r_points = np.array([lc_t-t_min, lc_rate]).T.reshape(-1, 1, 2)
				t_r_segments = np.concatenate([t_r_points[:-1], t_r_points[1:]], axis=1)
				
				t_b_points = np.array([lc_t-t_min, lc_bs]).T.reshape(-1, 1, 2)
				
				t_b_segments = np.concatenate([t_b_points[:-1], t_b_points[1:]], axis=1)
				
				seq = self.geometry.get_separation_with_time(lc_tc,positions)
				seq = seq[deteri].values
				
				t_r = LineCollection(t_r_segments, cmap=self.t_r_cmap, norm=t_r_norm)
				t_r.set_array(seq)
				t_r.set_linewidth(2)
				#lc.set_linewidth(2)
				axs[ind*2].add_collection(t_r)

				t_b = LineCollection(t_b_segments, cmap=self.t_b_cmap, norm=t_b_norm)
				t_b.set_array(seq)
				t_b.set_linewidth(2)
				axs[ind*2].add_collection(t_b)
				axs[ind*2+1].plot(lc_tc-t_min,seq,color = 'k')
			axs[ind*2].set_xticks([])
			axs[ind*2].set_xlim(0,t_max-t_min)
			axs[ind*2].set_ylim(0,lc_ratemax/0.8)
			axs[ind*2].set_ylabel('Count rate')
			axs[ind*2].legend(loc = 'upper left')
			axs[ind*2+1].set_xlim(0,t_max-t_min)
			axs[ind*2+1].set_ylim(0,180)
			axs[ind*2+1].set_ylabel('Separation degree')
			axs[ind*2+1].set_xlabel('Start at ' + str(utc_start)+' (s)')
			#axs[ind*2+1].legend()
	
	def plot_one_detector(self,sn,detector,positions,range_,axs):
		'''
		
		:param sn:
		:param detector:
		:param positions:
		:param axs:
		:return:
		'''
		t_r_norm = BoundaryNorm([0, self.geometry.radius+range_,181+range_], self.t_r_cmap.N)
		t_b_norm = BoundaryNorm([0, self.geometry.radius+range_,181+range_], self.t_b_cmap.N)
		tirg_data = self.result['lc']
		sn_serch_result = self.result[sn]
		sn_trig_0 = sn_serch_result['trig_0']
		sn_trig_1 = sn_serch_result['trig_1']
		sn_trig_all = sn_serch_result['trig_all']
		n0_c = tirg_data[detector]
		lc_list = n0_c['lc']
		lc_bs_list = n0_c['lc_bs']
		t_min = lc_list[0][0][0]
		utc_start = self.clock.met_to_utc(t_min).fits
		t_max = lc_list[-1][0][-1]
		
		det_trig_0 = sn_trig_0[sn_trig_0['detector'] == detector]
		det_trig_1 = sn_trig_1[sn_trig_1['detector'] == detector]
		det_trig_all = sn_trig_all[sn_trig_all['detector'] == detector]
		
		start = det_trig_all['start'].values
		stop  = det_trig_all['stop'].values
		bayes = det_trig_all['bayes'].values
		
		for i in range(len(start)):
			if bayes[i] == 0:
				axs[0].axvline(x=start[i] - t_min, color='#4f5555', alpha=0.2)
				axs[0].axvline(x=stop[i] - t_min, color='#4f5555', alpha=0.2)
			else:
				axs[0].axvline(x=start[i] - t_min, color='#c37e00', alpha=0.2)
				axs[0].axvline(x=stop[i] - t_min, color='#c37e00', alpha=0.2)
			
		start = det_trig_0['start'].values
		stop = det_trig_0['stop'].values
		for i in range(len(start)):
			axs[0].axvline(x=start[i] - t_min, color='#ea66a6')
			axs[0].axvline(x=stop[i] - t_min, color='#009ad6')
		
		start = det_trig_1['start'].values
		stop = det_trig_1['stop'].values
		for i in range(len(start)):
			axs[0].axvline(x=start[i] - t_min, color='#aa2116')
			axs[0].axvline(x=stop[i] - t_min, color='#225a1f')

		
		axs[0].plot(0,0,color = 'k',label = detector+' lightcurve')
		axs[0].plot(0,0,color = '#f47920',label = 'background')
		lc_ratemax = 0
		for indx,lc in enumerate(lc_list):
			lc_t,lc_rate = lc
			lcr_max = lc_rate.max()
			if lcr_max>lc_ratemax:
				lc_ratemax = lcr_max
			lc_bs = lc_bs_list[indx]
			lc_tc = 0.5 * (lc_t[:-1] + lc_t[1:])
			
			t_r_points = np.array([lc_t - t_min, lc_rate]).T.reshape(-1, 1, 2)
			t_r_segments = np.concatenate([t_r_points[:-1], t_r_points[1:]], axis=1)
			
			t_b_points = np.array([lc_t - t_min, lc_bs]).T.reshape(-1, 1, 2)
			t_b_segments = np.concatenate([t_b_points[:-1], t_b_points[1:]], axis=1)
			
			seq = self.geometry.get_separation_with_time(lc_tc, positions)
			seq = seq[detector].values
			
			t_r = LineCollection(t_r_segments, cmap=self.t_r_cmap, norm=t_r_norm)
			t_r.set_array(seq)
			t_r.set_linewidth(2)
			axs[0].add_collection(t_r)
			
			t_b = LineCollection(t_b_segments, cmap=self.t_b_cmap, norm=t_b_norm)
			t_b.set_array(seq)
			t_b.set_linewidth(2)
			axs[0].add_collection(t_b)
			
			axs[1].plot(lc_tc - t_min, seq, color='k')
		axs[0].set_xticks([])
		axs[0].set_xlim(0,t_max-t_min)
		axs[0].set_ylim(0,lc_ratemax/0.8)
		axs[0].set_ylabel('Count rate')
		axs[0].legend(loc = 'upper left')
		axs[1].set_xlim(0,t_max-t_min)
		axs[1].set_ylim(0,180)
		axs[1].set_ylabel('Separation degree')
		axs[1].set_xlabel('Start at ' + str(utc_start)+' (s)')



