import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
# from multiprocessing import Pool
import Data_analysis.file as myfile
from Data_analysis.geometry import Geometry, Detectors
from Data_analysis import Time_transform, Separate_source, ch_to_energy, TD_baseline
from Data_analysis import get_bayesian_flash, get_bayesian_duration, background_correction, get_bayesian_txx, \
	re_histogram
from Data_analysis import Plot, save_result,overlap_curve
import astropy.units as u
# from astropy.coordinates import SkyCoord
from astropy.stats import bayesian_blocks
import re


def get_sample_dir_list(yearlist, databaselink):
	sample_dir_list = []
	for year in yearlist:
		topdir = databaselink + str(year) + '/'
		dirlist1 = os.listdir(topdir)
		dirlist1 = np.sort(dirlist1)
		for dirl in dirlist1:
			if os.path.isdir(topdir + dirl):
				sample_dir_list.append([topdir + dirl + '/', dirl])
	return sample_dir_list


def light_curve_analysis(file, NaI, BGO, good_ni, good_bi, txtdir, plotsave, plotsave1, txx=False, WT=False):
	dt = 0.064
	maxx = 0
	new_c = {}
	if os.path.exists(txtdir) == False:
		os.makedirs(txtdir)
	myfile.printdatatofile(txtdir + 'N_good_ni.txt', data=[good_ni])
	myfile.printdatatofile(txtdir + 'N_good_bi.txt', data=[good_bi])
	for ni in NaI:
		if file[ni] is not None:
			hl = file[ni]
			trigtime = hl[0].header['TRIGTIME']
			time = hl[2].data.field(0)
			ch = hl[2].data.field(1)
			ch_n = hl[1].data.field(0)
			e1 = hl[1].data.field(1)
			e2 = hl[1].data.field(2)
			t = time - trigtime
			ch_index = np.where((ch >= 3) & (ch < 123))[0]
			ch_n1 = np.arange(3, 123, 1, dtype=int)
			t = t[ch_index]
			ch = ch[ch_index]
			bins = np.arange(t[0], t[-1], dt)
			bin_n, bin_edges = np.histogram(t, bins=bins)
			t_c = (bin_edges[1:] + bin_edges[:-1]) * 0.5
			rate = bin_n / dt
			t_c, cs_rate, bs_rate = TD_baseline(t_c, rate)
			new_c[ni] = [t_c, rate, bs_rate]
			if ni in good_ni:
				if rate.max() > maxx:
					maxx = rate.max()
				rate_sm = cs_rate + bs_rate.mean()
				bin_n_sm = np.round(rate_sm * dt)
				edges = bayesian_blocks(t_c, bin_n_sm, fitness='events', p0=0.05)
				result = background_correction(t_c, rate_sm, edges, degree=7)
				startedges, stopedges = get_bayesian_duration(result, sigma=3)
				new_c[ni + 'bb'] = [startedges, stopedges]
				if len(startedges) > 0:
					if os.path.exists(txtdir) == False:
						os.makedirs(txtdir)
					myfile.printdatatofile(txtdir + 'Z_' + ni + '_bayesian_duration.txt',
					                       data=[startedges, stopedges], format=['.5f', '.5f'])
					flash_start, flash_stop = get_bayesian_flash(result, startedges, stopedges)
					myfile.printdatatofile(txtdir + 'Y_' + ni + '_bayesian_flash.txt',
					                       data=[flash_start, flash_stop], format=['.5f', '.5f'])
					if txx:
						txx_result = get_bayesian_txx(result, startedges, stopedges, txx=0.9,
						                              it=400, lamd=200.)
						myplt = Plot(txx_result)
						plt.title(ni)
						myplt.plot_light_curve(sigma=5)
						plt.xlim(t[0], t[-1])
						plt.savefig(txtdir + 'X_' + ni + '_bayesian_txx.png')
						plt.close()
						print('***********', len(txx_result['txx']),
						      len(txx_result['txx_list']))
						for ij in range(len(txx_result['txx'])):
							plt.title(ni)
							myplt.plot_distribution('90', num=ij)
							plt.savefig(txtdir + 'W_' + ni + '_distribution_' + str(
								ij) + '.png')
							plt.close()
						plt.figure(figsize=(10, 10))
						plt.subplot(2, 1, 1)
						plt.title(ni)
						myplt.plot_Txx1('90')
						plt.xlim(t[0], t[-1])
						plt.subplot(2, 1, 2)
						myplt.plot_Txx2('90')
						plt.xlim(t[0], t[-1])
						plt.savefig(txtdir + 'U_' + ni + '_txx.png')
						plt.close()
						save_result(txx_result, txtdir + 'V_' + ni + '_distribution_T90.csv')
				
				if (ni == good_ni[0]):
					ni_event = Separate_source(t, ch, ch_n1, WT=WT)
					s_t, s_ch = ni_event.get_S_t_and_ch()
					new_t, new_energy = ch_to_energy(s_t, s_ch, ch_n, e1, e2)
					fig = plt.figure(figsize=(20, 20))
					ax1 = fig.add_subplot(2, 2, 1)
					ax1.set_title('light curve', size=20)
					ax1.step(t_c, rate, color='k', label=ni)
					ax1.set_xlabel('time (s)', size=20)
					ax1.set_ylabel('counts rate /s', size=20)
					ax1.set_xlim(t[0], t[-1])
					ax1.legend()
					ax2 = fig.add_subplot(2, 2, 2)
					ax2.set_title('point map', size=20)
					ax2.plot(new_t, new_energy, ',', color='k')
					ax2.set_xlabel('time (s)', size=20)
					ax2.set_ylabel('energy (kev)', size=20)
					ax2.set_yscale('log')
					ax2.set_xlim(t[0], t[-1])
					ax2.set_ylim(8, 9.1e2)
					ax3 = fig.add_subplot(2, 2, 3)
					ax3.step(t_c, rate, color='k', label=ni)
					if len(startedges) > 0:
						for i in range(len(startedges)):
							ax3.axvline(x=startedges[i], color='r')
							ax3.axvline(x=stopedges[i], color='g')
					ax3.set_xlabel('time (s)', size=20)
					ax3.set_ylabel('counts rate /s', size=20)
					ax3.set_xlim(t[0], t[-1])
					ax3.legend()
					ax4 = fig.add_subplot(2, 2, 4)
					ax4.plot(new_t, new_energy, ',', color='k')
					if len(startedges) > 0:
						for i in range(len(startedges)):
							ax4.axvline(x=startedges[i], color='r')
							ax4.axvline(x=stopedges[i], color='g')
					ax4.set_xlabel('time (s)', size=20)
					ax4.set_ylabel('energy (kev)', size=20)
					ax4.set_yscale('log')
					ax4.set_xlim(t[0], t[-1])
					ax4.set_ylim(8, 9.1e2)
					for k in plotsave:
						dir_, file_ = os.path.split(k)
						if os.path.exists(dir_) == False:
							os.makedirs(dir_)
						fig.savefig(k)
					plt.close(fig)
			else:
				new_c[ni + 'bb'] = None
		else:
			new_c[ni] = None
			new_c[ni + 'bb'] = None
	for bi in BGO:
		if file[bi] is not None:
			hl = file[bi]
			trigtime = hl[0].header['TRIGTIME']
			time = hl[2].data.field(0)
			t = time - trigtime
			bins = np.arange(t[0], t[-1], dt)
			bin_n, bin_edges = np.histogram(t, bins=bins)
			t_c = (bin_edges[1:] + bin_edges[:-1]) * 0.5
			rate = bin_n / dt
			t_c, cs_rate, bs_rate = TD_baseline(t_c, rate)
			new_c[bi] = [t_c, rate, bs_rate]
			if bi in good_bi:
				rate_sm = cs_rate + bs_rate.mean()
				bin_n_sm = np.round(rate_sm * dt)
				edges = bayesian_blocks(t_c, bin_n_sm, fitness='events', gamma=np.exp(-5))
				result = background_correction(t_c, rate_sm, edges, degree=7)
				startedges, stopedges = get_bayesian_duration(result, sigma=5)
				if len(startedges) > 0:
					if os.path.exists(txtdir) == False:
						os.makedirs(txtdir)
					myfile.printdatatofile(txtdir + 'Z_' + bi + '_bayesian_duration.txt',
					                       data=[startedges, stopedges], format=['.5f', '.5f'])
		else:
			new_c[bi] = None
	plt.figure(figsize=(30, 60))
	plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
	for index, value in enumerate(NaI):
		plt.subplot(14, 1, index + 1)
		if new_c[value] is not None:
			tm, rate, bs = new_c[value]
			plt.plot(tm, rate, color='k', label=value)
			plt.plot(tm, bs, color='r', label='back')
			if new_c[value + 'bb'] is not None:
				started, stoped = new_c[value + 'bb']
				for kk in range(len(started)):
					plt.axvline(x=started[kk], color='r')
					plt.axvline(x=stoped[kk], color='g')
			plt.ylabel('the count rate (N/s)')
			plt.xlim(tm[0], tm[-1])
			plt.ylim(0, maxx * 0.5 + maxx)
			plt.legend(loc='upper left')
	for index, value in enumerate(BGO):
		plt.subplot(14, 1, index + 13)
		if new_c[value] is not None:
			tm, rate, bs = new_c[value]
			plt.plot(tm, rate, color='k', label=value)
			plt.plot(tm, bs, color='r', label='back')
			plt.ylabel('the count rate (N/s)')
			plt.xlim(tm[0], tm[-1])
			plt.ylim(0, maxx * 0.5 + maxx)
			plt.legend(loc='upper left')
	for vv in plotsave1:
		dir_, file_ = os.path.split(vv)
		if os.path.exists(dir_) == False:
			os.makedirs(dir_)
		plt.savefig(vv)
	plt.close()


def get_fermi_geometry(hl):
	time1 = hl[5].data.field(0)
	time2 = hl[5].data.field(1)
	q4 = hl[5].data.field(2)
	sic = hl[5].data.field(3)
	t_c = (time2 + time1) * 0.5
	detectors = Detectors()
	fermi_time = Time_transform()
	fermi_gbm = Geometry(detector=detectors, time_base=fermi_time)
	fermi_gbm.input_pose(quaternion=q4, sc_pos=sic * u.km, time=t_c)
	return fermi_gbm


def plot_count_map(file, NaI, BGO, savedirlist):
	plt.figure(figsize=(40, 40))
	n = 1
	for ni in NaI:
		if file[ni] is not None:
			print('get ', ni)
			hl = file[ni]
			trigtime = hl[0].header['TRIGTIME']
			time = hl[2].data.field(0)
			ch = hl[2].data.field(1)
			t = time - trigtime
			ch_n = hl[1].data.field(0)
			e1 = hl[1].data.field(1)
			e2 = hl[1].data.field(2)
			plt.subplot(4, 4, n)
			plt.title(ni)
			ni_event = Separate_source(t, ch, ch_n, WT=False)
			s_t, s_ch = ni_event.get_S_t_and_ch()
			new_t, new_energy = ch_to_energy(s_t, s_ch, ch_n, e1, e2)
			plt.plot(new_t, new_energy, ',', color='k')
			plt.xlabel('time (s)')
			plt.ylabel('Energy (KeV)')
			plt.xlim(new_t[0], new_t[-1])
			plt.ylim(8, 900)
			plt.yscale('log')
			n = n + 1
	for bi in BGO:
		if file[bi] is not None:
			print('get ', bi)
			hl = file[bi]
			trigtime = hl[0].header['TRIGTIME']
			time = hl[2].data.field(0)
			ch = hl[2].data.field(1)
			t = time - trigtime
			ch_n = hl[1].data.field(0)
			e1 = hl[1].data.field(1)
			e2 = hl[1].data.field(2)
			plt.subplot(4, 4, n)
			plt.title(bi)
			ni_event = Separate_source(t, ch, ch_n, WT=False)
			s_t, s_ch = ni_event.get_S_t_and_ch()
			new_t, new_energy = ch_to_energy(s_t, s_ch, ch_n, e1, e2)
			plt.plot(new_t, new_energy, ',', color='k')
			plt.xlabel('time (s)')
			plt.ylabel('Energy (KeV)')
			plt.xlim(new_t[0], new_t[-1])
			plt.ylim(120, 20000)
			plt.yscale('log')
			n = n + 1
	for savedir in savedirlist:
		dir_, file = os.path.split(savedir)
		if os.path.exists(dir_) == False:
			os.makedirs(dir_)
		plt.savefig(savedir)
	plt.close()


def plot_sky_map(fermi_gbm, source, trigtime, savedirlist):
	t_c = fermi_gbm.Time_transition.batch_utc_to_met(fermi_gbm.time, astropyTime=True)
	index_0 = np.argmin((t_c - trigtime) ** 2)
	fermi_gbm.detector_plot(show_bodies=True, index=index_0, source=source)
	for savedir in savedirlist:
		dir_, file = os.path.split(savedir)
		if os.path.exists(dir_) == False:
			os.makedirs(dir_)
		plt.savefig(savedir)
	plt.close()


def get_file(input_list, NaI, BGO):
	sample = input_list[1]
	sampledir = input_list[0]
	data = {}
	loc_name_list = myfile.findfile(sampledir, 'glg_locprob_all_' + sample + '_v*')
	if len(loc_name_list) >= 1:
		data['loc'] = fits.open(sampledir + loc_name_list[0])
	else:
		loc_name_list = myfile.findfile(sampledir, 'glg_tcat_all_' + sample + '_v*')
		if len(loc_name_list) >= 1:
			data['loc'] = fits.open(sampledir + loc_name_list[0])
		else:
			data['loc'] = None
	trigdat_name_list = myfile.findfile(sampledir, 'glg_trigdat_all_' + sample + '_v*')
	if len(trigdat_name_list) >= 1:
		data['trigdat'] = fits.open(sampledir + trigdat_name_list[0])
	else:
		data['trigdat'] = None
	for ni in NaI:
		ni_list = myfile.findfile(sampledir, 'glg_tte_' + ni + '_' + sample + '_v*')
		if len(ni_list) >= 1:
			# print(ni)
			data[ni] = fits.open(sampledir + ni_list[0])
		else:
			data[ni] = None
	for bi in BGO:
		bi_list = myfile.findfile(sampledir, 'glg_tte_' + bi + '_' + sample + '_v*')
		if len(bi_list) >= 1:
			# print(bi)
			data[bi] = fits.open(sampledir + bi_list[0])
		else:
			data[bi] = None
	return data



def get_spectrum(t, ch, ch_n,bin_arr = None, edges=None, bg_dt=1.0):
	data = []
	data_err = []
	bins = np.arange(t[0], t[-1], bg_dt)
	for chi in ch_n:
		index = np.where(ch == chi)
		t_ch = t[index]
		bin_n, bin_edges = np.histogram(t_ch, bins=bins)
		bin_rate = bin_n / bg_dt
		bin_rate = np.concatenate((bin_rate[:1], bin_rate))
		bin_c, cs, bs = TD_baseline(bin_edges, bin_rate)
		if edges is not None:
			len_edges = edges[1:] - edges[:-1]
			bin_n1, bin_edges1 = np.histogram(t_ch, bins=edges)
			bin_c1 = (bin_edges1[1:] + bin_edges1[:-1]) * 0.5
			
		else:
			len_edges = bin_arr[:,-1]-bin_arr[:,0]
			bin_c1 = 0.5*(bin_arr[:,-1]+bin_arr[:,0])
			bin_n1 = overlap_curve(t_ch,bins_arr = bin_arr)
		bin_rate1 = bin_n1 / len_edges
		bs1 = np.interp(bin_c1, bin_c, bs)
		bin_err = np.sqrt(bin_n1) / len_edges
		bin_err[bin_err <= 0] = 1.0
		cs1 = bin_rate1 - bs1
		cs1[cs1 < 0] = 0
		data.append(list(cs1))
		data_err.append(list(bin_err))
	
	data = np.array(data).T
	data_err = np.array(data_err).T
	return data, data_err











