import numpy as np
import matplotlib.pyplot as plt
from ..Baseline import WhittakerSmooth

def background_correction(t,rate,edges,backgroundsize = 10,degree = 5,plot_save = None):
	'''
	
	:param t:
	:param rate:
	:param edges:
	:param degree:
	:return:
	'''
	re_rate,re_sigma,index_list = re_histogram(t,rate,edges)
	dt = t[1]-t[0]
	binsize = edges[1:]-edges[:-1]
	sort_index = np.argsort(-binsize)
	sort_binsize = binsize[sort_index]
	sort_sigma = re_sigma[sort_index]
	sort_re_rate = re_rate[sort_index]
	sort_index_list = np.array(index_list)[sort_index]
	background_pool = rate[sort_index_list[0]]
	correction_t = [edges[0],edges[-1]+dt,t[sort_index_list[0]][0],t[sort_index_list[0]][-1]]
	mean1 = sort_re_rate[0]
	correction_rate = [rate[0],rate[-1],mean1,mean1]
	sigma1 = sort_sigma[0]
	binsize1 = sort_binsize[0]
	n = 1
	for i in range(1,len(sort_binsize)):
		if binsize1>backgroundsize:
			binsize1_ = backgroundsize
		else:
			binsize1_ = binsize1
		if sort_binsize[i] > backgroundsize:
			sort_binsizei_ = backgroundsize
		else:
			sort_binsizei_ = sort_binsize[i]
		if confidence_analysis(mean1,sigma1,binsize1_,sort_re_rate[i],sort_sigma[i],sort_binsizei_,dt,degree) and len(t[sort_index_list[i]])>0:
			n = n+1
			correction_t.append(t[sort_index_list[i]][0])
			correction_t.append(t[sort_index_list[i]][-1])
			correction_rate.append(sort_re_rate[i])
			correction_rate.append(sort_re_rate[i])
			background_pool = np.concatenate((background_pool,rate[sort_index_list[i]]))
			mean1 = background_pool.mean()
			sigma1 = background_pool.std(ddof = 1)
			binsize1 = binsize1 + sort_binsize[i]
	#print('                                                   ',end='\r')
	#print('The number of background blocks is %d.'%n,end='\r')
	correction_t = np.array(correction_t)
	correction_rate = np.array(correction_rate)
	sort_t_index = np.argsort(correction_t)
	correction_t = correction_t[sort_t_index]
	correction_rate = correction_rate[sort_t_index]
	if plot_save is not None:
	
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.step(edges,np.concatenate((re_rate[:1],re_rate)))
		ax.plot(correction_t,correction_rate)
		fig.savefig(plot_save)
		plt.close(fig)
	
	correction_b = np.interp(t,correction_t,correction_rate)
	new_rate = rate - correction_b + correction_b.mean()
	re_rate,re_sigma,index_list = re_histogram(t,new_rate,edges)
	result = {'lc':(t,new_rate),
	          're_hist':(re_rate,re_sigma,index_list),
	          'bkg':(mean1,sigma1,binsize1),
	          'edges':edges,
	          'dt':dt
	          }
	return result

def get_SNR(edges,re_rate,background_mean,background_sigma,dt,non_negative = True):
	'''
	
	:param edges:
	:param re_rate:
	:param background_mean:
	:param background_sigma:
	:param dt:
	:param non_negative:
	:return:
	'''
	binsize = edges[1:] - edges[:-1]
	sigma = background_sigma * np.sqrt(dt/binsize)
	SNR = (re_rate - background_mean)/sigma
	if non_negative:
		if len(SNR[SNR<0])>0:
			SNR[SNR<0] = 0 #we only care about the part where the SNR is greater than 0.
	return SNR

def re_histogram(t,rate,edges):
	'''
	This is a stable function, which can re-bin the data 't' and 'rate' with the new bin edges 'edges'.
	
	:param t: The time of light curve.
	:param rate: The counts rate of light curve.
	:param edges: the new bin-edges, whose interval can different from each other.
	:return: three array ,The rebined rate and sigma of each bins.
	'''


	re_rate = np.zeros(len(edges)-1)
	re_sigma = np.zeros(len(edges)-1)
	
	index_list = []
	
	strat = edges[:-1]
	stop = edges[1:]
	for i in range(len(strat)):
		indexs = np.where((t>=strat[i])&(t<stop[i]))[0]
		index_list.append(indexs)
	'''
	index = np.where(t >= edges[0])[0]
	t = t[index]
	rate = rate[index]
	edges_num = 1
	ones = []
	for index1,value in enumerate(t):
		
		
		if value > edges[edges_num] or value == t[-1]:
			ones.append(index1)
			index_list.append(ones)
			ones = []
			edges_num = edges_num+1
			if edges_num == len(edges):
				break
		else:
			ones.append(index1)
	'''
	for index1,value in enumerate(index_list):
		one_rate = rate[value]
		index_list[index1] = value#+index[0]
		if len(one_rate) > 0:
			re_rate[index1] = one_rate.mean()
			if len(one_rate) == 1:
				re_sigma[index1] = one_rate.std(ddof=0)
			else:
				re_sigma[index1] = one_rate.std(ddof=1)
		else:
			if index1 == len(index_list)-1:
				re_rate[index1] = re_rate[index1-1]
				re_sigma[index1] = re_sigma[index1-1]
			else:
				re_rate[index1] = 0
				re_sigma[index1] = 1
				
	return re_rate,re_sigma,np.array(index_list)

def confidence_analysis(mean1,sigma1,T1,mean2,sigma2,T2,dt,degree = 3):
	
	'''
	conditional dependence of confidence
	:param mean1:
	:param sigma1:
	:param T1:
	:param mean2:
	:param sigma2:
	:param T2:
	:param dt:
	:param degree:
	:return: True or False
	'''
	
	sigma1_2 = sigma1*np.sqrt(dt/T2)
	sigma2_1 = sigma2*np.sqrt(dt/T1)
	return (mean1>=mean2-degree*sigma2_1)&(mean1<=mean2+degree*sigma2_1)&(mean2>=mean1-degree*sigma1_2)&(mean2<=mean1+degree*sigma1_2)


def get_bayesian_duration(data,sigma = 5,max_snr=False):
	'''
	
	:param data: the result from background_correction()
	:param sigma:
	:return:
	'''
	#start = 0
	#stop = 0
	pulse = 1
	cafe = 2
	fringe_up = 3
	fringe_down = 4
	fringe = 5
	#t,rate = data['lc']
	edges = data['edges']
	re_rate,re_sigma,index_list = data['re_hist']
	dt = data['dt']
	bkg,bkgsigma,bkgsize = data['bkg']
	SNR = get_SNR(edges,re_rate,bkg,bkgsigma,dt,non_negative=True)
	
	binstart = edges[:-1]
	binstop = edges[1:]
	center = 0.5*(binstart+binstop)
	trait = []
	for index1,hight in enumerate(re_rate):
		if (index1 == 0):
			if hight > re_rate[index1+1]:
				trait.append(fringe_down)
			else:
				trait.append(cafe)
			
		elif (index1 == len(re_rate)-1):
			
			if hight > re_rate[index1-1]:
				trait.append(fringe_up)
			else:
				trait.append(cafe)
				
		else:
			if (hight> re_rate[index1-1]) and (hight > re_rate[index1+1]):
				trait.append(pulse)
			elif (hight <= re_rate[index1-1]) and (hight <= re_rate[index1+1]):
				trait.append(cafe)
			elif (hight < re_rate[index1-1]) and (hight > re_rate[index1+1]):
				trait.append(fringe_down)
			elif (hight > re_rate[index1-1]) and (hight < re_rate[index1+1]):
				trait.append(fringe_up)
			else:
				trait.append(fringe)
	start_tag = False
	start_edges = []
	stop_edges = []
	good_edges = []
	for index,value in enumerate(trait):

		if (SNR[index]>sigma) and (start_tag == False):#get start
			
			if (value != fringe_down)and(value != cafe):
				good_edges.append(binstart[index])
				#start_edges.append(binstart[index])
				start_tag = True
				
		elif (SNR[index] <= sigma) and start_tag:#get stop
			
			if (value != fringe_up)and(value != pulse):
				good_edges.append(binstart[index])
				#stop_edges.append(binstart[index])
				start_tag = False
				if len(good_edges)==2:
					start_edges.append(min(good_edges))
					stop_edges.append(max(good_edges))
					good_edges = []
	
	#print(start_edges)
	if len(start_edges)>0:
		if start_edges[0] == binstart[0]:
			start_edges = start_edges[1:]
			stop_edges = stop_edges[1:]
	if max_snr:
		max_snr_list = []
		for i in range(len(start_edges)):
			indexi = np.where((center>=start_edges[i])&(center<=stop_edges[i]))[0]
			snri = SNR[indexi]
			max_snr_list.append(snri.max())
			
		return np.array(start_edges),np.array(stop_edges),np.array(max_snr_list)
	else:
		return np.array(start_edges),np.array(stop_edges)

def get_bayesian_flash(data,start_edges,stop_edges):
	'''
	
	:param data: the result from background_correction()
	:param start_edgs: the result from get_bayesian_duration()
	:param stop_edges:
	:param sigma:
	:return:
	'''
	start = 0
	stop = 0
	pulse = 1
	cafe = 2
	fringe = 3
	edges = data['edges']
	edges_c = (edges[1:] + edges[:-1])*0.5
	re_rate,re_sigma,index_list = data['re_hist']
	flash_start = []
	flash_stop = []
	for i in range(len(start_edges)):
		#select blocks and edges between the start edge and stop edges of a burst.
		indexi = np.where((edges_c>=start_edges[i])&(edges_c<=stop_edges[i]))[0]
		
		flash_start.append(start_edges[i])
		if len(indexi)>=3:
			#only the size of block larger than 3 can the burst has the bayesian flash.
			re_ratei = re_rate[indexi]
			edgesi_index = np.where((edges >= start_edges[i]) & (edges <= stop_edges[i]))[0]
			edgesi = edges[edgesi_index]
			edg_star = edgesi[:-1]
			edg_stop = edgesi[1:]
			trait = []
			for index1,hight in enumerate(re_ratei):
				if (index1 == 0):
					trait.append(start)
				elif (index1 == len(re_ratei)-1):
					trait.append(stop)
				else:
					if (hight> re_ratei[index1-1]) and (hight > re_ratei[index1+1]):
						trait.append(pulse)
					elif (hight < re_ratei[index1-1]) and (hight < re_ratei[index1+1]):
						trait.append(cafe)
					else:
						trait.append(fringe)
			trait = np.array(trait)
			indexj = np.where(trait == cafe)[0]
			
			if len(indexj) >= 3:
				for ij in indexj:
					flash_start.append(edg_stop[ij])  #the cafe stop edge is the start edge of a flash.
					flash_stop.append(edg_star[ij])   #the cafe start edge is the stop edge of a flash.
		flash_stop.append(stop_edges[i])
	return np.array(flash_start),np.array(flash_stop)
	
def get_bayesian_txx(data,t_start,t_stop,txx = 0.9,it = 400,lamd = 100.):
	'''
	
	:param data:
	:param t_start:
	:param t_stop:
	:param txx:
	:param it:
	:param lamd:
	:return:
	'''
	
	dt = data['dt']
	lamd = lamd / dt**2
	t,rate = data['lc']
	re_rate = data['re_hist'][0]
	n = rate*dt
	n_err = np.sqrt(n)
	dif = np.array(t_stop) - np.array(t_start)
	dif[dif > 20] = 20
	dif[dif <5] = 5
	tmin_ = t_start[0]-dif[0]#------------------
	if tmin_<t[0]:
		tmin_ = t[0]
	tmax_ = t_stop[-1]+dif[-1]#------------------
	if tmax_>t[-1]:
		tmax_ = t[-1]
	data_index = np.where((t>=tmin_)&(t<=tmax_))[0]
	txx = 1.-txx
	part_n = len(t_start)
	w = np.ones(len(t))
	for ij in range(len(t_start)):
		index_w = np.where((t>=t_start[ij])&(t<=t_stop[ij]))[0]
		w[index_w] = 0
	sigma = np.std(n[w > 0.5])
	bs1 = WhittakerSmooth(n, w, lamd)
	cs1 = n - bs1
	cs_f = np.cumsum(cs1)
	w1 = np.ones(len(cs_f))
	cs_fit = WhittakerSmooth(cs_f, w1, 1)
	t_e = t[data_index]
	cs_fe = cs_f[data_index]
	#durti = t_stop-t_start
	ns = 3*sigma#*durti
	duration = t_stop - t_start
	index_sort = np.argsort(duration)[0]


	if len(np.where((t>=t_start[index_sort]) & (t<= t_stop[index_sort]))[0])<100:
		d_t = (t_stop[index_sort]-t_start[index_sort])/100
		print('dt for interp:',d_t)
		t_l = np.arange(tmin_, tmax_ + d_t, d_t)
		cs_ff = np.interp(t_l, t, cs_fit)
	else:
		t_l = t[data_index]
		cs_ff = cs_fit[data_index]
	csf_fit_list = [np.mean(cs_fe[np.where(t_e < t_start[0])[0]])]
	for i in range(part_n):
		if (i < part_n - 1):
			cs1_fit_max = np.mean(cs_fe[np.where((t_e > t_stop[i]) & (t_e < t_start[i + 1]))[0]])
			csf_fit_list.append(cs1_fit_max)
		else:
			cs1_fit_max = np.mean(cs_fe[np.where(t_e > t_stop[i])[0]])
			csf_fit_list.append(cs1_fit_max)
	csf_fit_list = np.array(csf_fit_list)
	dcsf_fit_list = csf_fit_list[1:]-csf_fit_list[:-1]
	t90 = []
	index_i = []
	t1 = []
	t2 = []
	l1 = []
	l2 = []

	for index,dcsf in enumerate(dcsf_fit_list):
		if dcsf > ns:
			dd = txx * dcsf
			l1i = dd + csf_fit_list[index]
			l2i = csf_fit_list[index+1] - dd
			# ------------------------------------------------
			if 3 * duration[index] > 5:
				bb = 3 * duration[index]
			else:
				bb = 5
			t1_range1 = t_start[index] - bb
			t2_range2 = t_stop[index] + bb
			if index < len(t_start) - 1:
				
				if t2_range2 > t_start[index + 1]:
					t2_range2 = t_start[index + 1]
			if (index > 0):
				if t1_range1 < t_stop[index - 1]:
					t1_range1 = t_stop[index - 1]
			
			# ------------------------------------------------
			t90i, t1i, t2i = found_txx(t_l, cs_ff, l1i, l2i, csf_fit_list[index + 1], t1_range1, t2_range2)
			t90.append(t90i)
			t1.append(t1i)
			t2.append(t2i)
			l1.append(l1i)
			l2.append(l2i)
			index_i.append(index)
	fit_max = csf_fit_list[index_i+[index_i[-1]+1]]
	if len(t90)<1:
		return {'good':False}

	t90_list = []
	t1_list = []
	t2_list = []
	bs_list = []
	index_list = []
	nnn = 0
	nnnn=0
	while nnn < it:
		nnnn = nnnn+1
		try:
			
			bin_ratexx = n + n_err * np.random.randn(len(n_err))
			bs11 = WhittakerSmooth(bin_ratexx, w, lamd)

			cs11 = bin_ratexx - bs11
			cs11_f = np.cumsum(cs11)
			cs11_fit = WhittakerSmooth(cs11_f, w1, 1)
			cs11_fe = cs11_f[data_index]

			cs_ff1 = np.interp(t_l, t, cs11_fit)

			csf_fit_list1 = [np.mean(cs11_fe[np.where(t_e < t_start[0])[0]])]

			for i in range(part_n):
				if (i < part_n - 1):
					cs1_fit_max = np.mean(
						cs11_fe[np.where((t_e > t_stop[i]) & (t_e < t_start[i + 1]))[0]])
					csf_fit_list1.append(cs1_fit_max)
				else:
					cs1_fit_max = np.mean(cs11_fe[np.where(t_e > t_stop[i])[0]])
					csf_fit_list1.append(cs1_fit_max)
			csf_fit_list1 = np.array(csf_fit_list1)

			dcsf_fit_list1 = csf_fit_list1[1:] - csf_fit_list1[:-1]
			pp = 0
			for index, dcsf in enumerate(dcsf_fit_list1):
				if index in index_i:
					
					if dcsf > dcsf_fit_list[index]*0.1:

						#print(dcsf,ns)
						dd = txx * dcsf
						l11 = dd + csf_fit_list1[index]
						l21 = csf_fit_list1[index + 1] - dd
						
						if 3*t90[index]>5:
							bb = 3*t90[index]
						else:
							bb = 5
						t1_range1 = t_start[index]-bb
						t1_range2 = t_start[index]+bb
						t2_range1 = t_stop[index]-bb
						t2_range2 = t_stop[index]+bb
						
						if index <len(t_start)-1:
							if t1_range2>t_start[index+1]:
								t1_range2 = t_start[index+1]
							if t2_range2>t_start[index+1]:
								t2_range2 = t_start[index+1]
						if(index>0):
							if t1_range1<t_stop[index-1]:
								t1_range1 = t_stop[index-1]
							if t2_range1<t_stop[index-1]:
								t2_range1 = t_stop[index-1]
						#print(t1_range1,t1_range2,t2_range1,t2_range2)
						t90i, t1i, t2i = found_txx(t_l, cs_ff1, l11, l21,
						                           csf_fit_list1[index + 1], t1_range1,
						                           t2_range2)
						if ((t1i < t1_range2 and t1i > t1_range1) and (t2i > t2_range1 and t2i<t2_range2)):

							t90_list.append(t90i)
							t1_list.append(t1i)
							t2_list.append(t2i)
							index_list.append(index)
							pp = pp + 1

			if pp > 0:
				print(nnn,end = '\r')
				bs_list.append(bs11)
				nnn = nnn + 1
			
		except:
			continue
		if nnnn > it * 100:
			return {'good': False}

	t90_list = np.array(t90_list)
	t1_list = np.array(t1_list)
	t2_list = np.array(t2_list)
	index_list = np.array(index_list)

	new_t90_list = []
	new_t1_list = []
	new_t2_list = []
	t90_err1 = []
	t90_err2 = []
	t1_err1 = []
	t1_err2 = []
	t2_err1 = []
	t2_err2 = []
	for index,i in enumerate(index_i):
		i_index = np.where(index_list == i)
		t90_mean = np.mean(t90_list[i_index])
		t90_err = np.std(t90_list[i_index], ddof=0)
		t90_err1i = t90[index] - t90_mean + t90_err
		t90_err2i = t90_mean + t90_err - t90[index]
		if t90_err1i < 0:
			t90_err1i = 0
		if t90_err2i < 0:
			t90_err2i = 0
		t90_err1.append(t90_err1i)
		t90_err2.append(t90_err2i)
		t1_mean = np.mean(t1_list[i_index])
		t2_mean = np.mean(t2_list[i_index])
		
		t1_err = np.std(t1_list[i_index], ddof=0)
		t2_err = np.std(t2_list[i_index], ddof=0)
		
		t1_err1i = t1[index] - t1_mean + t1_err
		t1_err2i = t1_mean + t1_err - t1[index]
		if t1_err1i < 0:
			t1_err1i = 0
		if t1_err2i < 0:
			t1_err2i = 0
		t1_err1.append(t1_err1i)
		t1_err2.append(t1_err2i)
		t2_err1i = t2[index] - t2_mean + t2_err
		t2_err2i = t2_mean + t2_err - t2[index]
		if t2_err1i < 0:
			t2_err1i = 0
		if t2_err2i < 0:
			t2_err2i = 0
		t2_err1.append(t2_err1i)
		t2_err2.append(t2_err2i)
		new_t90_list.append(t90_list[i_index])
		new_t1_list.append(t1_list[i_index])
		new_t2_list.append(t2_list[i_index])
	
	
	result = {'good':True,'t_c':t,'rate':rate,'sigma':data['bkg'][1],'bs':WhittakerSmooth(rate,w,lambda_=lamd/dt),'bayesian_edges':[data['edges']],
	          'bayesian_rate':[np.concatenate((re_rate[:1], re_rate))],
		  'txx':t90,'txx_err':[t90_err1,t90_err2],
		  't1':t1,'t2':t2,'t1_err':[t1_err1,t1_err2],'t2_err':[t2_err1,t2_err2],
		  'txx_list':new_t90_list,'t1_list':new_t1_list,'t2_list':new_t2_list,
		  'cs_f_max':fit_max,'cs_f':cs_f,'xx':str(int(100*txx)),
		  't':t,'n':n,'bs_list':bs_list,'bs1':bs1,
		  'l':[l1,l2]}
	return result

def found_txx(t,v,st1,st2,base2,tmin,tmax):
	index_ = np.where((t>=tmin)&(t<=tmax))
	t = t[index_]
	v = v[index_]
	t1 = []
	for i in range(len(t)):
		if v[i] >= st1:
			if v[i] >= st2:
				if v[i]>base2:
					break
			else:
				t1.append(t[i])

		else:
			t1 = []
	t90 = t1[-1]-t1[0]
	return t90,t1[0],t1[-1]
