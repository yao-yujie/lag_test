from astropy.io import fits
import numpy as np
from scipy import optimize
from Data_analysis.Baseline import WhittakerSmooth
import Data_analysis.file as myfile
#import json
import pymultinest
from .analysis import Analyzer
from .PlotMarginalModes import PlotMarginalModes
from scipy.stats import chi2
import matplotlib.pyplot as plt
import ctypes
import sys
import os
paths = sys.path
c_lib_link = None
for path in paths:
	fand = path+'/Fermi_tool/c_lib/'
	if os.path.exists(fand):
		sonamelist = myfile.findfile(fand,'spectrum_tool.so')
		if len(sonamelist)>0:
			
			c_lib_link = fand+sonamelist[0]
			print('the C lib link is ',c_lib_link)
			break
if c_lib_link is not None:
	clib = ctypes.cdll.LoadLibrary(c_lib_link)
else:
	print('can not find the C lib of spectrum_tool.os!')
#clib = ctypes.cdll.LoadLibrary('/home/laojin/my_lat/python_c/spectrum_tool.so')

class Prior(object):
	def __init__(self,limit,log = False):
		self.log = log
		self.limit_width = max(limit)-min(limit)
		self.low = min(limit)
	def get_value(self,cube):
		if self.log:
			return 10**self.limit_width*cube+self.low
		else:
			return self.limit_width*cube+self.low
		
class Fit(object):
	
	def __init__(self,spectrumlist,model,priorlist,parameters,reference =False,reference_w = 0.0001):
		'''
		
		:param spectrumlist:
		:param modellist:
		'''
		
		self.spec_num = len(spectrumlist)
		self.spectrumlist = spectrumlist
		self.model = model
		self.prior_list = priorlist
		self.parameters = parameters
		self.n_params = len(parameters)
		self.reference = reference
		self.reference_rate = reference_w
		
	def prior(self,cube,ndim,nparams):
		for i in range(nparams):
			cube[i] = self.prior_list[i].get_value(cube[i])
		
	def log_like(self,cube,ndim,nparams):
		loglike = 0.
		#for spec_index in range(self.spec_num):
		for spec in self.spectrumlist:
			nn = len(spec.e_lo)
			spec1 = np.zeros(nn)
			#print('------------------------------------------------')
			#print('\r')
			for i in range(nn):
				e = np.linspace(spec.e_lo[i], spec.e_hi[i], 5)
				try:
					rate = self.model(e,cube)
					
					if True in np.isnan(rate):
						return -np.inf
					sp = rate.mean()
					spec1[i] = sp
				except:
					print('there are something wrong in your model!')
					print('the model`s return is ',cube)
					return -np.inf
			yu = spec.transform(spec1)
			
			
			if spec.effective_index is not None:
				if self.reference:
					loglike_data = -0.5*(((yu[spec.effective_index[0]:spec.effective_index[-1]] -spec.spectrum[spec.effective_index[0]:spec.effective_index[-1]])/spec.spectrum_err[spec.effective_index[0]:spec.effective_index[-1]])**2).sum()
					loglike_refer = - 0.5 * (((spec1[spec.effective_index1[0]:spec.effective_index1[-1]] - spec.reference[spec.effective_index1[0]:spec.effective_index1[-1]]) ) ** 2).sum()#/ spec.reference[spec.effective_index1[0]:spec.effective_index1[-1]]
					loglike = loglike + (1-self.reference_rate)*loglike_data + self.reference_rate*loglike_refer
				
				else:
					loglike = loglike-0.5*(((yu[spec.effective_index[0]:spec.effective_index[-1]] -spec.spectrum[spec.effective_index[0]:spec.effective_index[-1]])/spec.spectrum_err[spec.effective_index[0]:spec.effective_index[-1]])**2).sum()
			else:
				if self.reference:
					loglike_refer = - 0.5 * (((spec1 - spec.reference) ) ** 2).sum()#/ spec.reference
					loglike_data = -0.5*(((yu - spec.spectrum)/spec.spectrum_err)**2).sum()
					loglike = loglike + (1-self.reference_rate)*loglike_data + self.reference_rate*loglike_refer
				else:
					loglike = loglike-0.5*(((yu - spec.spectrum)/spec.spectrum_err)**2).sum()
		return loglike
	
	if c_lib_link is not None:
		def log_like_c(self,cube,ndim,nparams):
			loglike = 0.
			for spec in self.spectrumlist:
				e_add = spec.e_add
				try:
					rate = self.model(e_add,cube)
					if True in np.isnan(rate):
						return -np.inf
				except:
					print('there are something wrong in your model!')
					print('the model`s return is ',cube)
					return -np.inf
				spec1 = self.get_A(rate,spec.e_lo,spec.e_hi,spec.e_add_num)
				yu = spec.transform(spec1)
				if spec.effective_index is not None:
					if self.reference:
						loglike = loglike - 0.5 * (((spec1[spec.effective_index1[0]:spec.effective_index1[-1]] - spec.reference[spec.effective_index1[0]:spec.effective_index1[-1]]) / spec.reference[spec.effective_index1[0]:spec.effective_index1[-1]]) ** 2).sum()
					loglike = loglike-0.5*(((yu[spec.effective_index[0]:spec.effective_index[-1]] -spec.spectrum[spec.effective_index[0]:spec.effective_index[-1]])/spec.spectrum_err[spec.effective_index[0]:spec.effective_index[-1]])**2).sum()
				else:
					if self.reference:
						loglike = loglike - 0.5 * (((spec1 - spec.reference) / spec.reference) ** 2).sum()
					loglike = loglike-0.5*(((yu - spec.spectrum)/spec.spectrum_err)**2).sum()
			return loglike
	
	if c_lib_link is not None:
		def get_A(self,spe,e_lo,e_hi,add_n):
			n_spe = len(spe)
			n_e = len(e_lo)
			spe = (ctypes.c_double * n_spe)(*list(spe))
			e_lo = (ctypes.c_double * n_e)(*list(e_lo))
			e_hi = (ctypes.c_double * n_e)(*list(e_hi))
			ret = (ctypes.c_double * n_e)()
			clib.A_spec(spe,e_lo,e_hi,ret,n_spe,add_n,n_e)
			return np.array(ret)
	
	def run(self,outputfiles_basename,resume = False, verbose = True):
		'''
		
		:param outputfiles_basename:
		:param resume:
		:param verbose:
		:return:
		'''
		if c_lib_link is not None:
			pymultinest.run(self.log_like_c, self.prior, self.n_params, outputfiles_basename=outputfiles_basename,resume = resume, verbose = verbose)
		else:
			pymultinest.run(self.log_like, self.prior, self.n_params, outputfiles_basename=outputfiles_basename,resume = resume, verbose = verbose)
		#a1 = pymultinest.Analyzer(outputfiles_basename=outputfiles_basename, n_params = self.n_params)
		a1 = Analyzer(outputfiles_basename=outputfiles_basename, n_params = self.n_params)
		return a1
	
	def chi2_check(self,model_sp):
		
		retur_inde_list = []
		indexi = []
		sum = 0
		for index,i in enumerate(model_sp):
			sum = sum + i
			indexi.append(index)
			if sum > 5:
				retur_inde_list.append(indexi)
				sum = 0
				indexi = []
		if len(indexi)>0:
			retur_inde_list[-1] = retur_inde_list[-1] + indexi
		
		return retur_inde_list
		
	
	def get_chi2_text(self,a1):
		
		k = 0
		chi2_value = 0
		best_value = a1.get_best_fit()['parameters']
		r_ = len(best_value)
		for spec in self.spectrumlist:
			sp = spec.spectrum
			e_add = spec.e_add
			rate = self.model(e_add,best_value)
			spec1 = self.get_A(rate,spec.e_lo,spec.e_hi,spec.e_add_num)
			model_sp = spec.transform(spec1)
			effinde = spec.effective_index
			if effinde is not None:
				model_sp = model_sp[effinde[0]:effinde[-1]]
				sp = sp[effinde[0]:effinde[-1]]
				
			index_list = self.chi2_check(model_sp)
			new_model_sp = []
			new_sp = []
			for inde_i in index_list:
				new_model_sp.append(np.sum(model_sp[inde_i]))
				new_sp.append(np.sum(sp[inde_i]))
			new_model_sp = np.array(new_model_sp)
			new_sp = np.array(new_sp)
			chi2_s = (new_sp-new_model_sp)**2/new_model_sp
			k = k + len(chi2_s)
			chi2_value = chi2_value + chi2_s.sum()
			
			
		df = k - r_ - 1
		p = chi2.pdf(chi2_value,df)
		return {
			'p':p,
			'df':df,
			'chi2':chi2_value
		}
		
		
		
	def plot_model(self,a1,n = 0,ax = None,reference = True):
		'''
		
		:param a1:
		:param ax:
		:param reference:
		:return:
		'''
		e_c_list =[]
		for spec in self.spectrumlist:
			e_c = np.sqrt(spec.e_lo * spec.e_hi)
			e_c_list.append(e_c)
		for para in a1.get_equal_weighted_posterior()[::100,:-1]:
			for e_c in e_c_list:
				if ax is not None:
					ax.plot(e_c,e_c**n*self.model(e_c,para),'-', color='k', alpha=0.2)
				else:
					plt.plot(e_c,e_c**n*self.model(e_c,para),'-', color='k', alpha=0.2)
		best_value = a1.get_best_fit()['parameters']
		if reference:
			for index,spec in enumerate(self.spectrumlist):
				e_c,xxx = spec.get_reference()
				if ax is not None:
					ax.plot(e_c, e_c**n*xxx/(spec.e_hi-spec.e_lo), '.',label =spec.name +  ' reference')
				else:
					plt.plot(e_c,e_c**n*xxx/(spec.e_hi-spec.e_lo) , '.',label =spec.name + ' reference')
				
		for e_c in e_c_list:
			if ax is not None:
				ax.plot(e_c,e_c**n*self.model(e_c,best_value) , '-',label = 'best model')
			else:
				plt.plot(e_c,e_c**n*self.model(e_c,best_value) , '-',label = 'best model')
				
		
	def plot_data(self,a1,n = 0,ax = None,reference = True):
		'''
		
		:param a1:
		:param ax:
		:param reference:
		:return:
		'''
		best_value = a1.get_best_fit()['parameters']
		for spec in self.spectrumlist:
			e_c0 = spec.e_c
			e_c,xxx = spec.get_reference()
			xxx_sp = spec.transform(xxx)/(spec.e_max-spec.e_min)
			sp = spec.spectrum/(spec.e_max-spec.e_min)
			e_add = spec.e_add
			rate = self.model(e_add,best_value)
			spec1 = self.get_A(rate,spec.e_lo,spec.e_hi,spec.e_add_num)
			model_sp = spec.transform(spec1)/(spec.e_max-spec.e_min)
			sp_er = spec.spectrum_err/np.sqrt(spec.e_max-spec.e_min)
			effinde = spec.effective_index
			if effinde is not None:
				if ax is not None:
					ax.errorbar(e_c0[effinde[0]:effinde[-1]],e_c0[effinde[0]:effinde[-1]]**n*sp[effinde[0]:effinde[-1]],yerr = e_c0[effinde[0]:effinde[-1]]**n*sp_er[effinde[0]:effinde[-1]],elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
					if reference:
						ax.plot(e_c0[effinde[0]:effinde[-1]],e_c0[effinde[0]:effinde[-1]]**n*xxx_sp[effinde[0]:effinde[-1]], '-.',label = spec.name + ' reference')
					ax.plot(e_c0[effinde[0]:effinde[-1]],e_c0[effinde[0]:effinde[-1]]**n*model_sp[effinde[0]:effinde[-1]],label = spec.name + ' model')
				else:
					plt.errorbar(e_c0[effinde[0]:effinde[-1]],e_c0[effinde[0]:effinde[-1]]**n*sp[effinde[0]:effinde[-1]],yerr = e_c0[effinde[0]:effinde[-1]]**n*sp_er[effinde[0]:effinde[-1]],elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
					if reference:
						plt.plot(e_c0[effinde[0]:effinde[-1]],e_c0[effinde[0]:effinde[-1]]**n*xxx_sp[effinde[0]:effinde[-1]], '-.',label = spec.name + ' reference')
					plt.plot(e_c0[effinde[0]:effinde[-1]],e_c0[effinde[0]:effinde[-1]]**n*model_sp[effinde[0]:effinde[-1]],label = spec.name + ' model')
			else:
				if ax is not None:
					ax.errorbar(e_c0,e_c0**n*sp,yerr = e_c0**n*sp_er,fmt = '.',elinewidth=2,capsize=2,label = spec.name + ' data',alpha=0.3)
					if reference:
						ax.plot(e_c0,e_c0**n*xxx_sp, '-.',label = spec.name + ' reference')
					ax.plot(e_c0,e_c0**n*model_sp,label = spec.name + ' model')
				else:
					plt.errorbar(e_c0,e_c0**n*sp,yerr = e_c0**n*sp_er,fmt = '.',elinewidth=2,capsize=2,label = spec.name + ' data',alpha=0.3)
					if reference:
						plt.plot(e_c0,e_c0**n*xxx_sp, '-.',label = spec.name + ' reference')
					plt.plot(e_c0,e_c0**n*model_sp,label = spec.name + ' model')
	
	def plot_data_in_model(self,a1,ax = None):
		best_value = a1.get_best_fit()['parameters']
		for spec in self.spectrumlist:
			e_c0 = spec.e_c
			#e_c = spec.e_c1
			sp = spec.spectrum/(spec.e_max-spec.e_min)
			e_add = spec.e_add
			rate = self.model(e_add, best_value)
			spec1 = self.get_A(rate, spec.e_lo, spec.e_hi, spec.e_add_num)
			sp_model = spec.transform(spec1) / (spec.e_max - spec.e_min)
			sp_er = spec.spectrum_err/np.sqrt(spec.e_max-spec.e_min)
			effinde = spec.effective_index
			if effinde is not None:
				if ax is not None:
					ax.errorbar(e_c0[effinde[0]:effinde[-1]],sp[effinde[0]:effinde[-1]]-sp_model[effinde[0]:effinde[-1]],yerr = sp_er[effinde[0]:effinde[-1]],elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
				
				else:
					plt.errorbar(e_c0[effinde[0]:effinde[-1]],sp[effinde[0]:effinde[-1]]-sp_model[effinde[0]:effinde[-1]],yerr = sp_er[effinde[0]:effinde[-1]],elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
				
					
			else:
				if ax is not None:
					
					ax.errorbar(e_c0,sp-sp_model,yerr = sp_er,elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
					
				else:
					plt.errorbar(e_c0,sp-sp_model,yerr = sp_er,elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
		if ax is not None:
			ax.axhline(y=0,label = 'model',linestyle = '-.')
		else:
			plt.axhline(y=0,label = 'model',linestyle = '-.')
					
	def plot_data_in_reference(self,ax = None):
		for spec in self.spectrumlist:
			e_c0 = spec.e_c
			sp = spec.spectrum/(spec.e_max-spec.e_min)
			sp_er = spec.spectrum_err/np.sqrt(spec.e_max-spec.e_min)
			e_c,xxx = spec.get_reference()
			xxx_sp = spec.transform(xxx)/(spec.e_max-spec.e_min)
			effinde = spec.effective_index
			if effinde is not None:
				if ax is not None:
					ax.errorbar(e_c0[effinde[0]:effinde[-1]],sp[effinde[0]:effinde[-1]]-xxx_sp[effinde[0]:effinde[-1]],yerr = sp_er[effinde[0]:effinde[-1]],elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
				else:
					plt.errorbar(e_c0[effinde[0]:effinde[-1]],sp[effinde[0]:effinde[-1]]-xxx_sp[effinde[0]:effinde[-1]],yerr = sp_er[effinde[0]:effinde[-1]],elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
			else:
				if ax is not None:
					ax.errorbar(e_c0,sp-xxx_sp,yerr = sp_er,elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
				else:
					plt.errorbar(e_c0,sp-xxx_sp,yerr = sp_er,elinewidth=1,capsize=2,label = spec.name + ' data',alpha=0.5,fmt = '.')
		if ax is not None :
			ax.axhline(y = 0,label = 'reference',linestyle = '-.')
		else:
			plt.axhline(y=0,label = 'reference',linestyle = '-.')
			
			
			
	def plot_corner(self,a1):
		'''
		
		:param a1:
		:return:
		'''
		
		c = a1.get_stats()['modes'][0]
		#print('23232323')
		#print(a1.get_stats()['modes'])
		maximum = np.array(c['maximum'])
		
		mean = np.array(c['mean'])
		sigma = np.array(c['sigma'])

		'''
		equ_w = a1.get_equal_weighted_posterior()
		loglike = np.exp(equ_w.T[-1])
		loglikemin = loglike.min()
		loglikemax = loglike.max()
		d_like = loglikemax-loglikemin
	
		vv = loglikemax-0.954*d_like
		
		good_index = np.where(loglike>vv)[0]
		good_equ_w = equ_w[good_index][:,:-1]
		mean = good_equ_w.mean(axis = 0)
		sigma = 1*good_equ_w.std(ddof=1,axis =0)
		###
		'''
		dx = mean - maximum
		
		p = PlotMarginalModes(a1)
		sigmal = sigma - dx
		sigmah = sigma + dx
		uplims = np.zeros(len(sigmal),dtype=bool)
		lolims = np.zeros(len(sigmal),dtype=bool)
		#l_index = np.where(sigmal<0)[0]
		#h_index = np.where()
		lolims[sigmal<=0]=True
		uplims[sigmah<=0]=True
		sigmal[sigmal < 0] = 0
		sigmah[sigmah < 0] = 0
		plt.figure(figsize=(5*self.n_params,5*self.n_params))
		for i in range(self.n_params):
			for j in range(i,self.n_params):
				if(i == 0):
					plt.subplot(self.n_params, self.n_params,i*self.n_params+j+1)
					plt.title(self.parameters[j],size = 30)
					plt.tick_params(labelsize = 15)
					p.plot_marginal(j, with_ellipses = True, with_points = False, grid_points=50)
					plt.errorbar(maximum[j],0,xerr=[[sigmal[j]],[sigmah[j]]],xlolims=lolims[j],xuplims=uplims[j],
					             markersize=10,elinewidth = 3,
					             fmt = 'o',ecolor = 'r',capthick=5,color = 'r')
					if(j == 0):
						plt.ylabel("Probability",size = 30)
						plt.xlabel(self.parameters[j],size = 30)
					#plt.tight_layout()
				else:
					plt.subplot(self.n_params, self.n_params,i*self.n_params+j+1)
					plt.tick_params(labelsize = 15)
					p.plot_conditional(j, i-1, with_ellipses = False, with_points = False, grid_points=30)
					plt.errorbar(maximum[j], maximum[i-1], xerr=[[sigmal[j]],[sigmah[j]]], yerr=[[sigmal[i-1]],[sigmah[i-1]]],
					             xlolims=lolims[j],xuplims=uplims[j],uplims=uplims[i-1],lolims=lolims[i-1],
					             markersize=10,elinewidth = 3,
					             fmt='o', ecolor='r', capthick=5, color='r')
					if(j == i):
						plt.xlabel(self.parameters[j],size = 30)
						plt.ylabel(self.parameters[i-1],size = 30)
		
class Spectrum(object):
	'''
	
	'''
	def __init__(self,spectrum,spectrum_err,rsp_link,effective_band=None,
	             time = None,
	             spectrum_name = 'data',e_add_num = 20):
		hl = fits.open(rsp_link)
		matr = hl[2].data['MATRIX']
		matr[-1] = np.zeros(128)
		matr = np.vstack(matr)
		self.time = time
		self.matr = matr
		self.R = np.mat(matr)
		self.e_add_num = e_add_num
		self.name = spectrum_name
		self.e_lo = hl[2].data['ENERG_LO']#140
		self.e_hi = hl[2].data['ENERG_HI']
		self.e_min = hl[1].data['E_MIN']#128
		self.e_max = hl[1].data['E_MAX']
		
		self.e_c = np.sqrt(self.e_min*self.e_max)
		self.spectrum = np.array(spectrum)#/(self.e_max-self.e_min)
		self.spectrum_err = np.array(spectrum_err)#/np.sqrt(self.e_max-self.e_min)
		self.Y = np.mat(self.spectrum)
		self.Y_E = np.mat(self.spectrum_err)
		self.effective_band = effective_band
		if effective_band is not None:
			self.e_c1 = np.sqrt(self.e_lo*self.e_hi)
			index1 = np.where((self.e_c1>=effective_band[0])&(self.e_c1<=effective_band[1]))[0]
			self.effective_index1 = [index1[0],index1[-1]+1]
			#self.e_lo = self.e_lo[index1]
			#self.e_hi = self.e_hi[index1]
			#self.R = np.mat(self.matr[index1])
			
			index_ = np.where((self.e_c>=effective_band[0])&(self.e_c<=effective_band[1]))[0]
			self.effective_index = [index_[0],index_[-1]+1]
		else:
			self.effective_index1=None
			self.effective_index = None
		self.e_add = self.get_e_add(self.e_lo, self.e_hi, self.e_add_num)
		self.reference = self.get_reference()[1]
		
	def get_e_add(self,e_lo,e_hi,nn = 5):
		e_add = np.array([])
		for i in range(len(e_lo)):
			e = np.linspace(e_lo[i], e_hi[i], nn)
			e_add = np.concatenate((e_add,e))
		return e_add
	def change_spectrum(self,spectrum,spectrum_err,effective_band=None):
		self.spectrum = np.array(spectrum)
		self.spectrum_err = np.array(spectrum_err)
		self.Y = np.mat(self.spectrum)
		self.Y_E = np.mat(self.spectrum_err)
		if effective_band is not None:
			self.e_c1 = np.sqrt(self.e_lo*self.e_hi)
			index1 = np.where((self.e_c1>=effective_band[0])&(self.e_c1<=effective_band[1]))[0]
			self.effective_index1 = [index1[0],index1[-1]+1]
			#self.e_lo = self.e_lo[index1]
			#self.e_hi = self.e_hi[index1]
			#self.R = np.mat(self.matr[index1])
			index_ = np.where((self.e_c>=effective_band[0])&(self.e_c<=effective_band[1]))[0]
			self.effective_index = [index_[0],index_[-1]+1]
		self.e_add = self.get_e_add(self.e_lo, self.e_hi, self.e_add_num)
		self.reference = self.get_reference()[1]
		
	def get_reference(self):
		x0 = np.zeros(len(self.e_lo))
		bounds = [[0,+np.inf]]*len(self.e_lo)
		xxx = optimize.fmin_l_bfgs_b(self.fmin,x0,fprime = self.fprime,bounds = bounds,epsilon=1.e-10)[0]
		xxx1 = WhittakerSmooth(xxx,x0+1,1)
		e_c = np.sqrt(self.e_hi*self.e_lo)
		return e_c,xxx1
		
	def transform(self,model):
		model_mat = np.mat(model)
		return np.array(model_mat*self.R)[0]
	
	
	def fprime(self,x):
		X = np.mat(x)
		G = 2*((X*self.R-self.Y)/self.Y_E/self.Y_E)*self.R.T
		return np.array(G)[0]
	
	
	def fmin(self,x):
		X = np.mat(x)
		M = (X*self.R-self.Y)/self.Y_E
		S = M*M.T
		return S[0,0]








