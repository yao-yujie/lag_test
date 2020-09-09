
import numpy as np
from scipy.sparse import csc_matrix,eye,diags
from scipy.sparse.linalg import spsolve
import operator
from scipy import stats
from astropy.stats import sigma_clip,mad_std

def WhittakerSmooth(x,w,lambda_):
	'''

	:param x: array
	:param w: array .An array of weights corresponding to the values
	:param lambda_: Smoothing parameter
	:return: array Smoothing results
	'''
	
	X=np.mat(x)
	m=X.size
	#i=np.arange(0,m)
	E=eye(m,format='csc')
	D=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
	W=diags(w,0,shape=(m,m))
	A=csc_matrix(W+(lambda_*D.T*D))
	B=csc_matrix(W*X.T)
	background=spsolve(A,B)

	return np.array(background)

def get_w(cs,sigma):
	return np.exp(-0.5/(sigma**2)*(cs)**2)

def TD_bs(t,rate,it_ = 1,lambda_=4000,sigma = False,hwi = None,it = None,inti = None):
	dt = t[1]-t[0]
	t_c,cs,bs = TD_baseline(t,rate,hwi = hwi,it = it ,inti =inti)
	mask = sigma_clip(cs, sigma=5, maxiters=5, stdfunc=mad_std).mask
	myfilter = list(map(operator.not_, mask))
	lc_median_part = cs[myfilter]
	loc, scale = stats.norm.fit(lc_median_part)
	for i in range(it_):
		w = get_w(cs,scale)
		bs = WhittakerSmooth(rate,w,lambda_=lambda_/dt**1.5)
		cs = rate - bs
		mask = sigma_clip(cs, sigma=5, maxiters=5, stdfunc=mad_std).mask
		myfilter = list(map(operator.not_, mask))
		lc_median_part = cs[myfilter]
		loc, scale = stats.norm.fit(lc_median_part)
		
	if sigma:
		return cs,bs,scale
	else:
		return cs,bs


def TD_baseline(time,rate,lam = None,hwi = None,it = None,inti = None):
	'''
	
	:param time:
	:param rate:
	:param lam:
	:param hwi:
	:param it:
	:param inti:
	:return:
	'''
	dt = time[1]-time[0]
	if(lam is None):
		lam = 100/dt**1.5
	else:
		lam = lam/dt**1.5
	if(hwi is None):
		hwi = int(20/dt)
	else:
		hwi = int(hwi/dt)
	if(it is None):
		it = 5
	if(inti is None):

		fillpeak_int = int(len(rate)/10)

	else:
		fillpeak_int =inti
	if(lam < 1):
		lam = 1
	bs = baseline_kernel(rate,lambda_=lam,hwi=hwi,it = it,int_ = fillpeak_int)
	return time,rate-bs,bs


def get_smooth(spectra,lambda_):
	'''
	
	:param spectra:
	:param lambda_:
	:return:
	'''
	spectra = np.array(spectra)
	m = spectra.shape[0]
	w = np.ones(m)
	smooth = WhittakerSmooth(spectra,w,lambda_)
	cs = spectra-smooth
	cs_mean = cs.mean()
	cs_std = cs.std()
	for i in range(3):
		cs_index = np.where((cs>cs_mean+(1+1*i)*cs_std)|(cs<cs_mean-(1+1*i)*cs_std))
		w[cs_index] = 0
		smooth = WhittakerSmooth(spectra,w,lambda_)
		cs = spectra-smooth
		cs_mean = cs[w!=0].mean()
		cs_std = cs[w!=0].std()
	return smooth


def baseline_kernel(spectra,lambda_,hwi,it,int_):
	'''
	
	:param spectra:
	:param lambda_:
	:param hwi:
	:param it:
	:param int_:
	:return:
	'''
	spectra = np.array(spectra)
	spectra = get_smooth(spectra,lambda_)

	if it != 1 :
		d1 = np.log10(hwi)
		d2 = 0
		w = np.ceil(np.concatenate((10**(d1+np.arange(0,it-1,1)*(d2-d1)/(np.floor(it)-1)),[d2])))
		w = np.array(w,dtype = int)
	else:
		w = np.array([hwi],dtype = int)
	#print(w)

	lims = np.linspace(0,spectra.size -1,int_+1)
	lefts = np.array(np.ceil(lims[:-1]),dtype = int)#This is the index value
	rights = np.array(np.floor(lims[1:]),dtype = int)#Same as above
	minip = (lefts+rights)*0.5#The index
	xx = np.zeros(int_)
	for i in range(int_):
		xx[i] = spectra[lefts[i]:rights[i]+1].mean()

	
	for i in range(it):
		# Current window width
		w0 = w[i]
		# Point-wise iteration to the right
		for j in range(1,int_-1):
			# Interval cut-off close to edges
			v = min([j,w0,int_-j-1])
			# Baseline suppression
			a = xx[j-v:j+v+1].mean()
			xx[j] = min([a,xx[j]])
		for j in range(1,int_-1):
			k = int_-j-1
			v = min([j,w0,int_-j-1])
			a = xx[k-v:k+v+1].mean()
			xx[k] = min([a,xx[k]])

	minip = np.concatenate(([0],minip,[spectra.size-1]))
	xx = np.concatenate((xx[:1],xx,xx[-1:]))
	index = np.arange(0,spectra.size,1)
	xxx = np.interp(index,minip,xx)
	return xxx
























