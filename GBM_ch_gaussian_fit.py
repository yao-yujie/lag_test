# General analyses of GBM catalog bursts 
# last modified: Apr. 29, 2019

from astropy.io import fits
from astropy.time import Time
from astropy.stats import bayesian_blocks
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy import optimize
from scipy import signal
from glob import glob
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix,eye,diags
from scipy.sparse.linalg import spsolve
import operator
from scipy import stats
from astropy.stats import sigma_clip,mad_std
from scipy.interpolate import interp1d
import h5py
from scipy import stats
import os
import sys
import re
from multiprocessing import Pool
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)
import rpy2.robjects as robjects
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
robjects.numpy2ri.activate()
robjects.r("library(baseline,warn.conflicts = FALSE)")

#from xspec import *
import math
#import pymultinest
from PlotMarginalModes import PlotMarginalModes


#name=['bn150314205']
name = []
for line in open("sample.txt","r"):              
	name.append(line[:11])
nl=len(name)
#databasedir='/home/yao/burstdownloadyears'
#databasedir='/home/yao/文章结果/data/数据文件/data'
databasedir='/home/yao/GBM_burst_data/data'

NaI=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']
BGO=['b0','b1']
Det=['b0','b1','n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb']



ch1=3
ch2=124
ch3=25

ncore=10

mostbri_det=[]
time_slice=[]
time_slices=[]
epeak=[]
epeak_error_p=[]
epeak_error_n=[]

#
nnname=[]
t_start=[]
t_stop=[]
bn_year=[]
detector=[]

t90_start_time=[]
t90_stop_time=[]

def get_usersjar():
    usersjar = "/home/yao/Software/users.jar"
    return usersjar

def query_fermigbrst(cdir='./'):	
    fermigbrst = cdir+'/fermigbrst_test.txt'
    if not os.path.exists(fermigbrst):
        usersjar = get_usersjar()
        assert os.path.exists(usersjar), """'users.jar' is not available! 
            download users.jar at:
            https://heasarc.gsfc.nasa.gov/xamin/distrib/users.jar
            and update the path of usersjar in 'personal_settings.py'."""
        java_ready = os.system("java --version")
        assert not java_ready, """java not properly installed!
            Install Oracle Java 10 (JDK 10) in Ubuntu or Linux Mint from PPA
            $ sudo add-apt-repository ppa:linuxuprising/java
            $ sudo apt update
            $ sudo apt install oracle-java10-installer"""
        fields = ("trigger_name,t90,t90_error,t90_start,"
            "Flnc_Band_Epeak,scat_detector_mask")
        print('querying fermigbrst catalog using HEASARC-Xamin-users.jar ...')
        query_ready = os.system("java -jar "+usersjar+" table=fermigbrst fields="
                +fields+" sortvar=trigger_name output="+cdir+"/fermigbrst_test.txt")
        assert not query_ready, 'failed in querying fermigbrst catalog!'
        print('successful in querying fermigbrst catalog!')
    return fermigbrst

fermigbrst = query_fermigbrst()
df = pd.read_csv(fermigbrst,delimiter='|',header=0,skipfooter=3,engine='python')
trigger_name = df['trigger_name'].apply(lambda x:x.strip()).values
t90_str = df[df.columns[1]].apply(lambda x:x.strip()).values

t90_error_str = df[df.columns[2]].apply(lambda x:x.strip()).values
t90_start_str = df[df.columns[3]].apply(lambda x:x.strip()).values
Flnc_Band_Epeak_str = df[df.columns[4]].apply(lambda x:x.strip()).values
scat_detector_mask_str = df[df.columns[5]].apply(lambda x:x.strip()).values
burst_number = len(trigger_name)
print('burst_number = ',burst_number)

def norm_pvalue(sigma=3.0):
	p = stats.norm.cdf(sigma)-stats.norm.cdf(-sigma)
	return p

def poisson_k_shao(lamb,sigma=3.0):
	lamb = round(lamb)
	expected_pvalue = norm_pvalue_shao(sigma)
	k = max(lamb,1)
	current_pvalue = poisson.cdf(k,lamb)
	while (current_pvalue < expected_pvalue):
		k = k+1
		current_pvalue = poisson.cdf(k,lamb)
	return k # k in poisson distribution


def write_phaI(spectrum_rate,bnname,det,t1,t2,outfile):
	header0=fits.Header()
	header0.append(('creator', 'Shao', 'The name who created this PHA file'))
	header0.append(('telescop', 'Fermi', 'Name of mission/satellite'))
	header0.append(('bnname', bnname, 'Burst Name'))
	header0.append(('t1', t1, 'Start time of the PHA slice'))
	header0.append(('t2', t2, 'End time of the PHA slice'))
	
	hdu0=fits.PrimaryHDU(header=header0)
	
	a1 = np.arange(128)
	col1 = fits.Column(name='CHANNEL', format='1I', array=a1)
	col2 = fits.Column(name='COUNTS', format='1D', unit='COUNTS', array=spectrum_rate)
	hdu1 = fits.BinTableHDU.from_columns([col1, col2])
	header=hdu1.header
	header.append(('extname', 'SPECTRUM', 'Name of this binary table extension'))
	header.append(('telescop', 'GLAST', 'Name of mission/satellite'))
	header.append(('instrume', 'GBM', 'Specific instrument used for observation'))
	header.append(('filter', 'None', 'The instrument filter in use (if any)'))
	header.append(('exposure', 1., 'Integration time in seconds'))
	header.append(('areascal', 1., 'Area scaling factor'))
	header.append(('backscal', 1., 'Background scaling factor'))
	if outfile[-3:]=='pha':
		header.append(('backfile', det+'.bkg', 'Name of corresponding background file (if any)'))
		header.append(('respfile', det+'.rsp', 'Name of corresponding RMF file (if any)'))
	else:
		header.append(('backfile', 'none', 'Name of corresponding background file (if any)'))
		header.append(('respfile', 'none', 'Name of corresponding RMF file (if any)'))
	header.append(('corrfile', 'none', 'Name of corresponding correction file (if any)'))
	header.append(('corrscal', 1., 'Correction scaling file'))
	header.append(('ancrfile', 'none', 'Name of corresponding ARF file (if any)'))
	header.append(('hduclass', 'OGIP', 'Format conforms to OGIP standard'))
	header.append(('hduclas1', 'SPECTRUM', 'PHA dataset (OGIP memo OGIP-92-007)'))
	header.append(('hduclas2', 'TOTAL', 'Indicates gross data (source + background)'))
	header.append(('hduclas3', 'COUNT', 'Indicates data stored as counts'))
	header.append(('hduvers', '1.2.1', 'Version of HDUCLAS1 format in use'))
	header.append(('poisserr', True, 'Use Poisson Errors (T) or use STAT_ERR (F)'))
	header.append(('chantype', 'PHA', 'No corrections have been applied'))
	header.append(('detchans', 128, 'Total number of channels in each rate'))
	header.append(('hduclas4', 'TYPEI', 'PHA Type I (single) or II (mulitple spectra)'))
	
	header.comments['TTYPE1']='Label for column 1'
	header.comments['TFORM1']='2-byte INTERGER'
	header.comments['TTYPE2']='Label for column 2'
	header.comments['TFORM2']='8-byte DOUBLE'
	header.comments['TUNIT2']='Unit for colum 2'

	hdul = fits.HDUList([hdu0, hdu1])
	hdul.writeto(outfile)
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
def format_countmap_axes(ax, title, x1, x2,ymajor_ticks):
	ax.set_title(title,loc='right',fontsize=25,color='k')
	ax.set_xlim([x1,x2])
	ax.set_yscale('log')
	ax.yaxis.set_major_locator(ticker.FixedLocator(ymajor_ticks))
	#ax.yaxis.set_minor_locator(ticker.FixedLocator(yminor_ticks))
	#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
	#ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%d'))
	ax.tick_params(axis='both',top=True,right=True,length=5,width=2,\
								direction='out',which='both',labelsize=25)

def plot_countmap(bnname,resultdir,baseresultdir,datadir,content,tbins,viewt1,viewt2): 
	# content=['rate','base','net']
	BGOmaxcolorvalue=0.0
	NaImaxcolorvalue=0.0
	f=h5py.File(baseresultdir+'/base.h5',mode='r')
	fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
	for i in range(14):
		#data in firt and last two channels of BGO and NaI are not shown
		#ignore 0,1,126,127, notice 2-125
		if content=='rate':
			C=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch1,ch2+1) ])
		elif content=='base':
			C=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][1] \
									for ch in np.arange(ch1,ch2+1) ])
		elif content=='net':
			C=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][2] \
									for ch in np.arange(ch1,ch2+1) ])
		if i<=1:
			if BGOmaxcolorvalue < C.max():
				BGOmaxcolorvalue = C.max()
		else:
			if NaImaxcolorvalue < C.max():
				NaImaxcolorvalue = C.max()	
	for i in range(14):
		ttefile=glob(datadir+'/'+'glg_tte_'+Det[i]+'_'+bnname+'_v*.fit')
		hdu=fits.open(ttefile[0])
		ebound=hdu['EBOUNDS'].data
		emin=ebound.field(1)
		emin=emin[ch1:ch2+1]
		emax=ebound.field(2)
		emax=emax[ch1:ch2+1]				
		x = tbins
		y = np.concatenate((emin,[emax[-1]]))
		X, Y = np.meshgrid(x, y)
		if content=='rate':
			C=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch1,ch2+1) ])
		elif content=='base':
			C=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][1] \
									for ch in np.arange(ch1,ch2+1) ])
		elif content=='net':
			C=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][2] \
									for ch in np.arange(ch1,ch2+1) ])
		C[C<1]=1
		if i<=1:
			pcmBGO = axes[i//2,i%2].pcolormesh(X, Y, C,norm=\
					colors.LogNorm(vmin=1.0, vmax=BGOmaxcolorvalue),\
													cmap='rainbow')
			format_countmap_axes(axes[i//2,i%2],Det[i],tbins[0],\
											tbins[-1],[1000,10000])
		else:
			pcmNaI = axes[i//2,i%2].pcolormesh(X, Y, C,norm=\
					colors.LogNorm(vmin=1.0, vmax=NaImaxcolorvalue),\
													cmap='rainbow')
			format_countmap_axes(axes[i//2,i%2],Det[i],tbins[0],\
											tbins[-1],[10,100])
		axes[i//2,i%2].set_xlim([viewt1,viewt2])				
	cbarBGO=fig.colorbar(pcmBGO, ax=axes[0,], orientation='vertical', \
										fraction=0.005, aspect=100/6)
	cbarNaI=fig.colorbar(pcmNaI, ax=axes[1:,], orientation='vertical', \
										fraction=0.005, aspect=100)
	cbarBGO.ax.tick_params(labelsize=25)
	cbarNaI.ax.tick_params(labelsize=25)
	fig.text(0.07, 0.5, 'Energy (KeV)', ha='center', va='center', rotation='vertical',fontsize=30)
	fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)
	fig.text(0.5, 0.92, bnname, ha='center', va='center',fontsize=30)
	plt.savefig(resultdir+'/'+content+'_countmap.png')
	plt.close()
	f.close()

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
	#return time,rate-bs,bs
	return time,bs,rate

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

def copy_rspI(bnname,det,outfile):
	shortyear=bnname[2:4]
	fullyear='20'+shortyear
	datadir=databasedir+'/'+fullyear+'/'+bnname+'/'
	rspfile=glob(datadir+'/'+'glg_cspec_'+det+'_'+bnname+'_v*.rsp')
	assert len(rspfile)==1, 'response file is missing for '+'glg_cspec_'+det+'_'+bnname+'_v*.rsp'
	rspfile=rspfile[0]
	os.system('cp '+rspfile+' '+outfile)
	

class GRB:
	def __init__(self,bnname):
		self.bnname=bnname
		resultdir=os.getcwd()+'/results/'
		self.resultdir=resultdir+'/'+bnname+'/'

		shortyear=self.bnname[2:4]
		fullyear='20'+shortyear
		self.datadir=databasedir+'/'+fullyear+'/'+self.bnname+'/'
		self.dataready=True
		for i in range(14):
			ttefile=glob(self.datadir+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
			if not len(ttefile)==1:
				self.dataready=False
			else:
				hdu=fits.open(ttefile[0])
				event=hdu['EVENTS'].data.field(0)
				if len(event)<10:
					self.dataready=False
		if self.dataready:
			if not os.path.exists(resultdir):
				os.makedirs(resultdir)
			if not os.path.exists(self.resultdir):
				os.makedirs(self.resultdir)
			self.baseresultdir=self.resultdir+'/base/'
			self.phaIresultdir=self.resultdir+'/phaI/'

			# determine GTI1 and GTI2
			GTI_t1=np.zeros(14)
			GTI_t2=np.zeros(14)
			for i in range(14):
				ttefile=glob(self.datadir+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				time=data.field(0)-trigtime
				GTI0_t1=time[0]
				GTI0_t2=time[-1]
				timeseq1=time[:-1]
				timeseq2=time[1:]
				deltime=timeseq2-timeseq1
				delindex=deltime>5 
				if len(timeseq1[delindex])>=1:
					GTItmp_t1=np.array(np.append([GTI0_t1],timeseq2[delindex]))
					GTItmp_t2=np.array(np.append(timeseq1[delindex],[GTI0_t2]))
					for kk in np.arange(len(GTItmp_t1)):
						if GTItmp_t1[kk]<=0.0 and GTItmp_t2[kk]>=0.0:
							GTI_t1[i]=GTItmp_t1[kk]
							GTI_t2[i]=GTItmp_t2[kk]
				else:
					GTI_t1[i]=GTI0_t1
					GTI_t2[i]=GTI0_t2
			self.GTI1=np.max(GTI_t1)
			self.GTI2=np.min(GTI_t2)

		
	def rawlc(self,viewt1=-50,viewt2=300,binwidth=0.1):		
		viewt1=np.max([self.GTI1,viewt1])
		viewt2=np.min([self.GTI2,viewt2])
		assert viewt1<viewt2, self.bnname+': Inappropriate view times for rawlc!'
		if not os.path.exists(self.resultdir+'/'+'raw_lc.png'):
			#print('plotting raw_lc.png ...')
			tbins=np.arange(viewt1,viewt2+binwidth,binwidth)
			fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
			for i in range(14):
				ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				time=data.field(0)-trigtime
				ch=data.field(1)
				#data in firt and last two channels of BGO and NaI are not used
				#ignore 0,1,2,125,126,127, notice 3-124
				goodindex=(ch>=ch1) & (ch<=ch2)  
				time=time[goodindex]
				ebound=hdu['EBOUNDS'].data
				emin=ebound.field(1)
				emin=emin[ch1:ch2+1]
				emax=ebound.field(2)
				emax=emax[ch1:ch2+1]
				histvalue, histbin =np.histogram(time,bins=tbins)
				plotrate=histvalue/binwidth
				plotrate=np.concatenate(([plotrate[0]],plotrate))
				axes[i//2,i%2].plot(histbin,plotrate,drawstyle='steps')
				axes[i//2,i%2].set_xlim([viewt1,viewt2])
				axes[i//2,i%2].tick_params(labelsize=25)
				axes[i//2,i%2].text(0.05,0.85,Det[i],transform=\
									axes[i//2,i%2].transAxes,fontsize=25)
				axes[i//2,i%2].text(0.7,0.80,str(round(emin[0],1))+'-'+\
										str(round(emax[-1],1))+' keV',\
								transform=axes[i//2,i%2].transAxes,fontsize=25)
			fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',\
												 rotation='vertical',fontsize=30)
			fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
			fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)
			plt.savefig(self.resultdir+'/raw_lc.png')
			plt.close()


	def base(self,baset1=-50,baset2=300,binwidth=0.1):
		self.baset1=np.max([self.GTI1,baset1])
		self.baset2=np.min([self.GTI2,baset2])
		self.binwidth=binwidth
		self.tbins=np.arange(self.baset1,self.baset2+self.binwidth,self.binwidth)
		assert self.baset1<self.baset2, self.bnname+': Inappropriate base times!'
		if not os.path.exists(self.baseresultdir):
			os.makedirs(self.baseresultdir)
			expected_pvalue = norm_pvalue()
			f=h5py.File(self.baseresultdir+'/base.h5',mode='w')
			for i in range(14):
				grp=f.create_group(Det[i])
				ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[i]+'_'+\
                     							self.bnname+'_v*.fit')
				hdu=fits.open(ttefile[0])	
				trigtime=hdu['Primary'].header['TRIGTIME']
				data=hdu['EVENTS'].data
				timedata=data.field(0)-trigtime
				chdata=data.field(1)
				for ch in range(128):
					time_selected=timedata[chdata==ch]
					histvalue, histbin=np.histogram(time_selected,bins=self.tbins)
					rate=histvalue/binwidth
					ttime=(histbin[:-1]+histbin[1:])*0.5
					ttime,bs,cs=TD_baseline(ttime,rate)
					'''
					r.assign('rrate',rate) 
					r("y=matrix(rrate,nrow=1)")
					fillPeak_hwi=str(int(5/binwidth))
					fillPeak_int=str(int(len(rate)/10))
					r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi+", it=10,\
								 int ="+fillPeak_int+", method='fillPeaks')")
					r("bs=getBaseline(rbase)")
					r("cs=getCorrected(rbase)")
					bs=r('bs')#[0]
					cs=r('cs')[0]
					
					corrections_index= (bs<0)
					print('////p////////////////////////////////////////////////////',bs)
					bs[corrections_index]=0
					cs[corrections_index]=rate[corrections_index]
					'''
					
					f['/'+Det[i]+'/ch'+str(ch)]=np.array([rate,bs,cs])
			f.flush()
			f.close()

													
	def phaI(self,slicet1=0,slicet2=5):
		if not os.path.exists(self.phaIresultdir):
			os.makedirs(self.phaIresultdir)
		nslice=len(os.listdir(self.phaIresultdir))
		sliceresultdir=self.phaIresultdir+'/slice'+str(nslice)+'/'
		os.makedirs(sliceresultdir)
		fig, axes= plt.subplots(7,2,figsize=(32, 30),sharex=False,sharey=False)
		sliceindex= (self.tbins >=slicet1) & (self.tbins <=slicet2)
		valid_bins=np.sum(sliceindex)-1
		assert valid_bins>=1, self.bnname+': Inappropriate phaI slice time!'
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
		for i in range(14):
			total_rate=np.zeros(128)
			bkg_rate=np.zeros(128)
			total_uncertainty=np.zeros(128)
			bkg_uncertainty=np.zeros(128)
			ttefile=glob(self.datadir+'/glg_tte_'+Det[i]+'_'+self.bnname+'_v*.fit')
			hdu=fits.open(ttefile[0])
			ebound=hdu['EBOUNDS'].data
			emin=ebound.field(1)
			emax=ebound.field(2)
			energy_diff=emax-emin
			energy_bins=np.concatenate((emin,[emax[-1]]))
			for ch in range(128):
				base=f['/'+Det[i]+'/ch'+str(ch)][()][1]
				rate=f['/'+Det[i]+'/ch'+str(ch)][()][0]
				bkg=base[sliceindex[:-1]][:-1]
				total=rate[sliceindex[:-1]][:-1]
				bkg_rate[ch]=bkg.mean()
				total_rate[ch]=total.mean()
				exposure=len(bkg)*self.binwidth
				bkg_uncertainty[ch]=np.sqrt(bkg_rate[ch]/exposure)
				total_uncertainty[ch]=np.sqrt(total_rate[ch]/exposure)
			#plot both rate and bkg as count/s/keV
			write_phaI(bkg_rate,self.bnname,Det[i],slicet1,slicet2,sliceresultdir+'/'+Det[i]+'.bkg')
			write_phaI(total_rate,self.bnname,Det[i],slicet1,slicet2,sliceresultdir+'/'+Det[i]+'.pha')
			copy_rspI(self.bnname,Det[i],sliceresultdir+'/'+Det[i]+'.rsp')
			bkg_diff=bkg_rate/energy_diff
			total_diff=total_rate/energy_diff
			x=np.sqrt(emax*emin)
			axes[i//2,i%2].errorbar(x,bkg_diff,yerr=bkg_uncertainty/energy_diff,linestyle='None',color='blue')
			axes[i//2,i%2].errorbar(x,total_diff,yerr=total_uncertainty/energy_diff,linestyle='None',color='red')
			bkg_diff=np.concatenate(([bkg_diff[0]],bkg_diff))
			total_diff=np.concatenate(([total_diff[0]],total_diff))
			axes[i//2,i%2].plot(energy_bins,bkg_diff,drawstyle='steps',color='blue')
			axes[i//2,i%2].plot(energy_bins,total_diff,drawstyle='steps',color='red')
			axes[i//2,i%2].set_xscale('log')
			axes[i//2,i%2].set_yscale('log')
			axes[i//2,i%2].tick_params(labelsize=25)
			axes[i//2,i%2].text(0.85,0.85,Det[i],transform=\
										axes[i//2,i%2].transAxes,fontsize=25)
		fig.text(0.07, 0.5, 'Rate (count s$^{-1}$ keV$^{-1}$)', ha='center',\
							va='center', rotation='vertical',fontsize=30)
		fig.text(0.5, 0.05, 'Energy (keV)', ha='center', va='center',\
															fontsize=30)	
		plt.savefig(sliceresultdir+'/PHA_rate_bkg.png')
		plt.close()
		f.close()


	def specanalyze(self,slicename):
		slicedir=self.phaIresultdir+'/'+slicename+'/'
		os.chdir(slicedir)
		# select the most bright two NaIs (in channels 6-118) 
		# and more bright one BGO (in channels 4-124):
		BGOtotal=np.zeros(2)
		NaItotal=np.zeros(12)
		for i in range(2):
			phahdu=fits.open(slicedir+'/'+BGO[i]+'.pha')
			bkghdu=fits.open(slicedir+'/'+BGO[i]+'.bkg')
			pha=phahdu['SPECTRUM'].data.field(1)
			bkg=bkghdu['SPECTRUM'].data.field(1)
			src=pha-bkg
			plt.plot(src[4:125])
			plt.savefig(BGO[i]+'.png')
			plt.close()
			BGOtotal[i]=src[4:125].sum()
		for i in range(12):
			phahdu=fits.open(slicedir+'/'+NaI[i]+'.pha')
			bkghdu=fits.open(slicedir+'/'+NaI[i]+'.bkg')
			pha=phahdu['SPECTRUM'].data.field(1)
			bkg=bkghdu['SPECTRUM'].data.field(1)
			src=pha-bkg
			plt.plot(src[6:118])
			plt.savefig(NaI[i]+'.png')
			plt.close()
			NaItotal[i]=src[6:118].sum()
		BGOindex=np.argsort(BGOtotal)
		NaIindex=np.argsort(NaItotal)
		brightdet=[BGO[BGOindex[-1]],NaI[NaIindex[-1]],NaI[NaIindex[-2]]]
		
		# use xspec

		alldatastr=' '.join(det1[i]+'.pha' for i in mask)
		#alldatastr=' '.join([det+'.pha' for det in brightdet])
		print(alldatastr)
		#input('--wait--')
		AllData(alldatastr)
		AllData.show()
		AllData.ignore('1-(l-1):**-8.0,800.0-**  l:**-200.0,40000.0-**')
		print(AllData.notice)
		
		Model('grbm')
		Fit.statMethod='pgstat'
		Fit.nIterations=1000
		Fit.query = "yes"
		Fit.perform()
		
		
		Fit.error('3.0 3')
		Fit.perform()		
		Plot.device='/null'
		Plot.xAxis='keV'
		Plot.yLog=True
		Plot('eeufspec')



		for i in range(1,1+l):
			energies=Plot.x(i)
			rates=Plot.y(i)
			folded=Plot.model(i)
			xErrs=Plot.xErr(i)
			yErrs=Plot.yErr(i)
			plt.errorbar(energies,rates,xerr=xErrs,yerr=yErrs,zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('foldedspec.png')
		plt.close()
		Plot('eeufspec')

		for i in range(1,1+l):
			energies=Plot.x(i)
			ufspec=Plot.y(i)
			folded=Plot.model(i)
			xErrs=Plot.xErr(i)
			yErrs=Plot.yErr(i)
			plt.errorbar(energies,ufspec,xerr=xErrs,yerr=yErrs,zorder=1,ls='None')
			plt.plot(energies,folded,color='black',zorder=2)
		plt.xscale('log')
		plt.yscale('log')
		plt.savefig('eeufspec.png')
		plt.close()		

		par3=AllModels(1)(3)
		f = h5py.File(self.resultdir+"/data.h5", mode="w")
		epeak.append(par3.values[0])
		epeak_error_p.append(par3.error[0])
		epeak_error_n.append(par3.error[1])
		f = h5py.File("data.h5", mode="w")
		f["epeak"]=np.array(epeak)
		f["epeak_error_p"]=np.array(epeak_error_p)
		f["epeak_error_n"]=np.array(epeak_error_n)

		f.flush()
		f.close()		

		
		
	def removebase(self):
		os.system('rm -rf '+self.baseresultdir)



	def timeslice(self,lcbinwidth=0.1,gamma=1e-300):
		#fig = plt.figure()
		#ax1 = fig.add_subplot(111)
		#ax2 = ax1.twinx()
######
		file = glob(self.datadir+'glg_tte_'+Det[mostbri_det[0]]+'_'+self.bnname+'_v*.fit') 
		print(file)
		fitfile=file[0]	
		hdu=fits.open(fitfile)
		#data=hdu['events'].data['time']
		trigtime=hdu[0].header['TRIGTIME']
		data=hdu['EVENTS'].data
		time=data.field(0)-trigtime
		ch=data.field(1)
		index=np.where((ch<=124)&(ch>=3))
		time=time[index]
		tte=time[(time>-50)&(time<200)]

		#原始光变
		edges=np.arange(tte[0],tte[-1]+lcbinwidth,lcbinwidth)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/lcbinwidth
		plotrate=np.concatenate(([plotrate[0]],plotrate))
		#ax1.plot(plottime,plotrate,drawstyle='steps',color='red')
		plt.plot(plottime,plotrate,drawstyle='steps',color='red')
		#贝叶斯拟合
		edges = bayesian_blocks(plottime,plotrate,fitness='events',p0=1e-1, gamma=1e-300)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/(histbin[1:]-histbin[:-1])
		plotrate=np.concatenate(([plotrate[0]],plotrate)) 
		#ax1.plot(plottime,plotrate,drawstyle='steps',color='b')
		plt.plot(plottime,plotrate,drawstyle='steps',color='b')

		l=len(edges)		
		for i in range(1,l-1):
			time_slice.append(edges[i])
		print('time_slice',time_slice)
		#plt.vlines(edges,-100,60000,'blue',linestyle='dashed')
		plt.vlines(time_slice[0],-100,60000,'black',linestyle='dashed')
		plt.vlines(time_slice[-1],-100,60000,'black',linestyle='dashed')
		#plt.vlines(a,-100,60000,'orange',linestyle='dashed')
		#plt.vlines(b,-100,60000,'orange',linestyle='dashed')
		plt.ylim(0,12000)



		#输出epeak随时间的变化,用到光谱分析结果
		x=[]
		dx=[]
		for i  in range(l-3):    
			s=(edges[i+1]+edges[i+2])/2
			z=(edges[i+2]-edges[i+1])/2
			x.append(s)
			dx.append(z)
		print(len(x),len(epeak))
		dy=[epeak_error_p,epeak_error_n]
		
		#ax2.scatter(x,epeak,color='black', zorder=2,marker = '.',s=50.)    
		#ax2.errorbar(x,epeak,xerr=dx,yerr=dy,zorder=1, fmt='o',color = '0.15',markersize=1e-50)
		
		#ax2.set_ylim(0,600)
		#ax2.set_ylabel('Epeak')
		plt.savefig(self.resultdir+'bbdurations.png')
		plt.close()

	def BAT_timeslice(self,lcbinwidth=0.1,gamma=1e-300):
		#fig = plt.figure()
		#ax1 = fig.add_subplot(111)
		#ax2 = ax1.twinx()
		
######
		name=bnname[2:-3]
		ttefile=glob(self.datadir+'BAT/'+'all'+name+'.fits')
		hdu=fits.open(ttefile[0])
		trigtime=hdu['RATE'].header['TRIGTIME']
		data=hdu['RATE'].data
		alltime=data.field(0)-trigtime
		allrate=data.field(1)
		allerror=data.field(2)
		index=np.where((alltime>-10)&(alltime<30))
		time=alltime[index]
		rate=allrate[index]
		error=allerror[index]

		plt.plot(time,rate,drawstyle='steps',color='red')
		edges = bayesian_blocks(time,rate,error,fitness='measures',p0=1e-1,gamma=1e-200)
		
		l=len(edges)
		for i in range(1,l-1):
			time_slices.append(edges[i])
		print('time_slices',time_slices)
		plt.vlines(edges,0,15000,'blue',linestyle='dashed')
		plt.xlim(-10,30)
		plt.ylim(0,6)
		plt.savefig(self.resultdir+'BAT_bbdurations.png')
		plt.close()

	def bbhardness_ratio(self,lcbinwidth=0.1):
		file = glob(self.datadir+'glg_tte_'+Det[mostbri_det[0]]+'_'+self.bnname+'_v*.fit') 
		print(file)
		fitfile=file[0]	
		hdu=fits.open(fitfile)
		data=hdu['events'].data['time']
		trigtime=hdu[0].header['trigtime']
		time=data-trigtime
		tte=time[(time>-10)&(time<50)]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		#原始光变
		edges=np.arange(tte[0],tte[-1]+lcbinwidth,lcbinwidth)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/lcbinwidth
		plotrate=np.concatenate(([plotrate[0]],plotrate))
		ax1.plot(plottime,plotrate,drawstyle='steps',color='lightgreen')
		edges = bayesian_blocks(plottime,plotrate,fitness='events',p0=1e-1, gamma=1e-300)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/(histbin[1:]-histbin[:-1])
		plotrate=np.concatenate(([plotrate[0]],plotrate)) 
		ax1.plot(plottime,plotrate,drawstyle='steps',color='b')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Count')

		#硬度比     
		x=[]
		dx=[]
		l=len(edges)	
		print('edges',edges)
		for i  in range(l-3):    
			s=(edges[i+1]+edges[i+2])/2
			z=(edges[i+2]-edges[i+1])/2
			x.append(s)
			dx.append(z)
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
		
		cNetlo=np.array([f['/'+Det[mostbri_det[0]]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch1,ch3+1) ])
		cNethi=np.array([f['/'+Det[mostbri_det[0]]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch3,ch2+1) ])	
		totalNetlo=np.sum(cNetlo,axis=0)
		totalNethi=np.sum(cNethi,axis=0)
		hardness=totalNethi/totalNetlo
		hardness=np.concatenate(([hardness[0]],hardness))				
		edges=time_slice
		start=edges[:-1]
		stop=edges[1:]
		ever_rate=[]
		print('x',x)
		print(dx)
		for index,item in enumerate(start):
			t=np.where((self.tbins>=item)&(self.tbins<=stop[index]))[0]
			eva=hardness[t].mean()
			ever_rate.append(eva)		
		print('rate',ever_rate)

		ax2.scatter(x,ever_rate)
		ax2.errorbar(x,ever_rate,xerr=dx,zorder=1, fmt='o',color = '0.15',markersize=1e-50)
		ax2.set_ylim(0.5,2)
		ax2.set_ylabel('hardness ratio')
		
		plt.savefig(self.resultdir+bnname+'hardness_ratio.png')
		plt.close()
		
	def gbm_ch_netlc(self,lcbinwidth=1):
		fig, axes= plt.subplots(11,1,figsize=(10, 14),sharex='all',sharey=False)
		energy_edges=np.logspace(math.log(15,10),math.log(150,10),11)
		en_l=len(energy_edges)

		file = glob(self.datadir+'glg_tte_'+Det[mostbri_det[0]]+'_'+self.bnname+'_v*.fit') 
		ttefile=file[0]
		hdu=fits.open(ttefile)
		ebound=hdu['EBOUNDS'].data
		emin=ebound.field(1)
		emax=ebound.field(2)
		e=np.random.uniform(emin,emax)
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')

		for i in range(en_l):
			if i<en_l-1:
				indexs=np.where((e>=energy_edges[i])&(e<=energy_edges[i+1]))
				index=indexs[0]
				cNet=np.array([f['/'+Det[mostbri_det[0]]+'/ch'+str(ch)][()][2] \
										for ch in np.arange(index[0],index[-1])])		
				totalNet=np.sum(cNet,axis=0)
				totalNet=np.concatenate(([totalNet[0]],totalNet))	
				d=time_slice[0]
				s=time_slice[-1]
				xmajorlocator = MultipleLocator(5)
				xminorlocator = MultipleLocator(1)
				axes[-(i+1)].plot(self.tbins,totalNet,'blue',drawstyle='steps')
				axes[-(i+1)].vlines(d,0,15000,'red',linestyle='dashed')
				axes[-(i+1)].vlines(s,0,15000,'red',linestyle='dashed')
				axes[-(i+1)].set_xlim(-5,30)
				axes[-(i+1)].set_ylim(0,1000)
				axes[-(i+1)].xaxis.set_major_locator(xmajorlocator)
				axes[-(i+1)].xaxis.set_minor_locator(xminorlocator)
				axes[-(i+1)].text(20,800,str(round(energy_edges[i],1))+'-'+str(round(energy_edges[i+1],1))+' keV',fontsize=10)
				
			else:
				indexs=np.where((e>=energy_edges[0])&(e<=energy_edges[-1]))
				index=indexs[0]
				cNet=np.array([f['/'+Det[mostbri_det[0]]+'/ch'+str(ch)][()][2] \
											for ch in np.arange(index[0],index[-1])])
		
				totalNet=np.sum(cNet,axis=0)
				totalNet=np.concatenate(([totalNet[0]],totalNet))
				d=time_slice[0]
				s=time_slice[-1]

				xmajorlocator = MultipleLocator(5)
				xminorlocator = MultipleLocator(1)			
				axes[0].plot(self.tbins,totalNet,'blue',drawstyle='steps')		
				axes[0].vlines(d,0,70000,'red',linestyle='dashed')
				axes[0].vlines(s,0,70000,'red',linestyle='dashed')
				axes[0].xaxis.set_major_locator(xmajorlocator)
				axes[0].xaxis.set_minor_locator(xminorlocator)
				axes[0].set_xlim(-5,30)
				axes[0].set_ylim(0,5700)
				axes[0].text(20,4900,str(round(energy_edges[0],1))+'-'+str(round(energy_edges[-1],1))+' keV',fontsize=10)
				axes[0].set_title('GRB 150314A (GBM)')
		plt.subplots_adjust(hspace=0)						
		plt.savefig(self.resultdir+bnname+'_GBM_netlc.png')
		plt.close()

	def bat_ch_netlc(self,lcbinwidth=1):
		fig, axes= plt.subplots(11,1,figsize=(10, 14),sharex='all',sharey=False)
		energy_edges=np.logspace(math.log(15,10),math.log(150,10),11)
		en_l=len(energy_edges)
		name=bnname[2:-3]
		ttefile=glob(self.datadir+'BAT/'+name+'.fits')
		hdu=fits.open(ttefile[0])
		trigtime=hdu['RATE'].header['TRIGTIME']
		data=hdu['RATE'].data
		time=data.field(0)-trigtime
		rate=data.field(1)
		for i in range(en_l):
			if i<en_l-1:
				print(i)
				rrate=rate[:,i]
				d=time_slices[0]
				s=time_slices[-1]
				xmajorlocator = MultipleLocator(5)
				xminorlocator = MultipleLocator(1)
				axes[-(i+1)].xaxis.set_major_locator(xmajorlocator)
				axes[-(i+1)].xaxis.set_minor_locator(xminorlocator)
				axes[-(i+1)].plot(time,rrate,'blue',drawstyle='steps')
				axes[-(i+1)].vlines(d,0,1500,'red',linestyle='dashed')
				axes[-(i+1)].vlines(s,0,1500,'red',linestyle='dashed')
				axes[-(i+1)].set_xlim(-10,30)
				axes[-(i+1)].set_ylim(0,1)
				axes[-(i+1)].text(20,0.6,str(round(energy_edges[i],1))+'-'+str(round(energy_edges[i+1],1))+' keV',fontsize=10)
				
			else:
				ttefile=glob(self.datadir+'BAT/'+'all'+name+'.fits')
				hdu=fits.open(ttefile[0])
				trigtime=hdu['RATE'].header['TRIGTIME']
				data=hdu['RATE'].data
				time=data.field(0)-trigtime
				rate=data.field(1)
				d=time_slices[0]
				s=time_slices[-1]
				axes[0].plot(time,rate,'blue',drawstyle='steps')
				xmajorlocator = MultipleLocator(5)
				xminorlocator = MultipleLocator(1)
				axes[0].xaxis.set_major_locator(xmajorlocator)
				axes[0].xaxis.set_minor_locator(xminorlocator)			
				axes[0].set_xlim(-10,30)
				axes[0].set_ylim(0,5)
				axes[0].vlines(d,0,1500,'red',linestyle='dashed')
				axes[0].vlines(s,0,1500,'red',linestyle='dashed')
				axes[0].text(20,4,str(round(energy_edges[0],1))+'-'+str(round(energy_edges[-1],1))+' keV',fontsize=10)
				axes[0].set_title('GRB 150314A (BAT)')
		
		plt.savefig(self.resultdir+bnname+'_BAT_netlc.png')	
		plt.close()
	


	def netlc_ccf_test(self,viewt1=-50,viewt2=300,binwidth=0.1):
		
		energy_edges=np.logspace(math.log(15,10),math.log(150,10),11)
		en_l=len(energy_edges)
		
		def get_peak(x,corr,v=7,order = 12,num =10):
			max_index=np.argmax(corr)
			x_max = x[max_index]
			x_p = x[(x>=x_max-v)&(x<=x_max+v)]
			corr_p = corr[(x>=x_max-v)&(x<=x_max+v)]
			fit = np.polyfit(x_p, corr_p,order)
			x_new = np.linspace(x_p[0], x_p[-1], 1000)
			corr_new = np.polyval(fit,x_new)
			max_index = np.argmax(corr_new)
			x_max = x_new[max_index]
			x_array = np.zeros(num)
			for i in range(num):
				x_p = x[(x>=x_max-v)&(x<=x_max+v)]
				corr_p = corr[(x>=x_max-v)&(x<=x_max+v)]
				fit = np.polyfit(x_p, corr_p, order)
				x_new = np.linspace(x_p[0], x_p[-1], 1000)
				corr_new = np.polyval(fit,x_new)
				max_index = np.argmax(corr_new)
				x_max = x_new[max_index]
				x_array[i] = x_max
				#t_peak = x_array.sum()/x_array.size
			return x_max
######						
#根据能道区间生成的光子能量e与所要的区间进行对比决定是否选择该区间
#考虑找到正确的能道区间，得到索引值带入ch里13.750471949577264,
		#gbm
		viewt1=np.max([self.GTI1,viewt1])
		viewt2=np.min([self.GTI2,viewt2])
		tbins=np.arange(viewt1,viewt2+binwidth,binwidth)		
		ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[mostbri_det[0]]+'_'+self.bnname+'_v*.fit')
		hdu=fits.open(ttefile[0])
		trigtime=hdu['Primary'].header['TRIGTIME']
		data=hdu['EVENTS'].data
		time=data.field(0)-trigtime
		ch=data.field(1)
		ebound=hdu['EBOUNDS'].data
		emin=ebound.field(1)
		emax=ebound.field(2)
		e=np.random.uniform(emin,emax)
		
		#gbm基础数据	
		indexs=np.where((e>=energy_edges[0])&(e<=energy_edges[1]))
		index0=indexs[0]
		print(index0)
		needindex0=(ch>=index0[0]) & (ch<=index0[-1])
		time0=time[needindex0]
		histvalue0, histbin0 =np.histogram(time0,bins=tbins)
		plotrate0=histvalue0
		plotrate0=histvalue0/binwidth
		plotrate0_err = np.sqrt(plotrate0)
		
		time_newbin=(histbin0[1:]+histbin0[:-1])*0.5
		#time_index=(time_newbin>=time_slice[0])&(time_newbin<=time_slice[-1])
		time_index=(time_newbin>=t90_start[0])&(time_newbin<=t90_stop[0])
#	
		
		r.assign('rrate',plotrate0) 
		r("y=matrix(rrate,nrow=1)")
		fillPeak_hwi=str(int(5/binwidth))
		fillPeak_int=str(int(len(plotrate0)/10))
		r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi+", it=10,\
						 int ="+fillPeak_int+", method='fillPeaks')")
		r("bs=getBaseline(rbase)")
		r("cs=getCorrected(rbase)")
		bs0=r('bs')[0]
		cs0=r('cs')[0]
		
		cs_0=cs0[time_index]
	
		lag = [0]
		lag_err_l = [0]
		lag_err_h = [0]
		maxt=[]
		
		for i in range(1,en_l-1):
			index=np.where((e>=energy_edges[i])&(e<=energy_edges[i+1]))
			index1=index[0]
			print(index1)
			needindex1=(ch>=index1[0]) & (ch<=index1[-1])
			time1=time[needindex1]
			histvalue1, histbin1 =np.histogram(time1,bins=tbins)
			plotrate1=histvalue1
			plotrate1=histvalue1/binwidth
			plotrate1_err = np.sqrt(plotrate1)

			time_newbin=(histbin1[1:]+histbin1[:-1])*0.5
			r.assign('rrate',plotrate1) 
			r("y=matrix(rrate,nrow=1)")
			fillPeak_hwi=str(int(5/binwidth))
			fillPeak_int=str(int(len(plotrate1)/10))
			r("rbase=baseline(y,lam = 6,hwi="+fillPeak_hwi+", it=10,\
						 int ="+fillPeak_int+", method='fillPeaks')")
			r("bs=getBaseline(rbase)")
			r("cs=getCorrected(rbase)")
			bs1=r('bs')[0]
			cs1=r('cs')[0]
			cs_1=cs1[time_index]
			#corr =np.correlate(cs_1,cs_0,'full')
			corr =signal.correlate(cs_1,cs_0)
			n=len(cs_1)
			newtime=time_newbin[time_index]
			dt=(newtime[-1] - newtime[0])/len(newtime)
			x =  np.arange(-n+1,n,1)*dt
			max_index=np.argmax(corr)
			lags = get_peak(x,corr,order=12,v=7,num = 10)
			lag.append(lags)
			#ccf拟合图像
			index_array = np.where((x >= lags-7)&(x<=lags+7))
			fit_nccf = corr[index_array]
			fit_lag_t = x[index_array]
			new_t = np.linspace(fit_lag_t[0],fit_lag_t[-1],1000)
			c = np.polyfit(fit_lag_t,fit_nccf,12)
			yy = np.polyval(c,new_t)
			
			plt.plot(fit_lag_t,fit_nccf)
			plt.plot(new_t,yy)
			plt.vlines(lags,0,9000000,'red')
			plt.savefig(self.resultdir+'gbmccf'+str(i)+'.png')
			plt.close()
		
			#蒙特卡洛误差
			err_list=[]
			for i in range(10):
				rand = np.random.randn(n)
				cs_0_err=plotrate0_err[time_index]
				cs_1_err=plotrate1_err[time_index]
				cs_0_e = cs_0 + cs_0_err*rand
				cs_1_e = cs_1 + cs_1_err*rand
				err_corr =signal.correlate(cs_1_e,cs_0_e)
				#err_corr =np.correlate(cs_1_e,cs_0_e,'full')
				errlist = get_peak(x,err_corr,order=12,v=7,num = 5)
				err_list.append(errlist)
	
			lagerr_array = np.array(err_list)
			lag_sigma = lagerr_array.std()
			lag_ave = lagerr_array.mean()
			
			dx = lag_ave-lags
			lag_errh = lag_sigma-dx
			lag_errl = lag_sigma+dx
			lag_errh=lag_errh.tolist()
			lag_errl=lag_errl.tolist()
			lag_err_h.append(lag_errh)
			lag_err_l.append(lag_errl)
			
		lag_err=[lag_err_l,lag_err_h]		
		print('gbm',lag)
		
		'''
		#bat
		name=bnname[2:-3]
		bat_file=glob(self.datadir+'BAT/'+name+'.fits')
		bat_hdu=fits.open(bat_file[0])
		battrigtime=bat_hdu['RATE'].header['TRIGTIME']
		bat_data=bat_hdu['RATE'].data
		all_time=bat_data.field(0)-battrigtime
		all_rate=bat_data.field(1)
		all_error=bat_data.field(2)
		bd=time_slices[0]
		bs=time_slices[-1]	
		indexs=np.where((all_time>=bd)&(all_time<=bs))
		bat_time=all_time[indexs]	
		bat_rate=all_rate[indexs]
		bat_error=all_error[indexs]
		
		rrate=bat_rate[:,0]
		rerror=bat_error[:,0]
	
		bat_lag = [0]
		bat_lag_err_l = [0]
		bat_lag_err_h = [0]
		
		
		for i in range(1,en_l-1):
			new_rate=bat_rate[:,i]
			new_error=bat_error[:,i]
			#corr =np.correlate(cs_1,cs_0,'full')
			corr =signal.correlate(new_rate,rrate)
			n=len(new_rate)
			dt=(bat_time[-1] - bat_time[0])/len(bat_time)
			x =  np.arange(-n+1,n,1)*dt
			max_index=np.argmax(corr)
			#
			t=7
			bat_lags = get_peak(x,corr,order=12,v=t,num = 10)
			bat_lag.append(bat_lags)
			#ccf拟合图像
			index_array = np.where((x >= lags-t)&(x<=lags+t))
			fit_nccf = corr[index_array]
			fit_lag_t = x[index_array]
			new_t = np.linspace(fit_lag_t[0],fit_lag_t[-1],1000)
			c = np.polyfit(fit_lag_t,fit_nccf,12)
			yy = np.polyval(c,new_t)
			
			plt.plot(fit_lag_t,fit_nccf)
			plt.plot(new_t,yy)
			plt.vlines(bat_lags,0,7,'red')
			plt.savefig(self.resultdir+'batccf'+str(i)+'.png')
			plt.close()
	
			#蒙特卡洛误差
			bat_err_list=[]
			for i in range(10):
				rand = np.random.randn(n)
				n_rrate = rrate + rerror*rand
				n_new_rate = new_rate + new_error*rand
				err_corr =signal.correlate(n_new_rate,n_rrate)
				#err_corr =np.correlate(n_new_rate,n_rrate,'full')
				errlist = get_peak(x,err_corr,order=12,v=t,num = 5)
				bat_err_list.append(errlist)
			
			bat_lagerr_array = np.array(bat_err_list)
			bat_lag_sigma = bat_lagerr_array.std()
			bat_lag_ave = bat_lagerr_array.mean()
			
			dx = bat_lag_ave-bat_lags
			bat_lag_errh =bat_lag_sigma-dx
			bat_lag_errl =bat_lag_sigma+dx
			bat_lag_errh=bat_lag_errh.tolist()
			bat_lag_errl=bat_lag_errl.tolist()
			bat_lag_err_h.append(bat_lag_errh)
			bat_lag_err_l.append(bat_lag_errl)

		bat_lag_err=[bat_lag_err_l,bat_lag_err_h]
		print('bat',bat_lag)
		'''
		#横坐标:能量if not os.path.exists("chains"): os.mkdir("chains")	
		E=[]
		E_err=[]
		
		for i in range(en_l-1):
			a=(energy_edges[i]+energy_edges[i+1])/2
			E.append(a)
			b=(energy_edges[i+1]-energy_edges[i])/2
			E_err.append(b)
		'''
		#光谱延迟pymultinest,加入红移求延迟和指数
		E=np.array(E)
		gbm_t=np.array(lag)
		gbm_t_err=np.array(lag_err)
		gbm_t_err1 = gbm_t_err[0]
		gbm_t_err2 = gbm_t_err[1]
		
		bat_t=np.array(bat_lag)
		bat_t_err=np.array(bat_lag_err)
		bat_t_err1 = bat_t_err[0]
		bat_t_err2 = bat_t_err[1]
		os.chdir(self.resultdir)
		
		if not os.path.exists("batchains"): os.mkdir("batchains")
		if not os.path.exists("gbmchains"): os.mkdir("gbmchains")
		def test(nname,t,t_err,t_err1,t_err2):
			def model(E,b,s):
				return (1+1.7580)*s*((((1+1.7580)*E)**-b)-((1+1.7580)*16.95)**-b)
		

			def myprior(cube, ndim, nparams):
				cube[0] = cube[0] 
				cube[1] = cube[1] * 20 		

			def myloglike(cube, ndim, nparams):
				b = cube[0]
				s = cube[1]
				dm = t-model(E,b,s)
				t_err = t_err2
				index = np.where(dm>0)[0]
				t_err[index] = t_err1[index]
				t_err[t_err==0] =1
				return -0.5*np.sum(((t-model(E,b,s))/t_err)**2)

			parameters = ["β", "τ"]
			n_params = len(parameters)
			#progress=pymultinest.watch.ProgressPlotter(n_params, outputfiles_basename='chains/1-')
			#progress.start()
			pymultinest.run(myloglike, myprior, n_params, resume = 	False, verbose = 							False,write_output=True,outputfiles_basename=(nname+'chains/1-'))			
			a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename=(nname+'chains/1-'))
			bestfit_params = a.get_best_fit()
			print('bestfit_params',bestfit_params)
			stats = a.get_stats()

			pars=bestfit_params['parameters']
			values=stats['modes'][0]
			sigma=values['sigma']
			mean=values['mean']
			print('fit_pars',pars,sigma,mean)
			#progress.stop()
			p =PlotMarginalModes(a)
			plt.figure(figsize=(5*n_params,5*n_params))
			for i in range(n_params):
				for j in range(i,n_params):
					if(i == 0):	
						plt.subplot(n_params, n_params,i*n_params+j+1)
						plt.title(parameters[j],size = 30)
						plt.tick_params(labelsize = 15)
						p.plot_marginal(j, with_ellipses = True, with_points = False, grid_points=50)
						if(j == 0):
							plt.ylabel("Probability",size = 30)
							plt.xlabel(parameters[j],size = 30)
					else:
						plt.subplot(n_params, n_params,i*n_params+j+1)
						plt.tick_params(labelsize = 15)
						p.plot_conditional(j, i-1, with_ellipses = False, with_points = False, grid_points=30)
						if(j == i):
			
							plt.xlabel(parameters[j],size = 30)
							plt.ylabel(parameters[i-1],size = 30)
		
			
			plt.savefig(nname+'_test.png')	
			plt.close()
			return pars,sigma,mean

		gbm_pars,gbm_sigma,gbm_mean = test(nname = 'gbm',t = gbm_t,t_err = gbm_t_err,t_err1 = gbm_t_err1,t_err2 = gbm_t_err2)
		#bat_pars,bat_sigma,bat_mean = test(nname = 'bat',t = bat_t,t_err = bat_t_err,t_err1 = bat_t_err1,t_err2 = bat_t_err2)
		
		gbm_bv_err_h=[]
		gbm_bv_err_l=[]
		gbm_bv=gbm_pars[0]
		gbm_bv_err_h_v=gbm_sigma[0]+gbm_mean[0]-gbm_pars[0]
		gbm_bv_err_l_v=gbm_sigma[0]-gbm_mean[0]+gbm_pars[0]
		gbm_bv_err_h.append(gbm_bv_err_h_v)
		gbm_bv_err_l.append(gbm_bv_err_l_v)
		gbm_bv_err=[gbm_bv_err_l,gbm_bv_err_h]
		
		gbm_tv_err_h=[]
		gbm_tv_err_l=[]
		gbm_tv=gbm_pars[1]
		gbm_tv_err_h_v=gbm_sigma[1]+gbm_mean[1]-gbm_pars[1]
		gbm_tv_err_l_v=gbm_sigma[1]-gbm_mean[1]+gbm_pars[1]
		gbm_tv_err_h.append(gbm_tv_err_h_v)
		gbm_tv_err_l.append(gbm_tv_err_l_v)
		gbm_tv_err=[gbm_tv_err_l,gbm_tv_err_h]
		
		bat_bv_err_h=[]
		bat_bv_err_l=[]		
		bat_bv=bat_pars[0]
		bat_bv_err_h_v=bat_sigma[0]+bat_mean[0]-bat_pars[0]
		bat_bv_err_l_v=bat_sigma[0]-bat_mean[0]+bat_pars[0]
		bat_bv_err_h.append(bat_bv_err_h_v)
		bat_bv_err_l.append(bat_bv_err_l_v)
		bat_bv_err=[bat_bv_err_l,bat_bv_err_h]

		bat_tv_err_h=[]
		bat_tv_err_l=[]	
		bat_tv=bat_pars[1]
		bat_tv_err_h_v=bat_sigma[1]+bat_mean[1]-bat_pars[1]
		bat_tv_err_l_v=bat_sigma[1]-bat_mean[1]+bat_pars[1]
		bat_tv_err_h.append(bat_tv_err_h_v)
		bat_tv_err_l.append(bat_tv_err_l_v)
		bat_tv_err=[bat_tv_err_l,bat_tv_err_h]
		'''
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		left, bottom, width, height = 0.56, 0.55,  0.3, 0.3
		ax1.scatter(E,lag,c='blue',s=17)
		#ax1.scatter(E,bat_lag,c='red',s=17)
		ax1.errorbar(E,lag,xerr=E_err,yerr=lag_err,c='deepskyblue',fmt='o',linewidth=0.8,capthick=0.5,capsize=1.1)
		#ax1.errorbar(E,bat_lag,xerr=E_err,yerr=bat_lag_err,c='red',fmt='o',linewidth=0.8,capthick=0.5,capsize=1.1)
		'''		
		xmajorlocator = MultipleLocator(20)
		xminorlocator = MultipleLocator(5)			
		ax1.xaxis.set_major_locator(xmajorlocator)
		ax1.xaxis.set_minor_locator(xminorlocator)
		ymajorlocator = MultipleLocator(0.2)
		yminorlocator = MultipleLocator(0.05)			
		ax1.yaxis.set_major_locator(ymajorlocator)
		ax1.yaxis.set_minor_locator(yminorlocator)
		'''
		lag_arr=np.array(lag)
		l=lag_arr.min()
		l=l-0.2
		ax1.set_xlim(13,154)
		ax1.set_ylim(l,0.04)
		ax1.set_xlabel('E(kev)')
		ax1.set_ylabel('Mock arrival Time (s)')
		x=np.arange(E[0],E[-1],0.1)
		#gbm_y=(1+1.7580)*gbm_tv*((((1+1.7580)*x)**-gbm_bv)-((1+1.7580)*16.95)**-gbm_bv)
		#bat_y=(1+1.7580)*bat_tv*((((1+1.7580)*x)**-bat_bv)-((1+1.7580)*16.95)**-bat_bv)	
		#plt.plot(x,gbm_y,c='blue')
		#plt.plot(x,bat_y,c='red')
		'''
		ax2 = fig.add_axes([left, bottom, width, height])
		ax2.errorbar(gbm_bv,gbm_tv,xerr=gbm_bv_err,yerr=gbm_tv_err,c='blue',lw=0.6)
		#ax2.errorbar(bat_bv,bat_tv,xerr=bat_bv_err,yerr=bat_tv_err,c='red',lw=0.6)
		ax2.vlines(gbm_mean[0]+gbm_sigma[0],gbm_mean[1]-gbm_sigma[1],gbm_mean[1]+gbm_sigma[1],'blue',linestyle='solid',lw=0.6)
		ax2.vlines(gbm_mean[0]-gbm_sigma[0],gbm_mean[1]-gbm_sigma[1],gbm_mean[1]+gbm_sigma[1],'blue',linestyle='solid',lw=0.6)
		ax2.hlines(gbm_mean[1]+gbm_sigma[1],gbm_mean[0]-gbm_sigma[0],gbm_mean[0]+gbm_sigma[0],'blue',linestyle='solid',lw=0.6)
		ax2.hlines(gbm_mean[1]-gbm_sigma[1],gbm_mean[0]-gbm_sigma[0],gbm_mean[0]+gbm_sigma[0],'blue',linestyle='solid',lw=0.6)
		
		ax2.vlines(bat_mean[0]+bat_sigma[0],bat_mean[1]-bat_sigma[1],bat_mean[1]+bat_sigma[1],'red',linestyle='solid',lw=0.6)
		ax2.vlines(bat_mean[0]-bat_sigma[0],bat_mean[1]-bat_sigma[1],bat_mean[1]+bat_sigma[1],'red',linestyle='solid',lw=0.6)
		ax2.hlines(bat_mean[1]+bat_sigma[1],bat_mean[0]-bat_sigma[0],bat_mean[0]+bat_sigma[0],'red',linestyle='solid',lw=0.6)
		ax2.hlines(bat_mean[1]-bat_sigma[1],bat_mean[0]-bat_sigma[0],bat_mean[0]+bat_sigma[0],'red',linestyle='solid',lw=0.6)	
				
		xmajorlocators = MultipleLocator(0.2)
		xminorlocators = MultipleLocator(0.05)			
		ax2.xaxis.set_major_locator(xmajorlocators)
		ax2.xaxis.set_minor_locator(xminorlocators)
		ymajorlocators = MultipleLocator(5)
		yminorlocators = MultipleLocator(1)			
		ax2.yaxis.set_major_locator(ymajorlocators)
		ax2.yaxis.set_minor_locator(yminorlocators)
		ax2.set_xlim(0,1)
		ax2.set_ylim(-1,16)
		ax2.set_xlabel("β'")
		ax2.set_ylabel("τ'(s)")
		#ax1.set_title('GRB 150314A')
		'''
		plt.savefig(self.resultdir+bnname+'_T90.png')
		plt.close()


	def netlc(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/net_lc.png'):
			assert os.path.exists(self.baseresultdir), \
					'Should have run base() before running netlc()!'
			#print('plotting raw_lc_with_base.png ...')
		viewt1=np.max([self.baset1,viewt1])
		viewt2=np.min([self.baset2,viewt2])
		BGOplotymax=0.0
		NaIplotymax=0.0
		# raw lc with baseline
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')
		fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
		for i in range(14):
			cNet=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][2] \
								for ch in np.arange(ch1,ch2+1) ])
			totalNet=np.sum(cNet,axis=0)
			ymax=totalNet.max()
			if i<=1:
				if BGOplotymax<ymax:
					BGOplotymax=ymax
					BGOdetseq=i
			else:
				if NaIplotymax<ymax:
					NaIplotymax=ymax
					NaIdetseq=i
				
						
				
			cRate=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch1,ch2+1) ])
			totalRate=np.sum(cRate,axis=0)
			totalRate=np.concatenate(([totalRate[0]],totalRate))
			axes[i//2,i%2].plot(self.tbins,totalRate,drawstyle='steps',lw=3.0,\
											color='tab:blue')
			cBase=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][1] \
									for ch in np.arange(ch1,ch2+1) ])
			totalBase=np.sum(cBase,axis=0)
			plottime=self.tbins[:-1]+self.binwidth/2.0
			axes[i//2,i%2].plot(plottime,totalBase,linestyle='--',lw=4.0,\
											color='tab:orange')
			axes[i//2,i%2].set_xlim([viewt1,viewt2])
			axes[i//2,i%2].tick_params(labelsize=25)
			axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].\
											transAxes,fontsize=25)
		mostbri_det.append(NaIdetseq)
		fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',\
											 rotation='vertical',fontsize=30)
		fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
		fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)			
		plt.savefig(self.resultdir+'/raw_lc_with_base.png')
		plt.close()

			# net lc
		fig, axes= plt.subplots(7,2,figsize=(32, 20),sharex=True,sharey=False)
		for i in range(14):
			cNet=np.array([f['/'+Det[i]+'/ch'+str(ch)][()][2] \
									for ch in np.arange(ch1,ch2+1) ])
			totalNet=np.sum(cNet,axis=0)
			totalNet=np.concatenate(([totalNet[0]],totalNet))
			axes[i//2,i%2].plot(self.tbins,totalNet,linestyle='-',lw=3.0,\
															color='tab:blue')
			axes[i//2,i%2].set_xlim([viewt1,viewt2])
			if i<=1:
				axes[i//2,i%2].set_ylim([0,BGOplotymax])
			else:
				axes[i//2,i%2].set_ylim([0,NaIplotymax])
			axes[i//2,i%2].tick_params(labelsize=25)
			if i==BGOdetseq:
				axes[i//2,i%2].text(0.7,0.85,'Brightest BGO',transform=\
							axes[i//2,i%2].transAxes,color='red',fontsize=25)
			elif i==NaIdetseq:
				axes[i//2,i%2].text(0.7,0.85,'Brightest NaI',transform=\
							axes[i//2,i%2].transAxes,color='red',fontsize=25)
			axes[i//2,i%2].text(0.05,0.85,Det[i],transform=axes[i//2,i%2].\
														transAxes,fontsize=25)
		fig.text(0.07, 0.5, 'Count rate (count/s)', ha='center', va='center',\
												 rotation='vertical',fontsize=30)
		fig.text(0.5, 0.05, 'Time (s)', ha='center', va='center',fontsize=30)		
		fig.text(0.5, 0.92, self.bnname, ha='center', va='center',fontsize=30)			
		plt.savefig(self.resultdir+'/net_lc.png')
		plt.close()
		f.close()

	def hardness_ratio(self,lcbinwidth=0.1):
		brightest_det=Det[mostbri_det[0]]
		file = glob(self.datadir+'glg_tte_'+brightest_det+'_'+self.bnname+'_v*.fit') 
		#print(file)
		fitfile=file[0]	
		hdu=fits.open(fitfile)
		data=hdu['events'].data['time']
		trigtime=hdu[0].header['trigtime']
		time=data-trigtime
		tte=time[(time>-50)&(time<200)]
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		ax2 = ax1.twinx()
		#原始光变
		edges=np.arange(tte[0],tte[-1]+lcbinwidth,lcbinwidth)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/lcbinwidth
		plotrate=np.concatenate(([plotrate[0]],plotrate))
		ax1.plot(plottime,plotrate,drawstyle='steps',color='black')
		edges = bayesian_blocks(plottime,plotrate,fitness='events',p0=1e-1, gamma=1e-300)
		histvalue, histbin =np.histogram(tte,bins=edges)
		plottime=histbin
		plotrate=histvalue/(histbin[1:]-histbin[:-1])
		plotrate=np.concatenate(([plotrate[0]],plotrate)) 
		#ax1.plot(plottime,plotrate,drawstyle='steps',color='b')
		ax1.set_xlabel('time')
		ax1.set_ylabel('Count')

		#硬度比
		f=h5py.File(self.baseresultdir+'/base.h5',mode='r')		
		cNetlo=np.array([f['/'+Det[mostbri_det[0]]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch1,ch3+1) ])
		cNethi=np.array([f['/'+Det[mostbri_det[0]]+'/ch'+str(ch)][()][0] \
									for ch in np.arange(ch3,ch2+1) ])	
		totalNetlo=np.sum(cNetlo,axis=0)
		totalNethi=np.sum(cNethi,axis=0)
		hardness=totalNethi/totalNetlo
		hardness=np.concatenate(([hardness[0]],hardness))
		ax2.plot(self.tbins,hardness,drawstyle='steps',color='blue')
		ax2.set_xlim(-50,200)
		plt.savefig(self.resultdir+bnname+'hardness_ratio.png')
		plt.close()

	def countmap(self,viewt1=-50,viewt2=300):
		if not os.path.exists(self.resultdir+'/net_countmap.png'):
			assert os.path.exists(self.baseresultdir), \
					'Should have run base() before running countmap()!'
			viewt1=np.max([self.baset1,viewt1])
			viewt2=np.min([self.baset2,viewt2])
			#print('plotting rate_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,\
			#					self.datadir,'rate',self.tbins,viewt1,viewt2)
			#print('plotting base_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,\
			#					self.datadir,'base',self.tbins,viewt1,viewt2)
			#print('plotting net_countmap.png ...')
			plot_countmap(self.bnname,self.resultdir,self.baseresultdir,\
								self.datadir,'net',self.tbins,viewt1,viewt2)
			#print('plotting pois_countmap.png ...')
			#plot_countmap(self.bnname,self.resultdir,self.baseresultdir,\
			#					self.datadir,'pois',self.tbins,viewt1,viewt2)



	def ccf_test1(self,viewt1=-50,viewt2=300,binwidth=0.1):
		
		energy_edges=np.logspace(math.log(15,10),math.log(150,10),11)
		en_l=len(energy_edges)
		
		def get_peak(x,corr,v=7,order = 12,num =10):
			max_index=np.argmax(corr)
			x_max = x[max_index]
			x_p = x[(x>=x_max-v)&(x<=x_max+v)]
			corr_p = corr[(x>=x_max-v)&(x<=x_max+v)]
			fit = np.polyfit(x_p, corr_p,order)
			x_new = np.linspace(x_p[0], x_p[-1], 10000)
			corr_new = np.polyval(fit,x_new)
			max_index = np.argmax(corr_new)
			x_max = x_new[max_index]
			x_array = np.zeros(num)
			for i in range(num):
				x_p = x[(x>=x_max-v)&(x<=x_max+v)]
				corr_p = corr[(x>=x_max-v)&(x<=x_max+v)]
				fit = np.polyfit(x_p, corr_p, order)
				x_new = np.linspace(x_p[0], x_p[-1], 10000)
				corr_new = np.polyval(fit,x_new)
				max_index = np.argmax(corr_new)
				x_max = x_new[max_index]
				x_array[i] = x_max
				#t_peak = x_array.sum()/x_array.size
			return x_max
######						
#根据能道区间生成的光子能量e与所要的区间进行对比决定是否选择该区间
#考虑找到正确的能道区间，得到索引值带入ch里13.750471949577264,
		#gbm
		viewt1=np.max([self.GTI1,viewt1])
		viewt2=np.min([self.GTI2,viewt2])
		tbins=np.arange(viewt1,viewt2+binwidth,binwidth)		
		ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[mostbri_det[0]]+'_'+self.bnname+'_v*.fit')
		
		hdu=fits.open(ttefile[0])
		trigtime=hdu['Primary'].header['TRIGTIME']
		data=hdu['EVENTS'].data
		time=data.field(0)-trigtime
		ch=data.field(1)
		ebound=hdu['EBOUNDS'].data
		emin=ebound.field(1)
		emax=ebound.field(2)
		e=np.random.uniform(emin,emax)
		
		#gbm	
		lag = []
		lag_err_l = []
		lag_err_h = []
		
		for i in range(en_l-2):
			indexs=np.where((e>=energy_edges[i])&(e<=energy_edges[i+1]))
			index0=indexs[0]
			needindex0=(ch>=index0[0]) & (ch<=index0[-1])
			time0=time[needindex0]
			histvalue0, histbin0 =np.histogram(time0,bins=tbins)
			plotrate0=histvalue0
			plotrate0=histvalue0/binwidth
			plotrate0_err = np.sqrt(plotrate0)
		
			time_newbin=(histbin0[1:]+histbin0[:-1])*0.5
			time_index=(time_newbin>=time_slice[0])&(time_newbin<=time_slice[-1])
			#time_index=(time_newbin>=t90_start[0])&(time_newbin<=t90_stop[0])
#			
			'''
			r.assign('rrate',plotrate0) 
			r("y=matrix(rrate,nrow=1)")
			fillPeak_hwi=str(int(5/binwidth))
			fillPeak_int=str(int(len(plotrate0)/10))
			r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi+", it=10,\
						 	int ="+fillPeak_int+", method='fillPeaks')")
			r("bs=getBaseline(rbase)")
			r("cs=getCorrected(rbase)")
			bs0=r('bs')[0]
			cs0=r('cs')[0]
			cs_0=cs0[time_index]
			'''
			time_newbin,bs0,cs0=TD_baseline(time_newbin,plotrate0)
			cs_0=cs0[time_index]
			
			
			index=np.where((e>=energy_edges[i+1])&(e<=energy_edges[i+2]))
			index1=index[0]
			print(index0,index1)
			needindex1=(ch>=index1[0]) & (ch<=index1[-1])
			time1=time[needindex1]
			histvalue1, histbin1 =np.histogram(time1,bins=tbins)
			plotrate1=histvalue1
			plotrate1=histvalue1/binwidth
			plotrate1_err = np.sqrt(plotrate1)

			time_newbin=(histbin1[1:]+histbin1[:-1])*0.5
			time_newbin,bs1,cs1=TD_baseline(time_newbin,plotrate1)
			'''
			r.assign('rrate',plotrate1) 
			r("y=matrix(rrate,nrow=1)")
			fillPeak_hwi=str(int(5/binwidth))
			fillPeak_int=str(int(len(plotrate1)/10))
			r("rbase=baseline(y,lam = 6,hwi="+fillPeak_hwi+", it=10,\
						 int ="+fillPeak_int+", method='fillPeaks')")
			r("bs=getBaseline(rbase)")
			r("cs=getCorrected(rbase)")
			bs1=r('bs')[0]
			cs1=r('cs')[0]
			'''
			cs_1=cs1[time_index]
			#corr =np.correlate(cs_1,cs_0,'full')
			corr =signal.correlate(cs_1,cs_0)
			n=len(cs_1)
			newtime=time_newbin[time_index]
			dt=(newtime[-1] - newtime[0])/len(newtime)
			x =  np.arange(-n+1,n,1)*dt
			max_index=np.argmax(corr)
			lags = get_peak(x,corr,order=12,v=7,num = 10)
			lag.append(lags)
			#ccf拟合图像
			index_array = np.where((x >= lags-7)&(x<=lags+7))
			fit_nccf = corr[index_array]
			fit_lag_t = x[index_array]
			new_t = np.linspace(fit_lag_t[0],fit_lag_t[-1],1000)
			c = np.polyfit(fit_lag_t,fit_nccf,12)
			yy = np.polyval(c,new_t)
			
			plt.plot(fit_lag_t,fit_nccf)
			plt.plot(new_t,yy)
			plt.vlines(lags,0,yy.max(),'red')

 

			plt.savefig(self.resultdir+'gbmccf'+str(i)+'.png')
			plt.close()
		
			#蒙特卡洛误差
			err_list=[]
			for i in range(10):
				rand = np.random.randn(n)
				cs_0_err=plotrate0_err[time_index]
				cs_1_err=plotrate1_err[time_index]
				cs_0_e = cs_0 + cs_0_err*rand
	
	
				cs_1_e = cs_1 + cs_1_err*rand
				#err_corr =signal.correlate(cs_1_e,cs_0_e)
				err_corr =np.correlate(cs_1_e,cs_0_e,'full')
				errlist = get_peak(x,err_corr,order=12,v=7,num = 5)
				err_list.append(errlist)
	
			lagerr_array = np.array(err_list)
			lag_sigma = lagerr_array.std()
			lag_ave = lagerr_array.mean()
			
			dx = lag_ave-lags
			lag_errh = lag_sigma-dx
			lag_errl = lag_sigma+dx
			lag_errh=lag_errh.tolist()
			lag_errl=lag_errl.tolist()
			lag_err_h.append(lag_errh)
			lag_err_l.append(lag_errl)
		
			
		for i in range(len(lag)-1):
			lag[i+1]=lag[i]+lag[i+1]
		
		'''
		for i in range(len(lag)):
			lag[i]=lag[i]*(-1)
		'''
		print('gbm',lag)
		for i in range(len(lag_err_h)-1):
			lag_err_h[i+1]=np.sqrt(lag_err_h[i+1]**2+lag_err_h[i]**2)
			lag_err_l[i+1]=np.sqrt(lag_err_l[i+1]**2+lag_err_l[i]**2)
		lag_err=[lag_err_l,lag_err_h]		
		
		
		#横坐标:能量if not os.path.exists("chains"): os.mkdir("chains")	
		E=[]
		E_err=[]
		
		for i in range(1,en_l-1):
			a=(energy_edges[i]+energy_edges[i+1])/2
			E.append(a)
			b=(energy_edges[i+1]-energy_edges[i])/2
			E_err.append(b)

		print(E)
		#plt.scatter(E,lag,c='blue',s=17)
		#plt.errorbar(E,lag,xerr=E_err,yerr=lag_err,c='blue',fmt='.',linewidth=1.1,capthick=0.6,capsize=1.2)



###		
		'''
		#gbm基础数据	
		indexs=np.where((e>=energy_edges[0])&(e<=energy_edges[1]))
		index0=indexs[0]
		print(index0)
		needindex0=(ch>=index0[0]) & (ch<=index0[-1])
		time0=time[needindex0]
		histvalue0, histbin0 =np.histogram(time0,bins=tbins)
		plotrate0=histvalue0
		plotrate0=histvalue0/binwidth
		plotrate0_err = np.sqrt(plotrate0)
		
		time_newbin=(histbin0[1:]+histbin0[:-1])*0.5
		time_index=(time_newbin>=time_slice[0])&(time_newbin<=time_slice[-1])
		#time_index=(time_newbin>=t90_start[0])&(time_newbin<=t90_stop[0])
#	
		r.assign('rrate',plotrate0) 
		r("y=matrix(rrate,nrow=1)")
		fillPeak_hwi=str(int(5/binwidth))
		fillPeak_int=str(int(len(plotrate0)/10))
		r("rbase=baseline(y,lam = 6, hwi="+fillPeak_hwi+", it=10,\
						 int ="+fillPeak_int+", method='fillPeaks')")
		r("bs=getBaseline(rbase)")
		r("cs=getCorrected(rbase)")
		bs0=r('bs')[0]
		cs0=r('cs')[0]
		cs_0=cs0[time_index]
	
		lag2 = []
		lag_err_l2 = []
		lag_err_h2= []
		
		
		for i in range(1,en_l-1):
			index=np.where((e>=energy_edges[i])&(e<=energy_edges[i+1]))
			index1=index[0]
			print(index1)
			needindex1=(ch>=index1[0]) & (ch<=index1[-1])
			time1=time[needindex1]
			histvalue1, histbin1 =np.histogram(time1,bins=tbins)
			plotrate1=histvalue1
			plotrate1=histvalue1/binwidth
			plotrate1_err = np.sqrt(plotrate1)

			time_newbin=(histbin1[1:]+histbin1[:-1])*0.5
			r.assign('rrate',plotrate1) 
			r("y=matrix(rrate,nrow=1)")
			fillPeak_hwi=str(int(5/binwidth))
			fillPeak_int=str(int(len(plotrate1)/10))
			r("rbase=baseline(y,lam = 6,hwi="+fillPeak_hwi+", it=10,\
						 int ="+fillPeak_int+", method='fillPeaks')")
			r("bs=getBaseline(rbase)")
			r("cs=getCorrected(rbase)")
			bs1=r('bs')[0]
			cs1=r('cs')[0]


			cs_1=cs1[time_index]
			#corr =np.correlate(cs_1,cs_0,'full')
			corr =signal.correlate(cs_1,cs_0)
			n=len(cs_1)
			newtime=time_newbin[time_index]
			dt=(newtime[-1] - newtime[0])/len(newtime)
			x =  np.arange(-n+1,n,1)*dt
			max_index=np.argmax(corr)
			lags = get_peak(x,corr,order=12,v=7,num = 10)
			lag2.append(lags)
			#ccf拟合图像
			index_array = np.where((x >= lags-7)&(x<=lags+7))
			fit_nccf = corr[index_array]
			fit_lag_t = x[index_array]
			new_t = np.linspace(fit_lag_t[0],fit_lag_t[-1],1000)
			c = np.polyfit(fit_lag_t,fit_nccf,12)
			yy = np.polyval(c,new_t)
			
			plt.plot(fit_lag_t,fit_nccf)
			plt.plot(new_t,yy)
			plt.vlines(lags,0,yy.max(),'red')
			plt.savefig(self.resultdir+'gbmccf'+str(i)+'.png')
			plt.close()
		
			#蒙特卡洛误差
			err_list=[]
			for i in range(10):
				rand = np.random.randn(n)
				cs_0_err=plotrate0_err[time_index]
				cs_1_err=plotrate1_err[time_index]
				cs_0_e = cs_0 + cs_0_err*rand
				cs_1_e = cs_1 + cs_1_err*rand
				err_corr =signal.correlate(cs_1_e,cs_0_e)
				#err_corr =np.correlate(cs_1_e,cs_0_e,'full')
				errlist = get_peak(x,err_corr,order=12,v=7,num = 5)
				err_list.append(errlist)
	
			lagerr_array = np.array(err_list)
			lag_sigma = lagerr_array.std()
			lag_ave = lagerr_array.mean()
		
			dx = lag_ave-lags
			lag_errh = lag_sigma-dx
			lag_errl = lag_sigma+dx
			lag_errh=lag_errh.tolist()
			lag_errl=lag_errl.tolist()
			lag_err_h2.append(lag_errh)
			lag_err_l2.append(lag_errl)
			
		lag_err2=[lag_err_l2,lag_err_h2]
		laglim=lag
		for i in range(len(lag)):
			lag2[i]=lag2[i]*(-1)
			
				
		print('gbm2',lag2)
		'''
	
		l=np.min(lag)
		l=l+0.02
		m=np.max(lag)
		m=m-0.2
		plt.figure(figsize=(8,6))
		plt.errorbar(E,lag,xerr=E_err,yerr=lag_err,c='blue',fmt='.',linewidth=1.1,capthick=0.6,capsize=1.2)
		#plt.errorbar(E,lag2,xerr=E_err,yerr=lag_err2,c='red',fmt='.',linewidth=1.1,capthick=0.6,capsize=1.2)
		 	
		plt.xlim(14,157)
		#plt.xscale('log')
		#plt.yscale('log')
		plt.xlabel('E (kev)')
		plt.ylabel('Mock arrival Time (s)')
		plt.title('GRB'+bnname[2::])
		#plt.hlines(0.6,14,157,linestyle='dashed',color='lightgreen')		
		plt.savefig(self.resultdir+bnname+'结果_fit_ccf.png')
		plt.close()
		nnname.append(bnname)
		t_start.append(time_slice[0])
		t_stop.append(time_slice[-1])
	
	
		bn_year.append("20"+bnname[2:4])
		detector.append(Det[mostbri_det[0]])

	def ccf_test(self,viewt1=-50,viewt2=300,binwidth=0.1):
		
		energy_edges=np.logspace(math.log(10,10),math.log(800,10),11)
		en_l=len(energy_edges)
		band=get_band_in_log([10,800],10,ovelap=0)
		def get_nccf_peak_time(lag_t,nccf,precision = 0.001,save = None):
	
			dt = np.abs(lag_t[1]-lag_t[0])
			w = np.ones(len(nccf))
			smoo_nccf = WhittakerSmooth(nccf,w,0.15/dt**1.5)
			intef = interp1d(lag_t,smoo_nccf,kind = 'quadratic')
			new_lat_t = np.arange(lag_t[1],lag_t[-2]+precision,precision)
			new_nccf = intef(new_lat_t)
			index_ = np.argmax(new_nccf)
			t_max = new_lat_t[index_]
			

			if save is not None:
				fig = plt.figure(constrained_layout=True)
				ax = fig.add_subplot(1,1,1)
				ax.plot(lag_t,nccf,'.',color = 'k',label = 'nccf')
				ax.plot(new_lat_t,new_nccf,'-',color='#f47920',label = 'fit')
				ax.axvline(x =t_max,color = 'r' )
				ax.set_xlim(t_max-5,t_max+5)
				ax.set_xlabel('lag time (s)')
				ax.set_ylabel('nccf')
				ax.legend()
				plt.savefig('gbmccf'+str(i)+'.png')
				plt.close(fig)
			
			return t_max
######						
#根据能道区间生成的光子能量e与所要的区间进行对比决定是否选择该区间
#考虑找到正确的能道区间，得到索引值带入ch里1
		#gbm
		viewt1=np.max([self.GTI1,viewt1])
		viewt2=np.min([self.GTI2,viewt2])
		tbins=np.arange(viewt1,viewt2+binwidth,binwidth)		
		ttefile=glob(self.datadir+'/'+'glg_tte_'+Det[mostbri_det[0]]+'_'+self.bnname+'_v*.fit')
		
		hdu=fits.open(ttefile[0])
		trigtime=hdu['Primary'].header['TRIGTIME']
		data=hdu['EVENTS'].data
		time=data.field(0)
		t=data.field(0)-trigtime
		ch=data.field(1)
		ebound=hdu['EBOUNDS'].data
		ch_n = ebound.field(0)
		emin=ebound.field(1)
		emax=ebound.field(2)
		e=np.random.uniform(emin,emax)
		t,energy = ch_to_energy(t,ch,ch_n,emin,emax)
		event_band=[]
		
		for (emin,emax) in band:
			index = np.where((energy>=emin)&(energy<=emax))[0]
			t_i = time[index]
			e_i = energy[index]
			event_band.append([t_i,e_i])
		
			
		binsize=0.2
		binsize = tbins[1]-tbins[0]
		t_c = 0.5*(tbins[1:]+tbins[:-1])
		#gbm	
		cs0 = None
		cs_err0 = None
		lag_errh2 = 0
		lag_errl2 = 0
		lag_all = 0
		return_list = []
		lc_list = []
		
		for index_,(t_i,e_i) in enumerate(event_band):
			if cs0 is None or cs_err0 is None:
		
				num_i = np.histogram(t_i,bins=tbins)[0]
				num_err_i = np.sqrt(num_i)
				rate_i = num_i/binsize
				rate_err_i = num_err_i/binsize
				csi,bsi,sigma_i = TD_bs(t_c,rate_i,sigma=True)
				lc_list.append([rate_i,bsi,sigma_i])
				SNRi = csi/sigma_i
				if SNRi.max() > sigma:
					cs0 = csi
					cs_err0 = rate_err_i
					return_list.append([index_,0,1,1])
			else:
				num_i = np.histogram(t_i,bins=tbins)[0]
				num_err_i = np.sqrt(num_i)
				rate_i = num_i/binsize
				rate_err_i = num_err_i/binsize
				csi,bsi,sigma_i = TD_bs(t_c,rate_i,sigma=True)
				lc_list.append([rate_i,bsi,sigma_i])
				SNRi = csi/sigma_i
				if SNRi.max() > sigma:
					lag,lag_errl,lag_errh = get_one_lag(cs0[wind_index],csi[wind_index],cs_err0[wind_index],rate_err_i[wind_index],t_c[wind_index],mcmc_num=1000,save = savename)	
							
					lag_all = lag_all + lag*-1
					lag_errl2 = lag_errl2+lag_errl**2
					lag_errh2 = lag_errh2 + lag_errh**2
					return_list.append([index_,lag_all,np.sqrt(lag_errl2),np.sqrt(lag_errh2)])
					cs0 = csi
					cs_err0 = rate_err_i		
		
		print(return_list)
		
		
		
		
		
		
		'''
		
		for i in range(en_l-2):
			indexs=np.where((e>=energy_edges[i])&(e<=energy_edges[i+1]))
			index0=indexs[0]
			needindex0=(ch>=index0[0]) & (ch<=index0[-1])
			time0=time[needindex0]
			histvalue0, histbin0 =np.histogram(time0,bins=tbins)
			plotrate0=histvalue0
			plotrate0=histvalue0/binwidth
			plotrate0_err = np.sqrt(plotrate0)
		
			time_newbin=(histbin0[1:]+histbin0[:-1])*0.5
			time_index=(time_newbin>=time_slice[0])&(time_newbin<=time_slice[-1])
			#time_index=(time_newbin>=t90_start[0])&(time_newbin<=t90_stop[0])
#			
			time_newbin,bs0,cs0=TD_baseline(time_newbin,plotrate0)
			cs_0=cs0[time_index]
			
			
			index=np.where((e>=energy_edges[i+1])&(e<=energy_edges[i+2]))
			index1=index[0]
			print(index0,index1)
			needindex1=(ch>=index1[0]) & (ch<=index1[-1])
			time1=time[needindex1]
			histvalue1, histbin1 =np.histogram(time1,bins=tbins)
			plotrate1=histvalue1
			plotrate1=histvalue1/binwidth
			plotrate1_err = np.sqrt(plotrate1)

			time_newbin=(histbin1[1:]+histbin1[:-1])*0.5
			time_newbin,bs1,cs1=TD_baseline(time_newbin,plotrate1)

			cs_1=cs1[time_index]
			#corr =np.correlate(cs_1,cs_0,'full')
			corr = np.correlate(cs_1 / cs_1.max(), cs_0 / cs_0.max(), 'full')
			n=len(cs_1)
			newtime=time_newbin[time_index]
			dt=(newtime[-1] - newtime[0])/len(newtime)
			x =  np.arange(-n+1,n,1)*dt
			max_index=np.argmax(corr)
			lags = get_nccf_peak_time(x,corr,save=self.resultdir+'A_check/')
			lag.append(lags)
			
			
			#蒙特卡洛误差
			cs_0_err=plotrate0_err[time_index]
			cs_1_err=plotrate1_err[time_index]
			
			err_list=[]
			for i in range(1000):
				rand0 = np.random.randn(n)
				cs_0_e = cs_0 + cs_0_err*rand0
				rand1 = np.random.randn(n)
				cs_1_e = cs_1 + cs_1_err*rand1
				#err_corr =signal.correlate(cs_1_e,cs_0_e)
				err_corr =np.correlate(cs_1_e/cs_1_e.max(),cs_0_e/cs_0_e.max(),'full')
				errlist = get_nccf_peak_time(x,err_corr)
				err_list.append(errlist)
	
			lagerr_array = np.array(err_list)
			lag_sigma = lagerr_array.std(ddof = 1)
			lag_mean = lagerr_array.mean()
			
			dx = lag_mean-lags
			lag_errl = lag_sigma-dx
			lag_errh = lag_sigma+dx
			lag_errl=lag_errl.tolist()
			lag_errh=lag_errh.tolist()
			lag_err_l.append(lag_errl)
			lag_err_h.append(lag_errh)
		
			
		for i in range(len(lag)-1):
			lag[i+1]=lag[i]+lag[i+1]
		
	
		print('gbm',lag)
		
		for i in range(len(lag_err_h)-1):
			lag_err_h[i+1]=np.sqrt(lag_err_h[i+1]**2+lag_err_h[i]**2)
			lag_err_l[i+1]=np.sqrt(lag_err_l[i+1]**2+lag_err_l[i]**2)
		lag_err=[lag_err_l,lag_err_h]		
		
		
		#横坐标:能量if not os.path.exists("chains"): os.mkdir("chains")	
		E=[]
		E_err=[]
		
		for i in range(en_l-1):
			a=(energy_edges[i]+energy_edges[i+1])/2
			E.append(a)
			b=(energy_edges[i+1]-energy_edges[i])/2
			E_err.append(b)

		print(E)
		
		E_arr=np.array(E)
		l=np.min(lag)
		l=l+0.02
		m=np.max(lag)
		m=m-0.2
		plt.figure(figsize=(8,6))
		plt.errorbar(E,lag,xerr=E_err,yerr=lag_err,c='blue',fmt='.',linewidth=1.1,capthick=0.6,capsize=1.2)	
		plt.xlim(E_arr.min()-3,E_arr.max()+3)
		#plt.xscale('log')
		#plt.yscale('log')
		plt.xlabel('E (kev)')
		plt.ylabel('Mock arrival Time (s)')
		plt.title('GRB'+bnname[2::])	
		plt.savefig(self.resultdir+bnname+'结果_fit_ccf.png')
		plt.close()
		nnname.append(bnname)
		t_start.append(time_slice[0])
		t_stop.append(time_slice[-1])
		bn_year.append("20"+bnname[2:4])
		detector.append(Det[mostbri_det[0]])
		'''


		
for n in range(1,nl):

	os.chdir('/home/yao/Work/NEW_HR_TEST/样本挑选')    
	bnname=name[n]
	print(bnname)
	number=trigger_name.tolist().index(bnname)
	grb=GRB(bnname)
	grb.base(baset1=-50,baset2=200,binwidth=0.1)
	'''
	t90_start=[]
	t90_stop=[]
	a=float(t90_start_str[number])
	b=float(t90_str[number])+float(t90_start_str[number])
	print(a,b)
	t90_start.append(a)
	t90_stop.append(b)
	t90_start_time.append(a)
	t90_stop_time.append(b)
	Epeak=Flnc_Band_Epeak_str[number]
	mask=scat_detector_mask_str[number]
	det1=['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1']
	mask = [m.start() for m in re.finditer('1', scat_detector_mask_str[number])]
	l=len(mask)
	print(mask)
	'''
	

	grb.rawlc(viewt1=-50,viewt2=300,binwidth=0.1)
	
	
	#grb.bbhardness_ratio(lcbinwidth=0.1)
	#z=len(time_slice)
	grb.netlc(viewt1=-50,viewt2=300)
	print(mostbri_det)
	#grb.hardness_ratio(lcbinwidth=0.1)
	grb.timeslice(lcbinwidth=0.1,gamma=1e-300)
	#grb.BAT_timeslice(gamma=1e-300)
	
	#for i in range(z-1):
	#	grb.phaI(slicet1=time_slice[i],slicet2=time_slice[i+1])        
	#	grb.specanalyze('slice'+str(i))
	
	#print('epeak',epeak)ax1 = fig.add_subplot(111)
	#grb.gbm_ch_netlc(lcbinwidth=0.1)
	#grb.bat_ch_netlc(lcbinwidth=1)
	#grb.netlc_ccf_test(viewt1=-50,viewt2=200,binwidth=0.1)	
	
	grb.ccf_test(viewt1=-50,viewt2=200,binwidth=0.2)
	grb.countmap(viewt1=-50,viewt2=200)
	'''
	epeak=[]
	epeak_error_p=[]
	epeak_error_n=[]
	mostbri_det=[]
	time_slice=[]
	t90_start=[]
	t90_stop=[]
	'''

	
with open("ideat10没有t90.txt","w") as f:
	for i in range(len(nnname)):
		f.write(nnname[i]+"\t") 
		f.write(str(t_start[i])+"\t")
		f.write(str(t_stop[i])+"\t")
		f.write(str(bn_year[i])+"\t")
		f.write(str(detector[i])+"\n")
	
'''
t90_time=[]
real_time=[]
for ii in range(len(nnname)):
	real_time.append(t_stop[ii]-t_start[ii])
	t90_time.append(t90_stop_time[ii]-t90_start_time[ii])
xxx=np.arange(1,400)


plt.plot(xxx,xxx,color='black',linestyle='dashed')
plt.scatter(t90_time,real_time,color='blue',s=16)
plt.xlabel('T90')
plt.ylabel('t')
plt.savefig('对比结果.png')
plt.close()
'''

