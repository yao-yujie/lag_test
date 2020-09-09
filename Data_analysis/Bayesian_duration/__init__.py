
from .Bayesian_duration import *
from .Plot import *
import pandas as pd


def save_result(result,savename,float_format = '%.3f'):
	'''
	
	:param result: the result from get_txx()
	:param savename:
	:return:
	'''
	if result['good']:
		txx = result['txx']
		xx = result['xx']
		txx_err1 ,txx_err2 = result['txx_err']
		t1 = result['t1']
		t1_err1,t1_err2 = result['t1_err']
		t2 = result['t2']
		t2_err1,t2_err2 = result['t2_err']
		news = {'t'+xx:txx,"t"+xx+"_err-":txx_err1,"t"+xx+"_err+":txx_err2,'t'+xx+'_1':t1,"t"+xx+"_1_err-":t1_err1,
		        "t"+xx+"_1_err+":t1_err2,'t'+xx+'_2':t2,"t"+xx+"_2_err-":t2_err1,"t"+xx+"_2_err+":t2_err2}
		df = pd.DataFrame(news,columns=['t90',"t90_err-","t90_err+",'t90_1',"t90_1_err-", "t90_1_err+",'t90_2',"t90_2_err-","t90_2_err+"])
		df.to_csv(savename,index=False,float_format=float_format)
	else:
		print('result is not good!')




