

import matplotlib.pyplot as plt
import numpy as np


class Plot(object):

	def __init__(self,result):

		self.result = result
	
	def plot_light_curve(self,sigma=3,**k):
		if self.result['good']:
			t = self.result['t_c']
			rate = self.result['rate']
			sigma_ = self.result['sigma']
			bs = self.result['bs']
			plt.plot(t,rate,color = 'b',label = 'light curve',**k)
			plt.plot(t,bs,color = 'r',label = 'background',**k)
			plt.plot(t,bs+sigma*sigma_,color = 'y',label = r'%s $\sigma$'%sigma,**k)
	
			try:
				by_edges_list = self.result['bayesian_edges']
				by_rate_list = self.result['bayesian_rate']
				for index in range(len(by_edges_list)):
	
					plt.step(by_edges_list[index],by_rate_list[index],color = 'k',**k)
				plt.plot(0,0,color = 'k',label = 'bayesian blocks')
			except:
				print('have none bayesian blocks')
				pass
			plt.legend()
			plt.xlim([t[0],t[-1]])
		else:
			print('result is not good!')

	def plot_Txx1(self,txx,**k):

		if self.result['good']:

			t = self.result['t']
			dt = t[1] - t[0]
			n = self.result['n']/dt
			le = len(self.result['t1'])

			plt.plot(t,n,color = 'b',**k)
			plt.plot(t,self.result['bs1']/dt,color = 'r',**k)



			for i in self.result['bs_list']:
				plt.plot(t,i/dt,color = 'r',alpha = 0.01)
			for index in range(le):
				t1 = '%.2f' % self.result['t1'][index]
				if np.round(self.result['t1_err'][1][index]*100) == 0:
					t1_err1 = '%.3f' % self.result['t1_err'][1][index]
				else:
					t1_err1 = '%.2f' % self.result['t1_err'][1][index]
				if np.round(self.result['t1_err'][0][index]*100) == 0:
					t1_err2 = '%.3f' % self.result['t1_err'][0][index]
				else:
					t1_err2 = '%.2f' % self.result['t1_err'][0][index]
					
				t1_label = r'${T^{'+str(index+1)+'}_{'+txx+',1} = '+t1+' ^ {+ '+t1_err1+'}_{-'+t1_err2+'}}$ s '
				t2 = '%.2f' % self.result['t2'][index]
				if np.round(self.result['t2_err'][1][index]*100) == 0:
					t2_err1 = '%.3f' % self.result['t2_err'][1][index]
				else:
					t2_err1 = '%.2f' % self.result['t2_err'][1][index]
				if np.round(self.result['t2_err'][0][index]*100) == 0:
					t2_err2 = '%.3f' % self.result['t2_err'][0][index]
				else:
					t2_err2 = '%.2f' % self.result['t2_err'][0][index]
				t2_label = r'${T^{'+str(index+1)+'}_{'+txx+',2} = '+t2+' ^ {+ '+t2_err1+'}_{-'+t2_err2+'}}$ s  '
				
				plt.axvline(x = self.result['t1'][index],color = 'g',linestyle = '--')
				plt.axvline(x = self.result['t2'][index],color = 'g',linestyle = '--')
				plt.plot(0,0,',',color = 'g',label = t1_label+t2_label)

			plt.xlim([t[0],t[-1]])
			plt.xlabel('time (s)',**k)
			plt.ylabel('rate',**k)
			plt.legend()
		else:
			print('T'+txx+' is not good!')

	def plot_Txx2(self,txx,**k):

		if self.result['good']:
			le = len(self.result['t1'])
			t = self.result['t']
			cs_f = self.result['cs_f']
			cs_f_max = self.result['cs_f_max']
			l = self.result['l']
			for i in cs_f_max:
				plt.axhline(y = i,color = 'b')
			for i in l[0]:
				plt.axhline(y = i,color = 'b',linestyle = '--')
			for i in l[1]:
				plt.axhline(y = i,color = 'b',linestyle = '--')
			plt.plot(t,cs_f,color = 'k')
			for index in range(le):
				plt.axvline(x = self.result['t1'][index],color = 'g',linestyle = '--')
				plt.axvline(x = self.result['t2'][index],color = 'g',linestyle = '--')
				txx = '%.2f' % self.result['txx'][index]
				if np.round(self.result['txx_err'][1][index]*100) == 0:
					txx_err1 = '%.3f' % self.result['txx_err'][1][index]
				else:
					txx_err1 = '%.2f' % self.result['txx_err'][1][index]
				if np.round(self.result['txx_err'][0][index]*100) == 0:
					txx_err2 = '%.3f' % self.result['txx_err'][0][index]
				else:
					txx_err2 = '%.2f' % self.result['txx_err'][0][index]
				label1 = r'${T^{'+str(index+1)+'}_{'+txx+'} = '+txx+'^{+ '+txx_err1+'}_{-'+txx_err2+'}}$ s '
				plt.plot(0,0,',',label = label1)
			plt.xlabel('time (s)',**k)
			plt.ylabel('Accumulated counts',**k)
			plt.xlim([t[0],t[-1]])
			plt.legend()
		else:
			print('T'+txx+' is not good!')

	def plot_distribution(self,txx,num = 0,**k):

		if self.result['good']:
			txx_list = self.result['txx_list'][num]
			t1_list = self.result['t1_list'][num]
			t2_list = self.result['t2_list'][num]
			t90_list_sort = np.sort(txx_list)
			t90_err = np.std(txx_list)
			t90_two = np.percentile(txx_list,[40,60])

			t90_binsize = t90_two[-1]-t90_two[0]
			if t90_binsize >0:
				t90_bin = np.arange(t90_list_sort[0], t90_list_sort[-1]+t90_binsize,t90_binsize)
			else:
				t90_bin = np.linspace(t90_list_sort[0], t90_list_sort[-1],20)
			#t90_bin = np.linspace(t90_list_sort[0], t90_list_sort[-1], 100)
			t90_n, t90_edges = np.histogram(txx_list, bins=t90_bin)
			t90_n = np.concatenate((t90_n[:1], t90_n))
			plt.subplot(1,3,1)
			plt.title('T'+txx+' distribution',**k)
			plt.step(t90_edges, t90_n, color='k')
			plt.axvline(x=txx_list.mean(), color='r')
			plt.axvline(x=txx_list.mean() - t90_err, color='r')
			plt.axvline(x=txx_list.mean() + t90_err, color='r')
			plt.axvline(x=self.result['txx'][num], color='g')
			plt.xlabel('T'+txx+' (s)',**k)

			t1_list_sort = np.sort(t1_list)
			t90_two = np.percentile(t1_list_sort, [40, 60])
			t90_binsize = t90_two[-1] - t90_two[0]
			if t90_binsize > 0:
				t90_bin = np.arange(t1_list_sort[0], t1_list_sort[-1] + t90_binsize, t90_binsize)
			else:
				t90_bin = np.linspace(t1_list_sort[0], t1_list_sort[-1],20)
			#t90_bin = np.linspace(t1_list_sort[0], t1_list_sort[-1], 100)
			t90_n, t90_edges = np.histogram(t1_list, bins=t90_bin)
			t90_n = np.concatenate((t90_n[:1], t90_n))
			t1_err = np.std(t1_list)

			plt.subplot(1,3,2)
			plt.title('T'+txx+'1 distribution',**k)
			plt.step(t90_edges, t90_n, color='k')
			plt.axvline(x=t1_list.mean(), color='r')
			plt.axvline(x=t1_list.mean() - t1_err, color='r')
			plt.axvline(x=t1_list.mean() + t1_err, color='r')
			plt.axvline(x=self.result['t1'][num], color='g')
			plt.xlabel('T'+txx+'1 (s)')

			t2_list_sort = np.sort(t2_list)
			t90_two = np.percentile(t2_list, [40, 60])
			t90_binsize = t90_two[-1] - t90_two[0]
			if t90_binsize > 0:
				t90_bin = np.arange(t2_list_sort[0], t2_list_sort[-1] + t90_binsize, t90_binsize)
			else:
				t90_bin = np.linspace(t2_list_sort[0], t2_list_sort[-1], 20)

			t90_n, t90_edges = np.histogram(t2_list, bins=t90_bin)
			t90_n = np.concatenate((t90_n[:1], t90_n))
			t2_err = np.std(t2_list)

			plt.subplot(1, 3, 3)
			plt.title('T'+txx+'2 distribution')
			plt.step(t90_edges, t90_n, color='k')
			plt.axvline(x=t2_list.mean(), color='r')
			plt.axvline(x=t2_list.mean() - t2_err, color='r')
			plt.axvline(x=t2_list.mean() + t2_err, color='r')
			plt.axvline(x=self.result['t2'][num], color='g')
			plt.xlabel('T'+txx+'2 (s)')

		else:
			print('T' + txx + ' is not good!')

	


























