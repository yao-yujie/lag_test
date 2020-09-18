import csv
import matplotlib.pyplot as plt
import numpy as np
ty=[]
ty_errl=[]
ty_errh=[]
t_sce=[]
t90=[]
with open('B_lag_para.csv', 'r') as f:
	reader = csv.reader(f)
	for i in reader:
		ty.append(i[1])
		ty_errl.append(i[2])
		ty_errh.append(i[3])		
		t_sce.append(i[7])
		t90.append(i[8])
print(type(ty[0]))
for i in range(1,len(ty)):
	ty[i]=float(ty[i])
	t_sce[i]=float(t_sce[i])
	t90[i]=float(t90[i])
	ty_errl[i]=float(ty_errl[i])
	ty_errh[i]=float(ty_errh[i])
	#ty[i]=10**ty[i]
	#ty_errl[i]=10**ty_errl[i]
	#ty_errh[i]=10**ty_errh[i]
	

x=np.arange(0.005,130,0.1)
y=x
print(len(x))
plt.figure(figsize=(5,5))
plt.plot(x,y,linestyle='dashed',color='black')
plt.xscale('log')
plt.yscale('log')
plt.scatter(t_sce[1:],ty[1:],s=4)
plt.xlim(0.08,130)
plt.ylim(0.08,130)
plt.xlabel('ccf duration (s)')
plt.ylabel('Spectral lag τ (s)')
plt.vlines(2,0.08,130,linestyle='dashed',color='lightgreen')
plt.savefig('ccf.png')
plt.close()


plt.figure(figsize=(5,5))
plt.plot(x,y,linestyle='dashed',color='black')
plt.xscale('log')
#plt.yscale('log')
plt.scatter(t90[1:],ty[1:],s=4)
plt.errorbar(t90[1:],ty[1:],yerr=[ty_errl[1:],ty_errh[1:]],fmt='.',elinewidth=1)
plt.xlim(0.05,130)
#plt.ylim(0.05,130)
plt.xlabel('duration duration (s)')
plt.ylabel('Spectral lag τ (s)')
plt.vlines(2,0.05,130,linestyle='dashed',color='lightgreen')
plt.savefig('t90.png')
plt.close()


plt.figure(figsize=(5,5))
plt.plot(x,y,linestyle='dashed',color='black')
plt.xscale('log')
plt.yscale('log')
plt.scatter(t90[1:],t_sce[1:],s=4)
plt.xlim(0.05,130)
plt.ylim(0.05,130)
plt.xlabel('duration duration (s)')
plt.ylabel('ccf duration (s)')
plt.savefig('对比.png')
plt.close()










