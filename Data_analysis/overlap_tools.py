
import numpy as np


def overlap_bins(range_ ,binsize = 1.0 ,stepsize = 0.5):
	
	start = np.arange(range_[0] ,range_[-1]-binsize ,stepsize)
	stop = start + binsize
	
	return np.vstack([start ,stop]).T

def overlap_curve(t ,bins_arr):
	
	n = np.zeros(bins_arr.shape[0])
	t = np.array(t)
	for index,edges in enumerate(bins_arr):
		nn_ = np.where((t>=edges[0])&(t<edges[-1]))[0]
		n[index] = nn_.size
	return n
	
