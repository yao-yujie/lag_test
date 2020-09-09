
import numpy as np
#from spherical_geometry.polygon import SphericalPolygon
from astropy.coordinates import cartesian_to_spherical,SkyCoord,spherical_to_cartesian
from scipy.interpolate import interp1d
import  astropy.units as u

class Detectors(object):
	def __init__(self,local_az=None,local_zen = None,local_vector =None,
	             name_list = None,color_list = None,effective_angle = None):
		
		if (local_az is not None) and (local_zen is not None) and (local_vector is None) :
			self.vector = np.array(spherical_to_cartesian(1,local_az,local_zen)).T
			
		elif ((local_az is None) or (local_zen is None)) and (local_vector is not None):
			self.vector = np.array(local_vector)
		else:
			self.vector = np.array(
				[[0.2446677589, 0.2523893824, 0.9361823057],
				 [0.5017318971, 0.5036621127, 0.7032706462],
				 [0.5233876659, 0.8520868147, -0.0036651682],
				 [0.5009495177, -0.5032279093, 0.7041386753],
				 [0.5468267487, -0.8372325378, -0.0047123847],
				 [0.9982910766, 0.0584352143, 0.0005236008],
				 [-0.2471260191, -0.2465229020, 0.9370993606],
				 [-0.5135631636, -0.5067957667, 0.6923950822],
				 [-0.5503349679, -0.8349438131, 0.0005235846],
				 [-0.5064476761, 0.5030998708, 0.7002865795],
				 [-0.5552650628, 0.8316411478, -0.0073303046],
				 [-0.9978547710, -0.0652279514, -0.0055850266],
				 [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
			
		if ((local_az is None) or (local_zen is None)) and (local_vector is None) and (name_list is None):
			self.name_list = np.array(['n0','n1','n2','n3','n4','n5','n6','n7','n8','n9','na','nb','b0','b1'])
		else:
			if name_list is None :
				self.name_list = np.arange(len(self.vector))
			else:
				if len(name_list) == len(self.vector):
					self.name_list = name_list
				else:
					self.name_list = np.arange(len(self.vector))
					
		if ((local_az is None) or (local_zen is None)) and (local_vector is None) and (color_list is None):
			self.color_list = ['#74787c']*len(self.vector)
		else:
			if color_list is None :
				self.color_list = np.arange(len(self.vector))
			else:
				if len(color_list) == len(self.vector):
					self.color_list = color_list
				else:
					self.color_list = ['#74787c']*len(self.vector)
		if effective_angle is not None:
			self.effective_angle = effective_angle*u.deg
		else:
			self.effective_angle = 60*u.deg
		
	def get_angle_overlap(self):
		position_array = self.vector.T
		position = cartesian_to_spherical(position_array[0],position_array[1],position_array[2])
		ra = position[2].deg
		dec = position[1].deg
		center_all = SkyCoord(ra = ra,dec = dec,frame = 'icrs',unit = 'deg')
		c = {}
		for index,di in enumerate(self.name_list):
			centeri = center_all[index]
			seq = center_all.separation(centeri)
			dindx = (seq>0*u.deg)&(seq<=self.effective_angle)
			c[di] = self.name_list[dindx]
		return c
	def input_quaternion(self,quaternion,time=None):
		self.quaternion = quaternion
		self.time = time
		self.mat_list = []
		for quaternion_i in self.quaternion:
			self.mat_list.append(self.get_mat(quaternion_i[0], quaternion_i[1], quaternion_i[2], quaternion_i[3]))
		self.center_function,self.detector_index, self.center_all = self.get_center()

	def get_mat(self,p1,p2,p3,p0):
		mat = np.mat(np.zeros((3, 3)))
		mat[0, 0] = p0 ** 2 + p1 ** 2 - p2 ** 2 - p3 ** 2
		mat[0, 1] = 2 * (p1 * p2 - p0 * p3)
		mat[0, 2] = 2 * (p0 * p2 + p1 * p3)
		mat[1, 0] = 2 * (p3 * p0 + p2 * p1)
		mat[1, 1] = p0 ** 2 + p2 ** 2 - p3 ** 2 - p1 ** 2
		mat[1, 2] = 2 * (p2 * p3 - p1 * p0)
		mat[2, 0] = 2 * (p1 * p3 - p0 * p2)
		mat[2, 1] = 2 * (p0 * p1 + p3 * p2)
		mat[2, 2] = p0 ** 2 + p3 ** 2 - p1 ** 2 - p2 ** 2
		return mat
	def get_center(self):
		
		deter_f = {}
		ra_list = []
		dec_list = []
		indexlist = []
		for index, vector in enumerate(self.vector):
			indexlist.append(index)
			position_list = []
			detector_index_list = []
			
			for mat in self.mat_list:
				X = np.mat(vector).T
				X1 = np.array(mat * X).T[0]
				position_list.append([X1[0],X1[1],X1[2]])
				detector_index_list.append(index)
			position_array = np.array(position_list).T
			position = cartesian_to_spherical(position_array[0],position_array[1],position_array[2])
			ra = position[2].deg
			dec = position[1].deg
			ra_list.append(list(ra))
			dec_list.append(list(dec))
			if self.time is not None and len(self.time)>1:
				x_f = interp1d(self.time,position_array[0],kind = 'quadratic')
				y_f = interp1d(self.time,position_array[1],kind = 'quadratic')
				z_f = interp1d(self.time,position_array[2],kind = 'quadratic')
				#ra_f = interp1d(self.time, ra, kind='slinear')
				#dec_f = interp1d(self.time, dec, kind='slinear')
				# ra_f = interp1d(self.time,ra,kind = 'cubic')
				# dec_f = interp1d(self.time,dec,kind = 'cubic')
				deter_f[self.name_list[index]] = [x_f,y_f,z_f]
		ra_list = np.array(ra_list).T
		dec_list = np.array(dec_list).T
		center_all = SkyCoord(ra = ra_list,dec = dec_list,frame = 'icrs',unit = 'deg')
		if self.time is None or len(self.time)<=1:
			deter_f = None
		return deter_f,[indexlist]*len(self.mat_list),center_all
	
	'''
	def get_center_old(self):
		#print('get_center')
		detector_index_list1 = []
		center_all_list = []
		for mat in self.mat_list:
			position_list = []
			detector_index_list = []
			for index,vector in enumerate(self.vector):
				X = np.mat(vector).T
				X1 = np.array(mat * X).T[0]
				position_list.append([X1[0],X1[1],X1[2]])
				detector_index_list.append(index)
			position_array = np.array(position_list).T
			detector_index = np.array(detector_index_list)
			detector_index_list1.append(detector_index)
			position = cartesian_to_spherical(position_array[0],position_array[1],position_array[2])
			p_lon = position[2].deg
			p_lat = position[1].deg
			center_all = SkyCoord(ra = p_lon,dec = p_lat,frame = 'icrs',unit = 'deg')
			center_all_list.append(center_all)
		return detector_index_list1,center_all_list
	
	
	def get_fov(self,radius):
		if radius >= 60:
			steps = 500
		elif(radius >=30):
			steps = 300
		else:
			steps = 250
		j2000 = self.center_all.icrs
		c = []
		for index,center in enumerate(j2000):
			print(str(index))
			poly = SphericalPolygon.from_cone(center.ra.value,center.dec.value,radius,steps=steps)
			re = [p for p in poly.to_radec()][0]
			c.append(re)
		return c
	
	'''