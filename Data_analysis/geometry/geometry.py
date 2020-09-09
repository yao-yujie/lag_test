import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import cartesian_to_spherical,SkyCoord,get_sun,get_body_barycentric
from ..Time_transform import Time_transform
from .detectors import Detectors
from spherical_geometry.polygon import SphericalPolygon
from astropy.table import Table
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from matplotlib.patches import Polygon
import pandas as pd
import cartopy.crs as ccrs
from scipy.interpolate import interp1d
#import matplotlib.patches as mpatches
#from mpl_toolkits.basemap import Basemap

class Geometry(object):
	
	def __init__(self,detector=None,time_base = None):
		'''
		
		:param detector:
		:param time_base:
		'''
		if time_base is None:
			self.Time_transition = Time_transform()
		else:
			self.Time_transition = time_base
		if detector is None:
			self.detectors = Detectors()
		else:
			self.detectors = detector
	
		
	def input_pose(self,quaternion,sc_pos = None,pos_unit = None,time = None,ef_radius = 60):
		'''
		
		:param quaternion:[[q1,q2,q3,q0],[q1,q2,q3,q0]] or pd.DataFrame
		:param sc_pos:[[x,y,z],[x,y,z]]
		:param time:
		:param ef_radius:
		:return:
		'''
		
		if isinstance(quaternion,pd.DataFrame):
			self.quaternion = quaternion[['QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4']].values
			try:
				self.met_time = quaternion['SCLK_UTC'].values
				self.time = self.Time_transition.batch_met_to_utc(self.met_time)
				self.time_band = [np.min(self.met_time),np.max(self.met_time)]
			except:
				self.met_time = None
				self.time_band = None
				self.time = None
			try:
				if pos_unit is not None:
					self.pos_unit = pos_unit
					self.sc_pos = quaternion[['POS_X','POS_Y','POS_Z']].values * pos_unit
				else:
					self.pos_unit = u.m
					self.sc_pos = quaternion[['POS_X','POS_Y','POS_Z']].values * u.m
			except:
				self.sc_pos = None
		else:
			self.quaternion = quaternion
			if time is not None:
				if (isinstance(time, str)):
					self.met_time = self.Time_transition.utc_to_met(time)
					self.time = Time(time)
					self.time_band = None
				else:
					self.met_time = time
					self.time_band = [np.min(self.met_time),np.max(self.met_time)]
					self.time = self.Time_transition.batch_met_to_utc(time)
			else:
				self.met_time = None
				self.time = None
				self.time_band = None
			self.sc_pos = sc_pos
			if pos_unit is not None:
				self.pos_unit = pos_unit
				self.sc_pos = self.sc_pos * pos_unit
			else:
				self.pos_unit = u.m
		if self.sc_pos is not None and self.met_time is not None:
			if len(self.met_time)>1 and len(self.sc_pos[:,0])>1:
				
				x = self.sc_pos[:,0]
				y = self.sc_pos[:,1]
				z = self.sc_pos[:,2]
				print('me_time\n',self.met_time)
				print('x\n',x)
				x_f = interp1d(self.met_time,-x,kind = 'quadratic')
				y_f = interp1d(self.met_time,-y,kind = 'quadratic')
				z_f = interp1d(self.met_time,-z,kind = 'quadratic')
				self.sc_pos_f = [x_f,y_f,z_f]
			else:
				self.sc_pos_f = None
		else:
			self.sc_pos_f = None
			
		self.index = np.arange(len(self.quaternion),dtype = int)
		self.radius = ef_radius
		self.detectors.input_quaternion(self.quaternion,self.met_time)
		
		#self.all_sky = self.all_sky(num_points = 2000)
	def get_detector_centers_with_time(self,t):
		deter_name = self.detectors.name_list
		center_f = self.detectors.center_function
		if center_f is not None and self.time_band is not None:
			ra_t_all = []
			dec_t_all = []
			index_all = []
			try:
				
				n_=len(t)
				t = np.array(t)
				tband = t.max()-t.min()
				tband_sl = self.time_band[1] - self.time_band[0]
				if tband<=tband_sl+2:
					t[t<=self.time_band[0]] = self.time_band[0]+0.00001
					t[t>=self.time_band[1]] = self.time_band[1]-0.00001
				else:
					t[t<=self.time_band[0]] = np.nan
					t[t>=self.time_band[1]] = np.nan
				
				for index_,deteri in enumerate(deter_name):
					index_all.append(index_)
					xf,yf,zf = center_f[deteri]
					x = xf(t)
					y = yf(t)
					z = zf(t)
					position = cartesian_to_spherical(x,y,z)
					ra_t = position[2].deg
					dec_t = position[1].deg
					ra_t_all.append(list(ra_t))
					dec_t_all.append(list(dec_t))
				ra_t_all = np.array(ra_t_all).T
				dec_t_all = np.array(dec_t_all).T
				center = SkyCoord(ra = ra_t_all,dec = dec_t_all,frame='icrs', unit='deg')
				return center,[index_all]*n_
			except (TypeError):
				
				if (t<=self.time_band[0]):
					if (self.time_band[0]-t<=1):
						t = self.time_band[0]+0.00001
					else:
						t = np.nan
				if t>=self.time_band[1]:
					if (t-self.time_band[0]<=1):
						t = self.time_band[1]-0.00001
					else:
						t = np.nan
				for index_,deteri in enumerate(deter_name):
					index_all.append(index_)
					xf,yf,zf = center_f[deteri]
					x = xf(t)
					y = yf(t)
					z = zf(t)
					position = cartesian_to_spherical(x,y,z)
					ra_t = position[2].deg
					dec_t = position[1].deg
					ra_t_all.append(ra_t)
					dec_t_all.append(dec_t)
				center = SkyCoord(ra = ra_t_all,dec = dec_t_all,frame='icrs', unit='deg')
				return center,index_all
		else:
			return None
	def get_good_detector_centers_with_time(self,t,source):
		
		centers = self.get_detector_centers_with_time(t)
		if centers is None:
			return None
		good_detector_centers = []
		good_detector_index = []
		try:
			for i in range(len(t)):
				center_all = centers[i]
				condition = center_all.separation(source)<=self.radius * u.degree
				good_detector_index.append(condition)
				good_detector_centers.append(center_all[condition])
			return good_detector_index, good_detector_centers
		except (TypeError):
			
			condition = centers.separation(source)<=self.radius * u.degree
			return condition,centers[condition]
		
	
		
	def get_separation_with_time(self,t,source):
		deter_name = self.detectors.name_list
		center_f = self.detectors.center_function
		if center_f is not None and self.time_band is not None:
			retr = {'time':t}
			t = np.array(t)
			if t.size >0:
				tband = t.max()-t.min()
				tband_sl = self.time_band[1] - self.time_band[0]
				if tband<=tband_sl+2:
					t[t<=self.time_band[0]] = self.time_band[0]+0.00001
					t[t>=self.time_band[1]] = self.time_band[1]-0.00001
				else:
					t[t<=self.time_band[0]] = np.nan
					t[t>=self.time_band[1]] = np.nan
			for deteri in deter_name:
				xf,yf,zf = center_f[deteri]
				x = xf(t)
				y = yf(t)
				z = zf(t)
				position = cartesian_to_spherical(x,y,z)
				ra_t = position[2].deg
				dec_t = position[1].deg
				center = SkyCoord(ra = ra_t,dec = dec_t,frame='icrs', unit='deg')
				retr[deteri] = center.separation(source).value
			return 	pd.DataFrame(retr)
		else:
			return None
		
	def all_sky(self,num_points = 3000):
		
		ra_grid_tmp = np.linspace(0, 360, num_points)
		dec_range = [-90, 90]
		cosdec_min = np.cos(np.deg2rad(90.0 + dec_range[0]))
		cosdec_max = np.cos(np.deg2rad(90.0 + dec_range[1]))
		v = np.linspace(cosdec_min, cosdec_max, num_points)
		v = np.arccos(v)
		v = np.rad2deg(v)
		v -= 90.
		dec_grid_tmp = v
		ra_grid = np.zeros(num_points ** 2)
		dec_grid = np.zeros(num_points ** 2)
		itr = 0
		for ra in ra_grid_tmp:
			for dec in dec_grid_tmp:
				ra_grid[itr] = ra
				dec_grid[itr] = dec
				itr += 1
		return SkyCoord(ra=ra_grid, dec=dec_grid, frame='icrs', unit='deg')
	
		
	def get_detector_centers(self,index = None):
		if index is not None:
			retur_list = []
			for i in index:
				retur_list.append(self.detectors.center_all[i])
			return retur_list
		else:
			return self.detectors.center_all
	
	def get_detector_index(self,index = None):
		if index is not None:
			retur_list = []
			for i in index:
				retur_list.append(self.detectors.detector_index[i])
			return retur_list
		else:
			return self.detectors.detector_index
	
	def get_good_detector_centers(self,source = None,index = None):
		'''
		
		:param source:
		:return:
		'''
		
		if source is not None:
			good_detector_centers = []
			good_detector_index = []
			if index is not None:
				center_all_list = []
				for i in index:
					center_all_list.append(self.detectors.center_all[i])
				
			else:
				center_all_list = self.detectors.center_all
			for center_all in center_all_list:
				condition = center_all.separation(source)<=self.radius * u.degree
				good_detector_index.append(condition)
				good_detector_centers.append(center_all[condition])
			return good_detector_index, good_detector_centers
		else:
			print('No source! return []')
			return None
	
	
	def get_separation(self,index,source = None):
		'''
		
		:param source:
		:return:
		'''
		tab = Table(names=["Detector_index", "Separation"], dtype=[int, np.float64])
		if source is not None:
			for index1,center in enumerate(self.detectors.center_all[index]):
				sep = center.separation(source)
				tab.add_row([index1,sep])
			tab['Separation'].unit = u.degree
			#tab.sort('Separation')
			return tab
		else:
			print('No source! return None')
			return None

	def get_fov(self,conter,radius = 10.):
		fov_point_list = []
		for conter_i in conter:
			poly = SphericalPolygon.from_cone(conter_i.ra.value,conter_i.dec.value,radius,steps=180)
			x,y = np.array(list(poly.to_radec())[0])
			fov_point_list.append([radius,x,y])
		return fov_point_list
	def get_earth_point_with_time(self,t):
		
		earth_radius = 6371. * u.km
		if self.sc_pos_f is not None and self.time_band is not None:
			x_f, y_f, z_f = self.sc_pos_f
			try:
				n = len(t)
				t = np.array(t)
				tband = t.max() - t.min()
				tband_sl = self.time_band[1] - self.time_band[0]
				if tband <= tband_sl + 2:
					t[t <= self.time_band[0]] = self.time_band[0] + 0.00001
					t[t >= self.time_band[1]] = self.time_band[1] - 0.00001
				else:
					t[t <= self.time_band[0]] = np.nan
					t[t >= self.time_band[1]] = np.nan
					
				x = x_f(t) * self.pos_unit
				y = y_f(t) * self.pos_unit
				z = z_f(t) * self.pos_unit
				earth_point_list = []
				for i in range(n):
					position = cartesian_to_spherical(x[i],y[i],z[i])
					xyz_position = SkyCoord(position[2].deg,position[1].deg,frame='icrs',unit='deg')
					fermi_radius = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2)
					radius_deg = np.rad2deg(np.arcsin((earth_radius / fermi_radius).to(u.dimensionless_unscaled)).value)
					poly = SphericalPolygon.from_cone(position[2].deg,position[1].deg,radius_deg,steps=180)
					x_,y_ = np.array(list(poly.to_radec())[0])
					earth_point_list.append([xyz_position,radius_deg,x_,y_])
				return earth_point_list
			except 	(TypeError):
				
				if (t<=self.time_band[0]):
					if (self.time_band[0]-t<=1):
						t = self.time_band[0]+0.00001
					else:
						t = np.nan
				if t>=self.time_band[1]:
					if (t-self.time_band[0]<=1):
						t = self.time_band[1]-0.00001
					else:
						t = np.nan
				x = x_f(t) * self.pos_unit
				y = y_f(t) * self.pos_unit
				z = z_f(t) * self.pos_unit
				position = cartesian_to_spherical(x,y,z)
				xyz_position = SkyCoord(position[2].deg,position[1].deg,frame='icrs',unit='deg')
				fermi_radius = np.sqrt(x**2 + y**2 + z**2)
				radius_deg = np.rad2deg(np.arcsin((earth_radius / fermi_radius).to(u.dimensionless_unscaled)).value)
				poly = SphericalPolygon.from_cone(position[2].deg,position[1].deg,radius_deg,steps=180)
				x_,y_ = np.array(list(poly.to_radec())[0])
				return xyz_position,radius_deg,x_,y_
		else:
			return None
			
			
	def get_earth_point(self,index = None):
		if self.sc_pos is not None:
			earth_point_list = []
			if index is not None:
				sc_pos_list = self.sc_pos[index]
			else:
				sc_pos_list = self.sc_pos
			for sc_pos in sc_pos_list:
				position = cartesian_to_spherical(-sc_pos[0],-sc_pos[1],-sc_pos[2])
				xyz_position = SkyCoord(position[2].deg,position[1].deg,frame='icrs',unit='deg')
				earth_radius = 6371. * u.km
				fermi_radius = np.sqrt((sc_pos ** 2).sum())
				radius_deg = np.rad2deg(np.arcsin((earth_radius / fermi_radius).to(u.dimensionless_unscaled)).value)
				poly = SphericalPolygon.from_cone(position[2].deg,position[1].deg,radius_deg,steps=180)
				x,y = np.array(list(poly.to_radec())[0])
				earth_point_list.append([xyz_position,radius_deg,x,y])
			return earth_point_list
		else:
			print('No satellite position!')
			return None
			
	def detector_plot(self,radius = 10.0,source=None,points = None,good = False,highlight = None,
	                  highlight_color = '#f26522',
	                  lon_0 = 180,ax = None,show_bodies = False,avoid_pole = True,time =None,
	                  style = 'A',index = 0,size = 1):
		#pole = SkyCoord([0, 0], [90, -90], frame='icrs', unit='deg')
		if ax is None:
			fig = plt.figure(figsize = (20,10))
			ax = fig.add_subplot(1,1,1,projection=ccrs.Mollweide(central_longitude=lon_0),facecolor = '#f6f5ec')
			
		xticks = list(range(-180, 180, 30))
		yticks = list(range(-90, 90, 15))
			
		lons_x = np.arange(0,360,30)
		lons_y = np.zeros(lons_x.size)
		lats_y = np.arange(-75,76,15)
		lats_x = np.zeros(lats_y.size)
			
		ax.gridlines(xlocs=xticks, ylocs=yticks)
		lats_y_ticke = ax.projection.transform_points(ccrs.Geodetic(),lats_x+lon_0+180.0, lats_y*1.1)
		lats_y_x = lats_y_ticke[:,0]*0.86
		lats_y_y = lats_y_ticke[:,1]
			
		proj_xyz = ax.projection.transform_points(ccrs.Geodetic(),np.array([0,30]), np.array([0,0]))
		dx_ = np.abs(proj_xyz[0][0]-proj_xyz[1][0])
		for indexi,i in enumerate(lons_x):
			ax.text(i,lons_y[indexi],r'$%d^{\circ}$'%i,transform = ccrs.Geodetic(),size = 20*size)
		for indexi,i in enumerate(lats_y):
			ax.text(lats_y_x[indexi]+dx_,lats_y_y[indexi],r'$%d^{\circ}$'%i,size = 20*size,ha = 'right',va = 'center')
		ax.set_global()
		ax.invert_xaxis()
			
			
		if time is not None:
			utc = self.Time_transition.met_to_utc(time)
			ax.set_title(str(utc.fits),size = 20*size)
		else:
			ax.set_title(str(index),size = 20*size)
			
		if good and source :
			if time is not None and self.met_time is not None:
				index_,centor = self.get_good_detector_centers_with_time(time,source=source)
			else:
				index_,centor = self.get_good_detector_centers(source,index = [index])
				index_ = index_[0]
				centor = centor[0]
		else:
			if time is not None:
				centor,index_ = self.get_detector_centers_with_time(time)
			else:
				index_ = self.get_detector_index(index=[index])[0]
				centor = self.get_detector_centers(index = [index])[0]
			
			
		if show_bodies and self.sc_pos is not None :
			#print('plot_earth!')
			if time is not None and self.met_time is not None:
				postion, r, lon, lat = self.get_earth_point_with_time(time)
			else:
				postion, r, lon, lat = self.get_earth_point(index=[index])[0]
			if avoid_pole:
				lat[lat>88.0]=88.0
				lat[lat<-88.0]=-88.0
			#print(lat)
			#ax.plot( lon, lat,'-',color = 'k',transform=ccrs.Geodetic(),linewidth=5)
			earth = Polygon(list(zip(lon, lat))[::-1], facecolor='#90d7ec', edgecolor='#90d7ec',
			                linewidth=2*size, alpha=1,transform=ccrs.Geodetic())
			ax.add_patch(earth)
			#ax.add_patch(mpatches.Circle(xy=[postion.ra.value,postion.dec.value],transform=ccrs.Geodetic(), radius=r, color='#90d7ec', alpha=1, zorder=0))
			if self.time is not None:
				if time is not None:
					time_utc = self.Time_transition.met_to_utc(time)
					earth_r = get_body_barycentric('earth', time_utc)
					moon_r = get_body_barycentric('moon',time_utc )
					
				else:
					earth_r = get_body_barycentric('earth', self.time[index])
					moon_r = get_body_barycentric('moon', self.time[index])
					
				r_e_m = moon_r - earth_r
				
				if time is not None:
					x_f, y_f, z_f = self.sc_pos_f
					x = x_f(time)
					y = y_f(time)
					z = z_f(time)
					r = -np.array([x,y,z])*self.pos_unit - np.array([r_e_m.x.value, r_e_m.y.value, r_e_m.z.value]) * u.km
				else:
					r = self.sc_pos[index] - np.array([r_e_m.x.value, r_e_m.y.value, r_e_m.z.value]) * u.km
				
				moon_point_d = cartesian_to_spherical(-r[0], -r[1], -r[2])
				moon_ra, moon_dec = moon_point_d[2].deg, moon_point_d[1].deg
				moon_point = SkyCoord(moon_ra, moon_dec, frame='icrs', unit='deg')
				
				#moon_ra, moon_dec = map(moon_point.ra.deg, moon_point.dec.deg)
				ax.plot(moon_point.ra.deg, moon_point.dec.deg, 'o', color='#72777b', markersize=20*size,transform=ccrs.Geodetic())
				ax.text(moon_point.ra.deg, moon_point.dec.deg, 'moon', size=20*size,transform=ccrs.Geodetic(),va = 'center',ha='center')
			if show_bodies and self.time is not None:
				if time is not None:
					time_utc = self.Time_transition.met_to_utc(time)
					tmp_sun = get_sun(time_utc)
				else:
					tmp_sun = get_sun(self.time[index])
				sun_position = SkyCoord(tmp_sun.ra.deg, tmp_sun.dec.deg, unit='deg', frame='icrs')
				ax.plot(sun_position.ra.value, sun_position.dec.value, 'o', color='#ffd400', markersize=40*size,transform=ccrs.Geodetic())
				ax.text(sun_position.ra.value, sun_position.dec.value, 'sun', size=20*size,transform=ccrs.Geodetic(),va = 'center',ha='center')
		
		fovs = self.get_fov(centor, radius)
		
		for i,v in enumerate(index_):
			r,ra,dec = fovs[i]
			name_ = self.detectors.name_list[v]
			
			if avoid_pole:
				dec[dec>88.0]=88.0
				dec[dec<-88.0]=-88.0
			if highlight is not None:
				if name_ in highlight:
					color_ = highlight_color
				else:
					color_ = self.detectors.color_list[v]
			else:
				color_ = self.detectors.color_list[v]
			detec = Polygon(list(zip(ra,dec))[::-1],facecolor=color_,edgecolor=color_,linewidth=2*size, alpha=0.5,transform=ccrs.Geodetic())
			ax.add_patch(detec)
			plt.text(centor[i].ra.value, centor[i].dec.value,str(name_), color=self.detectors.color_list[v], size=22*size,transform=ccrs.Geodetic(),va = 'center',ha='center')
		
		if source:
			ax.plot(source.ra.value, source.dec.value, '*', color='#f36c21', markersize=20.*size,transform=ccrs.Geodetic())
		if points:
			
			ax.plot(points.ra.value, points.dec.value, '*', color='#c7a252', markersize=20.*size,transform=ccrs.Geodetic())
			
		return ax
		
	def detector_video(self,dt,savevdir,radius = 10.0,source=None,points = None,good = False,
	                   lon_0 = 180.,ax = None,show_bodies = False,style = 'A'):
		
		name_list = self.detectors.name_list
		color_list = self.detectors.color_list
		
		if good and source :
			index_list,centor_list = self.get_good_detector_centers(source)
		else:
			index_list = self.get_detector_index()
			centor_list = self.get_detector_centers()
		if show_bodies and self.sc_pos is not None:
			sc_pos = self.sc_pos
			earth_points_list = self.get_earth_point()
		else:
			sc_pos = None
			earth_points_list = None
		if self.time is not None:
			time = self.time
		else:
			time = None
		#fig,ax = plt.subplots()
		fig = plt.figure(figsize = (20,10))
		ax = fig.add_subplot(1,1,1,projection=ccrs.Mollweide(central_longitude=lon_0),facecolor = '#f6f5ec')
		duration = dt * len(self.index)
		xticks = list(range(-180, 180, 30))
		yticks = list(range(-90, 90, 15))
		
		lons_x = np.arange(0,360,30)
		lons_y = np.zeros(lons_x.size)
		lats_y = np.arange(-75,76,15)
		lats_x = np.zeros(lats_y.size)
		
		def make_frame(t):
			n = int(t/dt)
			ax.clear()
			ax.set_title(str(n))
			#map = Basemap(projection=projection,lat_0=lat_0,lon_0 = lon_0,resolution = 'l',area_thresh=1000.0,celestial=True,ax = ax)
			index_,centor = index_list[n],centor_list[n]
			
			if source:
				#ra, dec = map(source.ra.value, source.dec.value)
				ax.plot(source.ra.value, source.dec.value, '*', color='#f36c21', markersize=20., transform=ccrs.Geodetic())
			if points:
				ax.plot(points.ra.value, points.dec.value, '*', color='#c7a252', markersize=20., transform=ccrs.Geodetic())
			if show_bodies and sc_pos is not None:
				
				postion, r, lon, lat = earth_points_list[n]
				lat[lat > 88.0] = 88.0
				lat[lat < -88.0] = -88.0
				#lon, lat = map(lon, lat)
				earth = Polygon(list(zip(lon, lat))[::-1], facecolor='#90d7ec', edgecolor='#90d7ec',
					         linewidth=2, alpha=1, transform=ccrs.Geodetic())
				ax.add_patch(earth)
				#ax.add_patch(mpatches.Circle(xy=[postion.ra.value, postion.dec.value], radius=r,
				#                             color='#90d7ec', alpha=0.3, transform=ccrs.Geodetic(),
				#                             zorder=0))
				if time is not None:
					earth_r = get_body_barycentric('earth',time[n])
					moon_r = get_body_barycentric('moon',time[n])
					r_e_m = moon_r - earth_r
					r = sc_pos[n] - np.array([r_e_m.x.value,r_e_m.y.value,r_e_m.z.value])*u.km
					moon_point_d = cartesian_to_spherical(-r[0],-r[1],-r[2])
					moon_ra,moon_dec = moon_point_d[2].deg,moon_point_d[1].deg
					moon_point = SkyCoord(moon_ra,moon_dec,frame='icrs', unit='deg')
					ax.plot(moon_point.ra.deg,moon_point.dec.deg,'o',color = '#72777b',markersize = 20,transform=ccrs.Geodetic())
					ax.text(moon_point.ra.deg,moon_point.dec.deg,'moon',size = 20,transform=ccrs.Geodetic(),va = 'center',ha='center')
				if show_bodies and time is not None:
					tmp_sun = get_sun(time[n])
					sun_position = SkyCoord(tmp_sun.ra.deg,tmp_sun.dec.deg,unit='deg', frame='icrs')
					ax.plot(sun_position.ra.value,sun_position.dec.value ,'o',color = '#ffd400', markersize=40,transform=ccrs.Geodetic())
					ax.text(sun_position.ra.value,sun_position.dec.value,'sun',size = 20,transform=ccrs.Geodetic(),va = 'center',ha='center')
					
			fovs = get_fov(centor,radius)
			
			
			for i,v in enumerate(index_):
				r,ra,dec = fovs[i]
				dec[dec > 88.0] = 88.0
				dec[dec < -88.0] = -88.0
				detec = Polygon(list(zip(ra,dec))[::-1],facecolor=color_list[v],edgecolor=color_list[v],linewidth=2, alpha=0.5,transform=ccrs.Geodetic())
				ax.add_patch(detec)
				#ra_x,dec_y = map(centor[i].ra.value+2.5,centor[i].dec.value-1)
				ax.text(centor[i].ra.value, centor[i].dec.value,str(name_list[v]), color=color_list[v], size=22,transform=ccrs.Geodetic(),va = 'center',ha='center')
			
			ax.gridlines(xlocs=xticks, ylocs=yticks)
			lats_y_ticke = ax.projection.transform_points(ccrs.Geodetic(),lats_x+lon_0+180.0, lats_y*1.1)
			lats_y_x = lats_y_ticke[:,0]*0.86
			lats_y_y = lats_y_ticke[:,1]
			
			proj_xyz = ax.projection.transform_points(ccrs.Geodetic(),np.array([0,30]), np.array([0,0]))
			dx_ = np.abs(proj_xyz[0][0]-proj_xyz[1][0])
			for indexi,i in enumerate(lons_x):
				ax.text(i,lons_y[indexi],r'$%d^{\circ}$'%i,transform = ccrs.Geodetic(),size = 20)
			for indexi,i in enumerate(lats_y):
				ax.text(lats_y_x[indexi]+dx_,lats_y_y[indexi],r'$%d^{\circ}$'%i,size = 20,ha = 'right',va = 'center')
			ax.set_global()
			ax.invert_xaxis()
			
			#n = n + 1
			return mplfig_to_npimage(fig)
		animation = VideoClip(make_frame, duration=duration)
		#animation.write_videofile(savevdir, fps=1/dt,codec = 'h264')
		animation.write_videofile(savevdir, fps=1/dt)
		animation.close()
		

def get_fov(conter,radius = 10.):
		fov_point_list = []
		for conter_i in conter:
			poly = SphericalPolygon.from_cone(conter_i.ra.value,conter_i.dec.value,radius,steps=100)
			x,y = [p for p in poly.to_radec()][0]
			fov_point_list.append([radius,x,y])
		return fov_point_list

def loop_data(rad):
	rad = np.array(rad)
	rad[rad < 0] = rad[rad < 0] + 360
	rad[rad >= 360] = rad[rad >= 360] - 360
	return rad

def tranlation_lon_0(rad, lon_0):
	rad = np.array(rad)
	rad = rad - lon_0
	rad = loop_data(rad)
	return rad

def judge(x,n,xxdd,xxdd180):
	d_ra = np.abs(get_ra(x,n))
	case1 = (x>xxdd)&(xxdd>180)&(n<=xxdd)&(n>xxdd180)&(d_ra<20)
	case2 = (n>xxdd)&(xxdd>180)&(x<=xxdd)&(x>xxdd180)&(d_ra<20)
	case3 = (x<xxdd180)&(x>=xxdd)&(xxdd<=180)&(n<xxdd)&(d_ra<20)
	case4 = (n<xxdd180)&(n>=xxdd)&(xxdd<=180)&(x<xxdd)&(d_ra<20)
	return case1|case2|case3|case4

def get_ra(x,x0):
	xxx = np.arange(-5,6,1)
	xar = xxx*360+x
	dx = xar - x0
	dxabs = np.abs(dx)
	return dx[np.argmin(dxabs)]
def get_circle(position,radius,lon_0,map_,facecolor='coral', edgecolor='coral',linewidth=0., alpha=1.):
	'''
	
	:param position:
	:param radius:
	:param lon_0:
	:param map_:
	:param facecolor:
	:param edgecolor:
	:param linewidth:
	:param alpha:
	:return:
	'''
	x0 = position.ra.value
	y0 = position.dec.value
	xx_dd = tranlation_lon_0([lon_0 - 180],x0 - 180)[0]
	poly = SphericalPolygon.from_cone(180,y0,radius,steps=200)
	x,y = [p for p in poly.to_radec()][0]
	
	poly_arr = np.array([89.9999,-89.9999])
	d_deg = np.abs(poly_arr-y0)
	d_l = poly_arr[d_deg<radius]

	add_x = np.zeros(100) + xx_dd
	add_y = np.linspace(-89.9999, 89.9999, 100)
	if d_l.size > 0:
		xx_dd180 = loop_data([xx_dd+180])[0]
		#print('pole_in.size > 0')
		if d_l[0] == 89.9999:
			#print('pole_in.dec.deg == 90.')
			x_n = []
			y_n = []
			n0 = x[0]
			y0 = y[0]
			for i in range(1, len(x)):
				
				if judge(x[i],n0,xx_dd,xx_dd180) == False :
					x_n.append(x[i])
					y_n.append(y[i])
					n0 = x[i]
					y0 = y[i]
				else:
					#print(x[i],n0,xx_dd,xx_dd180)
					#print('nnn0',n0,y0)
					#print('xxxi',x[i],y[i])
					indexs = np.where(add_y >= y0)[0]
					add_y_a = add_y[indexs]
					add_x_a = add_x[indexs]
					indexssort = np.argsort(add_y_a)
					ds = get_ra(n0,x[i])
					n_v = ds / np.abs(ds)
					#print('n_v',n_v)
					add_y_a1 = add_y_a[indexssort]
					add_x_a1 = add_x_a[indexssort] + 0.1*n_v
					#print('add_x',add_x_a,add_y_a)
					#print(n0,x[i],xx_dd)
					#print(n0 - xx_dd)
					for i in range(len(add_y_a1)):
						x_n.append(add_x_a1[i])
						y_n.append(add_y_a1[i])
					indexssort = np.argsort(-add_y_a)
					add_y_a2 = add_y_a[indexssort]
					add_x_a2 = add_x_a[indexssort] - 0.1*n_v
					#print('add_x',add_x_a,add_y_a)
					for i in range(len(add_y_a2)):
						x_n.append(add_x_a2[i])
						y_n.append(add_y_a2[i])
					n0 = x[i]
					y0 = y[i]
			#print(x_n)
			#print(y_n)
			x = [tranlation_lon_0(x_n, -x0 + 180)]
			y = [np.array(y_n)]
		else:
			#print('pole_in.dec.deg == 90. else')
			x_n = []
			y_n = []
			n0 = x[0]
			y0 = y[0]
			
			for i in range(1, len(x)):
				if judge(x[i],n0,xx_dd,xx_dd180) == False:
					
					x_n.append(x[i])
					y_n.append(y[i])
					n0 = x[i]
					y0 = y[i]
				else:
					#print('dddddd',x[i],n0,xx_dd,xx_dd180)
					#print(i)
					indexs = np.where(add_y <= y0)[0]
					add_y_a = add_y[indexs]
					#add_x_a = np.zeros(add_y_a.size) + n0
					add_x_a = add_x[indexs]
					indexssort = np.argsort(-add_y_a)
					ds = get_ra(n0,x[i])
					n_v = ds / np.abs(ds)
					add_y_a1 = add_y_a[indexssort]
					add_x_a1 = add_x_a[indexssort]+0.1*n_v
					
					#print(n0,x[i],xx_dd)
					#print(n0 - xx_dd)
					for i in range(len(add_y_a1)):
						#pass
						x_n.append(add_x_a1[i])
						y_n.append(add_y_a1[i])
					indexssort = np.argsort(add_y_a)
					add_y_a2 = add_y_a[indexssort]
					add_x_a2 = add_x_a[indexssort] - 0.1*n_v
				
					for i in range(len(add_y_a2)):
						x_n.append(add_x_a2[i])
						y_n.append(add_y_a2[i])
					n0 = x[i]
					y0 = y[i]

			x = [tranlation_lon_0(x_n, -x0 + 180)]
			y = [np.array(y_n)]
	else:
		#print('pole_in.size > 0 else')
		x_1 = []
		y_1 = []
		x_2 = []
		y_2 = []
		
		if len(x[x > xx_dd]) > len(x[x <= xx_dd]):
			#print('len(x[x > xx_dd]) > len(x[x <= xx_dd])')
			if len(x[x <= xx_dd]) > 0:
				#print('len(x[x <= xx_dd]) > 0')
				# x大于xx_dd
				for i in range(len(x)):
					# x小于xx_dd一律等于xx_dd+0.1
					if x[i] <= xx_dd:
						x_1.append(xx_dd + 0.1)
						y_1.append(y[i])
					else:
						x_1.append(x[i])
						y_1.append(y[i])
				
				aa = y[x <= xx_dd]
				amax = aa.max()
				amin = aa.min()
				for i in range(len(x)):
					# x大于xx_dd一律等于xx_dd-0.1
					if x[i] >= xx_dd:
						x_2.append(xx_dd - 0.1)
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
					else:
						x_2.append(x[i])
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
				x = [tranlation_lon_0(x_1, -x0 + 180),
				     tranlation_lon_0(x_2, -x0 + 180)]
				y = [np.array(y_1), np.array(y_2)]
			else:
				x = [tranlation_lon_0(x, -x0 + 180)]
				y = [y]
		else:
			#print('len(x[x > xx_dd]) > len(x[x <= xx_dd]) else')
			if len(x[x > xx_dd]) > 0:
				#print('len(x[x > xx_dd]) > 0')
				# 小于xx_dd为主
				for i in range(len(x)):
					if x[i] >= xx_dd:
						x_1.append(xx_dd - 0.1)
						y_1.append(y[i])
					else:
						x_1.append(x[i])
						y_1.append(y[i])
				
				aa = y[x >= xx_dd]
				amax = aa.max()
				amin = aa.min()
				for i in range(len(x)):
					if x[i] <= xx_dd:
						x_2.append(xx_dd + 0.1)
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
					else:
						x_2.append(x[i])
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
				x = [tranlation_lon_0(x_1, -x0 + 180),
				     tranlation_lon_0(x_2, -x0 + 180)]
				y = [np.array(y_1), np.array(y_2)]
			else:
				#print('len(x[x > xx_dd]) > 0 else')
				x = [tranlation_lon_0(x, -x0 + 180)]
				y = [y]
	poly_list = []
	for i in range(len(x)):
		x_, y_ = map_(x[i], y[i])
		poly1 = Polygon(list(zip(x_, y_)), facecolor=facecolor, edgecolor=edgecolor,linewidth=linewidth, alpha=alpha)
		poly_list.append(poly1)
	return poly_list
	
def get_poly(position, radius, x, y, lon_0,map_,facecolor='coral', edgecolor='coral',linewidth=0., alpha=1.):
	x = np.array(x)
	y = np.array(y)
	center_ra = position.ra.value
	center_dec = position.dec.value
	xx_dd = tranlation_lon_0([lon_0 - 180], center_ra - 180)[0]
	x = tranlation_lon_0(x, center_ra - 180)
	poly_arr = np.array([89.9999, -89.9999])
	d_deg = np.abs(poly_arr - center_dec)
	d_l = poly_arr[d_deg < radius]
	add_x = np.zeros(100) + xx_dd
	add_y = np.linspace(-89.9999, 89.9999, 100)
	if d_l.size > 0:
		xx_dd180 = loop_data([xx_dd+180])[0]
		#print('pole_in.size > 0')
		if d_l[0] == 89.9999:
			#print('pole_in.dec.deg == 90.')
			x_n = []
			y_n = []
			n0 = x[0]
			y0 = y[0]
			for i in range(1, len(x)):
				
				if judge(x[i],n0,xx_dd,xx_dd180) == False :
					x_n.append(x[i])
					y_n.append(y[i])
					n0 = x[i]
					y0 = y[i]
				else:
					#print(x[i],n0,xx_dd,xx_dd180)
					indexs = np.where(add_y >= y0)[0]
					add_y_a = add_y[indexs]
					add_x_a = add_x[indexs]
					indexssort = np.argsort(add_y_a)
					ds = get_ra(n0,x[i])
					n_v = ds / np.abs(ds)
					add_y_a1 = add_y_a[indexssort]
					add_x_a1 = add_x_a[indexssort] + 0.1*n_v
					for i in range(len(add_y_a1)):
						x_n.append(add_x_a1[i])
						y_n.append(add_y_a1[i])
					indexssort = np.argsort(-add_y_a)
					add_y_a2 = add_y_a[indexssort]
					
					add_x_a2 = add_x_a[indexssort] - 0.1*n_v
					for i in range(len(add_y_a2)):
						x_n.append(add_x_a2[i])
						y_n.append(add_y_a2[i])
					n0 = x[i]
					y0 = y[i]
			x = [tranlation_lon_0(x_n, -center_ra + 180)]
			y = [np.array(y_n)]
		else:
			#print('pole_in.dec.deg == 90. else')
			x_n = []
			y_n = []
			n0 = x[0]
			y0 = y[0]
			
			for i in range(1, len(x)):
				if judge(x[i],n0,xx_dd,xx_dd180) == False:
					
					x_n.append(x[i])
					y_n.append(y[i])
					n0 = x[i]
					y0 = y[i]
				else:
					#print('dddddd',x[i],n0,xx_dd,xx_dd180)
					#print(i)
					indexs = np.where(add_y <= y0)[0]
					add_y_a = add_y[indexs]
					#add_x_a = np.zeros(add_y_a.size) + n0
					add_x_a = add_x[indexs]
					indexssort = np.argsort(-add_y_a)
					add_y_a1 = add_y_a[indexssort]
					ds = get_ra(n0,x[i])
					n_v = ds / np.abs(ds)
					add_x_a1 = add_x_a[indexssort]+0.1*n_v
					
					#print(n0,x[i],xx_dd)
					#print(n0 - xx_dd)
					for i in range(len(add_y_a1)):
						#pass
						x_n.append(add_x_a1[i])
						y_n.append(add_y_a1[i])
					indexssort = np.argsort(add_y_a)
					add_y_a2 = add_y_a[indexssort]
					
					add_x_a2 = add_x_a[indexssort] - 0.1*n_v
				
					for i in range(len(add_y_a2)):
						
						x_n.append(add_x_a2[i])
						y_n.append(add_y_a2[i])
					n0 = x[i]
					y0 = y[i]

			x = [tranlation_lon_0(x_n, -center_ra + 180)]
			y = [np.array(y_n)]
	else:
		#print('pole_in.size > 0 else')
		x_1 = []
		y_1 = []
		x_2 = []
		y_2 = []
		
		if len(x[x > xx_dd]) > len(x[x <= xx_dd]):
			#print('len(x[x > xx_dd]) > len(x[x <= xx_dd])')
			if len(x[x <= xx_dd]) > 0:
				#print('len(x[x <= xx_dd]) > 0')
				
				for i in range(len(x)):
					
					if x[i] <= xx_dd:
						x_1.append(xx_dd + 0.1)
						y_1.append(y[i])
					else:
						x_1.append(x[i])
						y_1.append(y[i])
				
				aa = y[x <= xx_dd]
				amax = aa.max()
				amin = aa.min()
				for i in range(len(x)):
					
					if x[i] >= xx_dd:
						x_2.append(xx_dd - 0.1)
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
					else:
						x_2.append(x[i])
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
				x = [tranlation_lon_0(x_1, -center_ra + 180),
				     tranlation_lon_0(x_2, -center_ra + 180)]
				y = [np.array(y_1), np.array(y_2)]
			else:
				x = [tranlation_lon_0(x, -center_ra + 180)]
				y = [y]
		else:
			#print('len(x[x > xx_dd]) > len(x[x <= xx_dd]) else')
			if len(x[x > xx_dd]) > 0:
				#print('len(x[x > xx_dd]) > 0')
				
				for i in range(len(x)):
					if x[i] >= xx_dd:
						x_1.append(xx_dd - 0.1)
						y_1.append(y[i])
					else:
						x_1.append(x[i])
						y_1.append(y[i])
				
				aa = y[x >= xx_dd]
				amax = aa.max()
				amin = aa.min()
				for i in range(len(x)):
					if x[i] <= xx_dd:
						x_2.append(xx_dd + 0.1)
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
					else:
						x_2.append(x[i])
						if y[i] > amax:
							y_2.append(amax)
						elif y[i] < amin:
							y_2.append(amin)
						else:
							y_2.append(y[i])
				x = [tranlation_lon_0(x_1, -center_ra + 180),
				     tranlation_lon_0(x_2, -center_ra + 180)]
				y = [np.array(y_1), np.array(y_2)]
			else:
				#print('len(x[x > xx_dd]) > 0 else')
				x = [tranlation_lon_0(x, -center_ra + 180)]
				y = [y]
	poly_list = []
	for i in range(len(x)):
		x_, y_ = map_(x[i], y[i])
		poly1 = Polygon(list(zip(x_, y_)), facecolor=facecolor, edgecolor=edgecolor,linewidth=linewidth, alpha=alpha)
		poly_list.append(poly1)
	return poly_list
