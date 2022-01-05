"""
This file contains classes to take hydro simulation data that was analyzed with MCRaT and process it to be used in
analyzing and understanding the MCRaT results

Written by: Tyler Parsotan

"""

import astropy as ap
import h5py as h5
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.units import UnitConversionError
from processmcrat import curdir
import tables as tb

class HydroSim(object):
    def __init__(self, fileroot_name, file_directory=None, hydrosim_type='flash', coordinate_sys='cartesian', density_scale=1*u.g/u.cm**3,\
                 length_scale=1*u.cm, velocity_scale=const.c.cgs, hydrosim_dim=2):
        """
        Initalized the hydrosimload class with the directory that the hydro files are located in, the
        :param file_directory:
        """

        if isinstance(density_scale, u.Quantity) and isinstance(length_scale, u.Quantity) and isinstance(velocity_scale, u.Quantity):
            if file_directory is not None:
                self.file_directory=file_directory
            else:
                self.file_directory=curdir()
            self.fileroot_name=fileroot_name

            self.coordinate_sys=coordinate_sys
            self.hydrosim_type=hydrosim_type
            self.density_scale=density_scale
            self.length_scale=length_scale
            self.velocity_scale=velocity_scale
            self.pressure_scale=density_scale*velocity_scale**2
            self.magnetic_scale=np.sqrt(4*np.pi*density_scale*velocity_scale**2)
            self.time_scale=length_scale/velocity_scale
            self.spatial_limit_idx=None
        else:
            print('Make sure that the density, length and velocity scales have units associated with them.')

    def load_frame(self, frame_num):

        sfrm = str(frame_num)
        if frame_num < 1000: sfrm = '0' + str(frame_num)
        if frame_num < 100: sfrm = '00' + str(frame_num)
        if frame_num < 10: sfrm = '000' + str(frame_num)

        self.frame_num=sfrm
        if 'flash' in self.hydrosim_type or 'Flash' in self.hydrosim_type:
            hydro_dict=self._read_flash_file(sfrm)
        elif 'pluto' in self.hydrosim_type or 'PLUTO' in self.hydrosim_type:
            print("Pluto and Pluto-chombo are not yet supported.")
            #hydro_dict=self.read_pluto_file(sfrm)
        else:
            print(self.hydrosim_type+" is not supported as this time.")

        self.hydro_data = hydro_dict

    def _read_flash_file(self, frame_num):

        file = tb.open_file(self.file_directory+"/"+self.fileroot_name+frame_num)
        print('Reading Flash frame %s positional, density, pressure, and velocity information.'%(frame_num))
        xy = file.get_node('/', 'coordinates')
        xy = xy.read()
        x = np.array(xy[:, 0])
        y = np.array(xy[:, 1])
        sz = file.get_node('/', 'block size')
        sz = sz.read()
        szx = np.array(sz[:, 0])
        szy = np.array(sz[:, 1])
        vx = file.get_node('/', 'velx')
        vx = vx.read()
        vy = file.get_node('/', 'vely')
        vy = vy.read()
        vv = np.sqrt(vx ** 2 + vy ** 2)
        dens = file.get_node('/', 'dens')
        dens = dens.read()
        pres = file.get_node('/', 'pres')
        pres = pres.read()

        xx = np.zeros(vx.shape)
        yy = np.zeros(vx.shape)
        szxx = np.zeros(vx.shape)  # for resolution
        szyy = np.zeros(vx.shape)
        x1 = np.array([-7., -5, -3, -1, 1, 3, 5, 7]) / 16.
        x2 = np.empty([8, 8])
        y2 = np.empty([8, 8])
        for ii in range(0, 8, 1):
            y2[:, ii] = np.array(x1)
            x2[ii, :] = np.array(x1)
        for ii in range(0, x.size):
            xx[ii, 0, :, :] = np.array(x[ii] + szx[ii] * x2)
            yy[ii, 0, :, :] = np.array(y[ii] + szy[ii] * y2)
        for i in range(0, x.size):
            szxx[i, 0, :, :] = np.array(szx[i] / 8)
            szyy[i, 0, :, :] = np.array(szy[i] / 8)

        nty = file.get_node('/', 'node type')
        nty = nty.read()
        file.close()
        jj = np.where(nty == 1)
        xx = np.array(xx[jj, 0, :, :]).flatten() * self.length_scale
        yy = np.array(yy[jj, 0, :, :]).flatten() * self.length_scale
        szxx = np.array(szxx[jj, 0, :, :]).flatten() * self.length_scale
        szyy = np.array(szyy[jj, 0, :, :]).flatten() * self.length_scale
        vx = np.array(vx[jj, 0, :, :]).flatten()*self.velocity_scale
        vy = np.array(vy[jj, 0, :, :]).flatten()*self.velocity_scale
        dens = np.array(dens[jj, 0, :, :]).flatten()*self.density_scale
        pres = np.array(pres[jj, 0, :, :]).flatten()*self.pressure_scale
        gg = 1. / np.sqrt(1. - ((vx.cgs/const.c.cgs) ** 2 + (vy.cgs/const.c.cgs) ** 2))

        hydro_dict=dict(x0=xx, x1=yy, dx0=szxx, dx1=szyy, pres=pres, dens=dens, v0=vx, v1=vy, gamma=gg)

        return hydro_dict

    def apply_spatial_limits(self, x0_min, x0_max, x1_min, x1_max, x2_min=None, x2_max=None):

        #make sure that all limits have units associated with them
        if isinstance(x0_min, u.Quantity) and isinstance(x0_max, u.Quantity) and isinstance(x1_min,u.Quantity) and \
                isinstance(x1_max,u.Quantity) and (isinstance(x2_min, u.Quantity) or x2_min is None) and \
                (isinstance(x2_max, u.Quantity) or x2_max is None):
            if x2_max is None and x2_min is None:
                idx = np.where((self.hydro_data['x0']  >= x0_min) & (self.hydro_data['x0'] < x0_max) \
                               & (self.hydro_data['x1'] >= x1_min) & (self.hydro_data['x1'] < x1_max))[0]
            else:
                idx = np.where((self.hydro_data['x0'] >= x0_min) & (self.hydro_data['x0'] < x0_max) \
                               & (self.hydro_data['x1'] >= x1_min) & (self.hydro_data['x1'] < x1_max)\
                               & (self.hydro_data['x2'] >= x2_min) & (self.hydro_data['x2'] < x2_max))[0]

            self.spatial_limit_idx=idx
        else:
            print('Make sure that each minimum and maximum coordinate value has astropy units associated with it.')

    def get_data(self, key):
        if self.spatial_limit_idx is not None:
            data=self.hydro_data[key][self.spatial_limit_idx]
        else:
            data=self.hydro_data[key]
        return data