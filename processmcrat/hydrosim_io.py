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

class HydroSimLoad(object):
    def __init__(self, fileroot_name, file_directory=None, sim_type='flash', coordinate_sys='cartesian', density_scale=1*u.g/u.cm**3,\
                 length_scale=1*u.cm, velocity_scale=const.c.cgs):
        """
        Initalized the hydrosimload class with the directory that the hydro files are located in, the
        :param file_directory:
        """
        if file_directory is not None:
            self.file_directory=file_directory
        else:
            self.file_directory=curdir()
        self.fileroot_name=fileroot_name

        self.coordinate_sys=coordinate_sys
        self.sim_type=sim_type
        self.density_scale=density_scale
        self.length_scale=length_scale
        self.velocity_scale=velocity_scale
        self.pressure_scale=density_scale*velocity_scale**2
        self.magnetic_scale=np.sqrt(4*np.pi*density_scale*velocity_scale**2)
        self.time_scale=length_scale/velocity_scale

    def load_frame(self, frame_num):

        sfrm = str(frame_num)
        if frame_num < 1000: sfrm = '0' + str(frame_num)
        if frame_num < 100: sfrm = '00' + str(frame_num)
        if frame_num < 10: sfrm = '000' + str(frame_num)

        self.frame_num=sfrm
        if 'flash' in self.coordinate_sys or 'Flash' in self.coordinate_sys:
            hydro_dict=self._read_flash_file(sfrm)
        elif 'pluto' in self.coordinate_sys or 'PLUTO' in self.coordinate_sys:
            print("Pluto and Pluto-chombo are not yet supported.")
            #hydro_dict=self.read_pluto_file(sfrm)

        self.hydro_data= hydro_dict

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
        xx = np.array(xx[jj, 0, :, :]) * self.length_scale
        #  yy=np.array(yy[jj,0,:,:]+1) this takes care of the fact that lngths scales with 1e9
        yy = np.array(yy[jj, 0, :, :]) * self.length_scale
        szxx = np.array(szxx[jj, 0, :, :]) * self.length_scale
        szyy = np.array(szyy[jj, 0, :, :]) * self.length_scale
        vx = np.array(vx[jj, 0, :, :])*self.velocity_scale
        vy = np.array(vy[jj, 0, :, :])*self.velocity_scale
        dens = np.array(dens[jj, 0, :, :])*self.density_scale
        pres = np.array(pres[jj, 0, :, :])*self.pressure_scale

        hydro_dict=dict(x0=xx, x1=yy, x2=None, dx0=szxx, dx1=szyy, dx2=None, pres=pres, dens=dens, v0=vx, v1=vy, )

        return hydro_dict
