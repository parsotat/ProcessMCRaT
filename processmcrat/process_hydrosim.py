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
import os

radiation_dens_const=(4*const.sigma_sb/const.c)

def calc_temperature(pres):
    return (3 * (pres  / radiation_dens_const.decompose(bases=u.cgs.bases))) ** (1.0 / 4.0)

class HydroSim(object):
    def __init__(self, fileroot_name, file_directory=None, hydrosim_type='flash', coordinate_sys='cartesian', density_scale=1*u.g/u.cm**3,\
                 length_scale=1*u.cm, velocity_scale=const.c.cgs, hydrosim_dim=2, datatype=None, amr_level=3):
        """
        Initalized the hydrosimload class with the directory that the hydro files are located in, the
        :param file_directory:
        """

        if 'pluto' in hydrosim_type or 'PLUTO' in hydrosim_type:
            self.amr_level=amr_level
            if datatype is None:
                raise ValueError('There is no datatype specified for the PLUTO simulation files.')
            else:
                self.datatype=datatype
        else:
            self.datatype='hdf5'


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
            self.dimensions=hydrosim_dim
        else:
            raise ValueError('Make sure that the density, length and velocity scales have units associated with them.')

    def load_frame(self, frame_num):

        sfrm = str(frame_num)
        if frame_num < 1000: sfrm = '0' + str(frame_num)
        if frame_num < 100: sfrm = '00' + str(frame_num)
        if frame_num < 10: sfrm = '000' + str(frame_num)

        self.frame_num=sfrm
        if 'flash' in self.hydrosim_type or 'Flash' in self.hydrosim_type:
            hydro_dict=self._read_flash_file(sfrm)
        elif 'pluto' in self.hydrosim_type or 'PLUTO' in self.hydrosim_type:
            hydro_dict=self._read_pluto_file(sfrm)
        else:
            print(self.hydrosim_type+" is not supported as this time.")

        self.hydro_data = hydro_dict

    def _read_flash_file(self, frame_num, make_1d=True):

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
        yy = np.array(yy[jj, 0, :, :]) * self.length_scale
        szxx = np.array(szxx[jj, 0, :, :]) * self.length_scale
        szyy = np.array(szyy[jj, 0, :, :]) * self.length_scale
        vx = np.array(vx[jj, 0, :, :])*self.velocity_scale
        vy = np.array(vy[jj, 0, :, :])*self.velocity_scale
        dens = np.array(dens[jj, 0, :, :])*self.density_scale
        pres = np.array(pres[jj, 0, :, :])*self.pressure_scale
        gg = 1. / np.sqrt(1. - ((vx.cgs/const.c.cgs) ** 2 + (vy.cgs/const.c.cgs) ** 2))

        temp=calc_temperature(pres)
        if make_1d:
            xx=xx.flatten()
            yy=yy.flatten()
            szxx=szxx.flatten()
            szyy=szyy.flatten()
            vx=vx.flatten()
            vy=vy.flatten()
            dens=dens.flatten()
            pres=pres.flatten()
            gg=gg.flatten()
            temp=temp.flatten()

        hydro_dict=dict(x0=xx, x1=yy, dx0=szxx, dx1=szyy, temp=temp, pres=pres, dens=dens, v0=vx, v1=vy, gamma=gg)

        return hydro_dict

    def _read_pluto_file(self, frame_num, amr_lvl=None, make_1d=True):

        if amr_lvl is None:
            amr_lvl=self.amr_level

        #user needs to do eg: pip install git+https://gitlab.mpcdf.mpg.de/sdoetsch/pypluto.git or other git with most up-to-date version
        #see if its installed
        try:
            import pyPLUTO as pp
        except ModuleNotFoundError as err:
            print('Need pyPLUTO installed to read in PLUTO files.')
            print(err)

        print('Reading Pluto file %s positional, density, pressure, and velocity information at level %d.' % (frame_num, amr_lvl))

        if 'hdf5' in self.datatype:
            D = pp.pload(frame_num, w_dir=self.file_directory, datatype=self.datatype, level=self.amr_level)
        else:
            D = pp.pload(frame_num, w_dir=self.file_directory, datatype=self.datatype)

        x0=D.x1 * self.length_scale
        if D.geometry in ['CARTESIAN', 'CYLINDRICAL']:
            x1=D.x2 * self.length_scale
        else:
            x1 = D.x2*u.radian
        szx0=D.dx1 * self.length_scale
        if D.geometry in ['CARTESIAN', 'CYLINDRICAL']:
            szx1 = D.dx2 * self.length_scale
        else:
            szx1 = D.dx2 *u.radian

        pres=D.prs * self.pressure_scale
        dens=D.rho * self.density_scale
        v0=D.vx1 * self.velocity_scale
        v1=D.vx2 * self.velocity_scale
        gg = 1. / np.sqrt(1. - ((v0.cgs / const.c.cgs) ** 2 + (v1.cgs / const.c.cgs) ** 2))
        temp = calc_temperature(pres)

        #broadcast x0 and x1 arrays to be the proper size for the cell centered values
        x0=x0[:,np.newaxis]*np.ones(pres.shape)
        x1=x1[np.newaxis,:]*np.ones(pres.shape)
        
        if make_1d:
            x0=x0.flatten()
            x1=x1.flatten()
            szx0=szx0.flatten()
            szx1=szx1.flatten()
            v0=v0.flatten()
            v1=v1.flatten()
            dens=dens.flatten()
            pres=pres.flatten()
            gg=gg.flatten()
            temp=temp.flatten()
            

        hydro_dict = dict(x0=x0, x1=x1, dx0=szx0, dx1=szx1, temp=temp, pres=pres, dens=dens, v0=v0, v1=v1, gamma=gg)

        self.coordinate_sys=D.geometry

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
            raise ValueError('Make sure that each minimum and maximum coordinate value has astropy units associated with it.')

    def reset_spatial_limits(self):
        self.spatial_limit_idx=None

    def get_data(self, key):
        if key in self.hydro_data:
            if self.spatial_limit_idx is not None:
                data=self.hydro_data[key][self.spatial_limit_idx]
            else:
                data=self.hydro_data[key]
            return data
        else:
            raise ValueError(key+" is not a key in the HydroSim object")

    def coordinate_to_cartesian(self):
        if ('flash' in self.hydrosim_type or 'Flash' in self.hydrosim_type) and self.dimensions==2:
            x=self.get_data('x0')
            y=self.get_data('x1')
        else:
            if 'SPHERICAL' in self.coordinate_sys:
                #theta measured from y axis
                x=self.get_data('x0')*np.sin(self.get_data('x1'))
                y=self.get_data('x0')*np.cos(self.get_data('x1'))
            elif self.coordinate_sys in ['CARTESIAN', 'CYLINDRICAL']:
                x = self.get_data('x0')
                y = self.get_data('x1')
            else:
                raise ValueError("Converting the hydro coordinate system %s to cartesian coordinates is not supported at this time."%(self.geometry))

        return x,y

    def coordinate_to_spherical(self):
        if ('flash' in self.hydrosim_type or 'Flash' in self.hydrosim_type) and self.dimensions==2:
            r=np.sqrt(self.get_data('x0') ** 2 + self.get_data('x1') ** 2)
            theta=np.arctan2(self.get_data('x0'), self.get_data('x1'))
        else:
            if 'SPHERICAL' in self.coordinate_sys:
                #theta measured from y axis
                r=self.get_data('x0')
                theta=self.get_data('x1')
            elif self.coordinate_sys in ['CARTESIAN', 'CYLINDRICAL']:
                r = np.sqrt(self.get_data('x0') ** 2 + self.get_data('x1') ** 2)
                theta = np.arctan2(self.get_data('x0'), self.get_data('x1'))
            else:
                raise ValueError("Converting the hydro coordinate system %s to spherical coordinates is not supported at this time."%(self.geometry))

        return r, theta*u.rad


    def make_spherical_outflow(self, luminosity, gamma_infinity, r_0):
        if isinstance(luminosity, u.Quantity) and isinstance(r_0, u.Quantity):
            luminosity=luminosity.decompose(bases=u.cgs.bases)
            r_0=r_0.decompose(bases=u.cgs.bases)

            r, theta=self.coordinate_to_spherical()
            gg=np.zeros(r.size)
            pp=np.zeros(r.size)*self.pressure_scale

            jj = np.where(r < (r_0 * gamma_infinity))
            kk = np.where(r >= (r_0 * gamma_infinity))

            if jj[0].size > 0: gg[jj] = r[jj] / r_0
            gg[kk] = gamma_infinity

            vv = np.sqrt(1. - 1. / gg ** 2)
            vx = const.c.cgs * vv * self.get_data('x0') / r
            vy = const.c.cgs * vv * self.get_data('x1') / r

            dd = luminosity / 4 / np.pi / r ** 2 / const.c.cgs ** 3 / gg / gamma_infinity

            if jj[0].size > 0:
                pp[jj] = luminosity / 12 / np.pi / const.c.cgs * r_0 ** 2 / r[jj] ** 4 #/ const.c.cgs ** 2
            pp[kk] = luminosity / 12. / np.pi / const.c.cgs * r_0 ** (2. / 3.) / gamma_infinity ** (4. / 3.) * r[kk] ** (
                        -8. / 3.) #/ const.c.cgs ** 2

            self.hydro_data['v0']=vx
            self.hydro_data['v1'] = vy
            self.hydro_data['dens'] = dd
            self.hydro_data['pres'] = pp
            self.hydro_data['gamma'] = gg
            self.hydro_data['temp'] = calc_temperature(pp)
        else:
            raise ValueError('Make sure that luminosity and saturation radius have units associated with them.')

