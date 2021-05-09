"""
This file is the main source file of the ProcessMCRaT library which is used to read and process
the results of a MCRaT simulation

Written by: Tyler Parsotan April 2021

"""
import os
import astropy as ap
import h5py as h5
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.units import UnitConversionError



class PhotonList(object):
    def __init__(self, r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, comv_p0=None, comv_p1=None, comv_p2=None,\
                 comv_p3=None, s0=None, s1=None, s2=None, s3=None, photon_type=None):
        """
        Iniitalizes the 4 momenta (lab and comoving), position, stokes parameters, weight, number of scatterings, and the photon type of each
        photon in the MCRaT file. units are cgs units
        :param r0:
        :param r1:
        :param r2:
        :param s0:
        :param s1:
        :param s2:
        :param s3:
        :param p0:
        :param p1:
        :param p2:
        :param p3:
        :param comv_p0:
        :param comv_p1:
        :param comv_p2:
        :param comv_p3:
        :param weight:
        :param scatterings:
        """
        self.p0=p0
        self.p1=p1
        self.p2=p2
        self.p3=p3
        self.comv_p0=comv_p0
        self.comv_p1=comv_p1
        self.comv_p2=comv_p2
        self.comv_p3=comv_p3
        self.r0=r0
        self.r1=r1
        self.r2=r2
        self.s0=s0
        self.s1=s1
        self.s2=s2
        self.s3=s3
        self.weight=weight
        self.scatterings=scatterings
        self.photon_type=photon_type
        self.file_index=file_index

    def get_energies(self, unit=u.keV):
        try:
            return self.p0 * (const.c.cgs.value * u.erg).to(unit).value
        except UnitConversionError:
            #trying to get wavelength so need to convert to si units for energy first
            x=self.p0 * (const.c.cgs.value * u.erg)
            return x.to(unit, equivalencies=u.spectral()).value

    def get_comv_energies(self, unit=u.keV):
        try:
            return self.comv_p0*(const.c.cgs.value*u.erg).to(unit).value
        except UnitConversionError:
            #trying to get wavelength so need to convert to si units for energy first
            x=self.comv_p0 * (const.c.cgs.value * u.erg)
            return x.to(unit, equivalencies=u.spectral()).value


def curdir():
    """
    Get the current working directory.
	"""
    curdir = os.getcwd() + '/'
    return curdir

class McratSimLoad(object):
    def __init__(self, file_directory=None):
        """
        Initalized the mload class with the directory that the MCRaT files are located in, and the frames per second of
        the simulation (this is found in the MCRaT mc.par file).
        :param file_directory:
        :param frames_per_second:
        """
        if file_directory is not None:
            self.file_directory=file_directory
        else:
            self.file_directory=curdir()

    def load_frame(self, frame_num, read_comv=False, read_stokes=False, read_type=False):
        """
        Reads in MCRaT data for current version of MCRaT that outputs data in hdf5 files. Also has support for various
        MCRaT switches that can be turned on by the user.
        :param frame_num:
        :param read_comv:
        :param read_stokes:
        :param read_type:
        :return:
        """

        with h5.File(self.file_directory+"mcdata_" + np.str_(frame_num) + '.h5', 'r') as f:
            pw = f['PW'][:]
            ns = f['NS'][:]
            p0 = f['P0'][:]
            p1 = f['P1'][:]
            p2 = f['P2'][:]
            p3 = f['P3'][:]
            r0 = f['R0'][:]
            r1 = f['R1'][:]
            r2 = f['R2'][:]
            if read_stokes:
                s0 = f['S0'][:]
                s1 = f['S1'][:]
                s2 = f['S2'][:]
                s3 = f['S3'][:]
            else:
                s0 = np.zeros(pw.size)
                s1 = np.zeros(pw.size)
                s2 = np.zeros(pw.size)
                s3 = np.zeros(pw.size)

            if read_comv:
                comv_p0 = f['COMV_P0'][:]
                comv_p1 = f['COMV_P1'][:]
                comv_p2 = f['COMV_P2'][:]
                comv_p3 = f['COMV_P3'][:]
            else:
                comv_p0 = np.zeros(pw.size)
                comv_p1 = np.zeros(pw.size)
                comv_p2 = np.zeros(pw.size)
                comv_p3 = np.zeros(pw.size)

            if read_type:
                pt = f['PT'][:]
                pt = np.array([i for i in bytes(pt).decode()])
            else:
                pt = np.full(pw.size, None)

        idx=np.arange(pw.size)

        photons=PhotonList(r0, r1, r2, p0, p1, p2, p3, pw, ns, idx, comv_p0=comv_p0, comv_p1=comv_p1,\
                        comv_p2=comv_p2, comv_p3=comv_p3, s0=s0, s1=s1, s2=s2, s3=s3, photon_type=pt)

        self.loaded_photons=photons
        self.read_stokes=read_stokes
        self.read_comv=read_comv
        self.read_type=read_type
        self.frame_num=frame_num
