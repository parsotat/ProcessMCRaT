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
        Iniitalizes the 4 momenta (lab and comoving), position, stokes parameters, weight, number of scatterings,
        and the photon type of each photon in the MCRaT file. Units are cgs units by default.

        :param r0: numpy/astropy quantity array of the photons' x position (in cartestian coordinates)
        :param r1: numpy/astropy quantity array of the photons' y position (in cartestian coordinates)
        :param r2: numpy/astropy quantity array of the photons' z position (in cartestian coordinates)
        :param p0: numpy/astropy quantity array of the photons' four momentum p0
        :param p1: numpy/astropy quantity array of the photons' four momentum p1
        :param p2: numpy/astropy quantity array of the photons' four momentum p2
        :param p3: numpy/astropy quantity array of the photons' four momentum p3
        :param weight: numpy/astropy quantity array of the photons' weights
        :param scatterings: numpy/astropy quantity array of the photons' scatterings that they have undergone
        :param file_index: numpy/astropy quantity of the index of the photons in the mcdata file
        :param comv_p0: Default None or numpy/astropy quantity array of the photons' comoving four momentum p0
        :param comv_p1: Default None or numpy/astropy quantity array of the photons' comoving four momentum p1
        :param comv_p2: Default None or numpy/astropy quantity array of the photons' comoving four momentum p2
        :param comv_p3: Default None or numpy/astropy quantity array of the photons' comoving four momentum p3
        :param s0: Default None or numpy/astropy quantity array of the photons' stokes s0 parameter.
        :param s1: Default None or numpy/astropy quantity array of the photons' stokes s1 parameter.
        :param s2: Default None or numpy/astropy quantity array of the photons' stokes s2 parameter.
        :param s3: Default None or numpy/astropy quantity array of the photons' stokes s3 parameter.
        :param photon_type: Default None or numpy/astropy quantity array of the photons' type (see MCRaT documentation for each photon type)
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
        """
        Gets the energies of the photons

        :param unit: Default is keV can pass in any astropy unit
        :return: array of astropy quantities with the photon energies in the specified unit
        """
        try:
            return self.p0 * (const.c.cgs.value * u.erg).to(unit).value
        except UnitConversionError:
            #trying to get wavelength so need to convert to si units for energy first
            x=self.p0 * (const.c.cgs.value * u.erg)
            return x.to(unit, equivalencies=u.spectral()).value

    def get_comv_energies(self, unit=u.keV):
        """
        Gets the comoving energies of the photons. Only works if there was comving photon energy loaded from the mcdata
        file.

        :param unit: Default is keV can pass in any astropy unit
        :return: array of astropy quantities with the photon energies in the specified unit
        """
        try:
            return self.comv_p0*(const.c.cgs.value*u.erg).to(unit).value
        except UnitConversionError:
            #trying to get wavelength so need to convert to si units for energy first
            x=self.comv_p0 * (const.c.cgs.value * u.erg)
            return x.to(unit, equivalencies=u.spectral()).value

    def get_spherical_coordinates(self, dimensions):
        """
        Converts the photons' cartesian coordinates to either 2D spherical coordinates (by projecting the photons onto a
        x,y grid) or in 3D spherical coordinates.

        :param dimensions: integer that denotes the number of dimensions that the resulting spherical coordinates should
            be calculated with respect to
        :return: numpy/astropy quantity array with photons' r coordinates in same units as r0/r1/r2, and
            numpy/astropy quantity array with photons' theta position in units of radians
        """
        if dimensions==2:
            mc_x_eff = np.sqrt(self.r0 ** 2 + self.r1 ** 2)  # effective x axis for MCRaT simulation to compare to
            # get radial positions of all photons
            R_photon = np.sqrt(mc_x_eff ** 2 + self.r2 ** 2)  # only concerned with z and x plane

            # get angle position of all photons in radians
            Theta_photon = np.arctan2(mc_x_eff, self.r2)
        else:
            raise ValueError("Calculating the photons spherical coordinate in 3D is not yet supported.")

        return R_photon, Theta_photon*u.rad

    def get_cartesian_coordinates(self, dimensions):
        """
        This method returns either the full 3D cartesian coordinate of each photon or it projects the 3D cartesian
        coordinates onto a 2D cartesian grid and returns the (x,y) coordinates

        :param dimensions: number of dimensions that the cartesian coordinates should be calculated for
        :return: tuple of (x,y) or (x,y,z), tuple elements are numpy/astropy quantity arrays with the same units as r0/r1/r2
        """
        if dimensions==2:
            mc_x_eff = np.sqrt(self.r0 ** 2 + self.r1 ** 2)  # effective x axis for MCRaT simulation to compare to
            # get radial positions of all photons
            y= self.r2
            coords=(mc_x_eff, y)
        else:
            coords=(self.r0, self.r1, self.r2)

        return coords



def curdir():
    """
    Get the current working directory.
	"""
    curdir = os.getcwd() + '/'
    return curdir

class McratSimLoad(object):
    def __init__(self, file_directory=None):
        """
        Initalized the mload class with the directory that the MCRaT files are located in. By default it uses the current
        working directory

        :param file_directory: Default is None or string to the direcotry with the mcdata files of interest
        """

        if file_directory is not None:
            self.file_directory=file_directory
        else:
            self.file_directory=curdir()

    def load_frame(self, frame_num, read_comv=False, read_stokes=False, read_type=False):
        """
        Reads in MCRaT data for current version of MCRaT that outputs data in hdf5 files. Also has support for various
        MCRaT switches that can be turned on by the user. If these switches arent on in the MCRaT simulation that is being
        loaded the stokes/comving four momentum/photon type datasets wont exist.

        :param frame_num: Integer for the mcdata frame that the user wants to read in
        :param read_comv: Default False, set to True if the MCRaT simulation was run with comoving parameters saved
        :param read_stokes: Default False, set to True if the MCRaT simulation was run with stokes parameters saved
        :param read_type: Default False, set to True if the MCRaT simulation was run with photon types saved
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
