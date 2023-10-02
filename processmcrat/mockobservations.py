"""
This file contains classes to take the MCRaT output and create mock observed light curves, spectra, and polarizations

Written by: Tyler Parsotan

"""
import numpy as np
import scipy.stats as ss
from scipy.optimize import curve_fit

from processmcrat import PhotonList, McratSimLoad, curdir
from astropy import constants as const
from astropy.stats import bayesian_blocks
from astropy import units as unit
from astropy.units import UnitsError
from astropy.modeling import InputParameterError

from .mclib import *


class ObservedPhotonList(PhotonList):
    """
    Observed photon class that inherits from Photon class. these are the photons that have a calculated time stamp and
    have been detected by at an observer at some observer viewing angle
    :param Photon:
    :return:
    """

    def __init__(self, r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, t_detect, comv_p0=None,
                 comv_p1=None, comv_p2=None, \
                 comv_p3=None, s0=None, s1=None, s2=None, s3=None, photon_type=None):
        """
        Creates a list of photons that have been mock detected by a specified observer.

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
        :param t_detect: numpy/astropy quantity of the times when each photon would be detected by the observer
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
        super().__init__(r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, comv_p0=comv_p0, comv_p1=comv_p1, \
                         comv_p2=comv_p2, comv_p3=comv_p3, s0=s0, s1=s1, s2=s2, s3=s3, photon_type=photon_type)
        self.detection_time = t_detect


class Instrument(object):
    """
    This instrument class allows users to specify various energy ranges for the mock observations that they want to
    calculate
    """

    @unit.quantity_input(spectral_energy_range=['energy','length'], lightcurve_energy_range=['energy','length'], polarization_energy_range=['energy','length'])
    def __init__(self, name='default instrument', spectral_energy_range=None, lightcurve_energy_range=None,\
                 polarization_energy_range=None):
        """
        Create an instrument that 'observes'/fit GRB spectra within an energy range. A separate energy range can be specified
        for the LC measurement and the polarization measurement. Creating an instrument and loading it in a mock observation
        automatically applies all the specified energy limits for the various mock observed quantities.

        :param name: string that denotes the name of the user created instrument
        :param spectral_energy_range: Default None or astropy auantity array with format [min,max]. None means that the
            default values are taken from the mockobservation methods to calculate/fit spectra
        :param lightcurve_energy_range: Default None astropy auantity array with format [min,max]. None means that the
            default values are taken from the mockobservation methods to calculate light curves
        :param polarization_energy_range: Default None astropy auantity array with format [min,max]. None means that the
            default values are taken from the mockobservation methods to calculate polarization degree and angle
        """
        self.name = name
        self.spectral_energy_range = spectral_energy_range
        #self.spectral_energy_unit = spectral_energy_unit
        self.lightcurve_energy_range = lightcurve_energy_range
        #self.lightcurve_energy_unit = lightcurve_energy_unit
        self.polarization_energy_range = polarization_energy_range
        #self.polarization_energy_unit = polarization_energy_unit


class MockObservation(object):
    def __init__(self, theta_observer, acceptancetheta_observer, r_observer, frames_per_sec, hydrosim_dim=2, \
                 phi_observer=0, acceptancephi_observer=0, mcratsimload_obj=None, id=None, directory=None):
        """
        Creates a mock observation for a loaded MCRaT data frame where the user specified information related to the
        location of the observer with respect to the jet axis, the size of the angle bin in which the observer will
        collect photons, the radial location of the observer. The user also needs to specify the hydrodynamic simulation
        parameter of frames per second. This should be identical to what was passed to MCRaT in the mc.par file for the
        MCRaT simulation that is being analyzed.

        The user can use a McratSimLoad object with all the loaded data from a given MCRaT data frame as the starting point
        to conduct a mock observation. It is also possible to load a created event file. This can be done by ignoring
        mcratsimload_obj and just passing in the unique id that was used in a prior call to the save_event_file method
        and also passing in the directory where the event file is located.

        :param theta_observer: observer location with respect to the jet axis, value needs to be in degrees
        :param acceptancetheta_observer: The width of the angle bin for which the observer will collect photons. ie for
            theta_observer=3 degrees and  acceptancetheta_observer=3 degrees, photons will be collected if they are moving
            towards the observer within the angle range of 1.5-4.5 degrees.
        :param r_observer: The radius that the detector is located. Units should be cm.
        :param frames_per_sec: The hydrodynamic simulation's frames per second. Should be identical to what is specified
            in the mc.par file of the MCRaT simulation that is being analyzed.
        :param hydrosim_dim: The number of dimensions of the hydrodynamic simulation that was analyzed with MCRaT. Should be
            identical to what is specified in the mcrat_input.h file of the MCRaT simulation that is being analyzed.
        :param phi_observer: Default 0 degrees. For 3D simulations, this is the azimuthal angle location of the observer
            in degrees
        :param acceptancephi_observer: Default 0 degrees. For 3D simulations, this is the azimuthal angle width of the
            angle bin for which the observer will collect photons
        :param mcratsimload_obj: Default None. The McratSimLoad object that contains the loaded Mcrat data frame that the
            user wants to utilize to conduct a mock observation
        :param id: Default None or string denoting the unique identifier for the simulation that is being analyzed. This
            should be identical to the id passed to a prior call to the save_event_file method.
        :param directory: The location of the event file that the user would like to load
        """



        self.theta_observer = theta_observer
        self.acceptancetheta_observer = acceptancetheta_observer
        if (phi_observer<0.0 or phi_observer>360.0):
            self.phi_observer = (phi_observer+ 360) % 360 #put it in domain of 0-360 degrees
        else:
            self.phi_observer = phi_observer
        self.acceptancephi_observer = acceptancephi_observer
        self.r_observer = r_observer
        self.fps = frames_per_sec
        self.hydrosim_dim = hydrosim_dim
        self.is_instrument_loaded = False
        self.is_set_spectral_fit_parameters = False

        if mcratsimload_obj is not None:
            self.read_stokes = mcratsimload_obj.read_stokes
            self.read_comv = mcratsimload_obj.read_comv
            self.read_type = mcratsimload_obj.read_type
            self.frame_num = mcratsimload_obj.frame_num

            loaded_photons = mcratsimload_obj.loaded_photons

            # test if the photon with the maximum radius is at a larger distance than the observer distance
            r_max=loaded_photons.get_spherical_coordinates(hydrosim_dim)[0].max()

            if r_max>r_observer:
                raise ValueError('The observer needs to be located further than the location of the furthest photon in \
                the simulation which has a radius of %e.'%(r_max))


            """ old way of calculating detection times
            # calculate projection of photons position vector onto vector in the observer's direction
            photon_radius = np.sqrt(loaded_photons.r0 ** 2 + loaded_photons.r1 ** 2 + loaded_photons.r2 ** 2)
            photon_theta_position = np.arctan2(np.sqrt(loaded_photons.r0 ** 2 + loaded_photons.r1 ** 2),
                                               loaded_photons.r2)
            position_theta_relative = photon_theta_position - np.deg2rad(self.theta_observer)
            projected_photon_radius = photon_radius * np.cos(position_theta_relative)

            photon_theta_velocity = np.arctan2(np.sqrt(loaded_photons.p1 ** 2 + loaded_photons.p2 ** 2),
                                               loaded_photons.p3)

            # identify photons moving in direction of observer
            jj = np.where((photon_theta_velocity >= np.deg2rad(self.theta_observer) - np.deg2rad(
                self.acceptancetheta_observer) / 2.) \
                          & (photon_theta_velocity < np.deg2rad(self.theta_observer) + np.deg2rad(
                self.acceptancetheta_observer) / 2.))
            # before had jj = np.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs) )

            self.total_observed_photons = jj[0].size

            # calculate the detection time of each photon based on the frame time, the time for the photon to reach the
            # detector, and the time for a virtual photon to reach the detector at time =0
            frame_time = self.frame_num / self.fps
            dr = projected_photon_radius[jj] - self.r_observer
            projected_velocity = const.c.cgs.value * np.cos(photon_theta_velocity[jj] - np.deg2rad(self.theta_observer))
            photon_travel_time = dr / projected_velocity
            detection_times = np.abs(photon_travel_time) + frame_time - (self.r_observer / const.c.cgs.value)
            """

            #calculate observer coordinates based on inputs from user and set up photon quantities to be in 2D or full 3D
            if hydrosim_dim ==2:
                observer_position = self.r_observer * np.array([np.sin(np.deg2rad(self.theta_observer)), \
                                                                np.cos(np.deg2rad(self.theta_observer))])
                photon_position = np.array([np.sqrt(loaded_photons.r0 ** 2 + loaded_photons.r1 ** 2), loaded_photons.r2])
                photon_velocity = np.array([np.sqrt(loaded_photons.p1 ** 2 + loaded_photons.p2 ** 2), loaded_photons.p3])\
                                  * const.c.cgs.value / loaded_photons.p0
                photon_theta_velocity = np.rad2deg(np.arctan2(photon_velocity[0], photon_velocity[1]))
            else:
                observer_position = self.r_observer * np.array(
                    [np.cos(np.deg2rad(self.phi_observer))*np.sin(np.deg2rad(self.theta_observer)), \
                     np.sin(np.deg2rad(self.phi_observer))*np.sin(np.deg2rad(self.theta_observer)), \
                     np.cos(np.deg2rad(self.theta_observer))])
                photon_position = np.array([loaded_photons.r0, loaded_photons.r1, loaded_photons.r2])
                photon_velocity = np.array([loaded_photons.p1, loaded_photons.p2, loaded_photons.p3]) \
                                  * const.c.cgs.value / loaded_photons.p0
                photon_theta_velocity = np.rad2deg(np.arccos(photon_velocity[2]/np.linalg.norm(photon_velocity, axis=0)))
                #convert arctan angle to degrees from 0-360 degrees for phi
                photon_phi_velocity = (np.rad2deg(np.arctan2(photon_velocity[1], photon_velocity[0])) + 360) % 360

            #identify which photons are moving towards the observer, angles are all in degrees
            jj = np.where((photon_theta_velocity >= self.theta_observer - self.acceptancetheta_observer / 2.) \
                          & (photon_theta_velocity < self.theta_observer + self.acceptancetheta_observer / 2.))[0]
            if self.hydrosim_dim == 3:
                phi_min=(self.phi_observer - self.acceptancephi_observer / 2. + 360) % 360
                phi_max=(self.phi_observer + self.acceptancephi_observer / 2. + 360) % 360
            
                #see if the acceptance phi goes from a negative number to a positive number
                if (phi_min > phi_max):
                    #if it is, make the condiiton be or to collect the values between the two limits which etend past the 0 degree limit
                    kk=np.where((photon_phi_velocity >= phi_min) | (photon_phi_velocity < phi_max))[0]
                else:
                    kk=np.where((photon_phi_velocity >= phi_min) & (photon_phi_velocity < phi_max))[0]
                jj=np.intersect1d(jj,kk) #combine both requirements and get indexes of photons that meet both

            #Calculate the difference between the location of the detector and the photon
            dr = observer_position[:, np.newaxis] - photon_position

            # project photon velocity onto the displacement vector dr
            photon_velocity_project_dr = observer_position[:, np.newaxis] * np.dot(observer_position, photon_velocity) / r_observer ** 2

            #calculate photon travel time as distance divided by speed
            photon_travel_time = np.linalg.norm(dr, axis=0)/np.linalg.norm(photon_velocity_project_dr, axis=0)

            #calculate detection time
            frame_time = self.frame_num / self.fps
            detection_times = np.abs(photon_travel_time) + frame_time - (self.r_observer / const.c.cgs.value)

            #apply condition
            detection_times=detection_times[jj]

            self.total_observed_photons = jj.size

            self.detected_photons = ObservedPhotonList(loaded_photons.r0[jj]*unit.cm, loaded_photons.r1[jj]*unit.cm,
                                                       loaded_photons.r2[jj]*unit.cm, loaded_photons.p0[jj], \
                                                       loaded_photons.p1[jj], loaded_photons.p2[jj],
                                                       loaded_photons.p3[jj], loaded_photons.weight[jj], \
                                                       loaded_photons.scatterings[jj], loaded_photons.file_index[jj],
                                                       detection_times, \
                                                       comv_p0=loaded_photons.comv_p0[jj],
                                                       comv_p1=loaded_photons.comv_p1[jj], \
                                                       comv_p2=loaded_photons.comv_p2[jj],
                                                       comv_p3=loaded_photons.comv_p3[jj], \
                                                       s0=loaded_photons.s0[jj], s1=loaded_photons.s1[jj],
                                                       s2=loaded_photons.s2[jj], \
                                                       s3=loaded_photons.s3[jj],
                                                       photon_type=loaded_photons.photon_type[jj])

        else:
            obs_id = self.create_obs_id(id)
            self.read_event_file(obs_id, directory=directory)

    def create_obs_id(self, id):
        """
        A convenience function to create the observation id which is used to specify the id that the user provides for
        the MCRaT simulation that is being observed, the observer viewing angle, and the detector location

        :param id: string to denote a unique identifier that is used with the location of the observer to create a unique
            event file name
        :return: string of the created event file name
        """
        if self.hydrosim_dim == 2:
            return str(id) + '_' + "%.2e" % self.r_observer + '_' + str(self.theta_observer)
        else:
            return str(id) + '_' + "%.2e" % self.r_observer + '_' + str(self.theta_observer) + '_' + str(
                self.phi_observer)

    def load_instrument(self, instrument_object):
        """
        Function that loads in an Instrument object that defines spectral, light curve, and polarization energy ranges

        :param instrument_object: An Instrument object that has been created with the user specified energy ranges
        :return: None
        """
        self.loaded_instrument_name = instrument_object.name
        if instrument_object.spectral_energy_range is not None:
            self.instrument_spectral_energy_range = instrument_object.spectral_energy_range.value
            #self.instrument_spectral_energy_unit = instrument_object.spectral_energy_unit
            self.instrument_spectral_energy_unit = instrument_object.spectral_energy_range.unit
        else:
            self.instrument_spectral_energy_range = None
            self.instrument_spectral_energy_unit = None


        if instrument_object.lightcurve_energy_range is not None:
            self.instrument_lightcurve_energy_range = instrument_object.lightcurve_energy_range.value
            #self.instrument_lightcurve_energy_unit = instrument_object.lightcurve_energy_unit
            self.instrument_lightcurve_energy_unit = instrument_object.lightcurve_energy_range.unit
        else:
            self.instrument_lightcurve_energy_range = None
            self.instrument_lightcurve_energy_unit = None

        if instrument_object.polarization_energy_range is not None:
            self.instrument_polarization_energy_range = instrument_object.polarization_energy_range.value
            #self.instrument_polarization_energy_unit = instrument_object.polarization_energy_unit
            self.instrument_polarization_energy_unit = instrument_object.polarization_energy_range.unit
        else:
            self.instrument_polarization_energy_range = None
            self.instrument_polarization_energy_unit = None

        self.is_instrument_loaded = True

    def unload_instrument(self):
        """
        Function that unloads a previously loaded instrument and prevents its energy ranges from being used.
        The class then goes back to defaulting to calculating energy integrated quantities, unless energy range is passed
        to each method of the class.
        :return: None
        """
        self.is_instrument_loaded = False
        self.loaded_instrument_name = ''
        self.instrument_spectral_energy_range = None
        self.instrument_spectral_energy_unit = None
        self.instrument_lightcurve_energy_range = None
        self.instrument_lightcurve_energy_unit = None
        self.instrument_polarization_energy_range = None
        self.instrument_polarization_energy_unit = None

    def save_event_file(self, id, save_directory=None, appendfile=False):
        """
        Save observed photons into text file. this allows the user to simply read this file in the future by providing
        the unique id and the directory where the event file is located.

        :param id: string which is the unique identifier that is combined with the observer location to create a unique
            event filename
        :param save_directory: None or string to denote the directory where the event file should be saved. The default
            of None means that the file will be saved in the current working directory
        :param appendfile: Future development! Boolean to denote if the code should append data to this file if it
            already exists.
        :return: None
        """
        obs_id = self.create_obs_id(id)
        file_name = obs_id + '.evt'

        if save_directory is None:
            dir = curdir()
        else:
            dir = save_directory

        outarr = np.zeros([self.total_observed_photons, 20], dtype=object)
        outarr[:, 0] = self.detected_photons.r0
        outarr[:, 1] = self.detected_photons.r1
        outarr[:, 2] = self.detected_photons.r2
        outarr[:, 3] = self.detected_photons.p0
        outarr[:, 4] = self.detected_photons.p1
        outarr[:, 5] = self.detected_photons.p2
        outarr[:, 6] = self.detected_photons.p3
        outarr[:, 7] = self.detected_photons.weight
        outarr[:, 8] = self.detected_photons.scatterings
        outarr[:, 9] = self.detected_photons.file_index
        outarr[:, 10] = self.detected_photons.detection_time

        if self.read_comv:
            outarr[:, 11] = self.detected_photons.comv_p0
            outarr[:, 12] = self.detected_photons.comv_p1
            outarr[:, 13] = self.detected_photons.comv_p2
            outarr[:, 14] = self.detected_photons.comv_p3
        else:
            outarr[:, 11] = np.zeros(self.detected_photons.r0.size)
            outarr[:, 12] = np.zeros(self.detected_photons.r0.size)
            outarr[:, 13] = np.zeros(self.detected_photons.r0.size)
            outarr[:, 14] = np.zeros(self.detected_photons.r0.size)

        if self.read_stokes:
            outarr[:, 15] = self.detected_photons.s0
            outarr[:, 16] = self.detected_photons.s1
            outarr[:, 17] = self.detected_photons.s2
            outarr[:, 18] = self.detected_photons.s3
        else:
            outarr[:, 15] = np.zeros(self.detected_photons.r0.size)
            outarr[:, 16] = np.zeros(self.detected_photons.r0.size)
            outarr[:, 17] = np.zeros(self.detected_photons.r0.size)
            outarr[:, 18] = np.zeros(self.detected_photons.r0.size)

        if self.read_type:
            outarr[:, 19] = self.detected_photons.photon_type
        else:
            outarr[:, 19] = np.zeros(self.detected_photons.r0.size)

        np.savetxt(dir + file_name, outarr, \
                   fmt='%.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %d %d %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %s')

    def read_event_file(self, obs_id, directory=None):
        """
        function to read in an event file

        :param obs_id: string unique event file name for the event file that should be loaded. Should exclude the '.evt' ending.
        :param directory: None or string that denotes the directory that contains the event file of interest. None denotes that the
            file exists in the current working directory.
        :return: None
        """
        if directory is None:
            dir = curdir()
        else:
            dir = directory

        r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, t_detect, comv_p0, comv_p1, comv_p2, \
        comv_p3, s0, s1, s2, s3 = np.loadtxt(dir + obs_id + '.evt', unpack=True, usecols=np.arange(0, 19))
        pt = np.loadtxt(dir + obs_id + '.evt', unpack=True, usecols=[19], dtype='|S15').astype(str)

        if np.sum(comv_p0) == 0:
            self.read_comv = False
        else:
            self.read_comv = True

        if np.sum(s0) == 0:
            self.read_stokes = False
        else:
            self.read_stokes = True

        try:
            if np.int_(pt).sum() == 0:
                self.read_type = False
                pt = np.full(weight.size, None)
        except ValueError:
            self.read_type = True

        self.detected_photons = ObservedPhotonList(r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index,
                                                   t_detect, \
                                                   comv_p0=comv_p0, comv_p1=comv_p1, comv_p2=comv_p2, \
                                                   comv_p3=comv_p3, s0=s0, s1=s1, s2=s2, s3=s3, photon_type=pt)

    def _select_photons(self, times, photon_type=None, energy_range=None, energy_unit=unit.keV, calc_comv=False):
        """
        Applies time cuts to a set of observed photons. Can also apply photon type and energy range cuts. The energy
        range cuts can be based on the lab frame or the comoving frame energies.

        :param times: list or array in the form of [min, max] that denote the time bin edges of interest
        :param photon_type: None or a string denoting the photon type of interest. (See MCRaT documentation for each photon type)
        :param energy_range: list or array in the form of [min, max] that denote the energy bin edges of interest
        :param energy_unit: Default keV or any astropy unit that specified the energy range units.
        :param calc_comv: Default False. Boolean to denote if the energy range cuts should be determined from the lab
            frame energy of the photons or the comoving frame energy of the photons
        :return: array of the indexes of photons that meet the time/type/energy restriction
        """

        if energy_range is not None:
            if calc_comv:
                ph_energy=self.detected_photons.get_comv_energies(unit=energy_unit)
            else:
                ph_energy=self.detected_photons.get_energies(unit=energy_unit)

        if photon_type is None:
            if energy_range is None:
                idx = np.where((self.detected_photons.detection_time >= times[0]) & \
                               (self.detected_photons.detection_time < times[1]) & (
                                   ~np.isnan(self.detected_photons.s0)))
            else:
                idx = np.where((self.detected_photons.detection_time >= times[0]) \
                               & (self.detected_photons.detection_time < times[1]) & (
                                   ~np.isnan(self.detected_photons.s0)) \
                               & (ph_energy >= energy_range[0]) \
                               & (ph_energy < energy_range[1]))
        else:
            if energy_range is None:
                idx = np.where((self.detected_photons.detection_time >= times[0]) \
                               & (self.detected_photons.detection_time < times[1]) & (
                                   ~np.isnan(self.detected_photons.s0)) \
                               & (self.detected_photons.photon_type == photon_type))
            else:
                idx = np.where((self.detected_photons.detection_time >= times[0]) & (
                        self.detected_photons.detection_time < times[1]) \
                               & (~np.isnan(self.detected_photons.s0)) & (
                                       self.detected_photons.photon_type == photon_type) \
                               & (ph_energy >= energy_range[0]) \
                               & (ph_energy < energy_range[1]))
        return idx

    def _time_iterator(self, times, lc_unit, photon_type=None, energy_range=None, energy_unit=unit.keV,
                       fit_spectrum=False, spectrum_energy_range=[10**-7, 10**5], \
                       spectrum_delta_energy=1.25, spectrum_energy_unit=unit.keV, spectral_sample_num=1e3):
        """
        Function to iterate over time bin edges to calculate various parameters of interest as a function of time. The
        time resolved polarization are based on the same cuts used to calculate the light curve.

        :param times: array of time bin edges
        :param lc_unit: astropy unit of the calculated light curve. Can only be erg/s or counts/s.
        :param photon_type: Default None or a string denoting a photon type of interest. (See MCRaT documentation for each photon type)
        :param energy_range: None or array of energy bin edges for the energy range of interest. None means that there
            will be no energy cuts made
        :param energy_unit: Default keV or any astropy units that denote the unit of the energy range that is specified
        :param fit_spectrum: Default False. Boolean to denote if the spectra collected in each energy bin should be fit
            with the Band and cutoff power law (COMP) functions. There can be no photon type cuts made for the spectral fits
        :param spectrum_energy_range: None or array of energy bin edges for the energy range where the fitting will be done.
            None means that there will be no energy cuts made. If there is an instrument loaded then the spectral energy limits
            specified in the instrument will automatically be used.
        :param spectrum_delta_energy: The energy bin width
        :param spectrum_energy_unit: The astropy unit denoting the units of the energy range bin edges
        :param spectral_sample_num: The number of bootstrap samples that should be taken to determine errors on the
            fitted spectral parameters
        :return: numpy arrays for the various mock observable quantities of interest.
        """

        pol_deg = np.empty(times.size) * np.nan
        pol_angle = np.empty(times.size) * np.nan
        stokes_i = np.empty(times.size) * np.nan  # should always be one but keep it as a check of the MCRaT code
        stokes_q = np.empty(times.size) * np.nan
        stokes_u = np.empty(times.size) * np.nan
        stokes_v = np.empty(times.size) * np.nan  # should always be zero, but including it for potential future use
        pol_err = np.empty((times.size, 2)) * np.nan
        lc = np.zeros(times.size)  # *np.nan
        lc_err = np.empty(times.size) * np.nan
        ph_num = np.empty(times.size) * np.nan
        num_scatt = np.zeros(times.size) * np.nan
        fit = np.zeros((times.size, 4)) * np.nan
        fit_error = np.zeros((times.size, 3)) * np.nan
        model_use = np.array([''] * times.size)

        for i in range(times.size - 1):
            # apply various constraints
            idx = self._select_photons([times[i], times[i + 1]], photon_type=photon_type, energy_range=energy_range,
                                       energy_unit=energy_unit)
            if idx[0].size > 0:
                if lc_unit == unit.erg / unit.s:
                    lc[i] = np.sum(self.detected_photons.weight[idx] * \
                                   self.detected_photons.get_energies(unit=unit.erg)[idx]) / (times[i + 1] - times[i])
                elif lc_unit == unit.count / unit.s:
                    lc[i] = np.sum(self.detected_photons.weight[idx]) / (times[i + 1] - times[i])
                else:
                    print('The light curve unit can only be set as erg/s or counts/s currently.')

                # implement same error here as for the spectrum which incl weight variance
                lc_err[i] =  np.sqrt(np.mean(self.detected_photons.weight[idx]**2)/np.mean(self.detected_photons.weight[idx])**2)\
                             *lc[i] / np.sqrt(idx[0].size)
                ph_num[i] = idx[0].size
                num_scatt[i] = np.average(self.detected_photons.scatterings[idx],
                                          weights=self.detected_photons.weight[idx])
                if self.read_stokes:
                    stokes_i[i], stokes_q[i], stokes_u[i], stokes_v[i], pol_deg[i], pol_angle[i], pol_err[i, 0], \
                    pol_err[i, 1] = \
                        self._calc_polarization(self.detected_photons.s0[idx], self.detected_photons.s1[idx], \
                                                self.detected_photons.s2[idx], self.detected_photons.s3[idx],
                                                self.detected_photons.weight[idx])

                if fit_spectrum:
                    print('Fitting between times:', times[i], times[i + 1])
                    spect_dict = self.spectrum(times[i], times[i + 1], spectrum_unit=unit.count / unit.s / unit.keV, \
                                               energy_range=spectrum_energy_range*spectrum_energy_unit,
                                               delta_energy=spectrum_delta_energy*spectrum_energy_unit, \
                                                photon_type=None,
                                               fit_spectrum=fit_spectrum, \
                                               sample_num=spectral_sample_num)  # [11:]
                    fit[i, :] = spect_dict['fit']['alpha'], spect_dict['fit']['beta'], spect_dict['fit'][
                        'break_energy'].value, spect_dict['fit']['normalization']
                    fit_error[i, :] = spect_dict['fit_errors']['alpha_errors'], spect_dict['fit_errors']['beta_errors'], \
                                      spect_dict['fit_errors']['break_energy_errors'].value
                    model_use[i] = spect_dict['model_use']

        return lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_error, model_use

    def _solid_angle(self):
        """
        Calculate the solid angle geometric factor to make the observations isotropic equivalent quantities.

        :return: None
        """
        if self.hydrosim_dim == 2:  # include this factor in the spectra
            factor = 2 * np.pi * (
                    np.cos(np.deg2rad(self.theta_observer - self.acceptancetheta_observer / 2.)) \
                    - np.cos(np.deg2rad(self.theta_observer + self.acceptancetheta_observer / 2.)))
        elif self.hydrosim_dim == 3:
            # this isnt fully implemented yet but is is added on for future support
            factor = np.deg2rad(self.acceptancephi_observer) * (
                        np.cos(np.deg2rad(self.theta_observer - self.acceptancetheta_observer / 2.)) \
                        - np.cos(np.deg2rad(self.theta_observer + self.acceptancetheta_observer / 2.)))

        return factor

    def _lightcurve_calc(self, times, lc_unit, photon_type=None, energy_range=None, energy_unit=unit.keV,
                         fit_spectrum=False, spectral_sample_num=1e4):
        """
        Calculates the light curve mock observation including the geometric factor that makes it an isotropic equvalent
        quantity. The associated time-resolved polarization is also calculated in the same energy range specified. While
        the spectral fitting energy range can be separately specified.

        :param times:  array of time bin edges
        :param lc_unit: astropy unit of the calculated light curve. Can only be erg/s or counts/s.
        :param photon_type: Default None or a string denoting a photon type of interest. (See MCRaT documentation for each photon type)
        :param energy_range: None or array of energy bin edges for the energy range of interest. None means that there
            will be no energy cuts made.
        :param energy_unit: Default keV or any astropy units that denote the unit of the energy range that is specified
        :param fit_spectrum: Default False. Boolean to denote if the spectra collected in each energy bin should be fit
            with the Band and cutoff power law (COMP) functions. There can be no photon type cuts made for the spectral fits.
            If set to True, the default energy range for spectral fitting is from 8-40e3 keV (see
            set_spectral_fit_parameters() method), otherwise an instrument needs to be loaded to change these defaults.
        :param spectrum_energy_range: None or array of energy bin edges for the energy range where the fitting will be done.
        :param spectral_sample_num: The number of bootstrap samples that should be taken to determine errors on the
            fitted spectral parameters
        :return: numpy arrays for the various mock observable quantities of interest.
        """

        lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use = \
            self._time_iterator(times, lc_unit, photon_type=photon_type, energy_range=energy_range,
                                energy_unit=energy_unit, fit_spectrum=fit_spectrum,
                                spectral_sample_num=spectral_sample_num)

        factor = self._solid_angle()
        lc = lc / factor
        lc_err = lc_err / factor

        return lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use

    @unit.quantity_input(energy_range=['energy', 'length'])
    def lightcurve(self, time_start=0, time_end=0, dt=0, time_array=None, lc_unit=unit.erg / unit.s, photon_type=None, \
                   energy_range=None, variable_t_bins=False, fit_spectrum=False,
                   spectral_sample_num=1e4):
        """
        Calculates the light curve and other time resolved parameters such as polarization and time resolved spectral fits.
        The energy range used to calculate the light curve is also used to calculate the time-resolved polarization degrees
        and angles.

        Before conducting time-resolved spectral fitting the user needs to use the set_spectral_fit_parameters() method
        to set the spectral fitting ranges. If an instrument is loaded with spectral fitting vlaues specified then those
        values will be used by default.

        The time-resolved quantities can be calculated using uniform or variable time bins, where a bayesian blocks
        algorithm is used to determine the variable time bin sizes.

        This method can be called by passing in a start time, end time, and dt otherwise the use can pass in an array with
        time bin edges which will be used instead.

        :param time_start: start time of the light curve in seconds
        :param time_end: end time of the light curve in seconds
        :param dt: uniform time bin sizes to be used for the light curve
        :param time_array: numpy array of time bin edges
        :param lc_unit: astropy unit of the calculated light curve. Can only be erg/s (default) or counts/s.
        :param photon_type: Default None or a string denoting a photon type of interest. (See MCRaT documentation for each photon type)
            None denotes that there will be no selection for calculating quantities based on photon type
        :param energy_range: None or astropy quantity array of energy bin edges for the energy range of interest. None means that there
            will be no energy cuts made. If passing in an array the format should be [min, max]
        :param variable_t_bins: Boolean to denote if the bayesian blocks alogithm should be used to determine light curve
            time bin widths
        :param fit_spectrum: Default False. Boolean to denote if the spectra collected in each energy bin should be fit
            with the Band and cutoff power law (COMP) functions. There can be no photon type cuts made for the spectral fits.
            If set to True, the default energy range for spectral fitting is from 8-40e3 keV (see
            set_spectral_fit_parameters() method), otherwise an instrument needs to be loaded to change these defaults.
        :param spectral_sample_num: The number of bootstrap samples that should be taken to determine errors on the
            fitted spectral parameters
        :return: a lightcurve dictionary denoting all the quantities that were able to be calculated. If stokes parameters
            are loaded the dictionary will include polarization quantities automatically. If the user requested that the
            time resolved spectra be fitted then the time-resolved spectra parameters will be included as well.
        """

        if (lc_unit != unit.erg / unit.s) & (lc_unit != unit.count / unit.s):
            raise UnitsError('The light curve unit can only be set as erg/s or counts/s currently.')

        # see if an instrument has been loaded and if the energy range parameter is none, if energy_range!=None then
        # we use the explicitly defined energy_range
        if self.is_instrument_loaded and energy_range is None and self.instrument_lightcurve_energy_range is not None:
            # load the instrumental constraints
            energy_range = self.instrument_lightcurve_energy_range
            energy_unit = self.instrument_lightcurve_energy_unit
        else:
            if energy_range is None:
                energy_unit=unit.keV
            else:
                energy_unit=energy_range.unit
                energy_range=energy_range.value


        if variable_t_bins:
            if (time_start + time_end + dt == 0) and time_array is not None:
                test_times = time_array
            else:
                test_times = np.arange(time_start, time_end, dt)
            data_init = self._lightcurve_calc(test_times, lc_unit, photon_type=photon_type, energy_range=energy_range,
                                              energy_unit=energy_unit)
            times = bayesian_blocks(test_times[~np.isnan(data_init[1])], data_init[0][~np.isnan(data_init[1])],
                                    sigma=data_init[1][~np.isnan(data_init[1])], fitness='measures')

            if (time_start + time_end + dt == 0) and time_array is not None:
                if times[-1] < test_times[-1]:
                    times[-1] = test_times[-1]

                if times[0] > test_times[0]:
                    times[0] = test_times[0]
            else:
                if times[-1] < time_end:
                    times[-1] = time_end

                if times[0] > time_start:
                    times[0] = time_start

        else:
            if (time_start + time_end + dt == 0) and time_array is not None:
                times = time_array
            else:
                times = np.arange(time_start, time_end, dt)

        lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use = \
            self._lightcurve_calc(times, lc_unit, photon_type=photon_type, energy_range=energy_range,
                                  energy_unit=energy_unit, fit_spectrum=fit_spectrum,
                                  spectral_sample_num=spectral_sample_num)

        # return lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use, times
        lc_dict = dict(lightcurve=lc * lc_unit, lightcurve_errors=lc_err * lc_unit,
                       ph_num=ph_num * unit.dimensionless_unscaled, \
                       num_scatt=num_scatt * unit.dimensionless_unscaled, times=times * unit.s, \
                       theta_observer=self.theta_observer * unit.deg)

        if self.read_stokes:
            lc_dict['pol_deg'] = pol_deg * unit.dimensionless_unscaled
            lc_dict['stokes_i'] = stokes_i * unit.dimensionless_unscaled
            lc_dict['stokes_q'] = stokes_q * unit.dimensionless_unscaled
            lc_dict['stokes_u'] = stokes_u * unit.dimensionless_unscaled
            lc_dict['stokes_v'] = stokes_v * unit.dimensionless_unscaled
            lc_dict['pol_angle'] = pol_angle * unit.deg
            lc_dict['pol_deg_errors'] = pol_err[:, 0] * unit.dimensionless_unscaled
            lc_dict['pol_angle_errors'] = pol_err[:, 1] * unit.deg

        if fit_spectrum:
            lc_dict['fit'] = dict(alpha=fit[:, 0], beta=fit[:, 1], break_energy=fit[:, 2] * energy_unit,
                                  normalization=fit[:, 3])
            lc_dict['fit_errors'] = dict(alpha_errors=fit_errors[:, 0], beta_errors=fit_errors[:, 1],
                                         break_energy_errors=fit_errors[:, 2] * energy_unit)
            lc_dict['model_use'] = model_use

        return lc_dict

    def _energy_iterator(self, time_min, time_max, spectrum_unit, energy_min, energy_max, energy_unit=unit.keV,
                         photon_type=None, calc_comv=False):
        """
        Iterates over a range of energies to calculate spectra and energy dependent polarization.
        Normalizes spectrum by the solid angle subtended by the detector to get isotropic equivalent values.

        :param time_min: min time in seconds to get the spectrum for
        :param time_max: max time in seconds to get the spectrum for
        :param spectrum_unit: astropy unit of the constructed spectrum. the units can only be erg/s/energy_unit or
            counts/s/energy_unit as of now.
        :param energy_min: min energy of the constructed spectrum
        :param energy_max: max energy of the constructed spectrum
        :param energy_unit: astropy unit of the energy_min and energy_max parameters
        :param photon_type: Default None or a string denoting a photon type of interest. (See MCRaT documentation for each photon type)
            The spectrum will be constructed for only photons of this type.
        :param calc_comv: Default False. Boolean to denote if the energy range cuts should be determined from the lab
            frame energy of the photons or the comoving frame energy of the photons
        :return: numpy arrays for the various mock observable quantities of interest.
        """
        delta_energy = energy_max - energy_min
        delta_t = time_max - time_min
        spectrum = np.zeros(energy_max.size)
        spectrum_error = np.zeros(energy_max.size)
        pol_deg = np.empty(energy_max.size) * np.nan
        pol_angle = np.empty(energy_max.size) * np.nan
        stokes_i = np.empty(energy_max.size) * np.nan  # should always be one but keep it as a check of the MCRaT code
        stokes_q = np.empty(energy_max.size) * np.nan
        stokes_u = np.empty(energy_max.size) * np.nan
        stokes_v = np.empty(
            energy_max.size) * np.nan  # should always be zero, but including it for potential future use
        pol_err = np.empty((energy_max.size, 2)) * np.nan
        num_scatt = np.zeros(energy_max.size)
        ph_num = np.zeros(energy_max.size)

        for i in range(energy_max.size):
            idx = self._select_photons([time_min, time_max], photon_type=photon_type, \
                                       energy_range=[energy_min[i], energy_max[i]], energy_unit=energy_unit, calc_comv=calc_comv)
            if idx[0].size > 0:
                if 'erg' in spectrum_unit.to_string():
                    if not calc_comv:
                        spectrum[i] = np.sum(self.detected_photons.weight[idx] * \
                                             self.detected_photons.get_energies(unit=unit.erg)[idx]) / delta_energy[i] / (
                                          delta_t)
                    else:
                        spectrum[i] = np.sum(self.detected_photons.weight[idx] * \
                                             self.detected_photons.get_comv_energies(unit=unit.erg)[idx]) / delta_energy[i] / (
                                          delta_t)
                elif 'ct' in spectrum_unit.to_string():
                    spectrum[i] = np.sum(self.detected_photons.weight[idx]) / delta_energy[i] / (delta_t)
                else:
                    print('The spectrum unit can only be set as erg/s/energy_unit or counts/s/energy_unit currently.')

                spectrum_error[i] = np.sqrt(
                    np.mean(self.detected_photons.weight[idx] ** 2) / np.mean(self.detected_photons.weight[idx]) ** 2) * \
                                    spectrum[i] / np.sqrt(idx[0].size)  # spectrum[i] / np.sqrt(idx[0].size) see https://arxiv.org/pdf/0909.0708.pdf for variance calculation
                ph_num[i] = idx[0].size
                num_scatt[i] = np.average(self.detected_photons.scatterings[idx],
                                          weights=self.detected_photons.weight[idx])

                if self.read_stokes:
                    stokes_i[i], stokes_q[i], stokes_u[i], stokes_v[i], pol_deg[i], pol_angle[i], pol_err[i, 0], \
                    pol_err[i, 1] = \
                        self._calc_polarization(self.detected_photons.s0[idx], self.detected_photons.s1[idx], \
                                                self.detected_photons.s2[idx], self.detected_photons.s3[idx], \
                                                self.detected_photons.weight[idx])

        factor = self._solid_angle()
        spectrum = spectrum/factor
        spectrum_error = spectrum_error/factor

        return spectrum, spectrum_error, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err

    @unit.quantity_input(energy_range=['energy', 'length'], delta_energy=['energy', 'length'])
    def spectrum(self, time_start, time_end, spectrum_unit=unit.erg / unit.s / unit.keV, energy_range=[10**-7, 10**5]*unit.keV, \
                 delta_energy=10**(0.1)*unit.keV, photon_type=None, fit_spectrum=False, sample_num=1e4, calc_comv=False):
        """
        Calculates the spectrum and other energy-resolved parameters such as polarization for a set time interval.

        This method can also fit the specrum but the details of the spectral fit need to be set before the fitting can
        be done in this function. This is accomplished by calling the set_spectral_fit_parameters() method. See the
        docstring for this method to see what the default values are.

        :param time_start: start time in seconds to start accumulating photons to calculate the spectrum
        :param time_end: end time in seconds to stop accumulating photons to calculate the spectrum
        :param spectrum_unit: Default unit.erg / unit.s / unit.keV or astropy unit of the constructed spectrum.
            The units can only be erg/s/energy_unit or counts/s/energy_unit as of now.
        :param energy_range: Default [10**-7, 10**5]*unit.keV or astropy quantity array of energy bin edges for the
            energy range of interest. The format of the array should be [min, max]. This will be the min energy range of the spectrum
            and the max energy of the spectrum. Photons with energies outside of this range will be ignored.
        :param delta_energy: Default 10**(0.1)*unit.keV or astropy quantity object. The width of the spectral energy bins for the spectra. Needs to be
            the same units as the energy_range values.
        :param photon_type: Default None or a string denoting a photon type of interest. (See MCRaT documentation for each photon type)
            The spectrum will be constructed for only photons of this type.
        :param fit_spectrum: Default False. Boolean to denote if the spectra collected in each energy bin should be fit
            with the Band and cutoff power law (COMP) functions. There can be no photon type cuts made for the spectral fits.
            If set to True, the default energy range for spectral fitting is from 8-40e3 keV (see
            set_spectral_fit_parameters() method), otherwise an instrument needs to be loaded to change these defaults.
        :param sample_num: Default 1e4. The number of bootstrap samples that should be taken to determine errors on the
            fitted spectral parameters
        :param calc_comv: Default False. Boolean to denote if the energy range cuts should be determined from the lab
            frame energy of the photons or the comoving frame energy of the photons
        :return: A dicgionary that holds all the energy dependednt quantities that were calculated for the spectrum
        """
        if ('erg' not in spectrum_unit.to_string()) & ('ct' not in spectrum_unit.to_string()):
            raise UnitsError(
                'The spectrum unit can only be set as erg/s/energy_unit or counts/s/energy_unit currently.')

        if energy_range.unit != delta_energy.unit:
            raise UnitsError('The units of the energy range and the energy bin sizes have to match.')

        if (np.sum(self.detected_photons.comv_p0) == 0) and calc_comv:
            raise InputParameterError('The comoving photon data has not been loaded to produce a comoving spectrum.')

        # see if an instrument has been loaded and if the energy range parameter is none, if energy_range!=None then
        # we use the explicitly defined energy_range
        if self.is_instrument_loaded and (
                np.equal([10**-7, 10**5], energy_range.value).sum() == 2) and self.instrument_spectral_energy_range is not None:
            # load the instrumental constraints
            log_energy_range = np.log10(self.instrument_spectral_energy_range)
            energy_unit = self.instrument_spectral_energy_unit
        else:
            log_energy_range = np.log10(energy_range.value)
            energy_unit = energy_range.unit

        delta_log_energy=np.log10(delta_energy.value)

        energy_min = 10 ** np.arange(log_energy_range[0], log_energy_range[1], delta_log_energy)
        energy_max = energy_min * 10 ** delta_log_energy
        energy_bin_center = np.sqrt(energy_min * energy_max)

        spectrum, spectrum_error, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err = \
            self._energy_iterator(time_start, time_end, spectrum_unit, energy_min, energy_max, energy_unit=energy_unit,
                                  photon_type=photon_type, calc_comv=calc_comv)

        fit, fit_errors, model_use = np.zeros(4) * np.nan, np.zeros(3) * np.nan, ''
        if fit_spectrum and ('ct' in spectrum_unit.to_string()):
            # break energy in fit always has same units as the energy bins
            fit, fit_errors, model_use = self.spectral_fit(spectrum, spectrum_error, ph_num, energy_bin_center*energy_unit,
                                                            sample_num=sample_num)
        else:
            if fit_spectrum:
                raise ValueError('The units of the spectrum needs to be counts/s/energy_unit to fit the spectrum.')

        # return spectrum, spectrum_error, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use, energy_bin_center
        spec_dict = dict(spectrum=spectrum * spectrum_unit, spectrum_errors=spectrum_error * spectrum_unit,
                         ph_num=ph_num * unit.dimensionless_unscaled, \
                         num_scatt=num_scatt * unit.dimensionless_unscaled,
                         energy_bin_center=energy_bin_center * energy_unit, \
                         theta_observer=self.theta_observer * unit.deg)


        if self.read_stokes:
            spec_dict['pol_deg'] = pol_deg * unit.dimensionless_unscaled
            spec_dict['stokes_i'] = stokes_i * unit.dimensionless_unscaled
            spec_dict['stokes_q'] = stokes_q * unit.dimensionless_unscaled
            spec_dict['stokes_u'] = stokes_u * unit.dimensionless_unscaled
            spec_dict['stokes_v'] = stokes_v * unit.dimensionless_unscaled
            spec_dict['pol_angle'] = pol_angle * unit.deg
            spec_dict['pol_deg_errors'] = pol_err[:, 0] * unit.dimensionless_unscaled
            spec_dict['pol_angle_errors'] = pol_err[:, 1] * unit.deg

        if fit_spectrum and ('ct' in spectrum_unit.to_string()):
            spec_dict['fit'] = dict(alpha=fit[0], beta=fit[1], break_energy=fit[2] * energy_unit, normalization=fit[3])
            spec_dict['fit_errors'] = dict(alpha_errors=fit_errors[0], beta_errors=fit_errors[1],
                                           break_energy_errors=fit_errors[2] * energy_unit)
            spec_dict['model_use'] = model_use

        return spec_dict

    def _calc_polarization(self, s0, s1, s2, s3, weights, mu=1):
        """
        function used to calculate the polarization and its error for a given set of mock observed photons following:
        	Kislat, F., Clark, B., Beilicke, M., & Krawczynski, H. 2015, Astroparticle Physics, 68, 45
    	In particular, we follow their appendix where photons have differing weights.

        :param s0: array of photons' s0
        :param s1: array of photons' s1
        :param s2: array of photons' s2
        :param s3: array of photons' s3
        :param weights: array of photons' weights
        :param mu: Default 1, this is indicative of how perfectly a detector can detect no polarization in sources where
            1 i perfect.
        :return: arrays of multiple polarization quantities
        """

        # calc the q and u normalized by I
        I = np.sum(weights)  # np.mean(weights) #np.sum(weights)
        i = np.average(s0, weights=weights)
        q = np.average(s1, weights=weights)
        u = np.average(s2, weights=weights)
        v = np.average(s3, weights=weights)
        p = np.sqrt(q ** 2 + u ** 2)
        chi = np.rad2deg(0.5 * np.arctan2(u, q))
        W_2 = np.sum(weights ** 2)  # np.mean(weights**2) #np.sum(weights**2)

        mu_factor = 2 / mu
        var_factor = W_2 / I ** 2

        # convert q and u to reconstructed values that kislat uses
        Q_r = mu_factor * q
        U_r = mu_factor * u
        p_r = mu_factor * p

        # calculate the standard deviation in Q_R and U_r and covariance
        sigma_Q_r = np.sqrt(var_factor * (mu_factor / mu - Q_r ** 2))
        sigma_U_r = np.sqrt(var_factor * (mu_factor / mu - U_r ** 2))
        if (np.isnan(sigma_Q_r)):
            # in case there is some error with the argument of sqrt being negative (happens rarely and am not sure why)
            sigma_Q_r = np.sqrt(var_factor * np.abs(mu_factor / mu - Q_r ** 2))
        if (np.isnan(sigma_U_r)):
            sigma_U_r = np.sqrt(var_factor * np.abs(mu_factor / mu - U_r ** 2))

        cov = -var_factor * Q_r * U_r
        # print('var factor', var_factor, 'W_2', W_2, 'I', I,  'mean value of W_2', np.mean(weights**2), 'mean value of I', np.mean(weights), 'leads to', np.mean(weights**2)/np.mean(weights)**2)
        # print(Q_r, U_r, sigma_U_r, sigma_Q_r, cov)
        # calculate the partial derivatives
        partial_pr_Qr = Q_r / p_r
        partial_pr_Ur = U_r / p_r
        partial_phir_Qr = -0.5 * u / p ** 2 / mu_factor  # dq/dQ_r=2/mu, and do (d phi/dq)*(dq/dQ_r)
        partial_phir_Ur = -0.5 * q / p ** 2 / mu_factor

        # calculate the error in pr and chi
        sigma_pr = np.sqrt((partial_pr_Qr * sigma_Q_r) ** 2 + (
                partial_pr_Ur * sigma_U_r) ** 2 + 2 * partial_pr_Qr * partial_pr_Ur * cov)
        sigma_chi = np.sqrt((partial_phir_Qr * sigma_Q_r) ** 2 + (
                partial_phir_Ur * sigma_U_r) ** 2 + 2 * partial_phir_Qr * partial_phir_Ur * cov)
        if (np.isnan(sigma_pr)):
            sigma_pr = np.sqrt(np.abs((partial_pr_Qr * sigma_Q_r) ** 2 + (
                    partial_pr_Ur * sigma_U_r) ** 2 + 2 * partial_pr_Qr * partial_pr_Ur * cov))

        return i, q, u, v, p, chi, sigma_pr / mu_factor, np.rad2deg(sigma_chi)

    @unit.quantity_input(energy_range=['energy', 'length'])
    def polarization(self, time_start, time_end, photon_type=None, energy_range=None):
        """
        Calculates the polarization degree, angle, summed stokes parameters and the errors of a set of photons detected
        in some time interval from time_start to time_end. The set of photons can be further selected by photon type
        and/or energy range.

        :param time_start: start time in seconds to start accumulating photons to calculate the polarization quantities
        :param time_end: end time in seconds to stop accumulating photons to calculate the polarization quantities
        :param photon_type: Default None or a string denoting a photon type of interest. (See MCRaT documentation for each photon type)
            The spectrum will be constructed for only photons of this type.
        :param energy_range: None or astropy quantity array of energy bin edges for the energy range of interest. None means that there
            will be no energy cuts made. If passing in an array the format should be [min, max].
        :return: A dictionary that contains all the relevant polarization parameters and errors
        """
        if self.read_stokes:
            # see if an instrument has been loaded and if the energy range parameter is none, if energy_range!=None then
            # we use the explicitly defined energy_range
            if self.is_instrument_loaded and energy_range is None and self.instrument_polarization_energy_range is not None:
                # load the instrumental constraints
                energy_range = self.instrument_polarization_energy_range
                energy_unit = self.instrument_polarization_energy_unit
            else:
                if energy_range is None:
                    energy_unit=None
                else:
                    energy_unit=energy_range.unit
                    energy_range=energy_range.value


            idx = self._select_photons([time_start, time_end], photon_type=photon_type, energy_range=energy_range, \
                                       energy_unit=energy_unit)

            pol_err = np.empty((1, 2)) * np.nan

            stokes_i, stokes_q, stokes_u, stokes_v, pol_deg, pol_angle, pol_deg_err, pol_angle_err = \
                self._calc_polarization(self.detected_photons.s0[idx], self.detected_photons.s1[idx], \
                                        self.detected_photons.s2[idx], self.detected_photons.s3[idx], \
                                        self.detected_photons.weight[idx])

            pol_dict = dict(stokes_i=stokes_i * unit.dimensionless_unscaled,
                            stokes_q=stokes_q * unit.dimensionless_unscaled, \
                            stokes_u=stokes_u * unit.dimensionless_unscaled,
                            stokes_v=stokes_v * unit.dimensionless_unscaled, \
                            pol_deg=pol_deg * unit.dimensionless_unscaled, pol_angle=pol_angle * unit.deg, \
                            pol_deg_errors=pol_deg_err * unit.dimensionless_unscaled,
                            pol_angle_errors=pol_angle_err * unit.deg, \
                            theta_observer=self.theta_observer * unit.deg)

            # return stokes_i, stokes_q, stokes_u, stokes_v, pol_deg, pol_angle, pol_err
            return pol_dict
        else:
            raise ValueError('The stokes parameters have not been loaded and polarization cannot be calculated.')

    @unit.quantity_input(spectral_fit_energy_range=['energy', 'length'])
    def set_spectral_fit_parameters(self, spectral_fit_energy_range=[8, 40e3]*unit.keV, \
                                    approx_gaussian_error_num=10):
        """
        Sets the spectral parameters for energy ranges that should be fitted with the spectral_fit method.
        This needs to be set before attempting to do any spectral fits for the class, either in the lightcurve method
        or the spectrum method.

        If an instrument is loaded, calling this method with no arguments will make the instrument spectral fit parameters
        the defualt.

        :param spectral_fit_energy_range: Default [8, 40e3]*unit.keV or an array with astropy quantities. Denotes the min
            and max energy for which the Band or cutoff powerlaw (COMP) functions should be fit to the spectrum.
        :param approx_gaussian_error_num: Default 10. The number of photons in a spectral energy bin which allows the code to
            approximate the errors as gaussian.
        :return: None
        """
        #see if an instrument is loaded with spectral parameters set use them
        if self.is_instrument_loaded and self.instrument_spectral_energy_range is not None:
            self.spectral_fit_energy_range = self.instrument_spectral_energy_range
            self.spectral_fit_energy_unit = self.instrument_spectral_energy_unit
        else:
            self.spectral_fit_energy_range = spectral_fit_energy_range.value
            self.spectral_fit_energy_unit = spectral_fit_energy_range.unit
        self.approx_gaussian_error_num = approx_gaussian_error_num
        self.is_set_spectral_fit_parameters = True

    def unset_spectral_fit_parameters(self):
        """
        Unloads the specified spectral fit parameters, replacing them with the defaults.
        :return: None
        """
        self.spectral_fit_energy_range = [8, 40e3]
        self.spectral_fit_energy_unit = unit.keV
        self.approx_gaussian_error_num = 10
        self.is_set_spectral_fit_parameters = False

    @unit.quantity_input(energy_bin_center=['energy', 'length'])
    def spectral_fit(self, spectrum, spectrum_error, ph_num, energy_bin_center, sample_num=1e4):
        """
        The function that takes a spectrum and fits it with either a cutoff powerlaw (COMP) or Band function in the
        specified energy range provided by the user through the set_spectral_fit_parameters() method.

        :param spectrum: array of spectral values (should be in units of counts/s/energy_unit)
        :param spectrum_error: The error in each spectral data point
        :param ph_num: The number of photons in each energy bin
        :param energy_bin_center: The center of each energy bin of the spectrum
        :param sample_num: Default 1e4. The number of bootstrap samples that should be taken to determine errors on the
            fitted spectral parameters
        :return: The best fitting spectral parameters and their errors and the type of function that provided the best fit.
            B for Band or C for Comp.
        """

        if not self.is_set_spectral_fit_parameters:
            raise InputParameterError(
                'The set_spectral_fit_parameters method needs to be called before spectral fitting can be done.')

        energy_bin_center = energy_bin_center.to(self.spectral_fit_energy_unit).value

        idx = np.where((energy_bin_center > self.spectral_fit_energy_range[0]) & \
                       (energy_bin_center <= self.spectral_fit_energy_range[1]) & (
                                   ph_num > self.approx_gaussian_error_num))

        if idx[0].size > 4:
            spectrum = spectrum[idx]
            spectrum_error = spectrum_error[idx]
            energy_bin_center = energy_bin_center[idx]
            normalization = np.trapz(spectrum, x=energy_bin_center)
            data_pts = spectrum.size

            # test the Band function fit first and make sure that the fitted energy cant be < 0
            band_fit, band_matrice = curve_fit(band_function, energy_bin_center, spectrum, sigma=spectrum_error, \
                                               p0=[.3, -5, 100, normalization], maxfev=10000)
            if (band_fit[2] < 0):
                band_fit, band_matrice = curve_fit(band_function, energy_bin_center, spectrum, sigma=spectrum_error, \
                                                   p0=[.3, -5, 100, normalization], maxfev=10000,
                                                   bounds=([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
            # calculate chi squared
            band_chi_sq = ((band_function(energy_bin_center, band_fit[0], band_fit[1], band_fit[2],
                                          band_fit[3]) - spectrum) ** 2 / spectrum_error ** 2).sum()

            # now test the comp function and make sure that the fitted energy cant be < 0
            comp_fit, comp_matrice = curve_fit(comptonized_function, energy_bin_center, spectrum, sigma=spectrum_error, \
                                               p0=[.3, 100, normalization])
            if (comp_fit[1] < 0):
                comp_fit, comp_matrice = curve_fit(comptonized_function, energy_bin_center, spectrum,
                                                   sigma=spectrum_error, \
                                                   p0=[.3, 100, normalization],
                                                   bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]))

            # calcualate chi squared
            comp_chi_sq = ((comptonized_function(energy_bin_center, comp_fit[0], comp_fit[1],
                                                 comp_fit[2]) - spectrum) ** 2 / spectrum_error ** 2).sum()

            # test for chi sq equivalence
            if band_chi_sq != comp_chi_sq:
                # do F test
                dof_c = data_pts - 2 - 1
                dof_b = data_pts - 3 - 1
                SS_c = ((comptonized_function(energy_bin_center, comp_fit[0], comp_fit[1],
                                              comp_fit[2]) - spectrum) ** 2).sum()
                SS_b = ((band_function(energy_bin_center, band_fit[0], band_fit[1], band_fit[2],
                                       band_fit[3]) - spectrum) ** 2).sum()

                alpha = 0.05  # false positive acceptance rate 5%
                F = ((comp_chi_sq - band_chi_sq) / (dof_c - dof_b)) / (band_chi_sq / dof_b)
                p = 1 - ss.f.cdf(F, (dof_c - dof_b), dof_b)
                # print(p, F)
                if (p < alpha):
                    model_use = 'b'
                    # print('Using The band function')
                else:
                    model_use = 'c'
                    # print('Using the Comp function')
            else:
                # if the two chi squares are equal have to choose the simpler model aka comp
                model_use = 'c'

            best_fit = np.zeros(4)
            best_fit_errors = np.zeros(3)  # have three elements for alpha, beta, e_0
            is_large_error = False  # parameter to determine if errors are extremely large
            # get errors in parameters
            try:
                if model_use == 'c':
                    avg_fit, fit_errors = bootstrap_parameters(energy_bin_center, spectrum, spectrum_error,
                                                               comptonized_function, \
                                                               comp_fit, sample_num=sample_num)
                    best_fit[0] = comp_fit[0]
                    best_fit[1] = np.nan  # the comp function has to beta parameter
                    best_fit[2] = comp_fit[1]
                    best_fit[3] = comp_fit[2]

                    best_fit_errors[0] = fit_errors[0]
                    best_fit_errors[1] = np.nan
                    best_fit_errors[2] = fit_errors[1]

                    for i in range(fit_errors.size):
                        if np.abs(fit_errors[i] + comp_fit[i]) > np.ceil(np.abs(3 * comp_fit[i])):
                            is_large_error = True

                else:
                    avg_fit, fit_errors = bootstrap_parameters(energy_bin_center, spectrum, spectrum_error,
                                                               band_function, \
                                                               band_fit, sample_num=sample_num)
                    best_fit = band_fit
                    best_fit_errors = fit_errors[:-1]

                    for i in range(best_fit_errors.size):
                        if np.abs(best_fit_errors[i] + best_fit[i]) > np.ceil(np.abs(3 * best_fit[i])):
                            is_large_error = True

                # print(best_fit, fit_errors, model_use) #can get really large errors so in processing output need to deal with that

                # if we have large errors or if the error is 0 (unconstrained parameters) set things to be null
                if (np.where(best_fit_errors == 0)[0].size > 0) or (is_large_error):
                    best_fit, best_fit_errors, model_use = np.zeros(4) * np.nan, np.zeros(3) * np.nan, ''

            except RuntimeError:
                # errors couldnt converge therefore set fit and errors to be 0
                best_fit, best_fit_errors, model_use = np.zeros(4) * np.nan, np.zeros(3) * np.nan, ''
        else:
            best_fit, best_fit_errors, model_use = np.zeros(4) * np.nan, np.zeros(3) * np.nan, ''

        return best_fit, best_fit_errors, model_use
