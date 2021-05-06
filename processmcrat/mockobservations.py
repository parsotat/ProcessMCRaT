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
    def __init__(self, r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, t_detect, comv_p0=None, comv_p1=None, comv_p2=None,\
                 comv_p3=None, s0=None, s1=None, s2=None, s3=None, photon_type=None):
        super().__init__(r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, comv_p0=comv_p0, comv_p1=comv_p1,\
                        comv_p2=comv_p2, comv_p3=comv_p3, s0=s0, s1=s1, s2=s2, s3=s3, photon_type=photon_type)
        self.detection_time=t_detect

class Instrument(object):
    """
    This instrument class allows users to specify various energy ranges for the mock observations that they want to
    calculate
    """
    def __init__(self, name='default instrument', spectral_energy_range=None, spectral_energy_unit=None,\
                 lightcurve_energy_range=None, lightcurve_energy_unit=None,\
                 polarization_energy_range=None, polarization_energy_unit=None):

        self.name=name
        self.spectral_energy_range=spectral_energy_range
        self.spectral_energy_unit=spectral_energy_unit
        self.lightcurve_energy_range=lightcurve_energy_range
        self.lightcurve_energy_unit=lightcurve_energy_unit
        self.polarization_energy_range=polarization_energy_range
        self.polarization_energy_unit=polarization_energy_unit



class MockObservation(object):
    def __init__(self, theta_observer, acceptancetheta_observer, r_observer, frames_per_sec, hydrosim_dim=2, mcratsimload_obj=None, id=None, directory=None):
        """
        Sets up the MockObservation object that allows the user to create mock observations. There are two ways to
        initialize this: 1) using a McratSimLoad object or 2) reading in a previously created event file

        :param theta_observer:
        :param acceptancetheta_observer:
        :param r_observer:
        :param frames_per_sec:
        :param mcratsimload_obj:
        :param id:
        :param directory:
        """

        self.theta_observer = theta_observer
        self.acceptancetheta_observer = acceptancetheta_observer
        self.r_observer = r_observer
        self.fps = frames_per_sec
        self.hydrosim_dim=hydrosim_dim
        self.is_instrument_loaded=False
        self.is_set_spectral_fit_parameters = False

        if mcratsimload_obj is not None:
            self.read_stokes=mcratsimload_obj.read_stokes
            self.read_comv=mcratsimload_obj.read_comv
            self.read_type=mcratsimload_obj.read_type
            self.frame_num=mcratsimload_obj.frame_num

            loaded_photons=mcratsimload_obj.loaded_photons

            #calculate projection of photons position vector onto vector in the observer's direction
            photon_radius=np.sqrt(loaded_photons.r0**2 + loaded_photons.r1**2 + loaded_photons.r2**2)
            photon_theta_position=np.arctan2(np.sqrt(loaded_photons.r0 ** 2 + loaded_photons.r1 ** 2), loaded_photons.r2)
            position_theta_relative = photon_theta_position - np.deg2rad(self.theta_observer)
            projected_photon_radius=photon_radius*np.cos(position_theta_relative)

            photon_theta_velocity=np.arctan2(np.sqrt(loaded_photons.p1 ** 2 + loaded_photons.p2 ** 2), loaded_photons.p3)

            #identify photons moving in direction of observer
            jj=np.where((photon_theta_velocity >= np.deg2rad(self.theta_observer) - np.deg2rad(self.acceptancetheta_observer) / 2.)\
                        & (photon_theta_velocity < np.deg2rad(self.theta_observer) + np.deg2rad(self.acceptancetheta_observer) / 2.))
            #before had jj = np.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs) )

            self.total_observed_photons=jj[0].size

            #calculate the detection time of each photon based on the frame time, the time for the photon to reach the
            # detector, and the time for a virtual photon to reach the detector at time =0
            frame_time=self.frame_num/self.fps
            dr=projected_photon_radius[jj] - self.r_observer
            projected_velocity=const.c.cgs.value*np.cos(photon_theta_velocity[jj]-np.deg2rad(self.theta_observer))
            photon_travel_time=dr/projected_velocity
            detection_times=np.abs(photon_travel_time)+frame_time-(self.r_observer/const.c.cgs.value)


            self.detected_photons=ObservedPhotonList(loaded_photons.r0[jj], loaded_photons.r1[jj], loaded_photons.r2[jj], loaded_photons.p0[jj],\
                                                loaded_photons.p1[jj], loaded_photons.p2[jj], loaded_photons.p3[jj], loaded_photons.weight[jj],\
                                                loaded_photons.scatterings[jj], loaded_photons.file_index[jj], detection_times, \
                                                comv_p0=loaded_photons.comv_p0[jj],comv_p1=loaded_photons.comv_p1[jj],\
                                                comv_p2=loaded_photons.comv_p2[jj], comv_p3=loaded_photons.comv_p3[jj],\
                                                s0=loaded_photons.s0[jj], s1=loaded_photons.s1[jj], s2=loaded_photons.s2[jj],\
                                                s3=loaded_photons.s3[jj], photon_type=loaded_photons.photon_type[jj])

        else:
            obs_id=self.create_obs_id(id)
            self.read_event_file(obs_id, directory=directory)

    def create_obs_id(self, id):
        """
        A convenience function to create the observation id which is used to specify the id that the user provides for
        the MCRaT simulation that is being observed, the observer viewing angle, and the detector location
        :param id:
        :return:
        """
        return np.str(id) + '_' + "%.2e" % self.r_observer + '_' + np.str(self.theta_observer)

    def load_instrument(self, instrument_object):
        """
        Function that loads in an Instrument object that defines spectral, light curve, and polarization energy ranges
        :param instrument_object:
        :return:
        """
        self.loaded_instrument_name=instrument_object.name
        self.instrument_spectral_energy_range = instrument_object.spectral_energy_range
        self.instrument_spectral_energy_unit = instrument_object.spectral_energy_unit
        self.instrument_lightcurve_energy_range = instrument_object.lightcurve_energy_range
        self.instrument_lightcurve_energy_unit = instrument_object.lightcurve_energy_unit
        self.instrument_polarization_energy_range = instrument_object.polarization_energy_range
        self.instrument_polarization_energy_unit = instrument_object.polarization_energy_unit
        self.is_instrument_loaded = True

    def unload_instrument(self):
        """
        Function that unloads a previously loaded instrument and prevents its energy ranges from being used.
        The class then goes back to defaulting to calculating energy integrated quantities, unless energy range is passed
        to each method of the class.
        :return:
        """
        self.is_instrument_loaded = False
        self.loaded_instrument_name=''
        self.instrument_spectral_energy_range = None
        self.instrument_spectral_energy_unit = None
        self.instrument_lightcurve_energy_range = None
        self.instrument_lightcurve_energy_unit = None
        self.instrument_polarization_energy_range = None
        self.instrument_polarization_energy_unit = None


    def save_event_file(self, id, save_directory=None, appendfile=False):
        """
        function to save observed photons into text file
        :param id:
        :param save_directory:
        :param appendfile:
        :return:
        """
        obs_id=self.create_obs_id(id)
        file_name=obs_id +'.evt'

        if save_directory is None:
            dir=curdir()
        else:
            dir=save_directory

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

        np.savetxt(dir+file_name,outarr,\
                   fmt='%.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %d %d %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %s')


    def read_event_file(self, obs_id, directory=None):
        """
        function to read in an event file
        :param obs_id:
        :param directory:
        :return:
        """
        if directory is None:
            dir=curdir()
        else:
            dir=directory

        r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, t_detect, comv_p0, comv_p1, comv_p2,\
                 comv_p3, s0, s1, s2, s3 =np.loadtxt(dir + obs_id +'.evt', unpack=True, usecols=np.arange(0,19))
        pt = np.loadtxt(dir + obs_id +'.evt', unpack=True, usecols=[19], dtype='|S15').astype(str)

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

        self.detected_photons = ObservedPhotonList(r0, r1, r2, p0, p1, p2, p3, weight, scatterings, file_index, t_detect, \
                                                   comv_p0=comv_p0, comv_p1=comv_p1, comv_p2=comv_p2, \
                                                   comv_p3=comv_p3, s0=s0, s1=s1, s2=s2, s3=s3, photon_type=pt)

    def _select_photons(self, times, photon_type=None, energy_range=None, energy_unit=unit.keV):
        """

        :param times:
        :param photon_type:
        :param energy_range:
        :param energy_unit:
        :return:
        """
        if photon_type is None:
            if energy_range is None:
                idx = np.where((self.detected_photons.detection_time >= times[0]) & \
                               (self.detected_photons.detection_time < times[1]) & (
                                   ~np.isnan(self.detected_photons.s0)))
            else:
                idx = np.where((self.detected_photons.detection_time >= times[0]) \
                               & (self.detected_photons.detection_time < times[1]) & (
                                   ~np.isnan(self.detected_photons.s0)) \
                               & (self.detected_photons.get_energies(unit=energy_unit) >= energy_range[0]) \
                               & (self.detected_photons.get_energies(unit=energy_unit) < energy_range[1]))
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
                               & (self.detected_photons.get_energies(unit=energy_unit) >= energy_range[0]) \
                               & (self.detected_photons.get_energies(unit=energy_unit) < energy_range[1]))
        return idx

    def _time_iterator(self, times, lc_unit, photon_type=None, energy_range=None, energy_unit=unit.keV, fit_spectrum=False, spectrum_log_energy_range=[-7, 5],\
                 spectrum_delta_log_energy=0.1, spectrum_energy_unit=unit.keV, spectral_sample_num=1e3):
        """
        Function to iterate over time bin edges to calculate various parameters of interest as a function of time
        :param times:
        :param lc_unit:
        :param photon_type:
        :param energy_range:
        :param energy_unit:
        :param fit_spectrum:
        :param spectrum_log_energy_range:
        :param spectrum_delta_log_energy:
        :param spectrum_energy_unit:
        :param spectral_sample_num:
        :return:
        """


        pol_deg = np.empty(times.size) * np.nan
        pol_angle = np.empty(times.size) * np.nan
        stokes_i = np.empty(times.size) * np.nan #should always be one but keep it as a check of the MCRaT code
        stokes_q = np.empty(times.size) * np.nan
        stokes_u = np.empty(times.size) * np.nan
        stokes_v = np.empty(times.size) * np.nan #should always be zero, but including it for potential future use
        pol_err = np.empty((times.size, 2)) * np.nan
        lc = np.zeros(times.size) #*np.nan
        lc_err = np.empty(times.size) * np.nan
        ph_num = np.empty(times.size) * np.nan
        num_scatt = np.zeros(times.size) * np.nan
        fit= np.zeros((times.size,4)) * np.nan
        fit_error = np.zeros((times.size,3)) * np.nan
        model_use = np.array(['']*times.size)

        for i in range(times.size-1):
            #apply various constraints
            idx=self._select_photons([times[i], times[i+1]], photon_type=photon_type, energy_range=energy_range, energy_unit=energy_unit)
            if idx[0].size > 0:
                if lc_unit==unit.erg/unit.s:
                    lc[i] = np.sum(self.detected_photons.weight[idx] *\
                                   self.detected_photons.get_energies(unit=unit.erg)[idx]) / (times[i + 1] - times[i])
                elif lc_unit==unit.count/unit.s:
                    lc[i] = np.sum(self.detected_photons.weight[idx]) / (times[i + 1] - times[i])
                else:
                    print('The light curve unit can only be set as erg/s or counts/s currently.')

                lc_err[i] = lc[i] / np.sqrt(idx[0].size)
                ph_num[i] = idx[0].size
                num_scatt[i] = np.average(self.detected_photons.scatterings[idx], weights=self.detected_photons.weight[idx])
                if self.read_stokes:
                    stokes_i[i], stokes_q[i], stokes_u[i], stokes_v[i], pol_deg[i], pol_angle[i], pol_err[i, 0], pol_err[i, 1]=\
                        self._calc_polarization(self.detected_photons.s0[idx], self.detected_photons.s1[idx],\
                                          self.detected_photons.s2[idx], self.detected_photons.s3[idx], self.detected_photons.weight[idx])

                if fit_spectrum:
                    print('Fitting between times:', times[i], times[i+1])
                    spect_dict = self.spectrum(times[i], times[i+1], spectrum_unit=unit.count/unit.s/unit.keV,\
                                               log_energy_range=spectrum_log_energy_range, delta_log_energy=spectrum_delta_log_energy,\
                                               energy_unit=spectrum_energy_unit, photon_type=None, fit_spectrum=fit_spectrum,\
                                               sample_num=spectral_sample_num)#[11:]
                    fit[i, :]=spect_dict['fit']['alpha'], spect_dict['fit']['beta'], spect_dict['fit']['break_energy'].value, spect_dict['fit']['normalization']
                    fit_error[i, :]=spect_dict['fit_errors']['alpha_errors'], spect_dict['fit_errors']['beta_errors'], spect_dict['fit_errors']['break_energy_errors'].value
                    model_use[i]=spect_dict['model_use']


        return lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_error, model_use

    def _lightcurve_calc(self, times, lc_unit, photon_type=None, energy_range=None, energy_unit=unit.keV, fit_spectrum=False, spectral_sample_num=1e4):
        """
        Function that allows for the calculation of the light curve including geometrical factors
        :param times:
        :param lc_unit:
        :param photon_type:
        :param energy_range:
        :param energy_unit:
        :param fit_spectrum:
        :param spectral_sample_num:
        :return:
        """
        lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use=\
            self._time_iterator(times, lc_unit, photon_type=photon_type, energy_range=energy_range, energy_unit=energy_unit, fit_spectrum=fit_spectrum, spectral_sample_num=spectral_sample_num)

        if self.hydrosim_dim == 2:
            factor = 2 * np.pi * (
                        np.cos(np.deg2rad(self.theta_observer - self.acceptancetheta_observer / 2.))\
                        - np.cos(np.deg2rad(self.theta_observer + self.acceptancetheta_observer / 2.)))
        elif self.hydrosim_dim == 3:
            #this isnt fully implemented yet but is is added on for future support
            factor = (dphi * np.pi / 180) * (np.cos(np.deg2rad(self.theta_observer - self.acceptancetheta_observer / 2.))\
                        - np.cos(np.deg2rad(self.theta_observer + self.acceptancetheta_observer / 2.)))

        lc = lc / factor
        lc_err = lc_err / factor

        return lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use

    def lightcurve(self, time_start=0, time_end=0, dt=0, time_array=None, lc_unit=unit.erg/unit.s, photon_type=None,\
                   energy_range=None, energy_unit=unit.keV, variable_t_bins=False, fit_spectrum=False, spectral_sample_num=1e4):
        """
        Calculates the light curve and other time resolved parameters such as polarization and time resolved spectral fits.
        Can conduct these calculations for uniform or variable time bins, where a bayesian binning algorithm is used to
        determine the variable time bins.
        :param time_start:
        :param time_end:
        :param dt:
        :param time_array:
        :param lc_unit:
        :param photon_type:
        :param energy_range:
        :param energy_unit:
        :param variable_t_bins:
        :param fit_spectrum:
        :param spectral_sample_num:
        :return:
        """

        if (lc_unit != unit.erg/unit.s) & (lc_unit != unit.count/unit.s):
            raise UnitsError('The light curve unit can only be set as erg/s or counts/s currently.')

        #see if an instrument has been loaded and if the energy range parameter is none, if energy_range!=None then
        # we use the explicitly defined energy_range
        if self.is_instrument_loaded and energy_range is None and self.instrument_lightcurve_energy_range is not None:
            #load the instrumental constraints
            energy_range=self.instrument_lightcurve_energy_range
            energy_unit=self.instrument_lightcurve_energy_unit

        if variable_t_bins:
            if (time_start + time_end + dt == 0) and time_array is not None:
                test_times=time_array
            else:
                test_times = np.arange(time_start, time_end, dt)
            data_init = self._lightcurve_calc(test_times, lc_unit, photon_type=photon_type, energy_range=energy_range, energy_unit=energy_unit)
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
                times=time_array
            else:
                times = np.arange(time_start, time_end, dt)

        lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use=\
            self._lightcurve_calc(times, lc_unit, photon_type=photon_type, energy_range=energy_range, energy_unit=energy_unit, fit_spectrum=fit_spectrum, spectral_sample_num=spectral_sample_num)

        #return lc, lc_err, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use, times
        lc_dict = dict(lightcurve=lc*lc_unit, lightcurve_errors=lc_err*lc_unit, ph_num=ph_num*unit.dimensionless_unscaled, \
                         num_scatt=num_scatt*unit.dimensionless_unscaled, times=times*unit.s,\
                       theta_observer=self.theta_observer*unit.deg)

        if self.read_stokes:
            lc_dict['pol_deg']=pol_deg*unit.dimensionless_unscaled
            lc_dict['stokes_i'] = stokes_i*unit.dimensionless_unscaled
            lc_dict['stokes_q'] = stokes_q*unit.dimensionless_unscaled
            lc_dict['stokes_u'] = stokes_u*unit.dimensionless_unscaled
            lc_dict['stokes_v'] = stokes_v*unit.dimensionless_unscaled
            lc_dict['pol_angle'] = pol_angle*unit.deg
            lc_dict['pol_deg_errors'] = pol_err[:, 0]*unit.dimensionless_unscaled
            lc_dict['pol_angle_errors'] = pol_err[:, 1]*unit.deg


        if fit_spectrum:
            lc_dict['fit'] = dict(alpha=fit[:,0], beta=fit[:,1], break_energy=fit[:,2]*energy_unit, normalization=fit[:,3])
            lc_dict['fit_errors'] = dict(alpha_errors=fit_errors[:,0], beta_errors=fit_errors[:,1], break_energy_errors=fit_errors[:,2]*energy_unit)
            lc_dict['model_use'] = model_use


        return lc_dict

    def _energy_iterator(self, time_min, time_max, spectrum_unit, energy_min, energy_max, energy_unit=unit.keV, photon_type=None):
        """
        Function that iterates over a range of energies to calculate spectra and energy dependent polarization.
        :param time_min:
        :param time_max:
        :param spectrum_unit:
        :param energy_min:
        :param energy_max:
        :param energy_unit:
        :param photon_type:
        :return:
        """
        delta_energy = energy_max - energy_min
        delta_t=time_max-time_min
        spectrum = np.zeros(energy_max.size)
        spectrum_error = np.zeros(energy_max.size)
        pol_deg = np.empty(energy_max.size) * np.nan
        pol_angle = np.empty(energy_max.size) * np.nan
        stokes_i = np.empty(energy_max.size) * np.nan #should always be one but keep it as a check of the MCRaT code
        stokes_q = np.empty(energy_max.size) * np.nan
        stokes_u = np.empty(energy_max.size) * np.nan
        stokes_v = np.empty(energy_max.size) * np.nan #should always be zero, but including it for potential future use
        pol_err = np.empty((energy_max.size, 2)) * np.nan
        num_scatt = np.zeros(energy_max.size)
        ph_num=np.zeros(energy_max.size)

        for i in range(energy_max.size):
            idx = self._select_photons([time_min, time_max], photon_type=photon_type,\
                                       energy_range=[energy_min[i], energy_max[i]], energy_unit=energy_unit)
            if idx[0].size > 0:
                if 'erg' in spectrum_unit.to_string():
                    spectrum[i] = np.sum(self.detected_photons.weight[idx] * \
                                   self.detected_photons.get_energies(unit=unit.erg)[idx])/delta_energy[i]  #/ (delta_t)
                elif 'ct' in spectrum_unit.to_string():
                    spectrum[i] = np.sum(self.detected_photons.weight[idx])/delta_energy[i] #/ (delta_t)
                else:
                    print('The spectrum unit can only be set as erg/s/energy_unit or counts/s/energy_unit currently.')

                spectrum_error[i] = spectrum[i] / np.sqrt(idx[0].size)
                ph_num[i] = idx[0].size
                num_scatt[i] = np.average(self.detected_photons.scatterings[idx], weights=self.detected_photons.weight[idx])

                if self.read_stokes:
                    stokes_i[i], stokes_q[i], stokes_u[i], stokes_v[i], pol_deg[i], pol_angle[i], pol_err[i, 0], \
                    pol_err[i, 1] = \
                        self._calc_polarization(self.detected_photons.s0[idx], self.detected_photons.s1[idx],\
                                                self.detected_photons.s2[idx], self.detected_photons.s3[idx],\
                                                self.detected_photons.weight[idx])

        return spectrum, spectrum_error, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err

    def spectrum(self, time_start, time_end, spectrum_unit=unit.erg/unit.s/unit.keV, log_energy_range=[-7, 5],\
                 delta_log_energy=0.1, energy_unit=unit.keV, photon_type=None, fit_spectrum=False, sample_num=1e4):
        """
        Function that calculates the mock observed spectrum and also fit the function with a Comp or Band function.

        :param time_start:
        :param time_end:
        :param spectrum_unit:
        :param log_energy_range:
        :param delta_log_energy:
        :param energy_unit:
        :param photon_type:
        :param fit_spectrum:
        :param sample_num:
        :return:
        """
        if ('erg' not in spectrum_unit.to_string()) & ('ct' not in spectrum_unit.to_string()):
            raise UnitsError('The spectrum unit can only be set as erg/s/energy_unit or counts/s/energy_unit currently.')

        #see if an instrument has been loaded and if the energy range parameter is none, if energy_range!=None then
        # we use the explicitly defined energy_range
        if self.is_instrument_loaded and (np.equal([-7,5],log_energy_range).sum()==2) and self.instrument_spectral_energy_range is not None:
            #load the instrumental constraints
            log_energy_range=np.log10(self.instrument_spectral_energy_range)
            energy_unit=self.instrument_spectral_energy_unit


        energy_min=10**np.arange(log_energy_range[0], log_energy_range[1], delta_log_energy)
        energy_max=energy_min*10**delta_log_energy
        energy_bin_center=np.sqrt(energy_min*energy_max)

        spectrum, spectrum_error, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err =\
            self._energy_iterator(time_start, time_end, spectrum_unit, energy_min, energy_max, energy_unit=energy_unit, photon_type=photon_type)

        fit, fit_errors, model_use=np.zeros(4)*np.nan, np.zeros(3) * np.nan, ''
        if fit_spectrum and ('ct' in spectrum_unit.to_string()):
            #break energy in fit always has same units as the energy bins
            fit, fit_errors, model_use=self.spectral_fit(spectrum, spectrum_error, ph_num, energy_bin_center, energy_unit=energy_unit, sample_num=sample_num)
        else:
            if fit_spectrum:
                print('The units of the spectrum needs to be counts/s/energy_unit to fit the spectrum.')

        #return spectrum, spectrum_error, ph_num, num_scatt, pol_deg, stokes_i, stokes_q, stokes_u, stokes_v, pol_angle, pol_err, fit, fit_errors, model_use, energy_bin_center
        spec_dict = dict(spectrum=spectrum*spectrum_unit, spectrum_errors=spectrum_error*spectrum_unit, ph_num=ph_num*unit.dimensionless_unscaled,\
                         num_scatt=num_scatt*unit.dimensionless_unscaled, energy_bin_center = energy_bin_center*energy_unit,\
                         theta_observer=self.theta_observer*unit.deg)

        if self.read_stokes:
            spec_dict['pol_deg']=pol_deg*unit.dimensionless_unscaled
            spec_dict['stokes_i'] = stokes_i*unit.dimensionless_unscaled
            spec_dict['stokes_q'] = stokes_q*unit.dimensionless_unscaled
            spec_dict['stokes_u'] = stokes_u*unit.dimensionless_unscaled
            spec_dict['stokes_v'] = stokes_v*unit.dimensionless_unscaled
            spec_dict['pol_angle'] = pol_angle*unit.deg
            spec_dict['pol_deg_errors'] = pol_err[:, 0]*unit.dimensionless_unscaled
            spec_dict['pol_angle_errors'] = pol_err[:, 1]*unit.deg

        if fit_spectrum and ('ct' in spectrum_unit.to_string()):
            spec_dict['fit'] = dict(alpha=fit[0], beta=fit[1], break_energy=fit[2]*energy_unit, normalization=fit[3])
            spec_dict['fit_errors'] = dict(alpha_errors=fit_errors[0], beta_errors=fit_errors[1], break_energy_errors=fit_errors[2]*energy_unit)
            spec_dict['model_use'] = model_use

        return spec_dict

    def _calc_polarization(self, s0, s1, s2, s3, weights, mu=1):
        """
        function used to calculate the polarization and its error for a given set of mock observed photons following:
        	Kislat, F., Clark, B., Beilicke, M., & Krawczynski, H. 2015, Astroparticle Physics, 68, 45
    	In particular we follow their appendix where photons have differing weights
        :param s0:
        :param s1:
        :param s2:
        :param s3:
        :param weights:
        :param mu:
        :return:
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

    def polarization(self, time_start, time_end, photon_type=None, energy_range=None, energy_unit=None):
        """
        Function that sets up and calculates the polarization for the time and energy range that is provided as input.
        :param time_start:
        :param time_end:
        :param photon_type:
        :param energy_range:
        :param energy_unit:
        :return:
        """
        if self.read_stokes:
            #see if an instrument has been loaded and if the energy range parameter is none, if energy_range!=None then
            # we use the explicitly defined energy_range
            if self.is_instrument_loaded and energy_range is None and self.instrument_polarization_energy_range is not None:
                #load the instrumental constraints
                energy_range=self.instrument_polarization_energy_range
                energy_unit=self.instrument_polarization_energy_unit

            idx = self._select_photons([time_start, time_end], photon_type=photon_type, energy_range=energy_range,\
                                       energy_unit=energy_unit)

            pol_err=np.empty((1, 2)) * np.nan

            stokes_i, stokes_q, stokes_u, stokes_v, pol_deg, pol_angle, pol_deg_err, pol_angle_err = \
                self._calc_polarization(self.detected_photons.s0[idx], self.detected_photons.s1[idx], \
                                        self.detected_photons.s2[idx], self.detected_photons.s3[idx], \
                                        self.detected_photons.weight[idx])

            pol_dict=dict(stokes_i=stokes_i*unit.dimensionless_unscaled, stokes_q=stokes_q*unit.dimensionless_unscaled,\
                          stokes_u=stokes_u*unit.dimensionless_unscaled, stokes_v=stokes_v*unit.dimensionless_unscaled,\
                          pol_deg=pol_deg*unit.dimensionless_unscaled, pol_angle=pol_angle*unit.deg,\
                          pol_deg_errors=pol_deg_err*unit.dimensionless_unscaled, pol_angle_errors=pol_angle_err*unit.deg,\
                          theta_observer=self.theta_observer*unit.deg)

            #return stokes_i, stokes_q, stokes_u, stokes_v, pol_deg, pol_angle, pol_err
            return pol_dict
        else:
            print('The stokes parameters have not been loaded and polarization cannot be calculated.')

    def set_spectral_fit_parameters(self, spectral_fit_energy_range=[8, 40e3], spectral_fit_energy_unit=unit.keV,\
                                    approx_gaussian_error_num=10):
        """
        Function that sets the spectral parameters for energy ranges that should be fitted with the spectral_fit method.
        Needs to be set before attempting to do any spectral fits for the class.
        :param spectral_fit_energy_range:
        :param spectral_fit_energy_unit:
        :param approx_gaussian_error_num:
        :return:
        """
        self.spectral_fit_energy_range=spectral_fit_energy_range
        self.spectral_fit_energy_unit=spectral_fit_energy_unit
        self.approx_gaussian_error_num=approx_gaussian_error_num
        self.is_set_spectral_fit_parameters=True

    def unset_spectral_fit_parameters(self):
        self.spectral_fit_energy_range=[8, 40e3]
        self.spectral_fit_energy_unit=unit.keV
        self.approx_gaussian_error_num=10
        self.is_set_spectral_fit_parameters=False


    def spectral_fit(self, spectrum, spectrum_error, ph_num, energy_bin_center, energy_unit=unit.keV, sample_num=1e4):
        """
        The function that takes a spectrum and fits it with either a Comp or Band function in the energy range specified
        by the user through the set_spectral_fit_parameters method.
        :param spectrum:
        :param spectrum_error:
        :param ph_num:
        :param energy_bin_center:
        :param energy_unit:
        :param sample_num:
        :return:
        """

        if not self.is_set_spectral_fit_parameters:
            raise InputParameterError('The set_spectral_fit_parameters method needs to be called before spectral fitting can be done.')

        energy_bin_center=energy_bin_center*energy_unit.to(self.spectral_fit_energy_unit)

        idx=np.where((energy_bin_center > self.spectral_fit_energy_range[0]) &\
                     (energy_bin_center <= self.spectral_fit_energy_range[1]) & (ph_num>self.approx_gaussian_error_num))

        if idx[0].size > 4:
            spectrum=spectrum[idx]
            spectrum_error=spectrum_error[idx]
            energy_bin_center=energy_bin_center[idx]
            normalization=np.trapz(spectrum,x=energy_bin_center)
            data_pts = spectrum.size

            #test the Band function fit first and make sure that the fitted energy cant be < 0
            band_fit, band_matrice = curve_fit(band_function, energy_bin_center, spectrum, sigma=spectrum_error,\
                                          p0=[.3, -5, 100, normalization], maxfev=5000)
            if (band_fit[2] < 0):
                band_fit, band_matrice = curve_fit(band_function, energy_bin_center, spectrum, sigma=spectrum_error,\
                                              p0=[.3, -5, 100, normalization], maxfev=5000, bounds=([-np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
            #calculate chi squared
            band_chi_sq = ((band_function(energy_bin_center, band_fit[0], band_fit[1], band_fit[2], band_fit[3]) - spectrum) ** 2 / spectrum_error ** 2).sum()

            #now test the comp function and make sure that the fitted energy cant be < 0
            comp_fit, comp_matrice = curve_fit(comptonized_function, energy_bin_center, spectrum, sigma=spectrum_error,\
                                           p0=[.3, 100,normalization])
            if (comp_fit[1] < 0):
                comp_fit, comp_matrice = curve_fit(comptonized_function, energy_bin_center, spectrum, sigma=spectrum_error,\
                                               p0=[.3, 100,normalization], bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]))

            #calcualate chi squared
            comp_chi_sq = ((comptonized_function(energy_bin_center, comp_fit[0], comp_fit[1], comp_fit[2]) - spectrum) ** 2 / spectrum_error ** 2).sum()

            #test for chi sq equivalence
            if band_chi_sq != comp_chi_sq:
                #do F test
                dof_c = data_pts - 2 - 1
                dof_b = data_pts - 3 - 1
                SS_c = ((comptonized_function(energy_bin_center, comp_fit[0], comp_fit[1], comp_fit[2]) - spectrum) ** 2).sum()
                SS_b = ((band_function(energy_bin_center, band_fit[0], band_fit[1], band_fit[2], band_fit[3]) - spectrum) ** 2).sum()

                alpha = 0.05  # false positive acceptance rate 5%
                F = ((comp_chi_sq - band_chi_sq) / (dof_c - dof_b)) / (band_chi_sq / dof_b)
                p = 1 - ss.f.cdf(F, (dof_c - dof_b), dof_b)
                #print(p, F)
                if (p < alpha):
                    model_use = 'b'
                    # print('Using The band function')
                else:
                    model_use = 'c'
                    # print('Using the Comp function')
            else:
                # if the two chi squares are equal have to choose the simpler model aka comp
                model_use = 'c'

            best_fit=np.zeros(4)
            best_fit_errors=np.zeros(3) # have three elements for alpha, beta, e_0
            is_large_error=False #parameter to determine if errors are extremely large
            #get errors in parameters
            try:
                if model_use=='c':
                    avg_fit, fit_errors=bootstrap_parameters(energy_bin_center, spectrum, spectrum_error, comptonized_function,\
                                                             comp_fit, sample_num=sample_num)
                    best_fit[0]=comp_fit[0]
                    best_fit[1] = np.nan #the comp function has to beta parameter
                    best_fit[2] = comp_fit[1]
                    best_fit[3] = comp_fit[2]

                    best_fit_errors[0]=fit_errors[0]
                    best_fit_errors[1] = np.nan
                    best_fit_errors[2] = fit_errors[1]

                    for i in range(fit_errors.size):
                        if np.abs(fit_errors[i]+comp_fit[i])>np.ceil(np.abs(3*comp_fit[i])):
                            is_large_error = True

                else:
                    avg_fit, fit_errors = bootstrap_parameters(energy_bin_center, spectrum, spectrum_error, band_function, \
                                                               band_fit, sample_num=sample_num)
                    best_fit=band_fit
                    best_fit_errors=fit_errors[:-1]

                    for i in range(best_fit_errors.size):
                        if np.abs(best_fit_errors[i]+best_fit[i])>np.ceil(np.abs(3*best_fit[i])):
                            is_large_error = True

                #print(best_fit, fit_errors, model_use) #can get really large errors so in processing output need to deal with that

                #if we have large errors or if the error is 0 (unconstrained parameters) set things to be null
                if (np.where(best_fit_errors==0)[0].size>0) or (is_large_error):
                    best_fit, best_fit_errors, model_use = np.zeros(4) * np.nan, np.zeros(3) * np.nan, ''

            except RuntimeError:
                #errors couldnt converge therefore set fit and errors to be 0
                best_fit, best_fit_errors, model_use = np.zeros(4) * np.nan, np.zeros(3) * np.nan, ''
        else:
            best_fit, best_fit_errors, model_use = np.zeros(4)*np.nan, np.zeros(3) * np.nan, ''

        return best_fit, best_fit_errors, model_use

