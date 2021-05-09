"""
Basic library for plotting MCRaT simulation data.
Written by Tyler Parsotan @ OregonState

"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from astropy import units as unit
from astropy import constants as const
from astropy.modeling import InputParameterError
from astropy.units import UnitsError
from astropy.visualization import quantity_support
quantity_support()

from .mclib import *

#set some default plotting markers and fonts for all plots here
_angle_symbols=np.array(['*', 'D', '^', 'P', '8', 's', '<', 'o', '>', 'X', 'd', 'h',  'v', 'p','H'])
_theta_1 = mlines.Line2D([], [], color='grey', marker='*', ls='None', markersize=10, label=r': $\theta_v= 1^ \circ$')
_theta_2 = mlines.Line2D([], [], color='grey', marker='D', ls='None', markersize=6, label=r': $\theta_v= 2^ \circ$')
_theta_3 = mlines.Line2D([], [], color='grey', marker='^', ls='None', markersize=8, label=r': $\theta_v= 3^ \circ$')
_theta_4 = mlines.Line2D([], [], color='grey', marker='P', ls='None', markersize=8, label=r': $\theta_v= 4^ \circ$')
_theta_5 = mlines.Line2D([], [], color='grey', marker='8', ls='None', markersize=8, label=r': $\theta_v= 5^ \circ$')
_theta_6 = mlines.Line2D([], [], color='grey', marker='s', ls='None', markersize=8, label=r': $\theta_v= 6^ \circ$')
_theta_7 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[6], ls='None', markersize=8,
                        label=r': $\theta_v= 7^ \circ$')
_theta_8 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[7], ls='None', markersize=8,
                        label=r': $\theta_v= 8^ \circ$')
_theta_9 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[8], ls='None', markersize=8,
                        label=r': $\theta_v= 9^ \circ$')
_theta_10 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[9], ls='None', markersize=8,
                         label=r': $\theta_v= 10^ \circ$')
_theta_11 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[10], ls='None', markersize=8,
                         label=r': $\theta_v= 11^ \circ$')
_theta_12 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[11], ls='None', markersize=8,
                         label=r': $\theta_v= 12^ \circ$')
_theta_13 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[12], ls='None', markersize=8,
                         label=r': $\theta_v= 13^ \circ$')
_theta_14 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[13], ls='None', markersize=8,
                         label=r': $\theta_v= 14^ \circ$')
_theta_15 = mlines.Line2D([], [], color='grey', marker=_angle_symbols[14], ls='None', markersize=8,
                         label=r': $\theta_v= 15^ \circ$')
_angle_handles=[_theta_1, _theta_2, _theta_3, _theta_4, _theta_5, _theta_6, _theta_7, _theta_8, _theta_9, _theta_10,\
                _theta_11, _theta_12, _theta_13, _theta_14, _theta_15]


def plot_spectral_fit_hist(spect_dict_list, observational_data=None):
    """
    A convenience function that plots the distribution of the spectral parameters specified in spect_dict_list alongside the
    Yu et al. (2016) Fermi BEST spectral parameters if observational_data=None otherwise the user can set observational_data
    to be a list with Nx3 columns where the first, second and third columns correspond to the alpha, beta and E_pk parameters.
    :param spect_dict_list:
    :param observational_data:
    :return:
    """
    plt.rcParams.update({'font.size': 14})
    figs=[]
    axes=[]
    alphas=np.array([i['fit']['alpha'] for i in spect_dict_list]).flatten()
    betas = np.array([i['fit']['beta'] for i in spect_dict_list]).flatten()
    break_energies = np.array([i['fit']['break_energy'] for i in spect_dict_list]).flatten()

    peak_energies=calc_epk_error(alphas, break_energies)[0]
    if observational_data is None:
        observational_data = get_FERMI_best_data()#load the FERMI data

    plot_quantities=[alphas[~np.isnan(alphas)], betas[~np.isnan(betas)], peak_energies[~np.isnan(peak_energies)]]
    for i in range(3):
        f, axarr = plt.subplots(1)

        scale = 1
        bin_edges=np.arange(np.floor(plot_quantities[i].min()), np.ceil(plot_quantities[i].max() + 0.1), 0.1)
        if i == 0:
            #do stuff specific to alpha
            c='r'
            y_label='N(' + r'$\alpha$' + ')'
            x_label=r'$\alpha$'
            xlim=[-2, 6]
            scale=1
            d=observational_data[:, i]
            bins_obs_data = np.arange(np.floor(d.min()), np.ceil(d.max() + 0.1), 0.1)
        elif i==1:
            #do stuff specific to beta
            c='b'
            y_label='N(' + r'$\beta$' + ')'
            x_label=r'$\beta$'
            xlim=[-6, -1]
            d = observational_data[~np.isnan(observational_data[:,1]),1]
            bins_obs_data = np.arange(np.floor(d.min()), np.ceil(d.max() + 0.1), 0.1)

        elif i==2:
            bin_edges=10**np.arange(np.log10(np.floor(plot_quantities[i].min())), np.log10(np.ceil(plot_quantities[i].max() + 10)), 0.1)
            c='g'
            y_label='N('+r'E$_{\mathrm{pk}}$'+')'
            x_label=r'E$_{\mathrm{pk}}$' + ' ('+spect_dict_list[0]['fit']['break_energy'].unit.to_string('latex_inline')+')'
            xlim=[bin_edges.min(), 5000]
            bins_obs_data=bin_edges
            d=observational_data[:,2]


        num, bin, stuff = axarr.hist(plot_quantities[i], bins=bin_edges, color=c, alpha=.75, zorder=5)
        hist_all, bin_edges_all = np.histogram(d, bins=bins_obs_data)
        bin_centers = (bin_edges_all[:-1] + bin_edges_all[1:]) / 2.0
        axarr.bar(bin_centers, hist_all / scale, width=(bin_edges_all[1:] - bin_edges_all[:-1]), align='center',
                color='darkorange', alpha=1, zorder=1)

        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.xlim(xlim)

        if i==2:
            axarr.set_xscale("log")

        figs.append(f)
        axes.append(axarr)

        plt.show()

    return figs, axes, alphas, betas, peak_energies

def plot_spectrum(spectrum_dict, photon_num_min=10, plot_polarization=False, plot_fit=False):
    """
    A convenience function that allows the user to plot a mock observed spectrum including energy resolved polarization
    and the best fit function.
    :param spectrum_dict:
    :param photon_num_min:
    :param plot_polarization:
    :param plot_fit:
    :return:
    """

    if plot_polarization and spectrum_dict.get('pol_angle') is None:
        raise InputParameterError(
            'The spectrum_dict dictionary does not have the polarization data to plot. Please make sure that the\
             spectrum was calculated with energy dependent polarization.')
    if plot_fit and spectrum_dict.get('fit') is None:
        raise InputParameterError(
            'The spectrum_dict dictionary does not have the fitted parameters to plot. Please make sure that the\
             spectrum was fitted in the call to calculate the spectrum.')


    if not plot_polarization:
        f, axarr = plt.subplots(1, sharex=True)
        axarr_spex = axarr
    else:
        f, axarr = plt.subplots(2, sharex=True)
        axarr_spex = axarr[0]
        axarr_pol = axarr[1]
    plt.rcParams.update({'font.size': 14})

    idx=np.where(spectrum_dict['ph_num']>photon_num_min)[0]
    axarr_spex.loglog(spectrum_dict['energy_bin_center'][idx], spectrum_dict['spectrum'][idx], 'b.')
    axarr_spex.errorbar(spectrum_dict['energy_bin_center'][idx], spectrum_dict['spectrum'][idx],\
                        yerr=spectrum_dict['spectrum_errors'][idx], color='b', marker='o', ls='None', markersize=10,
                        label='Total Spectrum')

    if plot_polarization:
        axarr_pol.semilogx(spectrum_dict['energy_bin_center'][idx], spectrum_dict['pol_deg'][idx] * 100, 'k.')
        axarr_pol.errorbar(spectrum_dict['energy_bin_center'][idx], spectrum_dict['pol_deg'][idx] * 100,\
                           yerr=spectrum_dict['pol_deg_errors'][idx] * 100, color='k', marker='o', ls='None')

        axarr_pol.set_ylabel(r'$\Pi (\%)$', fontsize=14)
        axarr_pol.set_xlabel(r'E' + ' ('+spectrum_dict['energy_bin_center'].unit.to_string('latex_inline')+')', fontsize=14)
        if (axarr_pol.get_ylim()[1] > 100):
            axarr_pol.set_ylim([0, 100])
            axarr_pol.set_yticks([0, 25, 50, 75, 100])
        if (axarr_pol.get_ylim()[0] < 0):
            axarr_pol.set_ylim([0, axarr_pol.get_ylim()[1]])

        ax_pol_angle = axarr_pol.twinx()
        ax_pol_angle.errorbar(spectrum_dict['energy_bin_center'][idx], spectrum_dict['pol_angle'][idx],\
                              yerr=spectrum_dict['pol_angle_errors'][idx], color='darkmagenta', marker='o', ls='None')
        ax_pol_angle.plot(np.arange(spectrum_dict['energy_bin_center'][idx].value.min(), spectrum_dict['energy_bin_center'][idx].value.max()),
                          np.zeros(np.size(np.arange(spectrum_dict['energy_bin_center'][idx].value.min(), spectrum_dict['energy_bin_center'][idx].value.max()))),
                          ls='--', color='darkmagenta', alpha=0.5)

        ax_pol_angle.set_ylabel(r'$\chi$ ($^\circ$)', color='darkmagenta', fontsize=14)
        ax_pol_angle.set_ylim([-90, 90])
        ax_pol_angle.set_yticks([-90, -45, 0, 45, 90])

    if spectrum_dict.get('fit') is not None and plot_fit:
        if spectrum_dict['model_use'] == 'b':
            band_in_energy_range = band_function(spectrum_dict['energy_bin_center'][idx], spectrum_dict['fit']['alpha'], spectrum_dict['fit']['beta'], spectrum_dict['fit']['break_energy'], spectrum_dict['fit']['normalization'])
            axarr_spex.plot(spectrum_dict['energy_bin_center'][idx], band_in_energy_range, color='k', label='Fitted Band', ls='solid', lw=3, zorder=3)
            #full_band = Band(nucen_all, best[0], best[1], best[2], best[
            #	3])  # this is for the extrapolated portion below energy range keV, also normalize it so its continuous with other plotted band function >energy range
            #full_band = full_band * band_in_energy_range[-1] / full_band[-1]
            #axarr_spex.plot(nucen_all[nucen_all <= fermi_gbm_e_min], full_band[nucen_all <= fermi_gbm_e_min], 'k--', lw=3,
            #				label='Extrapolated Band', zorder=3)

            axarr_spex.annotate(r'$\alpha$' + '=' + np.str(spectrum_dict['fit']['alpha']).split('.')[0] + '.' + np.str(spectrum_dict['fit']['alpha']).split('.')[1][0] +
                                '\n' + r'$\beta$' + '=' + np.str(spectrum_dict['fit']['beta']).split('.')[0] + '.' +
                                np.str(spectrum_dict['fit']['beta']).split('.')[1][0] + '\n' + r'E$_{\mathrm{o}}$' + '=' +
                                np.str(spectrum_dict['fit']['break_energy']).split('.')[0] + '.' + np.str(spectrum_dict['fit']['break_energy']).split('.')[1][0]\
                                + ' '+spectrum_dict['energy_bin_center'].unit.to_string('latex_inline'), xy=(0, 0),
                                xycoords='axes fraction', fontsize=18, xytext=(10, 10),
                                textcoords='offset points', ha='left', va='bottom')

        if spectrum_dict['model_use'] == 'c':
            axarr_spex.plot(nucen, comptonized_function(spectrum_dict['energy_bin_center'][idx], spectrum_dict['fit']['alpha'], spectrum_dict['fit']['break_energy'], spectrum_dict['fit']['normalization']), color='k', label='Fitted COMP',
                            ls='solid', lw=3, zorder=3)

            axarr_spex.annotate(r'$\alpha$' + '=' + np.str(spectrum_dict['fit']['alpha']).split('.')[0] + '.' + np.str(spectrum_dict['fit']['alpha']).split('.')[1][0] +
                                '\n' + r'E$_{\mathrm{o}}$' + '=' + np.str(spectrum_dict['fit']['break_energy']).split('.')[0] + '.' +
                                np.str(spectrum_dict['fit']['break_energy']).split('.')[1][0] + ' '+spectrum_dict['energy_bin_center'].unit.to_string('latex_inline'),\
                                xy=(0, 0), xycoords='axes fraction', fontsize=18,xytext=(10, 10),
                                textcoords='offset points', ha='left', va='bottom')
    elif plot_fit:
        print('There is no fit keyword in the spectrum dictionary to plot the best spectral fit. Make sure that '+\
              'the fit_spectrum key in the method call is set to True.')

    if 'ct' in spectrum_dict['spectrum'][idx].unit.to_string():
        axarr_spex.set_ylabel('N(' + r'E' + r') (photons s$^{-1}$ keV$^{-1})$', fontsize=14)
    else:
        axarr_spex.set_ylabel(r'L$_E$ ('+spectrum_dict['spectrum'][idx].unit.to_string('latex_inline')+')', fontsize=14)

    if not plot_polarization:
        axarr_spex.set_xlabel(r'E' + ' ('+spectrum_dict['energy_bin_center'].unit.to_string('latex_inline')+')', fontsize=14)
    else:
        axarr_spex.set_xlabel('', fontsize=14)

    axarr_spex.tick_params(labelsize=14)
    if plot_polarization:
        axarr_pol.tick_params(labelsize=14)

    plt.show()

    return f, axarr

def plot_lightcurve(lightcurve_dict_list, plot_polarization=False, plot_spectral_params=False):
    """
    Convenience function to plot the light curves(s) that are provided and the time resolved polarization and spectral
    fit parameters for the first light curve dictionary provided in lightcurve_dict_list
    :param lightcurve_dict_list:
    :param plot_polarization:
    :param plot_spectral_params:
    :return:
    """
    plt.rcParams.update({'font.size': 14})
    #see how many panels we need for the plot and how many light curves the user wants plotted
    num_panels=1+plot_polarization+plot_spectral_params


    if isinstance(lightcurve_dict_list, list):
        main_lightcurve_dict = lightcurve_dict_list[0]
        num_lc = len(lightcurve_dict_list)
    else:
        main_lightcurve_dict=lightcurve_dict_list
        num_lc = 1

    if plot_spectral_params and main_lightcurve_dict.get('fit') is None:
        raise InputParameterError(
            'The first element of the lightcurve_dict_list list does not have the fitted parameters to plot. Please make sure that the\
             call to calculate the light curve included the time-resolved spectral fitting keyword.')


    #calculate the center of the time bins
    difference=np.zeros(main_lightcurve_dict['times'].size)
    difference[:main_lightcurve_dict['times'].size - 1] = np.diff(main_lightcurve_dict['times'])
    difference[-1] = np.diff(main_lightcurve_dict['times']).min().value
    t_cen = (main_lightcurve_dict['times'].value + (main_lightcurve_dict['times'].value + difference)) / 2
    x_err=difference/2

    #calculate Epk and its errors
    if plot_spectral_params:
        peak_e, peak_e_err = calc_epk_error(main_lightcurve_dict['fit']['alpha'], main_lightcurve_dict['fit']['break_energy'],\
                                        alpha_error=main_lightcurve_dict['fit_errors']['alpha_errors'],\
                                        break_energy_error=main_lightcurve_dict['fit_errors']['break_energy_errors'])

    f, axarr = plt.subplots(num_panels, sharex=True)

    #decide which panels will plot what based on input
    lc_panel = np.nan
    pol_panel = np.nan
    spec_panel = np.nan
    beta_ax = np.nan
    epk_ax = np.nan
    pol_angle_ax=np.nan

    if num_panels==1:
        lc_panel = axarr

    if num_panels==2:
        lc_panel=axarr[0]
        if plot_polarization:
            pol_panel=axarr[1]
        else:
            spec_panel=axarr[1]

    if num_panels==3:
        lc_panel=axarr[0]
        pol_panel=axarr[1]
        spec_panel=axarr[2]


    formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 1))

    if num_panels>1:
        axarr[0].yaxis.set_major_formatter(formatter)
    else:
        axarr.yaxis.set_major_formatter(formatter)

    #plot the light curve(s)
    if num_lc==1:
        lc_panel.plot(main_lightcurve_dict['times'],main_lightcurve_dict['lightcurve'],ds='steps-post', color='k',lw=2)
        #if 'ct' in spectrum_dict['lightcurve'].unit.to_string():
        lc_panel.set_ylabel(r'L$_\mathrm{iso}$ ('+main_lightcurve_dict['lightcurve'].unit.to_string('latex_inline')+')')
    else:
        for i in lightcurve_dict_list:
            lc_panel.plot(i['times'], i['lightcurve']/i['lightcurve'].max(), ds='steps-post', lw=2)

        lc_panel.set_ylabel(r'L/L$_\mathrm{max}$' +' ('+main_lightcurve_dict['lightcurve'].unit.to_string('latex_inline')+')')

    if num_panels==1:
        lc_panel.set_xlabel('Time since Jet Launch (s)')
    else:
        lc_panel.set_xlabel('')

    #plot the spectral parameters if necessary
    if plot_spectral_params:
        #identify where alpha <0
        neg_alpha_index = np.where(main_lightcurve_dict['fit']['alpha'] < 0)[0]
        pos_alpha_index = np.where(main_lightcurve_dict['fit']['alpha'] >= 0)[0]

        #plot epk for positive alpha and COMP model
        epk_ax = lc_panel.twinx()
        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'c'), pos_alpha_index)
        epk_ax.errorbar(t_cen[index], peak_e[index], yerr=np.abs(peak_e_err[index].T),\
                          xerr=x_err[index], color='g', marker='.', ls='None')

        # plot epk for positive alpha and Band model
        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'b'), pos_alpha_index)
        epk_ax.errorbar(t_cen[index], peak_e[index], yerr=np.abs(peak_e_err[index,].T),\
                          xerr=x_err[index], color='g', marker='.', mfc='white', ls='None')

        # plot epk for negative alpha and COMP model
        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'c'), neg_alpha_index)
        epk_ax.errorbar(t_cen[index], peak_e[index], yerr=np.abs(peak_e_err[index].T), \
                          xerr=x_err[index], color='g', marker='*', ls='None')

        #plot epk for negative alpha and Band model
        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'b'), neg_alpha_index)
        epk_ax.errorbar(t_cen[index], peak_e[index], yerr=np.abs(peak_e_err[index].T), \
                          xerr=x_err[index], color='g', marker='*', mfc='white', ls='None')

        epk_ax.set_ylabel(r'E$_{\mathrm{pk}}$' + ' ('+main_lightcurve_dict['fit']['break_energy'].unit.to_string('latex_inline')+')', color='g')

        #plot alpha and beta for same combos of positive and negative alphas and the model fit as for epk
        #alphas
        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'c'), pos_alpha_index)
        spec_panel.errorbar(t_cen[index],main_lightcurve_dict['fit']['alpha'][index],\
                            yerr=np.abs(main_lightcurve_dict['fit_errors']['alpha_errors'][index].T), xerr=x_err[index],\
                            color='r',marker='.',ls='None')

        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'b'), pos_alpha_index)
        spec_panel.errorbar(t_cen[index],main_lightcurve_dict['fit']['alpha'][index],\
                            yerr=np.abs(main_lightcurve_dict['fit_errors']['alpha_errors'][index].T), xerr=x_err[index],\
                            color='r',marker='.',ls='None', mfc='white')

        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'c'), neg_alpha_index)
        spec_panel.errorbar(t_cen[index],main_lightcurve_dict['fit']['alpha'][index],\
                            yerr=np.abs(main_lightcurve_dict['fit_errors']['alpha_errors'][index].T), xerr=x_err[index],\
                            color='r',marker='*',ls='None')

        index = np.intersect1d(np.where(main_lightcurve_dict['model_use'] == 'b'), neg_alpha_index)
        spec_panel.errorbar(t_cen[index],main_lightcurve_dict['fit']['alpha'][index],\
                            yerr=np.abs(main_lightcurve_dict['fit_errors']['alpha_errors'][index].T), xerr=x_err[index],\
                            color='r',marker='*',ls='None', mfc='white')

        spec_panel.set_ylabel(r'$\alpha$', color='r')

        #beta
        if main_lightcurve_dict['fit']['beta'][~np.isnan(main_lightcurve_dict['fit']['beta'])].size > 0:
            beta_ax = spec_panel.twinx()
            beta_ax.errorbar(t_cen[pos_alpha_index], main_lightcurve_dict['fit']['beta'][pos_alpha_index],\
                             yerr=np.abs(main_lightcurve_dict['fit_errors']['alpha_errors'][pos_alpha_index].T),\
                      xerr=x_err[pos_alpha_index], color='b', marker='.', ls='None', mfc='white', mec='b')

            beta_ax.errorbar(t_cen[neg_alpha_index], main_lightcurve_dict['fit']['beta'][neg_alpha_index],\
                             yerr=np.abs(main_lightcurve_dict['fit_errors']['alpha_errors'][neg_alpha_index].T),\
                      xerr=x_err[neg_alpha_index], color='b', marker='*', ls='None', mfc='white', mec='b')
            beta_ax.set_ylabel(r'$\beta$', color='b')

        if num_panels==2:
            spec_panel.set_xlabel('Time since Jet Launch (s)')
        else:
            spec_panel.set_xlabel('')

    if plot_polarization:
        #plot the polarization in its own separate panel
        index = np.where((main_lightcurve_dict['model_use'] == 'c') | (main_lightcurve_dict['model_use'] == 'b'))[0]
        pol_panel.errorbar(t_cen[index],main_lightcurve_dict['pol_deg'][index]*100,\
                           yerr=np.abs(main_lightcurve_dict['pol_deg_errors'][index]*100),xerr=x_err[index],color='k',\
                           marker='.', ls='None')
        pol_panel.set_ylabel(r'$\Pi (\%)$')
        if (pol_panel.get_ylim()[1] > 100):
            pol_panel.set_ylim([0, 100])
            pol_panel.set_yticks([0, 25, 50, 75, 100])
        if (pol_panel.get_ylim()[0] < 0):
            pol_panel.set_ylim([0, pol_panel.get_ylim()[1]])

        pol_angle_ax=pol_panel.twinx()
        pol_angle_ax.errorbar(t_cen[index], main_lightcurve_dict['pol_angle'][index],\
                              yerr=np.abs(main_lightcurve_dict['pol_angle_errors'][index]), xerr=x_err[index],\
                                          color='darkmagenta', marker='.', ls='None')
        line_x=np.arange(main_lightcurve_dict['times'].min().value, main_lightcurve_dict['times'].max().value, difference[-1])
        pol_angle_ax.plot(line_x, np.zeros(line_x.size), ls='--', color='darkmagenta', alpha=0.5 )

        pol_angle_ax.set_ylabel(r'$\chi$ ($^\circ$)', color='darkmagenta')
        pol_angle_ax.set_ylim([-90, 90])
        pol_angle_ax.set_yticks([-90, -45, 0, 45, 90])

        if num_panels==2:
            pol_panel.set_xlabel('Time since Jet Launch (s)')
        else:
            pol_panel.set_xlabel('')

    if num_panels == 3:
        spec_panel.set_xlabel('Time since Jet Launch (s)')

    plt.show()

    return f, [lc_panel, epk_ax, spec_panel, beta_ax, pol_panel, pol_angle_ax]

def plot_yonetoku_relationship(spect_dict_list, lightcurve_dict_list, observational_data=None, yonetoku_func=get_yonetoku_relationship,\
                               plot_polarization=False, polarization_list=None, labels=None):
    """
    A convenience function that plots the observations specified by the user alongside the Yonetoku relationship. The spectra
    need to be calculated in keV and the light curves need to be binned into 1 s time bins with units of erg/s. Observational
    data can be passed in if it is organized in a certain way (see parameters), while a function that provides the yonetoku
    relationship can also be passed in. Labels for the various simulations that will be plotted can also be provided.

    :param spect_dict_list:
    :param lightcurve_dict_list:
    :param observational_data:
    :param yonetoku_func:
    :param plot_polarization:
    :param polarization_list:
    :param labels:
    :return:
    """
    plt.rcParams.update({'font.size': 14})

    #make sure the inputs are formatted correctly
    if not isinstance(spect_dict_list[0], list):
        spect_dict_list=[spect_dict_list]
        lightcurve_dict_list=[lightcurve_dict_list]
        if plot_polarization:
            polarization_list=[polarization_list]

    #get the shape of the inputs
    num_sims=np.shape(np.array(spect_dict_list, dtype=object))[0]

    #test the inputs before doing work
    num_angles=len(spect_dict_list)
    if len(spect_dict_list) != len(lightcurve_dict_list):
        raise InputParameterError('The lengths of the spectral lists and the light curves lists do not match. Make sure'+\
         'that they have the same number of dictionaries')
    if plot_polarization and polarization_list is None:
        raise InputParameterError('The plot polarization keyword is set to True but there is no provided list of '+\
                                  'polarization dictionaries for the function to plot the polarization.')

    if not isinstance(spect_dict_list[0], list):
        for i in range(len(spect_dict_list)):
            if (spect_dict_list[i]['energy_bin_center'].unit != unit.keV) or (lightcurve_dict_list[i]['lightcurve'].unit != unit.erg/unit.s):
                raise UnitsError('The units of the spectra can only be in keV and the units of the light curve can only be in erg/s.'+\
                                 'Make sure that all the spectra and light curves were created with the proper units.')
    else:
        for i in range(num_sims):
            for j in range(len(spect_dict_list[i])):
                if (spect_dict_list[i][j]['energy_bin_center'].unit != unit.keV) or (lightcurve_dict_list[i][j]['lightcurve'].unit != unit.erg/unit.s):
                    raise UnitsError('The units of the spectra can only be in keV and the units of the light curve can only be in erg/s.'+\
                                     'Make sure that all the spectra and light curves were created with the proper units.')


    if observational_data is None:
        obs_E_p, obs_E_p_err, obs_L_iso, obs_L_iso_err=get_yonetoku_data()
    else:
        obs_E_p, obs_E_p_err, obs_L_iso, obs_L_iso_err=observational_data[:,0], observational_data[:,1], \
                                                       observational_data[:,2], observational_data[:,3]
    energies=np.linspace(obs_E_p[obs_E_p!=0].min()/10,obs_E_p[obs_E_p!=0].max()*10,100)
    yonetoku_liso=yonetoku_func(energies)

    fig, ax = plt.subplots()
    ax.errorbar(obs_L_iso[obs_L_iso != 0] , obs_E_p[obs_E_p != 0], yerr=[obs_E_p_err[obs_E_p != 0], obs_E_p_err[obs_E_p != 0]],
                xerr=[obs_L_iso_err[obs_L_iso != 0] , obs_L_iso_err[obs_L_iso != 0] ], color='grey', marker='o', ls='None')
    ax.loglog(yonetoku_liso, energies, color='grey')


    #collect data and scale lightcurve and its error to Yonetoku paper (/(1e52))
    all_angles=[]
    all_L_iso_sim=[]
    all_L_err_sim=[]
    all_E_p_sim=[]
    all_E_p_err_sim=[]
    all_polarization_deg=[]
    all_polarization_deg_error=[]
    lines=[]
    num_angle_list = []  # count the number fo angles in each
    for i in range(num_sims):
        if plot_polarization:
            L_iso_sim, L_err_sim, E_p_sim, E_p_err_sim, polarization_deg, polarization_angle, polarization_deg_error,\
            polarization_angle_error, angles= calc_yonetoku_values(spect_dict_list[i], lightcurve_dict_list[i], polarization_list=polarization_list[i])
        else:
            L_iso_sim, L_err_sim, E_p_sim, E_p_err_sim, polarization_deg, polarization_angle, polarization_deg_error, \
            polarization_angle_error, angles = calc_yonetoku_values(spect_dict_list[i], lightcurve_dict_list[i])
        for j in angles: all_angles.append(j)
        for j in L_iso_sim: all_L_iso_sim.append(j)
        for j in L_err_sim: all_L_err_sim.append(j)
        for j in E_p_sim: all_E_p_sim.append(j)
        for j in E_p_err_sim: all_E_p_err_sim.append(j)
        for j in polarization_deg: all_polarization_deg.append(j*100)
        for j in polarization_deg_error: all_polarization_deg_error.append(j*100)
        if labels is not None:
            lines.append(ax.plot(L_iso_sim, E_p_sim, linewidth=2, label=labels[i])[0])
        else:
            lines.append(ax.plot(L_iso_sim, E_p_sim, linewidth=2, label='Simulation %d'%(i))[0])
        num_angle_list.append(len(spect_dict_list[i]))



    all_L_iso_sim=np.array(all_L_iso_sim)
    all_L_err_sim=np.array(all_L_err_sim)
    all_E_p_sim=np.array(all_E_p_sim)
    all_E_p_err_sim=np.array(all_E_p_err_sim)
    all_polarization_deg=np.array(all_polarization_deg)
    all_polarization_deg_error=np.array(all_polarization_deg_error)


    ax.set_ylabel(r'E$_{\mathrm{pk}}$' + ' (' + spect_dict_list[0][0]['energy_bin_center'].unit.to_string('latex_inline') + ')',
                          fontsize=14)
    ax.set_xlabel(r'$L_\mathrm{iso}$' + ' ('+lightcurve_dict_list[0][0]['lightcurve'].unit.to_string('latex_inline')+')')

    clb=None
    if plot_polarization:
        polarization_deg_copy = all_polarization_deg.copy()
        polarization_deg_copy[polarization_deg_copy.argmin()] = 0
        sc = ax.scatter(all_L_iso_sim, all_E_p_sim, s=0, c=polarization_deg_copy)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.00)
        clb = plt.colorbar(sc, cax=cax, ticklocation='right')
        clb.set_label(r'$\Pi$ (%)')

        norm = matplotlib.colors.Normalize(vmin=min(polarization_deg_copy), vmax=max(polarization_deg_copy), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap='viridis')

    count_beg=0
    count_end=count_beg+num_angle_list[0]
    if plot_polarization:
        for i in range(num_sims):
            L_iso_sim=all_L_iso_sim[count_beg:count_end]
            E_p_sim=all_E_p_sim[count_beg:count_end]
            L_err_sim=all_L_err_sim[count_beg:count_end]
            E_p_err_sim=all_E_p_err_sim[count_beg:count_end]
            markers=_angle_symbols[:num_angle_list[i]]
            pol_color = np.array([(mapper.to_rgba(v)) for v in all_polarization_deg[count_beg:count_end]])
            for x, y, xe, ye, color, sym in zip(L_iso_sim, E_p_sim, L_err_sim, E_p_err_sim, pol_color, markers):
                ax.plot(x, y, marker=sym, color=color, markeredgecolor=lines[i].get_color())
                ax.errorbar(x, y, xerr=xe, yerr=ye, marker=sym, color=color, markersize='10', markeredgecolor=lines[i].get_color())
            count_beg = count_end
            if i != num_sims-1:
                count_end = count_beg + num_angle_list[i+1]
    else:
        for i in range(num_sims):
            L_iso_sim=all_L_iso_sim[count_beg:count_end]
            E_p_sim=all_E_p_sim[count_beg:count_end]
            L_err_sim=all_L_err_sim[count_beg:count_end]
            E_p_err_sim=all_E_p_err_sim[count_beg:count_end]
            markers=_angle_symbols[:num_angle_list[i]]
            for x, y, xe, ye, sym in zip(L_iso_sim, E_p_sim, L_err_sim, E_p_err_sim, markers):
                ax.plot(x, y, marker=sym, color=lines[i].get_color(), markeredgecolor=lines[i].get_color())
                ax.errorbar(x, y, xerr=xe, yerr=ye, marker=sym, color=lines[i].get_color(), markersize='10', markeredgecolor=lines[i].get_color())

            count_beg = count_end
            if i != num_sims-1:
                count_end = count_beg + num_angle_list[i+1]

    ax.set_ylim([4, 3e3])
    ax.set_xlim([1e52*2e-4, 1e52*150])

    leg_handles=[l for l in lines]
    for i in _angle_handles[:np.int_(np.max(num_angle_list))]: leg_handles.append(i)
    ax.legend(loc='upper left', ncol=2, handles=leg_handles)

    return fig, ax, lines, clb

def plot_golenetskii_relationship(lightcurve_dict_list, golenetskii_func=get_golenetskii_relationship, labels=None, luminosity_cutoff=None):
    """
    A convience funtion to plot MCRaT mock observations alongside the golenetskii relationship. The user can pass in a
    function that provides the golenteskii relationship and its upper and lower limits. The user can also specify a cutoff
    in the plotted luminosities. They can also specify labels that will populate the legend of the figure.
    :param lightcurve_dict_list:
    :param golenetskii_func:
    :param labels:
    :param luminosity_cutoff:
    :return:
    """
    plt.rcParams.update({'font.size': 14})

    #make sure the inputs are formatted as a list
    if not isinstance(lightcurve_dict_list, list):
        lightcurve_dict_list=[lightcurve_dict_list]


    num_angles = np.shape(np.array(lightcurve_dict_list, dtype=object))[0]

    #check that the light curve dicts have fitted spectra
    for i in range(num_angles):
        if lightcurve_dict_list[i].get('fit') is None:
            raise InputParameterError('The lightcurve_dict_list list does not have the fitted'+\
             ' parameters to plot. Please make sure that the call to calculate the light curves included the time'+\
                                      '-resolved spectral fitting keyword.')

    for i in lightcurve_dict_list:
        if (i['fit']['break_energy'].unit != unit.keV) or (i['lightcurve'].unit != unit.erg / unit.s):
            raise UnitsError(
                'The units of the spectra can only be in keV and the units of the light curve can only be in erg/s.' + \
                'Make sure that all the spectra and light curves were created with the proper units.')

    #get the goleketskii data
    x,y=golenetskii_func()
    x_p,y_p=golenetskii_func(value='+')
    x_m,y_m=golenetskii_func(value='-')

    #initalize plotting
    fig, ax = plt.subplots()
    ax.loglog(x, y, color='grey')
    ax.loglog(x_p, y_p,color='grey', ls='-.')
    ax.loglog(x_m, y_m, color='grey', ls='-.')

    lines=[]
    angles=[]
    for i in range(num_angles):
        lc=lightcurve_dict_list[i]['lightcurve'].value
        lc_e = lightcurve_dict_list[i]['lightcurve_errors'].value
        peak_e, peak_e_err = calc_epk_error(lightcurve_dict_list[i]['fit']['alpha'],
                                            lightcurve_dict_list[i]['fit']['break_energy'], \
                                            alpha_error=lightcurve_dict_list[i]['fit_errors']['alpha_errors'], \
                                            break_energy_error=lightcurve_dict_list[i]['fit_errors']['break_energy_errors'])

        m=_angle_symbols[np.int_(lightcurve_dict_list[i]['theta_observer'].value-1)]
        angles.append(np.int_(lightcurve_dict_list[i]['theta_observer'].value))

        if labels is None:
            l=ax.plot([np.nan], [np.nan], label='Simulation %d'%(i))[0]
        else:
            l=ax.plot([np.nan], [np.nan], label=labels[i])[0]
        lines.append(l)
        if luminosity_cutoff is not None:
            idx=np.where(lc>luminosity_cutoff)[0]
        else:
            idx = np.where(lc>-1)[0]
        ax.errorbar(lc[idx], peak_e[idx], xerr=lc_e[idx], yerr=np.abs(peak_e_err[idx]), marker=m, ls='None', c=l.get_color())

    ax.set_xlabel('Time Interval ' +r'Luminosity' + ' ('+lightcurve_dict_list[0]['lightcurve'].unit.to_string('latex_inline')+')')
    ax.set_ylabel('Time Interval '+ r'E$_{\mathrm{pk}}$' + \
                  ' (' + lightcurve_dict_list[0]['fit']['break_energy'].unit.to_string('latex_inline') + ')', fontsize=14)
    ax.set_xlim([8e49,2e53])
    ax.set_ylim([10,1.5e3])

    leg_handles = [l for l in lines]
    for i in _angle_handles[:np.int_(np.max(angles))]: leg_handles.append(i)
    ax.legend( loc='upper left', ncol=2,  handles=leg_handles, numpoints=1,fontsize='10') #ti, ob,ti_e150, ti_e150_g100, cmc_oi,

    return  fig, ax

def plot_amati_relationship(spect_dict_list, lightcurve_dict_list, amati_func=get_amati_relationship, labels=None):
    """
    A convience funtion to plot MCRaT mock observations alongside the amati relationship. The user can pass in a
    function that specifies the amati relationship and its upper and lower limits. They can also specify labels that will
    populate the legend of the figure.
    :param spect_dict_list:
    :param lightcurve_dict_list:
    :param amati_func:
    :param labels:
    :return:
    """

    plt.rcParams.update({'font.size': 14})

    #make sure the inputs are formatted correctly
    if not isinstance(spect_dict_list[0], list):
        spect_dict_list=[spect_dict_list]
        lightcurve_dict_list=[lightcurve_dict_list]

    #get the shape of the inputs
    num_sims=np.shape(np.array(spect_dict_list, dtype=object))[0]

    #test the inputs before doing work
    num_angles=len(spect_dict_list)
    if len(spect_dict_list) != len(lightcurve_dict_list):
        raise InputParameterError('The lengths of the spectral lists and the light curves lists do not match. Make sure'+\
         'that they have the same number of dictionaries')

    if not isinstance(spect_dict_list[0], list):
        for i in range(len(spect_dict_list)):
            if (spect_dict_list[i]['energy_bin_center'].unit != unit.keV) or (lightcurve_dict_list[i]['lightcurve'].unit != unit.erg/unit.s):
                raise UnitsError('The units of the spectra can only be in keV and the units of the light curve can only be in erg/s.'+\
                                 'Make sure that all the spectra and light curves were created with the proper units.')
    else:
        for i in range(num_sims):
            for j in range(len(spect_dict_list[i])):
                if (spect_dict_list[i][j]['energy_bin_center'].unit != unit.keV) or (lightcurve_dict_list[i][j]['lightcurve'].unit != unit.erg/unit.s):
                    raise UnitsError('The units of the spectra can only be in keV and the units of the light curve can only be in erg/s.'+\
                                     'Make sure that all the spectra and light curves were created with the proper units.')

    #initalize plotting with amati relationship and its error
    fig, ax = plt.subplots()
    x,y=amati_func()
    x_m,y_m=amati_func(value='-')
    x_p,y_p=amati_func(value='+')
    ax.loglog(x, y, color='grey')
    ax.loglog(x_p, y_p, color= 'grey', ls='-.')
    ax.loglog(x_m, y_m, color= 'grey', ls='-.')

    #collect data and scale lightcurve and its error to Yonetoku paper (/(1e52))
    all_angles=[]
    all_E_iso_sim=[]
    all_E_err_sim=[]
    all_E_p_sim=[]
    all_E_p_err_sim=[]
    lines=[]
    num_angle_list = []  # count the number fo angles in each
    for i in range(num_sims):
        E_iso_sim, E_err_sim, E_p_sim, E_p_err_sim, angles = calc_amati_values(spect_dict_list[i], lightcurve_dict_list[i])
        for j in angles: all_angles.append(j)
        for j in E_iso_sim: all_E_iso_sim.append(j)
        for j in E_err_sim: all_E_err_sim.append(j)
        for j in E_p_sim: all_E_p_sim.append(j)
        for j in E_p_err_sim: all_E_p_err_sim.append(j)
        if labels is not None:
            lines.append(ax.plot(E_iso_sim, E_p_sim, linewidth=2, label=labels[i])[0])
        else:
            lines.append(ax.plot(E_iso_sim, E_p_sim, linewidth=2, label='Simulation %d'%(i))[0])
        num_angle_list.append(len(spect_dict_list[i]))

    all_E_iso_sim=np.array(all_E_iso_sim)
    all_E_err_sim=np.array(all_E_err_sim)
    all_E_p_sim=np.array(all_E_p_sim)
    all_E_p_err_sim=np.array(all_E_p_err_sim)

    ax.set_ylabel(r'E$_{\mathrm{pk}}$' + ' (' + spect_dict_list[0][0]['energy_bin_center'].unit.to_string('latex_inline') + ')')
    ax.set_xlabel(r'$E_\mathrm{iso}$' + ' ('+(lightcurve_dict_list[0][0]['lightcurve'].unit*lightcurve_dict_list[0][0]['times'].unit).to_string('latex_inline')+')')

    count_beg=0
    count_end=count_beg+num_angle_list[0]
    for i in range(num_sims):
        E_iso_sim = all_E_iso_sim[count_beg:count_end]
        E_p_sim = all_E_p_sim[count_beg:count_end]
        E_err_sim = all_E_err_sim[count_beg:count_end]
        E_p_err_sim = all_E_p_err_sim[count_beg:count_end]
        markers = _angle_symbols[:num_angle_list[i]]
        for x, y, xe, ye, sym in zip(E_iso_sim, E_p_sim, E_err_sim, E_p_err_sim, markers):
            ax.plot(x, y, marker=sym, color=lines[i].get_color(), markeredgecolor=lines[i].get_color())
            ax.errorbar(x, y, xerr=xe, yerr=ye, marker=sym, color=lines[i].get_color(), markersize='10',
                        markeredgecolor=lines[i].get_color())

        count_beg = count_end
        if i != num_sims - 1:
            count_end = count_beg + num_angle_list[i + 1]

    ax.set_ylim([4,3e3])
    ax.set_xlim([1e52*4e-3, 1e52*1e2])

    leg_handles=[l for l in lines]
    for i in _angle_handles[:np.int_(np.max(num_angle_list))]: leg_handles.append(i)
    ax.legend(loc='upper left', ncol=2, handles=leg_handles)


    return fig, ax, lines

def plot_polarization_observer_angle(polarization_dict_list, plot_pol_angle=False, labels=None):
    """
    A convience function that allows MCRaT mock observed polarizations to be plotted as a function of observer viewing
    angle. The function can just plot the polarization degree or it can plot both the degree and the angle. The user can also
    pass in a list of labels that will populate the legend of the figure.
    :param polarization_dict_list:
    :param plot_pol_angle:
    :param labels:
    :return:
    """
    plt.rcParams.update({'font.size': 14})
    num_panels = 1 + plot_pol_angle

    #make sure the inputs are formatted correctly
    if not isinstance(polarization_dict_list[0], list):
        polarization_dict_list=[polarization_dict_list]

    #get the shape of the inputs
    num_sims=np.shape(np.array(polarization_dict_list, dtype=object))[0]

    f, axarr = plt.subplots(num_panels, sharex=True)

    if num_panels==1:
        ax=axarr
    else:
        ax=axarr[0]
        ax_2=axarr[1]

    #collect data and plot it
    all_angles=[]
    all_polarization_deg=[]
    all_polarization_deg_error=[]
    all_polarization_angle=[]
    all_polarization_angle_error=[]
    lines=[]
    num_angle_list = []  # count the number fo angles in each
    for i in range(num_sims):
        angles = []
        polarization_deg = []
        polarization_deg_error = []
        polarization_angle = []
        polarization_angle_error = []

        for j in polarization_dict_list[i]:
            angles.append(j['theta_observer'].value)
            polarization_angle.append(j['pol_angle'].value)
            polarization_angle_error.append(j['pol_angle_errors'].value)
            polarization_deg.append(j['pol_deg'].value*100)
            polarization_deg_error.append(j['pol_deg_errors'].value*100)

        #make np arrays to sort by observer viewing angle
        angles = np.array(angles)
        polarization_deg = np.array(polarization_deg)[angles.argsort()]
        polarization_deg_error = np.array(polarization_deg_error)[angles.argsort()]
        polarization_angle = np.array(polarization_angle)[angles.argsort()]
        polarization_angle_error = np.array(polarization_angle_error)[angles.argsort()]
        angles = angles[angles.argsort()]


        for j in angles: all_angles.append(j)
        for j in polarization_angle: all_polarization_angle.append(j)
        for j in polarization_angle_error: all_polarization_angle_error.append(j)
        for j in polarization_deg: all_polarization_deg.append(j)
        for j in polarization_deg_error: all_polarization_deg_error.append(j)

        num_angle_list.append(len(polarization_dict_list[i]))

    all_polarization_deg=np.array(all_polarization_deg)
    all_polarization_deg_error=np.array(all_polarization_deg_error)
    all_polarization_angle=np.array(all_polarization_angle)
    all_polarization_angle_error=np.array(all_polarization_angle_error)
    all_angles=np.array(all_angles)

    count_beg=0
    count_end=count_beg+num_angle_list[0]
    for i in range(num_sims):
        if labels is None:
            l=ax.plot([np.nan], [np.nan], label='Simulation %d'%(i))[0]
        else:
            l=ax.plot([np.nan], [np.nan], label=labels[i])[0]
        lines.append(l)
        ax.errorbar(all_angles[count_beg:count_end], all_polarization_deg[count_beg:count_end], \
                    yerr=all_polarization_deg_error[count_beg:count_end], linewidth=2, marker='o', color=l.get_color())

        count_beg = count_end
        if i != num_sims - 1:
            count_end = count_beg + num_angle_list[i + 1]

    if plot_pol_angle:
        count_beg = 0
        count_end = count_beg + num_angle_list[0]
        for i in range(num_sims):
            if labels is not None:
                ax_2.errorbar(all_angles[count_beg:count_end], all_polarization_angle[count_beg:count_end], \
                                         yerr=all_polarization_angle_error[count_beg:count_end], linewidth=2,
                                         label=labels[i], color=lines[i].get_color(), marker='o',)
            else:
                ax_2.errorbar(all_angles[count_beg:count_end], all_polarization_angle[count_beg:count_end], \
                                         yerr=all_polarization_angle_error[count_beg:count_end], linewidth=2,
                                         label='Simulation %d' % (i), color=lines[i].get_color(), marker='o',)

            count_beg = count_end
            if i != num_sims - 1:
                count_end = count_beg + num_angle_list[i + 1]

        ax_2.plot(np.linspace(all_angles.min(), all_angles.max()), np.zeros(np.size(np.linspace(all_angles.min(), all_angles.max()))),\
                          ls='--', color='darkmagenta', alpha=0.5)

    if (ax.get_ylim()[1] > 100):
        ax.set_ylim([0, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
    if (ax.get_ylim()[0] < 0):
        ax.set_ylim([0, np.ceil(ax.get_ylim()[1])])

    ax.set_ylabel(r'$\Pi$ (%)', fontsize=14)
    if num_panels==1:
        ax.set_xlabel(r'$\theta_{\mathrm{v}} (^\circ)$', fontsize=14)
    else:
        ax.set_xlabel(r'', fontsize=14)
        ax_2.set_ylabel(r'$\chi  (^\circ)$', color='darkmagenta', fontsize=14)
        ax_2.set_xlabel(r'$\theta_{\mathrm{v}} (^\circ)$', fontsize=14)
        ax_2.set_ylim([-90, 90])
        ax_2.set_yticks([-90, -45, 0, 45, 90])


    leg_handles=[l for l in lines]
    ax.legend(loc='best', ncol=2, handles=leg_handles)

    return f, axarr, all_angles, all_polarization_deg, all_polarization_deg_error, all_polarization_angle, all_polarization_angle_error


def plot_polarization_peak_energy(spect_dict_list, polarization_dict_list, labels=None):
    plt.rcParams.update({'font.size': 14})

    #make sure the inputs are formatted correctly
    if not isinstance(spect_dict_list[0], list):
        spect_dict_list=[spect_dict_list]
        polarization_dict_list=[polarization_dict_list]

    #get the shape of the inputs
    num_sims=np.shape(np.array(spect_dict_list, dtype=object))[0]

    #test the inputs before doing work
    if len(spect_dict_list) != len(polarization_dict_list):
        raise InputParameterError('The lengths of the spectral lists and the polarization lists do not match. Make sure'+\
         'that they have the same number of dictionaries')

    #collect data and plot it
    all_angles=[]
    all_E_p_sim=[]
    all_E_p_err_sim=[]
    all_polarization_deg=[]
    all_polarization_deg_error=[]
    all_polarization_angle=[]
    all_polarization_angle_error=[]
    lines=[]
    num_angle_list = []  # count the number fo angles in each
    for i in range(num_sims):
        angles = []
        polarization_deg = []
        polarization_deg_error = []
        polarization_angle = []
        polarization_angle_error = []
        E_p_sim = []
        E_p_err_sim = []

        for p, s in zip(polarization_dict_list[i], spect_dict_list[i]):
            angles.append(p['theta_observer'].value)
            polarization_angle.append(p['pol_angle'].value)
            polarization_angle_error.append(p['pol_angle_errors'].value)
            polarization_deg.append(p['pol_deg'].value*100)
            polarization_deg_error.append(p['pol_deg_errors'].value*100)
            val= calc_epk_error(s['fit']['alpha'], s['fit']['break_energy'].value, \
															alpha_error=s['fit_errors']['alpha_errors'], \
															break_energy_error=s['fit_errors']['break_energy_errors'].value)
            E_p_sim.append(val[0])
            E_p_err_sim.append(val[1])

        #make np arrays to sort by observer viewing angle
        angles = np.array(angles)
        polarization_deg = np.array(polarization_deg)[angles.argsort()]
        polarization_deg_error = np.array(polarization_deg_error)[angles.argsort()]
        polarization_angle = np.array(polarization_angle)[angles.argsort()]
        polarization_angle_error = np.array(polarization_angle_error)[angles.argsort()]
        E_p_sim = np.array(E_p_sim)[angles.argsort()]
        E_p_err_sim = np.array(E_p_err_sim)[angles.argsort()]
        angles=angles[angles.argsort()]

        for j in angles: all_angles.append(j)
        for j in polarization_angle: all_polarization_angle.append(j)
        for j in polarization_angle_error: all_polarization_angle_error.append(j)
        for j in polarization_deg: all_polarization_deg.append(j)
        for j in polarization_deg_error: all_polarization_deg_error.append(j)
        for j in E_p_sim: all_E_p_sim.append(j)
        for j in E_p_err_sim: all_E_p_err_sim.append(j)

        num_angle_list.append(len(polarization_dict_list[i]))

    all_polarization_deg=np.array(all_polarization_deg)
    all_polarization_deg_error=np.array(all_polarization_deg_error)
    all_polarization_angle=np.array(all_polarization_angle)
    all_polarization_angle_error=np.array(all_polarization_angle_error)
    all_angles=np.array(all_angles)
    all_E_p_sim=np.array(all_E_p_sim)
    all_E_p_err_sim=np.array(all_E_p_err_sim)


    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xlabel(r'E$_{\mathrm{pk}}$' + ' (' + spect_dict_list[0][0]['energy_bin_center'].unit.to_string('latex_inline') + ')')
    ax.set_ylabel(r'$\Pi$ (%)', fontsize=14)

    count_beg=0
    count_end=count_beg+num_angle_list[0]
    for i in range(num_sims):
        if labels is None:
            l=ax.plot([np.nan], [np.nan], label='Simulation %d'%(i))[0]
        else:
            l=ax.plot([np.nan], [np.nan], label=labels[i])[0]
        lines.append(l)
        markers = _angle_symbols[:num_angle_list[i]]
        for x, y, xe, ye, sym in zip(all_E_p_sim[count_beg:count_end], all_polarization_deg[count_beg:count_end], all_E_p_err_sim[count_beg:count_end], all_polarization_deg_error[count_beg:count_end], markers):
            ax.errorbar(x, y, xerr=xe, yerr=ye, marker=sym, color=l.get_color(), markersize='10', markeredgecolor=l.get_color())

        count_beg = count_end
        if i != num_sims - 1:
            count_end = count_beg + num_angle_list[i + 1]

    leg_handles=[l for l in lines]
    for i in _angle_handles[:np.int_(np.max(num_angle_list))]: leg_handles.append(i)
    ax.legend(loc='best', ncol=2, handles=leg_handles)
    if (ax.get_ylim()[1] > 100):
        ax.set_ylim([0, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
    if (ax.get_ylim()[0] < 0):
        ax.set_ylim([0, np.ceil(ax.get_ylim()[1])])


    return fig, ax
