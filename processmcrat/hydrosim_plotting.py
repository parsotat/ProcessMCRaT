import numpy as np
from scipy import interpolate
from astropy.visualization import quantity_support
quantity_support()
from hydrosim_lib import load_photon_vs_fluid_quantities
from .mclib import calc_optical_depth
import matplotlib.pyplot as plt
import os

from .plotting import random_photon_index
from .mclib import lc_time_to_radius


def create_image(hydro_obj, key, logscale=True):
    x0 = np.linspace((hydro_obj.get_data('x0').min()), (hydro_obj.get_data('x0').max()), num=1000)
    x1 = np.linspace((hydro_obj.get_data('x1').min()), (hydro_obj.get_data('x1').max()), num=1000)
    data=hydro_obj.get_data(key)


    points = np.empty([x0.size, 2])
    points[:, 0] = x0
    points[:, 1] = x1

    X, Y = np.meshgrid(x0, x1)

    Z = interpolate.griddata(points, data, (X, Y), method='nearest', rescale=True)

    if logscale:
        data=np.log10(Z)
    else:
        data=Z

    img_data = np.zeros([data.shape[0], 2 * data.shape[1]])
    img_data[:, :data.shape[1]] = np.fliplr(data)
    img_data[:, data.shape[1]:] = data

    return img_data

def plot_photon_vs_fluid_quantities(savefile, hydro_keys, lc_dict_list, theta=None, tmin=None, tmax=None, savedir=None):
    """

    """

    #see if the savedir exists
    if savedir is not None and not os.path.isdir(savedir):
        raise ValueError('Make sure that the directory %s exists'%(savedir))

    data_dict=load_photon_vs_fluid_quantities(savefile)

    key_in_dict=[(k in data_dict) for k in hydro_keys]

    #see if the user wants to plot the optical depth
    count=0
    for k in hydro_keys:
        if 'optical_depth' in k:
            key_in_dict[count]=True
        count+=1


    if np.sum(key_in_dict)==0:
        raise ValueError("No keys that have been passed are in the dictionary loaded from %s"%(savefile))


    for i in range(len(lc_dict_list)):
        lc=lc_dict_list[i]
        #if the user specified a certain theta to plot
        if theta is not None:
            #see if the ith lc_dict has this theta
            if lc['theta_observer'] in theta:
                plot_angle_switch=True
            else:
                plot_angle_switch=False
        else:
            plot_angle_switch=True

        #make sure that the theta of the passed lc dict is included in the output of the phoot_fluid analysis
        if lc['theta_observer'] not in data_dict['obs_theta']:
            plot_angle_switch = False
        else:
            #get the index in which the angle specific data is saved in teh large array
            theta_idx=np.where(lc['theta_observer']==data_dict['obs_theta'])[0]

        if plot_angle_switch:
            for t_idx in range(np.size(lc['times'])-1):
                # if we want to plot this observer angle, we also need to check the times that the user specified if they have been specified
                if tmin is None and tmax is None:
                    #plot everything
                    plot_time_switch=True
                else:
                    #see if the times of the light curve correspond to the provided time limits
                    if tmin is None:
                        tmin=-1e12 #if the user didnt set tmin, set to a very negative value
                    if tmax is None:
                        tmax=1e12 #if the user didnt set tmin, set to a very large value

                    if (lc['times'][t_idx].value >= tmin) & (lc['times'][t_idx+1].value<tmax):
                        plot_time_switch=True
                    else:
                        plot_time_switch=False

                if plot_time_switch:
                    #get the number of plots
                    num_plots=np.sum(key_in_dict)

                    #if the user wants both fluid and photon temp subract one plot b/c we will plot them on the same axes
                    if 'hydro_temp' in hydro_keys and 'photon_temp' in hydro_keys:
                        num_plots=num_plots-1

                    #can go ahead and plot things
                    fig, axes = plt.subplots(num_plots,sharex=True)
                    axes_queue=[i for i in range(num_plots)]

                    ax_temp=None

                    count=0
                    for i in hydro_keys:

                        r=data_dict['avg_r'][t_idx,theta_idx, :]

                        #plot the photon and hydro temp
                        if 'temp' in i:
                            if ax_temp is None:
                                ax_temp=axes[axes_queue[0]]
                                axes_queue.pop(0)
                            if 'hydro' in i:
                                data=data_dict['hydro_temp'][t_idx,theta_idx, :]
                                ls='-'
                                string = r'$T_\mathrm{fl}$'
                            else:
                                data = data_dict['photon_temp'][t_idx,theta_idx, :]
                                ls='.-'
                                string=r'$T_\mathrm{ph}$'

                            ax_temp.loglog(r, data, ls=ls, label=string)
                            ax_temp.ylabel('Temperature' + ' ('+data.unit.to_string('latex_inline')+')')
                            ax_temp.legend(loc='best')

                        if 'avg_scatt' in i:
                            ax = axes[axes_queue[0]]
                            axes_queue.pop(0)

                            data=data_dict[i][t_idx,theta_idx, :]

                            ax.loglog(r, data, ls='-')
                            ax.ylabel(r'$<N_\mathrm{scatt}>$')

                        if 'optical_depth' in i:
                            ax = axes[axes_queue[0]]
                            axes_queue.pop(0)

                            data=calc_optical_depth(data_dict['avg_scatt'][t_idx,theta_idx, :])

                            ax.loglog(r, data, ls='-')
                            ax.ylabel(r'$\tau$')

                        if 'avg_pres' in i:
                            ax = axes[axes_queue[0]]
                            axes_queue.pop(0)

                            data=data_dict[i][t_idx,theta_idx, :]

                            ax.loglog(r, data, ls='-')
                            ax.ylabel(r'$<P>$'+ ' ('+data.unit.to_string('latex_inline')+')')

                        if 'avg_dens' in i:
                            ax = axes[axes_queue[0]]
                            axes_queue.pop(0)

                            data=data_dict[i][t_idx,theta_idx, :]

                            ax.loglog(r, data, ls='-')
                            ax.ylabel(r'$<\rho>$'+ ' ('+data.unit.to_string('latex_inline')+')')

                        if 'avg_gamma' in i:
                            ax = axes[axes_queue[0]]
                            axes_queue.pop(0)

                            data=data_dict[i][t_idx,theta_idx, :]

                            ax.loglog(r, data, ls='-')
                            ax.ylabel(r'$<\Gamma>$'+ ' ('+data.unit.to_string('latex_inline')+')')


                        if 'avg_pol' in i:
                            ax = axes[axes_queue[0]]
                            axes_queue.pop(0)

                            data=data_dict[i][t_idx,theta_idx, :]

                            ax.loglog(r, data, ls='-')
                            ax.ylabel(r'$<\Pi>$'+ ' (%)')


                        if 'photon_num' in i:
                            ax = axes[axes_queue[0]]
                            axes_queue.pop(0)

                            data=data_dict[i][t_idx,theta_idx, :]

                            ax.loglog(r, data, ls='-')
                            ax.ylabel(r'$N$')

                        axes[-1].set_xlabel(r'$<R>$'+ ' ('+r.unit.to_string('latex_inline')+')')

                        #put x axis ticks on top to see radius ticks easily
                        for ax in axes[1:]:
                            ax.xaxis.tick_top()

                        axes[-1].tick_params(labelbottom=True, labeltop=False, bottom=True)

                        if savedir is not None:
                            filename='photon_fluid_params_theta_%d_time_'+\
                                        np.str_(np.round(lc['times'][t_idx].value, decimals=1))+'-'+\
                                        np.str_(np.round(lc['times'][t_idx+1].value, decimals=1))+'.pdf'
                            fig.savefig(os.path.join(savedir,filename), bbox_inches='tight')

