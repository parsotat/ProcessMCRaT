import numpy as np
from scipy import interpolate
from astropy.visualization import quantity_support
quantity_support()
from .hydrosim_lib import load_photon_vs_fluid_quantities
from .mclib import calc_optical_depth, calc_equal_arrival_time_surface, calc_line_of_sight
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .plotting import random_photon_index
from .mclib import lc_time_to_radius

def create_image(hydro_obj, key, logscale=True):
    """
    Convience function to obtain an image of a frame of a hydrodynamic simulation where the grid values are reflected about
    the axis of symmetry, which is assumed to be the z axis. Can be plotted with imshow.

    :param hydro_obj: A HydroSim object with a loaded frame
    :param key: the hydrodynamic quantity that will be plotted
    :param logscale: Boolean (default True) that denotes if a log color scale should be used in the plot
    :return: 2D array that represents the image of the hydrodynamic outflow
    """

    x,y=hydro_obj.coordinate_to_cartesian()
    data=hydro_obj.get_data(key)

    x0 = np.linspace((x.min()), (x.max()), num=1000)
    x1 = np.linspace((y.min()), (y.max()), num=1000)

    X, Y = np.meshgrid(x0, x1)

    points = np.empty([x.size, 2])
    points[:, 0] = x.flatten()
    points[:, 1] = y.flatten()

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
    Convience function to plot the hydrodynamic fluid quantity as a function of photon location in the outflow.
    This is done for a set of observer viewing angles and for a set of observed photons in time bins of the light curves.

    :param savefile: string of the photon_vs_fluid_quantities pickle file that should be loaded without the .pickle ending.
    :param hydro_keys: String that denotes the hydrodynamic grid value that will be plotted
    :param lc_dict_list: list of MockObservation calculated light curve dictionaries. Need to be identical to what was
        passed into calculate_photon_vs_fluid_quantities function
    :param theta: none or a value of observer viewing angle that the user would like to plot. None defaults to plotting all the
        observer viweing angles specified in the lc_dict_list
    :param tmin: min time in the light curve for which we want to see the evolution of the outflow values for the photons
        within the outflow. Default None denotes that the photons from the very first light curve time bin will be plotted.
    :param tmax: max time of the light curve for which we want to see the evolution of the outflow values for the photons
        within the outflow. Default None denotes that the photons to the very last light curve time bin will be plotted.
    :param savedir: strign of the directory to save the plots that are produced. Default is None which denotes that no plots should be
        saved
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
            if type(theta) is not list:
                theta=[theta]
            #see if the ith lc_dict has this theta
            if lc['theta_observer'].value in theta:
                plot_angle_switch=True
            else:
                plot_angle_switch=False
        else:
            plot_angle_switch=True

        #make sure that the theta of the passed lc dict is included in the output of the phoot_fluid analysis
        if lc['theta_observer'].value not in data_dict['obs_theta']:
            plot_angle_switch = False
        else:
            #get the index in which the angle specific data is saved in teh large array
            theta_idx=np.where(lc['theta_observer'].value==data_dict['obs_theta'])[0][0]
        

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

                    if (lc['times'][t_idx].value >= tmin) & (lc['times'][t_idx+1].value<=tmax):
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
                    fig.set_size_inches(17, 10)
                    plt.subplots_adjust(hspace=0, wspace=0)
                    axes_queue=[i for i in range(num_plots)]

                    ax_temp=None
                    
                    count=0
                    for i in hydro_keys:
                    
                        if num_plots>1:
                            ax = axes[axes_queue[0]]
                            if 'temp' in i and ax_temp is None:
                                ax_temp=ax
                                axes_queue.pop(0)
                            elif 'temp' in i and ax_temp is not None:
                                ax=ax_temp
                            else:
                                axes_queue.pop(0)
                        else:
                            ax = axes

                        r=data_dict['avg_r'][t_idx,theta_idx, :]
                        
                        if "optical_depth" in i:
                            data=calc_optical_depth(data_dict['avg_scatt'][t_idx,theta_idx, :][::-1])#order the data from smallest radii to largest, therefore reverse it
                            data=data[::-1]#revert it to how it was for plotting with r
                            #print(data)
                            #r=r[:-1]
                        else:
                            data=data_dict[i][t_idx,theta_idx, :]
                        
                        #print(i, data.shape, r.shape)
                        
                        ls='-'
                        string=None
                        

                        #plot the photon and hydro temp
                        if 'temp' in i:
                            if 'hydro' in i:
                                ls='-'
                                string = r'$T_\mathrm{fl}$'
                            else:
                                ls='-.'
                                string=r'$T_\mathrm{ph}$'

                            ylab='Temperature' + ' (K)'

                        if 'avg_scatt' in i:
                            ylab=r'$<N_\mathrm{scatt}>$'

                        if 'optical_depth' in i:
                            ylab=r'$\tau$'

                        if 'avg_pres' in i:
                            ylab=r'$<P>$'+ r' (g cm$^{-1}$ s$^{-2}$)'

                        if 'avg_dens' in i:
                            ylab=r'$<\rho>$'+ r' (g cm$^{-3}$)'

                        if 'avg_gamma' in i:
                            ylab=r'$<\Gamma>$'


                        if 'avg_pol' in i:
                            ylab=r'$<\Pi>$'+ ' (%)'


                        if 'photon_num' in i:
                            ylab=r'$N_\mathrm{ph}$'
                            
                        ax.loglog(r, data, ls=ls, label=string)
                        ax.set_ylabel(ylab)
                        
                        if 'temp' in i:
                            ax.legend(loc='best')
                    
                    if num_plots>1:
                        #put x axis ticks on top to see radius ticks easily
                        for ax in axes[1:]:
                            ax.xaxis.tick_top()
                        
                        axes[-1].tick_params(labelbottom=True, labeltop=False, bottom=True)
                        
                        ax=axes[-1]
                    else:
                        ax=axes
                    ax.set_xlabel(r'$<R>$'+ ' (cm)')


                    if savedir is not None:
                        filename='photon_fluid_params_theta_%d_time_'%(lc['theta_observer'].value)+\
                                    np.str_(np.round(lc['times'][t_idx].value, decimals=1))+'-'+\
                                    np.str_(np.round(lc['times'][t_idx+1].value, decimals=1))+'.pdf'
                        fig.savefig(os.path.join(savedir,filename), bbox_inches='tight')

def scale_photon_weights(weights, scale, power=2):
    """
    Scales the photon weights for plotting purposes. The scaling follows this function:
    new_weight=scale*(1+np.log10(weights/weights.min()))**power

    :param weights: array of weights that should be scaled.
    :param scale: a scale that can be used as a multiplier
    :param power: The power of raising the log weights
    :return: array of the new rescaled weights
    """
    return scale*(1+np.log10(weights/weights.min()))**power

def plot_photon_fluid_EATS(tmin, tmax, ph_obs, hydro_obj, x0lim=None, x1lim=None, hydro_key="dens", photon_type=None,\
                           photon_colors=["b","r"],photon_markers=['o', 'P'], photon_zorder=[5,4], photon_alpha=[0.1, 0.5], plot_weight=False):
    """
    Convience function to plot the location of the photons in the hydrodynamic frame. The plotted photons are detected by
    an observer at a specified time interval. This function can plot photons of different types and it can show the
    weights of each photons.

    :param tmin: min time that the user wants to plot photons detected after alongside the hydrodynamic simulation frame
    :param tmax: max time that the user wants to plot photons detected before alongside the hydrodynamic simulation frame
    :param ph_obs: MockObservtion object
    :param hydro_obj: HydroSim object with the loaded frame of interest
    :param x0lim: Default None or a quantity object that denotes the min and max x0 limits that should be used for
        plotting the hydrosimulation frame. None automatically sets the limits to be the limits of the photons.
    :param x1lim: Default None or a quantity object that denotes the min and max x1 limits that should be used for
        plotting the hydrosimulation frame. None automatically sets the limits to be the limits of the photons.
    :param hydro_key: The value of the hydrodynamic grid that will be plotted (eg dens)
    :param photon_type: None or a list of strings denoting photon types that should be plotted. None will plot all photons of
        all types.
    :param photon_colors: For each photon type specified, a list of the colors that they will be plotted with
    :param photon_markers: For each photon type specified, a list of the markers that will be used for each type
    :param photon_zorder: For each photon type specified, a list of the zorder that will be used for each type (see matplotlib documentation)
    :param photon_alpha: For each photon type specified, a list of the alpha values that will be used for each type (see matplotlib documentation)
    :param plot_weight: Boolean (default True) that denotes if the photon weight should be plotted by adjusting the
        sizes of each photon plotted
    :return: figure object, axis object
    """

    if photon_type is not None:
        if len(photon_type) > len(photon_markers) or len(photon_type) > len(photon_zorder) \
                or len(photon_type) > len(photon_alpha) or len(photon_type) > len(photon_colors):
            raise ValueError("Make sure that the number of photon types that are requested to be plotted are the same as the number of "+\
                             "elements in the lists passed to  photon_markers, photon_zorder, and photon_alpha.")

        if ph_obs.detected_photons.photon_type[0] is None:
            raise ValueError('The MCRaT simulation object does not contain information on the type of photons. Make sure that read_type=True when loading a frame.')

    #get the photons that fall within tmin and tmax
    index=np.where((ph_obs.detected_photons.detection_time>tmin) & (ph_obs.detected_photons.detection_time<=tmax))[0]

    if x0lim is None and x1lim is None:
        #use the hydro limits
        x0=hydro_obj.get_data("x0")
        x1=hydro_obj.get_data("x1")
        x0lim=[x0.min(), x0.max()]
        x1lim = [x1.min(), x1.max()]
    else:
        #get the values of the hydro frame around the specified limits
        hydro_obj.apply_spatial_limits(x0lim[0], x0lim[1], x1lim[0], x1lim[1])

    #get the image
    img = create_image(hydro_obj, hydro_key)

    #only look at half
    img=img[:, int(0.5*img.shape[1]):]

    #get the EATS
    x,y=hydro_obj.coordinate_to_cartesian()
    x_t_min, y_t_min = calc_equal_arrival_time_surface(ph_obs.theta_observer, int(hydro_obj.frame_num), ph_obs.fps, x.min().value, x.max().value,
                                                 tmin)
    x_t_max, y_t_max = calc_equal_arrival_time_surface(ph_obs.theta_observer, int(hydro_obj.frame_num), ph_obs.fps, x.min().value, x.max().value,
                                              tmax)
    #get the LOS
    x_LOS, y_LOS=calc_line_of_sight(ph_obs.theta_observer, x.min().value, x.max().value, y.min().value, y.max().value)

    #plot the image
    fig, ax = plt.subplots(figsize=(10, 10))
    e = [x.min().value, x.max().value, y.min().value, y.max().value]
    im = ax.imshow(img, origin='lower', extent=e, cmap=plt.get_cmap('BuPu'))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.00)
    cbar = plt.colorbar(im, cax=cax, ticklocation='right')


    if "dens" in hydro_key:
        color_label=r'Log($\frac{\rho}{\mathrm{g/cm}^3}$)'
    elif "gamma" in hydro_key:
        color_label=r'Log($\Gamma$)'
    elif "pres" in hydro_key:
        color_label = r'Log($\frac{P}{\mathrm{dyne/cm}^2}$)'
    elif "temp" in hydro_key:
        color_label = r'Log($\frac{T}{\mathrm{K}}$)'

    cbar.set_label(color_label, fontsize=14)
    ax.ticklabel_format(style='scientific', useMathText=True)
    ax.yaxis.offsetText.set_fontsize(14)
    ax.xaxis.offsetText.set_fontsize(14)
    ax.set_xlabel('x (cm)', fontsize=14)
    ax.set_ylabel('y (cm)', fontsize=14)

    ax.plot(x_LOS, y_LOS, 'r-', label=r'$\theta_\mathrm{v}=$%d$^\circ$'%(ph_obs.theta_observer))
    ax.plot(x_t_min, y_t_min, 'k--', label='t=%0.1f s' % (tmin))
    ax.plot(x_t_max, y_t_max, 'k--', label='t=%0.1f s' % (tmax))
    ax.legend(loc='best')

    #plot photons
    phx, phy=ph_obs.detected_photons.get_cartesian_coordinates(hydro_obj.dimensions)
    if photon_type is None:
        # need to determine if we need to plot the photons' weights
        if plot_weight:
            sizes = scale_photon_weights(ph_obs.detected_photons.weight, 1)[index]
        else:
            sizes=None

        ax.scatter(phx[index], phy[index], color=photon_colors[0], marker=photon_markers[0], ls='None', zorder=photon_zorder[0], \
                   alpha=photon_alpha[0], s=sizes)
    else:
        for i, type in enumerate(photon_type):

            #get the indexes of the photon types
            type_idx=np.where(type==ph_obs.detected_photons.photon_type)

            #identfy which indexes are the same based on time constraint and type constraint
            idx=np.intersect1d(type_idx, index)

            # need to determine if we need to plot the photons' weights
            if plot_weight:
                sizes = scale_photon_weights(ph_obs.detected_photons.weight, 1)[idx]
            else:
                sizes = None

            ax.scatter(phx[idx], phy[idx], color=photon_colors[i], marker=photon_markers[i], ls='None',
                       zorder=photon_zorder[i], alpha=photon_alpha[i], s=sizes)

    return fig, ax 
