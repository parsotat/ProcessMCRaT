import numpy as np
import astropy.units as u
from scipy.interpolate import griddata
from astropy import units as unit
from astropy import constants as const
import tempfile
import os
import shutil
from joblib import Parallel, delayed
from joblib import load, dump
import pickle
from .processmcrat import *
from .process_hydrosim import *
#from process_hydrosim import HydroSim
from .mclib import calc_equal_arrival_time_surface, lorentzBoostVectorized, calc_photon_temp


def hydro_position_interpolate(photons, hydro_obj, key):
    """
    Interpolates the values of the hydrodynamic grid at the locations of a set of MCRaT photons.

    :param photons: a PhotonLlist object of photons that will be used to interpolate the grid values near
    :param hydro_obj: a HydroSim object with a loaded frame that will be used to interpolate values within
    :param key: String of the hydrodynamic value of interest (eg. gamma, v0, etc)
    :return: array of the interpolated hydro data vaues for each photon, the max distance from a grid location between
        the location of a photon and the grid value (can be used for confidence checking)
    """

    #try to identify where the observed photon coordinates are encompassed by the hydrogrid
    i=2
    #convert MCRaT 3D coord to coord system of hydro simulation
    if hydro_obj.coordinate_sys in ['CARTESIAN', 'cartesian', 'CYLINDRICAL']:
        ph_r0, ph_r1=photons.get_cartesian_coordinates(hydro_obj.dimensions)*hydro_obj.length_scale.unit
    else:
         ph_r0, ph_r1=photons.get_spherical_coordinates(hydro_obj.dimensions)
         ph_r0=ph_r0*hydro_obj.length_scale.unit

    #apply spatial limits in hydro coordinate system
    hydro_obj.apply_spatial_limits(ph_r0.min()*(1-(.05*i)), ph_r0.max()*(1+(.05*i)),\
                                   ph_r1.min()*(1-(.05*i)), ph_r1.max()*(1+(.05*i)))

    while (hydro_obj.spatial_limit_idx.size == 0):
        i+=1
        #find indexes where x and z are close to points of interest to limit possibilities for finding differences
        hydro_obj.apply_spatial_limits(ph_r0.min() * (1 - (.05 * i)), ph_r0.max() * (1 + (.05 * i)),
                                       ph_r1.min() * (1 - (.05 * i)), ph_r1.max() * (1 + (.05 * i)))

    #want to do things in cartesian here
    gridpoints=np.zeros((hydro_obj.spatial_limit_idx.size,2))
    hydro_x, hydro_y=hydro_obj.coordinate_to_cartesian()
    gridpoints[:,0]=hydro_x
    gridpoints[:,1]=hydro_y

    data=hydro_obj.get_data(key)


    photonpoints=np.zeros((np.size(ph_r0),2))
    ph_x, ph_y=photons.get_cartesian_coordinates(hydro_obj.dimensions)*hydro_obj.length_scale.unit
    photonpoints[:,0]=ph_x
    photonpoints[:,1]=ph_y

    data_points=griddata(gridpoints,data,photonpoints,method='nearest', rescale=True)
    if 'gamma' not in key:
        data_points *= data.unit
    distances=griddata(gridpoints,gridpoints,photonpoints,method='nearest', rescale=True)
    distances *= hydro_obj.length_scale.unit

    #print(key, hydro_obj.get_data('x1')[0], ph_r0[0], gridpoints[0,0], photonpoints[0,0], data_points[0], distances[0])

    differences=np.hypot((ph_x - distances[:,0]), (ph_y - distances[:,1]))

    max_diff=differences.max()

    hydro_obj.reset_spatial_limits()

    return data_points, max_diff

def load_photon_vs_fluid_quantities(savefile):
    """
    Loads a saved pickle file with hydrodynamic grid values as a function of photon position in the simulation.

    :param savefile: string of the pickle file that should be loaded without the .pickle ending.
    :return: dictionary with hydrodyanmic values as a function of photons' locations within the outflow
    """
    with open(savefile + '.pickle', 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        Temp_photon_2, Temp_flash_2, Avg_R_2, ph_num_2, avg_scatt_2, avg_gamma_2, avg_pres_2, avg_dens_2, P_2, Q_2, U_2, V_2, obs_theta = u.load()

        return_dict = dict(hydro_temp=Temp_flash_2, photon_temp=Temp_photon_2, avg_r=Avg_R_2, avg_scatt=avg_scatt_2, \
                           avg_gamma=avg_gamma_2, avg_pres=avg_pres_2, avg_pol=P_2,
                           avg_stokes=dict(Q=Q_2, U=U_2, V=V_2), \
                           photon_num=ph_num_2, obs_theta=obs_theta, avg_dens=avg_dens_2)

        return return_dict


def calculate_photon_vs_fluid_quantities(mcratload_obj, mcrat_obs_list, lc_dict_list, hydrosim_obj, savefile,\
                                         hydro_frame_min_max, hydrosim_dim=2, spherical_simulation=None):
    """
    Calculates, in parallel, the location of photons of interest in each hydrodynamic frame and the values of the outflow
    near these photons of interest. Thus, we can follow the evolution of the outflow as photons detected in different
    mock observable time bins interact with the specific regions of the outflow.

    :param mcratload_obj: The McratLoad object that was used to produce the MockObservation objects
    :param mcrat_obs_list: list of MockObservation objects
    :param lc_dict_list: list of the  MockObservation calculated light curves to specify the time bins of interest.
        In the same order as the mcrat_obs_list
    :param hydrosim_obj: A HydroSim object that contains all the information to access the hydrosimulation frames
    :param savefile: string of the file that will be saved with the calculated results without the .pickle ending
    :param hydro_frame_min_max: an array of the min/max hydro frames from which the analysis should be done to the frame
        at which the analysis will end
    :param hydrosim_dim: This is not necessary
    :param spherical_simulation: None or array of [lumnosity, gamma_infinity, saturation radius] to use for overwriting
        the hydro grid values with values of a spherical outflow with the specified parameters
    :return: dictionary with hydrodyanmic values as a function of photons' locations within the outflow
    """

    #find the maximum number of time bins in mcrat_obs_list that we need to account for
    max_t_steps=-1
    for i in lc_dict_list:
        if i['times'].size > max_t_steps:
            max_t_steps=i['times'].size

    if hydro_frame_min_max[1]<hydro_frame_min_max[0]:
        print("The list passed in denoting the minimum and maximum frames to conduct the analysis for is not ordered correctly.")

    photon_temp=np.zeros((max_t_steps, len(mcrat_obs_list), len(range(hydro_frame_min_max[1], hydro_frame_min_max[0]-1,-1))))

    folder = tempfile.mkdtemp()

    data_name = os.path.join(folder, 'Temp_photon')
    data_name_1 = os.path.join(folder, 'Temp_flash')
    data_name_2 = os.path.join(folder, 'Avg_R')
    data_name_3 = os.path.join(folder, 'dist_nearest_photon')
    data_name_4 = os.path.join(folder, 'ph_num')
    data_name_5 = os.path.join(folder, 'avg_scatt')
    data_name_6 = os.path.join(folder, 'avg_gamma')
    data_name_7 = os.path.join(folder, 'avg_pres')
    data_name_8 = os.path.join(folder, 'avg_dens')
    data_name_9 = os.path.join(folder, 'P')
    data_name_10 = os.path.join(folder, 'Q')
    data_name_11 = os.path.join(folder, 'U')
    data_name_12 = os.path.join(folder, 'V')

    #create memmaps for parallel processing of data
    Temp_photon = np.memmap(data_name, dtype=photon_temp.dtype, shape=photon_temp.shape, mode='w+')
    Temp_flash = np.memmap(data_name_1, dtype=Temp_photon.dtype, shape=photon_temp.shape, mode='w+')
    Avg_R = np.memmap(data_name_2, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    dist_nearest_photon = np.memmap(data_name_3, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    ph_num = np.memmap(data_name_4, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    avg_scatt = np.memmap(data_name_5, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    avg_gamma = np.memmap(data_name_6, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    avg_pres = np.memmap(data_name_7, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    avg_dens = np.memmap(data_name_8, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    P = np.memmap(data_name_9, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    Q = np.memmap(data_name_10, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    U = np.memmap(data_name_11, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')
    V = np.memmap(data_name_12, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')


    #actually call the function to calculate photon and fluid properties as functions of radius
    Parallel(n_jobs=-2, max_nbytes='1G')(
        delayed(calculate_photon_vs_fluid)(i, hydro_frame_min_max[1],
                                           mcratload_obj.file_directory,
                                           mcrat_obs_list, lc_dict_list,
                                           hydrosim_obj.fileroot_name, hydrosim_obj.file_directory,
                                           hydrosim_obj.hydrosim_type, hydrosim_obj.coordinate_sys,
                                           hydrosim_obj.density_scale, hydrosim_obj.length_scale, hydrosim_obj.velocity_scale, hydrosim_obj.datatype,
                                            spherical_simulation, Temp_photon, Temp_flash, Avg_R,
                                        dist_nearest_photon, ph_num,
                                        avg_scatt,avg_gamma, avg_pres,avg_dens,
                                         P, Q, U, V) for i in
                   range(hydro_frame_min_max[1], hydro_frame_min_max[0] - 1, -1))

    dump(Temp_photon, 'test_dump.dat')
    Temp_photon_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(Temp_flash, 'test_dump.dat')
    Temp_flash_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(Avg_R, 'test_dump.dat')
    Avg_R_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(dist_nearest_photon, 'test_dump.dat')
    dist_nearest_photon_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(ph_num, 'test_dump.dat')
    ph_num_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(avg_scatt, 'test_dump.dat')
    avg_scatt_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(avg_gamma, 'test_dump.dat')
    avg_gamma_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(avg_pres, 'test_dump.dat')
    avg_pres_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(avg_dens, 'test_dump.dat')
    avg_dens_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(P, 'test_dump.dat')
    P_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(Q, 'test_dump.dat')
    Q_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(U, 'test_dump.dat')
    U_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    dump(V, 'test_dump.dat')
    V_2 = load('test_dump.dat')
    os.remove('test_dump.dat')

    #print(np.array([Temp_photon_2, Temp_flash_2, Avg_R_2, ph_num_2, avg_scatt_2, avg_gamma_2, avg_pres_2, avg_dens_2, P_2, Q_2, U_2,
    #     V_2]).shape, Temp_photon_2.shape)

    obs_theta=np.array([i.theta_observer for i in mcrat_obs_list])

    # save file as pickle file
    f = open(savefile + '.pickle', 'wb')
    pickle.dump(
        [Temp_photon_2, Temp_flash_2, Avg_R_2, ph_num_2, avg_scatt_2, avg_gamma_2, avg_pres_2, avg_dens_2, P_2, Q_2, U_2,
         V_2, obs_theta], f)
    f.close()

    shutil.rmtree(folder)

    return_dict=dict(hydro_temp=Temp_flash_2, photon_temp=Temp_photon_2, avg_r=Avg_R_2, avg_scatt=avg_scatt_2, \
                     avg_gamma=avg_gamma_2, avg_pres=avg_pres_2, avg_pol=P_2, avg_stokes=dict(Q=Q_2, U=U_2, V=V_2), \
                     photon_num=ph_num_2, obs_theta=obs_theta, avg_dens=avg_dens_2)

    return return_dict

@unit.quantity_input(density_scale=['mass density'], length_scale=['length'], velocity_scale=['speed'])
def calculate_photon_vs_fluid(file_num, file_num_max, file_directory, mcrat_obs_list, lc_dict_list, fileroot_name, \
                              hydro_file_directory, hydrosim_type, coordinate_sys, \
                              density_scale, length_scale, velocity_scale, datatype, spherical_simulation, \
                            Temp_photon, Temp_flash, Avg_R, dist_nearest_photon, \
                              ph_num, avg_scatt, avg_gamma, avg_pres, avg_dens, P, Q, U, V):
    """
    Function that actually calculates the location of the photons of interest in a given hydro frame. It interpolates
    the properties of the outflow near the photons.

    :param file_num: frame number
    :param file_num_max: max frame number of the hydro simulation
    :param file_directory: the directory of the MCRaT simulation results
    :param mcrat_obs_list: list of MockObservation objects
    :param lc_dict_list: list of the  MockObservation calculated light curves to specify the time bins of interest
    :param fileroot_name: name of the save file of the produced pickle file
    :param hydro_file_directory: the directory of the hydro simulation files
    :param hydrosim_type: the type of the hydro simulation that is being analyzed
    :param coordinate_sys: the coordinate system of the hydro simulation that is being analyzed
    :param density_scale: the density scale of the hydro simulation that is being analyzed
    :param length_scale: the length scale of the hydro simulation that is being analyzed
    :param velocity_scale: the velocity scale of the hydro simulation that is being analyzed
    :param datatype: the hydro simulation frames' datatype
    :param spherical_simulation: None or array of [lumnosity, gamma_infinity, saturation radius] to use for overwriting
        the hydro grid values with values of a spherical outflow with the specified parameters
    :param Temp_photon: an empty array that will hold the calculated values
    :param Temp_flash: an empty array that will hold the calculated values
    :param Avg_R: an empty array that will hold the calculated values
    :param dist_nearest_photon: an empty array that will hold the calculated values
    :param ph_num: an empty array that will hold the calculated values
    :param avg_scatt: an empty array that will hold the calculated values
    :param avg_gamma: an empty array that will hold the calculated values
    :param avg_pres: an empty array that will hold the calculated values
    :param avg_dens: an empty array that will hold the calculated values
    :param P: an empty array that will hold the calculated values
    :param Q: an empty array that will hold the calculated values
    :param U: an empty array that will hold the calculated values
    :param V: an empty array that will hold the calculated values
    """

    print('Working on File Number: ' + np.str_(file_num))

    #load in the frame data
    mcrat_sim = McratSimLoad(file_directory)
    mcrat_sim.load_frame(file_num, read_comv=True)

    hydrosim_obj=HydroSim(fileroot_name, file_directory=hydro_file_directory, hydrosim_type=hydrosim_type, datatype=datatype, \
                          coordinate_sys=coordinate_sys, density_scale=density_scale,\
                 length_scale=length_scale, velocity_scale=velocity_scale, hydrosim_dim=mcrat_obs_list[0].hydrosim_dim)

    hydrosim_obj.load_frame(file_num)

    if spherical_simulation is not None:
        hydrosim_obj.make_spherical_outflow(spherical_simulation[0], spherical_simulation[1], spherical_simulation[2])

    #get photon r and theta
    r, theta=mcrat_sim.loaded_photons.get_spherical_coordinates(mcrat_obs_list[0].hydrosim_dim)

    ph_x,ph_y=mcrat_sim.loaded_photons.get_cartesian_coordinates(mcrat_obs_list[0].hydrosim_dim)

    #also look at lab frame 4 momentum's theta angle
    photon_velocity = np.array(
        [np.sqrt(mcrat_sim.loaded_photons.p1 ** 2 + mcrat_sim.loaded_photons.p2 ** 2), mcrat_sim.loaded_photons.p3]) \
                      * const.c.cgs.value / mcrat_sim.loaded_photons.p0
    photon_theta_velocity = np.rad2deg(np.arctan2(photon_velocity[0], photon_velocity[1]))

    #get each photons closest hydro value of velocity etc
    hydro_v0=hydro_position_interpolate(mcrat_sim.loaded_photons, hydrosim_obj, 'v0')[0]
    hydro_v1=hydro_position_interpolate(mcrat_sim.loaded_photons, hydrosim_obj, 'v1')[0]
    hydro_dens=hydro_position_interpolate(mcrat_sim.loaded_photons, hydrosim_obj, 'dens')[0]
    hydro_pres=hydro_position_interpolate(mcrat_sim.loaded_photons, hydrosim_obj, 'pres')[0]
    hydro_temp=hydro_position_interpolate(mcrat_sim.loaded_photons, hydrosim_obj, 'temp')[0]
    hydro_gamma=hydro_position_interpolate(mcrat_sim.loaded_photons, hydrosim_obj, 'gamma')[0]

    #for each observation of interest
    for i in range(len(mcrat_obs_list)):
        obs=mcrat_obs_list[i]
        lc =lc_dict_list[i]


        #for each time bin in the LC up to np.size(lc['times'])-1
        for t_idx in range(np.size(lc['times'])-1):
            #print(file_num, i,t_idx)
            #print(obs.theta_observer, lc['times'][t_idx], lc['times'][t_idx+1], mcrat_sim.loaded_photons.r0)
            x, y_t_min = calc_equal_arrival_time_surface(obs.theta_observer, file_num, obs.fps, None, None, lc['times'][t_idx].value, ph_x)
            x, y_t_max = calc_equal_arrival_time_surface(obs.theta_observer, file_num, obs.fps, None, None, lc['times'][t_idx+1].value, ph_x)


            #this may work for any type photon
            idx=np.where((ph_y<y_t_min) & (ph_y>=y_t_max)& (photon_theta_velocity<obs.theta_observer+0.5*obs.acceptancetheta_observer) \
                           & (photon_theta_velocity>obs.theta_observer-0.5*obs.acceptancetheta_observer))[0]

            #this works for just 'i' type potons
            #index=np.where((obs.detected_photons.detection_time>lc['times'][t_idx].value) & (obs.detected_photons.detection_time<=lc['times'][t_idx+1].value))[0]
            #if np.size(index)>0:
            #    index2=np.where(obs.detected_photons.file_index[index]<ph_x.size)[0]
            #    idx=obs.detected_photons.file_index[index][index2]
            #else:
            #    idx=[]

            #print(file_num, i,t_idx, w.size, y_t_min[0], y_t_max[0], ph_y[0], mcrat_sim.loaded_photons.comv_p0.sum(), idx)
            #if i==1 and file_num==259:
            #    stop

            if np.size(idx) > 0:
                w = mcrat_sim.loaded_photons.weight[idx]

                #get the phootn averaged values
                R_photon_avg=np.average(r[idx], weights=w)

                #if we dont have the photons comv 4 momentum we need to calculate it
                if mcrat_sim.loaded_photons.comv_p0.sum() == 0: #thsi is the correct one
                    #if mcrat_sim.loaded_photons.comv_p0 is None:
                    # get photon phi
                    ph_phi = np.arctan2(mcrat_sim.loaded_photons.r1[idx], mcrat_sim.loaded_photons.r0[idx])

                    #use the positions of the photons in the grid to get their local velocities
                    fluid_vx=hydro_v0[idx]
                    fluid_vy = hydro_v1[idx]

                    # have to build up the fluid beta
                    fluid_beta = np.array([fluid_vx * np.cos(ph_phi), fluid_vx * np.sin(ph_phi), fluid_vy])/const.c.cgs.value

                    # and the photon 4 momentum
                    ph_lab_4_p = np.array([mcrat_sim.loaded_photons.p0[idx], mcrat_sim.loaded_photons.p1[idx], \
                                           mcrat_sim.loaded_photons.p2[idx], mcrat_sim.loaded_photons.p3[idx]])

                    # deboost photon 4 momentum to fluid frame
                    deboosted_photon_4_p = lorentzBoostVectorized(fluid_beta, ph_lab_4_p)

                    # calculate average photon temp
                    #NEED TO PAY ATTENTION HERE IF I ADD UNITS TO PHOTON 4 MOMENTUM
                    photon_temp = np.average(calc_photon_temp(deboosted_photon_4_p[0, :] * const.c.cgs.value * u.erg), \
                                             weights=w)

                else:
                    photon_temp=mcrat_sim.loaded_photons.get_comv_energies( u.erg )[idx]*u.erg
                    #print(photon_temp)
                    #stop
                    photon_temp = np.average(calc_photon_temp(photon_temp), weights=w)


                #calculate averages for other quantities
                size = np.size(idx)
                scatt = np.average(mcrat_sim.loaded_photons.scatterings[idx], weights=w)
                d = np.average(hydro_dens[idx], weights=w)
                p = np.average(hydro_pres[idx], weights=w)
                g = np.average(hydro_gamma[idx], weights=w)
                fluid_temp =np.average(hydro_temp[idx], weights=w)
                if mcrat_sim.loaded_photons.s0 is not None:
                    stokes_q = np.average(mcrat_sim.loaded_photons.s1[idx], weights=w)
                    stokes_u = np.average(mcrat_sim.loaded_photons.s2[idx], weights=w)
                    stokes_v = np.average(mcrat_sim.loaded_photons.s3[idx], weights=w)
                    stokes_I = np.average(mcrat_sim.loaded_photons.s0[idx], weights=w)
                    pol = np.sqrt(stokes_q ** 2 + stokes_u ** 2) / stokes_I
                else:
                    stokes_q=np.nan
                    stokes_u = np.nan
                    stokes_v = np.nan
                    stokes_I = np.nan
                    pol = np.nan

                #print(photon_temp, fluid_temp)
                Temp_photon[t_idx, i, file_num_max - file_num] = photon_temp.value
                Temp_flash[t_idx, i, file_num_max - file_num] = fluid_temp.value
                Avg_R[t_idx, i, file_num_max - file_num] = R_photon_avg
                ph_num[t_idx, i, file_num_max - file_num] = size
                avg_scatt[t_idx, i, file_num_max - file_num] = scatt
                avg_gamma[t_idx, i, file_num_max - file_num] = g
                avg_pres[t_idx, i, file_num_max - file_num] = p.value
                avg_dens[t_idx, i, file_num_max - file_num] = d.value
                dist_nearest_photon[t_idx, i, file_num_max - file_num] = np.nan
                P[t_idx, i, file_num_max - file_num] = pol
                Q[t_idx, i, file_num_max - file_num] = stokes_q
                U[t_idx, i, file_num_max - file_num] = stokes_u
                V[t_idx, i, file_num_max - file_num] = stokes_v

                #print(Temp_photon[t_idx, i, file_num_max - file_num], avg_gamma[t_idx, i, file_num_max - file_num])

            else:
                Temp_photon[t_idx, i, file_num_max - file_num] = np.nan
                Temp_flash[t_idx, i, file_num_max - file_num] = np.nan
                Avg_R[t_idx, i, file_num_max - file_num] = np.nan
                ph_num[t_idx, i, file_num_max - file_num] = np.nan
                avg_scatt[t_idx, i, file_num_max - file_num] = np.nan
                avg_gamma[t_idx, i, file_num_max - file_num] = np.nan
                avg_pres[t_idx, i, file_num_max - file_num] = np.nan
                avg_dens[t_idx, i, file_num_max - file_num] = np.nan
                dist_nearest_photon[t_idx, i, file_num_max - file_num] = np.nan
                P[t_idx, i, file_num_max - file_num] = np.nan
                Q[t_idx, i, file_num_max - file_num] = np.nan
                U[t_idx, i, file_num_max - file_num] = np.nan
                V[t_idx, i, file_num_max - file_num] = np.nan

        if t_idx<Temp_photon.shape[0]:
            Temp_photon[t_idx:, i, file_num_max - file_num]=np.nan
            Temp_flash[t_idx:, i, file_num_max - file_num] = np.nan
            Avg_R[t_idx:, i, file_num_max - file_num] = np.nan
            ph_num[t_idx:, i, file_num_max - file_num] = np.nan
            avg_scatt[t_idx:, i, file_num_max - file_num] = np.nan
            avg_gamma[t_idx:, i, file_num_max - file_num] = np.nan
            avg_pres[t_idx:, i, file_num_max - file_num] = np.nan
            avg_dens[t_idx:, i, file_num_max - file_num] = np.nan
            dist_nearest_photon[t_idx:, i, file_num_max - file_num] = np.nan
            P[t_idx:, i, file_num_max - file_num] = np.nan
            Q[t_idx:, i, file_num_max - file_num] = np.nan
            U[t_idx:, i, file_num_max - file_num] = np.nan
            V[t_idx:, i, file_num_max - file_num] = np.nan
