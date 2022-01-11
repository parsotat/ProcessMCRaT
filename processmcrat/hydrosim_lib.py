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

    #try to identify where the observed photon coordinates are encompassed by the hydrogrid
    i=2
    #for flash, in cartesian coordinates therefore convert MCRaT 3D coord to 2D cartesian coord
    ph_r0, ph_r1=photons.get_cartesian_coordinates(hydro_obj.dimensions)*hydro_obj.length_scale.unit

    #apply spatial limits in hydro coordinate system
    hydro_obj.apply_spatial_limits(ph_r0.min()*(1-(.05*i)), ph_r0.max()*(1+(.05*i)),\
                                   ph_r1.min()*(1-(.05*i)), ph_r1.max()*(1+(.05*i)))

    while (hydro_obj.spatial_limit_idx.size == 0):
        i+=1
        #find indexes where x and z are close to points of interest to limit possibilities for finding differences
        hydro_obj.apply_spatial_limits(ph_r0.min() * (1 - (.05 * i)), ph_r0.max() * (1 + (.05 * i)),
                                       ph_r1.min() * (1 - (.05 * i)), ph_r1.max() * (1 + (.05 * i)))

    gridpoints=np.zeros((hydro_obj.spatial_limit_idx.size,2))
    gridpoints[:,0]=hydro_obj.get_data('x0')
    gridpoints[:,1]=hydro_obj.get_data('x1')

    if key in hydro_obj.hydro_data:
        data=hydro_obj.get_data(key)
    else:
        print(key+" is not a key in the HydroSim object")


    photonpoints=np.zeros((np.size(ph_r0),2))
    photonpoints[:,0]=ph_r0
    photonpoints[:,1]=ph_r1

    data_points=griddata(gridpoints,data,photonpoints,method='nearest')
    data_points *= data.unit
    distances=griddata(gridpoints,gridpoints,photonpoints,method='nearest')
    distances *= hydro_obj.length_scale.unit

    #print(key, hydro_obj.get_data('x1')[0], ph_r0[0], gridpoints[0,0], photonpoints[0,0], data_points[0], distances[0])

    differences=np.hypot((ph_r0 - distances[:,0]), (ph_r1 - distances[:,1]))

    max_diff=differences.max()

    hydro_obj.reset_spatial_limits()

    return data_points, max_diff

def calculate_photon_vs_fluid_quantities(mcratload_obj, mcrat_obs_list, lc_dict_list, hydrosim_obj, savefile,\
                                         hydro_frame_min_max, hydrosim_dim=2, spherical_simulation=None):

    #find the maximum number of time bins in mcrat_obs_list that we need to account for
    max_t_steps=-1
    for i in lc_dict_list:
        if i['times'].size > max_t_steps:
            max_t_steps=i['times'].size

    if hydro_frame_min_max[1]<hydro_frame_min_max[0]:
        print("The list passed in denoting the minimum and maximum frames to conduct the analysis for is not oredered correctly.")

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
    Temp_photon = np.memmap(data_name, dtype=photon_temp.dtype, shape=photon_temp.shape, mode='w+')* np.nan
    Temp_flash = np.memmap(data_name_1, dtype=Temp_photon.dtype, shape=photon_temp.shape, mode='w+')* np.nan
    Avg_R = np.memmap(data_name_2, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    dist_nearest_photon = np.memmap(data_name_3, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    ph_num = np.memmap(data_name_4, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    avg_scatt = np.memmap(data_name_5, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    avg_gamma = np.memmap(data_name_6, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    avg_pres = np.memmap(data_name_7, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    avg_dens = np.memmap(data_name_8, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    P = np.memmap(data_name_9, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    Q = np.memmap(data_name_10, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    U = np.memmap(data_name_11, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan
    V = np.memmap(data_name_12, dtype=Temp_photon.dtype, shape=Temp_photon.shape, mode='w+')* np.nan

    #actually call the function to calculate photon and fluid properties as functions of radius
    Parallel(n_jobs=1)(
        delayed(calculate_photon_vs_fluid)(file_num=i, file_num_max=hydro_frame_min_max[1],
                                           file_directory=mcratload_obj.file_directory,
                                           mcrat_obs_list=mcrat_obs_list, lc_dict_list=lc_dict_list,
                                           fileroot_name=hydrosim_obj.fileroot_name, hydro_file_directory=hydrosim_obj.file_directory,
                                           hydrosim_type=hydrosim_obj.hydrosim_type, coordinate_sys=hydrosim_obj.coordinate_sys,
                                           density_scale=hydrosim_obj.density_scale, length_scale=hydrosim_obj.length_scale, velocity_scale=hydrosim_obj.velocity_scale,
                                            spherical_simulation=spherical_simulation, Temp_photon=Temp_photon, Temp_flash=Temp_flash, Avg_R=Avg_R,
                                        dist_nearest_photon=dist_nearest_photon, ph_num=ph_num,
                                        avg_scatt=avg_scatt, avg_gamma=avg_gamma, avg_pres=avg_pres, avg_dens=avg_dens,
                                         P=P, Q=Q, U=U, V=V) for i in
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

    print(np.array([Temp_photon_2, Temp_flash_2, Avg_R_2, ph_num_2, avg_scatt_2, avg_gamma_2, avg_pres_2, avg_dens_2, P_2, Q_2, U_2,
         V_2]).shape, Temp_photon_2.shape)

    # save file as pickle file
    f = open(savefile + '.pickle', 'wb')
    pickle.dump(
        [Temp_photon_2, Temp_flash_2, Avg_R_2, ph_num_2, avg_scatt_2, avg_gamma_2, avg_pres_2, avg_dens_2, P_2, Q_2, U_2,
         V_2], f)
    f.close()

    shutil.rmtree(folder)

    return_dict=dict(hydro_temp=Temp_flash_2, photon_temp=Temp_photon_2, avg_r=Avg_R_2, avg_scatt=avg_scatt_2, \
                     avg_gamma=avg_gamma_2, avg_pres=avg_pres_2, avg_pol=P_2, avg_stokes=dict(Q=Q_2, U=U_2, V=V_2))

    return return_dict


def calculate_photon_vs_fluid(file_num, file_num_max, file_directory, mcrat_obs_list, lc_dict_list, fileroot_name, \
                              hydro_file_directory, hydrosim_type, coordinate_sys, \
                              density_scale, length_scale, velocity_scale, \
                            Temp_photon, Temp_flash, Avg_R, dist_nearest_photon, \
                              ph_num, avg_scatt, avg_gamma, avg_pres, avg_dens, P, Q, U, V, spherical_simulation=None):

    print('Working on File Number: ' + np.str_(file_num))

    #load in the frame data
    mcrat_sim = McratSimLoad(file_directory)
    mcrat_sim.load_frame(file_num)

    hydrosim_obj=HydroSim(fileroot_name, file_directory=hydro_file_directory, hydrosim_type=hydrosim_type, \
                          coordinate_sys=coordinate_sys, density_scale=density_scale,\
                 length_scale=length_scale, velocity_scale=velocity_scale, hydrosim_dim=mcrat_obs_list[0].hydrosim_dim)

    hydrosim_obj.load_frame(file_num)

    if spherical_simulation is not None:
        hydrosim_obj.make_spherical_outflow(spherical_simulation[0], spherical_simulation[1], spherical_simulation[2])

    #get photon r and theta
    r, theta=mcrat_sim.loaded_photons.get_spherical_coordinates(mcrat_obs_list[0].hydrosim_dim)


    ph_x,ph_y=mcrat_sim.loaded_photons.get_cartesian_coordinates(mcrat_obs_list[0].hydrosim_dim)

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
            print(i,t_idx)
            #print(lc['times'][t_idx], lc['times'][t_idx+1], mcrat_sim.loaded_photons.r0)
            x, y_t_min = calc_equal_arrival_time_surface(obs.theta_observer, file_num, obs.fps, None, None, lc['times'][t_idx].value, ph_x)
            x, y_t_max = calc_equal_arrival_time_surface(obs.theta_observer, file_num, obs.fps, None, None, lc['times'][t_idx+1].value, ph_x)

            idx=np.where((ph_y<y_t_min) & (ph_y>=y_t_max))[0]

            w=mcrat_sim.loaded_photons.weight[idx]

            print(w.size, y_t_min[0], y_t_max[0], ph_y[0], mcrat_sim.loaded_photons.comv_p0.sum())
            #stop

            if np.size(idx) > 0:

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
                    photon_temp = np.average(calc_photon_temp(deboosted_photon_4_p[0, :] * const.c.cgs.value), \
                                             weights=w)


                else:
                    photon_temp=mcrat_sim.loaded_photons.get_comv_energies( u.erg )[idx]*u.erg
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