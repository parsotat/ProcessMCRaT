# function to read files and assign data to variables
#and process data from MC_RAT and FLASH
from __future__ import division
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys

#import os
#import glob

#plt.style.use('ggplot')

def read_mcrat(file_name):
    """
    Legacy code to read in MCRaT data when it used to output data in text files

    :param file_name: the file name of the MCRaT data file including the directory that it is located
    :return: returns all the data in the given file of interest
    """
    print('>> Reading Files: '+file_name+'...')

    ns = np.fromfile(file_name + '_NS.dat', sep='\t', dtype=float, count=-1)
    p0 = np.fromfile(file_name + '_P0.dat', sep='\t', dtype=float, count=-1)
    p1 = np.fromfile(file_name + '_P1.dat', sep='\t', dtype=float, count=-1)
    p2 = np.fromfile(file_name + '_P2.dat', sep='\t', dtype=float, count=-1)
    p3 = np.fromfile(file_name + '_P3.dat', sep='\t', dtype=float, count=-1)
    r1 = np.fromfile(file_name + '_R1.dat', sep='\t', dtype=float, count=-1)
    r2 = np.fromfile(file_name + '_R2.dat', sep='\t', dtype=float, count=-1)
    r3 = np.fromfile(file_name + '_R3.dat', sep='\t', dtype=float, count=-1)

    return ns, p0, p1, p2, p3, r1, r2, r3

def read_mcrat_h5(file_name, read_comv=False, read_stokes=False, read_type=False):
    """
    Reads in MCRaT data for current version of MCRaT that outputs data in hdf5 files. Also has support for various
    MCRaT switches that can be turned on by the user.

    :param file_name: the file name of the MCRaT data file including the directory that it is located
    :param read_comv: switch that lets the function know if it should expect/ return comoving 4 momenta data, set to true
                    if this switch is set to ON in mcrat_input.h
    :param read_stokes: switch that lets the function know if it should expect/ return stokes parameters, set to true
                    if this switch is set to ON in mcrat_input.h
    :param read_type: switch that lets the function know if it should expect/ return photon type, set to true
                    if this switch is set to ON in mcrat_input.h
    :return: returns the data read in from the MCRaT data frame
    """
    import h5py as h5
    with h5.File(file_name+'.h5', 'r') as f:
        pw = f['PW'].value
        ns=f['NS'].value
        p0=f['P0'].value
        p1=f['P1'].value
        p2=f['P2'].value
        p3=f['P3'].value
        r0=f['R0'].value
        r1=f['R1'].value
        r2=f['R2'].value
        if read_stokes:
            s0=f['S0'].value
            s1=f['S1'].value
            s2=f['S2'].value
            s3=f['S3'].value
        if read_comv:
            comv_p0 = f['COMV_P0'].value
            comv_p1 = f['COMV_P1'].value
            comv_p2 = f['COMV_P2'].value
            comv_p3 = f['COMV_P3'].value
        if read_type:
            pt = f['PT'].value
            pt=np.array([i for i in bytes(pt).decode()])

    if read_comv and read_stokes:
        if read_type:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2, s0, s1, s2, s3, comv_p0, comv_p1, comv_p2, comv_p3, pt
        else:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2, s0, s1, s2, s3, comv_p0, comv_p1, comv_p2, comv_p3
    elif read_comv and not read_stokes:
        if read_type:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2, comv_p0, comv_p1, comv_p2, comv_p3, pt
        else:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2, comv_p0, comv_p1, comv_p2, comv_p3
    elif not read_comv and read_stokes:
        if read_type:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2, s0, s1, s2, s3, pt
        else:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2, s0, s1, s2, s3
    else:
        if read_type:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2, pt
        else:
            return pw, ns, p0, p1, p2, p3, r0, r1, r2


def read_flash(fnam, length_scale=1e9, make1D=True):
    """
    Legacy code that was used to read in FLASH data frames in the old python version of the code, still is used in some
    functions comparing MCRaT information to fluid quantities

    :param fnam: the file name fo the FLASH file, including directory it is located in
    :param make1D: switch to set the output to be either 1D, if set to True (the default), or keep the returned
                    information as 2D arrays
    :return: returns all of the FLASH file information (x, y, dx, dy, vx, vy, gamma, density, lab density, radius,
            theta, pressure) at each hydro node
    """
    # read FLASH data, modified from readanddecimate
    import tables as t
    file = t.open_file(fnam)
    print( '>> read_flash: Reading positional, density, pressure, and velocity information...')
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
    #print( '>> read_flash: DONE Reading positional, density, pressure, and velocity information...')

    #print( '>> read_flash: Creating the full x and y arrays...')
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

    #print( '>> read_flash: Selecting good node types (=1)...')
    nty = file.get_node('/', 'node type')
    nty = nty.read()
    file.close()
    jj = np.where(nty == 1)
    xx = np.array(xx[jj, 0, :, :]) * length_scale
    #  yy=np.array(yy[jj,0,:,:]+1) this takes care of the fact that lngths scales with 1e9
    yy = np.array(yy[jj, 0, :, :]) * length_scale
    szxx = np.array(szxx[jj, 0, :, :]) * length_scale
    szyy = np.array(szyy[jj, 0, :, :]) * length_scale
    vx = np.array(vx[jj, 0, :, :])
    vy = np.array(vy[jj, 0, :, :])
    dens = np.array(dens[jj, 0, :, :])
    pres = np.array(pres[jj, 0, :, :])

    if make1D:
        xx = np.reshape(xx, xx.size)
        vx = np.reshape(vx, xx.size)
        yy = np.reshape(yy, yy.size)
        szxx = np.reshape(szxx, xx.size)
        szyy = np.reshape(szyy, yy.size)
        vy = np.reshape(vy, yy.size)
        gg = 1. / np.sqrt(1. - (vx ** 2 + vy ** 2))
        dd = np.reshape(dens, dens.size)
        dd_lab = dd * gg
        rr = np.sqrt(xx ** 2 + yy ** 2)
        tt = np.arctan2(xx, yy)
        pp = np.reshape(pres, pres.size)
    else:
        dd=dens
        pp=pres
        gg = 1. / np.sqrt(1. - (vx ** 2 + vy ** 2))
        dd_lab = dd * gg
        rr = np.sqrt(xx ** 2 + yy ** 2)
        tt = np.arctan2(xx, yy)


    return xx, yy, szxx, szyy, vx, vy, gg, dd, dd_lab, rr, tt, pp

def calc_kislat_error(s0,s1,s2,s3,weights,mu=1):
    #calc the q and u normalized by I
    I=np.sum(weights) #np.mean(weights) #np.sum(weights)
    i=np.average(s0, weights=weights)
    q=np.average(s1, weights=weights)
    u=np.average(s2, weights=weights)
    v = np.average(s3, weights=weights)
    p=np.sqrt(q**2+u**2)
    chi = (0.5 * np.arctan2(u, q) * 180 / np.pi)
    W_2=np.sum(weights**2) #np.mean(weights**2) #np.sum(weights**2)

    mu_factor=2/mu
    var_factor=W_2/I**2

    #convert q and u to reconstructed values that kislat uses
    Q_r=mu_factor*q
    U_r=mu_factor*u
    p_r=mu_factor*p

    #calculate the standard deviation in Q_R and U_r and covariance
    sigma_Q_r=np.sqrt(var_factor*(mu_factor/mu-Q_r**2))
    sigma_U_r=np.sqrt(var_factor*(mu_factor/mu-U_r**2))
    if (np.isnan(sigma_Q_r)):
        #in case there is some error with the argument of sqrt being negative (happens rarely and am not sure why)
        sigma_Q_r = np.sqrt(var_factor * np.abs(mu_factor / mu - Q_r ** 2))
    if (np.isnan(sigma_U_r)):
        sigma_U_r = np.sqrt(var_factor * np.abs(mu_factor / mu - U_r ** 2))

    cov= -var_factor*Q_r*U_r
    #print('var factor', var_factor, 'W_2', W_2, 'I', I,  'mean value of W_2', np.mean(weights**2), 'mean value of I', np.mean(weights), 'leads to', np.mean(weights**2)/np.mean(weights)**2)
    #print(Q_r, U_r, sigma_U_r, sigma_Q_r, cov)
    #calculate the partial derivatives
    partial_pr_Qr=Q_r/p_r
    partial_pr_Ur=U_r/p_r
    partial_phir_Qr=-0.5*u/p**2/mu_factor #dq/dQ_r=2/mu, and do (d phi/dq)*(dq/dQ_r)
    partial_phir_Ur=-0.5*q/p**2/mu_factor

    #calculate the error in pr and chi
    sigma_pr=np.sqrt( (partial_pr_Qr*sigma_Q_r)**2 + (partial_pr_Ur*sigma_U_r)**2 + 2*partial_pr_Qr*partial_pr_Ur*cov )
    sigma_chi=np.sqrt( (partial_phir_Qr*sigma_Q_r)**2 + (partial_phir_Ur*sigma_U_r)**2 + 2*partial_phir_Qr*partial_phir_Ur*cov )
    if (np.isnan(sigma_pr)):
        sigma_pr = np.sqrt(np.abs((partial_pr_Qr * sigma_Q_r) ** 2 + (
                    partial_pr_Ur * sigma_U_r) ** 2 + 2 * partial_pr_Qr * partial_pr_Ur * cov))

    return i, q, u, v, p, chi, sigma_pr/mu_factor, (180/np.pi)*sigma_chi


def get_times_angles(v, t, min_angle, max_angle, change_in_angle,angle_range, r , angle):

    #get radial positions of all photons
    #r=np.sqrt(x**2 + z**2) #only concerned with z and x plane

    #get angle position of all photons in degrees
    #angle=np.arctan2(x,z)*(180/np.pi)

    #print('>> Finding where virtual observer Is')
    vir_obs=r.max()

    #print('>> Finding How long it takes each photon to get to the virtual observer ')
    #find out how long it takes each photon to get to the flat virtual observer

    #rotate photons to z axis

    #calculate how long it takes for them to get to flat virtual observer
    times=np.zeros(r.size)
    #times= (vir_obs-r) / v #how to find v?????????????????????????????????????

    #make matrix with all of indices of photons at different angles
    #get size of matrix
    #len(range(0,max_angle,delta_angle)) number of columns for each angle

    #find number of rows or max number of photons in a given angle range
    #convert to radians if need be
    #delta_angle_rad=delta_angle*(np.pi/180)
    #max_angle_rad=max_angle*(np.pi/180)
    #angle_range_rad=angle_range*(np.pi/180)

    #print('>> Initializing and filling the arrays telling what angle each photon is at and how long each photon takes to get to the virtual observer ')

    #make matrix, with proper number of rows and columns

    angle_indexes=np.empty((r.size, max_angle),dtype=bool)

    #fill up matrix

    for theta in range(min_angle,max_angle,change_in_angle):
        #print(theta)
        #find indexes
        angle_indexes[:,theta]= np.logical_and(angle>theta-angle_range,angle<=theta+angle_range)

        #get photons at the angle theta and project them to z axis
        #rotated_r= r[angle_indexes[:,theta]] * np.cos(theta*(np.pi/180))
        vir_obs_at_theta=vir_obs/np.cos(theta*(np.pi/180))

        times[angle_indexes[:,theta]]=(vir_obs_at_theta-r[angle_indexes[:,theta]])/v #look into this here!!
        #times=(vir_obs_at_theta-r)/v

    #find max time to get to virtual observer
    max_time=times.max()
    #print(max_time)
    #stop

    #find number of time steps
    time_steps=np.int(np.ceil(max_time/t))


    return time_steps, times, angle_indexes

def get_times_angles_jet(r_obs,lastfile,fps, R1,R2,R3,P1,P2,P3, delta_t, max_angle, change_in_angle,angle_range):
    c_light=3e10
    #print(R1.size,max_angle, change_in_angle,angle_range,delta_t)
    times=np.zeros(R1.size)
    angle_indexes=np.empty((R1.size, max_angle),dtype=bool)
    angle_indexes[:]=False
    times[:]=np.nan

    RR=np.sqrt(R1**2+R2**2+R3**2)  # radius of propagation
    theta_pos=np.arctan2(np.sqrt(R1**2+R2**2),R3) # angle between position vector and polar axis
    theta_pho=np.arctan2(np.sqrt(P1**2+P2**2),P3) # angle between velocity vector and polar axis
    total=0
    for theta in range(0,max_angle,change_in_angle):

        theta_rel=theta_pos-(theta*np.pi/180)  # angle between position vector and line of sight
        RR_prop=RR*np.cos(theta_rel)
        print(RR_prop.min(), RR_prop.max())


        jj=np.where((theta_pho>((theta-angle_range)*np.pi/180) )&(theta_pho<= ((theta+angle_range)*np.pi/180))&(RR_prop>=r_obs))
        angle_indexes[jj,theta]=True

        print( 'accepted photons ',jj[0].size)
        nnn=jj[0].size
        tnow=lastfile/fps
        dr=RR_prop[jj]-r_obs
        vel_proj=c_light*np.cos(theta_pho[jj]-(theta*np.pi/180))
        dt=dr/vel_proj
        times[jj]=tnow-dt-r_obs/c_light
        total+=nnn
    print(times)
    print(np.where(times>0), times, (times[np.where(times>0)].max())/delta_t, np.ceil((times[np.where(times>0)].max())/delta_t), np.int_(np.ceil(times[np.where(times>0)].max()/delta_t)))
    time_steps=np.int_(np.ceil(times[np.where(times>0)].max()/delta_t))

    return time_steps, times, angle_indexes

def flash_position(r_photon_avg, theta_photon_avg, rr, tt_deg):
    #get position in FLASH
    i=1
    size=0

    #find indexes of angles very close to angle of interest
    #Flash_deg_indexes=np.logical_and(theta_photon_avg<tt_deg+0.1, theta_photon_avg>=tt_deg-0.1)
    Flash_deg_indexes=np.logical_and(theta_photon_avg*(1-(.05*i))<tt_deg, theta_photon_avg*(1+(.05*i))>=tt_deg)
    #Flash_r_indexes=np.logical_and(rr<R_photon_avg+(1*10**i), rr>R_photon_avg-(1*10**i))
    #size=np.size(rr[Flash_deg_indexes*Flash_r_indexes])

    while (size == 0):

        #if indicies of two conditions dont match up, keep increasing range for rr
        Flash_r_indexes=np.logical_and(rr<r_photon_avg*(1+(.05*i)), rr>r_photon_avg*(1-(.05*i)))
        closest_index=Flash_deg_indexes*Flash_r_indexes
        size=np.size(rr[closest_index])
        i+=1
        #    i+=1
        #print(i)
        #Flash_r_indexes=np.logical_and(rr<r_photon_avg+(1*10**i), rr>r_photon_avg-(1*10**i))
        #    size=np.size(rr[Flash_deg_indexes*Flash_r_indexes])
        #another_method=np.logical_and(np.logical_and(Theta_photon_avg<tt_deg+0.1, Theta_photon_avg>=tt_deg-0.1), np.logical_and(rr<R_photon_avg+(1*10**i), rr>R_photon_avg-(1*10**i)) )
    #print('Size='+np.str(size))

    #refine which rr and tt_deg is closest to desired value
    size_count=0
    while (size>1):
        if size_count==0:
            #find out which one is closest in rr
            #selected_r=rr[Flash_deg_indexes*Flash_r_indexes]
            selected_r=rr[closest_index]
            #selected_theta=tt_deg[Flash_deg_indexes*Flash_r_indexes]
            #find index of min of selected_r, sine already have angles tht are close to what we want
            closest_index=selected_r[np.abs(selected_r-r_photon_avg).argmin()]==rr
            closest_index*=Flash_deg_indexes #multiply by indexes for angles again to make sure there is only 1
                                             #one value that we get out
            size_count+=1
        else:
            #find out which one is closest in theta
            selected_tt_deg=tt_deg[closest_index]
            closest_index_2=selected_tt_deg[np.abs(selected_tt_deg-theta_photon_avg).argmin()]==tt_deg
            closest_index*=closest_index_2
            size_count=0 # may need to restart loop if size !=1

        size=np.size(rr[closest_index])


    #print('Average R='+np.str(r_photon_avg)+ ' The acquired r is: '+ np.str(rr[closest_index]))
    #print('Average theta='+np.str(theta_photon_avg)+ ' The acquired theta is: '+ np.str(tt_deg[closest_index]))

    return closest_index

def find_angles_only(max_angle, change_in_angle,angle_range, angle):
    #print('Finding angles only')
    angle_indexes=np.empty((angle.size, max_angle),dtype=bool)

    for theta in range(0,max_angle,change_in_angle):
        #print(theta)
        #find indexes
        angle_indexes[:,theta]= np.logical_and(angle>theta-angle_range,angle<=theta+angle_range)

    return angle_indexes

def nearest_photon_dist(flash_r, flash_theta, photon_r, photon_theta):
    #find closest photon to flash position and take difference
    flash_x=flash_r*np.sin(flash_theta*(np.pi/180))
    flash_z=flash_r*np.cos(flash_theta*(np.pi/180))

    photon_x=photon_r*np.sin(photon_theta*(np.pi/180))
    photon_z=photon_r*np.cos(photon_theta*(np.pi/180))

    diff_x=photon_x-flash_x
    diff_z=photon_z-flash_z

    total_diff=np.sqrt(diff_x**2 + diff_z**2)

    return total_diff.min()


def flash_position_ordered(x_photon_avg, z_photon_avg, flash_x, flash_z):
    diff_magnitude=np.sqrt((x_photon_avg-flash_x)**2 + (z_photon_avg-flash_z)**2)

    index=diff_magnitude.argmin()

    print('Average X='+np.str(x_photon_avg)+ ' The acquired X is: '+ np.str(flash_x[index]))
    print('Average Z='+np.str(z_photon_avg)+ ' The acquired Z is: '+ np.str(flash_z[index]))
    print('Difference is: ',diff_magnitude[index] )

    return index

def flash_position_broadcasting(mc_x_eff,R3,xx,yy,photon_2_constraints):

    i=0
    test_index=np.where(np.logical_and(np.logical_and(mc_x_eff[photon_2_constraints].min()*(1-(.05*i))<xx, mc_x_eff[photon_2_constraints].max()*(1+(.05*i))>=xx),np.logical_and(R3[photon_2_constraints].min()*(1-(.05*i))<yy, R3[photon_2_constraints].max()*(1+(.05*i))>=yy) ))

    while (xx[test_index].size == 0):
        i+=1
        #find indexes where x and z are close to points of interest to limit possibilities for finding differences
        test_index=np.where(np.logical_and(np.logical_and(mc_x_eff[photon_2_constraints].min()*(1-(.05*i))<xx, mc_x_eff[photon_2_constraints].max()*(1+(.05*i))>=xx),np.logical_and(R3[photon_2_constraints].min()*(1-(.05*i))<yy, R3[photon_2_constraints].max()*(1+(.05*i))>=yy) ))
    #stop
    #set up array
    differences=np.zeros((np.size(mc_x_eff[photon_2_constraints]), np.size(xx[test_index])))

    #count=0
    #for i in range(0,len(R_photon[photon_2_constraints])):
    #    index=r_p.flash_position(R_photon[photon_2_constraints][i],Theta_photon[photon_2_constraints][i], rr,tt_deg)
    #    Flash_pressure_at_photon[i]=pp[index]
    #    gamma_at_photon[i]=gg[index]

    #calculate distances
    differences=np.sqrt( (mc_x_eff[photon_2_constraints,np.newaxis] - xx[np.newaxis,test_index])**2 + (R3[photon_2_constraints,np.newaxis] - yy[np.newaxis,test_index])**2 )[0,:,:]

    #get arguments of min differences for each photon
    min_arg=differences.argmin(axis=1)

    #get index for these specific values in array with all indexes that we are interested in
    val_index=test_index[0][min_arg]

    return val_index

def flash_position_interpolate(mc_x_eff,R3,xx,yy,pressure, gamma,dens, photon_2_constraints):

    i=2
    test_index=np.where(np.logical_and(np.logical_and(mc_x_eff[photon_2_constraints].min()*(1-(.05*i))<xx, mc_x_eff[photon_2_constraints].max()*(1+(.05*i))>=xx),np.logical_and(R3[photon_2_constraints].min()*(1-(.05*i))<yy, R3[photon_2_constraints].max()*(1+(.05*i))>=yy) ))
    #
    while (xx[test_index].size == 0):
        i+=1
        #find indexes where x and z are close to points of interest to limit possibilities for finding differences
        test_index=np.where(np.logical_and(np.logical_and(mc_x_eff[photon_2_constraints].min()*(1-(.05*i))<xx, mc_x_eff[photon_2_constraints].max()*(1+(.05*i))>=xx),np.logical_and(R3[photon_2_constraints].min()*(1-(.05*i))<yy, R3[photon_2_constraints].max()*(1+(.05*i))>=yy) ))
    #stop

    gridpoints=np.zeros((np.size(xx[test_index]),2))
    gridpoints[:,0]=xx[test_index]
    gridpoints[:,1]=yy[test_index]

    #gridpoints=np.zeros((np.size(xx),2))
    #gridpoints[:,0]=xx
    #gridpoints[:,1]=yy

    photonpoints=np.zeros((np.size(mc_x_eff[photon_2_constraints]),2))
    photonpoints[:,0]=mc_x_eff[photon_2_constraints]
    photonpoints[:,1]=R3[photon_2_constraints]

    pressure_points=griddata(gridpoints,pressure[test_index],photonpoints,method='nearest')
    gamma_points=griddata(gridpoints,gamma[test_index],photonpoints,method='nearest')
    dens_points=griddata(gridpoints,dens[test_index],photonpoints,method='nearest')
    #gamma_points=griddata(gridpoints,gamma,photonpoints,method='nearest')
    distances=griddata(gridpoints,gridpoints,photonpoints,method='nearest')
    #print(distances[:,0].shape)
    #find magnitude of distances
    #calculate distances
    #differences=np.sqrt( (mc_x_eff[photon_2_constraints] - distances[:,0])**2 + (R3[photon_2_constraints,np.newaxis] - yy[np.newaxis,test_index])**2 )[0,:,:]
    differences=np.hypot((mc_x_eff[photon_2_constraints] - distances[:,0]), (R3[photon_2_constraints] - distances[:,1]))

    #plt.scatter(mc_x_eff[photon_2_constraints],R3[photon_2_constraints], marker='o', c='b')
    #plt.scatter(distances[:,0], distances[:,1], marker='.', c='r')
    #plt.show()
    #stop
    #get arguments of min differences for each photon
    max_diff=differences.max()

    return pressure_points,gamma_points,dens_points,max_diff

def Band(x,a,b,c,d):
    """
    Function that evaluates the Band function

    :param x: x is the energy in keV
    :param a: alpha parameter
    :param b: beta parameter
    :param c: break energy in keV
    :param d: normalization of the returned spectrum
    :return: Returns the Band function evaluated for the parameters of interest (units of cts/s)
    """
    #a is alpha, b is beta, c is break energy
    model=np.empty(x.size)
    kk=np.where(x<((a-b)*c))
    if kk[0].size>0:
        model[kk]=x[kk]**a*np.exp(-x[kk]/c)
        kk=np.where(x>=((a-b)*c))
    if kk[0].size>0:
        model[kk]=((a-b)*c)**(a-b)*x[kk]**(b)*np.exp(b-a)
    model=model/np.trapz(model,x=x)*d
    return model

def wien(x, t,d):
    """
    Function that produces a Wien spectrum.

    :param x: Energy in keV
    :param t: Temp in kelvin
    :param d: Normalization of the spectrum
    :return: returns the Wien spectrum parameterized by some temperature at the energies of interest
    """
    h=6.6260755e-27
    c_light=29979245800.0
    kB=1.380658e-16

    model =np.empty(x.size)
    model=((((1.6e-9)*x)**3)/((h*c_light)**2))*np.exp(-(x*(1.6e-9))/(kB*(t)))
    model=model/np.trapz(model,x=x)*d

    return model

def blackbody(x, t,d):
    """
    Function that produces a blackbody spectrum.

    :param x: Energy in keV
    :param t: temp in kelvin
    :param d: normalization of the spectrum
    :return: returns the blackbody spectrum parameterized by some temperature at the energies of interest
    """
    h=6.6260755e-27
    c_light=29979245800.0
    kB=1.380658e-16

    model =np.empty(x.size)
    model=((((1.6e-9)*x)**3)/((h*c_light)**2))*((np.exp((x*(1.6e-9))/(kB*(t))) -1)**-1)
    model=model/np.trapz(model,x=x)*d

    return model


def comp(x, a, c,d):
    """
    returns the Comptonized (COMP) spectrum for a given set of energies

    :param x: the energies that the Comptonized function will be evaluated at (should be in keV)
    :param a: the alpha parameter
    :param c: the break energy
    :param d: the normalization of the spectrum
    :return: returns the COMP spectrum at the energies of interest (units of cts/s)
    """
    #a is alpha, b is beta, and c is break energy

    model=np.empty(x.size)
    model=(x**a)*np.exp(-x/c)
    model=model/np.trapz(model,x=x)*d

    return model

def Goodman(max_spex, max_energy):
    """
    Function that returns Goodman's scalable spherical explosion spectra to compare against a spectra acquired by a
    spherical explosion run in MCRAT. To compare this to simulation data, the simulation spectrum needs to be in units
    of erg/s.

    :param max_spex: maximum value of the data's spectrum.
    :param max_energy: The energy bin where the maximum of the data lies
    :return: returns the scaled version of Goodman's spectrum
    """
    nux=10**np.array([-3,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4,-1.2,-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1.,1.2,1.4])
    spy=10**np.array([-5.2,-4.8,-4.5,-4.1,-3.7,-3.4,-3,-2.7,-2.3,-1.95,-1.6,-1.3,-1.1,-0.8,-0.6,-0.4,-0.2,-0.1,-0.2,-0.6,-1,-2.4,-4])

    y_shift=max_spex/spy.max()
    x_shift=max_energy/nux[spy.argmax()]

    spy_shift=spy*y_shift
    nux_shift=nux*x_shift

    return [nux_shift, spy_shift]

def cfit(event_file,time_start, time_end, dnulog=0.1, hdf5=True, plotting=False, save_plot=False, save_spex=None, append=False, photon_type=None, calc_pol=False, calc_synch=False):
    """
    Function that fits either the COMP function of the Band function to time integrated or time resolved spectra from
    MCRaT and also acquires the 1 sigma error in the fitted values. The best fit spectra, accounting for the extra
    degree of freedom is chosen by conducting an F-test.

    :param event_file: the base name of the event file that holds all of the info about the mock observed photons
            from MCRaT
    :param time_start: the start time of a given time bin to collect photons into a spectrum
    :param time_end: the end of the time bin for collecting photons into a spectrum to be fit by this function
    :param dnulog: the spacing between energy bins when binning the mock observed MCRaT photons (in keV)
    :param hdf5: switch to let the function know if the MCRaT output files are hdf5 files or not
    :param plotting: switch to let the function knwo that it should plot the MCraT spectra and the fits to the spectra
    :param save_plot: switch to let the function know if it should save the plots
    :param save_spex: string to let the function save the tabulated spectrum to a file given by this string
    :param append: switch to let the function know if it should append data to the file given by save_spex
    :param photon_type: can be set to 's', 'i', or left as None in order to select thermal synchrotron photons, injected photons, or all the photons in the simulation for analysis
    :param calc_pol: switch to determine of polarization as a function of energy should be calculated and plotted
    :return: returns the best fitted parameters to the function, a (3,2) array that has each parameters' (alpha, beta
            and break energy) upper and lower 1 sigma errors, and a character denoting the function that provided the best fit ('c' for
            COMP or 'b' for Band)
    """
    from mclib import spex as sp
    import scipy.stats as ss
    import os.path

    #dnulog=.01
    numin=10**np.arange(-7,5,dnulog) # before was -2
    numax=numin*10**dnulog
    nucen=np.sqrt(numax*numin)

    spex,spee,grade, i, q, u, v, p, p_angle, perr, ns=sp(event_file,numin,numax,time_start,time_end, units='cts/s', h5=hdf5, photon_type=photon_type, calc_pol=calc_pol)

    if calc_synch:
        type='s'
        if '16TI' in event_file:
            type='o'
        s_spex, s_spee, s_grade, s_i, s_q, s_u, s_v, s_p, s_p_angle, s_perr, s_ns = sp(event_file, numin, numax,
                                                                                       time_start, time_end,
                                                                                       units='cts/s', h5=hdf5,
                                                                                       photon_type=type,
                                                                                       calc_pol=calc_pol)
        s_kk = np.where(s_grade > 0)
        s_spex = s_spex[s_kk]  # /1e50
        s_spexe = s_spee[s_kk]  # /1e50
        s_pol = s_p[s_kk]
        s_p_angle = s_p_angle[s_kk]
        s_perr = s_perr[s_kk[0], :]
        s_numin = numin[s_kk]
        s_numax = numax[s_kk]
        s_nucen = nucen[s_kk]
        s_data_pts = s_spex.size

    kk=np.where(grade>0)
    spex=spex[kk]#/1e50
    spexe=spee[kk]#/1e50
    pol=p[kk]
    p_angle=p_angle[kk]
    perr=perr[kk[0],:]
    numin=numin[kk]
    numax=numax[kk]
    nucen=nucen[kk]
    data_pts=spex.size



    #print(grade[kk].mean(),grade[kk].max(), grade[kk].min(), nucen, spex, grade[kk] )

    if save_spex:
        #to tabulate the spectra to pass to Dr. Veres and put through GBM IRF
        if append:
            file_exists = os.path.isfile(save_spex)
            if file_exists:
                h=''
            else:
                h='Time Interval of Detection Start (s)\tTime Interval of Detection End (s)\tMin Energy of Bin (keV)\tMax Energy of Bin (keV)\tNumber of Photons per Second per keV\tError in the Number of Photons per Second'

            with open(save_spex, 'ab') as f:
                np.savetxt(f, np.array([np.ones(numin.size)*time_start, np.ones(numin.size)*time_end, numin, numax,spex, spexe ]).T,\
                   header=h)
        else:
            np.savetxt(save_spex, np.array([np.ones(numin.size)*time_start, np.ones(numin.size)*time_end, numin, numax,spex, spexe ]).T,\
                   header='Time Interval of Detection Start (s)\tTime Interval of Detection End (s)\tMin Energy of Bin (keV)\tMax Energy of Bin (keV)\tNumber of Photons per Second per keV\tError in the Number of Photons per Second')

    d=np.trapz(spex,x=nucen)
    #spex=Band(nucen,1,-3,100,d)   # delete this and next line to fit real data ###
    #spexe=spex/10.
    if plotting:
        #plt.close(1)
        if not calc_pol:
            f, axarr = plt.subplots(1)
            axarr_spex=axarr
        else:
            f, axarr = plt.subplots(2, sharex=True)
            axarr_spex = axarr[0]
            axarr_pol = axarr[1]
        axarr_spex.loglog(nucen,spex,'r.')
        axarr_spex.errorbar(nucen,spex,yerr=spexe,color='r',marker='o',ls='None')
        if calc_synch:
            axarr_spex.errorbar(s_nucen, s_spex, yerr=s_spexe, color='b',marker='o',ls='None')
        if calc_pol:
            axarr_pol.semilogx(nucen, pol*100, 'k.')
            axarr_pol.errorbar(nucen, pol*100, yerr=perr[:,0]*100, color='k', marker='o',ls='None')
            ax_pol_angle = axarr_pol.twinx()
            ax_pol_angle.errorbar(nucen, p_angle, yerr=perr[:,1], color='darkmagenta', marker='.', ls='None')
            ax_pol_angle.plot(np.arange(nucen.min(), nucen.max()), np.zeros(np.size(np.arange(nucen.min(), nucen.max()))),
                  ls='--', color='darkmagenta', alpha=0.5)

            ax_pol_angle.set_ylabel(r'$\chi$ ($^\circ$)', color='darkmagenta')
            ax_pol_angle.set_ylim([-90, 90])
            ax_pol_angle.set_yticks([-90, -45, 0, 45, 90])

    nucen_all = nucen.copy()
    spex_all = spex.copy()
    spexe_all = spexe.copy()

    # restrict to 8 keV and on
    fermi_gbm_e_min=8
    fermi_gbm_e_max=40e3
    idx = np.where((nucen > fermi_gbm_e_min) & (nucen <= fermi_gbm_e_max))
    spex = spex[idx]
    nucen = nucen[idx]
    spexe = spexe[idx]

    best,matrice=curve_fit(Band,nucen,spex,sigma=spexe,p0=[.3,-5,100,d], maxfev =5000)#spexe*10
    print(best,matrice)
    zachimin=((Band(nucen,best[0],best[1],best[2],best[3])-spex)**2/spexe**2).sum()

    #try to fit with a comp spectrum
    p0_init=[.3, 100,d]

    best_comp,matrice=curve_fit(comp,nucen,spex,sigma=spexe,p0=p0_init)#spexe*10
    if (best_comp[1]<0):
        best_comp,matrice=curve_fit(comp,nucen,spex,sigma=spexe,p0=p0_init, bounds=([-np.inf, 0, 0],[np.inf,np.inf,np.inf]))

    zachimin_comp=((comp(nucen,best_comp[0],best_comp[1],best_comp[2])-spex)**2/spexe**2).sum()
    #print('Chis:', zachimin, zachimin_comp, 'Number of points: ', data_pts)
    #print('Band: ', best, 'Comp: ', best_comp)

    if plotting:
        axarr_spex.plot(nucen,Band(nucen,best[0],best[1],best[2],best[3]),'g', label='Band')
        axarr_spex.plot(nucen, comp(nucen,best_comp[0],best_comp[1],best_comp[2]), 'r', label='Comp')
        axarr_spex.legend(loc='best')
        #plt.show()

    #if the two chi squares are equal have to choose the simpler model aka comp otherwise do F test
    if (zachimin==zachimin_comp):
        #use comp
        model_use='c'
        #print('Equal chi squared Values: Using Comp Model')
    else:
        #do F test
        dof_c=data_pts-2-1
        dof_b=data_pts-3-1
        SS_c=((comp(nucen,best_comp[0],best_comp[1],best_comp[2])-spex)**2).sum()
        SS_b=((Band(nucen,best[0],best[1],best[2],best[3])-spex)**2).sum()

        alpha=0.05 #false positive acceptance rate 5%
        F=((zachimin_comp-zachimin)/(dof_c-dof_b))/(zachimin/dof_b)
        p=1-ss.f.cdf(F,(dof_c-dof_b), dof_b)
        print(p,F)
        if (p < alpha):
            model_use='b'
            #print('Using The band function')
        else:
            model_use='c'
            #print('Using the Comp function')


    if model_use=='b' :
        previous_best=np.zeros_like(best)
        while not np.array_equal(previous_best,best):
            previous_best=best.copy() #get errors on parameters
            try:
                a_err=param_err(nucen,spex, spexe,best,zachimin,'a')
            except RuntimeError:
                print('No parameter errors for a')
                a_err=np.array([np.nan,np.nan])
                #print(a_err)
            try:
                b_err=param_err(nucen,spex, spexe,best,zachimin,'b')
            except RuntimeError:
                print('No parameter errors for b')
                b_err=np.array([np.nan,np.nan])  #print(b_err)
            except ValueError:
                print('No parameter errors for b')
                b_err=np.array([np.nan,np.nan])
            try:
                c_err=param_err(nucen,spex, spexe,best,zachimin,'c')
            except RuntimeError:
                print('No parameter errors for E_0')
                c_err=np.array([np.nan,np.nan])
    else:

        previous_best=np.zeros_like(best_comp)
        while not np.array_equal(previous_best,best_comp):
            previous_best=best_comp.copy()
            #get errors on parameters
            try:
                a_err=param_err_comp(nucen,spex, spexe,best_comp,zachimin_comp,'a')
            except RuntimeError:
                print('No parameter errors for a')
                a_err=np.array([np.nan,np.nan])
            try:
                c_err=param_err_comp(nucen,spex, spexe,best_comp,zachimin_comp,'c')
            except RuntimeError:
                print('No parameter errors for E_0')
                c_err=np.array([np.nan,np.nan])

        best[0]=best_comp[0]
        best[1]=np.nan
        best[2]=best_comp[1]
        best[3]-best_comp[2]
        b_err=np.array([np.nan,np.nan])



    if plotting:
        if not calc_pol:
            f, axarr = plt.subplots(1, sharex=True)
            axarr_spex=axarr
        else:
            f, axarr = plt.subplots(2, sharex=True)
            #f.subplots_adjust(right=0.75) #went with new_ax.spines
            axarr_spex = axarr[0]
            axarr_pol = axarr[1]
        plt.rcParams.update({'font.size':14})

        axarr_spex.loglog(nucen_all,spex_all,'b.')
        axarr_spex.errorbar(nucen_all,spex_all,yerr=spexe_all,color='b',marker='o',ls='None', markersize=10, label='Total Spectrum')
        if calc_synch:
            axarr_spex.errorbar(s_nucen, s_spex, yerr=s_spexe, color='r',marker='P',ls='None', markersize=8, label='CS Spectrum')#, mfc='w'
            max_spex=s_spex[s_nucen<1].max()
            max_E=s_nucen[s_nucen<1][s_spex[s_nucen<1].argmax()]
            x=np.linspace(max_E, nucen_all[spex_all.argmax()], 100)
            axarr_spex.plot(x, x**-1*max_spex/x[0]**-1, 'k-.', zorder=3)
            axarr_spex.annotate(r'$\propto E^{-1}$', (x[-1], x[-1]**-1*max_spex/x[0]**-1 ), textcoords='offset points', xytext=(-10, -15))
        if calc_pol:
            axarr_pol.semilogx(nucen_all, pol*100, 'k.')
            axarr_pol.errorbar(nucen_all, pol*100, yerr=perr[:,0]*100, color='k', marker='o',ls='None')
            axarr_pol.set_ylabel(r'$\Pi (\%)$', fontsize=14)
            axarr_pol.set_xlabel(r'E' + ' (keV)', fontsize=14)
            if (axarr_pol.get_ylim()[1] > 100):
                axarr_pol.set_ylim([0, 100])
                axarr_pol.set_yticks([ 0, 25, 50, 75, 100])
            if (axarr_pol.get_ylim()[0] < 0):
                axarr_pol.set_ylim([0, axarr_pol.get_ylim()[1]])



            ax_pol_angle = axarr_pol.twinx()
            ax_pol_angle.errorbar(nucen_all, p_angle, yerr=perr[:,1], color='darkmagenta', marker='o', ls='None')
            ax_pol_angle.plot(np.arange(nucen_all.min(), nucen_all.max()), np.zeros(np.size(np.arange(nucen_all.min(), nucen_all.max()))),
                  ls='--', color='darkmagenta', alpha=0.5)

            ax_pol_angle.set_ylabel(r'$\chi$ ($^\circ$)', color='darkmagenta', fontsize=14)
            ax_pol_angle.set_ylim([-90, 90])
            ax_pol_angle.set_yticks([-90, -45, 0, 45, 90])

            def make_patch_spines_invisible(ax):
                ax.set_frame_on(True)
                ax.patch.set_visible(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)


            #axvspan for energy limits of different instruments

            #new_ax = axarr_pol.twinx()
            #new_ax.spines["right"].set_position(("axes", 1.2))
            #make_patch_spines_invisible(new_ax)
            #new_ax.spines["right"].set_visible(True)

            #modify this to show either number of scatterings or number of photons
            new_ax = axarr_spex.twinx()
            #new_ax.plot(nucen_all, grade[kk], 'g.')
            #new_ax.set_ylabel('Number of MC Photons', color='g', fontsize=14)
            new_ax.plot(nucen_all, ns[kk], 'g.')
            new_ax.set_ylabel('Avg number of scatterings', color='g', fontsize=14)
            new_ax.set_yscale('log')



        if model_use=='b':
            band_in_energy_range=Band(nucen, best[0], best[1], best[2], best[3])
            axarr_spex.plot(nucen,band_in_energy_range, color='k', label='Fitted Band', ls='solid', lw=3, zorder=3)
            full_band = Band(nucen_all, best[0], best[1], best[2], best[3]) # this is for the extrapolated portion below energy range keV, also normalize it so its continuous with other plotted band function >energy range
            full_band = full_band*band_in_energy_range[-1]/full_band[-1]
            axarr_spex.plot(nucen_all[nucen_all<=fermi_gbm_e_min],full_band[nucen_all<=fermi_gbm_e_min],'k--', lw=3, label='Extrapolated Band', zorder=3)

            axarr_spex.annotate(r'$\alpha$'+'='+np.str(best[0]).split('.')[0] + '.'+ np.str(best[0]).split('.')[1][0]+
                '\n'+r'$\beta$'+'='+np.str(best[1]).split('.')[0] + '.'+ np.str(best[1]).split('.')[1][0]+'\n'+r'E$_{\mathrm{o}}$'+'=' +
                    np.str(best[2]).split('.')[0] + '.'+ np.str(best[2]).split('.')[1][0] +' keV', xy=(0, 0), xycoords='axes fraction', fontsize=18, xytext=(10, 10),
                      textcoords='offset points', ha='left', va='bottom')

        if model_use=='c':
            axarr_spex.plot(nucen, comp(nucen,best_comp[0],best_comp[1],best_comp[2]), color='k',label='Fitted COMP',ls='solid', lw=3, zorder=3)
            full_comp = comp(nucen_all, best_comp[0], best_comp[1], best_comp[2])  # this is for the extrapolated portion below 1 keV
            axarr_spex.plot(nucen_all[nucen_all <= fermi_gbm_e_min], full_comp[nucen_all <= fermi_gbm_e_min], 'k--', lw=3,
                            label='Extrapolated COMP', zorder=3)

            axarr_spex.annotate(r'$\alpha$'+'='+np.str(best[0]).split('.')[0] + '.'+ np.str(best[0]).split('.')[1][0]+
                '\n'+r'E$_{\mathrm{o}}$'+'=' +np.str(best[2]).split('.')[0] + '.'+ np.str(best[2]).split('.')[1][0]+' keV', xy=(0, 0), xycoords='axes fraction', fontsize=18, xytext=(10, 10),
                      textcoords='offset points', ha='left', va='bottom')
        axarr_spex.set_ylabel('N('+r'E'+') (photons/s/keV)', fontsize=14)
        if not calc_pol:
            axarr_spex.set_xlabel(r'E'+' (keV)', fontsize=14)
            axarr_spex.axvspan(8, 40e3, ymin=0, ymax=1, alpha=0.5, facecolor='g')
            energy_range = [1.5e-3, 7.7e-3]
            axarr_spex.axvspan(energy_range[0], energy_range[1], ymin=0, ymax=1, alpha=0.5, facecolor='r')
            #f.legend(loc='upper center', fontsize=12, ncol=2, bbox_to_anchor=(0.5, 1.05))
            #axarr_spex.annotate('(b) 40sp_down\n'+'\t'+r'$\theta_\mathrm{v}=%s^\circ$'%(event_file.split('_')[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=14, xytext=(-120, -30),textcoords='offset points')
            axarr_spex.annotate('(c) 16TI\n'+'     '+r'$\theta_\mathrm{v}=%s^\circ$'%(event_file.split('_')[-1]), xy=(1, 1), xycoords='axes fraction', fontsize=14, xytext=(-90, -30),textcoords='offset points')


        axarr_spex.tick_params(labelsize=14)
        if calc_pol:
            axarr_pol.tick_params(labelsize=14)

        if save_plot:
            ang_str = event_file.split('_')[-1]
            base_name=event_file.split(ang_str)[0]
            if not calc_pol:
                savefilename='EVENT_FILE_ANALYSIS_PLOTS/'+event_file.replace('.', '_')+'_t_s_'+np.str(time_start).replace('.','_')+'_t_e_'+np.str(time_end).replace('.','_')+'.pdf'
            else:
                savefilename='EVENT_FILE_ANALYSIS_PLOTS/' + event_file.replace('.', '_') + '_t_s_' + np.str(
                    time_start).replace('.', '_') + '_t_e_' + np.str(time_end).replace('.', '_') + '_w_pol_w_scatt.pdf'
            f.savefig(savefilename, bbox_inches='tight')
        plt.show()

    return best, np.array([a_err,b_err,c_err]), model_use

def param_err_comp(x,data,err, best_par,chi_sq,par):
    """
    Function to calculate the errors in the comptonized fit to the spectrum. Returns 1 sigma error bars.


    :param x: energies of the spectrum in keV
    :param data: The spectrum data points that are being fitted
    :param err: the errors in each data point
    :param best_par: the best fit parameters of the COMP function to the MCRaT spectrum
    :param chi_sq: The value of chi squared of the best fit (value of best_par)
    :param par: The COMP spectrum parameter of interest that we would like to find the errors of ('a', 'b', 'c' which
            corresponds to the definitions of the COMP function)
    :return: returns a (1x2) array of the negative error bar and the positive error bar
    """

    d_par_start=0.1

    d_par_p,new_best=param_err_loop_comp(x,data,err, best_par.copy(),chi_sq,par,d_par_start)

    if not np.array_equal(best_par,new_best):
        #print('Old:',chi_sq, best_par)
        best_par=new_best #if function finds new chi min, set as new best_par
        zachimin=((comp(x,best_par[0],best_par[1],best_par[2])-data)**2/err**2).sum()
        #print('New:',zachimin,x.size)

    print('Getting Negative Error Bar')

    if (~np.isnan(d_par_p)):
        d_par_m, new_best=param_err_loop_comp(x,data,err, best_par.copy(),chi_sq,par,-d_par_start)
    else:
        d_par_p=np.nan
        d_par_m=np.nan

    if not np.array_equal(best_par,new_best):
        #print('Old:',chi_sq, best_par)
        best_par=new_best
        zachimin=((comp(x,best_par[0],best_par[1],best_par[2])-data)**2/err**2).sum()
        #print('New:',zachimin,x.size)


    return np.array([d_par_m,d_par_p])

def param_err_loop_comp(x,data,err, best_par,chi_sq,par,d_par):
    """
    Function to calculate the deviation in a COMP spectrum parameter such that there is a change of ~1 in the chi squared
    value of the fit.

    :param x: energies of the spectrum in keV
    :param data: The spectrum data points that are being fitted
    :param err: the errors in each data point
    :param best_par: the best fit parameters of the COMP function to the MCRaT spectrum
    :param chi_sq: The value of chi squared of the best fit (value of best_par)
    :param par: The COMP spectrum parameter of interest that we would like to find the errors of
    :param d_par: The initial guess in the change in parameter that would lead to a change in chi squared of ~1
    :return: returns the amount the parameter of interest can change before the fitted chi squared changes by ~1
    """
    #set limits for delta chi squared to get
    #NUMERICAL RECIPES IN C: THE ART OF SCIENTIFIC COMPUTING, delta_chi=1 gives 1 sigma (68.3%) for 1 degree of freedom
    count=0
    d_chi_sq_min=0.9
    d_chi_sq_max=1.1

    if par=='a':
        fudge_factor=1e-12
    else:
        fudge_factor=1e-5

    original_best=best_par.copy()
    print('@ start:',original_best)
    original_d_par=d_par
    refit_chi=0

    while (((refit_chi-chi_sq)> d_chi_sq_max) or ((refit_chi-chi_sq)<d_chi_sq_min) and count<1000):
        #based on refit_chi modify d_par
        if ((refit_chi-chi_sq)> d_chi_sq_max):
            d_par=d_par/10
        if ((refit_chi-chi_sq)<d_chi_sq_min):
            d_par=d_par*2

        if par=='a':
            best_par[0]=d_par+original_best[0]
            #fudge_factor=10**-(np.str_(best_par[0])[::-1].find('.')) #factor to make by pass curve_fit criteria for upper and lower bounds to not be equal
            bound_array_l=[best_par[0], -np.inf,-np.inf]
            bound_array_u=[best_par[0]+fudge_factor, np.inf,np.inf]
            guess=[best_par[0], original_best[1], original_best[2]]
        elif par=='c':
            best_par[1]=d_par+original_best[1]
            fudge_factor=10**-(np.str_(best_par[1])[::-1].find('.')-1)
            #if theres an e, number is too big and need larger fudge factor
            if (np.str_(best_par[1]).find('e') != -1):
                fudge_factor=10**(np.float(np.str_(best_par[1])[np.str_(best_par[1]).find('e')+2:])-10)
            bound_array_l=[ -np.inf,best_par[1],-np.inf]
            bound_array_u=[np.inf,best_par[1]+(2*fudge_factor),np.inf]
            guess=[ original_best[0], best_par[1], original_best[2]]

        #print(bound_array_l,bound_array_u)

        #find new parameter fit
        #print(x,data,err,guess,bound_array_l,bound_array_u)
        best,covar=curve_fit(comp,x,data,sigma=err, p0=guess, bounds=(bound_array_l,bound_array_u))

        #use new best to calculate new refit_chi
        refit_chi=(((comp(x,best[0],best[1],best[2])-data)**2)/err**2).sum()

        #print(refit_chi-chi_sq,best, d_par)

        if (refit_chi-chi_sq)<=0:
            print('The refitted curve has the same or better chi squared value as the original best parameters',refit_chi-chi_sq )
            if np.round(refit_chi-chi_sq)<=-1:
                #if chi squared is better by at least one
                #set as new original best
                print('Restarting parameter error finding')
                original_best=best.copy()
                d_par=original_d_par
        count+=1
    print('@ end:',original_best)
    if count>1000:
        d_par=np.nan

    return d_par, original_best

def param_err(x,data,err, best_par,chi_sq,par):
    """
    Function to calculate the 1 sigma errors in the comptonized fit to the spectrum.

    :param x: energies of the spectrum in keV
    :param data: The spectrum data points that are being fitted
    :param err: the errors in each data point
    :param best_par: the best fit parameters of the Band function to the MCRaT spectrum
    :param chi_sq: The value of chi squared of the best fit (value of best_par)
    :param par: The Band spectrum parameter of interest that we would like to find the errors of
    :return: returns a (1x2) array of the negative error bar and the positive error bar
    """

    d_par_start=0.1

    d_par_p,new_best=param_err_loop(x,data,err, best_par.copy(),chi_sq,par,d_par_start)

    if not np.array_equal(best_par,new_best):
        #print('Old:',chi_sq, best_par)
        best_par=new_best #if function finds new chi min, set as new best_par
        zachimin=((Band(x,best_par[0],best_par[1],best_par[2],best_par[3])-data)**2/err**2).sum()
        #print('New:',zachimin,x.size)

    print('Getting Negative Error Bar')

    d_par_m, new_best=param_err_loop(x,data,err, best_par.copy(),chi_sq,par,-d_par_start)

    if not np.array_equal(best_par,new_best):
        #print('Old:',chi_sq, best_par)
        best_par=new_best
        zachimin=((Band(x,best_par[0],best_par[1],best_par[2],best_par[3])-data)**2/err**2).sum()
        #print('New:',zachimin,x.size)


    return np.array([d_par_m,d_par_p])

def param_err_loop(x,data,err, best_par,chi_sq,par,d_par):
    """
    Function to calculate the deviation in a Band spectrum parameter such that there is a change of ~1 in the chi squared
    value of the fit.

    :param x: energies of the spectrum in keV
    :param data: The spectrum data points that are being fitted
    :param err: the errors in each data point
    :param best_par: the best fit parameters of the Band function to the MCRaT spectrum
    :param chi_sq: The value of chi squared of the best fit (value of best_par)
    :param par: The Band spectrum parameter of interest that we would like to find the errors of
    :param d_par: The initial guess in the change in parameter that would lead to a change in chi squared of ~1
    :return: returns the amount the parameter of interest can change before the fitted chi squared changes by ~1
    """
    #set limits for delta chi squared to get
    count=0
    d_chi_sq_min=0.9
    d_chi_sq_max=1.1

    if par=='a':
        fudge_factor=1e-12
    elif par=='b':
        fudge_factor=1e-8
    else:
        fudge_factor=1e-5

    original_best=best_par.copy()
    print('@ start:',original_best)
    original_d_par=d_par
    refit_chi=0

    while (((refit_chi-chi_sq)> d_chi_sq_max) or ((refit_chi-chi_sq)<d_chi_sq_min) and count<1000):
        #based on refit_chi modify d_par
        if ((refit_chi-chi_sq)> d_chi_sq_max):
            d_par=d_par/10
        if ((refit_chi-chi_sq)<d_chi_sq_min):
            d_par=d_par*2.3

        if par=='a':
            best_par[0]=d_par+original_best[0]
            #fudge_factor=10**-(np.str_(best_par[0])[::-1].find('.')) #factor to make by pass curve_fit criteria for upper and lower bounds to not be equal
            bound_array_l=[best_par[0], -np.inf,-np.inf,-np.inf]
            bound_array_u=[best_par[0]+fudge_factor, np.inf,np.inf,np.inf]
            guess=[best_par[0], original_best[1], original_best[2], original_best[3]]
        elif par=='b':
            best_par[1]=d_par+original_best[1]
            fudge_factor=10**-(np.str_(best_par[1])[::-1].find('.')-1)
            #if theres an e, number is too big and need larger fudge factor
            if (np.str_(best_par[1]).find('e') != -1):
                fudge_factor=10**(np.float(np.str_(best_par[1])[np.str_(best_par[1]).find('e')+2:])-10)

            #print(np.str_(best_par[1])[::-1].find('.'),np.str_(best_par[1])[::-1].find('.')-1)
            bound_array_l=[ -np.inf,best_par[1],-np.inf,-np.inf]
            bound_array_u=[np.inf,best_par[1]+fudge_factor,np.inf,np.inf]
            guess=[ original_best[0], best_par[1], original_best[2], original_best[3]]
        elif par=='c':
            best_par[2]=d_par+original_best[2]
            #fudge_factor=10**-(np.str_(best_par[2])[::-1].find('.'))
            bound_array_l=[ -np.inf,-np.inf,best_par[2],-np.inf]
            bound_array_u=[np.inf,np.inf,best_par[2]+fudge_factor,np.inf]
            guess=[ original_best[0], original_best[1], best_par[2],original_best[3]]

        #print(bound_array_l,bound_array_u)

        #find new parameter fit
        #print(x,data,err,guess,bound_array_l,bound_array_u)
        best,covar=curve_fit(Band,x,data,sigma=err, p0=guess, bounds=(bound_array_l,bound_array_u))

        #use new best to calculate new refit_chi
        refit_chi=(((Band(x,best[0],best[1],best[2],best[3])-data)**2)/err**2).sum()

        print(refit_chi-chi_sq,best, d_par)

        if (refit_chi-chi_sq)<=0:
            print('The refitted curve has the same or better chi squared value as the original best parameters',refit_chi-chi_sq )
            if np.round(refit_chi-chi_sq)<=-1:
                #if chi squared is better by at least one
                #set as new original best
                print('Restarting parameter error finding')
                original_best=best.copy()
                d_par=original_d_par
        else:
            d_chi_sq_min=d_chi_sq_min-0.1
            d_chi_sq_max=d_chi_sq_max+0.1
            print(d_chi_sq_min, d_chi_sq_max)
        count+=1

    if count>1000:
        d_par=np.nan

    print('@ end:',original_best)
    return d_par, original_best

def band_hist_data(pickle_file, event_files, time_start, time_end, dt, plotting=False, save_plot=False, choose_best=False):
    """
    Function that collects a set of parameters from time resolved spectral fits of MCRaT data. This allows them to be
    histogrammed to look at the distribution of alpha, beta and e_pk for example.

    :param pickle_file: the name of the pickle file that will contain all of the spectral parameter values so it is only
            necessary to run this lengthy calculation once.
    :param event_files: The event files that will be analyzed to fit time resolved spectra and acquire the spectral
            parameters, a 1D array of strings
    :param time_start: The start time of the light curve that will be analyzed, a 1D array that is the same size as
            event_files
    :param time_end: The end time of the light curve that will be analyzed, a 1D array that is the same size as
            event_files
    :param dt: The width of time bins that photons will be collected within to produce the time resolved spectra, this
            will be across all of the event files that will be analyzed
    :param plotting: Switch to determine if the call to produce the light curves for each event file should plot
            all the light curves that it produces so the user can inspect the spectral fits in each time interval
    :param save_plot: Switch to determine if the call to produce the light curves for each event file should
            save all the light curves that it produces (saves time in that the user doesn't have to recall the
            lcur_param function to make plots of the light curves)
    :param choose_best: Switch to ignore spectral parameters that haven't been well constrained (they may not have
            error bars)
    :return: Returns 2D arrays of alpha, beta, and break energies that contain all the time resolved parameter fits for
            each event file that was passed to the function.
    """
    #takes an array of event files and times and goes through them to get the alphas, betas, and energies, and saves them in a
    #pickle file for later plotting

    import pickle
    time=np.arange(time_start.min(),time_end.max()+dt.min(),dt.min())#get the most number of time bins that have to be accounted for
    alphas=np.zeros([event_files.size, time.size])
    betas=np.zeros_like(alphas)
    e_0s=np.zeros_like(alphas)
    model=np.zeros_like(alphas, dtype='S')

    #set with impossible values
    alphas[:,:]=np.inf
    betas[:,:]=np.inf
    e_0s[:,:]=np.inf

    j=0
    for i in event_files:
        lcur, lcur_e, a, b, e, err, time, m, P, I, Q, U, V, Perr, P_angle, num_ph, num_scatt=\
            lcur_param_var_t(i, time_start[j], time_end[j], dt[j], plotting=plotting, save_plot=save_plot, choose_best=choose_best)
        alphas[j, :time.size]=a
        betas[j, :time.size]=b
        e_0s[j, :time.size]=e
        model[j, :time.size]=m
        #alphas[j,:time.size][lcur<1e50]=np.nan
        #e_0s[j,:time.size][lcur<1e50]=np.nan
        #betas[j,:time.size][lcur<1e50]=np.nan

        j+=1

    f=open(pickle_file + '.pickle', 'wb')
    pickle.dump([alphas, betas, e_0s, model],f)
    f.close()

    return alphas, betas, e_0s, model

def band_hist_plot(pickle_file, plotting=True, save_plot=False, scale=1):
    """
    Function to plot the histogrammed Band/COMP spectral parameters.

    :param pickle_file: The produced pickle file name from the band_hist_data function
    :param plotting: Switch to show the produced plots
    :param save_plot: Switch to save the plotted histograms
    :param scale: Scaling to scale the histograms if necessary
    :return: Returns 2D arrays of alpha, beta, and break energies that contain all the time resolved parameter fits for
            each event file that was passed to the band_hist_data function.
    """
    import pickle
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    import matplotlib.lines as mlines
    import matplotlib
    #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
    #plt.style.use('seaborn-paper')
    #plt.rcParams.update({'font.size':24})
    plt.rcParams.update({'font.size':14})

    with open(pickle_file + '.pickle', 'rb') as pickle_f:
        u=pickle._Unpickler(pickle_f)
        u.encoding='latin1'
        alphas, betas, e_0s, model=u.load()

        #u.encoding='latin1'
        #Temp_photon, Temp_flash, Avg_R, ph_num=u.load()
        #file=open('EVENT_FILE_ANALYSIS_PLOTS/'+pickle_file + '_3.0.pickle')

    model=model.astype('U')

    theta_1=mlines.Line2D([],[],color='darkmagenta', ls='dashed', lw=3, label=r'16TI')
    theta_2=mlines.Line2D([],[],color='darkmagenta', ls='solid', lw=3, label=r'40sp_down')
    theta_3=mlines.Line2D([],[],color='darkmagenta', ls='dotted', lw=1, label=r'$\theta_v= 3^ \circ$')

    #alphas_bin_size=0.5
    #betas_bin_size=0.5
    #energies_bin_size=5

    #make the peak energies match the FERMI Brightest Burst paper
    energies=(e_0s*(2+alphas))
    energies=energies[energies>0]

    all_alphas=alphas[np.logical_and(~np.isnan(alphas),~np.isinf(alphas)) ]
    all_betas=betas[np.logical_and(~np.isnan(betas),~np.isinf(betas)) ]
    all_energies=energies[np.logical_and(~np.isnan(energies),~np.isinf(energies)) ]
    d=get_FERMI_best_data()

    fig_all_a=plt.figure()
    a_ax=fig_all_a.add_subplot(111)
    num,bin, stuff=plt.hist(all_alphas, bins=np.arange(np.floor(all_alphas.min()), np.ceil(all_alphas.max()+0.1), 0.1), color='r', alpha=.75, edgecolor='None', zorder=5)#
    #plt.hist(d[:,0], bins=np.arange(np.floor(d[:,0].min()), np.ceil(d[:,0].max()+0.1), 0.1), color="#3F5D7D",alpha=0.2)
    hist_all,bin_edges_all=np.histogram(d[:,0], bins=np.arange(np.floor(d[:,0].min()), np.ceil(d[:,0].max()+0.1), 0.1))
    bin_centers=(bin_edges_all[:-1] +bin_edges_all[1:])/2.0
    plt.bar(bin_centers, hist_all/scale, width=( bin_edges_all[1:]-bin_edges_all[:-1]), align='center', color='darkorange',alpha=1, zorder=1)
    """
    if sim:
        #plot alphas for each angle
        for i in range(3):
            if (i==0):
                line='dashed'
                j=1
            elif (i==1):
                line='solid'
                j=1
            else:
                line='dotted'
                j=1

            plt.hist(alphas[i,:][~np.isnan(alphas[i,:][~np.isinf(alphas[i,:])])], bins=np.arange(np.floor(all_alphas.min()), np.ceil(all_alphas.max()+0.1), 0.1), histtype='step', color='darkmagenta', linestyle=line, lw=j)
    else:
        #plot angles for all the simulations
        for i in range(3):
            if (i==0):
                line='dashed'
                j=1
            elif (i==1):
                line='solid'
                j=1
            else:
                line='dotted'
                j=1
            flat_arr=alphas[i::3,:].flatten()
            plt.hist(flat_arr[np.where(np.logical_and(~np.isnan(flat_arr),~np.isinf(flat_arr)))], bins=np.arange(np.floor(all_alphas.min()), np.ceil(all_alphas.max()+0.1), 0.1), histtype='step', color='darkmagenta', linestyle=line, lw=j)


    """
    alpha_16ti=alphas[:15,:][np.logical_and(~np.isnan(alphas[:15,:]),~np.isinf(alphas[:15,:])) ]
    alpha_40sp_down=alphas[15:,:][np.logical_and(~np.isnan(alphas[15:,:]),~np.isinf(alphas[15:,:])) ]

    a_ax.hist(alpha_16ti, bins=np.arange(np.floor(all_alphas.min()), np.ceil(all_alphas.max() + 0.1), 0.1),
             histtype='step', color='darkmagenta', linestyle='dashed', lw=3, zorder=6)

    a_ax.hist(alpha_40sp_down, bins=np.arange(np.floor(all_alphas.min()), np.ceil(all_alphas.max() + 0.1), 0.1),
             histtype='step', color='darkmagenta', linestyle='solid', lw=3, zorder=6)

    #plt.ylim([0,(hist_all.max()/scale)+45]) #+45 for 35OB otherwise +10
    plt.ylabel('N('+r'$\alpha$'+')')
    plt.xlabel(r'$\alpha$')
    plt.xlim([-2,6])
    plt.legend(loc='upper right', handles=[theta_1, theta_2],fontsize='12')
    """
    #plot alphas divided by comp vs Band fit spectra
    inset_axis=inset_axes(a_ax, width="40%",height=1.5, loc=1)
    #inset_axis=inset_axes(a_ax, width=3,height=2.0, loc=3, bbox_to_anchor=(0.55,0.5), bbox_transform=a_ax.figure.transFigure)
    inset_axis.hist(all_alphas[model[model!=""]=='c'],bins=np.arange(np.floor(all_alphas.min()), np.ceil(all_alphas.max()+0.1), 0.1),color='r', histtype='step', lw=2)
    inset_axis.hist(all_alphas[model[model!=""]=='b'],bins=np.arange(np.floor(all_alphas.min()), np.ceil(all_alphas.max()+0.1), 0.1),color='b', histtype='step', lw=2)

    hist,bin_edges=np.histogram(d[np.isnan(d[:,1]),0], bins=np.arange(np.floor(d[:,0].min()), np.ceil(d[:,0].max()+0.1), 0.1))
    bin_centers=(bin_edges[:-1] +bin_edges[1:])/2.0
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="r",ls='steps', lw=2)
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="k",ls='steps--', lw=2)

    hist,bin_edges=np.histogram(d[~np.isnan(d[:,1]),0], bins=np.arange(np.floor(d[:,0].min()), np.ceil(d[:,0].max()+0.1), 0.1))
    bin_centers=(bin_edges[:-1] +bin_edges[1:])/2.0
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="b",ls='steps', lw=2)
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="k",ls='steps--', lw=2)

    #plt.hist(d[np.isnan(d[:,1]),0], bins=np.arange(np.floor(d[:,0].min()), np.ceil(d[:,0].max()+0.1), 0.1), color='r', histtype='step', ls='dashed')
    #plt.hist(d[~np.isnan(d[:,1]),0], bins=np.arange(np.floor(d[:,0].min()), np.ceil(d[:,0].max()+0.1), 0.1), color='b', histtype='step', ls='dashed')
    
    index=np.where(num>0)
    index_fermi=np.where(hist_all>0)
    inset_axis.set_xlim(bin_edges_all[index_fermi[0][0]], bin[index[0][-1]+1])
    inset_axis.set_ylim(0, (hist_all.max()/scale)+30) #+30 for 35OB
    plt.ylabel('N('+r'$\alpha$'+')')
    plt.xlabel(r'$\alpha$')
    """

    #for spine in ['left', 'right', 'top', 'bottom']:
    #	plt.spines[spine].set_color('k')
    #plt.grid(b=False)
    #plt.set_axis_bgcolor('white')
    #plt.get_xaxis().tick_bottom()
    #plt.get_yaxis().tick_left()

    fig_all_b=plt.figure()
    b_ax=fig_all_b.add_subplot(111)
    n, b, stuff=plt.hist(all_betas, bins=np.arange(np.floor(all_betas.min()), np.ceil(all_betas.max()+0.1), 0.1), color='b', alpha=.75, edgecolor='None', zorder=1)#
    num, bin, stuff=plt.hist(d[~np.isnan(d[:,1]),1], bins=np.arange(np.floor(d[~np.isnan(d[:,1]),1].min()), np.ceil(d[~np.isnan(d[:,1]),1].max()+0.1), 0.1), color='darkorange',alpha=1, zorder=5, linestyle='solid', lw=4) #"#3F5D7D"

    """
    if sim:
        for i in range(3):
            if (i==0):
                line='dashed'
                j=1
            elif (i==1):
                line='solid'
                j=1
            else:
                line='dotted'
                j=1
            plt.hist(betas[i,:][~np.isnan(betas[i,:][~np.isinf(betas[i,:])])], bins=np.arange(np.floor(all_betas.min()), np.ceil(all_betas.max()+0.1), 0.1), histtype='step', color='darkmagenta', linestyle=line, lw=j)
    else:
        for i in range(3):
            if (i==0):
                line='dashed'
                j=1
            elif (i==1):
                line='solid'
                j=1
            else:
                line='dotted'
                j=1
            flat_arr=betas[i::3,:].flatten()
            plt.hist(flat_arr[np.where(np.logical_and(~np.isnan(flat_arr),~np.isinf(flat_arr)))], bins=np.arange(np.floor(all_betas.min()), np.ceil(all_betas.max()+0.1), 0.1), histtype='step', color='darkmagenta', linestyle=line, lw=j)

    """
    beta_16ti=betas[:15,:][np.logical_and(~np.isnan(betas[:15,:]),~np.isinf(betas[:15,:])) ]
    beta_40sp_down=betas[15:,:][np.logical_and(~np.isnan(betas[15:,:]),~np.isinf(betas[15:,:])) ]

    b_ax.hist(beta_16ti, bins=np.arange(np.floor(all_betas.min()), np.ceil(all_betas.max()+0.1), 0.1),
             histtype='step', color='darkmagenta', linestyle='dashed', lw=3, zorder=6)

    b_ax.hist(beta_40sp_down, bins=np.arange(np.floor(all_betas.min()), np.ceil(all_betas.max()+0.1), 0.1),
             histtype='step', color='darkmagenta', linestyle='solid', lw=3, zorder=6)

    #plt.legend(loc='best', handles=[theta_1, theta_2, theta_3],fontsize='12')
    plt.ylabel('N('+r'$\beta$'+')')
    plt.xlabel(r'$\beta$')
    print(num.max())
    plt.ylim([0,n.max()+1])
    plt.xlim([-6, -1])

    fig_all_e=plt.figure()
    e_ax=fig_all_e.add_subplot(111)
    bin_e=10**(np.arange(np.log10(np.floor(all_energies.min())), np.log10(np.ceil(d[:,2].max()+10)), 0.1)) #(np.arange((np.floor(all_energies.min())), (np.ceil(d[:,2].max()+10)), 10))

    num,bin, stuff=plt.hist(all_energies, bins=bin_e, color='g', alpha=0.75, edgecolor='None', zorder=5)#
    #plt.hist(d[:,2], bins=np.arange((np.floor(all_energies.min())), (np.ceil(d[:,2].max()+50)), 50 ), color="#3F5D7D",alpha=0.2)
    hist_all,bin_edges_all=np.histogram(d[:,2], bins=bin_e)
    bin_centers=(bin_edges_all[:-1] +bin_edges_all[1:])/2.0
    plt.bar(bin_centers, hist_all/scale, width=( bin_edges_all[1:]-bin_edges_all[:-1]), align='center', color='darkorange',alpha=1, zorder=1)
    plt.xlim([bin.min(), 5000])
    """
    if sim:
        for i in range(3):
            if (i==0):
                line='dashed'
                j=1
            elif (i==1):
                line='solid'
                j=1
            else:
                line='dotted'
                j=1
            plt.hist(energies[i,:][~np.isnan(energies[i,:][~np.isinf(energies[i,:])])], bins=bin_e, histtype='step', color='darkmagenta', linestyle=line, lw=j)
    else:
        for i in range(3):
            if (i==0):
                line='dashed'
                j=1
            elif (i==1):
                line='solid'
                j=1
            else:
                line='dotted'
                j=1
            flat_arr=energies[i::3,:].flatten()
            plt.hist(flat_arr[np.where(np.logical_and(~np.isnan(flat_arr),~np.isinf(flat_arr)))], bins=bin_e, histtype='step', color='darkmagenta', linestyle=line, lw=j)

    """
    energies=(e_0s*(2+alphas))
    energies_16ti=energies[:15,:][np.logical_and(~np.isnan(energies[:15,:]),~np.isinf(energies[:15,:])) ]
    energies_40sp_down=energies[15:,:][np.logical_and(~np.isnan(energies[15:,:]),~np.isinf(energies[15:,:])) ]
    energies_16ti=energies_16ti[energies_16ti>0]
    energies_40sp_down=energies_40sp_down[energies_40sp_down>0]

    plt.hist(energies_16ti, bins=bin_e, histtype='step', color='darkmagenta', linestyle='dashed', lw=3, zorder=6)
    plt.hist(energies_40sp_down, bins=bin_e, histtype='step', color='darkmagenta', linestyle='solid', lw=3, zorder=6)

    plt.ylabel('N('+r'E$_{\mathrm{pk}}$'+')')
    plt.xlabel(r'E$_{\mathrm{pk}}$ (keV)')
    #plt.ylim([0,num.max()+172])
    plt.gca().set_xscale("log")
    #plt.legend(loc='best', handles=[theta_1, theta_2],fontsize='12')

    """
    inset_axis=inset_axes(e_ax, width="50%",height=2.0, loc=1)
    num,bin,stuff=inset_axis.hist(all_energies[model[model!=""]=='c'],bins=bin_e,color='r',histtype='step', lw=2)
    inset_axis.hist(all_energies[model[model!=""]=='b'],bins=bin_e,color='b',histtype='step', lw=2)

    hist,bin_edges=np.histogram(d[np.isnan(d[:,1]),2], bins=bin_e)
    bin_centers=(bin_edges[:-1] +bin_edges[1:])/2.0
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="r",ls='steps', lw=2)
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="k",ls='steps--', lw=2)

    hist,bin_edges=np.histogram(d[~np.isnan(d[:,1]),2], bins=bin_e)
    bin_centers=(bin_edges[:-1] +bin_edges[1:])/2.0
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="b",ls='steps', lw=2)
    inset_axis.plot(bin_centers-(bin_edges_all[1:]-bin_edges_all[:-1])/2, hist/scale, color="k",ls='steps--', lw=2)

    inset_axis.set_xscale("log")

    index=np.where(num>0)
    inset_axis.set_xlim(bin[index[0][0]], 2000)
    inset_axis.set_ylim(0, num.max()+1)
    #inset_axis.gca().set_xscale("log")

    """
    #plt.xlim([np.floor(all_energies.min())*0.5, (np.ceil(d[:,2].max()+10))*2])
    plt.ylabel('N('+r'E$_{\mathrm{pk}}$'+')')
    plt.xlabel(r'E$_{\mathrm{pk}}$ (keV)')
    #e_ax.legend(loc='upper left', handles=[theta_1, theta_2, theta_3],fontsize='12')

    if plotting:
        plt.show()

    if save_plot:
        #f_a.savefig('EVENT_FILES/'+pickle_file+'_alphas.eps', bbox_inches='tight')
        #f_b.savefig('EVENT_FILES/'+pickle_file+'_betas.eps', bbox_inches='tight')
        #f_e.savefig('EVENT_FILES/'+pickle_file+'_energies.eps', bbox_inches='tight')
        fig_all_a.savefig(pickle_file.replace('.','_')+'_all_alphas.pdf', bbox_inches='tight',dpi=100)
        fig_all_b.savefig(pickle_file.replace('.','_')+'_all_betas.pdf', bbox_inches='tight',dpi=100)
        fig_all_e.savefig(pickle_file.replace('.','_')+'_all_energies.pdf', bbox_inches='tight',dpi=100)

    return all_alphas, all_betas, all_energies, d


def lcur_param(event_file,time_start,time_end, dt=1, phi_angle=0, delta_theta=1, unit='erg/s', plotting=True, save_plot=False, choose_best=False, lc_only=False, pol_only=False, dim=2, hdf5=True, save_spex=None, photon_type=None, energy_range=None ):
    """
    This function takes the MCRaT 'detected' photons from the event file and time bins them to produce light curves. The
    photons in a given time bin are then fitted with either the COMP or Band functions.

    :param event_file: the base name of the event file that holds all of the info about the mock observed photons
            from MCRaT
    :param time_start: the start time to begin binning the photons
    :param time_end: the end time to stop binning the photons
    :param dt: the light curve time bin width
    :param phi_angle: unused parameter, meant for future 3D development
    :param delta_theta: delta theta of the observer viewing angle for accepting photons, in degrees. Should be the same
            value as what is passed into event_h5's dtheta_deg
    :param unit: A string of the units that will be used for the light curve/spectrum 'erg/s' or 'cts/s'
    :param plotting: Switch to allow the function to show the plots of the light curve/spectral parameters
    :param save_plot: Switch to allow the function to save the produced plot
    :param choose_best: Switch that permits the function to overwrite spectral parameters that do not have error bars
            to be ignored if set to True. If this is set to False all spectral parameters will be considered/plotted
    :param lc_only: switch that tells the function to only plot the light curve and peak energy of the time resolved
        spectra and none of the other parametres
    :param pol_only: Switch to allow the function to only plot the light curve and the polarization when set to True
    :param dim: The number of dimensions of the MCRaT simulation that is being analyzed. There is no support for 3D at
            this time so this switch should be set to 2, which is it's default value
    :param hdf5: Switch to specify if the MCRaT output files were HDF5 format or not, default is True. Should be set to
            False when an old version of MCRaT was used which outputtted the data in text files
    :param save_spex: Switch that allows the time resolved spectra to be saved to a text file
    :param photon_type: can be set to 's', 'i', or left as None in order to select thermal synchrotron photons, injected photons, or all the photons in the simulation for analysis
    :param energy_range: has units of keV, can be left as None to choose photons of all energy ranges for analysis or it can be set to an array with [min energy, max energy] e.g. [1, 10] for 1 to 10 keV (limits inclusive)

    :return: returns arrays of the various time resolved parameters and their errors. The order of the variables
            returned are: light curve, light curve error, alpha,beta,break energy, errors in the spectral parameters
            (a (times.size,3,2) array where times is the start of the light curve time bins), start times of the time bins, the best fit model in each time bin,
            polarization, stokes I, stokes Q, stokes U, stokes V, polarization and polarization angle errors (a ((times.size,2)) array), polarization angle,
            number of photons in each time bin
    """
    from mclib import lcur
    plt.ion()
    if save_spex:
        append=True
    else:
        append=False

    angle=np.double(event_file.split('_')[-1])

    times=np.arange(time_start,time_end+dt,dt)


    lc,lcur_e, num_ph, avg_e_ph, P, I, Q, U, V, Perr, P_angle, num_scatt, times=lcur(event_file,times,units=unit, theta=angle, dtheta=delta_theta, sim_dims=dim, h5=hdf5, photon_type=photon_type, energy_range=energy_range)


    #initalize matrices to hold best fit values, loop goes to one less value to encompass range from time_end-1 to time_end
    #have nan's to make sure values @ end dont get plotted or analyzed later on
    e_o=np.zeros(times.size)
    alpha=np.zeros(times.size)
    beta=np.zeros(times.size)
    beta[:]=np.nan
    alpha[:]=np.nan
    e_o[:]=np.nan
    err=np.zeros((times.size,3,2))
    if sys.version_info < (3,0):
        model_use=np.zeros(times.size, dtype='S')
    else:
        model_use=np.array([""]*times.size)

    #get curve fit of spectrum for each time period
    count=0
    for i in times[:-1]:
        print(i,i+dt)
        try:
            fit, error, model=cfit(event_file,i,i+dt, hdf5=hdf5, save_spex=save_spex, append=append, photon_type=photon_type)
            fit=np.array(fit)
            error=np.array(error)
            if choose_best:
                #see if any error bars are nan, if so replace values of band parameters with nan so they're not plotted
                if ((np.where(np.isnan(error)==True)[0].size >0) or (np.where(np.isnan(error)==True)[1].size >0) or (np.abs(error[1,0])> np.abs(1*fit[1])) or (np.abs(error[1,1])> np.abs(1*fit[1]))  ) and model=='b':
                    fit[:]=np.nan
                    model=""
                if ((np.where(np.isnan(error[0])==True)[0].size >0) or (np.where(np.isnan(error[2])==True)[0].size >0)  ) and model=='c':
                    fit[:]=np.nan
                    model=""
                if (np.abs(error[2][0])> np.abs(fit[2])/3) or (np.abs(error[2][1])> np.abs(fit[2])/3) and model=='c':
                    fit[:]=np.nan
                    model=""

        except TypeError:
            print('No data points for time interval')
            fit=np.array([np.nan,np.nan,np.nan])
            error=np.zeros((3,2))
            error[:,:]=np.nan
            model=""
            #print('In exception')
        except ValueError:
            print('No data points for time interval')
            fit=np.array([np.nan,np.nan,np.nan])
            error=np.zeros((3,2))
            error[:,:]=np.nan
            model=""
            #print('In exception')
        except RuntimeError:
            print('No optimal parameters found for time interval')
            fit=np.array([np.nan,np.nan,np.nan])
            error=np.zeros((3,2))
            error[:,:]=np.nan
            model=""



        alpha[count]=fit[0]
        beta[count]=fit[1]
        e_o[count]=fit[2]
        model_use[count]=model
        #get rid of huge errors that skew graph and huge betas that skew it as well
        #beta[beta<-10]=np.nan
        #error[error<-10**2]=np.nan
        err[count,:,:]=error
        count+=1

    if plotting:
        lcur_param_plot(event_file,lc, lcur_e, alpha,beta,e_o,err,np.arange(time_start,time_end+dt,dt), model_use, P, I, Q, U, V, Perr, P_angle, num_scatt, dt=dt, plotting=save_plot, lc_only=lc_only, pol_only=pol_only, h5=hdf5)

    return lc, lcur_e, alpha,beta,e_o,err, np.arange(time_start,time_end+dt,dt), model_use, P, I, Q, U, V, Perr, P_angle, num_ph, num_scatt

def lcur_param_var_t(event_file,time_start, time_end, dt=0.2, dt_min=0.2, phi_angle=0, delta_theta=1, liso_c=None, unit='erg/s', plotting=True, save_plot=False, choose_best=False, lc_only=False, pol_only=False, dim=2,riken_switch=False, hdf5=True, save_spex=None,  photon_type=None, energy_range=None, use_Lcrit=False, plot3=False ):
    """
    This function takes the MCRaT 'detected' photons from the event file and time bins them to produce light curves with
    variable time bins. The photons in a given time bin are then fitted with either the COMP or Band functions.

    This function should be called after viewing the data in uniform time bins. It uses a minimum luminosity cutoff to
    form time bins, if this critical limit is achieved, the time bin is recorded otherwise the function will keep
    increasing a giving time bin in a light curve to ensure that the critical luminosity is achieved.

    :param event_file: the base name of the event file that holds all of the info about the mock observed photons
            from MCRaT
    :param time: a python list or numpy array with the start time and the end time in chronological order e.g. time=[0, 35]
    :param dt: the light curve time bin width
    :param phi_angle: unused parameter, meant for future 3D development
    :param delta_theta: delta theta of the observer viewing angle for accepting photons, in degrees. Should be the same
            value as what is passed into event_h5's dtheta_deg
    :param liso_c: The critical luminosity that has to be achieved in order for the function to close a given time bin
    :param unit: A string of the units that will be used for the light curve/spectrum 'erg/s' or 'cts/s'
    :param plotting: Switch to allow the function to show the plots of the light curve/spectral parameters
    :param save_plot: Switch to allow the function to save the produced plot
    :param choose_best: Switch that permits the function to overwrite spectral parameters that do not have error bars
            to be ignored if set to True. If this is set to False all spectral parameters will be considered/plotted
    :param lc_only: switch that tells the function to only plot the light curve and peak energy of the time resolved
        spectra and none of the other parametres
    :param pol_only: Switch to allow the function to only plot the light curve and the polarization when set to True
    :param dim: The number of dimensions of the MCRaT simulation that is being analyzed. There is no support for 3D at
            this time so this switch should be set to 2, which is it's default value
    :param hdf5: Switch to specify if the MCRaT output files were HDF5 format or not, default is True. Should be set to
            False when an old version of MCRaT was used which outputted the data in text files
    :param save_spex: Switch that allows the time resolved spectra to be saved to a text file
    :param photon_type: can be set to 's', 'i', or left as None in order to select thermal synchrotron photons, injected photons, or all the photons in the simulation for analysis
    :param energy_range: has units of keV, can be left as None to choose photons of all energy ranges for analysis or it can be set to an array with [min energy, max energy] e.g. [1, 10] for 1 to 10 keV (limits inclusive)

    :return: returns arrays of the various time resolved parameters and their errors. The order of the variables
            returned are: light curve, light curve error, alpha,beta,break energy, errors in the spectral parameters
            (a (times.size,3,2) array where times is the start of the light curve time bins), start times of the time bins, the best fit model in each time bin,
            polarization, stokes I, stokes Q, stokes U, stokes V, polarization and polarization angle errors (a ((times.size,2)) array), polarization angle,
            number of photons in each time bin
    """


    from mclib import lcur_var_t
    plt.ion()
    if save_spex:
        append=True
    else:
        append=False

    angle=np.double(event_file.split('_')[-1])


    lc,lcur_e, num_ph, avg_e_ph, P, I, Q, U, V, Perr, P_angle, num_scatt, time=lcur_var_t(event_file,time_start,time_end, dt, dt_min, liso_c=liso_c, units=unit, theta=angle, dtheta=delta_theta, sim_dims=dim, h5=hdf5, photon_type=photon_type, energy_range=energy_range, use_Lcrit=use_Lcrit)


    #initalize matrices to hold best fit values, loop goes to one less value to encompass range from time_end-1 to time_end
    #have nan's to make sure values @ end dont get plotted or analyzed later on
    e_o=np.zeros(time.size)
    alpha=np.zeros(time.size)
    beta=np.zeros(time.size)
    beta[:]=np.nan
    alpha[:]=np.nan
    e_o[:]=np.nan
    err=np.zeros((time.size,3,2))
    if sys.version_info < (3,0):
        model_use=np.zeros(time.size, dtype='S')
    else:
        model_use=np.array([""]*time.size)

    #get curve fit of spectrum for each time period
    count=0
    for i in range(time.size-1):
        print(i)
        #print(i,time[count+1])
        print(lc[np.where(time[i]==time)], P[np.where(time[i]==time)])
        try:
            t_start=time[i]

            if (i==time.size-1):
                t_end=time[i]+np.abs(0.2)
            else:
                t_end = time[i + 1]
            fit, error, model=cfit(event_file,t_start,t_end, hdf5=hdf5, save_spex=save_spex, append=append, photon_type=photon_type)
            fit=np.array(fit)
            error=np.array(error)
            #print('In curve fit')
            if choose_best:
                #see if any error bars are nan, if so replace values of band parameters with nan so they're not plotted
                if ((np.where(np.isnan(error)==True)[0].size >0) or (np.where(np.isnan(error)==True)[1].size >0) or (np.abs(error[1,0])> np.abs(1*fit[1])) or (np.abs(error[1,1])> np.abs(1*fit[1]))  ) and model=='b':
                    fit[:]=np.nan
                    model=""
                if ((np.where(np.isnan(error[0])==True)[0].size >0) or (np.where(np.isnan(error[2])==True)[0].size >0)  ) and model=='c':
                    fit[:]=np.nan
                    model=""
                if (np.abs(error[2][0])> np.abs(fit[2])/3) or (np.abs(error[2][1])> np.abs(fit[2])/3) and model=='c':
                    fit[:]=np.nan
                    model=""

                if fit[2]<0:
                    #if e peak is negative dont plot points
                    fit[:]=np.nan
                    error[:,:]=np.nan

                if fit[0]>10:
                    #if e peak is negative dont plot points
                    fit[:]=np.nan
                    error[:,:]=np.nan


        except TypeError:
            print('No data points for time interval')
            fit=np.array([np.nan,np.nan,np.nan])
            error=np.zeros((3,2))
            error[:,:]=np.nan
            model=""
            #print('In exception')
        except ValueError:
            print('No data points for time interval')
            fit=np.array([np.nan,np.nan,np.nan])
            error=np.zeros((3,2))
            error[:,:]=np.nan
            model=""
            #print('In exception')
        except RuntimeError:
            print('No optimal parameters found for time interval')
            fit=np.array([np.nan,np.nan,np.nan])
            error=np.zeros((3,2))
            error[:,:]=np.nan
            model=""



        alpha[count]=fit[0]
        beta[count]=fit[1]
        e_o[count]=fit[2]
        model_use[count]=model
        #get rid of huge errors that skew graph and huge betas that skew it as well
        #beta[beta<-10]=np.nan
        #error[error<-10**2]=np.nan
        err[count,:,:]=error
        count+=1

    if plotting:
        lcur_param_plot(event_file,lc, lcur_e, alpha,beta,e_o,err,time, model_use, P, I, Q, U, V, Perr, P_angle, num_scatt, dt=-1, plotting=save_plot, lc_only=lc_only, pol_only=pol_only, h5=plot3, liso_c=liso_c)

    return lc, lcur_e, alpha,beta,e_o,err, time, model_use, P, I, Q, U, V, Perr, P_angle, num_ph, num_scatt


def lcur_param_plot(event_file,lcur, lcur_e, alpha,beta,e_o,err,t, model, P, I, Q, U, V, Perr, P_angle, num_scatt, dt=1, lc_only=False, pol_only=False, plotting=True, h5=False, liso_c=None, plot_optical=True):
    """
    The plotting function for the results of lcur_param and lcur_param_var_t functions.

    :param event_file: The event file name, a string that will be ncorporated into the saved pdf of the plot that is produced
    :param lcur: the values of the light curve at each time bin as an array
    :param lcur_e: the values of the error in he light curve values at each time bin
    :param alpha: an array of the fitted alpha parameters for each time resolved spectrum
    :param beta: an array of the fitted beta parameters for each time resolved spectrum
    :param e_o: an array of the fitted break energies for each time resolved spectrum
    :param err: A (t.size,3,2) array of errors in the spectral parameters
    :param t: An array of time bins both beginning and end that will be plotted
    :param model: An array of model types for the types of spectrum that best fit each time binned spectrum
    :param P: An array of he polarization at each time bin
    :param I:
    :param Q:
    :param U:
    :param V:
    :param Perr: The error in the polarization at each time bin
    :param P_angle: The polarization angle of the photons detected in each time bin
    :param dt: the size of the time bin if the time bins are uniformly size otherwise set it to be a negative umber
    :param lc_only: Switch to denote if the function should only plot the light curve or not
    :param pol_only: Switch to denote if the function should plot just the light curve and the polarization parameters
    :param plotting: Switch, set to True by default, to show the plot produced
    :param h5: A switch to denote that the MCRaT simulation that is being analyzed produced HDF5 files (for backwatds compatibility)
    :param liso_c: The value of the critical isotropic luminosity that was used to create the variable time bins
    :return: No returns
    """
    import matplotlib
    from mclib import lcur as lc
    import matplotlib.patheffects as pe

    plt.rcParams.update({'font.size':14})
    #plt.figure()
    #print(err[:,0,:].T.shape, alpha.shape)
    theta=np.double(event_file.split('_')[-1])
    #get center of time bin old
    if dt>0:
        t_cen=(t+(t+dt))/2
        x_err=dt/2
    else:
        difference=np.zeros(t.size)
        difference[:t.size-1]=np.diff(t)
        difference[-1]=np.diff(t).min()
        t_cen=(t+(t+difference))/2
        x_err=difference/2

    #for var_t

    #this if for figures for paper
    string=''

    if '16TI' in event_file:
        if theta==1 :
            #if event_file[0:4]=='16OI':
            string='(a) 16TI\n     '+r'$\theta_\mathrm{v}=1^\circ$'
        elif theta==7:
            string = '(b) 16TI\n     '+r'$\theta_\mathrm{v}=7^\circ$'
        else:
            string='(c) 16TI\n     '+r'$\theta_\mathrm{v}=12^\circ$'
    else:
        if theta==3:
            string = '(a) 40sp_down\n\t'+r'$\theta_\mathrm{v}=3^\circ$'
        else:
            string = '(b) 40sp_down\n\t'+r'$\theta_\mathrm{v}=7^\circ$'



    #do this for plotting optical alongside bolometri
    if plot_optical:
        #s_lc = lc(event_file, t, theta=theta, energy_range=[1.5e-3, 7.7e-3])[0] # just want the light curve
        s_lc = lc(event_file, t, theta=theta, energy_range=[1.81e-3, 2.62e-3])[0]
        lcur_max = lcur[~np.isnan(lcur)].max()
        lcur = lcur / lcur_max
        s_lc_max = s_lc[~np.isnan(s_lc)].max()
        s_lc = s_lc / s_lc_max
        str_lc_max=np.format_float_scientific(lcur_max, precision=2)
        str_s_lc_max=np.format_float_scientific(s_lc_max, precision=2)

    #calc proper error propagation for e_pk
    err[:,2,0]=np.sqrt(((2+alpha)* err[:,2,0])**2 + (e_o*err[:,0,0])**2  )
    err[:,2,1]=np.sqrt(((2+alpha)* err[:,2,1])**2 + (e_o*err[:,0,1])**2  )

    #plot
    if lc_only:
        f, axarr = plt.subplots(1,sharex=True, figsize=(10,5))
        formatter=matplotlib.ticker.ScalarFormatter( useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,1))

        #plt.rcParams.update({'mathtext.fontset':'stix'})
        axarr.yaxis.set_major_formatter(formatter)
        #axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #axarr[0].bar( t_cen,lcur,width=dt, align='center', color='None')
        axarr.plot( t,lcur,ls='steps', color='k',lw=2)
        axarr.set_ylabel('Light Curve (erg/s)')

        new_ax_2=axarr.twinx()
        index=np.where(model=='c')
        if np.size(xerr.shape)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',marker='.',ls='None', markersize='10')
        index=np.where(model=='b')
        if np.size(xerr.shape)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',mec='g',marker='.',mfc='white', ls='None', markersize='10')
        new_ax_2.set_ylabel(r'E$_{\mathrm{pk}}$ (keV)', color='g')
        axarr.set_xlabel('Time since Jet Launch (s)')
        #new_ax_2.plot(t_cen,e_o*(2+alpha), 'g-')

    elif pol_only:
        matplotlib.rcParams['axes.ymargin'] = 0
        matplotlib.rcParams['axes.xmargin'] = 0

        neg_alpha_index=np.where(alpha<0)
        pos_alpha_index=np.where(alpha>0)

        f, axarr = plt.subplots(2, sharex=True)

        formatter=matplotlib.ticker.ScalarFormatter( useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,1))

        #plt.rcParams.update({'mathtext.fontset':'stix'})
        axarr[0].yaxis.set_major_formatter(formatter)
        #axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #axarr[0].bar( t_cen,lcur,width=dt, align='center', color='None')
        axarr[0].plot( t,lcur,ds='steps-post', color='k',lw=2)
        axarr[0].set_ylabel('Light\nCurve (erg/s)')

        #annotate is for paper figures
        if 'down' in event_file:
            axarr[0].annotate('(c) 40sp_down\n\t'+r'$\theta_\mathrm{v}=7^\circ$',xy=(1, 1), xycoords='axes fraction', fontsize=16, xytext=(-120, -10), textcoords='offset points', ha='left', va='top')
        else:
            axarr[0].annotate('(a) 16TI\n  '+r'$\theta_\mathrm{v}=7^\circ$',xy=(0, 1), xycoords='axes fraction', fontsize=16, xytext=(10, -10), textcoords='offset points', ha='left', va='top')

        new_ax_2=axarr[0].twinx()
        index=np.intersect1d(np.where(model=='c'), pos_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',marker='.',ls='None')
        index=np.intersect1d(np.where(model=='b'), pos_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',mec='g',marker='.', ls='None')#mfc='white'

        index=np.intersect1d(np.where(model=='c'), neg_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',marker='.',ls='None')
        index=np.intersect1d(np.where(model=='b'), neg_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',mec='g',marker='.', ls='None')#,mfc='white'

        new_ax_2.set_ylabel(r'E$_{\mathrm{pk}}$ (keV)', color='g')



        index = np.where((model == 'c') | (model == 'b') | (model == ''))[0]
        if np.size(x_err) > 1:
            xerr_2 = x_err[index]
        else:
            xerr_2 = x_err
        # print(Perr, index )
        axarr[1].errorbar(t_cen[index], P[index] * 100, yerr=np.abs(Perr[index, 0] * 100),
                          xerr=np.ones(t[index].size) * xerr_2, color='k', marker='.', ls='None')
        # axarr[2].autoscale_view(scalex=False)
        # axarr[2].set_ylim([0,P[index].max()*100])
        axarr[1].set_ylabel(r'$\Pi (\%)$')
        new_ax_3 = axarr[1].twinx()
        new_ax_3.errorbar(t_cen[index], P_angle[index], yerr=np.abs(Perr[index, 1]),
                          xerr=np.ones(t[index].size) * xerr_2, color='darkmagenta', marker='.', ls='None')
        new_ax_3.plot(np.arange(t_cen.min(), t_cen.max()), np.zeros(np.size(np.arange(t_cen.min(), t_cen.max()))),
                      ls='--', color='darkmagenta', alpha=0.5)

        new_ax_3.set_ylabel(r'$\chi$ ($^\circ$)', color='darkmagenta')
        # new_ax_3.set_ylim([0, 180])
        new_ax_3.set_ylim([-90, 90])

        axarr[1].set_xlabel('Time since Jet Launch (s)')


    else:
        #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'

        matplotlib.rcParams['axes.ymargin'] = 0
        matplotlib.rcParams['axes.xmargin'] = 0
        if h5:
            f, axarr = plt.subplots(3, sharex=True)
        else:
            f, axarr = plt.subplots(2, sharex=True)
        formatter=matplotlib.ticker.ScalarFormatter( useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0,1))

        #plt.rcParams.update({'mathtext.fontset':'stix'})
        axarr[0].yaxis.set_major_formatter(formatter)
        #axarr[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #axarr[0].bar( t_cen,lcur,width=dt, align='center', color='None')
        if not plot_optical:
            axarr[0].plot( t,lcur,ds='steps-post', color='b',lw=2)
            axarr[0].set_ylabel('L (erg/s)')
        else:
            bolo_line=axarr[0].plot( t,lcur,ds='steps-post', color='deepskyblue',lw=2, label=r'Bolometric, L$_\mathrm{max}$=%s x10$^{%s}$ erg/s' % (str_lc_max.split('e+')[0], str_lc_max.split('e+')[1]))
            synch_line = axarr[0].plot(t, s_lc, ds='steps-post', color='deeppink', lw=2,
                                       label=r'Optical, L$_\mathrm{max}$=%s x10$^{%s}$ erg/s' % (
                                       str_s_lc_max.split('e+')[0], str_s_lc_max.split('e+')[1]),ls='-')
                                       #path_effects=[pe.Stroke(linewidth=3, foreground='k'), pe.Normal()])
            f.legend(loc='upper center', fontsize=10, ncol=2)
            axarr[0].set_ylabel(r'L/L$_\mathrm{max}$')
        #axarr[0].set_ylim([0,lcur.max()])

        #axarr[0].margins=(1.2)
        #axarr[0].autoscale_view(tight=True, scalex=False)
        #axarr[0].plot( t,lcur/lcur.max(),ls='steps-post', color='k',lw=2)
        #axarr[0].set_ylabel('L/'+ r'$L_\mathrm{max}$')


        #annotate is for paper figures
        if 'down' in event_file:
            axarr[0].annotate(string,xy=(1, 1), xycoords='axes fraction', fontsize=14, xytext=(-120, -10), textcoords='offset points', ha='left', va='top')
        else:
            axarr[0].annotate(string,xy=(0, 1), xycoords='axes fraction', fontsize=14, xytext=(65, -10), textcoords='offset points', ha='left', va='top')#5 for xytext x
        #plt.annotate(string,xy=(1, 1), xycoords='axes fraction', fontsize=16, xytext=(-30, -10), textcoords='offset points', ha='left', va='top')

        #axarr[0].locator_params(axis='y', nbins=6)


        #make plot such that band are open circles and comp are filled
        if not h5:
            spex_handle_idx=1
        else:
            spex_handle_idx=2

        #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
        neg_alpha_index=np.where(alpha<0)[0]
        pos_alpha_index=np.where(alpha>0)[0]



        index=np.intersect1d(np.where(model=='c'), pos_alpha_index)

        if np.size(x_err)>1:
            xerr_2=x_err[index]
            print('In if')
        else:
            xerr_2=x_err
            print('in else')
        #print(t_cen[index].size, alpha[index].size, err[index, 0, :].size, t[index].size, np.size(x_err.shape), xerr_2.size, x_err[index].shape)

        axarr[spex_handle_idx].errorbar(t_cen[index],alpha[index],yerr=np.abs(err[index,0,:].T),xerr=np.ones(t[index].size)*xerr_2 ,color='r',marker='.',ls='None')
        index=np.intersect1d(np.where(model=='b'), pos_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        axarr[spex_handle_idx].errorbar(t_cen[index],alpha[index],yerr=np.abs(err[index,0,:].T),xerr=np.ones(t[index].size)*xerr_2 ,color='r', mec='r',marker='.',mfc='white', ls='None')

        index=np.intersect1d(np.where(model=='c'), neg_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        axarr[spex_handle_idx].errorbar(t_cen[index],alpha[index],yerr=np.abs(err[index,0,:].T),xerr=np.ones(t[index].size)*xerr_2 ,color='r',marker='*',ls='None')
        index=np.intersect1d(np.where(model=='b'), neg_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        axarr[spex_handle_idx].errorbar(t_cen[index],alpha[index],yerr=np.abs(err[index,0,:].T),xerr=np.ones(t[index].size)*xerr_2 ,color='r', mec='r',marker='*',mfc='white', ls='None')

        axarr[spex_handle_idx].set_ylabel(r'$\alpha$', color='r')
        #if not h5:
        axarr[spex_handle_idx].set_xlabel('Time since Jet Launch (s)')
        if beta[~np.isnan(beta)].size >0:
            new_ax=axarr[spex_handle_idx].twinx()
            if np.size(x_err) > 1:
                xerr_2 = x_err[pos_alpha_index]
            else:
                xerr_2 = x_err

            new_ax.errorbar(t_cen[pos_alpha_index],beta[pos_alpha_index],yerr=np.abs(err[pos_alpha_index,1,:].T),xerr=np.ones(t[pos_alpha_index].size)*xerr_2,color='b',marker='.',ls='None', mfc='white', mec='b')
            if np.size(x_err) > 1:
                xerr_2 = x_err[neg_alpha_index]
            else:
                xerr_2 = x_err
            new_ax.errorbar(t_cen[neg_alpha_index],beta[neg_alpha_index],yerr=np.abs(err[neg_alpha_index,1,:].T),xerr=np.ones(t[neg_alpha_index].size)*xerr_2,color='b',marker='*',ls='None', mfc='white', mec='b')

            new_ax.set_ylabel(r'$\beta$', color='b')
        #matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'
        #axarr[1].autoscale_view(scalex=False)
        new_ax_2=axarr[0].twinx()
        index=np.intersect1d(np.where(model=='c'), pos_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',marker='.',ls='None')
        index=np.intersect1d(np.where(model=='b'), pos_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',mec='g',marker='.',mfc='white', ls='None')

        index=np.intersect1d(np.where(model=='c'), neg_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',marker='*',ls='None')
        index=np.intersect1d(np.where(model=='b'), neg_alpha_index)
        if np.size(x_err)>1:
            xerr_2=x_err[index]
        else:
            xerr_2=x_err

        new_ax_2.errorbar(t_cen[index],e_o[index]*(2+alpha[index]),yerr=np.abs(err[index,2,:].T),xerr=np.ones(t[index].size)*xerr_2,color='g',mec='g',marker='*',mfc='white', ls='None')

        new_ax_2.set_ylabel(r'E$_{\mathrm{pk}}$ (keV)', color='g')

        #new_ax_2.autoscale_view(scalex=False)
        if h5:
            index=np.where((model=='c') | (model=='b'))[0]
            if np.size(x_err) > 1:
                xerr_2 = x_err[index]
            else:
                xerr_2 = x_err
            #print(Perr, index )
            axarr[1].errorbar(t_cen[index],P[index]*100,yerr=np.abs(Perr[index,0]*100),xerr=np.ones(t[index].size)*xerr_2,color='k', marker='.',ls='None')
            #axarr[2].autoscale_view(scalex=False)
            #axarr[2].set_ylim([0,P[index].max()*100])
            axarr[1].set_ylabel(r'$\Pi (\%)$')
            if (axarr[1].get_ylim()[1] > 100):
                axarr[1].set_ylim([0, 100])
                axarr[1].set_yticks([ 0, 25, 50, 75, 100])
            if (axarr[1].get_ylim()[0] < 0):
                axarr[1].set_ylim([0, axarr[1].get_ylim()[1]])



            new_ax_3=axarr[1].twinx()
            new_ax_3.errorbar(t_cen[index], P_angle[index] , yerr=np.abs(Perr[index,1] ), xerr=np.ones(t[index].size) * xerr_2, color='darkmagenta', marker='.', ls='None')
            new_ax_3.plot(np.arange(t_cen.min(), t_cen.max()), np.zeros(np.size(np.arange(t_cen.min(), t_cen.max()))), ls='--', color='darkmagenta', alpha=0.5 )

            new_ax_3.set_ylabel(r'$\chi$ ($^\circ$)', color='darkmagenta')
            #new_ax_3.set_ylim([0, 180])
            new_ax_3.set_ylim([-90, 90])
            new_ax_3.set_yticks([-90, -45, 0, 45, 90])


            #axarr[1].set_xlabel('Time since Jet Launch (s)')

        #new_ax_2.locator_params(axis='y', nbins=6)

        #to point out where in lc we are analyzing
        """
        if 'sp' in event_file:
            if '1' in event_file:
                t=[9.1, 12.1, 18.1]
            if '40' in event_file:
                t=[20.1, 25.1, 36.1]
            if 'down' in event_file:
                t=[20.1,26.1,38.1]
        
            min=axarr[0].get_ylim()[0]
            max=axarr[0].get_ylim()[1]
            axarr[0].vlines(t[0], min,max, lw=1, linestyle='dotted', color='gray')
            axarr[0].vlines(t[1], min,max, lw=1, linestyle='dotted', color='gray')
            axarr[0].vlines(t[2], min,max, lw=1, linestyle='dotted', color='gray')
        """
        #extraticks=[]#[20.1, 25.1, 36.1]
        #axarr[0].set_xticks(list(axarr[0].get_xticks()) + extraticks)

    #plt.tight_layout()
    plt.show()
    #axarr[0].set_title('Lightcurve and Peak Energy, with Band Function Parameters\nfor $ \\theta $='+np.str_(theta))
    #axarr[0].set_title('Lightcurve and Peak Energy')
    ##axarr.bar( t_cen,lcur,width=dt, align='center', color='None')
    ##axarr.set_ylabel('Light Curve (cts/s)')
    ##new_ax_2=axarr.twinx()
    ##new_ax_2.errorbar(t_cen,e_pk*(2+alpha),yerr=np.abs(err[:,2,:].T),xerr=np.ones(t.size)*x_err,color='g',marker='o',ls='None')
    ##new_ax_2.set_ylabel(r'E$_{\mathrm{pk}}$ (keV)')
    ##axarr.set_title('Lightcurve and Peak Energy')
    if plotting:
        if lc_only:
            if dt > 0:
                savestring=event_file.replace('.','_')+'_lc'
            else:
                savestring= event_file.replace('.', '_') + '_lc_liso_c_%s_dt_var'%(np.str(liso_c))
        else:
            if dt > 0:
                savestring=event_file.replace('.','_')+'_dt_'+np.str(dt).replace('.','_')
            else:
                savestring= event_file.replace('.', '_') + '_liso_c_%s_dt_var'%(np.str(liso_c))

        if plot_optical:
            savestring=savestring+'_w_optical'

        f.savefig('EVENT_FILE_ANALYSIS_PLOTS/' + savestring +'.pdf', bbox_inches='tight')

def get_yonetoku_rel(simid_array, t_start, t_end, dt=1, h5=False, save_plot=False):
    """
    A function to plot the MCRaT analyzed GRB alongside the yonetoku relationship and a list of observed GRBs as
    provided by GRB_list.dat (observed GRBS are from Nava et al. (2012))

    :param simid_array: An array of the event file name bases of the MCRaT result that will be plotted
    :param t_start: An array of the start times for each event file passed to simid_array
    :param t_end: An array of the end times for each event file passed to simid_array
    :param dt: The time bin width, default is 1 following how the relationship is derived
    :param h5: Switch to specify if the MCRaT output files were HDF5 format or not, default is True. Should be set to
            False when an old version of MCRaT was used which outputtted the data in text files
    :param save_plot: Switch to save the plot as a PDF or not
    :return: Returns arrays of the plotted peak energy, its error, the luminosity and the erro in the luminosity for
                each event file that was provided to simid_array
    """
    import matplotlib
    plt.rcParams.update({'font.size':14})
    from mclib import lcur
    from matplotlib.font_manager import FontProperties
    import matplotlib.lines as mlines
    import matplotlib
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    symbols=np.array(['*', 'D', '^', 'P', '8', 's', '|', 'o', '4', 'X', 'd', 'h',  '1', 'p','+'])

    simid_array=np.array(simid_array)
    #file_data=np.genfromtxt('yoketoku_relation.dat')
    file_data=np.genfromtxt('Data_files/GRB_list.dat',dtype='S',usecols=(8,10), delimiter='\t')

    E_p=np.zeros(file_data.shape[0])
    L_iso=np.zeros(file_data.shape[0])

    E_p_err=np.zeros(file_data.shape[0])
    L_iso_err=np.zeros(file_data.shape[0])

    #ll_e_p=np.array([29.7,32.1,63.2,-35,-785,-150,56.9,87.8,32.1,41.3,83.1])
    #ul_e_p=np.array([37.8,29.4,48.6,100,1085,0,49.8,69.2,22.3,37.3,97.5])

    #ll_l=np.array([0.01,0.15,0.17,-0.24,-12.49,0.05,0.23,0.1,0.06,0.11,7.88])
    #ul_l=np.array([0.01,0.15,0.17,0.4,72.38,0.05,0.23,0.1,0.06,0.11,7.88])

    #peak_luminosity *10^51
    #E_p(1+z) in keV
    count=0
    for i in range(file_data.shape[0]):
        if ((np.size(np.where(np.fromstring(file_data[i,0], sep=' \xc2\xb1 ')!=-1)))!=0) and ((np.size(np.where(np.fromstring(file_data[i,1], sep=' \xc2\xb1 ')!=-1)))!=0):
            E_p[count]=np.fromstring(file_data[i,0], sep=' \xc2\xb1 ')
            E_p_err[count]=np.float(np.fromstring(file_data[i,0][::-1], sep=' \xc2\xb1 ').astype(int).astype('U')[0][::-1])
            L_iso[count]=np.fromstring(file_data[i,1], sep=' \xc2\xb1 ')
            L_iso_err[count]=np.float(np.fromstring(file_data[i,1][::-1], sep=' \xc2\xb1 ').astype(float).astype('U')[0][::-1])
            count+=1

    #divide L by ten to have same normalization factor as in yonetoku paper
    #plt.errorbar(E_p[E_p!=0],L_iso[L_iso!=0]/10,xerr=[E_p_err[E_p!=0],E_p_err[E_p!=0]], yerr=[L_iso_err[L_iso!=0]/10,L_iso_err[L_iso!=0]/10], marker='o', ls='None')
    fig, ax = plt.subplots()
    ax.errorbar(L_iso[L_iso!=0]/10,E_p[E_p!=0],yerr=[E_p_err[E_p!=0],E_p_err[E_p!=0]], xerr=[L_iso_err[L_iso!=0]/10,L_iso_err[L_iso!=0]/10], color='grey',marker='o', ls='None')
    e=np.linspace(E_p[E_p!=0].min()/10,E_p[E_p!=0].max()*10,100)
    #plt.loglog(e,(2.34e-5)*e**2)
    ax.loglog((2.34e-5)*e**2, e, color='grey')

    #zoom_ax=zoomed_inset_axes(ax,7, loc=4, borderpad=1.7)
    #zoom_ax.set_xscale("log")
    #zoom_ax.set_yscale("log")


    L_iso_sim=np.zeros([simid_array.size])
    L_err_sim=np.zeros([simid_array.size])
    E_p_sim=np.zeros([simid_array.size,1])
    E_p_err_sim=np.zeros((simid_array.size,2))
    #info=np.array(["                " for x in range(simid_array.size)])
    count=0
    for i in simid_array:
        print(i)
        t=np.arange(0,t_end[count],dt) #should change for sims that arent 105 seconds long
        lc=np.zeros([t.size])
        lc_e=np.zeros([t.size])

        try:
            angle=np.double(i[-2:])
        except ValueError:
            angle=np.float(i[::-1][0])
        lc[:],lc_e[:], num_ph, dum, p, l, q, u, v, perr, p_angle, num_scatt, t=lcur(i,np.arange(0,t_end[count],dt), units='erg/s', theta=angle, h5=h5)
        L_iso_sim[count]=lc.max()/(1e52) #scale to yonetoku paper
        L_err_sim[count]=lc_e[lc.argmax()]/(1e52)

        best,err, model=cfit(i,t_start[count],t_end[count], hdf5=h5) # t[lc.argmax()],t[lc.argmax()]+1)
        E_p_sim[count]=best[2]*(2+best[0])

        #if E_p_sim[count]<0:
        #    E_p_sim[count]=np.nan

        #calc proper error propagation for e_pk
        err[2,0]=np.sqrt(((2+best[0])* err[2,0])**2 + (best[2]*err[0,0])**2  )
        err[2,1]=np.sqrt(((2+best[0])* err[2,1])**2 + (best[2]*err[0,1])**2  )

        E_p_err_sim[count,:]=err[-1,:]
        info=i[0:4]+r': $\theta$='+i[-1]
        #print(i)
        #print(E_p_err_sim[count,:],E_p_sim[count])
        #print(E_p_err_sim[count,:].T.shape,E_p_sim[count].shape)
        if angle==1:
            symbol='*'
            m_size='10'
        elif angle==2:
            symbol='D'
            m_size='6'
        elif angle==3:
            symbol='^'
            m_size='9'
        elif angle==4:
            symbol='P'
            m_size='9'
        elif angle==5:
            symbol='8'
            m_size='9'
        elif angle==6:
            symbol='s'
            m_size='9'
        elif angle==7:
            symbol=symbols[np.int(angle)-1]
            m_size='12'
        elif angle==8:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==9:
            symbol=symbols[np.int(angle)-1]
            m_size='12'
        elif angle==10:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==11:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==12:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==13:
            symbol=symbols[np.int(angle)-1]
            m_size='12'
        elif angle==14:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==15:
            symbol=symbols[np.int(angle)-1]
            m_size='12'

        if '1spike' in i:
            c='r'
        if '40spikes' in i:
            c='b'
        if '40sp_down' in i:
            c='g'
        if '16TI' in i:
            c = 'r'

        elif i[0:4]=='16TI':
            c='goldenrod'
            #sub ifs for the two other kinds of 16TI
            if i[4:9]=='.e150':
                c='g'
                zoom_ax.errorbar(L_iso_sim[count],E_p_sim[count],xerr=L_err_sim[count], yerr=[np.abs(err[-1].T)], color=c, marker=symbol,ls='None',markersize=m_size, label=info)

            if i[4:14]=='.e150.g100':
                c='darkmagenta'
                zoom_ax.errorbar(L_iso_sim[count],E_p_sim[count],xerr=L_err_sim[count], yerr=[np.abs(err[-1].T)], color=c, marker=symbol,ls='None',markersize=m_size, label=info)

        elif i[0:4]=='35OB':
            c='darkmagenta'
        elif i[0:4]=='CMC_':
            c='k'

        #if count==0 or count==3 or count==6:
        #plt.errorbar(E_p_sim[count],L_iso_sim[count],yerr=L_err_sim[count], xerr=[np.abs(err[-1].T)], color=c, marker=symbol,ls='None',markersize=m_size, label=info)
        #elif count==2 or count==4 or count==7:
        ax.errorbar(L_iso_sim[count],E_p_sim[count],xerr=L_err_sim[count], yerr=np.reshape(err[-1], (2,1)) , color=c, marker=symbol,ls='None',markersize=m_size, label=info)

        #else:
        #	plt.errorbar(E_p_sim[count],L_iso_sim[count],yerr=L_err_sim[count], xerr=[np.abs(err[-1].T)], color=c,marker=symbol,ls='None',markersize='8', label=info)

        count+=1
    print(L_iso_sim[:],E_p_sim[:], np.where(E_p_sim>0)[0])

    idx=np.where(E_p_sim>0)[0]
    E_p_sim = E_p_sim[idx]
    L_iso_sim=L_iso_sim[idx]

    #oi,  =plt.plot(E_p_sim[0:3],L_iso_sim[0:3],'r', linewidth=2, label='16OI')
    #ti,  =plt.plot(E_p_sim[3:6],L_iso_sim[3:6],'goldenrod', linewidth=2, label='16TI')
    #ob,  =plt.plot(E_p_sim[6:9],L_iso_sim[6:9],'k', linewidth=2, label='35OB')
    #ti_e150_g100,  =plt.plot(E_p_sim[9:12],L_iso_sim[9:12],'darkmagenta', linewidth=2, label='16TI.e150.g100')
    #plt.plot(E_p_sim[12:15],L_iso_sim[12:15],'g', linewidth=2, label='16TI.e150')

    #oi,  =ax.plot(L_iso_sim[12:18],E_p_sim[12:18],'r', linewidth=2, label='1spike')
    #forty_spikes,  =ax.plot(L_iso_sim[6:12],E_p_sim[6:12],'b', linewidth=2, label='40spikes')
    forty_sp,  =ax.plot(L_iso_sim[15:],E_p_sim[15:],'g', linewidth=2, label='40sp_down')
    ti,  =ax.plot(L_iso_sim[:15],E_p_sim[:15],'r', linewidth=2, label='16TI')
    #ob,  =ax.plot(L_iso_sim[24:],E_p_sim[24:],'darkmagenta', linewidth=2, label='35OB')
    #ti_e150_g100,  =ax.plot(L_iso_sim[9:12],E_p_sim[9:12],'darkmagenta', linewidth=2, label='16TI.e150.g100')
    #ti_e150,  =ax.plot(L_iso_sim[12:15],E_p_sim[12:15],'g', linewidth=2, label='16TI.e150')
    #cmc_oi,  =ax.plot(L_iso_sim[18:21],E_p_sim[18:21],'k', linewidth=2, label='16OI')
    #plt.plot(E_p_sim[12:15],L_iso_sim[12:15],'g', linewidth=2, label='16TI.e150')


    theta_1=mlines.Line2D([],[],color='grey', marker='*', ls='None', markersize=10, label=r': $\theta_v= 1^ \circ$')
    theta_2=mlines.Line2D([],[],color='grey', marker='D', ls='None', markersize=6, label=r': $\theta_v= 2^ \circ$')
    theta_3=mlines.Line2D([],[],color='grey', marker='^', ls='None', markersize=8, label=r': $\theta_v= 3^ \circ$')
    theta_4=mlines.Line2D([],[],color='grey', marker='P', ls='None', markersize=8, label=r': $\theta_v= 4^ \circ$')
    theta_5=mlines.Line2D([],[],color='grey', marker='8', ls='None', markersize=8, label=r': $\theta_v= 5^ \circ$')
    theta_6=mlines.Line2D([],[],color='grey', marker='s', ls='None', markersize=8, label=r': $\theta_v= 6^ \circ$')
    theta_7 = mlines.Line2D([], [], color='grey', marker=symbols[6], ls='None', markersize=8, label=r': $\theta_v= 7^ \circ$')
    theta_8 = mlines.Line2D([], [], color='grey', marker=symbols[7], ls='None', markersize=8, label=r': $\theta_v= 8^ \circ$')
    theta_9 = mlines.Line2D([], [], color='grey', marker=symbols[8], ls='None', markersize=8, label=r': $\theta_v= 9^ \circ$')
    theta_10 = mlines.Line2D([], [], color='grey', marker=symbols[9], ls='None', markersize=8, label=r': $\theta_v= 10^ \circ$')
    theta_11 = mlines.Line2D([], [], color='grey', marker=symbols[10], ls='None', markersize=8, label=r': $\theta_v= 11^ \circ$')
    theta_12 = mlines.Line2D([], [], color='grey', marker=symbols[11], ls='None', markersize=8, label=r': $\theta_v= 12^ \circ$')
    theta_13 = mlines.Line2D([], [], color='grey', marker=symbols[12], ls='None', markersize=8, label=r': $\theta_v= 13^ \circ$')
    theta_14 = mlines.Line2D([], [], color='grey', marker=symbols[13], ls='None', markersize=8, label=r': $\theta_v= 14^ \circ$')
    theta_15 = mlines.Line2D([], [], color='grey', marker=symbols[14], ls='None', markersize=8, label=r': $\theta_v= 15^ \circ$')


    fontP=FontProperties()
    #fontP.set_size('small') prop=fontP,
    ax.legend( loc='upper left', ncol=2, handles=[ti, forty_sp,  theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9, theta_10, theta_11, theta_12, theta_13, theta_14, theta_15], numpoints=1,fontsize='10') #cmc_oi, ti, ob,ti_e150, ti_e150_g100, cmc_oi,
    #plt.legend(loc='upper left', bbox_to_anchor=(1,0.5))
    ax.set_ylabel(r'E$_{\mathrm{pk}}$ (keV)')
    ax.set_xlabel(r'$\frac{L_\mathrm{iso}}{10^{52}}$ (erg/s)')
    ax.set_ylim([4,3e3])
    ax.set_xlim([2e-4, 150])

    #plt.rcParams.update({'font.size':12})
    #zoom_ax.plot(L_iso_sim[12:15],E_p_sim[12:15],'g', linewidth=2, label='16TI.e150')
    #zoom_ax.plot(L_iso_sim[9:12],E_p_sim[9:12],'darkmagenta', linewidth=2, label='16TI.e150.g100')
    #zoom_ax.set_xlim([0.14,0.2])
    #zoom_ax.set_ylim([55,65])
    ##plt.yticks(visible=False)
    ##plt.xticks(visible=False)
    #mark_inset(ax, zoom_ax, loc1=2, loc2=3, fc="None", ec="0.5")

    plt.tight_layout()
    plt.show()
    if save_plot:
        fig.savefig('yonetoku.pdf',bbox_inches='tight')

    return E_p[E_p!=0],E_p_err[E_p!=0],L_iso[E_p!=0]/10,L_iso_err[E_p!=0]/10


def get_FERMI_best_data():
    """
    A function to acquire data about the FERMI Best GRB sample, as is saved in the file named FERMI_BEST_GRB.dat.
    The data is from Yu et al. (2016).

    :return: returns arrays of the Band or COMP function fitted GRB spectral parameters
    """

    data=np.genfromtxt('Data_files/FERMI_BEST_GRB.dat', dtype='U', usecols=(4,7,9,11 ))

    #only want BAND and COMP ones
    Band_Comp_data=data[np.logical_or(data[:,0]=='BAND', data[:,0]=='COMP') ,:]

    parameters=np.zeros([Band_Comp_data.shape[0], 3])
    parameters[:]=np.nan
    parameters[:,0]=Band_Comp_data[:,1].astype("f8") #alpha
    parameters[:,2]=Band_Comp_data[:,3].astype("f8") #peak energy
    parameters[Band_Comp_data[:,0]=='BAND' ,1]=Band_Comp_data[Band_Comp_data[:,0]=='BAND' ,2].astype("f8") #band beta

    #alphas in 1st column, betas in 2nd, etc.
    return parameters

def get_amati_rel(simid_array, time_start, time_end, save_plot=False, h5=False):
    """
    A function to plot the MCRaT analyzed GRB alongside the amati relationship and a list of observed GRBs as
    provided by GRB_list.dat (observed GRBS are from Nava et al. (2012))

    :param simid_array: An array of the event file name bases of the MCRaT result that will be plotted
    :param t_start: An array of the start times for each event file passed to simid_array
    :param t_end: An array of the end times for each event file passed to simid_array
    :param save_plot: Switch to save the plot as a PDF or not
    :param h5: Switch to specify if the MCRaT output files were HDF5 format or not, default is True. Should be set to
            False when an old version of MCRaT was used which outputtted the data in text files
    :return: No returns
    """
    import matplotlib
    plt.rcParams.update({'font.size':14})
    from mclib import lcur
    from matplotlib.font_manager import FontProperties
    import matplotlib.lines as mlines
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    simid_array = np.array(simid_array)
    symbols=np.array(['*', 'D', '^', 'P', '8', 's', '|', 'o', '4', 'X', 'd', 'h',  '1', 'p','+'])


    file_data=np.genfromtxt('Data_files/GRB_list.dat',dtype='S',usecols=(8,9), delimiter='\t')
    E_p=np.zeros(file_data.shape[0])
    E_iso=np.zeros(file_data.shape[0])

    E_p_err=np.zeros(file_data.shape[0])
    E_iso_err=np.zeros(file_data.shape[0])

    count=0
    for i in range(file_data.shape[0]):
        if ((np.size(np.where(np.fromstring(file_data[i,0], sep=' \xc2\xb1 ')!=-1)))!=0) and ((np.size(np.where(np.fromstring(file_data[i,1], sep=' \xc2\xb1 ')!=-1)))!=0):
            #print(i)
            E_p[count]=np.fromstring(file_data[i,0], sep=' \xc2\xb1 ')
            E_p_err[count]=np.float(np.fromstring(file_data[i,0][::-1], sep=' \xc2\xb1 ').astype(int).astype('U')[0][::-1])
            E_iso[count]=np.fromstring(file_data[i,1], sep=' \xc2\xb1 ')
            E_iso_err[count]=np.float(np.fromstring(file_data[i,1][::-1], sep=' \xc2\xb1 ').astype(float).astype('U')[0][::-1])
            count+=1

    #Data already normalized by 10^52
    #d=np.abs(np.trapz(E_p[E_p!=0],x=E_iso[E_iso!=0]))
    fig, ax = plt.subplots()
    #ax.errorbar(E_iso[E_iso!=0],E_p[E_p!=0],yerr=[E_p_err[E_p!=0],E_p_err[E_p!=0]], xerr=[E_iso_err[E_iso!=0],E_iso_err[E_iso!=0]], color='grey', marker='o', ls='None')
    #e=np.linspace(E_iso[E_iso!=0].min(),E_iso[E_iso!=0].max(),100)
    #f=np.trapz(((e)**0.52), x=e)
    #e=np.linspace(E_iso[E_iso!=0].min()/10,E_iso[E_iso!=0].max()*10,100)
    #ax.loglog(e, ((e)**0.52)*d/f, 'grey') #normalize relationship to the data
    x,y=amati()
    x_m,y_m=amati(value='-')
    x_p,y_p=amati(value='+')
    ax.loglog(10**x, 10**y, color='grey')
    ax.loglog(10**x_p, 10**y_p, color= 'grey', ls='-.')
    ax.loglog(10**x_m, 10**y_m, color= 'grey', ls='-.')

    #zoom_ax=zoomed_inset_axes(ax,10, loc=4, borderpad=1.5)
    #zoom_ax.set_xscale("log")
    #zoom_ax.set_yscale("log")

    #get best fit, time integrated model spectrum and integrate it from 1 keV to 10 MeV and use peak energy of that spectrum for E_pk
    E_iso_sim=np.zeros([simid_array.size])
    E_err_sim=np.zeros([simid_array.size])
    E_p_sim=np.zeros([simid_array.size,1])
    E_p_err_sim=np.zeros((simid_array.size,2))
    #info=np.array(["                " for x in range(simid_array.size)])
    count=0
    for i in simid_array:

        try:
            angle=np.double(i[-2:])
        except ValueError:
            angle=np.float(i[::-1][0])

        t=np.arange(time_start[count],time_end[count],1)
        lc=np.zeros([t.size])
        lc_e=np.zeros([t.size])
        lc[:],lc_e[:], num_ph, dum, p, l, q, u, v, perr, p_angle, num_scatt=lcur(i,np.arange(time_start[count],time_end[count],1),units='erg/s', theta=angle, h5=h5)



        best,err, model=cfit(i,time_start[count],time_end[count], hdf5=h5)
        E_p_sim[count]=best[2]*(2+best[0])

        #calc proper error propagation for e_pk
        #print(err)
        err[2,0]=-np.sqrt(((2+best[0])* err[2,0])**2 + (best[2]*err[0,0])**2  )
        err[2,1]=np.sqrt(((2+best[0])* err[2,1])**2 + (best[2]*err[0,1])**2  )
        #print(err)
        E_p_err_sim[count,:]=err[-1]

        #dnulog=.1
        #numin=10**np.arange(0,4,dnulog)
        #numax=numin*10**dnulog
        #nucen=np.sqrt(numax*numin)
        #data_pts=spex.size
        #if model=='b':
        #	spex=Band(nucen,best[0],best[1],best[2],best[3])
        #else:
        #	spex=comp(nucen,best[0],best[2],best[3])

        E_iso_sim[count]=np.trapz(lc,x=t)/(1e52) #integrate and normalize by same amount data is normalized or by the integral of the
        E_err_sim[count]=np.trapz(lc_e,x=t)/(1e52) #not sure how to calculate this error??????

        info=i[0:4]+r': $\theta$='+i[-1]
        #print(i)
        #print(E_p_err_sim[count,:],E_p_sim[count])
        #print(E_p_err_sim[count,:].T.shape,E_p_sim[count].shape)
        #c='k'
        if angle==1:
            symbol='*'
            m_size='10'
        elif angle==2:
            symbol='D'
            m_size='6'
        elif angle==3:
            symbol='^'
            m_size='9'
        elif angle==4:
            symbol='P'
            m_size='9'
        elif angle==5:
            symbol='8'
            m_size='9'
        elif angle==6:
            symbol='s'
            m_size='9'
        elif angle==7:
            symbol=symbols[np.int(angle)-1]
            m_size='12'
        elif angle==8:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==9:
            symbol=symbols[np.int(angle)-1]
            m_size='12'
        elif angle==10:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==11:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==12:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==13:
            symbol=symbols[np.int(angle)-1]
            m_size='12'
        elif angle==14:
            symbol=symbols[np.int(angle)-1]
            m_size='10'
        elif angle==15:
            symbol=symbols[np.int(angle)-1]
            m_size='12'


        if '1spike' in i:
            c='r'
        if '40spikes' in i:
            c='b'
        if '40sp_down' in i:
            c='g'
        if '16TI' in i:
            c = 'r'

        elif i[0:4]=='16TI':
            c='goldenrod'
            #sub ifs for the two other kinds of 16TI
            if i[4:9]=='.e150':
                c='g'
                zoom_ax.errorbar(E_iso_sim[count],E_p_sim[count],xerr=E_err_sim[count], yerr=[np.abs(err[-1].T)], color=c, marker=symbol,ls='None',markersize=m_size, label=info)
            if i[4:14]=='.e150.g100':
                c='darkmagenta'
                #m_size=np.str((np.int(m_size)+4))
                zoom_ax.errorbar(E_iso_sim[count],E_p_sim[count],xerr=E_err_sim[count], yerr=[np.abs(err[-1].T)], color=c, marker=symbol,ls='None',markersize=m_size, label=info)

        elif i[0:4]=='35OB':
            c='darkmagenta'
        elif i[0:4]=='CMC_':
            c='k'

        #if count==0 or count==3 or count==6:
        #plt.errorbar(E_p_sim[count],L_iso_sim[count],yerr=L_err_sim[count], xerr=[np.abs(err[-1].T)], color=c, marker=symbol,ls='None',markersize=m_size, label=info)
        #elif count==2 or count==4 or count==7:
        ax.errorbar(E_iso_sim[count],E_p_sim[count],xerr=E_err_sim[count], yerr=[np.abs(err[-1].T)], color=c, marker=symbol,ls='None',markersize=m_size, label=info)
        #else:
        #	plt.errorbar(E_p_sim[count],L_iso_sim[count],yerr=L_err_sim[count], xerr=[np.abs(err[-1].T)], color=c,marker=symbol,ls='None',markersize='8', label=info)

        count+=1
    print(E_iso_sim, E_p_sim)
    idx = np.where(E_p_sim > 0)[0]
    E_p_sim = E_p_sim[idx]
    E_iso_sim = E_iso_sim[idx]

    #plt.plot(E_p_sim[12:15],L_iso_sim[12:15],'g', linewidth=2, label='16TI.e150')
    #oi,  =ax.plot(E_iso_sim[12:18],E_p_sim[12:18],'r', linewidth=2, label='1spike')
    #forty_spikes,  =ax.plot(E_iso_sim[6:12],E_p_sim[6:12],'b', linewidth=2, label='40spikes')
    forty_sp,  =ax.plot(E_iso_sim[14:],E_p_sim[14:],'g', linewidth=2, label='40sp_down')
    ti,  =ax.plot(E_iso_sim[:14],E_p_sim[:14],'r', linewidth=2, label='16TI')
    #ob,  =ax.plot(E_iso_sim[24:],E_p_sim[24:],'darkmagenta', linewidth=2, label='35OB')
    #ti_e150_g100,  =ax.plot(E_iso_sim[9:12],E_p_sim[9:12],'darkmagenta', linewidth=2, label='16TI.e150.g100')
    #ti_e150,  =ax.plot(E_iso_sim[12:15],E_p_sim[12:15],'g', linewidth=2, label='16TI.e150')
    #cmc_oi,  =ax.plot(E_iso_sim[18:21],E_p_sim[18:21],'k', linewidth=2, label='16OI')

    theta_1=mlines.Line2D([],[],color='grey', marker='*', ls='None', markersize=10, label=r': $\theta_v= 1^ \circ$')
    theta_2=mlines.Line2D([],[],color='grey', marker='D', ls='None', markersize=6, label=r': $\theta_v= 2^ \circ$')
    theta_3=mlines.Line2D([],[],color='grey', marker='^', ls='None', markersize=8, label=r': $\theta_v= 3^ \circ$')
    theta_4=mlines.Line2D([],[],color='grey', marker='P', ls='None', markersize=8, label=r': $\theta_v= 4^ \circ$')
    theta_5=mlines.Line2D([],[],color='grey', marker='8', ls='None', markersize=8, label=r': $\theta_v= 5^ \circ$')
    theta_6=mlines.Line2D([],[],color='grey', marker='s', ls='None', markersize=8, label=r': $\theta_v= 6^ \circ$')
    theta_7 = mlines.Line2D([], [], color='grey', marker=symbols[6], ls='None', markersize=8, label=r': $\theta_v= 7^ \circ$')
    theta_8 = mlines.Line2D([], [], color='grey', marker=symbols[7], ls='None', markersize=8, label=r': $\theta_v= 8^ \circ$')
    theta_9 = mlines.Line2D([], [], color='grey', marker=symbols[8], ls='None', markersize=8, label=r': $\theta_v= 9^ \circ$')
    theta_10 = mlines.Line2D([], [], color='grey', marker=symbols[9], ls='None', markersize=8, label=r': $\theta_v= 10^ \circ$')
    theta_11 = mlines.Line2D([], [], color='grey', marker=symbols[10], ls='None', markersize=8, label=r': $\theta_v= 11^ \circ$')
    theta_12 = mlines.Line2D([], [], color='grey', marker=symbols[11], ls='None', markersize=8, label=r': $\theta_v= 12^ \circ$')
    theta_13 = mlines.Line2D([], [], color='grey', marker=symbols[12], ls='None', markersize=8, label=r': $\theta_v= 13^ \circ$')
    theta_14 = mlines.Line2D([], [], color='grey', marker=symbols[13], ls='None', markersize=8, label=r': $\theta_v= 14^ \circ$')
    theta_15 = mlines.Line2D([], [], color='grey', marker=symbols[14], ls='None', markersize=8, label=r': $\theta_v= 15^ \circ$')


    fontP=FontProperties()
    #fontP.set_size('small') prop=fontP,
    #ax.legend( loc='upper left', handles=[oi,forty_spikes, forty_sp,  theta_1, theta_2, theta_3, theta_4, theta_5,theta_6], numpoints=1,fontsize='10')# ti, ob, cmc_oi, ti, ob,ti_e150, ti_e150_g100, cmc_oi
    ax.legend( loc='upper left', ncol=2, handles=[ti, forty_sp,  theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8, theta_9, theta_10, theta_11, theta_12, theta_13, theta_14, theta_15], numpoints=1,fontsize='10') #cmc_oi, ti, ob,ti_e150, ti_e150_g100, cmc_oi,

    #plt.legend(loc='upper left', bbox_to_anchor=(1,0.5))
    ax.set_ylabel(r'E$_{\mathrm{pk}}$ (keV)')
    ax.set_xlabel(r'$\frac{E_\mathrm{iso}}{10^{52}}$ (erg)')
    ax.set_ylim([4,3e3])
    ax.set_xlim([4e-3, 1e2])

    #zoom_ax.plot(E_iso_sim[9:12],E_p_sim[9:12],'darkmagenta', linewidth=2, label='16TI.e150.g100')
    #zoom_ax.plot(E_iso_sim[12:15],E_p_sim[12:15],'g', linewidth=2, label='16TI.e150')
    #zoom_ax.set_ylim([58,62])
    ##plt.yticks(visible=False)
    ##plt.xticks(visible=False)
    #mark_inset(ax, zoom_ax, loc1=1, loc2=2, fc="None", ec="0.5")

    #plt.tight_layout()
    plt.show()
    if save_plot:
        fig.savefig('amati_variable_KN_CMC.pdf',bbox_inches='tight')

def amati(value='o'):
    """
    Return the Amati relationship and it's 1 sigma dispersion as given by Tsutsui et al. (2009).

    :param value: a string that can be 'o', '+', or '-'. The default is set to 'o' for the actual Amati relationship.
        '+' gives the upper bound of uncertainty and '-' gives the lower bound of uncertainty.
    :return: returns arrays of the a and y values of the amati relation/ error in the relation
    """
    #plot the amati relation given by:
    #http://iopscience.iop.org/article/10.1088/1475-7516/2009/08/015/pdf
    x=np.linspace(-3,3,100) #log(E_iso/10**52), for caluclation of E_p, add 52 to x @ end to get back normal values

    if value=='o':
        y=(1/2.01)*(x+3.87) #y is log(E_p/1keV)
    elif value=='+':
        y=(1/(2.01))*(x+(3.87+0.33))
    elif value=='-':
        y=(1/(2.01))*(x+(3.87-0.33))
    else:
        print('This isnt a correct option for value\n')

    return x,y



def golenetskii(value='o'):
    """
    Return the golenetskii relationship and it's 2 sigma dispersion as given by Lu et al. (2012).

    :param value: a string that can be 'o', '+', or '-'. The default is set to 'o' for the actual golenetskii relationship.
        '+' gives the upper bound of uncertainty and '-' gives the lower bound of uncertainty.
    :return: returns arrays of the a and y values of the relation/ error in the relation
    """
    #plot the golenetskii relation given in:
    # Lu R.-J.,  Wei J.-J.,  Liang E.-W.,  Zhang B.-B.,  Lu H.-J.,  Lu L.-Z.,  Lei W.-H.,  Zhang B.. , ApJ , 2012, vol. 756 pg. 112
    #http://iopscience.iop.org/article/10.1088/0004-637X/756/2/112/meta

    #log Ep = −(29.854  \pm  0.178) + (0.621  ±  0.003)log L_gamma, iso

    x=np.linspace(46,54,100) #Peak L_iso

    if value=='o':
        y=-29.854 + 0.621*x
    elif value=='+':
        y=(-29.854+0.178)+(0.621+0.003)*x
    elif value=='-':
        y=(-29.854-0.178)+(0.621-0.003)*x
    else:
        print('This isnt a correct option for value\n')

    return x,y

def plot_golenetskii(sims, t_end, delta_t=1, detectable=False, save_plot=False, h5=True):
    """
    Function to plot the MCRaT results alongside the golenetskii relationship

    :param sims: An array containing strings of event file base names that will be plotted
    :param t_end: An array of times that the light curves of the above sim values end
    :param delta_t: The size of the time binds used to get the synthetic GRB light curve/time resolved spectra
    :param detectable: Switch to plot the luminosities below 10^50 ergs/s or not, set to False by default
    :param save_plot: Switch to save the plot as a PDF or not
    :param h5: Switch to specify if the MCRaT output files were HDF5 format or not, default is True. Should be set to
            False when an old version of MCRaT was used which outputted the data in text files
    :return:
    """
    import matplotlib
    plt.rcParams.update({'font.size':14})
    from mclib import lcur
    from matplotlib.font_manager import FontProperties
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots()

    sims=np.array(sims)
    t_end=np.array(t_end)

    x,y=golenetskii()
    x_p,y_p=golenetskii(value='+')
    x_m,y_m=golenetskii(value='-')

    ax.loglog(10**x, 10**y, color='grey')
    ax.loglog(10**x_p, 10**y_p,color='grey', ls='-.')
    ax.loglog(10**x_m, 10**y_m, color='grey', ls='-.')

    count=0
    for i in sims:
        #t=np.arange(0,t_end[i],1) #should change for sims that arent 105 seconds long
        #lc=np.zeros([t.size])
        #lc_e=np.zeros([t.size])

        #angle=np.float(sims[i][-1])
        lc, lcur_e, alpha,beta,e_o,err, time, model_use, P, I, Q, U, V, Perr, P_angle, num_ph, num_scatt = \
            lcur_param(i, 0, t_end[count], dt=delta_t, plotting=False, choose_best=True, hdf5=h5)

        #calc proper error propagation for e_pk
        err[:,2,0]=np.sqrt(((2+alpha)* err[:,2,0])**2 + (e_o*err[:,0,0])**2  )
        err[:,2,1]=np.sqrt(((2+alpha)* err[:,2,1])**2 + (e_o*err[:,0,1])**2  )

        if detectable:
            #just plot greater than L_iso>10^50
            id=np.where(lc>=1e50)[0]
            #id=np.where(alpha<0)[0]
            lc=lc[id]
            lcur_e=lcur_e[id]
            alpha=alpha[id]
            e_o=e_o[id]
            err=err[id,:,:]
            print(err.shape, lc.size, lcur_e.size, e_o.size, alpha.size)

        if '1spike' in i:
            c='r'
        if '40spikes' in i:
            #c='sienna'
            sym='*'
        if '40sp_down' in i:
            #c='lime'
            sym='^'
        m_size='5'

        if 'CMC_16OI' in i:
            c='k'
        if '35OB' in i:
            c='b'
        if '16TI' in i:
            c='goldenrod'
            if '.e150' in i:
                c='g'
            if '.e150.g100' in i:
                c='darkmagenta'

        if i[-1]=='1':
            #sym='*'
            c='sienna'
        elif i[-1]=='2':
            #sym='D'
            c='r'
        elif i[-1]=='3':
            #sym='^'
            c='lime'
        elif i[-1]=='4':
            #sym='+'
            c='b'
        elif i[-1]=='5':
            #sym='x'
            c='k'
        elif i[-1]=='6':
            #sym='s'
            c='goldenrod'
        elif i[-1]=='7':
            #sym='s'
            c='g'
        elif i[-1]=='8':
            #sym='s'
            c='darkmagenta'
        elif i[-1]=='9':
            #sym='s'
            c='grey'


        ax.errorbar(lc, e_o*(2+alpha), xerr=lcur_e, yerr=np.abs(err[:,2,:].T), color=c,marker=sym, markersize=m_size, ls='None', label=i)

        count+=1


    ax.set_xlabel('Time Interval Luminosity (erg/s)')
    ax.set_ylabel('Time Interval '+r'E$_{\mathrm{pk}}$ (keV)')
    one_spike=mlines.Line2D([],[], color='r', marker='^', ls='None', label='1spike')
    forty_spike=mlines.Line2D([],[], color='grey', marker='^', ls='None', label='40spikes')
    forty_sp_down=mlines.Line2D([],[], color='grey', marker='^', ls='None', label='40sp_down')
    cmc_oi=mlines.Line2D([],[], color='k', ls='-', label='CMC_16OI')
    ti=mlines.Line2D([],[], color='goldenrod', ls='-', label='16TI')
    ob=mlines.Line2D([],[], color='b', ls='-', label='35OB')
    ti_e150=mlines.Line2D([],[], color='g', ls='-', label='16TI.e150')
    ti_e150_g100=mlines.Line2D([],[], color='darkmagenta', ls='-', label='16TI.e150.g100')
    theta_1=mlines.Line2D([],[],color='sienna', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 1^ \circ$')
    theta_2=mlines.Line2D([],[],color='r', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 2^ \circ$')
    theta_3=mlines.Line2D([],[],color='lime', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 3^ \circ$')
    theta_4=mlines.Line2D([],[],color='b', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 4^ \circ$')
    theta_5=mlines.Line2D([],[],color='k', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 5^ \circ$')
    theta_6=mlines.Line2D([],[],color='goldenrod', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 6^ \circ$')
    theta_7=mlines.Line2D([],[],color='g', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 7^ \circ$')
    theta_8=mlines.Line2D([],[],color='darkmagenta', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 8^ \circ$')
    theta_9=mlines.Line2D([],[],color='grey', marker=sym, ls='None', markersize=6, label=r': $\theta_v= 9^ \circ$')

    ax.set_xlim([8e49,2e53])
    ax.set_ylim([10,1.5e3])

    ax.legend( loc='upper left', ncol=2,  handles=[ theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7, theta_8], numpoints=1,fontsize='10') #ti, ob,ti_e150, ti_e150_g100, cmc_oi,


    plt.show()
    if save_plot:
        fig.savefig('golenetskii_40sp_down_KN_CMC.pdf',bbox_inches='tight')


    return lc, e_o, alpha


def polarizationVsAngle(event_files, time_start,time_end, dt=1, phi_angle=0, delta_theta=1, unit='erg/s', plotting=True, save_plot=False, dim=2, hdf5=True, compare_to_lund=False, p=4, thetaj=0.01, plot_lumi=False ):
    """
    Function to plot how time integrated polarization of the MCRaT simulation changes as a function of observer angle.
    There are option to compare the results to those of Lundman et al. (2014b) and to also overplot the peak luminoity
    of the light curve at the given angle.

    :param event_files: An array of the MCRaT event file base names that are numerically ordered from smallest to largest
    :param time_start: An array of the start times of the light curves for the event files specified above
    :param time_end: An array of the end times of the light curves for the event files specified above
    :param dt: If also plotting the peak of the light curve, dt specifies the width of the time bins of the light curves
    :param phi_angle: A placeholder variable for future 3D compatability
    :param delta_theta: delta theta of the observer viewieng angle for accepting photons, also in degrees.
            Should be the same as what was used in the function call to produce the event files passed to this function
    :param unit: The units of the produced light curve if plotting the light curve peak as well. The default is 'erg/s'
            but can also pass 'cts/s'
    :param plotting: Switch, set to True by default, to show the plot produced
    :param save_plot: Switch to save the plot as a PDF or not
    :param dim: Sets the number of dimensions of the MCraT simulation analyzed, should be set to 2 since there is no 3d
            compatability yet.
    :param hdf5: Switch to specify if the MCRaT output files were HDF5 format or not, default is True. Should be set to
            False when an old version of MCRaT was used which outputted the data in text files
    :param compare_to_lund: Switch to specify if the user wants to compare their results to those of
            Lundman et al. (2014b)
    :param p: If compare_to_lund is set to True, the user needs to specify the proper p value for the lorentz factor
            profile. The default is p=4, also accepts p=2.
    :param thetaj: If compare_to_lund is set to True, the user needs to specify the opening angel of the jet in radians,
            the default is set to 0.01 rad, also accepts 0.1 rad when p=4.
    :param plot_lumi: A switch to plot the peak luminosity of the light curve associated with the passed event_files
    :return: Arrays of the plotted time integrated polarizations and their errors and the polarization angles and their
            associated 1 sigma errors
    """

    def gamma(angle_ratio, p, gamma0=100):
        return gamma0/np.sqrt(1+angle_ratio**(2*p))

    from mclib import lcur

    if compare_to_lund:
        plot_lumi=False

    avg_pol=np.zeros(np.size(event_files))
    avg_pol_angle = np.zeros(np.size(event_files))
    std_pol=np.zeros((np.size(event_files),2))
    angles=np.zeros(np.size(event_files))

    if compare_to_lund or plot_lumi:
        lumi = np.zeros(np.size(event_files))
        lumi_e = np.zeros(np.size(event_files))

    for i in range(np.size(event_files)):
        if i>9:
            angles[i] = np.int(event_files[i][-2:])
        else:
            angles[i] = np.int(event_files[i][-1])

        angles[i] = np.double(event_files[i].split('_')[-1])#*np.pi/180
        if compare_to_lund:
            angles[i]=np.deg2rad(angles[i])
            angles[i] /= thetaj #for POL test
            angles[i] *=thetaj*180/np.pi
        if compare_to_lund or plot_lumi:
            lc,lcur_e, num_ph, avg_e_ph, P, I, Q, U, V, Perr, P_angle, num_scatt=lcur(event_files[i],np.array([time_start[i], time_end[i]]),units=unit, theta=angles[i], dtheta=delta_theta, sim_dims=dim, h5=hdf5)
            print('printing info:', ' angle: ', angles[i], ' Polarization: ',P, ' Polarization Error: ', Perr, ' Polarization Angle: ', P_angle, ' Q: ', Q, ' U: ', U, ' num_ph: ', num_ph)
            avg_pol[i]=P[0] #np.average(P, weights=lc)
            avg_pol_angle[i] = P_angle[0]
            std_pol[i,:] = Perr[0,:] #np.sqrt(np.average((P-avg_pol[i])**2, weights=lc))
            #print(Perr.mean())
            if compare_to_lund :
                lumi[i]=np.average(lc[lc>0])
                lumi_e[i]=np.average(lcur_e[lc>0])
                avg_pol_angle[i] = np.nan

        if plot_lumi:
            t = np.arange(0, time_end[i], dt)  # should change for sims that arent 105 seconds long
            lc = np.zeros([t.size])
            lc_e = np.zeros([t.size])

            lc[:], lc_e[:], num_ph, dum, P, l, q, u, v, perr, p_angle = lcur(event_files[i], np.arange(0, time_end[i], dt), units=unit, theta=angles[i], dtheta=delta_theta, sim_dims=dim, iso_lumi=riken_switch, h5=hdf5)

            lumi[i] = lc.max() #np.average(lc[lc > 0])
            lumi_e[i] = lc_e[lc.argmax()]  #np.average(lcur_e[lc > 0])

    if compare_to_lund:
        angles[i] /= thetaj  # for POL test

        gammas=gamma(angles, p)
        one_over_gamma_error_pos=angles+(gammas**-1)/thetaj
        one_over_gamma_error_neg = angles - (gammas ** -1) / thetaj

    if plotting==True:
        if not plot_lumi:
            f, axarr = plt.subplots(1, sharex=True)
            line1 =axarr.errorbar(angles[:-1], avg_pol[:-1]*100, yerr=std_pol[:-1,0]*100, marker='o', label='Polarization Degree', color='k') #xerr= (gammas ** -1) / thetaj
            #new_ax_2 = axarr.twinx()
            #line2 =new_ax_2.errorbar(angles, avg_pol_angle, yerr=std_pol[:,1], marker='o', color='darkmagenta', label='Polarization Angle')
            #new_ax_2.set_ylabel(r'$\chi  (^\circ)$', color='darkmagenta',  fontsize=14)

            #new_ax_2.set_ylim([-90, 90])
            axarr.set_xlabel(r'$\theta_{\mathrm{v}} (^\circ)$', fontsize=14)
            axarr.set_ylabel(r'$\Pi$ (%)', fontsize=14)

            # added these three lines
            #lns = line1 + line2
            #labs = [l.get_label() for l in lns]
            #axarr.legend(lns, labs, loc='best', fontsize=14)
            #axarr.legend(loc='best', fontsize=14)
            #new_ax_2.legend(loc='best', fontsize=14)

            if compare_to_lund:
                axarr.set_xlabel(r'$\theta_{\mathrm{v}}/\theta_{\mathrm{j}}$',  fontsize=14)
                all_data=np.abs(get_lundman_pol_data(p, thetaj=thetaj))
                axarr.plot(all_data[:,0], all_data[:,1]*100, 'k--', label='Lundman Polarization')

                #new_ax_2.errorbar(angles, lumi/lumi.max(), yerr=lumi_e/lumi.max(), marker='o', color='r',
                #                  label='Luminosity')

                #new_ax_2.plot(all_data[:,2], all_data[:,3], 'r--', label='Lundman Luminosity')
                #new_ax_2.set_ylabel(r'L/L($\theta=0$)', color='r', fontsize=14)
                #new_ax_2.set_ylim([0, 1])

                if p==4:
                    axarr.set_xlim([0,1.8])
        else:
            f, axarr = plt.subplots()
            f.subplots_adjust(right=0.75)
            new_ax_2 = axarr.twinx()
            new_ax_3 = axarr.twinx()

            def make_patch_spines_invisible(ax):
                ax.set_frame_on(True)
                ax.patch.set_visible(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)

            # Offset the right spine of par2.  The ticks and label have already been
            # placed on the right by twinx above.
            new_ax_3.spines["right"].set_position(("axes", 1.2))
            # Having been created by twinx, par2 has its frame off, so the line of its
            # detached spine is invisible.  First, activate the frame but make the patch
            # and spines invisible.
            make_patch_spines_invisible(new_ax_3)
            # Second, show the right spine.
            new_ax_3.spines["right"].set_visible(True)


            line1 = axarr.errorbar(angles, avg_pol * 100, yerr=std_pol[:, 0] * 100, marker='o',
                                   label='Polarization Degree')  # xerr= (gammas ** -1) / thetaj



            line2 = new_ax_2.errorbar(angles, avg_pol_angle, yerr=std_pol[:, 1], marker='o', color='darkmagenta',
                                      label='Polarization Angle')

            line3 =new_ax_3.errorbar(angles, lumi , yerr=lumi_e , marker='o', color='k', label='Luminosity')

            # new_ax_2.set_ylabel(r'$\chi$ ($^\circ$)', color='darkmagenta')
            new_ax_2.set_ylabel(r'$\chi  (^\circ)$', color='darkmagenta', fontsize=14)

            new_ax_2.set_ylim([-90, 90])
            # axarr.set_xlabel(r'$\theta_{\mathrm{v}}$')
            axarr.set_xlabel(r'$\theta_{\mathrm{v}} (^\circ)$', fontsize=14)
            # axarr.set_ylabel(r'$\Pi (\%)$')
            axarr.set_ylabel(r'$\Pi$ (%)', fontsize=14)


            new_ax_3.set_ylabel(r'L$_\mathrm{pk}$ (erg/s)', fontsize=14)
            new_ax_3.set_yscale('log')

            axarr.yaxis.label.set_color(line1[0].get_color())
            new_ax_2.yaxis.label.set_color(line2[0].get_color())
            new_ax_3.yaxis.label.set_color(line3[0].get_color())

            tkw = dict(size=4, width=1.5)
            axarr.tick_params(axis='y', colors=line1[0].get_color(), **tkw)
            new_ax_2.tick_params(axis='y', colors=line2[0].get_color(), **tkw)
            new_ax_3.tick_params(axis='y', colors=line3[0].get_color(), **tkw)

            axarr.annotate('(b) 40sp_down',xy=(0, 0), xycoords='axes fraction', fontsize=14, xytext=(10, 20), textcoords='offset points', ha='left', va='top')



        if save_plot==True:
            if not compare_to_lund:
                if not plot_lumi:
                    plt.savefig('EVENT_FILE_ANALYSIS_PLOTS/%s_pol_vs_angle.pdf'%(event_files[0][:-2]),bbox_inches='tight')
                else:
                    plt.savefig('EVENT_FILE_ANALYSIS_PLOTS/%s_pol_vs_angle_vs_lumi.pdf'%(event_files[0][:-2]),bbox_inches='tight')
            else:
                plt.savefig('EVENT_FILE_ANALYSIS_PLOTS/comp_lundman_p_%d_thetaj_%0.1e.pdf' % (p, thetaj),
                            bbox_inches='tight')

    return avg_pol, std_pol, avg_pol_angle, angles

def get_lundman_pol_data(p=4, thetaj=0.01):
    """
    This function gets the appropriate values of polarization vs theta for various values of p and theta_j from the
    plots of Lundman et al. (2014b). The only accepted parameter combinations are p=4 with thetaj=0.01 or 0.1 and
    p=2 with thetaj=0.01

    :param p: The p value for the width of the analytic jet from Lundman et al. (2014b)
    :param thetaj: The opening angle of the analytic jet profile from Lundman et al. (2014b)
    :return: Returns a (nx2) array where n is the number of angles points read in fro the file and the columns are the
        theta/thetaj values followed by the polarization degree
    """
    #p parameter says which p result to get, p=4 or p=2
    if p==4:
        if thetaj==0.01:
            all_data=np.genfromtxt('Data_files/Dataset_lundman_1.csv', delimiter=',', usecols=(0,1,3,4))
        if thetaj==0.1:
            all_data=np.genfromtxt('Data_files/lundman_p_4_thetaj_0.1.csv', delimiter=',', usecols=(0,1,3,4))
    elif p==2:
        all_data=np.genfromtxt('Data_files/Dataset_lundman_2.csv', delimiter=',', usecols=(0,1,3,4))

    return all_data


def fluidGammaVsTheta(fluid_dir, r_interest, t_interest, fps, theta_max):
    """
    Function to calculate the lorentz factor profile as a function of theta directly from the hydrodynamic files.
    It calculates which frame to analyze based off of what radius the user is interested in and the times in the outflow
    the user is interested in

    :param fluid_dir: String that points to the directory of the hydrodynamic files, should include the base name of the
        hydro file (excluding any numbers)
    :param r_interest: The radius at which the user wants the flow to be analyzed
    :param t_interest: An array of times in the ouflow that the user wants to be analyzed
    :param fps: The frames per second of the passed hydrodynamic simulation
    :param theta_max: The maximum angle to get the lorentz factor profile
    :return: All the theta data points, in degrees, the average lorentz profile at the angle data points, and the array
        of the times that the user was interested in
    """
    from scipy import stats

    c_light = 2.99792458e10
    frames=np.round(fps*(np.array(t_interest)+(r_interest)/c_light))
    c_dt=c_light/fps

    count=0
    all_data_theta=[]
    all_data_gamma=[]
    for frame in frames:
        sfrm=np.str(np.int(frame))
        if frame < 1000: sfrm = '0' + np.str(np.int(frame))
        if frame < 100: sfrm = '00' + np.str(np.int(frame))
        if frame < 10: sfrm = '000' + np.str(np.int(frame))

        xx, yy, szxx, szyy, vx, vy, gg, dd, dd_lab, rr, tt, pp=read_flash(fluid_dir+sfrm)

        idx=np.where((rr<r_interest+c_dt) &  (rr>r_interest-c_dt) & (tt<np.deg2rad(theta_max)))

        bin_means, bin_edges, binnumber = stats.binned_statistic(tt[idx], gg[idx], statistic='mean', bins=100) #bins the data and finds the mean gamma in each angle bin

        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width / 2

        idx = np.where(np.isnan(bin_means))
        not_idx = np.where(~np.isnan(bin_means))

        bin_means[idx]=np.interp(bin_centers[idx], bin_centers[not_idx], bin_means[not_idx])

        all_data_theta.append(np.rad2deg(bin_centers))
        all_data_gamma.append(bin_means)


    return all_data_theta, all_data_gamma, t_interest

def lundmanPolarizationGamma(p, theta_j, gamma_0, max_angle):
    """
    Function that calculates Lundman et al.'s (2014b) lorentz factor profile as a function of theta

    :param p: The p value for the width of the analytic jet from Lundman et al. (2014b)
    :param theta_j: The opening angle of the analytic jet profile from Lundman et al. (2014b)
    :param gamma_0: The jet core lorentz factor
    :param max_angle: The maximum angle to calculate the lorentz factor of
    :return: Returns arrays of the angles and the lorentz factors at those angles
    """
    theta=np.linspace(0, max_angle)

    gamma=gamma_0/np.sqrt((np.deg2rad(theta)/theta_j)**(2*p)+1 )

    return theta, gamma


def plotfluidGammaVsTheta(derivative=False, comp_lundman=False, saveplot=False):
    """
    This function plots the lorentz factor profile of hydrodynamic outflows, as a function of theta, alongside
    Lundman et al.'s (2014b) lorentz factor profile

    :param derivative: Switch to plot the derivative of the lorentz factor with respect to theta
    :param comp_lundman: Switch to compare the hydro simulation to the lorentz factor profile from Lundman et al. (2014b)
    :param saveplot: Switch to save the plot as a PDF or not
    :return:
    """
    tt_16ti, gg_16ti, times_16ti = fluidGammaVsTheta(
        '/Volumes/DATA6TB/Collapsars/2D/HUGE_BOXES/CONSTANT/16TI/rhd_jet_big_13_hdf5_plt_cnt_', 15e12, [30, 60, 90], 5,
        17)

    tt_40sp_down, gg_40sp_down, times_40sp_down = fluidGammaVsTheta(
        '/Volumes/DATA6TB/Collapsars/2D/HUGE_BOXES/VARY/40sp_down/m0_rhop0.1big_hdf5_plt_cnt_', 2.5e12, [13, 26, 39],
        10, 11)

    fig, ax=plt.subplots(1, sharex=True)
    if derivative:
        fig_2, ax_2=plt.subplots(1)

    for i in range(np.size(times_16ti)):
        ax.semilogy((tt_16ti[i]), gg_16ti[i], label='16TI: %ds'%(times_16ti[i]))
        if derivative:
            ax_2.semilogy((tt_16ti[i]), np.abs(np.gradient(gg_16ti[i], tt_16ti[i])), label='16TI: %ds'%(times_16ti[i]))

    for i in range(np.size(times_40sp_down)):
        ax.semilogy((tt_40sp_down[i]), gg_40sp_down[i], '-.', label='40sp_down: %ds' % (times_40sp_down[i]))
        if derivative:
            ax_2.semilogy((tt_40sp_down[i]), np.abs(np.gradient(gg_40sp_down[i], tt_40sp_down[i])),'-.', label='16TI: %ds' % (times_40sp_down[i]))

    if comp_lundman:
        theta, gamma=lundmanPolarizationGamma(4, 10/100, 100, 17) #parameters from thier figure 5

        ax.plot(theta, gamma, 'k--', label='Lundman wide jet')

        ax.set_ylim([2, 400])

        if derivative:
            ax_2.plot(theta, np.abs(np.gradient(gamma, theta)), 'k-', label='Lundman wide jet')




    ax.legend()
    ax.set_ylabel(r'$\Gamma$')
    ax.set_xlabel(r'$\theta (^\circ)$')

    if derivative:
        ax_2.set_ylabel(r'$\left| \frac{d\Gamma}{d\theta} \right|$')
        ax_2.set_ylim([1e-4, 2.5e2])
        ax_2.legend()

    plt.show()

    if saveplot:
        fig.savefig('EVENT_FILE_ANALYSIS_PLOTS/gamma_vs_theta.pdf')
        if derivative:
            fig_2.savefig('EVENT_FILE_ANALYSIS_PLOTS/dgamma_vs_dtheta.pdf')

    return
