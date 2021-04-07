"""
This routine allows the user to make movies of the photons propagating through the hydro simulation.
This routine can also plot the average temperature of the photons vs the average temperature of the matter near the photons

currently this code only works with FLASH hydrodynamic simulations

1st: read FLASH File and MCRaT file

2nd: plot flash temperature and mcrat positions of a bunch of photons

3rd: plot temp vs radius for photons and matter

"""
import mclib as m
import matplotlib.animation as animation
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import  pylab as pl # was: from pylab imprt *
import pickle
import scipy as s
from scipy import interpolate
import tables as t
import read_process_files as rp
import random
import sys
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_flash_data(file_num, f, max_x, min_y, max_y, flash_l_scale):
    """
    This function collects all the data from a given FLASH file and prepares it to be plotted with imshow.

    :param file_num: The frame number
    :param f: The path to the file as well as the FLASH file name preceeding the file number
    :param max_x: The maximum x that should be plotted (the input should be in physical units, not in code units)
    :param min_y: The minimum y of the portion of the hydro frame that will be plotted (the input should be in code units)
    :param max_y: The maximum y of the portion of the hydro frame that will be plotted (the input should be in code units)
    :return: returns a 2D array of the density for plotting with imshow
    """
    

    sfrm=np.str(file_num)
    if file_num < 1000: sfrm = '0' + np.str(file_num)
    if file_num < 100: sfrm = '00' + np.str(file_num)
    if file_num < 10: sfrm = '000' + np.str(file_num)

    xx, yy, szxx, szyy, vx, vy, gg, dd, dd_lab, rr, tt, pp = rp.read_flash(f + sfrm, make1D=True)
    xx=xx/flash_l_scale
    yy=yy/flash_l_scale

    print ('>> creating image...')
    idx = np.where((xx  > 0) & (xx  < max_x) & (yy  > min_y) & (yy  < max_y))[0]
    x = np.linspace((xx[idx].min()), (xx[idx].max()), num=1000)
    y = np.linspace((yy[idx].min()), (yy[idx].max()), num=1000)

    points = np.empty([idx.size, 2])
    points[:, 0] = xx[idx]
    points[:, 1] = yy[idx]

    X, Y = np.meshgrid(x, y)
    Z = interpolate.griddata(points, dd[idx], (X, Y), method='nearest', rescale=True)

    LTima = (np.log10(Z))

    # employ axis symmetry
    data = np.zeros([LTima.shape[0], 2 * LTima.shape[1]])
    data[:, :LTima.shape[1]] = np.fliplr(LTima)
    data[:, LTima.shape[1]:] = LTima

    return data

def get_indexes_data(mcrat_f, num, ph_num, angle, t, dt, fps, read_comv=False, read_stokes=False, read_type=False, dtheta_deg=1, energy_range=None):
    """
    This function gets the set of indexes of the photons that will be plotted in the imshow window as well as the
    indexes of all the photons in order to calculate the spectrum

    :param mcrat_f:
    :param num:
    :param ph_num:
    :param angle:
    :param t:
    :param dt:
    :param fps:
    :param read_comv:
    :param read_stokes:
    :param dtheta_deg:
    :return:
    """


    #get frame data, want times that photons would be detected without considering detector position
    times, energy, weight, mcrat_indexes, S0, S1, S2, S3, comv_energy, num_scatt, photon_type = m.event_h5(0, angle, dtheta_deg, mcrat_f.replace('mcdata_', '') , num, '', fps=fps,
                                                read_comv=read_comv, read_stokes=read_stokes, read_type=read_type, save_event_file=False)

    #full_indexes = np.zeros(times.size, dtype=np.bool)
    #indexes = np.zeros(times.size, dtype=np.bool)

    #choose the photons that would be detected during a certain time
    if energy_range is None:
        idx = np.where((times >= (t)) & (times < (t+dt)))[0]
    else:
        idx = np.where((times >= (t)) & (times < (t + dt)) & (energy>=energy_range[0]) & (energy<=energy_range[1]) )[0]

    if ph_num>np.size(mcrat_indexes[idx]):
        choose_ph_num=np.size(mcrat_indexes[idx])
        rnd_index=range(0, np.size(mcrat_indexes[idx]))
    else:
        choose_ph_num=ph_num
        #choose ph_num indexes
        rnd_index = random.sample(range(0, np.size(mcrat_indexes[idx])), choose_ph_num)

    full_indexes = idx #to calculate spectrum
    indexes = mcrat_indexes[idx][rnd_index]  #to plot a random set of photons on imshow plot, have to do something different here to keep photons plotted the same across frames

    data = rp.read_mcrat_h5(mcrat_f+np.str(num), read_comv=read_comv, read_stokes=read_stokes, read_type=read_type)
    if read_comv and read_stokes:
        if read_type:
            PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3, PT = data
        else:
            PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3 = data
            PT = np.zeros((np.size(P0))) * np.nan
    elif read_comv and not read_stokes:
        if read_type:
            PW, NS, P0, P1, P2, P3, R1, R2, R3, COMV_P0, COMV_P1, COMV_P2, COMV_P3, PT = data
        else:
            PW, NS, P0, P1, P2, P3, R1, R2, R3, COMV_P0, COMV_P1, COMV_P2, COMV_P3 = data
            PT = np.zeros((np.size(P0))) * np.nan
        S0, S1, S2, S3 = np.zeros((4, np.size(P0))) * np.nan
    elif not read_comv and read_stokes:
        if read_type:
            PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3, PT = data
        else:
            PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3 = data
            PT = np.zeros((np.size(P0))) * np.nan
        COMV_P0, COMV_P1, COMV_P2, COMV_P3 = np.zeros((4, np.size(P0))) * np.nan
    else:
        if read_type:
            PW, NS, P0, P1, P2, P3, R1, R2, R3, PT = data
        else:
            PW, NS, P0, P1, P2, P3, R1, R2, R3 = data
            PT = np.zeros((np.size(P0))) * np.nan
        S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3 = np.zeros((8, np.size(P0))) * np.nan

    return full_indexes, indexes, times[full_indexes], energy[full_indexes], weight[full_indexes], comv_energy[full_indexes], R1[indexes], R2[indexes], R3[indexes], photon_type[full_indexes]

def calc_spectrum(p0,weight):
    """
    produces spectrum in keV

    :param p0:
    :param weight:
    :return:
    """

    hnu=p0

    dnulog=.1
    numin=10**np.arange(-7,5,dnulog)
    numax=numin*10**dnulog
    nucen=np.sqrt(numax*numin)

    dnu=numax-numin
    sp=np.zeros(numin.size)
    spe=np.zeros(numin.size)
    goodones=np.zeros(numin.size)

    for i in range(numin.size):
        jj=np.where((hnu>=numin[i])&(hnu<numax[i]))
        if jj[0].size>0:

            sp[i]=np.sum(weight[jj])/dnu[i] #np.sum(weight[jj]*hnu[jj])/dnu[i]

            #spe[i]=sp[i]/np.sqrt(jj[0].size)#/dnu[i]
            if jj[0].size>0: goodones[i]=1
            #print(jj, jj[0].size, weight[jj])

    kk=np.where(goodones>0)
    sp=sp[kk]
    nucen=nucen[kk]

    return sp, nucen

def get_plot_data(frame_number, flash_file, mcrat_file, t_ph=None, t_f=None, plot_temp_curves=False, read_comv=False, read_stokes=False, read_type=False, dtheta_deg=1):
    """

    :param frame_number:
    :param flash_file:
    :param mcrat_file:
    :param t_ph:
    :param t_f:
    :param plot_temp_curves:
    :param read_comv:
    :param read_stokes:
    :param read_type:
    :param dtheta_deg:
    :return:
    """
    global maxx, maxy, miny, sim_edge, last_frame, im, pts, ph_t, flash_t, spex, flash_l_scale, num_photon, view_percent, photon_start_frame, time_detect_init, dt_detect, fps, theta

    min_y=miny
    max_y=maxy
    max_x=maxx

    print('On frame:', frame_number)

    view_percent=0.10
    delta_y=1000

    if frame_number>=photon_start_frame:

        full_idx, plot_idx, times, energy, weight, comv_energy, x, y, z, photon_types = get_indexes_data(mcrat_file, frame_number, num_photon, theta, time_detect_init, dt_detect, fps, read_comv=read_comv, read_stokes=read_stokes, read_type=read_type,dtheta_deg=dtheta_deg)
        pts.set_data(x,z)


        #if the photons are outside of the field of view of plot, modify min and max y
        view_percent=0.10 #10 percent view on both sides
        if ((z.size>0)  and (z.max()>= (0.7)*maxy*flash_l_scale) and ((max_y+6)*flash_l_scale <= (sim_edge)*flash_l_scale)):
        #and (z.min()>1.2*miny*1e9)
        #if ((z.size>0) and (maxy<(1+view_percent)*(z.mean()/1e9)) ):
            print('Printing Here',z.size, z.max(), (0.7)*maxy*flash_l_scale, (max_y+6)*flash_l_scale, (sim_edge)*flash_l_scale, z.min(), 1.2*miny*flash_l_scale)
            avg_z=z.mean()/flash_l_scale

            maxy+= 3e10/fps/flash_l_scale #(z.max()/0.8) #3#(1+view_percent)*avg_z #was +3
            miny= maxy-delta_y  #+=3#(1-view_percent)*avg_z

            if ((max_y+3)>=sim_edge):
                maxy=sim_edge
                miny=sim_edge-0.5e3 #*(1-view_percent)/(1+view_percent)


            max_y = maxy
            min_y = miny

        sp, nucen = calc_spectrum(energy, weight)
        spex.set_data(nucen, sp)

    else:
        if (((max_y + 3) * flash_l_scale <= (sim_edge) * flash_l_scale)):
            # and (z.min()>1.2*miny*1e9)
            # if ((z.size>0) and (maxy<(1+view_percent)*(z.mean()/1e9)) ):
            print('Printing Else', (0.8) * maxy * flash_l_scale, (max_y + 4) * flash_l_scale, (sim_edge) * flash_l_scale,
                  1.2 * miny * flash_l_scale)

            maxy +=3 #(1+view_percent)*avg_z #was +3
            miny += 3#(1-view_percent)*avg_z

            if ((max_y + 3) >= sim_edge):
                view_percent = 0.10
                maxy = sim_edge
                miny = sim_edge * (1 - view_percent) / (1 + view_percent)

            max_y = maxy
            min_y = miny

    """ stuff for plotting photon/matter temperature which I dont want to consider right now
    print(number, (last_frame-r.size), (last_frame-r.size)-number)
    if number > (last_frame-r.size):
        if plot_temp_curves:
            ph_t.set_data(r[((last_frame-r.size)-number):],t_ph[((last_frame-r.size)-number):])
            flash_t.set_data(r[((last_frame-r.size)-number):],t_f[((last_frame-r.size)-number):])
        else:
            ph_t=np.nan
            flash_t=np.nan

        if ~np.isnan(t_ph[((last_frame-r.size)-number)]):
            #calculate the spectrum
            sp, nucen=calc_spectrum(p0,pw_full_indexes)
            spex.set_data(nucen, sp)
            #ax3.set_xlim([nucen.min()-(0.05*nucen.min()),nucen.max()+(0.05*nucen.max())])
            #ax3.set_ylim([sp.min()-(0.05*sp.min()),sp.max()+(0.05*sp.max())])
    """


    #ph_t.set_data(r,t_ph)
    #flash_t.set_data(r,t_f)

    flash_data = get_flash_data(frame_number, flash_file, max_x, min_y, max_y, flash_l_scale)
    im.set_data(flash_data)
    im.set_clim(flash_data.min(), flash_data.max())
    im.set_extent([-max_x * flash_l_scale, max_x * flash_l_scale, min_y*flash_l_scale, max_y * flash_l_scale])
    ax.set_xlim([-max_x * flash_l_scale, max_x * flash_l_scale])
    ax.set_ylim([min_y*flash_l_scale, (max_y-5) * flash_l_scale])

    return im, pts, ph_t, flash_t, spex


if __name__ == "__main__":
    plt.rcParams['animation.ffmpeg_path'] = '/Users/parsotat/anaconda3/bin/ffmpeg'
    #main parameters
    #flash_file = '/Volumes/LACIE_RAID/Collapsars/2D/HUGE_BOXES/VARY/40spikes/m0_rhop0.1big_hdf5_plt_cnt_'
    #mcrat_file = '/Volumes/DATA6TB/Collapsars/2D/HUGE_BOXES/VARY/40spikes/CMC_40spikes/0.0-2.0/mcdata_'
    #pickle_file = '/Users/parsotat/Box Sync/PY_FILES/40spikes_1350_620_dt_0_2_0-2deg_jet_further.pickle'

    # flash_file = '/Users/Tylerparsotan/Documents/16OI_TEST/rhd_jet_big_16OI_hdf5_plt_cnt_'
    # mcrat_file = '/Users/Tylerparsotan/Documents/16OI_TEST/SKN_16OI_SPHERICAL/0.0-3.0/mcdata_'

    flash_file = '/Volumes/LACIE_RAID/Collapsars/2D/HUGE_BOXES/CONSTANT/16TI/rhd_jet_big_13_hdf5_plt_cnt_'
    mcrat_file = '/Volumes/LACIE_RAID/Collapsars/2D/HUGE_BOXES/CONSTANT/16TI/SKN_16TI/ALL_DATA/mcdata_'


    theta = 1
    dtheta=1
    time_detect_init = 45  # 180 #(36.0)s
    dt_detect = 5  # 0.2
    last_frame =3092
    r_obs = 0
    fps = 5
    f_num_start = 60 #292 #60 #250 #883
    f_num_end = last_frame
    photon_start_frame = 65  # 884 # set to correspond to when the photons of interest are injected
    num_photon = 100
    maxx = 5000  # 600 #5000  # 5e3
    miny = 6000 #0.00e3  # 1.44e3
    maxy = 10000 #1.00e3  # 2.06e3 #1000 #1200 #1200 #15000  # 25e3
    sim_edge = 25600
    dpi = 300
    flash_l_scale=1e9

    """
    for loading photon/matter temperature
    if sys.version_info < (3,0):
        pickle_f=open(pickle_file, 'rb')
        Temp_photon, Temp_flash, Avg_R, ph_num= pickle.load(pickle_f)
    else:
        with open(pickle_file, 'rb') as pickle_f:
            u=pickle._Unpickler(pickle_f)
            u.encoding='latin1'
            Temp_photon, Temp_flash, Avg_R, ph_num,scatt_num, avg_gamma, avg_pres, avg_dens=u.load()
    
    Avg_R=Avg_R[time,theta,:]
    Temp_photon=Temp_photon[time,theta,:]
    Temp_flash=Temp_flash[time,theta,:]
    """


    plot_temp_curves = False

    #setup figure for plotting
    fig = plt.figure(figsize=(12, 10))  # figsize=(6,5), dpi=1000
    gs = gridspec.GridSpec(11, 4)
    gs.update(wspace=1.05)
    ax = plt.subplot(gs[:, 0:2])

    if plot_temp_curves:
        ax2 = plt.subplot(gs[6:9, 2:])
        ax3 = plt.subplot(gs[2:5, 2:])
    else:
        ax3 = plt.subplot(gs[4:7, 2:])

    #plot hydro sim
    im = ax.imshow(get_flash_data(f_num_end, flash_file, maxx, miny, maxy, flash_l_scale), origin='lower left',
                   extent=[-maxx * flash_l_scale, maxx * flash_l_scale, miny * flash_l_scale, maxy * flash_l_scale], animated=True,
                   cmap=plt.get_cmap('BuPu'))  # , cmap=plt.get_cmap('gist_heat_r')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.00)
    cbar = plt.colorbar(im, cax=cax, ticklocation='right')
    cbar.set_label(r'Log($\rho$)', size=10)

    #get mcrat data
    full_indexes, indexes, time, energy, weight, comv_energy, x, y, z, photon_type=\
        get_indexes_data(mcrat_file, f_num_end, num_photon, theta, time_detect_init, dt_detect, fps, dtheta_deg=dtheta)
    print(full_indexes.size)

    #plot mcrat photons
    #pts, = ax.plot([], [], 'red', marker='.', ls='None', markersize='1')
    pts, = ax.plot(x, z, 'red', marker='o', ls='None')

    #plot mcrat photon spectrum
    sp, nucen = calc_spectrum(energy, weight)
    spex, = ax3.plot(nucen, sp, 'red', ls='-', marker='o')
    spex.set_data([], [])

    matplotlib.rcParams['axes.autolimit_mode'] = 'round_numbers'

    if plot_temp_curves:
        ph_t, = ax2.plot([], [], 'red', marker='.', ls='None', markersize='12')
        flash_t, = ax2.plot([], [], 'blue', ls='-', markersize='12')
        # ph_t, =ax2.plot(Avg_R,Temp_photon,'red', marker='.', ls='None', markersize='12')
        # flash_t, =ax2.plot(Avg_R,Temp_flash,'blue', ls='-', markersize='12')
    else:
        ph_t = np.nan
        flash_t = np.nan

    #format axis
    ax.set_xlim([-maxx * 1e9, maxx * 1e9])
    ax.set_ylim([miny * 1e9, (maxy - 100) * 1e9])

    ax.ticklabel_format(style='scientific', useMathText=True)

    if plot_temp_curves:
        ax2.set_xlim([Avg_R[~np.isnan(Avg_R)].min() - (0.1 * Avg_R[~np.isnan(Avg_R)].min()),
                      Avg_R[~np.isnan(Avg_R)].max() + (0.1 * Avg_R[~np.isnan(Avg_R)].max())])
        max = np.maximum(Temp_photon[~np.isnan(Temp_photon)], Temp_flash[~np.isnan(Temp_flash)]).max()
        min = np.minimum(Temp_photon[~np.isnan(Temp_photon)], Temp_flash[~np.isnan(Temp_flash)]).min()
        ax2.set_ylim([min - (0.1 * min), max + (0.1 * max)])
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('R (cm)')
        ax2.set_ylabel('T (K)')

    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_ylim([1e46, 1e56])
    ax3.set_xlim([1e-4, 1e5])
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')

    ax3.set_xlabel('E (keV)')
    ax3.set_ylabel(r'N(E) (photons/s/keV)')
    stop

    ani = animation.FuncAnimation(fig, get_plot_data, np.arange(f_num_start, f_num_end + 1, 1),
                                  fargs=(
                                  flash_file, mcrat_file), blit=False, repeat=False)
    writer = animation.FFMpegWriter(fps=60, bitrate=5e3, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])  # ,  '-loglevel', 'debug' extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '0'] "libx265" #'-crf','5',

    if sys.version_info < (3, 0):
        ani.save('test.mp4', writer=writer, bitrate=5e3, codec="libx264",
                 extra_args=['-pix_fmt', 'yuv420p', '-crf',
                             '5'], dpi=300)  # , extra_args=[ '-vcodec', 'libx264', '-pix_fmt', 'yuv420p','-crf', '0']
    else:
        ani.save('test_0-20.mp4', writer=writer, dpi=200)

    plt.show()


def follow_lc_data(hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, time, dt, fps, num_photon_plot=1e7, read_type=False, energy_range=None):
    """

    :param hydro_sim_dir:
    :param hydro_sim_name:
    :param last_frame:
    :param mcrat_sim_data_dir:
    :param maxx: in real units, not code units
    :param miny:
    :param maxy:
    :param theta: in degrees
    :param time:
    :param dt:
    :param fps:
    :param num_photon_plot:
    :param read_type:
    :return:
    """

    img_data = get_flash_data(last_frame, hydro_sim_dir+hydro_sim_name,
                   maxx, miny, maxy, 1)
    photon_data = get_indexes_data(hydro_sim_dir+mcrat_sim_data_dir+'mcdata_', last_frame, num_photon_plot,
                      theta, time, dt, fps, read_type=read_type, energy_range=energy_range)

    return img_data, photon_data

def convert_lc_time(last_frame, fps, time):
    return ((last_frame/fps)-time)*3e10

def plot_lc_jet(time_index, hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, time, dt, fps, num_photon_plot, energy_range, axes, lines, points, highlights, plot_ani):

    if  plot_ani:
        print('Working on times: %0.1lf - %0.1lf'%(time[time_index], time[time_index+1]))
        img_data, photon_data=follow_lc_data(hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, time[time_index], time[time_index+1]-time[time_index],
                   fps, num_photon_plot=num_photon_plot, read_type=False)
        if energy_range is not None:
            img_data_energy, photon_data_energy = follow_lc_data(hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx,
                                                   miny, maxy, theta, time[time_index],
                                                   time[time_index + 1] - time[time_index],
                                                   fps, num_photon_plot=num_photon_plot, read_type=False, energy_range=energy_range)
    else:
        img_data, photon_data=follow_lc_data(hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, time, dt,
                  fps, num_photon_plot=num_photon_plot, read_type=False)
        if energy_range is not None:
            img_data_energy, photon_data_energy = follow_lc_data(hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx,
                                                   miny, maxy, theta, time, dt,
                                                   fps, num_photon_plot=num_photon_plot, read_type=False, energy_range=energy_range)


    x_range = np.linspace(0, maxx, 100000)
    y_angle = np.tan(np.deg2rad(theta)) ** -1 * x_range
    idx = np.where((y_angle > miny) & (y_angle < maxy))

    if axes is None:
        fig, ax = plt.subplots(figsize=(10,10))
        plt.rcParams.update({'font.size': 16})
        plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
        plt.tick_params(labelsize=16)
    else:
        ax=axes[0]
    if not plot_ani:
        im = ax.imshow(img_data[:, np.int_(img_data.shape[-1] / 2):], origin='lower left', extent=[0, maxx, miny, maxy],
                       cmap=plt.get_cmap('BuPu'))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.00)
        cbar = plt.colorbar(im, cax=cax, ticklocation='right')
        cbar.set_label(r'Log($\frac{\rho}{\mathrm{g/cm}^3}$)', fontsize=16)
        ax.ticklabel_format(style='scientific', useMathText=True)
        ax.yaxis.offsetText.set_fontsize(16)
        ax.xaxis.offsetText.set_fontsize(16)
        ax.set_xlabel('x (cm)', fontsize=16)
        ax.set_ylabel('y (cm)', fontsize=16)
    if not plot_ani:
        ax.plot(x_range[idx], y_angle[idx], 'r--', label=r'$\theta_\mathrm{v}=$%d$^\circ$'%(theta))

    if plot_ani:
        t=time[time_index]
    else:
        t=time

    r=convert_lc_time(last_frame, fps, t)
    y_range_1 = -np.tan(np.deg2rad(theta)) * (x_range - r * np.sin(np.deg2rad(theta))) + r * np.cos(np.deg2rad(theta))
    if not plot_ani:
        img_dash1, =ax.plot(x_range, y_range_1, 'k--', label='t=%0.1f s'%(time))
    else:
        lines[1].set_data(x_range, y_range_1)
        lines[3].set_xdata(time[time_index])
        lines[5].set_xdata(time[time_index])


    if plot_ani:
        t=time[time_index+1]
    else:
        t=time+dt

    r=convert_lc_time(last_frame, fps, t)
    y_range_2 = -np.tan(np.deg2rad(theta)) * (x_range - r * np.sin(np.deg2rad(theta))) + r * np.cos(np.deg2rad(theta))
    if not plot_ani:
        img_dash2, =ax.plot(x_range, y_range_2, 'k--', label='t=%0.1f s'%(time+dt))
        if highlights is not None:
            for h in highlights[:-1]:
                h.set_alpha(0.15)
    else:
        lines[2].set_data(x_range, y_range_2)
        lines[4].set_xdata(time[time_index+1])
        lines[6].set_xdata(time[time_index+1])
        for h in highlights[:-1]:
            lims=h.get_xy()
            min_lim=lims[:,0].min()
            max_lim=lims[:,0].max()
            #print('before',time_index,lims, min_lim, max_lim)
            lims[lims[:,0]==max_lim,0]=time[time_index+1]
            lims[lims[:,0]==min_lim,0]=time[time_index]
            #print('after',time_index,lims)
            h.set_xy(lims)
        ax.collections.clear()
        ax.fill_between(x_range, y_range_1, y_range_2, color='k', alpha=0.15)

    photon_flash_x = np.sqrt(photon_data[6] ** 2 + photon_data[7] ** 2)
    if energy_range is not None:
        photon_flash_x_energy = np.sqrt(photon_data_energy[6] ** 2 + photon_data_energy[7] ** 2)
        #print(photon_flash_x_energy, photon_data_energy[4])

    if photon_data[4].size>0 and plot_ani:
        sizes=(1+np.log10(photon_data[4]/photon_data[4].min()))**2
    else:
        sizes=1*(1+np.log10(photon_data[4]/photon_data[4].min()))#**2 #None
    ph_1 =ax.scatter(photon_flash_x, photon_data[8], color='b', marker='o', ls='None', zorder=3, alpha=0.1, s=sizes)

    if energy_range is not None:
        if photon_data_energy[4].size>0 and plot_ani:
            if photon_data_energy[4].size>1:
                sizes=4*(1+np.log10(photon_data_energy[4]/photon_data_energy[4].min()))**2
            else:
                sizes=sizes.max()
        else:
            if 'down' in mcrat_sim_data_dir:
                scale=4
            else:
                scale=2
            sizes=scale*(1+np.log10(photon_data_energy[4]/photon_data_energy[4].min()))**2 #None
        ph_2 =ax.scatter(photon_flash_x_energy, photon_data_energy[8], color='r', ls='None', marker='P', zorder=4, alpha=0.5, s=sizes)
            
    #lgnd=ax.legend(loc='best')
    lgnd=ax.legend(loc='upper right', fontsize=12)
    if plot_ani:
        lgnd.get_texts()[1].set_text('t=%0.1f s'%(time[time_index]))
        lgnd.get_texts()[2].set_text('t=%0.1f s'%(time[time_index+1]))
    
    ax.set_xlim([0, maxx])
    ax.set_ylim([miny, maxy])
    
    if plot_ani:
        sp, nucen = calc_spectrum(photon_data[3], photon_data[4])
        lines[0].set_data(nucen, sp)
        
    if not plot_ani:
        return ph_1, ph_2, img_dash1, img_dash2

def animate_lc_jet(simid, hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, time, dt, fps, lc_tmin, lc_tmax, lc_dt,num_photon_plot=1e7, energy_range=None, lc_type='uniform'):

    #setup figure for plotting
    fig = plt.figure(figsize=(12, 10))  # figsize=(6,5), dpi=1000
    gs = gridspec.GridSpec(12, 4)
    gs.update(wspace=1.05)
    ax_img = plt.subplot(gs[:, 0:2])
    ax_spex = plt.subplot(gs[0:4, 2:])
    ax_lc_1 = plt.subplot(gs[5:8, 2:])
    ax_lc_1.set_ylabel(r'L$_\mathrm{bolo}$ (erg/s)', color='b')

    if energy_range is not None:
        ax_lc_2 = plt.subplot(gs[9:12, 2:])
        plt.setp(ax_lc_1.get_xticklabels(), visible=False)
        ax_lc_2.set_ylabel(r'L$_\mathrm{opt}$ (erg/s)', color='r')
        ax_lc_2.set_xlabel('Time Since Jet Launch (s)')
    else:
        ax_lc_1.set_xlabel('Time Since Jet Launch (s)')


    t = np.arange(lc_tmin, lc_tmax, lc_dt)
    if lc_type=='uniform':
        lc_1=m.lcur(simid, t, theta=theta, energy_range=None)[0]
    else:
        data_bolo=m.lcur_var_t(simid,lc_tmin,lc_tmax,lc_dt, theta=theta, energy_range=None)
        lc_1, t= data_bolo[0], data_bolo[-1]
    ax_lc_1.plot(t,lc_1, ds='steps-post', color='b')
    ax_lc_1.ticklabel_format(style='scientific', useMathText=True)
    
    lc_highlight_1=ax_lc_1.axvspan(time, time+dt, ymin=0, ymax=1, alpha=0.5, color='k')
    lc_highlight_1_line1=ax_lc_1.axvline(x=time, ymin=0, ymax=1, color='k', ls='--')
    lc_highlight_1_line2=ax_lc_1.axvline(x=time+dt, ymin=0, ymax=1, color='k', ls='--')

    if energy_range is not None:
        lc_2 = m.lcur(simid, t, theta=theta, energy_range=energy_range)[0]
        ax_lc_2.plot(t, lc_2, ds='steps-post', color='r')
        ax_lc_2.ticklabel_format(style='scientific', useMathText=True)

        lc_highlight_2=ax_lc_2.axvspan(time, time+dt, ymin=0, ymax=1, alpha=0.5, color='k')
        lc_highlight_2_line1=ax_lc_2.axvline(x=time, ymin=0, ymax=1, color='k', ls='--')
        lc_highlight_2_line2=ax_lc_2.axvline(x=time+dt, ymin=0, ymax=1, color='k', ls='--')

        
    #get mcrat data
    data=get_indexes_data(hydro_sim_dir+mcrat_sim_data_dir+'mcdata_', last_frame, 1e7, theta, lc_tmin, lc_tmax-lc_tmin, fps)

    #plot mcrat photon spectrum
    sp, nucen = calc_spectrum(data[3], data[4])
    spex, = ax_spex.loglog(nucen, sp, 'k', ls='-', marker='o')
    #spex.set_data([], [])
    ax_spex.set_xlabel('E (keV)')
    ax_spex.set_ylabel(r'N(E) (photons/s/keV)')
    ax_spex.axvspan(8, 40e3, ymin=0, ymax=1, alpha=0.5, facecolor='g')
    if energy_range is not None:
        ax_spex.axvspan(energy_range[0], energy_range[1], ymin=0, ymax=1, alpha=0.5, facecolor='r')
    
    r=convert_lc_time(last_frame, fps, time)
    x_range = np.linspace(0, maxx, 100000)
    y_range1 = -np.tan(np.deg2rad(theta)) * (x_range - r * np.sin(np.deg2rad(theta))) + r * np.cos(np.deg2rad(theta))
    r=convert_lc_time(last_frame, fps, time+dt)
    y_range2 = -np.tan(np.deg2rad(theta)) * (x_range - r * np.sin(np.deg2rad(theta))) + r * np.cos(np.deg2rad(theta))
    img_highlight=ax_img.fill_between(x_range, y_range1, y_range2, color='k', alpha=0.15)
    ax_img.ticklabel_format(style='scientific', useMathText=True)
    ax_img.set_xlabel('x (cm)')
    ax_img.set_ylabel('y (cm)')

    
    ph1, ph2, img_dash1, img_dash2=plot_lc_jet(0, hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, time, dt, fps, num_photon_plot, energy_range, [ax_img, ax_lc_1, ax_lc_2], [spex], [lc_highlight_1, lc_highlight_2, img_highlight], [], False)
    plt.pause(1)
    
    #for i in range(3):
    #    plot_lc_jet(i, hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, t, dt, fps, num_photon_plot, energy_range, [ax_img, ax_lc_1, ax_lc_2], [spex, img_dash1, img_dash2, lc_highlight_1_line1, lc_highlight_1_line2, lc_highlight_2_line1, lc_highlight_2_line2], [ph1, ph2], [lc_highlight_1, lc_highlight_2, img_highlight],True)
    #    plt.pause(1)

    
    #ani = animation.FuncAnimation(fig, plot_lc_jet, np.arange(t.size), fargs=(hydro_sim_dir, hydro_sim_name, last_frame, mcrat_sim_data_dir, maxx, miny, maxy, theta, t, dt, fps, num_photon_plot, energy_range, [ax_img, ax_lc_1, ax_lc_2], [spex, img_dash1, img_dash2, lc_highlight_1_line1, lc_highlight_1_line2, lc_highlight_2_line1, lc_highlight_2_line2], [ph1, ph2], [lc_highlight_1, lc_highlight_2, img_highlight],True), blit=False, repeat=False)
    #writer = animation.FFMpegWriter(fps=1, bitrate=5e3, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])  # ,  '-loglevel', 'debug' extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '0'] "libx265" #'-crf','5',
    
    if 'down' in simid:
        save_string='40sp_down_lc_theta_%d.mp4'%(theta)
    else:
        save_string='16TI_lc_theta_%d.mp4'%(theta)
    
    #ani.save(save_string, writer=writer, dpi=200)
    
    return fig, ax_img, ax_lc_1, lc_highlight_1

