"""
Python script meant to calculate various properties of
two time series such as their spearman rank coefficient
and their time lag with respect to one another

Tyler Parsotan 2021

"""
import mclib as m
from scipy import signal
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

event_file='SKN_16TI_1.00e+13_1'

time_start=0
time_end=100
dt=1

t=np.arange(time_start,time_end,dt)

#lc_type='uniform' #can also be variable
lc_type='variable'

lc_energy_range_1=None #Bolometric
lc_energy_range_2=[1.5e-3, 7.7e-3] #optical

def get_lc(event_file, time_start, time_end, dt, energy_range, lc_type, dt_min=None,liso_c=None ):
    theta = np.double(event_file.split('_')[-1])
    if 'uniform' in lc_type:
        t = np.arange(time_start, time_end, dt)
        return  m.lcur(event_file, t, theta=theta,  energy_range=energy_range)[0], t
    else:
        data=m.lcur_var_t(event_file, time_start, time_end, dt, dt_min, liso_c=liso_c, theta=theta,  energy_range=energy_range)
        return data[0], data[-1]

def find_time_lag(signal_1,signal_2,t):
    corr = signal.correlate(signal_1, signal_2)
    corr /= np.max(corr)

    lags = signal.correlation_lags(len(signal_1), len(signal_2))

    return lags[corr.argmax()], (t[signal_1.argmax()-lags[corr.argmax()]]-t[signal_1.argmax()])

def get_spearman(signal_1,signal_2):
    r_s=spearmanr(signal_1 / signal_1.max(), signal_2 / signal_2.max())
    confidence_interval=[np.tanh(np.arctanh( r_s[0])- (1.96/np.sqrt(signal_1.size-3))), \
                         np.tanh(np.arctanh( r_s[0])+(1.96/np.sqrt(signal_1.size-3)))]  #this is the 95% interval

    """
    can also verify confidence interval through stationary bootstrap e.g.
    from recombinator.block_bootstrap import circular_block_bootstrap 
    from recombinator.optimal_block_length import optimal_block_length 
    block_length=np.max(optimal_block_length(signal_1),optimal_block_length(signal_2))
    all_data=np.zeros((signal_1.size,2))
    all_data[:,0]=signal_1
    all_data[:,0]=signal_2
    B = 5000
    y_star_cb=circular_block_bootstrap(all_data, block_length=np.int(block_length), replications=B, replace=True)  
    all_spearman=np.zeros(B)
    for i in range(y_star_cb.shape[0]): all_spearman[i]=spearmanr(y_star_cb[i,:,0],y_star_cb[i,:,1])[0]
    all_spearman.mean() 
    all_spearman.std() 
    all_spearman.mean()-2*all_spearman.std(), all_spearman.mean()+2*all_spearman.std() #for 95% confidence interval
    """

    return r_s, confidence_interval


if __name__ == "__main__":
    #lc_1, t=get_lc(event_file,time_start, time_end, dt, lc_energy_range_1, lc_type)
    #lc_2, t=get_lc(event_file,time_start, time_end, dt, lc_energy_range_2, lc_type)

    #time_index_lag,delta_t_lag = find_time_lag(lc_1, lc_2, t)

    #print(time_index_lag,delta_t_lag)
    #print(get_spearman(lc_1, lc_2))


    #To look at the relationships between gamma and optical
    active_time_min=[0,0,0,0,0,0, 0, 0, 0, 0, 0, 0 , 0 , 0 , 0]
    active_time_max=[100,100,100,100,100,100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    angles= np.arange(1,16)
    lags=np.zeros(angles.size)
    spearman_values=np.zeros((3,angles.size))
    for i in range(angles.size):
        print('Working on angle',i+1)
        #event_file='SKN_16TI_subdir%d-%d_1.00e+13_%d'%(3*np.floor_divide(angles[i],3),3*np.floor_divide(angles[i],3)+3,angles[i])
        #event_file = 'SKN_40sp_down_2.50e+12_%d'%(angles[i])
        event_file = 'SKN_16TI_1.00e+13_%d' % (angles[i])
        if lc_type == 'uniform':
            data_bolo = m.lcur(event_file, t, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_1)
            data_opt = m.lcur(event_file, t, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_2)
        else:
            data_bolo = m.lcur_var_t(event_file, time_start, time_end, dt, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_1)
            t=data_bolo[-1]
            data_opt = m.lcur(event_file, t, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_2)

        lc_1= data_bolo[0]
        lc_2 = data_opt[0]
        lags[i]=find_time_lag(lc_1, lc_2, t)[-1]
        data=get_spearman(lc_1, lc_2)
        spearman_values[0,i]=data[0][0]
        spearman_values[1, i] = data[1][0]#lower 95% limit
        spearman_values[2, i] = data[1][1]#upper

    """

    #To look at the relationship between optical and the number of scatterings and bolometric and the number of scatterings
    angles= np.arange(1,4)
    spearman_values_bolo=np.zeros((3,angles.size))
    spearman_values_opt=np.zeros((3,angles.size))

    for i in range(angles.size):
        print('Working on angle',i+1)
        #event_file='SKN_40sp_down_%d-%dsubdir_2.50e+12_%d'%(3*np.floor_divide(angles[i],3),3*np.floor_divide(angles[i],3)+3,angles[i])
        #event_file='SKN_16TI_subdir%d-%d_1.00e+13_%d'%(3*np.floor_divide(angles[i],3),3*np.floor_divide(angles[i],3)+3,angles[i])

        event_file='SKN_16TI_1.00e+13_%d'%(angles[i])
        #event_file = 'SKN_40sp_down_2.50e+12_%d'%(angles[i])
        if lc_type == 'uniform':
            data_bolo = m.lcur(event_file, t, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_1)
            data_opt = m.lcur(event_file, t, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_2)
            idx=-2
        else:
            data_bolo = m.lcur_var_t(event_file, time_start, time_end, dt, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_1)
            t=data_bolo[-1]
            data_opt = m.lcur(event_file, t, theta=np.double(event_file.split('_')[-1]), energy_range=lc_energy_range_2)
            idx = -2

        data=get_spearman(data_bolo[0][~np.isnan(data_bolo[idx])], data_bolo[idx][~np.isnan(data_bolo[idx])])
        spearman_values_bolo[0,i]=data[0][0]
        spearman_values_bolo[1, i] = data[1][0]#lower 95% limit
        spearman_values_bolo[2, i] = data[1][1]#upper

        data=get_spearman(data_opt[0][~np.isnan(data_opt[idx])], data_opt[idx][~np.isnan(data_opt[idx])])
        spearman_values_opt[0,i]=data[0][0]
        spearman_values_opt[1, i] = data[1][0]#lower 95% limit
        spearman_values_opt[2, i] = data[1][1]#upper

        f, ax = plt.subplots(2, sharex=True)
        lc_1, =ax[0].plot(t, data_bolo[0],  ds='steps-post', label='LC')
        ax[1].plot(t, data_opt[0],ds='steps-post')

        ax_bolo_scatt = ax[0].twinx()
        ax_opt_scatt = ax[1].twinx()
        ns_1, =ax_bolo_scatt.semilogy(t, data_bolo[idx], ls='--', ds='steps-post', label='NS')
        ax_opt_scatt.semilogy(t, data_opt[idx], ds='steps-post',ls='--')

        ax[0].legend(handles=[lc_1, ns_1])
        ax[0].set_ylabel('Bolometric LC')
        ax_bolo_scatt.set_ylabel('Average Scatterings')
        ax[1].set_ylabel('Optical LC')
        ax_opt_scatt.set_ylabel('Average Scatterings')
        ax[1].set_xlabel('Time Since Jet Launch (s)')
        ax[0].set_title(event_file)
        #f.savefig('EVENT_FILE_ANALYSIS_PLOTS/LIGHT_CURVE_NUM_SCATT_SPEARMAN_ANALYSIS/'+event_file.split('_')[0]+'_'+event_file.split('_')[1]+'_'+event_file.split('_')[-1]+'_bolo_opt_lc_ns_var.pdf')

    plt.figure()
    bolo_fill=plt.fill_between(angles, spearman_values_bolo[1, :], spearman_values_bolo[2, :],'b', alpha=0.2)
    bolo_line=plt.plot(angles, spearman_values_bolo[0, :], 'b-')
    plt.plot(angles, np.zeros(angles.size), 'k--')
    opt_fill=plt.fill_between(angles, spearman_values_opt[1, :], spearman_values_opt[2, :],'r', alpha=0.2)
    opt_line=plt.plot(angles, spearman_values_opt[0,:],'r-')
    plt.ylabel(r'$r_s$')
    plt.xlabel(r'$\theta_\mathrm{v} (^\circ)$')
    plt.ylim([-1, 1])
    plt.legend([(bolo_fill, bolo_line[0]), (opt_fill, opt_line[0])], ['Bolometric', 'Optical'], loc='upper center', ncol=2)
    plt.title('16TI Light Curve - Number of Scattering Spearman\n0-100 s dt=variable s')
    """