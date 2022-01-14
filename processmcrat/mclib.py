"""
Basic library for processing MCRaT simulation data.
Written by Tyler Parsotan @ OregonState

"""
import numpy as np
from astropy import units as unit
from astropy import constants as const
from astropy.modeling import InputParameterError
from scipy.optimize import curve_fit



def band_function(energies, alpha, beta, break_energy, normalization, energy_unit=unit.keV):
	"""

	:param energies:
	:param alpha:
	:param beta:
	:param break_energy:
	:param normalization:
	:return:
	"""
	try:
		energies=energies.value
	except AttributeError:
		energies = energies

	try:
		break_energy=break_energy.value
	except AttributeError:
		break_energy = break_energy

	try:
		normalization=normalization.value
	except AttributeError:
		normalization = normalization


	model=np.empty(energies.size)
	kk=np.where(energies<((alpha-beta)*break_energy))
	if kk[0].size>0:
		model[kk]=energies[kk]**alpha*np.exp(-energies[kk]/break_energy)
	kk=np.where(energies>=((alpha-beta)*break_energy))
	if kk[0].size>0:
		model[kk]=((alpha-beta)*break_energy)**(alpha-beta)*energies[kk]**(beta)*np.exp(beta-alpha)
	model=model/np.trapz(model,x=energies)*normalization

	return model

def wien_function(energies, temp, normalization, energy_unit=unit.keV):
	"""

	:param energies:
	:param temp:
	:param normalization:
	:param energy_unit:
	:return:
	"""

	energies=energies*energy_unit.to(unit.erg)
	try:
		energies=energies.value
	except AttributeError:
		energies = energies

	try:
		temp=temp.value
	except AttributeError:
		temp = temp
  
	try:
		normalization=normalization.value
	except AttributeError:
		normalization = normalization



	model =np.empty(energies.size)
	model=(energies**3/(const.h.cgs.value*const.c.cgs.value)**2)*np.exp(-energies/(const.k_B.cgs.value*temp))
	energies=energies*unit.erg.to(energy_unit)
	model=model/np.trapz(model,x=energies)*normalization

	return model
 
def blackbody_function(energies, temp, normalization, energy_unit=unit.keV):
	"""

	:param energies:
	:param temp:
	:param normalization:
	:param energy_unit:
	:return:
	"""

	energies=energies*energy_unit.to(unit.erg)
	try:
		energies=energies.value
	except AttributeError:
		energies = energies

	try:
		temp=temp.value
	except AttributeError:
		temp = temp
  
	try:
		normalization=normalization.value
	except AttributeError:
		normalization = normalization


	model =np.empty(energies.size)
	model=(energies**3/(const.h.cgs.value*const.c.cgs.value)**2)/(np.exp(energies/(const.k_B.cgs.value*temp))-1)
	energies = energies * unit.erg.to(energy_unit)
	model=model/np.trapz(model,x=energies)*normalization

	return model

def comptonized_function(energies, alpha, break_energy, normalization, energy_unit=unit.keV):
	"""

	:param energies:
	:param alpha:
	:param break_energy:
	:param normalization:
	:param energy_unit:
	:return:
	"""
	try:
		energies=energies.value
	except AttributeError:
		energies = energies

	try:
		break_energy=break_energy.value
	except AttributeError:
		break_energy = break_energy
  
	try:
		normalization=normalization.value
	except AttributeError:
		normalization = normalization


	model=np.empty(energies.size)
	model=(energies**alpha)*np.exp(-energies/break_energy)
	model=model/np.trapz(model,x=energies)*normalization

	return model

def goodman_function(energy_maximum, spectrum_maximum):
	"""
	Function that returns Goodman's scalable spherical explosion spectra to compare against a spectra acquired by a
	spherical explosion run in MCRAT. To compare this to simulation data, the simulation spectrum needs to be in units
	of erg/s/energy_unit.

	:param energy_maximum:
	:param spectrum_maximum:
	:return:
	"""

	goodman_energy=10**np.array([-3,-2.8,-2.6,-2.4,-2.2,-2,-1.8,-1.6,-1.4,-1.2,-1,-.8,-.6,-.4,-.2,0,.2,.4,.6,.8,1.,1.2,1.4])
	goodman_spectrum=10**np.array([-5.2,-4.8,-4.5,-4.1,-3.7,-3.4,-3,-2.7,-2.3,-1.95,-1.6,-1.3,-1.1,-0.8,-0.6,-0.4,-0.2,-0.1,-0.2,-0.6,-1,-2.4,-4])

	y_shift=spectrum_maximum/goodman_spectrum.max()
	x_shift=energy_maximum/goodman_energy[goodman_spectrum.argmax()]

	goodman_spectrum_shift=goodman_spectrum*y_shift
	goodman_energy_shift=goodman_energy*x_shift

	return goodman_energy_shift, goodman_spectrum_shift

def bootstrap_parameters(x, y, yerr, function, best_fit, sample_num=1e4):
	"""
	Function that conducts the bootstrapping of the spectral fit to a given spectrum in order to get the errors on the
	parameters of the fitting function.
	:param x:
	:param y:
	:param yerr:
	:param function:
	:param best_fit:
	:param sample_num:
	:return:
	"""
	#can potentially speed up with lmfit package need to require it in setup.py and install with pip install lmfit
	#double checked that this method gives a change in reduced chi squared of 1 which is ~1 sigma
	sample_num=np.int(sample_num)
	resampled_data=np.random.default_rng().normal(y, yerr, size=(sample_num, y.size))

	parameters=np.zeros((best_fit.size, sample_num))
	for i in range(sample_num):
		fit, matrice = curve_fit(function, x, resampled_data[i,:], sigma=yerr,  p0=best_fit, maxfev=5000)
		parameters[:,i]=fit

	return np.mean(parameters, axis=1), np.std(parameters, axis=1)

def calc_epk_error(alpha, break_energy, alpha_error=None, break_energy_error=None):
	"""
	Function that calculates the spectral Epk from a spectral fit and the errors on Epk if the errors ar not set to None
	:param alpha:
	:param break_energy:
	:param alpha_error:
	:param break_energy_error:
	:return:
	"""
	epk=break_energy*(2+alpha)

	if alpha_error is not None and break_energy_error is not None:
		epk_error=np.sqrt(((2 + alpha) * break_energy_error) ** 2 + (break_energy * alpha_error) ** 2)
	else:
		epk_error = np.nan

	return epk, epk_error

def get_FERMI_best_data():
	"""
	A function to acquire data about the FERMI Best GRB sample, as is saved in the file named FERMI_BEST_GRB.dat.
	The data is from Yu et al. (2016).

	:return: returns arrays of the Band or COMP function fitted GRB spectral parameters
	"""
	# need to get the file name off to get the dir mclib is located in
	dir=__file__[::-1].partition('/')[-1][::-1]
	data=np.genfromtxt(dir+'/Data_files/FERMI_BEST_GRB.dat', dtype='U', usecols=(4,7,9,11 ))

	#only want BAND and COMP ones
	Band_Comp_data=data[np.logical_or(data[:,0]=='BAND', data[:,0]=='COMP') ,:]

	parameters=np.zeros([Band_Comp_data.shape[0], 3])
	parameters[:]=np.nan
	parameters[:,0]=Band_Comp_data[:,1].astype("f8") #alpha
	parameters[:,2]=Band_Comp_data[:,3].astype("f8") #peak energy
	parameters[Band_Comp_data[:,0]=='BAND' ,1]=Band_Comp_data[Band_Comp_data[:,0]=='BAND' ,2].astype("f8") #band beta

	#alphas in 1st column, betas in 2nd, etc.
	return parameters

def get_yonetoku_relationship(energies):
	"""
	Returns the Yonetoku relationship for a given set of energies. The original paper scaled L_iso by 10^52 so undo that
	here
	:param energies:
	:return:
	"""
	return 1e52*(2.34e-5)*energies**2

def get_yonetoku_data():
	"""
	Gets the list of observed GRBs taken from Nava et al. (2012) and gets their values and errors on the Yonetoku plane
	:return:
	"""
	# need to get the file name off to get the dir mclib is located in
	dir=__file__[::-1].partition('/')[-1][::-1]
	file_data = np.genfromtxt(dir+'/Data_files/GRB_list.dat', dtype='S', usecols=(8, 10), delimiter='\t')

	E_p = np.zeros(file_data.shape[0])
	L_iso = np.zeros(file_data.shape[0])
	E_p_err = np.zeros(file_data.shape[0])
	L_iso_err = np.zeros(file_data.shape[0])

	count = 0
	for i in range(file_data.shape[0]):
		if ((np.size(np.where(np.fromstring(file_data[i, 0], sep=' \xc2\xb1 ') != -1))) != 0) and (
				(np.size(np.where(np.fromstring(file_data[i, 1], sep=' \xc2\xb1 ') != -1))) != 0):
			E_p[count] = np.fromstring(file_data[i, 0], sep=' \xc2\xb1 ')
			E_p_err[count] = np.float(
				np.fromstring(file_data[i, 0][::-1], sep=' \xc2\xb1 ').astype(int).astype('U')[0][::-1])
			L_iso[count] = np.fromstring(file_data[i, 1], sep=' \xc2\xb1 ')
			L_iso_err[count] = np.float(
				np.fromstring(file_data[i, 1][::-1], sep=' \xc2\xb1 ').astype(float).astype('U')[0][::-1])
			count += 1
	L_iso, L_iso_err= L_iso*1e51, L_iso_err*1e51
	return E_p, E_p_err, L_iso, L_iso_err

def calc_yonetoku_values(spectra_list, lightcurve_list, polarization_list=None):
	"""
	Function that takes a list of spectra and lightcurve dictionaries and calculated the appropriate values for
	where the mock observations would lie on the Yonetoku relationship
	:param spectra_list:
	:param lightcurve_list:
	:param polarization_list:
	:return:
	"""
	num_angles = len(spectra_list)
	# collect data and unscale lightcurve and its error
	L_iso_sim = np.zeros(num_angles)
	L_err_sim = np.zeros(num_angles)
	E_p_sim = np.zeros(num_angles)
	E_p_err_sim = np.zeros(num_angles)
	polarization_deg = np.zeros(num_angles)
	polarization_angle = np.zeros(num_angles)
	polarization_deg_error = np.zeros(num_angles)
	polarization_angle_error = np.zeros(num_angles)
	angles = np.zeros(num_angles)

	count = 0
	for spec, lc in zip(spectra_list, lightcurve_list):
		angles[count] = spec['theta_observer'].value
		E_p_sim[count], E_p_err_sim[count] = calc_epk_error(spec['fit']['alpha'], spec['fit']['break_energy'].value, \
															alpha_error=spec['fit_errors']['alpha_errors'], \
															break_energy_error=spec['fit_errors'][
																'break_energy_errors'].value)
		L_iso_sim[count], L_err_sim[count] = lc['lightcurve'].max().value , lc['lightcurve_errors'][lc['lightcurve'].argmax()].value
		if polarization_list is not None:
			polarization_deg[count], polarization_angle[count], polarization_deg_error[count], polarization_angle_error[
				count] \
				= polarization_list[count]['pol_deg'].value, polarization_list[count]['pol_angle'].value, \
				  polarization_list[count]['pol_deg_errors'].value, polarization_list[count]['pol_angle_errors'].value

		count+=1

	# sort the data by observer viewing angle in case it isnt properly ordered
	L_iso_sim = L_iso_sim[angles.argsort()]
	L_err_sim = L_err_sim[angles.argsort()]
	E_p_sim = E_p_sim[angles.argsort()]
	E_p_err_sim = E_p_err_sim[angles.argsort()]
	polarization_deg = polarization_deg[angles.argsort()]
	polarization_angle = polarization_angle[angles.argsort()]
	polarization_deg_error = polarization_deg_error[angles.argsort()]
	polarization_angle_error = polarization_angle_error[angles.argsort()]
	angles = angles[angles.argsort()]

	return L_iso_sim, L_err_sim, E_p_sim, E_p_err_sim, polarization_deg, polarization_angle, polarization_deg_error,\
		   polarization_angle_error, angles

def get_golenetskii_relationship(value='o'):
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

	return 10**x, 10**y

def get_amati_relationship(value='o'):
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

	return 1e52*10**x,10**y

def calc_amati_values(spectra_list, lightcurve_list):
	"""
	Calculates the mock observed Amati values: isotropic energies and time integrated spectral peak energy for a list of
	mock observed spectra and lightcurves
	:param spectra_list:
	:param lightcurve_list:
	:return:
	"""
	num_angles = len(spectra_list)
	# collect data and unscale lightcurve and its error
	E_iso_sim = np.zeros(num_angles)
	E_iso_err_sim = np.zeros(num_angles)
	E_p_sim = np.zeros(num_angles)
	E_p_err_sim = np.zeros(num_angles)
	angles = np.zeros(num_angles)

	count = 0
	for spec, lc in zip(spectra_list, lightcurve_list):
		angles[count] = spec['theta_observer'].value
		E_p_sim[count], E_p_err_sim[count] = calc_epk_error(spec['fit']['alpha'], spec['fit']['break_energy'].value, \
															alpha_error=spec['fit_errors']['alpha_errors'], \
															break_energy_error=spec['fit_errors'][
																'break_energy_errors'].value)
		E_iso_sim[count]=np.trapz(lc['lightcurve'].value, x=lc['times'].value)
		E_iso_err_sim[count]=E_iso_sim[count]/np.sqrt(spec['ph_num'].value.sum())
		count+=1

	# sort the data by observer viewing angle in case it isnt properly ordered
	E_iso_sim = E_iso_sim[angles.argsort()]
	E_iso_err_sim = E_iso_err_sim[angles.argsort()]
	E_p_sim = E_p_sim[angles.argsort()]
	E_p_err_sim = E_p_err_sim[angles.argsort()]
	angles = angles[angles.argsort()]

	return E_iso_sim, E_iso_err_sim, E_p_sim, E_p_err_sim, angles

def lc_time_to_radius(frame, fps, time):
	return ((frame/fps)-time)*const.c.cgs.value

def calc_line_of_sight(theta_observer, x0_min, x0_max, x1_min, x1_max):
	x_range = np.linspace(x0_min, x0_max, 100000)
	y_angle = np.tan(np.deg2rad(theta_observer)) ** -1 * x_range
	idx = np.where((y_angle > x1_min) & (y_angle < x1_max))

	return x_range[idx], y_angle[idx]

def calc_equal_arrival_time_surface(theta_observer, frame, fps, x0_min, x0_max, time, individual_point=None):
	r=lc_time_to_radius(frame, fps, time)
	if individual_point is None:
		x_range = np.linspace(x0_min, x0_max, 100000)
	else:
		x_range=individual_point
	y_range = -np.tan(np.deg2rad(theta_observer)) * (x_range - r * np.sin(np.deg2rad(theta_observer))) + r * np.cos(np.deg2rad(theta_observer))

	return x_range, y_range

def calc_photon_temp(comov_energy):
	return comov_energy/(3*const.k_B.cgs)


def lorentzBoostVectorized(boost, P_ph):
	save_result=np.zeros_like(P_ph)*np.nan
	indexes = np.where((boost ** 2).sum(axis=0) > 0)[0]
	zero_beta_idx = np.where((boost ** 2).sum(axis=0) == 0)[0]
	Lambda1 = np.zeros([4, 4, indexes.size])

	# fill in matrix for each photon
	beta = np.sqrt((boost[:, indexes] ** 2).sum(axis=0))
	gamma = 1. / np.sqrt(1. - beta ** 2)
	Lambda1[0, 0, :] = gamma
	Lambda1[0, 1, :] = -boost[0, indexes] * gamma
	Lambda1[0, 2, :] = -boost[1, indexes] * gamma
	Lambda1[0, 3, :] = -boost[2, indexes] * gamma
	Lambda1[1, 1, :] = 1. + (gamma - 1.) * boost[0, indexes] ** 2 / (beta ** 2)
	Lambda1[1, 2, :] = (gamma - 1.) * boost[0, indexes] * boost[1, indexes] / (beta ** 2)
	Lambda1[1, 3, :] = (gamma - 1.) * boost[0, indexes] * boost[2, indexes] / (beta ** 2)
	Lambda1[2, 2, :] = 1. + (gamma - 1.) * boost[1, indexes] ** 2 / (beta ** 2)
	Lambda1[2, 3, :] = (gamma - 1.) * boost[1, indexes] * boost[2, indexes] / (beta ** 2)
	Lambda1[3, 3, :] = 1. + (gamma - 1.) * boost[2, indexes] ** 2 / (beta ** 2)

	Lambda1[1, 0, :] = Lambda1[0, 1, :]
	Lambda1[2, 0, :] = Lambda1[0, 2, :]
	Lambda1[3, 0, :] = Lambda1[0, 3, :]
	Lambda1[2, 1, :] = Lambda1[1, 2, :]
	Lambda1[3, 1, :] = Lambda1[1, 3, :]
	Lambda1[3, 2, :] = Lambda1[2, 3, :]

	# perform dot product for each photon with beta>0
	result = np.einsum('ijk,jk->ik', Lambda1, P_ph[:, indexes])
	save_result[:,indexes]=result
	# print(result.shape, indexes.shape, np.where((boost**2).sum(axis=0)<=0)[0], boost[:,np.where((boost**2).sum(axis=0)<=0)[0]], np.sqrt((boost[:,np.where((boost**2).sum(axis=0)<=0)[0]]**2).sum(axis=0)))
	#if (zero_beta_idx.size > 0):
	#	result = np.insert(result, zero_beta_idx, P_ph[:, zero_beta_idx], axis=1)
	#save photns 4 momentum of photons in stationary elements
	save_result[:, zero_beta_idx] =  P_ph[:, zero_beta_idx]

	# print(result.shape)

	save_result = zero_norm(save_result)
	return save_result

def zero_norm(P):
	#test zero norm condition of 4 momenta, if its violated correct the 4 momenta assuming that the energy is correct

	if P.ndim >1:
		not_norm=np.where(P[0,:]**2 != np.linalg.norm(P[1:,:], axis=0)**2)[0]

		P[1:,not_norm]=(P[1:,not_norm]/np.linalg.norm(P[1:,not_norm], axis=0))*P[0,not_norm]

	else:
		#print('Normalizing factor', np.linalg.norm(P[1:]))
		if (P[0]**2 != np.linalg.norm(P[1:])**2):
			#correct the 4 momentum
			P[1:]=(P[1:]/np.linalg.norm(P[1:]))*P[0]

	return P

def calc_optical_depth(scatt_vs_r):
	# reverse because the frame data is saved from largest frame value to smallest (from smallest radius to largest)
	scatt = np.ediff1d(scatt_vs_r[::-1])
	tau = np.cumsum(scatt[::-1])[::-1]

	return tau
