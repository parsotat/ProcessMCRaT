import numpy as np

"""
Basic library for processing MCRaT simulation data.
Written by Tyler Parsotan and D. Lazzati @ OregonState

This includes some legacy code that is kept to allow compatibility with older versions of MCRaT. At some point these may
be deleted from the scripts as MCRaT approaches a well developed stage.
"""

def single_electron(T, P_ph):
	"""
	This routine simulates one electron from a thermal distribution at a
	temperature T. The electron has the correct angular distribution for a
	photon traveling with direction P_ph
	"""
	from scipy import special
	me = 9.1093897e-28
	c_light = 2.99792458e10
	kB = 1.380658e-16
	if T >= 1e7:
		theta = kB * T / me / c_light ** 2
		xdum = np.random.rand(10000) * (1 + 100 * theta)
		betax = np.sqrt(1. - 1. / xdum ** 2)
		ydum = np.random.rand(10000) / 2.
		funx = xdum ** 2 * betax / special.kn(2, 1. / theta) * np.exp(-xdum / theta)
		jj = np.where(ydum <= funx)
		thegamma = xdum[jj[0][0]]
	else:
		sigma = np.sqrt(kB * T / me)
		v_el = np.random.normal(scale=sigma, size=3) / c_light
		g_el = 1. / np.sqrt(1. - np.sum((v_el) ** 2))
		thegamma = g_el
	# print 'Temperature (1e7 K)',T/1e7
	#print ('thegamma',thegamma)
	thebeta = np.sqrt(1. - 1. / thegamma ** 2)
	thedum = np.random.rand(100) * np.pi
	ydum = np.random.rand(100) * 1.3
	yy = np.sin(thedum) * (1 - thebeta * np.cos(thedum))
	jj = np.where(ydum <= yy)
	thetheta = thedum[jj[0][0]]
	# print 'thetheta',thetheta*180/np.pi
	thephi = np.random.rand() * 2 * np.pi
	P_el = me * c_light * thegamma * np.array([1, thebeta * np.cos(thetheta), thebeta * np.sin(thetheta) * np.sin(thephi),
											  thebeta * np.sin(thetheta) * np.cos(thephi)])
	# print 'P_el',P_el
	theta_ph = np.arctan2(np.sqrt(P_ph[2] ** 2 + P_ph[3] ** 2), P_ph[1])
	phi_ph = np.arctan2(P_ph[2], P_ph[3])
	# print theta_ph,phi_ph
	#	First rotate around x-axis to get phi out of the way
	Rot2 = np.zeros([3, 3])
	Rot2[1, 1] = 1.
	Rot2[0, 0] = np.cos(theta_ph)
	Rot2[2, 2] = np.cos(theta_ph)
	Rot2[0, 2] = -np.sin(theta_ph)
	Rot2[2, 0] = np.sin(theta_ph)
	p_el_prime = np.dot(Rot2, P_el[1:])
	P_el_prime = P_el.copy()
	P_el_prime[1:] = p_el_prime
	Rot1 = np.zeros([3, 3])
	Rot1[0, 0] = 1.
	Rot1[1, 1] = np.cos(-phi_ph)
	Rot1[2, 2] = np.cos(-phi_ph)
	Rot1[1, 2] = -np.sin(-phi_ph)
	Rot1[2, 1] = np.sin(-phi_ph)
	p_el_second = np.dot(Rot1, P_el_prime[1:])
	P_el_second = P_el_prime.copy()
	P_el_second[1:] = p_el_second

	return P_el_second


def lorentzBoostVectorized(boost, P_ph):
	"""
	Function to quickly lorentz boost a set of photon 4 momenta, and ensures that the 0 norm condition is met.

	:param boost: The velocity vector of the frame that the photon will be boosted into. The shape of the array
				  should be (3,N), where N is the number of photons that will be lorentz boosted
	:param P_ph: The photon's 4 momentum that will be boosted into the desired frame of reference. The shape of the array
				  should be (4,N)
	:return: returns (4,N) array of the boosted photon's 4 momenta
	"""

	indexes=np.where((boost**2).sum(axis=0)>0)[0]
	zero_beta_idx=np.where((boost**2).sum(axis=0)==0)[0]
	Lambda1=np.zeros([4,4,indexes.size])

	#fill in matrix for each photon
	beta=np.sqrt((boost[:,indexes]**2).sum(axis=0))
	gamma = 1. / np.sqrt(1. - beta ** 2)
	Lambda1[0, 0,:] = gamma
	Lambda1[0, 1,:] = -boost[0,indexes] * gamma
	Lambda1[0, 2,:] = -boost[1,indexes] * gamma
	Lambda1[0, 3,:] = -boost[2,indexes] * gamma
	Lambda1[1, 1,:] = 1. + (gamma - 1.) * boost[0,indexes] ** 2 / (beta ** 2)
	Lambda1[1, 2,:] = (gamma - 1.) * boost[0,indexes] * boost[1,indexes] / (beta ** 2)
	Lambda1[1, 3,:] = (gamma - 1.) * boost[0,indexes] * boost[2,indexes] / (beta ** 2)
	Lambda1[2, 2,:] = 1. + (gamma - 1.) * boost[1,indexes] ** 2 / (beta ** 2)
	Lambda1[2, 3,:] = (gamma - 1.) * boost[1,indexes] * boost[2,indexes] / (beta ** 2)
	Lambda1[3, 3,:] = 1. + (gamma - 1.) * boost[2,indexes] ** 2 / (beta ** 2)

	Lambda1[1, 0,:] = Lambda1[0, 1,:]
	Lambda1[2, 0,:] = Lambda1[0, 2,:]
	Lambda1[3, 0,:] = Lambda1[0, 3,:]
	Lambda1[2, 1,:] = Lambda1[1, 2,:]
	Lambda1[3, 1,:] = Lambda1[1, 3,:]
	Lambda1[3, 2,:] = Lambda1[2, 3,:]

	#perform dot product for each photon
	result=np.einsum('ijk,jk->ik', Lambda1, P_ph[:,indexes])
	if (zero_beta_idx.size>0):
		result=np.insert(result, zero_beta_idx, P_ph[:,zero_beta_idx], axis=1)

	result=zero_norm(result)
	return result

def Lorentz_Boost(boost, P_ph):
	"""
	This routine performs a Lorentz boost of a 4-momentum. The boost is specified with a 3-vel.

	:param boost: The velocity vector of the frame that the photon will be boosted into.
	:param P_ph: The photon 4 momentum that will be boosted into the desired frame of reference.
	:return:  the boosted photon 4 mometum
	"""

	# First, performs Lorentz boost so that the electron is at rest.
	if (boost ** 2).sum() > 0:
		beta = np.sqrt((boost ** 2).sum())
		gamma = 1. / np.sqrt(1. - beta ** 2)
		Lambda1 = np.empty([4, 4])
		Lambda1[0, 0] = gamma
		Lambda1[0, 1] = -boost[0] * gamma
		Lambda1[0, 2] = -boost[1] * gamma
		Lambda1[0, 3] = -boost[2] * gamma
		Lambda1[1, 1] = 1. + (gamma - 1.) * boost[0] ** 2 / (beta ** 2)
		Lambda1[1, 2] = (gamma - 1.) * boost[0] * boost[1] / (beta ** 2)
		Lambda1[1, 3] = (gamma - 1.) * boost[0] * boost[2] / (beta ** 2)
		Lambda1[2, 2] = 1. + (gamma - 1.) * boost[1] ** 2 / (beta ** 2)
		Lambda1[2, 3] = (gamma - 1.) * boost[1] * boost[2] / (beta ** 2)
		Lambda1[3, 3] = 1. + (gamma - 1.) * boost[2] ** 2 / (beta ** 2)

		Lambda1[1, 0] = Lambda1[0, 1]
		Lambda1[2, 0] = Lambda1[0, 2]
		Lambda1[3, 0] = Lambda1[0, 3]
		Lambda1[2, 1] = Lambda1[1, 2]
		Lambda1[3, 1] = Lambda1[1, 3]
		Lambda1[3, 2] = Lambda1[2, 3]

		P_ph_prime = np.dot(Lambda1, P_ph)

		return P_ph_prime
	else:

		return P_ph

def zero_norm(P):
	"""
	Takes a photon 4 momentum and checks if it satisfies the 0 norm condition of photons. If not, it corrects the 4
	mometum by assuming that the energy is correct.

	:param P: photon 4 monetum
	:return: returns the correct 0 normed photon 4 momentum
	"""

	if P.ndim >1:
		not_norm=np.where(P[0,:]**2 != np.linalg.norm(P[1:,:], axis=0)**2)[0]
		P[1:,not_norm]=(P[1:,not_norm]/np.linalg.norm(P[1:,not_norm], axis=0))*P[0,not_norm]
	else:
		if (P[0]**2 != np.linalg.norm(P[1:])**2):
			#correct the 4 momentum
			P[1:]=(P[1:]/np.linalg.norm(P[1:]))*P[0]

	return P


def single_cs(P_el, P_ph):
	"""
	This function conducts a single compton scatter in the electron rest frame. The photon has to be in the comoving
	frame first. Legacy code from Python version of MCRaT.

	:param P_el: electron 4 momentum
	:param P_ph: photon comoving 4 momentum
	:return: returns post-scattered photon 4 momentum in the comoving frame
	"""

	#	This routine performs a Compton scattering between a photon and a
	#	moving electronp. Takes the 4-momenta as inputs
	#
	# Fist let's define some constants:
	me = 9.1093897e-28
	c = 2.99792458e10
	#
	# First, performs Lorentz boost so that the electron is at rest.
	P_el_prime = Lorentz_Boost(P_el[1:] / P_el[0], P_el)
	P_ph_prime = Lorentz_Boost(P_el[1:] / P_el[0], P_ph)

	###################################################################################################################
	#test norm of photon
	P_ph_prime=zero_norm(P_ph_prime)
	###################################################################################################################

	#
	# Second, we rotate the axes so that the photon incomes along the x-axis
	# 2.1 we first rotate it to the xz plane
	phi = np.arctan2(P_ph_prime[2], P_ph_prime[1])
	Rot1 = np.zeros([3, 3])
	Rot1[2, 2] = 1.
	Rot1[0, 0] = np.cos(-phi)
	Rot1[1, 1] = np.cos(-phi)
	Rot1[0, 1] = -np.sin(-phi)
	Rot1[1, 0] = np.sin(-phi)
	p_ph_second = np.dot(Rot1, P_ph_prime[1:])

	#y_tilde_rot_1=np.dot(-Rot1, y_tilde_1) #rotate the stokes plane as well
	#x_tilde_rot_1=np.dot(-Rot1, x_tilde_1)

	P_ph_second = P_ph_prime.copy()
	P_ph_second[1:] = p_ph_second
	P_ph_second[2] = 0
	P_el_second = P_el_prime.copy()

	#print(np.arccos(np.dot(P_ph_second[1:], y_tilde_rot_1)/np.sqrt(np.dot(P_ph_second[1:],P_ph_second[1:])))*180/np.pi, np.arccos(np.dot(P_ph_second[1:], x_tilde_rot_1)/np.sqrt(np.dot(P_ph_second[1:],P_ph_second[1:])))*180/np.pi)


	# 2.2 now we rotate around y to bring it all along x
	phi2 = np.arctan2(p_ph_second[2], p_ph_second[0])
	Rot2 = np.zeros([3, 3])
	Rot2[1, 1] = 1.
	Rot2[0, 0] = np.cos(-phi2)
	Rot2[2, 2] = np.cos(-phi2)
	Rot2[0, 2] = -np.sin(-phi2)
	Rot2[2, 0] = np.sin(-phi2)
	p_ph_third = np.dot(Rot2, P_ph_second[1:])

	#y_tilde_rot_2=np.dot(Rot2, y_tilde_rot_1)
	#x_tilde_rot_2=np.dot(Rot2, x_tilde_rot_1)

	P_ph_third = P_ph_second.copy()
	P_ph_third[1:] = p_ph_third
	P_ph_third[1] = P_ph_third[0]
	P_ph_third[3] = 0
	P_el_third = P_el_second.copy()

	#print(np.arccos(np.dot(P_ph_third[1:], y_tilde_rot_2)/np.sqrt(np.dot(P_ph_third[1:],P_ph_third[1:])))*180/np.pi, np.arccos(np.dot(P_ph_third[1:], x_tilde_rot_2)/np.sqrt(np.dot(P_ph_third[1:],P_ph_third[1:])))*180/np.pi)

	#rotate y_yilde_rot_2 such that it is aligned with the z axis
	#t=np.arccos(np.dot([0,0,1], y_tilde_rot_2))
	#Rot_stokes = np.zeros([3, 3])
	#Rot_stokes[0, 0] = 1.
	#Rot_stokes[1, 1] = np.cos(t)
	#Rot_stokes[2, 2] = np.cos(t)
	#Rot_stokes[1, 2] = -np.sin(t)
	#Rot_stokes[2, 1] = np.sin(t)

	#y_tilde_rot_3=np.dot(Rot_stokes, y_tilde_rot_2)
	#x_tilde_rot_3=np.dot(Rot_stokes, x_tilde_rot_2)


	#
	# Third we perform the scattering
	#
	# 3.1 we generate a phi and a theta angles
	phi3 = np.random.rand() * 2 * np.pi
	dumx = np.random.rand(100) * np.pi
	dumy = np.random.rand(100) * 1.09
	dumyy = (1. + np.cos(dumx) ** 2) * np.sin(dumx)
	jj = np.where(dumyy >= dumy)
	theta3 = dumx[jj[0][0]]
	# 3.2 compute new 4-momenta of electron and photon
	P_ph_fourth = np.zeros(4)
	P_ph_fourth[0] = P_ph_third[0] / (1. + P_ph_third[0] / me / c * (1. - np.cos(theta3)))
	P_ph_fourth[3] = P_ph_fourth[0] * np.sin(theta3) * np.cos(phi3)
	P_ph_fourth[2] = P_ph_fourth[0] * np.sin(theta3) * np.sin(phi3)
	P_ph_fourth[1] = P_ph_fourth[0] * np.cos(theta3)
	#	print 'norm of ph 4',P_ph_fourth[0]**2-np.sum(P_ph_fourth[1:]**2)
	P_el_fourth = P_el_third + P_ph_third - P_ph_fourth  ###
	if np.sqrt(np.sum((P_el_fourth[1:] / P_el_fourth[0]) ** 2)) > 1:
		print('Electron 4 momentum  not normalized')
		exit()
	#
	# Fourth we rotate back into the original boosted frame
	#
	Rot3 = np.zeros([3, 3])
	Rot3[1, 1] = 1.
	Rot3[0, 0] = np.cos(-phi2)
	Rot3[2, 2] = np.cos(-phi2)
	Rot3[0, 2] = np.sin(-phi2)
	Rot3[2, 0] = -np.sin(-phi2)
	p_ph_fifth = np.dot(Rot3, P_ph_fourth[1:])
	P_ph_fifth = P_ph_fourth.copy()
	P_ph_fifth[1:] = p_ph_fifth
	p_el_fifth = np.dot(Rot3, P_el_fourth[1:])
	P_el_fifth = P_el_fourth.copy()
	P_el_fifth[1:] = p_el_fifth
	if np.sqrt(np.sum((P_el_fifth[1:] / P_el_fifth[0]) ** 2)) > 1:
		print('Electron 4 momentum  not normalized')
		exit()

	Rot4 = np.zeros([3, 3])
	Rot4[2, 2] = 1.
	Rot4[0, 0] = np.cos(-phi)
	Rot4[1, 1] = np.cos(-phi)
	Rot4[0, 1] = np.sin(-phi)
	Rot4[1, 0] = -np.sin(-phi)
	p_ph_sixth = np.dot(Rot4, P_ph_fifth[1:])
	P_ph_sixth = P_ph_fifth.copy()
	P_ph_sixth[1:] = p_ph_sixth
	p_el_sixth = np.dot(Rot4, P_el_fifth[1:])
	P_el_sixth = P_el_fifth.copy()
	P_el_sixth[1:] = p_el_sixth
	#
	# Fifth we de-boost to the lab
	#
	P_el_seventh = Lorentz_Boost(-P_el[1:] / P_el[0], P_el_sixth)
	P_ph_seventh = Lorentz_Boost(-P_el[1:] / P_el[0], P_ph_sixth)
	###################################################################################################################
	#test norm of photon
	P_ph_seventh=zero_norm(P_ph_seventh)
	###################################################################################################################

	return P_el_seventh, P_ph_seventh


def event3D(r_obs, theta_deg, phi_deg, dtheta_deg, path, lastfile, sim_type, riken_switch=False):
	"""
	Place holder function to conduct a synthetic observation of MCRaT simulated outflow using a 3D hydro simulationp.

	:param r_obs:
	:param theta_deg:
	:param phi_deg:
	:param dtheta_deg:
	:param path:
	:param lastfile:
	:param sim_type:
	:param riken_switch:
	:return:
	"""
	theta = theta_deg * np.pi / 180  # angle between jet axis and line of sight
	dtheta = dtheta_deg * np.pi / 180  # acceptance
	phi=phi_deg* np.pi / 180
	foutn='events.dat'
	fout=np.str(sim_type)+'_'+"%.1e"% r_obs+'_'+np.str(theta_deg)+'.evt'

	#comment out when using mc.par file for weight
	weight = np.loadtxt(path + 'mcdata_PW.dat')
	print( weight.shape)

	c_light = 2.99792458e10
	i = lastfile
	P0 = np.loadtxt(path + 'mcdata_' + str(i) + '_P0.dat')
	sizze = P0.shape
	print(sizze, P0.size)
	P1 = np.loadtxt(path + 'mcdata_' + str(i) + '_P1.dat')
	P2 = np.loadtxt(path + 'mcdata_' + str(i) + '_P2.dat')
	P3 = np.loadtxt(path + 'mcdata_' + str(i) + '_P3.dat')
	R0 = np.loadtxt(path + 'mcdata_' + str(i) + '_R0.dat')
	R1 = np.loadtxt(path + 'mcdata_' + str(i) + '_R1.dat')
	R2 = np.loadtxt(path + 'mcdata_' + str(i) + '_R2.dat')
	P0 = np.reshape(P0, P0.size)
	P1 = np.reshape(P1, P0.size)
	P2 = np.reshape(P2, P0.size)
	P3 = np.reshape(P3, P0.size)
	R1 = np.reshape(R1, P0.size)
	R2 = np.reshape(R2, P0.size)
	R3 = np.reshape(R3, P0.size)
	RR = np.sqrt(R0 ** 2 + R1 ** 2 + R2 ** 2)  # radius of propagation

	if not riken_switch:
		theta_pos = np.arccos(R2/RR)  # angle between position vector and polar axis
		phi_pos=np.arctan2(R1, R0)
	else:
		theta_pos = np.arccos(R1/RR)  # angle between position vector and y axis for RIKEN Hydro
		phi_pos=np.arctan2(R2, R0)

	theta_rel = theta_pos - theta  # angle between position vector and line of sight
	RR_prop = RR * np.cos(theta_rel)

	#comment out when using mc.par file for weight
	weight=np.reshape(weight[:sizze[0]],P0.size)

	#read photon weight from mc.par file
	#weight=np.ones(P0.size)*(np.genfromtxt(path+'mc.par')[-3])
	#weight = np.reshape(weight, P0.size)

	if not riken_switch:
		theta_pho = np.arccos(P3/np.sqrt(P1 ** 2 + P2 ** 2 + P3**2))  # angle between velocity vector and polar axis
		phi_pho = np.arctan2(P2, P1)
	else:
		theta_pho = np.arccos(P2/np.sqrt(P1 ** 2 + P2 ** 2  + P3**2))  # angle between velocity vector and y axis
		phi_pho = np.arctan2(P3, P1)

	# dtheta applies for theta and phi directions, make sure light curve function also has correct factors for the 3D case
	jj = np.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.)  & (phi_pho >= phi - dtheta / 2.) & (phi_pho < phi + dtheta / 2.) & (RR_prop >= r_obs))
	print ('accepted photons ', jj[0].size)
	nnn = jj[0].size
	if not riken_switch:
		fps=5.0
	else:
		fps=10.0


	tnow = lastfile / fps
	dr = RR_prop[jj] - r_obs
	vel_proj = c_light * np.cos(theta_pho[jj])
	dt = dr / vel_proj
	crossing_time = tnow - dt - r_obs / c_light
	hnukeV = P0[jj] * 3e10 / 1.6e-9
	outarr = np.zeros([nnn, 3])
	outarr[:, 0] = crossing_time
	outarr[:, 1] = hnukeV
	outarr[:, 2] = weight[jj]
	np.savetxt(path+foutn,outarr)
	np.savetxt('EVENT_FILES/'+fout,outarr)

def event4(r_obs, theta_deg, dtheta_deg, path, lastfile, sim_type, withintheta_deg=False,riken_switch=False, save_event=True):
	"""
	Function to collect MCRaT photons in space that will be observed. legacy code from when MCRaT produced text files
	as its output.

	:param r_obs: radius of where the detector will be placed, should be at a radius such that all the photons have
		   propagated past the detector by the end of the MCRaT simulation
	:param theta_deg: observer viewing angle in degrees
	:param dtheta_deg: delta theta of the observer viewieng angle for accepting photons, also in degrees
	:param path: path to the directory that holds ll the output MCRaT files, should be a string
	:param lastfile: the number of the last MCRaT output file, this should be an int
	:param sim_type: A string that will be the name of the output file of this function
	:param withintheta_deg: this is a depreciated key
	:param riken_switch: this is a depreciated key
	:param save_event: switch to create the event file that will be read in by other functions
	:return: returns photon's that would be observed and thier energy, detector crossing time, and their weight
	"""
	theta = theta_deg * np.pi / 180  # angle between jet axis and line of sight
	dtheta = dtheta_deg * np.pi / 180  # acceptance
	foutn='events.dat'
	if not withintheta_deg:
		fout=np.str(sim_type)+'_'+"%.1e"% r_obs+'_'+np.str(theta_deg)+'.evt'
	else:
		fout=np.str(sim_type)+'_within_theta_'+"%.1e"% r_obs+'_'+np.str(theta_deg)+'.evt'

	#comment out when using mc.par file for weight
	weight = np.loadtxt(path + 'mcdata_PW.dat')
	print( weight.shape)

	c_light = 2.99792458e10
	i = lastfile
	P0 = np.loadtxt(path + 'mcdata_' + str(i) + '_P0.dat')
	sizze = P0.shape
	print(sizze, P0.size)
	P1 = np.loadtxt(path + 'mcdata_' + str(i) + '_P1.dat')
	P2 = np.loadtxt(path + 'mcdata_' + str(i) + '_P2.dat')
	P3 = np.loadtxt(path + 'mcdata_' + str(i) + '_P3.dat')
	R1 = np.loadtxt(path + 'mcdata_' + str(i) + '_R0.dat')
	R2 = np.loadtxt(path + 'mcdata_' + str(i) + '_R1.dat')
	R3 = np.loadtxt(path + 'mcdata_' + str(i) + '_R2.dat')
	P0 = np.reshape(P0, P0.size)
	P1 = np.reshape(P1, P0.size)
	P2 = np.reshape(P2, P0.size)
	P3 = np.reshape(P3, P0.size)
	R1 = np.reshape(R1, P0.size)
	R2 = np.reshape(R2, P0.size)
	R3 = np.reshape(R3, P0.size)
	RR = np.sqrt(R1 ** 2 + R2 ** 2 + R3 ** 2)  # radius of propagation
	theta_pos = np.arctan2(np.sqrt(R1 ** 2 + R2 ** 2), R3)  # angle between position vector and polar axis
	theta_rel = theta_pos - theta  # angle between position vector and line of sight
	RR_prop = RR * np.cos(theta_rel)

	#comment out when using mc.par file for weight
	weight=np.reshape(weight[:sizze[0]],P0.size)


	theta_pho = np.arctan2(np.sqrt(P1 ** 2 + P2 ** 2), P3)  # angle between velocity vector and polar axis

	if not withintheta_deg:
		jj = np.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs))
	else:
		jj = np.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs) & (theta_pos >= theta - dtheta / 2.) & (theta_pos < theta + dtheta / 2.))

	print ('accepted photons ', jj[0].size)
	nnn = jj[0].size
	if not riken_switch:
		fps=5.0
	else:
		fps=10.0


	tnow = lastfile / fps
	dr = RR_prop[jj] - r_obs
	vel_proj = c_light * np.cos(theta_pho[jj]-theta)
	dt = dr / vel_proj
	crossing_time = tnow - dt - r_obs / c_light
	hnukeV = P0[jj] * 3e10 / 1.6e-9
	outarr = np.zeros([nnn, 4])
	outarr[:, 0] = crossing_time
	outarr[:, 1] = hnukeV
	outarr[:, 2] = weight[jj]
	outarr[:, 3] =  jj[0] #save indexes of the photons in the whole set of photons
	if save_event:
		#np.savetxt(path+foutn,outarr)
		np.savetxt('EVENT_FILES/'+fout,outarr)
		
	return  crossing_time, hnukeV, weight[jj], jj[0]

def event_h5(r_obs, theta_deg, dtheta_deg, path, lastfile, sim_type, fps=5, read_comv=False, read_stokes=False, read_type=False, save_event_file=True, append=False):
	"""
	Function to collect MCRaT photons in space that will be observed. Saves event files in EVENT_FILES/ directory that
	must be created before calling this functionp.

	:param r_obs: radius of where the detector will be placed, should be at a radius such that all the photons have
		   propagated past the detector by the end of the MCRaT simulation
	:param theta_deg: observer viewing angle in degrees
	:param dtheta_deg: delta theta of the observer viewing angle for accepting photons, also in degrees
	:param path: path to the directory that holds ll the output MCRaT files, should be a string
	:param lastfile: the number of the last MCRaT output file, this should be an int
	:param sim_type: A string that will be the name of the output file of this function
	:param fps: an int that represents the number fo frames per second in the hydro simulation used
	:param read_comv: switch to denote of the MCRaT files contain the comoving photon 4 momenta
	:param read_stokes: switch to denote if the MCRaT output files have the stokes parameters
	:param save_event_file: switch to determine if the function should save the event file or not
	:param append: switch to denote if the function should append the extracted data to a pre-existing event file
	:return: returns (N,8) array where N is the number of photons 'detected'. The array contains:
			  the crossing time of the photon, the lab energy in keV, the weight, the index in the list of photons,
			  the stokes parameters and the comoving energy in keV
	"""
	from read_process_files import read_mcrat_h5
	import h5py as h5
	
	c_light = 2.99792458e10
	
	data=read_mcrat_h5(path+"mcdata_"+np.str_(lastfile), read_comv=read_comv, read_stokes=read_stokes, read_type=read_type)
	if read_comv and read_stokes:
		if read_type:
			PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3, PT=data
		else:
			PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3=data
			PT = np.zeros((np.size(P0))) * np.nan
	elif read_comv and not read_stokes:
		if read_type:
			PW, NS, P0, P1, P2, P3, R1, R2, R3, COMV_P0, COMV_P1, COMV_P2, COMV_P3, PT = data
		else:
			PW, NS, P0, P1, P2, P3, R1, R2, R3, COMV_P0, COMV_P1, COMV_P2, COMV_P3=data
			PT = np.zeros((np.size(P0))) * np.nan
		S0, S1, S2, S3 = np.zeros((4, np.size(P0))) * np.nan
	elif not read_comv and read_stokes:
		if read_type:
			PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3, PT = data
		else:
			PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3=data
			PT = np.zeros((np.size(P0))) * np.nan
		COMV_P0, COMV_P1, COMV_P2, COMV_P3 = np.zeros((4, np.size(P0))) * np.nan
	else:
		if read_type:
			PW, NS, P0, P1, P2, P3, R1, R2, R3, PT = data
		else:
			PW, NS, P0, P1, P2, P3, R1, R2, R3=data
			PT = np.zeros((np.size(P0))) * np.nan
		S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3 = np.zeros((8, np.size(P0))) * np.nan


	print('Total Number of Photons: ', np.size(P0), P0)

	weight = PW
		
	theta = theta_deg * np.pi / 180  # angle between jet axis and line of sight
	dtheta = dtheta_deg * np.pi / 180  # acceptance
	foutn='events.dat'
	fout=np.str(sim_type)+'_'+"%.2e"% r_obs+'_'+np.str(theta_deg)+'.evt'
	#fout = np.str(sim_type) + '_' + "%.2e" % r_obs + '_' + "%0.15lf" %(theta_deg) + '.evt' #thsi was used for the lundman comparisons
	
	RR = np.sqrt(R1 ** 2 + R2 ** 2 + R3 ** 2)  # radius of propagation
	theta_pos = np.arctan2(np.sqrt(R1 ** 2 + R2 ** 2), R3)  # angle between position vector and polar axis
	theta_rel = theta_pos - theta  # angle between position vector and line of sight
	RR_prop = RR * np.cos(theta_rel)

	theta_pho = np.arctan2(np.sqrt(P1 ** 2 + P2 ** 2), P3)  # angle between velocity vector and polar axis

	#print(np.histogram(theta_pho, bins=50)[0], np.histogram(theta_pho, bins=50)[1]*180/np.pi)

	jj = np.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs) ) #& (S1 != 0) &  (S2 != 0) added last two to properly calc polarization in test cases
	print ('accepted photons ', jj[0].size)
	nnn = jj[0].size


	tnow = lastfile / fps
	dr = RR_prop[jj] - r_obs
	vel_proj = c_light * np.cos(theta_pho[jj]-theta)
	dt = dr / vel_proj
	crossing_time = tnow - dt - r_obs / c_light
	hnukeV = P0[jj] * 3e10 / 1.6e-9
	comv_hnukeV= COMV_P0[jj] * 3e10 / 1.6e-9
	outarr = np.zeros([nnn, 11], dtype=object)
	outarr[:, 0] = crossing_time
	outarr[:, 1] = hnukeV
	outarr[:, 2] = weight[jj]
	outarr[:, 3] =  jj[0] #save indexes of the photons in the whole set of photons
	outarr[:, 4] = S0[jj]
	outarr[:, 5] = S1[jj]
	outarr[:, 6] = S2[jj]
	outarr[:, 7] = S3[jj]
	outarr[:, 8] = comv_hnukeV
	outarr[:, 9] = NS[jj]
	outarr[:, 10] = PT[jj]


	if save_event_file:
		if append:
			np.savetxt('EVENT_FILES/'+fout,outarr, fmt='%.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %s')
		else:
			with open('EVENT_FILES/'+fout, 'ba') as f:
				np.savetxt(f, outarr, fmt='%.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %.18e %s')


	return crossing_time, hnukeV, weight[jj], jj[0], S0[jj], S1[jj], S2[jj], S3[jj], comv_hnukeV, NS[jj], PT[jj]

def read_event_file(simid, h5=True):
	"""
	An example function for reading the event files produced by event_h5.

	:param simid: the name of the event file to be read
	:param h5: switch to set whether the MCRaT simulation used hdf5 files (should always be set to true for the current versions of MCRaT)
	:return: returns the data from the event file
	"""
	if h5:
		try:
			time, hnu, weight, indexes, s0, s1, s2, s3, comv_hnu, num_scatt = np.loadtxt('EVENT_FILES/' + simid + '.evt',
																			  unpack=True,
																			  usecols=[0, 1, 2, 3, 4, 5, 6, 7,
																					   8, 9])  # comv_hnu
			pt = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True, usecols=[10], dtype='|S15').astype(str)
			return time, hnu, weight, indexes, s0, s1, s2, s3, comv_hnu, pt
		except IndexError:
			time, hnu, weight, s0, s1, s2, s3 = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)
			return time, hnu, weight, s0, s1, s2, s3
	else:
		try:
			time, hnu, weight, indexes = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)
			return time, hnu, weight, indexes
		except ValueError:
			time, hnu, weight = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)
			return time, hnu, weight

def lcur(simid, t, units='erg/s', theta=1., dtheta=1., phi=0, dphi=1, sim_dims=2, h5=True, photon_type=None, energy_range=None):
	"""
	reads in the event file and bins photons in uniform time bins to create light curves

	:param simid: a string of the event file base name of the event file created in event_h5 (everything but the .evt)
	:param t: an array of time bin edges
	:param units: string specifying units, can be erg/s or cts/s
	:param theta: observer viewing angle in degrees
	:param dtheta: the size of the observer viewing angle bin in degrees (same as is specified in event_h5 function)
	:param phi: azimuthal angle for mock observer in degrees (only for 3D simulations, not fully supported)
	:param dphi: the size of the observer azimuthal viewing angle bin in degrees (only for 3D simulations, not fully
				supported)
	:param sim_dims: The number of dimensions of the hydro simulation used
	:param h5: specify if the format of the MCRaT output files is hdf5 files or not (denotes if using an old format or
			   a newer format of saving files)
	:param photon_type: can be set to 's', 'i', or left as None in order to select thermal synchrotron photons, injected photons, or all the photons in the simulation for analysis
	:param energy_range: has units of keV, can be left as None to choose photons of all energy ranges for analysis or it can be set to an array with [min energy, max energy] e.g. [1, 10] for 1 to 10 keV (limits inclusive)
	:return: returns the time binned quantities of luminosity, luminosity error, number of photons in each bin, the
			 average eenrgy of photons in each bin, polarization, the stokes parameters, polarization error and
			 polarization angle
	"""
	from read_process_files import calc_kislat_error

	if (units != 'cts/s') & (units != 'erg/s'):
		print( 'Wrong units')
		print( 'The only allowed units are: erg/s and cts/s')
	else:
		if h5:
			try:
				#time,hnu,weight, indexes, s0, s1, s2, s3=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)#comv_hnu
				time,hnu,weight, indexes, s0, s1, s2, s3, comv_hnu, num_scatt=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) #comv_hnu
				pt = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True, usecols=[10], dtype='|S15').astype(str)
			except IndexError:
				time,hnu,weight, indexes, s0, s1, s2, s3=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			
		else:
			try:
				time,hnu,weight, indexes=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			except ValueError:
				time,hnu,weight=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)

		p = np.empty(t.size)*np.nan
		p_angle = np.empty(t.size)*np.nan
		l = np.empty(t.size)*np.nan
		q = np.empty(t.size)*np.nan
		u = np.empty(t.size)*np.nan
		v = np.empty(t.size)*np.nan
		perr = np.empty((t.size,2))*np.nan
		lc = np.zeros(t.size)#np.empty(t.size)*np.nan
		lce = np.empty(t.size)*np.nan
		ph_num = np.empty(t.size)*np.nan
		ph_avg_energy = np.empty(t.size)*np.nan
		weight_std=np.zeros(t.size)
		ns = np.zeros(t.size)*np.nan

		for i in range(t.size-1):
			# print(tmin[i], tmax[i])
			if photon_type==None:
				if energy_range == None:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)))
				else:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)) & (hnu>=energy_range[0]) & (hnu<=energy_range[1]))
			else:
				if energy_range == None:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)) & (pt==photon_type))
				else:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)) & (pt==photon_type) & (hnu>=energy_range[0]) & (hnu<=energy_range[1]))

			if jj[0].size > 0:
				if units == 'erg/s':
					lc[i] = np.sum(weight[jj] * hnu[jj] * 1.6e-9)/(t[i+1]-t[i])
				if units == 'cts/s':
					lc[i] = np.sum(weight[jj])/(t[i+1]-t[i])
				lce[i] = lc[i] / np.sqrt(jj[0].size)
				ph_num[i] = jj[0].size
				ph_avg_energy[i] = np.average(hnu[jj] * 1.6e-9, weights=weight[jj])  # in ergs
				weight_std[i]=(np.sqrt(np.mean(weight[jj] ** 2) / np.mean(weight[jj]) ** 2))
				ns[i] = np.average(num_scatt[jj], weights=weight[jj])
				if h5:
					# p[i]=np.average(np.sqrt(np.sum(s1[jj])**2+np.sum(s2[jj])**2)/np.sum(s0[jj]), weights=weight[jj])
					l[i] = np.average(s0[jj], weights=weight[jj])
					q[i] = np.average(s1[jj], weights=weight[jj])
					u[i] = np.average(s2[jj], weights=weight[jj])
					v[i] = np.average(s3[jj], weights=weight[jj])
					I = np.average(s0[jj], weights=weight[jj])
					p[i] = np.sqrt(q[i] ** 2 + u[i] ** 2) / I
					p_angle[i] = (
								0.5 * np.arctan2(u[i], q[i]) * 180 / np.pi)


					#from Kislat
					l[i], q[i], u[i], v[i], p[i], p_angle[i], perr[i,0], perr[i,1] = calc_kislat_error(s0[jj], s1[jj], s2[jj], s3[jj], weight[jj])


		if sim_dims == 2:
			factor = 2 * np.pi * (np.cos((theta - dtheta / 2.) * np.pi / 180) - np.cos((theta + dtheta / 2.) * np.pi / 180))
		elif sim_dims == 3:
			factor = (dphi * np.pi / 180) * (
						np.cos((theta - dtheta / 2.) * np.pi / 180) - np.cos((theta + dtheta / 2.) * np.pi / 180))


		lc =  lc  / factor
		lce =  lce  / factor
		return lc, lce, ph_num, ph_avg_energy, p, l, q, u, v, perr, p_angle, ns, t


def lcur_old(simid, t, units='erg/s', theta=1., dtheta=1., phi=0, dphi=1, sim_dims=2, h5=True, photon_type=None,
		 energy_range=None):
	"""
	reads in the event file and bins photons in uniform time bins to create light curves

	:param simid: a string of the event file base name of the event file created in event_h5 (everything but the .evt)
	:param t: an array of time bin edges
	:param units: string specifying units, can be erg/s or cts/s
	:param theta: observer viewing angle in degrees
	:param dtheta: the size of the observer viewing angle bin in degrees (same as is specified in event_h5 function)
	:param phi: azimuthal angle for mock observer in degrees (only for 3D simulations, not fully supported)
	:param dphi: the size of the observer azimuthal viewing angle bin in degrees (only for 3D simulations, not fully
				supported)
	:param sim_dims: The number of dimensions of the hydro simulation used
	:param h5: specify if the format of the MCRaT output files is hdf5 files or not (denotes if using an old format or
			   a newer format of saving files)
	:param photon_type: can be set to 's', 'i', or left as None in order to select thermal synchrotron photons, injected photons, or all the photons in the simulation for analysis
	:param energy_range: has units of keV, can be left as None to choose photons of all energy ranges for analysis or it can be set to an array with [min energy, max energy] e.g. [1, 10] for 1 to 10 keV (limits inclusive)
	:return: returns the time binned quantities of luminosity, luminosity error, number of photons in each bin, the
			 average eenrgy of photons in each bin, polarization, the stokes parameters, polarization error and
			 polarization angle
	"""
	from read_process_files import calc_kislat_error

	if (units != 'cts/s') & (units != 'erg/s'):
		print('Wrong units')
		print('The only allowed units are: erg/s and cts/s')
	else:
		if h5:
			try:
				# time,hnu,weight, indexes, s0, s1, s2, s3=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)#comv_hnu
				time, hnu, weight, indexes, s0, s1, s2, s3, comv_hnu, num_scatt = np.loadtxt(
					'EVENT_FILES/' + simid + '.evt', unpack=True, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # comv_hnu
				pt = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True, usecols=[10], dtype='|S15').astype(str)
			except IndexError:
				time, hnu, weight, indexes, s0, s1, s2, s3 = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)

		else:
			try:
				time, hnu, weight, indexes = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)
			except ValueError:
				time, hnu, weight = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)

		p = np.zeros(t.size) * np.nan
		p_angle = np.zeros(t.size) * np.nan
		l = np.zeros(t.size)
		q = np.zeros(t.size)
		u = np.zeros(t.size)
		v = np.zeros(t.size)
		perr = np.zeros((t.size, 2)) * np.nan
		lc = np.zeros(t.size)
		lce = np.zeros(t.size)
		ph_num = np.zeros(t.size)
		ph_avg_energy = np.zeros(t.size)
		dt = t[1] - t[0]
		tmin = t  # - dt / 2.
		tmax = t + dt  # / 2.
		weight_std = np.zeros(t.size)
		ns = np.zeros(t.size)
		for i in range(t.size):
			# print(tmin[i], tmax[i])
			if photon_type == None:
				if energy_range == None:
					jj = np.where((time >= tmin[i]) & (time < tmax[i]) & (~np.isnan(s0)))
				else:
					jj = np.where((time >= tmin[i]) & (time < tmax[i]) & (~np.isnan(s0)) & (hnu >= energy_range[0]) & (
								hnu <= energy_range[1]))
			else:
				if energy_range == None:
					jj = np.where((time >= tmin[i]) & (time < tmax[i]) & (~np.isnan(s0)) & (pt == photon_type))
				else:
					jj = np.where((time >= tmin[i]) & (time < tmax[i]) & (~np.isnan(s0)) & (pt == photon_type) & (
								hnu >= energy_range[0]) & (hnu <= energy_range[1]))

			if jj[0].size > 0:
				if units == 'erg/s':
					lc[i] = np.sum(weight[jj] * hnu[jj] * 1.6e-9)
				if units == 'cts/s':
					lc[i] = np.sum(weight[jj])
				lce[i] = lc[i] / np.sqrt(jj[0].size)
				ph_num[i] = jj[0].size
				ph_avg_energy[i] = np.average(hnu[jj] * 1.6e-9, weights=weight[jj])  # in ergs
				weight_std[i] = (np.sqrt(np.mean(weight[jj] ** 2) / np.mean(weight[jj]) ** 2))
				ns[i] = np.average(num_scatt[jj], weights=weight[jj])
				if h5:
					# p[i]=np.average(np.sqrt(np.sum(s1[jj])**2+np.sum(s2[jj])**2)/np.sum(s0[jj]), weights=weight[jj])
					l[i] = np.average(s0[jj], weights=weight[jj])
					q[i] = np.average(s1[jj], weights=weight[jj])
					u[i] = np.average(s2[jj], weights=weight[jj])
					v[i] = np.average(s3[jj], weights=weight[jj])
					I = np.average(s0[jj], weights=weight[jj])
					# print(u[i])
					p[i] = np.sqrt(q[i] ** 2 + u[i] ** 2)  # /I
					p_angle[i] = (0.5 * np.arctan2(u[i], q[
						i]) * 180 / np.pi)  # gives angle between +90 degrees and -90 degrees

					# from Kislat
					l[i], q[i], u[i], v[i], p[i], p_angle[i], perr[i, 0], perr[i, 1] = calc_kislat_error(s0[jj], s1[jj],
																										 s2[jj], s3[jj],
																										 weight[jj])

		# calulate d \Omega
		if sim_dims == 2:
			factor = 2 * np.pi * (
						np.cos((theta - dtheta / 2.) * np.pi / 180) - np.cos((theta + dtheta / 2.) * np.pi / 180))
		elif sim_dims == 3:
			factor = (dphi * np.pi / 180) * (
						np.cos((theta - dtheta / 2.) * np.pi / 180) - np.cos((theta + dtheta / 2.) * np.pi / 180))

		lc = lc / dt / factor
		lce = lce / dt / factor
		return lc, lce, ph_num, ph_avg_energy, p, l, q, u, v, perr, p_angle, ns, t


def lcur_var_t(simid, time_start, time_end, dt, dt_min=0.2, liso_c = 1e50, units='erg/s', theta=1., dtheta=1., phi=0, dphi=1, sim_dims=2, h5=True, photon_type=None, energy_range=None, use_Lcrit=False):
	"""
	Produces time binned quantities for non-uniform time bins. The time bins must be larger than some critical
	luminosity and some minimum dt that the user specifies.

	:param simid: a string of the event file base name of the event file created in event_h5 (everything but the .evt)
	:param time_start: starting time of the light curve
	:param time_end: end time of the light curve binning
	:param dt: initial dt of the time bins
	:param dt_min: the minimum acceptable dt for the light curve
	:param liso_c: the mimimum isotropic luminosity for a given time bin (in the same units specified by units)
	:param units: a string of the units of the light curve that will be produced (erg/s or cts/s)
	:param theta: the observer viewing angle in degrees
	:param dtheta: the size of the observer viewing angle bin in degrees (same as is specified in event_h5 function)
	:param phi: azimuthal angle for mock observer in degrees (only for 3D simulations, not fully supported)
	:param dphi: the size of the observer azimuthal viewing angle bin in degrees (only for 3D simulations, not fully
				supported)
	:param sim_dims: The number of dimensions of the hydro simulation used
	:param h5: specify if the format of the MCRaT output files is hdf5 files or not (denotes if using an old format or
			   a newer format of saving files)
	:param photon_type: can be set to 's', 'i', or left as None in order to select thermal synchrotron photons, injected photons, or all the photons in the simulation for analysis
	:param energy_range: has units of keV, can be left as None to choose photons of all energy ranges for analysis or it can be set to an array with [min energy, max energy] e.g. [1, 10] for 1 to 10 keV (limits inclusive)

	:return: returns the time binned quantities of luminosity, luminosity error, number of photons in each bin, the
			 average eenrgy of photons in each bin, polarization, the stokes parameters, polarization error and
			 polarization angle
	"""

	from read_process_files import calc_kislat_error
	from astropy.stats import bayesian_blocks

	if (units != 'cts/s') & (units != 'erg/s'):
		print('Wrong units')
		print('The only allowed units are: erg/s and cts/s')
	else:
		if h5:
			try:
				time,hnu,weight, indexes, s0, s1, s2, s3, comv_hnu, num_scatt=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) #comv_hnu
				pt = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True, usecols=[10], dtype='|S15').astype(str)
			except IndexError:
				time, hnu, weight, indexes, s0, s1, s2, s3 = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)

		else:
			try:
				time, hnu, weight, indexes = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)
			except ValueError:
				time, hnu, weight = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)



		# bayes_blocks_bins = a[1]
		test_t = np.arange(time_start, time_end, dt)  # before had 15 to 40
		new_bins = np.zeros(test_t.size)
		new_bins[:] = np.nan


		if use_Lcrit:
			i = 1
			count = 1
			j = 0
			new_bins[0] = test_t[0]
			while (i < test_t.size):
				print(i)
				if j == 0:
					if (i + 1 < test_t.size):
						val = lcur(simid, np.array([test_t[i], test_t[i + 1]]), units=units, h5=True, theta=theta, photon_type=photon_type, energy_range=energy_range)[0][0]
						print(val)
						if (np.round(test_t[i + 1] - test_t[i], decimals=2) >= dt_min) & (val > liso_c):
							print(test_t[i + 1], test_t[i], test_t[i + 1] - test_t[i])
							new_bins[count] = test_t[i + 1]
							count += 1
						else:
							j = i
					else:
						if (test_t[i] - test_t[i - 1] < dt_min):
							new_bins[count] = test_t[i - 1] + dt_min
						else:
							new_bins[count] = test_t[i]
				else:
					if (i + 1 < test_t.size):
						print('In else')
						val = lcur(simid, np.array([test_t[j], test_t[i + 1]]), units=units, h5=True,theta=theta, photon_type=photon_type, energy_range=energy_range)[0][0]
						print(val)
						if (np.round(test_t[i + 1] - test_t[i], decimals=2) >= dt_min) & (val > liso_c):
							print(test_t[i + 1], test_t[j], test_t[i + 1] - test_t[j])
							new_bins[count] = test_t[i + 1]
							count += 1
							j = 0
							print('j=0')

					else:
						if (test_t[i] - test_t[i - 1] < dt_min):
							new_bins[count] = test_t[i - 1] + dt_min
						else:
							new_bins[count] = test_t[i]
				i += 1


			new_bins_no_nan = new_bins[~np.isnan(new_bins)]
		else:
			data_bolo = lcur(simid, test_t, theta=np.double(simid.split('_')[-1]),
							   energy_range=energy_range, photon_type=photon_type)
			new_bins_no_nan=bayesian_blocks(test_t[~np.isnan(data_bolo[1])], data_bolo[0][~np.isnan(data_bolo[1])], sigma=data_bolo[1][~np.isnan(data_bolo[1])],fitness='measures')

		# fix last histogram bin
		if new_bins_no_nan[-1]< time_end:
			new_bins_no_nan[-1] = time_end

		if new_bins_no_nan[0] > time_start:
			new_bins_no_nan[0] = time_start


		t = new_bins_no_nan

		p = np.empty(t.size)*np.nan
		p_angle = np.empty(t.size)*np.nan
		l = np.empty(t.size)*np.nan
		q = np.empty(t.size)*np.nan
		u = np.empty(t.size)*np.nan
		v = np.empty(t.size)*np.nan
		perr = np.empty((t.size,2))*np.nan
		lc = np.zeros(t.size)#np.empty(t.size)*np.nan
		lce = np.empty(t.size)*np.nan
		ph_num = np.empty(t.size)*np.nan
		ph_avg_energy = np.empty(t.size)*np.nan
		weight_std=np.zeros(t.size)
		ns = np.zeros(t.size)*np.nan

		for i in range(t.size-1):
			# print(tmin[i], tmax[i])
			if photon_type==None:
				if energy_range == None:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)))
				else:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)) & (hnu>=energy_range[0]) & (hnu<=energy_range[1]))
			else:
				if energy_range == None:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)) & (pt==photon_type))
				else:
					jj = np.where((time >= t[i]) & (time < t[i+1]) & (~np.isnan(s0)) & (pt==photon_type) & (hnu>=energy_range[0]) & (hnu<=energy_range[1]))

			if jj[0].size > 0:
				if units == 'erg/s':
					lc[i] = np.sum(weight[jj] * hnu[jj] * 1.6e-9)/(t[i+1]-t[i])
				if units == 'cts/s':
					lc[i] = np.sum(weight[jj])/(t[i+1]-t[i])
				lce[i] = lc[i] / np.sqrt(jj[0].size)
				ph_num[i] = jj[0].size
				ph_avg_energy[i] = np.average(hnu[jj] * 1.6e-9, weights=weight[jj])  # in ergs
				weight_std[i]=(np.sqrt(np.mean(weight[jj] ** 2) / np.mean(weight[jj]) ** 2))
				ns[i] = np.average(num_scatt[jj], weights=weight[jj])
				if h5:
					# p[i]=np.average(np.sqrt(np.sum(s1[jj])**2+np.sum(s2[jj])**2)/np.sum(s0[jj]), weights=weight[jj])
					l[i] = np.average(s0[jj], weights=weight[jj])
					q[i] = np.average(s1[jj], weights=weight[jj])
					u[i] = np.average(s2[jj], weights=weight[jj])
					v[i] = np.average(s3[jj], weights=weight[jj])
					I = np.average(s0[jj], weights=weight[jj])
					p[i] = np.sqrt(q[i] ** 2 + u[i] ** 2) / I
					p_angle[i] = (
								0.5 * np.arctan2(u[i], q[i]) * 180 / np.pi)


					#from Kislat
					l[i], q[i], u[i], v[i], p[i], p_angle[i], perr[i,0], perr[i,1] = calc_kislat_error(s0[jj], s1[jj], s2[jj], s3[jj], weight[jj])


		if sim_dims == 2:
			factor = 2 * np.pi * (np.cos((theta - dtheta / 2.) * np.pi / 180) - np.cos((theta + dtheta / 2.) * np.pi / 180))
		elif sim_dims == 3:
			factor = (dphi * np.pi / 180) * (
						np.cos((theta - dtheta / 2.) * np.pi / 180) - np.cos((theta + dtheta / 2.) * np.pi / 180))


		lc =  lc  / factor
		lce =  lce  / factor
		return lc, lce, ph_num, ph_avg_energy, p, l, q, u, v, perr, p_angle, ns, t


def spex(simid,numin,numax,tmin,tmax,units='erg/s', h5=True, photon_type=None, calc_pol=False):
	"""
	Produces spectra of phtons detected within any time interval

	:param simid: a string of the event file base name of the event file created in event_h5 (everything but the .evt)
	:param numin: array of energy values of the left most cutoff of the energy bins in the spectrum in keV
	:param numax: array of energy values of the right most cutoff of the energy bins in the spectrum in keV
	:param tmin: minimum of time bin that we are interested in collecting photons in to analyze
	:param tmax: max of time bin that we are interested in collecting photons in to analyze
	:param units: a string of the units of the spectrum that will be produced (erg/s or cts/s), each unit is then divided by the width of the enrgy bin in keV
	:param h5: specify if the format of the MCRaT output files is hdf5 files or not (denotes if using an old format or
			   a newer format of saving files)
	:param photon_type: can be set to 's', 'i', or left as None in order to select thermal synchrotron photons, injected photons, or all the photons in the simulation for analysis
	:param energy_range: has units of keV, can be left as None to choose photons of all energy ranges for analysis or it can be set to an array with [min energy, max energy] e.g. [1, 10] for 1 to 10 keV (limits inclusive)

	:return: returns the spectrum with the specified units, for the photons in each energy bin
	"""
	from read_process_files import calc_kislat_error

	if (units!='cts/s')&(units!='erg/s'):
		print( 'Wrong units')
		print( 'The only allowed units are: erg/s and cts/s')
	else:
		if h5:
			try:
				#time,hnu,weight, indexes, s0, s1, s2, s3=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)#comv_hnu
				time,hnu,weight, indexes, s0, s1, s2, s3, comv_hnu, num_scatt=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) #comv_hnu
				pt = np.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True, usecols=[10], dtype='|S15').astype(str)
			except IndexError:
				time,hnu,weight, indexes, s0, s1, s2, s3=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
		else:
			try:
				time,hnu,weight, indexes=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			except ValueError:
				time,hnu,weight=np.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
		
		dnu=numax-numin
		sp=np.zeros(numin.size)
		spe=np.zeros(numin.size)
		p = np.empty(numin.size)*np.nan
		p_angle = np.empty(numin.size)*np.nan
		l = np.empty(numin.size)*np.nan
		q = np.empty(numin.size)*np.nan
		u = np.empty(numin.size)*np.nan
		v = np.empty(numin.size)*np.nan
		perr = np.zeros((numin.size,2))*np.nan
		ns = np.zeros(numin.size)

		goodones=np.zeros(numin.size)
		for i in range(numin.size):
			if photon_type==None:
				jj = np.where((time >= tmin) & (time < tmax) & (hnu >= numin[i]) & (hnu < numax[i]) & (~np.isnan(s0)))
			else:
				jj = np.where((time >= tmin) & (time < tmax) & (hnu >= numin[i]) & (hnu < numax[i]) & (~np.isnan(s0)) & (pt == photon_type))

			if jj[0].size>0:
				if units=='erg/s':
					sp[i]=(1.6e-9)*np.sum(weight[jj]*hnu[jj])/dnu[i]
				else:
					sp[i]=np.sum(weight[jj])/dnu[i]
				spe[i]=sp[i]/np.sqrt(jj[0].size)

				l[i], q[i], u[i], v[i], p[i], p_angle[i], perr[i, 0], perr[i, 1] = calc_kislat_error(s0[jj], s1[jj],
																									 s2[jj], s3[jj],
																									 weight[jj])
				ns[i]=np.average(num_scatt[jj], weights=weight[jj])
				if (np.isnan(perr[i, 0]) & (jj[0].size==811)):
					print('this is nan')
				if jj[0].size>10:
					goodones[i]=jj[0].size

	return sp,spe,goodones, l, q, u, v, p, p_angle, perr, ns

def readanddecimate(fnam, inj_radius):
	"""
	Legacy code from the python version of MCRaT to read and process FLASH files

	:param fnam: string of directory and file name of the FLASH file that will be loaded and processed
	:param inj_radius: The radius that photons are injected in cm
	:return: returns FLASH values of various quantities at cells
	"""
	import tables as t
	file = t.open_file(fnam)
	print( '>> mc.py: Reading positional, density, pressure, and velocity informationp...')
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
	print(szx.shape, vx.shape)


	print ('>> mc.py: Creating the full x and y arrays...')
	xx = np.zeros(vx.shape)
	yy = np.zeros(vx.shape)
	x1 = np.array([-7., -5, -3, -1, 1, 3, 5, 7]) / 16.
	x2 = np.empty([8, 8])
	y2 = np.empty([8, 8])
	szxx=np.zeros(vx.shape)
	szyy=np.zeros(vx.shape)
	szxx[:,0,:,:]=szx[:, np.newaxis, np.newaxis]
	szyy[:,0,:,:]=szy[:, np.newaxis, np.newaxis]
	for ii in range(0, 8, 1):
		y2[:, ii] = np.array(x1)
		x2[ii, :] = np.array(x1)
	for ii in range(0, x.size):
		xx[ii, 0, :, :] = np.array(x[ii] + szx[ii] * x2)
		yy[ii, 0, :, :] = np.array(y[ii] + szy[ii] * y2)

	print ('>> mc.py: Selecting good node types (=1)...')
	nty = file.get_node('/', 'node type')
	nty = nty.read()
	file.close()
	jj = np.where(nty == 1)
	xx = np.array(xx[jj, 0, :, :]) * 1e9
	#	yy=np.array(yy[jj,0,:,:]+1) this takes care of the fact that y starts at 1e9 and not at 0
	yy = np.array(yy[jj, 0, :, :]) * 1e9
	szx=1e9*np.array(szxx[jj, 0, :, :])/8
	szy=1e9*np.array(szyy[jj, 0, :, :])/8
	vx = np.array(vx[jj, 0, :, :])
	vy = np.array(vy[jj, 0, :, :])
	dens = np.array(dens[jj, 0, :, :])
	pres = np.array(pres[jj, 0, :, :])

	print( '>> mc.py: Reshaping arrays...')
	xx = np.reshape(xx, xx.size)
	vx = np.reshape(vx, xx.size)
	yy = np.reshape(yy, yy.size)
	vy = np.reshape(vy, yy.size)
	szx=np.reshape(szx, yy.size)
	szy=np.reshape(szy, yy.size)
	gg = 1. / np.sqrt(1. - (vx ** 2 + vy ** 2))
	dd = np.reshape(dens, dens.size)
	dd_lab = dd * gg
	rr = np.sqrt(xx ** 2 + yy ** 2)
	tt = np.arctan2(xx, yy)
	pp = np.reshape(pres, pres.size)
	del pres, dens, x, y

	jj = np.where(rr > (0.95 * inj_radius))
	xx = xx[jj]
	vx = vx[jj]
	yy = yy[jj]
	vy = vy[jj]
	gg = gg[jj]
	dd = dd[jj]
	dd_lab = dd_lab[jj]
	rr = rr[jj]
	tt = tt[jj]
	pp = pp[jj]
	szx=szx[jj]
	szy=szy[jj]

	return xx, yy, vx, vy, gg, dd, dd_lab, rr, tt, pp, szx, szy
