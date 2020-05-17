import numpy as n

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
		xdum = n.random.rand(10000) * (1 + 100 * theta)
		betax = n.sqrt(1. - 1. / xdum ** 2)
		ydum = n.random.rand(10000) / 2.
		funx = xdum ** 2 * betax / special.kn(2, 1. / theta) * n.exp(-xdum / theta)
		jj = n.where(ydum <= funx)
		thegamma = xdum[jj[0][0]]
	else:
		sigma = n.sqrt(kB * T / me)
		v_el = n.random.normal(scale=sigma, size=3) / c_light
		g_el = 1. / n.sqrt(1. - n.sum((v_el) ** 2))
		thegamma = g_el
	# print 'Temperature (1e7 K)',T/1e7
	#print ('thegamma',thegamma)
	thebeta = n.sqrt(1. - 1. / thegamma ** 2)
	thedum = n.random.rand(100) * n.pi
	ydum = n.random.rand(100) * 1.3
	yy = n.sin(thedum) * (1 - thebeta * n.cos(thedum))
	jj = n.where(ydum <= yy)
	thetheta = thedum[jj[0][0]]
	# print 'thetheta',thetheta*180/n.pi
	thephi = n.random.rand() * 2 * n.pi
	P_el = me * c_light * thegamma * n.array([1, thebeta * n.cos(thetheta), thebeta * n.sin(thetheta) * n.sin(thephi),
											  thebeta * n.sin(thetheta) * n.cos(thephi)])
	# print 'P_el',P_el
	theta_ph = n.arctan2(n.sqrt(P_ph[2] ** 2 + P_ph[3] ** 2), P_ph[1])
	phi_ph = n.arctan2(P_ph[2], P_ph[3])
	# print theta_ph,phi_ph
	#	First rotate around x-axis to get phi out of the way
	Rot2 = n.zeros([3, 3])
	Rot2[1, 1] = 1.
	Rot2[0, 0] = n.cos(theta_ph)
	Rot2[2, 2] = n.cos(theta_ph)
	Rot2[0, 2] = -n.sin(theta_ph)
	Rot2[2, 0] = n.sin(theta_ph)
	p_el_prime = n.dot(Rot2, P_el[1:])
	P_el_prime = P_el.copy()
	P_el_prime[1:] = p_el_prime
	Rot1 = n.zeros([3, 3])
	Rot1[0, 0] = 1.
	Rot1[1, 1] = n.cos(-phi_ph)
	Rot1[2, 2] = n.cos(-phi_ph)
	Rot1[1, 2] = -n.sin(-phi_ph)
	Rot1[2, 1] = n.sin(-phi_ph)
	p_el_second = n.dot(Rot1, P_el_prime[1:])
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

	indexes=n.where((boost**2).sum(axis=0)>0)[0]
	zero_beta_idx=n.where((boost**2).sum(axis=0)==0)[0]
	Lambda1=n.zeros([4,4,indexes.size])

	#fill in matrix for each photon
	beta=n.sqrt((boost[:,indexes]**2).sum(axis=0))
	gamma = 1. / n.sqrt(1. - beta ** 2)
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
	result=n.einsum('ijk,jk->ik', Lambda1, P_ph[:,indexes])
	if (zero_beta_idx.size>0):
		result=n.insert(result, zero_beta_idx, P_ph[:,zero_beta_idx], axis=1)

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
		beta = n.sqrt((boost ** 2).sum())
		gamma = 1. / n.sqrt(1. - beta ** 2)
		Lambda1 = n.empty([4, 4])
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

		P_ph_prime = n.dot(Lambda1, P_ph)

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
		not_norm=n.where(P[0,:]**2 != n.linalg.norm(P[1:,:], axis=0)**2)[0]
		P[1:,not_norm]=(P[1:,not_norm]/n.linalg.norm(P[1:,not_norm], axis=0))*P[0,not_norm]
	else:
		if (P[0]**2 != n.linalg.norm(P[1:])**2):
			#correct the 4 momentum
			P[1:]=(P[1:]/n.linalg.norm(P[1:]))*P[0]

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
	#	moving electron. Takes the 4-momenta as inputs
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
	phi = n.arctan2(P_ph_prime[2], P_ph_prime[1])
	Rot1 = n.zeros([3, 3])
	Rot1[2, 2] = 1.
	Rot1[0, 0] = n.cos(-phi)
	Rot1[1, 1] = n.cos(-phi)
	Rot1[0, 1] = -n.sin(-phi)
	Rot1[1, 0] = n.sin(-phi)
	p_ph_second = n.dot(Rot1, P_ph_prime[1:])

	#y_tilde_rot_1=n.dot(-Rot1, y_tilde_1) #rotate the stokes plane as well
	#x_tilde_rot_1=n.dot(-Rot1, x_tilde_1)

	P_ph_second = P_ph_prime.copy()
	P_ph_second[1:] = p_ph_second
	P_ph_second[2] = 0
	P_el_second = P_el_prime.copy()

	#print(n.arccos(n.dot(P_ph_second[1:], y_tilde_rot_1)/n.sqrt(n.dot(P_ph_second[1:],P_ph_second[1:])))*180/n.pi, n.arccos(n.dot(P_ph_second[1:], x_tilde_rot_1)/n.sqrt(n.dot(P_ph_second[1:],P_ph_second[1:])))*180/n.pi)


	# 2.2 now we rotate around y to bring it all along x
	phi2 = n.arctan2(p_ph_second[2], p_ph_second[0])
	Rot2 = n.zeros([3, 3])
	Rot2[1, 1] = 1.
	Rot2[0, 0] = n.cos(-phi2)
	Rot2[2, 2] = n.cos(-phi2)
	Rot2[0, 2] = -n.sin(-phi2)
	Rot2[2, 0] = n.sin(-phi2)
	p_ph_third = n.dot(Rot2, P_ph_second[1:])

	#y_tilde_rot_2=n.dot(Rot2, y_tilde_rot_1)
	#x_tilde_rot_2=n.dot(Rot2, x_tilde_rot_1)

	P_ph_third = P_ph_second.copy()
	P_ph_third[1:] = p_ph_third
	P_ph_third[1] = P_ph_third[0]
	P_ph_third[3] = 0
	P_el_third = P_el_second.copy()

	#print(n.arccos(n.dot(P_ph_third[1:], y_tilde_rot_2)/n.sqrt(n.dot(P_ph_third[1:],P_ph_third[1:])))*180/n.pi, n.arccos(n.dot(P_ph_third[1:], x_tilde_rot_2)/n.sqrt(n.dot(P_ph_third[1:],P_ph_third[1:])))*180/n.pi)

	#rotate y_yilde_rot_2 such that it is aligned with the z axis
	#t=n.arccos(n.dot([0,0,1], y_tilde_rot_2))
	#Rot_stokes = n.zeros([3, 3])
	#Rot_stokes[0, 0] = 1.
	#Rot_stokes[1, 1] = n.cos(t)
	#Rot_stokes[2, 2] = n.cos(t)
	#Rot_stokes[1, 2] = -n.sin(t)
	#Rot_stokes[2, 1] = n.sin(t)

	#y_tilde_rot_3=n.dot(Rot_stokes, y_tilde_rot_2)
	#x_tilde_rot_3=n.dot(Rot_stokes, x_tilde_rot_2)


	#
	# Third we perform the scattering
	#
	# 3.1 we generate a phi and a theta angles
	phi3 = n.random.rand() * 2 * n.pi
	dumx = n.random.rand(100) * n.pi
	dumy = n.random.rand(100) * 1.09
	dumyy = (1. + n.cos(dumx) ** 2) * n.sin(dumx)
	jj = n.where(dumyy >= dumy)
	theta3 = dumx[jj[0][0]]
	# 3.2 compute new 4-momenta of electron and photon
	P_ph_fourth = n.zeros(4)
	P_ph_fourth[0] = P_ph_third[0] / (1. + P_ph_third[0] / me / c * (1. - n.cos(theta3)))
	P_ph_fourth[3] = P_ph_fourth[0] * n.sin(theta3) * n.cos(phi3)
	P_ph_fourth[2] = P_ph_fourth[0] * n.sin(theta3) * n.sin(phi3)
	P_ph_fourth[1] = P_ph_fourth[0] * n.cos(theta3)
	#	print 'norm of ph 4',P_ph_fourth[0]**2-n.sum(P_ph_fourth[1:]**2)
	P_el_fourth = P_el_third + P_ph_third - P_ph_fourth  ###
	if n.sqrt(n.sum((P_el_fourth[1:] / P_el_fourth[0]) ** 2)) > 1:
		print('Electron 4 momentum  not normalized')
		exit()
	#
	# Fourth we rotate back into the original boosted frame
	#
	Rot3 = n.zeros([3, 3])
	Rot3[1, 1] = 1.
	Rot3[0, 0] = n.cos(-phi2)
	Rot3[2, 2] = n.cos(-phi2)
	Rot3[0, 2] = n.sin(-phi2)
	Rot3[2, 0] = -n.sin(-phi2)
	p_ph_fifth = n.dot(Rot3, P_ph_fourth[1:])
	P_ph_fifth = P_ph_fourth.copy()
	P_ph_fifth[1:] = p_ph_fifth
	p_el_fifth = n.dot(Rot3, P_el_fourth[1:])
	P_el_fifth = P_el_fourth.copy()
	P_el_fifth[1:] = p_el_fifth
	if n.sqrt(n.sum((P_el_fifth[1:] / P_el_fifth[0]) ** 2)) > 1:
		print('Electron 4 momentum  not normalized')
		exit()

	Rot4 = n.zeros([3, 3])
	Rot4[2, 2] = 1.
	Rot4[0, 0] = n.cos(-phi)
	Rot4[1, 1] = n.cos(-phi)
	Rot4[0, 1] = n.sin(-phi)
	Rot4[1, 0] = -n.sin(-phi)
	p_ph_sixth = n.dot(Rot4, P_ph_fifth[1:])
	P_ph_sixth = P_ph_fifth.copy()
	P_ph_sixth[1:] = p_ph_sixth
	p_el_sixth = n.dot(Rot4, P_el_fifth[1:])
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
	Place holder function to conduct a synthetic observation of MCRaT simulated outflow using a 3D hydro simulation.

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
	theta = theta_deg * n.pi / 180  # angle between jet axis and line of sight
	dtheta = dtheta_deg * n.pi / 180  # acceptance
	phi=phi_deg* np.pi / 180
	foutn='events.dat'
	fout=n.str(sim_type)+'_'+"%.1e"% r_obs+'_'+n.str(theta_deg)+'.evt'

	#comment out when using mc.par file for weight
	weight = n.loadtxt(path + 'mcdata_PW.dat')
	print( weight.shape)

	c_light = 2.99792458e10
	i = lastfile
	P0 = n.loadtxt(path + 'mcdata_' + str(i) + '_P0.dat')
	sizze = P0.shape
	print(sizze, P0.size)
	P1 = n.loadtxt(path + 'mcdata_' + str(i) + '_P1.dat')
	P2 = n.loadtxt(path + 'mcdata_' + str(i) + '_P2.dat')
	P3 = n.loadtxt(path + 'mcdata_' + str(i) + '_P3.dat')
	R0 = n.loadtxt(path + 'mcdata_' + str(i) + '_R0.dat')
	R1 = n.loadtxt(path + 'mcdata_' + str(i) + '_R1.dat')
	R2 = n.loadtxt(path + 'mcdata_' + str(i) + '_R2.dat')
	P0 = n.reshape(P0, P0.size)
	P1 = n.reshape(P1, P0.size)
	P2 = n.reshape(P2, P0.size)
	P3 = n.reshape(P3, P0.size)
	R1 = n.reshape(R1, P0.size)
	R2 = n.reshape(R2, P0.size)
	R3 = n.reshape(R3, P0.size)
	RR = n.sqrt(R0 ** 2 + R1 ** 2 + R2 ** 2)  # radius of propagation

	if not riken_switch:
		theta_pos = n.arccos(R2/RR)  # angle between position vector and polar axis
		phi_pos=n.arctan2(R1, R0)
	else:
		theta_pos = n.arccos(R1/RR)  # angle between position vector and y axis for RIKEN Hydro
		phi_pos=n.arctan2(R2, R0)

	theta_rel = theta_pos - theta  # angle between position vector and line of sight
	RR_prop = RR * n.cos(theta_rel)

	#comment out when using mc.par file for weight
	weight=n.reshape(weight[:sizze[0]],P0.size)

	#read photon weight from mc.par file
	#weight=n.ones(P0.size)*(n.genfromtxt(path+'mc.par')[-3])
	#weight = n.reshape(weight, P0.size)

	if not riken_switch:
		theta_pho = n.arccos(P3/n.sqrt(P1 ** 2 + P2 ** 2 + P3**2))  # angle between velocity vector and polar axis
		phi_pho = n.arctan2(P2, P1)
	else:
		theta_pho = n.arccos(P2/n.sqrt(P1 ** 2 + P2 ** 2  + P3**2))  # angle between velocity vector and y axis
		phi_pho = n.arctan2(P3, P1)

	# dtheta applies for theta and phi directions, make sure light curve function also has correct factors for the 3D case
	jj = n.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.)  & (phi_pho >= phi - dtheta / 2.) & (phi_pho < phi + dtheta / 2.) & (RR_prop >= r_obs))
	print ('accepted photons ', jj[0].size)
	nnn = jj[0].size
	if not riken_switch:
		fps=5.0
	else:
		fps=10.0


	tnow = lastfile / fps
	dr = RR_prop[jj] - r_obs
	vel_proj = c_light * n.cos(theta_pho[jj])
	dt = dr / vel_proj
	crossing_time = tnow - dt - r_obs / c_light
	hnukeV = P0[jj] * 3e10 / 1.6e-9
	outarr = n.zeros([nnn, 3])
	outarr[:, 0] = crossing_time
	outarr[:, 1] = hnukeV
	outarr[:, 2] = weight[jj]
	n.savetxt(path+foutn,outarr)
	n.savetxt('EVENT_FILES/'+fout,outarr)

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
	theta = theta_deg * n.pi / 180  # angle between jet axis and line of sight
	dtheta = dtheta_deg * n.pi / 180  # acceptance
	foutn='events.dat'
	if not withintheta_deg:
		fout=n.str(sim_type)+'_'+"%.1e"% r_obs+'_'+n.str(theta_deg)+'.evt'
	else:
		fout=n.str(sim_type)+'_within_theta_'+"%.1e"% r_obs+'_'+n.str(theta_deg)+'.evt'

	#comment out when using mc.par file for weight
	weight = n.loadtxt(path + 'mcdata_PW.dat')
	print( weight.shape)

	c_light = 2.99792458e10
	i = lastfile
	P0 = n.loadtxt(path + 'mcdata_' + str(i) + '_P0.dat')
	sizze = P0.shape
	print(sizze, P0.size)
	P1 = n.loadtxt(path + 'mcdata_' + str(i) + '_P1.dat')
	P2 = n.loadtxt(path + 'mcdata_' + str(i) + '_P2.dat')
	P3 = n.loadtxt(path + 'mcdata_' + str(i) + '_P3.dat')
	R1 = n.loadtxt(path + 'mcdata_' + str(i) + '_R0.dat')
	R2 = n.loadtxt(path + 'mcdata_' + str(i) + '_R1.dat')
	R3 = n.loadtxt(path + 'mcdata_' + str(i) + '_R2.dat')
	P0 = n.reshape(P0, P0.size)
	P1 = n.reshape(P1, P0.size)
	P2 = n.reshape(P2, P0.size)
	P3 = n.reshape(P3, P0.size)
	R1 = n.reshape(R1, P0.size)
	R2 = n.reshape(R2, P0.size)
	R3 = n.reshape(R3, P0.size)
	RR = n.sqrt(R1 ** 2 + R2 ** 2 + R3 ** 2)  # radius of propagation
	theta_pos = n.arctan2(n.sqrt(R1 ** 2 + R2 ** 2), R3)  # angle between position vector and polar axis
	theta_rel = theta_pos - theta  # angle between position vector and line of sight
	RR_prop = RR * n.cos(theta_rel)

	#comment out when using mc.par file for weight
	weight=n.reshape(weight[:sizze[0]],P0.size)


	theta_pho = n.arctan2(n.sqrt(P1 ** 2 + P2 ** 2), P3)  # angle between velocity vector and polar axis

	if not withintheta_deg:
		jj = n.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs))
	else:
		jj = n.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs) & (theta_pos >= theta - dtheta / 2.) & (theta_pos < theta + dtheta / 2.))

	print ('accepted photons ', jj[0].size)
	nnn = jj[0].size
	if not riken_switch:
		fps=5.0
	else:
		fps=10.0


	tnow = lastfile / fps
	dr = RR_prop[jj] - r_obs
	vel_proj = c_light * n.cos(theta_pho[jj]-theta)
	dt = dr / vel_proj
	crossing_time = tnow - dt - r_obs / c_light
	hnukeV = P0[jj] * 3e10 / 1.6e-9
	outarr = n.zeros([nnn, 4])
	outarr[:, 0] = crossing_time
	outarr[:, 1] = hnukeV
	outarr[:, 2] = weight[jj]
	outarr[:, 3] =  jj[0] #save indexes of the photons in the whole set of photons
	if save_event:
		#n.savetxt(path+foutn,outarr)
		n.savetxt('EVENT_FILES/'+fout,outarr)
		
	return  crossing_time, hnukeV, weight[jj], jj[0]

def event_h5(r_obs, theta_deg, dtheta_deg, path, lastfile, sim_type, fps=5, read_comv=False, read_stokes=False, append=False):
	"""
	Function to collect MCRaT photons in space that will be observed. Saves event files in EVENT_FILES/ directory that
	must be created before calling this function.

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
	:param append: switch to denote if the function should append the extracted data to a pre-existing event file
	:return: returns (N,8) array where N is the number of photons 'detected'. The array contains:
	          the crossing time of the photon, the lab energy in keV, the weight, the index in the list of photons,
	          the stokes parameters and the comoving energy in keV
	"""
	from read_process_files import read_mcrat_h5
	import h5py as h5
	
	c_light = 2.99792458e10
	
	data=read_mcrat_h5(path+"mcdata_"+n.str_(lastfile), read_comv=read_comv, read_stokes=read_stokes)
	if read_comv and read_stokes:
		PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3=data
	elif read_comv and not read_stokes:
		PW, NS, P0, P1, P2, P3, R1, R2, R3, COMV_P0, COMV_P1, COMV_P2, COMV_P3=data
		S0, S1, S2, S3 = n.zeros((4, n.size(P0))) * n.nan
	elif not read_comv and read_stokes:
		PW, NS, P0, P1, P2, P3, R1, R2, R3, S0, S1, S2, S3=data
		COMV_P0, COMV_P1, COMV_P2, COMV_P3 = n.zeros((4, n.size(P0))) * n.nan
	else:
		PW, NS, P0, P1, P2, P3, R1, R2, R3=data
		S0, S1, S2, S3, COMV_P0, COMV_P1, COMV_P2, COMV_P3 = n.zeros((8, n.size(P0))) * n.nan



	print('Total Number of Photons: ', n.size(P0), P0)

	weight = PW
		
	theta = theta_deg * n.pi / 180  # angle between jet axis and line of sight
	dtheta = dtheta_deg * n.pi / 180  # acceptance
	foutn='events.dat'
	fout=n.str(sim_type)+'_'+"%.2e"% r_obs+'_'+n.str(theta_deg)+'.evt'
	#fout = n.str(sim_type) + '_' + "%.2e" % r_obs + '_' + "%0.15lf" %(theta_deg) + '.evt' #thsi was used for the lundman comparisons
	
	RR = n.sqrt(R1 ** 2 + R2 ** 2 + R3 ** 2)  # radius of propagation
	theta_pos = n.arctan2(n.sqrt(R1 ** 2 + R2 ** 2), R3)  # angle between position vector and polar axis
	theta_rel = theta_pos - theta  # angle between position vector and line of sight
	RR_prop = RR * n.cos(theta_rel)

	theta_pho = n.arctan2(n.sqrt(P1 ** 2 + P2 ** 2), P3)  # angle between velocity vector and polar axis

	print(n.histogram(theta_pho, bins=50)[0], n.histogram(theta_pho, bins=50)[1]*180/n.pi)

	jj = n.where((theta_pho >= theta - dtheta / 2.) & (theta_pho < theta + dtheta / 2.) & (RR_prop >= r_obs) ) #& (S1 != 0) &  (S2 != 0) added last two to properly calc polarization in test cases
	print ('accepted photons ', jj[0].size)
	nnn = jj[0].size



	tnow = lastfile / fps
	dr = RR_prop[jj] - r_obs
	vel_proj = c_light * n.cos(theta_pho[jj]-theta)
	dt = dr / vel_proj
	crossing_time = tnow - dt - r_obs / c_light
	hnukeV = P0[jj] * 3e10 / 1.6e-9
	comv_hnukeV= COMV_P0[jj] * 3e10 / 1.6e-9
	outarr = n.zeros([nnn, 9])
	outarr[:, 0] = crossing_time
	outarr[:, 1] = hnukeV
	outarr[:, 2] = weight[jj]
	outarr[:, 3] =  jj[0] #save indexes of the photons in the whole set of photons
	outarr[:, 4] = S0[jj]
	outarr[:, 5] = S1[jj]
	outarr[:, 6] = S2[jj]
	outarr[:, 7] = S3[jj]
	outarr[:, 8] = comv_hnukeV

	#n.savetxt(path+foutn,outarr)
	if not append:
		n.savetxt('EVENT_FILES/'+fout,outarr)
	else:
		with open('EVENT_FILES/'+fout, 'ba') as f:
			n.savetxt(f, outarr)


def lcur(simid, t, units='erg/s', theta=1., dtheta=1., phi=0, dphi=1, sim_dims=2, h5=True):
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
	:return: returns the time binned quantities of luminosity, luminosity error, number of photons in each bin, the
			 average eenrgy of photons in each bin, polarization, the stokes parameters, polarization error and
			 polarization angle
	"""
	if (units != 'cts/s') & (units != 'erg/s'):
		print( 'Wrong units')
		print( 'The only allowed units are: erg/s and cts/s')
	else:
		if h5:
			try:
				time,hnu,weight, indexes, s0, s1, s2, s3, comv_hnu=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			except ValueError:
				time,hnu,weight, s0, s1, s2, s3=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			
		else:
			try:
				time,hnu,weight, indexes=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			except ValueError:
				time,hnu,weight=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)

		p=n.zeros(t.size)
		p_angle = n.zeros(t.size)
		l=n.zeros(t.size)
		q=n.zeros(t.size)
		u=n.zeros(t.size)
		v=n.zeros(t.size)
		perr=n.zeros((t.size,2))
		lc = n.zeros(t.size)
		lce = n.zeros(t.size)
		ph_num=n.zeros(t.size)
		ph_avg_energy=n.zeros(t.size)
		dt = t[1] - t[0]
		tmin = t #- dt / 2.
		tmax = t + dt #/ 2.
		for i in range(t.size):
			#print(tmin[i], tmax[i])
			jj = n.where((time >= tmin[i]) & (time < tmax[i]) & (~n.isnan(s0)))
			if jj[0].size > 0:
				if units == 'erg/s':
					lc[i] = n.sum(weight[jj] * hnu[jj] * 1.6e-9) /lc_factor
				if units == 'cts/s':
					lc[i] = n.sum(weight[jj])
				lce[i] = lc[i] / n.sqrt(jj[0].size)
				ph_num[i]=jj[0].size
				ph_avg_energy[i]=n.average(hnu[jj] * 1.6e-9, weights=weight[jj]) #in ergs
				if h5:
					#p[i]=n.average(n.sqrt(n.sum(s1[jj])**2+n.sum(s2[jj])**2)/n.sum(s0[jj]), weights=weight[jj])
					l[i]=n.average(s0[jj], weights=weight[jj])
					q[i]=n.average(s1[jj], weights=weight[jj])
					u[i]=n.average(s2[jj], weights=weight[jj])
					v[i]=n.average(s3[jj], weights=weight[jj])
					I=n.average(s0[jj],weights=weight[jj])
					#print(u[i])
					p[i]=n.sqrt(q[i]**2+u[i]**2)#/I
					p_angle[i]=(0.5*n.arctan2(u[i],q[i])*180/n.pi)#gives angle between +90 degrees and -90 degrees

					#from Kislat eqn 36 and 37
					mu=1
					perr[i,:]= n.sqrt(2.0-p[i]**2*mu**2) / n.sqrt((jj[0].size-1)*mu**2), (180/n.pi)*1/(mu*p[i]*n.sqrt(2*(jj[0].size-1)))

		#calulate d \Omega
		if sim_dims==2:
			factor = 2 * n.pi * (n.cos((theta - dtheta / 2.) * n.pi / 180) - n.cos((theta + dtheta / 2.) * n.pi / 180))
		elif sim_dims==3:
			factor = (dphi* n.pi / 180) * (n.cos((theta - dtheta / 2.) * n.pi / 180) - n.cos((theta + dtheta / 2.) * n.pi / 180))


		lc =  lc / dt / factor
		lce =  lce / dt / factor
		return lc, lce, ph_num, ph_avg_energy, p, l, q, u, v, perr, p_angle


def lcur_var_t(simid, time_start, time_end, dt, dt_min, liso_c = 1e50, units='erg/s', theta=1., dtheta=1., phi=0, dphi=1, sim_dims=2, h5=True):
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
	:return: returns the time binned quantities of luminosity, luminosity error, number of photons in each bin, the
			 average eenrgy of photons in each bin, polarization, the stokes parameters, polarization error and
			 polarization angle
	"""

	if (units != 'cts/s') & (units != 'erg/s'):
		print('Wrong units')
		print('The only allowed units are: erg/s and cts/s')
	else:
		if h5:
			try:
				time, hnu, weight, indexes, s0, s1, s2, s3 = n.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)
			except ValueError:
				time, hnu, weight, s0, s1, s2, s3 = n.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)

		else:
			try:
				time, hnu, weight, indexes = n.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)
			except ValueError:
				time, hnu, weight = n.loadtxt('EVENT_FILES/' + simid + '.evt', unpack=True)



		# bayes_blocks_bins = a[1]
		test_t = n.arange(time_start, time_end, dt)  # before had 15 to 40
		new_bins = n.zeros(test_t.size)
		new_bins[:] = n.nan

		i = 1
		count = 1
		j = 0
		new_bins[0] = test_t[0]
		while (i < test_t.size):
			print(i)
			if j == 0:
				if (i + 1 < test_t.size):
					val = lcur(simid, n.array([test_t[i], test_t[i + 1]]), units=units, h5=True)[0][0]
					print(val)
					if (n.round(test_t[i + 1] - test_t[i], decimals=2) >= dt_min) & (val > liso_c):
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
					val = lcur(simid, n.array([test_t[j], test_t[i + 1]]), units=units, h5=True)[0][0]
					print(val)
					if (n.round(test_t[i + 1] - test_t[i], decimals=2) >= dt_min) & (val > liso_c):
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

		new_bins_no_nan = new_bins[~n.isnan(new_bins)]

		# fix last histogram bin
		# new_bins_no_nan[-1] = new_bins_no_nan[-2] + 0.2

		t = new_bins_no_nan

		p = n.empty(t.size)*n.nan
		p_angle = n.empty(t.size)*n.nan
		l = n.empty(t.size)*n.nan
		q = n.empty(t.size)*n.nan
		u = n.empty(t.size)*n.nan
		v = n.empty(t.size)*n.nan
		perr = n.empty((t.size,2))*n.nan
		lc = n.empty(t.size)*n.nan
		lce = n.empty(t.size)*n.nan
		ph_num = n.empty(t.size)*n.nan
		ph_avg_energy = n.empty(t.size)*n.nan

		for i in range(t.size-1):
			# print(tmin[i], tmax[i])
			jj = n.where((time >= t[i]) & (time < t[i+1]) & (~n.isnan(s0)))
			if jj[0].size > 0:
				if units == 'erg/s':
					lc[i] = n.sum(weight[jj] * hnu[jj] * 1.6e-9)/(t[i+1]-t[i])
				if units == 'cts/s':
					lc[i] = n.sum(weight[jj])/(t[i+1]-t[i])
				lce[i] = lc[i] / n.sqrt(jj[0].size)
				ph_num[i] = jj[0].size
				ph_avg_energy[i] = n.average(hnu[jj] * 1.6e-9, weights=weight[jj])  # in ergs
				if h5:
					# p[i]=n.average(n.sqrt(n.sum(s1[jj])**2+n.sum(s2[jj])**2)/n.sum(s0[jj]), weights=weight[jj])
					l[i] = n.average(s0[jj], weights=weight[jj])
					q[i] = n.average(s1[jj], weights=weight[jj])
					u[i] = n.average(s2[jj], weights=weight[jj])
					v[i] = n.average(s3[jj], weights=weight[jj])
					I = n.average(s0[jj], weights=weight[jj])
					p[i] = n.sqrt(q[i] ** 2 + u[i] ** 2) / I
					p_angle[i] = (
								0.5 * n.arctan2(u[i], q[i]) * 180 / n.pi)


					#from Kislat eqn 36 and 37
                    mu=1
                    perr[i,:]= n.sqrt(2.0-p[i]**2*mu**2) / n.sqrt((jj[0].size-1)*mu**2), (180/n.pi)*1/(mu*p[i]*n.sqrt(2*(jj[0].size-1)))


		if sim_dims == 2:
			factor = 2 * n.pi * (n.cos((theta - dtheta / 2.) * n.pi / 180) - n.cos((theta + dtheta / 2.) * n.pi / 180))
		elif sim_dims == 3:
			factor = (dphi * n.pi / 180) * (
						n.cos((theta - dtheta / 2.) * n.pi / 180) - n.cos((theta + dtheta / 2.) * n.pi / 180))


		lc =  lc  / factor
		lce =  lce  / factor
		return lc, lce, ph_num, ph_avg_energy, p, l, q, u, v, perr, p_angle, t


def spex(simid,numin,numax,tmin,tmax,units='erg/s', h5=True):
	"""
	Produces spectra of phtons detected within any time interval

	:param simid: a string of the event file base name of the event file created in event_h5 (everything but the .evt)
	:param numin: array of energy values of the left most cutoff of the energy bins in the spectrum in keV
	:param numax: array of energy values of the right most cutoff of the energy bins in the spectrum in keV
	:param tmin: minimum of time bin that we are interested in collecting photons in to analyze
	:param tmax: max of time bin that we are interested in collecting photons in to analyze
	:param units: a string of the units of the spectrum that will be produced (erg/s or cts/s)
	:param h5: pecify if the format of the MCRaT output files is hdf5 files or not (denotes if using an old format or
	           a newer format of saving files)
	:return: returns the spectrum with the specified units, for the photons in each energy bin
	"""
	if (units!='cts/s')&(units!='erg/s'):
		print( 'Wrong units')
		print( 'The only allowed units are: erg/s and cts/s')
	else:
		if h5:
			try:
				time,hnu,weight, indexes, s0, s1, s2, s3, comv_hnu=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			except ValueError:
				time,hnu,weight, s0, s1, s2, s3=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
		else:
			try:
				time,hnu,weight, indexes=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
			except ValueError:
				time,hnu,weight=n.loadtxt('EVENT_FILES/'+simid+'.evt',unpack=True)
		
		dnu=numax-numin
		sp=n.zeros(numin.size)
		spe=n.zeros(numin.size)
		goodones=n.zeros(numin.size)
		for i in range(numin.size):
			jj=n.where((time>=tmin)&(time<tmax)&(hnu>=numin[i])&(hnu<numax[i]))
			if jj[0].size>0:
				if units=='erg/s':
					sp[i]=(1.6e-9)*n.sum(weight[jj]*hnu[jj])/dnu[i]
				else:
					sp[i]=n.sum(weight[jj])/dnu[i]
				spe[i]=sp[i]/n.sqrt(jj[0].size)
				if jj[0].size>10: 
					goodones[i]=jj[0].size

	return sp,spe,goodones

def readanddecimate(fnam, inj_radius):
	"""
	Legacy code from the python version of MCRaT to read and process FLASH files

	:param fnam: string of directory and file name of the FLASH file that will be loaded and processed
	:param inj_radius: The radius that photons are injected in cm
	:return: returns FLASH values of various quantities at cells
	"""
	import tables as t
	file = t.open_file(fnam)
	print( '>> mc.py: Reading positional, density, pressure, and velocity information...')
	xy = file.get_node('/', 'coordinates')
	xy = xy.read()
	x = n.array(xy[:, 0])
	y = n.array(xy[:, 1])
	sz = file.get_node('/', 'block size')
	sz = sz.read()
	szx = n.array(sz[:, 0])
	szy = n.array(sz[:, 1])
	vx = file.get_node('/', 'velx')
	vx = vx.read()
	vy = file.get_node('/', 'vely')
	vy = vy.read()
	vv = n.sqrt(vx ** 2 + vy ** 2)
	dens = file.get_node('/', 'dens')
	dens = dens.read()
	pres = file.get_node('/', 'pres')
	pres = pres.read()
	print(szx.shape, vx.shape)


	print ('>> mc.py: Creating the full x and y arrays...')
	xx = n.zeros(vx.shape)
	yy = n.zeros(vx.shape)
	x1 = n.array([-7., -5, -3, -1, 1, 3, 5, 7]) / 16.
	x2 = n.empty([8, 8])
	y2 = n.empty([8, 8])
	szxx=n.zeros(vx.shape)
	szyy=n.zeros(vx.shape)
	szxx[:,0,:,:]=szx[:, n.newaxis, n.newaxis]
	szyy[:,0,:,:]=szy[:, n.newaxis, n.newaxis]
	for ii in range(0, 8, 1):
		y2[:, ii] = n.array(x1)
		x2[ii, :] = n.array(x1)
	for ii in range(0, x.size):
		xx[ii, 0, :, :] = n.array(x[ii] + szx[ii] * x2)
		yy[ii, 0, :, :] = n.array(y[ii] + szy[ii] * y2)

	print ('>> mc.py: Selecting good node types (=1)...')
	nty = file.get_node('/', 'node type')
	nty = nty.read()
	file.close()
	jj = n.where(nty == 1)
	xx = n.array(xx[jj, 0, :, :]) * 1e9
	#	yy=n.array(yy[jj,0,:,:]+1) this takes care of the fact that y starts at 1e9 and not at 0
	yy = n.array(yy[jj, 0, :, :]) * 1e9
	szx=1e9*n.array(szxx[jj, 0, :, :])/8
	szy=1e9*n.array(szyy[jj, 0, :, :])/8
	vx = n.array(vx[jj, 0, :, :])
	vy = n.array(vy[jj, 0, :, :])
	dens = n.array(dens[jj, 0, :, :])
	pres = n.array(pres[jj, 0, :, :])

	print( '>> mc.py: Reshaping arrays...')
	xx = n.reshape(xx, xx.size)
	vx = n.reshape(vx, xx.size)
	yy = n.reshape(yy, yy.size)
	vy = n.reshape(vy, yy.size)
	szx=n.reshape(szx, yy.size)
	szy=n.reshape(szy, yy.size)
	gg = 1. / n.sqrt(1. - (vx ** 2 + vy ** 2))
	dd = n.reshape(dens, dens.size)
	dd_lab = dd * gg
	rr = n.sqrt(xx ** 2 + yy ** 2)
	tt = n.arctan2(xx, yy)
	pp = n.reshape(pres, pres.size)
	del pres, dens, x, y

	jj = n.where(rr > (0.95 * inj_radius))
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
