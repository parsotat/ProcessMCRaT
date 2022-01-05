import numpy as np
from astropy import units as unit
from astropy import constants as const

radiation_dens_const=(4*const.sigma_sb/const.c)

def calc_temperature(hydro_obj):
    return (3 * hydro_obj.get_data('pres')  / radiation_dens_const.cgs) ** (1.0 / 4.0)