import numpy as np
from scipy import interpolate
from astropy.visualization import quantity_support
quantity_support()

from .plotting import random_photon_index
from .mclib import lc_time_to_radius


def create_image(hydro_obj, key, logscale=True):
    x0 = np.linspace((hydro_obj.get_data('x0').min()), (hydro_obj.get_data('x0').max()), num=1000)
    x1 = np.linspace((hydro_obj.get_data('x1').min()), (hydro_obj.get_data('x1').max()), num=1000)
    data=hydro_obj.get_data(key)


    points = np.empty([x0.size, 2])
    points[:, 0] = x0
    points[:, 1] = x1

    X, Y = np.meshgrid(x0, x1)

    Z = interpolate.griddata(points, data, (X, Y), method='nearest', rescale=True)

    if logscale:
        data=np.log10(Z)
    else:
        data=Z

    img_data = np.zeros([data.shape[0], 2 * data.shape[1]])
    img_data[:, :data.shape[1]] = np.fliplr(data)
    img_data[:, data.shape[1]:] = data

    return img_data


