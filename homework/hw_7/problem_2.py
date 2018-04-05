import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits

sns.set_style('darkgrid')
sns.set_context('talk')


def read_fits_data(fname='f61final.fits'):
    hdulist = fits.open(fname)
    img = hdulist[0].data
    hdulist.close()
    return img[54:64, 136:146]


def get_noise(data, gain=2.72, readout=20.0):
    return np.sqrt((data / gain) + (readout**2.0 / gain**2.0))


if __name__ == '__main__':
    img_data = read_fits_data()
    ax = sns.heatmap(img_data)
    ax.set_aspect('equal')
    plt.show()
    noise = get_noise(img_data)
    ax2 = sns.heatmap(img_data / noise)
    ax2.set_aspect('equal')
    plt.show()
