import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import erf
sns.set_style('darkgrid')
sns.set_context('talk')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def get_erf(sigma_array):
    erf_term = erf(45.0*np.sqrt(2.0)/sigma_array)
    denom = 2.0*sigma_array*np.sqrt(2.0*np.pi)
    return erf_term/denom


if __name__ == '__main__':
    sigma_array = np.linspace(0.21, 0.23, 1000)
    erf_line = get_erf(sigma_array)
    intersect_line = np.ones_like(sigma_array)*0.9
    intersect_sigma = sigma_array[np.argmin(np.abs(erf_line - 0.9))]
    erf_label = r'$P(r < 90 \textrm{km})=\frac{1}{2\sigma \sqrt{2\pi}}'
    erf_label += r'\mathrm{erf}\left(\frac{45\sqrt{2}}{\sigma}\right)$'
    temp_label = "{:.3f}".format(intersect_sigma*1000.0) + ' m'
    title_label = r'Error Ellipse for Martian Lander: $\sigma = $' + temp_label
    answer_label = r'Intersection $\sigma = $' + temp_label
    plt.plot(sigma_array*1000.0, erf_line, label=erf_label)
    plt.plot(sigma_array*1000.0, intersect_line,
             label=r'$P(r < 90 \textrm{km})=0.9$')
    plt.axvline(intersect_sigma*1000.0, linestyle=':',
                color='k', label=answer_label, alpha=0.5)
    plt.legend()
    plt.setp(plt.gca().get_legend().get_texts(), fontsize='10')
    plt.xlabel(r'$\sigma$ [m]')
    plt.ylabel(r'$P(r)$')
    plt.suptitle(title_label)
    plt.savefig('problem_2.pdf')
    plt.show()
