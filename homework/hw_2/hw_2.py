import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from astropy import units as u
sns.set_style('darkgrid')
sns.set_context('talk')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def make_pdf(t_array=np.logspace(5.0, 10.0, 10**4)*(u.yr), k=3.0,
             A=10.0*((u.km)**2.0), r=0.01/((u.km**2)*(10.0**6.0*u.yr))):
    mu = (r*A*t_array).decompose()
    norm_constant = r*A
    return (r*A)*(mu**k)*np.exp(-mu)/(np.math.factorial(int(k))), t_array


def calc_pdf_things(t_array=np.logspace(5.0, 10.0, 10**4)*(u.yr),
                    k=3.0,
                    A=10.0*((u.km)**2.0),
                    r=0.01/((u.km**2)*(10.0**6.0*u.yr))):
    pdf, t_array = make_pdf(t_array, k, A, r)
    max_loc = np.argmax(pdf.value)
    t_max = t_array[max_loc]
    pdf_max_array = np.linspace(0.0*pdf[0], pdf[max_loc], 1000)
    t_max_array = np.ones_like(pdf_max_array)*t_max
    equal_points = np.array([np.where(pdf[max_loc:] <= pdf[i])[0][0]
                             for i in range(0, max_loc)])
    integral_value = np.array([np.trapz(pdf[i:max_loc + equal_points[i]],
                                        t_array[i:max_loc + equal_points[i]])
                               for i in range(0, max_loc)])
    test_integral = np.array([np.abs(integral_value[i] - 0.68)
                              for i in range(0, max_loc)])
    loc_low = np.argmin(test_integral)
    loc_high = max_loc + equal_points[np.argmin(test_integral)]
    t_low = t_array[loc_low]
    t_high = t_array[loc_high]

    max_label = r'max$(P(t\vert k=' + str(int(k)) + r'))$ at $t=$'
    max_label += "{:.1f}".format(t_max.value/(10**6)) + r' Myrs'
    fill_label = "{:.2f}".format(t_low.value/(10**6)) + r' '
    fill_label += r'\/$< t \leq $\/' + r' '
    fill_label += "{:.2f}".format(t_high.value/(10**6))
    fill_label += r'\textrm{ Myr}, $k=' + str(int(k)) + r'$'
    return [[pdf, t_array],
            [pdf_max_array, t_max_array],
            [loc_low, loc_high, max_loc, t_low, t_high],
            max_label, fill_label]


def plot_pdf_things(t_array=np.logspace(5.0, 10.0, 10**4)*(u.yr),
                    k=3.0,
                    A=10.0*((u.km)**2.0),
                    r=0.01/((u.km**2)*(10.0**6.0*u.yr)),
                    diff_plot=2000,
                    save_name='meteor_1.pdf',
                    title_label=r'Likelihood $P(t \vert k)$',
                    figure_size=(12, 10),
                    bbox=(0.98, 1.0),
                    suptitle_y=0.998):
    [[pdf, t_array], [pdf_max_array, t_max_array],
     [loc_low, loc_high, max_loc, t_low, t_high],
     max_label, fill_label] = calc_pdf_things()
    mask = np.arange(max_loc - diff_plot, max_loc + diff_plot)
    [[pdf_2, t_array_2], [pdf_max_array_2, t_max_array_2],
     [loc_low_2, loc_high_2, max_loc_2, t_low_2, t_high_2],
     max_label_2, fill_label_2] = calc_pdf_things(k=37.0, A=100.0*(u.km**2.0))
    mask_2 = np.arange(max_loc_2 - diff_plot, max_loc_2 + diff_plot)
    plt.figure(figsize=figure_size)
    plt.semilogx(t_array[mask], pdf[mask], '--b', label=r'$P(t\vert k=3)$')
    plt.plot(t_max_array, pdf_max_array, ':b', label=max_label)
    plt.fill_between(t_array[loc_low:loc_high],
                     pdf[loc_low:loc_high], 0.0,
                     color='b', alpha=0.5,
                     label=fill_label)
    plt.semilogx(t_array_2[mask_2],
                 pdf_2[mask_2], '-k', label=r'$P(t\vert k=37)$')
    plt.plot(t_max_array_2, pdf_max_array_2, ':k', label=max_label_2)
    t_fill_array = np.array(t_array_2[loc_low_2:loc_high_2], dtype=float)
    pdf_fill_array = np.array(pdf_2[loc_low_2:loc_high_2], dtype=float)
    plt.fill_between(t_fill_array,
                     pdf_fill_array, 0.0,
                     color='k', alpha=0.5,
                     label=fill_label_2)
    plt.ylabel(r'$P(t\vert k)$')
    plt.xlabel(r'$t$ [years]')
    plt.legend(bbox_to_anchor=bbox)
    plt.suptitle(title_label,
                 y=suptitle_y)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


def do_single_press_Poisson(mean_rate=10):
    old_mean = -1.0
    mean_now = mean_rate
    logfact = np.ones((1024), dtype=np.longdouble)*(-1.0)
    if mean_now != old_mean:
        sqlam = np.sqrt(mean_now)
        loglam = np.math.log(mean_now)
    blah = True
    while blah:
        u = 0.64*np.random.uniform()
        v = -0.68 + 1.28*np.random.uniform()
        k = int(np.floor(sqlam*(v/u)+mean_now+0.5))
        if k < 0:
            continue
        u2 = u*u
        if k < 1024:
            if logfact[k] < 0:
                logfact[k] = np.math.log(np.math.factorial(k))
            lfac = logfact[k]
        else:
            lfac = np.math.log(np.math.factorial(k))
        p = sqlam*np.exp(-mean_now + k*loglam - lfac)
        if u2 < p:
            break
    mean_old = mean_now
    return k


def do_N_Poisson_samples(mean_rate, N):
    return np.array([do_single_press_Poisson(mean_rate)
                     for i in range(0, N)])


def problem_4(mean_rate=10, N=1000):
    sample_results = do_N_Poisson_samples(mean_rate, N)
    plt.figure(figsize=(8, 6))
    plt.hist(sample_results,
             label='Poisson Number Generator', alpha=0.6, bins='auto',
             color='green')
    k_array = np.arange(0, 25)
    factorial_array = np.array([np.math.factorial(k) for k in k_array],
                               dtype=float)
    pdf_array = np.array([(1.0/np.math.factorial(k))*(
        mean_rate**k)*np.exp(-mean_rate)
                          for k in range(0, 25)])
    plt.plot(k_array, N*pdf_array,
             label=r'$\frac{N \lambda^k e^{-\lambda}}{k!}$', color='k', drawstyle='steps-mid')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$n(k)$')
    title_label = r'Poisson Number Generator with $\lambda=$' + str(mean_rate)
    title_label += r' and $N=$' + str(N) + r' samples'
    plt.suptitle(title_label)
    plt.legend()
    plt.savefig('poisson.pdf')
    plt.show()


if __name__ == '__main__':
    plot_pdf_things()
    problem_2()
    problem_4()
