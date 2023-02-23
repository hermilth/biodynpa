# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the graphic functions dedicated module.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''

import matplotlib.pyplot as plt

# Imports


from maps import *


# Functions


def TL_var(var):

    '''
    A reprensentation of the temporal coverage of the given variable in your dataset.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param var: The variable of interest. (str)
    :return: The matplotlib axes. (matplotlib axes)
    '''
    fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3, 1]})
    fig.suptitle('Floats {} timeline and Multivariate Enso Index'.format(var))

    for i, wmo in enumerate(WMOS()):

        prof = PRC(wmo)

        if var in prof.data_vars:

            mask = ~np.isnan(prof[var].values).all(axis=1)
            t = FMT_date(prof.JULD.values[mask], 'dt', verbose=False)
            tmin, tmax = np.min(t), np.max(t)
            ax[0].plot([tmin, tmax], [i, i], c=VAR.clusters_colors[VAR.floats_names[FMT_wmo(wmo)][0]], linewidth=14)

    ax[0].set_yticks(np.arange(len(WMOS())))
    ax[0].set_yticklabels(list(VAR.floats_names.values()))
    ax[0].invert_yaxis()
    ax[0].grid()

    # MEI file has to be reformatted as csv
    file = 'Files/MEI/meiv2.data'
    data = np.genfromtxt(file, delimiter=',')
    MEI = [data[i + 1, j + 1] for i in range(np.shape(data)[0] - 1) for j in range(np.shape(data)[1] - 1)]
    T = [dt.datetime(int(y), int(m), 15) for y in data[1:, 0] for m in data[0, 1:]]
    MEI = np.where(np.abs(MEI) > 10., np.nan, MEI)

    xlims = ax[0].get_xlim()

    ax[1].plot(T, MEI, c='k')
    ax[1].set_ylabel('MEI')
    ax[1].set_xlabel('Time')
    ax[1].set_xlim(xlims)
    ax[1].grid()

    fig.tight_layout()

    return ax


def HST_sprof(pathorprof, var, bins=50, min=None, max=None, log=False, plot_gaussian=False, ax=None):
    '''
    Plots the values histogram of a float variable.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param var: the variable to display. (str)
    :param bins: The number of color category to use. If None, the maximum will be used. (int, optional)
    :param min: Limits the histogram bins to this minimum. (float, optional)
    :param max: Limits the histogram bins to this maximum. (float, optional)
    :param log: For logarithmic bins. (boolean, default to False)
    :param ax: The axes on which to plot the histogram. (matplotlib axes, optional)
    :return: The matplotlib axes.
    '''
    war.filterwarnings('ignore', category=RuntimeWarning)

    if log:
        ylab = '$\log_{10} (N_{points})$'
    else:
        ylab = '$N_{points}$'

    if type(pathorprof) is list:

        prof_list = PRC(pathorprof)
        data = np.array([])
        for prof in prof_list:
            if prof is None:
                pass
            elif not(var in prof.data_vars):
                pass
            else:

                new_data = prof[var].values.flatten()
                new_data = new_data[~np.isnan(new_data)]
                data = np.hstack([data, new_data])

                if var == 'LONGITUDE':

                    data = np.array([FMT_lon(lon) for lon in data])

    else:

        prof = PRC(pathorprof)
        data = prof[var].values.flatten()

        if var == 'LONGITUDE':

            data = np.array([FMT_lon(lon) for lon in data])

    if ax is None:
        _, ax = plt.subplots()

    if plot_gaussian:

        m = np.nanmean(data)
        sig = np.sqrt(np.nanvar(data))
        x = np.linspace(m - 3 * sig, m + 3 * sig, 100)
        ax.plot(x, stats.norm.pdf(x, m, sig)*np.size(data[~np.isnan(data)]), c='red')
        ax.legend(['Associated gaussian distribution'])

    if min is None:
        min = np.nanmean(data) - 4 * np.nanstd(data)
    if max is None:
        max = np.nanmean(data) + 4*np.nanstd(data)
    if np.isnan(min) or np.isnan(max):
        hist = ax.hist(np.array([]), bins=bins, log=log)
    else:
        span = max - min
        tot = np.size(data)
        count = np.sum((data>max).astype(int)) + np.sum((data<min).astype(int))
        ax.set_xlim([min, max])
        hist = ax.hist(data, bins=bins, range=[min, max], log=log, color='darkblue')
        height = 6/7*hist[0].max()
        if log:
            height = 10**np.log10(0.5*height)
        ax.text(max - 3*span/4, height, '{} points outside\n window ({:.2f}\%)'.format(count.astype(int),
                                                                                           100*count/tot),
                fontsize=10., horizontalalignment='center')
        ax.set_xlabel(VAR.var_labels[var])
        ax.set_ylabel(ylab)

    plt.tight_layout()

    return ax


def HST_hours(pathorprof, ax=None, c='royalblue', bins=None, combine=True, absolute=True, verbose=True):
    '''
    Plots an histogram of profile hours.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param ax: The matplotlib axes. (matplotlib axes)
    :param c: The bar colors. (matplotlib colors)
    :param bins: The histogram categories limits. (list or numpy array, optional)
    :param combine: Whether to combine profiles in one histogram or superimpose several histograms. (boolean,
     default is True)
    :param absolute: Whether to use the bins count or divide it by the bin values for bar heights. (boolean,
     default is True)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''
    ti = t.time()

    if ax is None:

        _, ax = plt.subplots()

    if bins is None:

        bins = np.arange(0, 25)

    ax.set_title('Profile times')
    ax.set_xlim((0, 24))
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xlabel('Hour of the day')

    if type(pathorprof) is list:

        cmap = plt.get_cmap('gist_rainbow')
        cycle = [cmap(1. * i / (len(pathorprof)-1)) for i in range(len(pathorprof))]

        if combine:

            proxies = [plt.Rectangle((-9999, -9999.), 0.1, 0.1, color=c) for c in cycle]
            hours = []
            labels = []
            wmo_list = []

            for i, p in enumerate(pathorprof):

                if verbose:
                    CPRINT('Computing histogram.{}'.format('.'*(i%3)), attrs='BLUE', end='\r')

                prof = FLT_qcs(p, verbose=False)
                labels.append(int(prof.PLATFORM_NUMBER.values[0]))
                hours.append(CMP_localtimes(prof.JULD.values, prof.LONGITUDE.values))

            mapax = plt.axes(position=(0.18, 0.62, 0.27, 0.2), projection=ccrs.PlateCarree(central_longitude=180))
            CHART(ax=mapax, verbose=False)
            PT_floats(labels, ax=mapax, verbose=False)

            if absolute:
                ax.hist(hours, color=cycle, histtype='barstacked', bins=bins, density=False)
                ax.set_ylabel('Profile count')
            else:
                ax.hist(hours, color=cycle, histtype='barstacked', bins=bins, density=True)
                ax.set_ylabel('Normalized profile count')

            LGD(ax, proxies, labels, loc='upper right')

        else:

            for i, p in enumerate(pathorprof):

                HST_hours(p, ax=ax, c=(0.2, 0.1, 0.9, 0.3), absolute=absolute, verbose=False)

    else:

        prof = RAW(pathorprof)

        if ax is None:

            _, ax = plt.subplots()

        hours = CMP_localtimes(prof)

        if absolute:
            ax.hist(hours, color=c, bins=bins, label=int(prof.PLATFORM_NUMBER.values[0]))
            ax.set_ylabel('Profile count')
        else:
            ax.hist(hours, color=c, bins=bins, density=True, label=int(prof.PLATFORM_NUMBER.values[0]))
            ax.set_ylabel('Normalized profile count')

    TINFO(ti, 2., 'Hours histogram', verbose)

    return ax


def OV_sensordrift(pathorprof, var, verbose=True):
    '''
    Fits a linear function on the mean of variable values below 600m (or 200m if no values are available below) in
    order to detect an eventual sensor drift.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param var: the variable to inspect. (str)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: None.
    '''
    ti = t.time()

    war.filterwarnings('ignore', category=RuntimeWarning)
    get_exponent = lambda x: np.floor(np.log10(np.abs(x))).astype(int)
    thresh = 600

    prof = RAW(pathorprof)
    wmo = int(prof.PLATFORM_NUMBER.values[0])

    if var[-9:] == '_ADJUSTED':
        pr = prof.PRES_ADJUSTED
    else:
        pr = prof.PRES

    std = np.nanstd(prof[var].values)
    valmean = np.nanmean(np.where(pr.values > thresh * VAR.hf, prof[var].values, np.nan), axis=1)
    vals = np.where(pr.values > thresh * VAR.hf, prof[var].interpolate_na(dim='N_LEVELS').values, np.nan)
    dates = FMT_date(prof.JULD, 'dt', verbose=False)
    dates -= dates[0]
    time = np.array([date.days + date.seconds/(3600*24) for date in dates])

    if vals[~np.isnan(valmean)].size > 2:

        mask = ~np.isnan(valmean)
        time, valmean, vals = time[mask], valmean[mask], vals[mask]
        ind = np.where(np.sum((~np.isnan(vals)).astype(int), axis=0) > 5,
                       np.arange(np.size(vals[0])), np.nan)

    else:

        thresh = 200
        mask = ~np.isnan(valmean)
        valmean = np.nanmean(np.where(pr.values > thresh * VAR.hf, prof[var].values, np.nan), axis=1)
        valstd = np.nanstd(np.where(pr.values > thresh * VAR.hf, prof[var].values, np.nan), axis=1)
        vals = np.where(pr.values > thresh * VAR.hf, prof[var].interpolate_na(dim='N_LEVELS').values, np.nan)
        time, valmean, vals = time[mask], valmean[mask], vals[mask]
        ind = np.where(np.sum((~np.isnan(vals)).astype(int), axis=0) > 5,
                       np.arange(np.size(vals[0])), np.nan)

    res = stats.linregress(time, valmean)
    slope, inter = res.slope * 365, res.intercept
    exp = get_exponent(slope)

    if np.abs(slope*time[-1]/365)/std*100 > 10.:
        col = 'red'
    elif np.abs(slope*time[-1]/365)/std*100 > 5.:
        col = 'orange'
    elif np.abs(slope*time[-1]/365)/std*100 > 2.:
        col = 'yellow'
    else:
        col = 'blue'

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle('Float \#{} {} drift based on values below {}m'.format(wmo, var, thresh), y=.95)

    ax[0].tick_params(labelbottom=True)
    ax[0].set_position([0.12, 0.07, 0.75, 0.36])

    Z = np.linspace(thresh * VAR.hf, np.nanmax(pr.values), 200)

    X, Y = np.meshgrid(time, Z)
    V = INT_1D(prof[var].values, pr * VAR.hf, Z)
    vmin, vmax = CMP_lims(V, 5.)
    mappable = ax[0].pcolormesh(X, Y, V[mask, :].T, cmap=VAR.var_cmaps[var], vmin=vmin, vmax=vmax, shading='gouraud')
    CBAR(mappable, ax[0], cbarlab=VAR.var_labels[var])
    ax[0].set_xlabel('Time (days)')
    ax[0].set_ylabel('Depth (m)')
    ax[0].set_ylim(np.nanmax(prof.PRES.values), thresh * VAR.hf)

    ax[1].set_position([0.12, 0.48, 0.75, 0.36])
    ax[1].tick_params(labelbottom=False)

    ax[1].plot(time, valmean, c='k', linewidth=1., label='Vertical mean value')
    ax[1].plot(ax[1].get_xlim(), [slope/365.*ax[1].get_xlim()[0] + inter, slope/365.*ax[1].get_xlim()[1] + inter],
            linestyle='dashed', c=col, linewidth=2., label='Least squares fit')
    ax[1].set_ylabel(VAR.var_names[var] + ' mean\n(' + VAR.var_units[var] + ')')
    ax[1].legend()
    ax[1].set_title('Lifetime: ${:.1f} years$ $|$ Slope: ${:.1f}\\times10^{{{}}}${}$/year$ (${:.1f}\%$ of std)'
                    ' $|$ Drift $\\times$ Age: ${:.1f}\\times10^{{{}}}$ {} (${:.1f}\%$ of std)'
                    .format(time[-1]/365., slope*10.**-exp, exp, VAR.var_units[var], slope/std*100,
                            slope*time[-1]/365 *10.**-exp, exp, VAR.var_units[var], slope*time[-1]/365/std*100),
                    fontsize=14)

    TINFO(ti, 2., 'Drift plot', verbose)


def ZV_sprof(pathorprof, vars, ax=None, dates=None, nprof=-1, dmax = 300., vmin=None, vmax=None, color='k', back=True,
             fancy=True, toscale=False, verbose=True):
    '''
    Plots the variable individual profiles, as curves.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param vars: The variables that you want to plot. (str or list of str)
    :param dates: The date at which to plot the profile. (datetime or str 'YYYY-MM-DD', optional)
    :param nprof: The number of the profile in the pathorprof file. Prefer date arg. (int)
    :param dmax: The maximum depth of the plot. (float, default is 500)
    :param color: The color of the curves. (matplotlib color, default is 'black')
    :param back: Plots the other profiles in a light grey in the back. (boolean, default is True)
    :param fancy: Adds the DCM if CHLA is the variable plot, z_1% if DOWNWELLING_PAR is. (boolean, default is True)
    :param toscale: Sets the variable range the same.
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''

    ti = t.time()

    def aux(prof, var, nprof, back, ax, color, fancy):

        if var[-9:] == '_ADJUSTED':
            pres = prof.PRES_ADJUSTED.values
        else:
            pres = prof.PRES.values

        Ytemp = prof[var].values

        for k in range(np.shape(pres)[0]):

            log = False
            if var[:4] == 'DOWN':
                war.filterwarnings('ignore', category=RuntimeWarning)
                log = True
                X, Y = pres[k][~np.isnan(pres[k])], np.log10(Ytemp[k][~np.isnan(pres[k])])
            else:
                X, Y = pres[k][~np.isnan(pres[k])]*VAR.hf, Ytemp[k][~np.isnan(pres[k])]

            X, Y = X[~np.isnan(Y)], Y[~np.isnan(Y)]

            if k == nprof:

                if np.size(Y) > 0:

                    minval, maxval = np.min(Y), np.max(Y)
                    diff = maxval - minval

                    if 'CHLA' in var and fancy:

                        dcm = prof.SCM.values[k]
                        if not np.isnan(dcm):
                            ax.plot([-1., 20.], [dcm, dcm], color=(0., 0.4, 0.), label='SCM: ${:0.1f}m$'.format(dcm), zorder=3,
                                    linewidth=1.5, linestyle='dashed')
                            ax.legend(loc='lower right')
                        mld = prof.MLD_S03.values[k]
                        if not np.isnan(mld):
                            ax.plot([-1., 20.], [mld, mld], color='b', label=VAR.var_labels['MLD_S03']
                                    .format(mld), zorder=3, linewidth=1.5, linestyle=':')
                            ax.legend(loc='lower right')
                        iso15 = prof.ISO15.values[k]
                        if not np.isnan(iso15):
                            ax.plot([-1., 20.], [iso15, iso15], color='orange', label='Isolume$_{{15}}$: ${:0.1f}$m'
                                    .format(iso15), zorder=3, linewidth=1.5, linestyle='-.')
                            ax.legend(loc='lower right')

                    if var[:15] == 'DOWNWELLING_PAR' and fancy:

                        i = 0
                        par_surf = Y[i]
                        while np.isnan(par_surf):
                            i += 1
                            par_surf = Y[i]

                        photic_ind = 0
                        while Y[photic_ind]>0.01*par_surf and photic_ind<np.size(Y):
                            photic_ind += 1
                        if photic_ind!=np.size(Y):
                            xmin, xmax = ax.get_xlim()
                            span = xmax - xmin
                            ax.plot([xmin-span, xmax+span], [X[photic_ind], X[photic_ind]], color='blue',
                                    label='z$_{{1\%}}$: {:0.1f}m'.format(X[photic_ind]))
                            ax.set_xlim(xmin, xmax)
                            ax.legend(loc='lower right')

                    ax.plot(Y, X, color=color, zorder = 3)
                    ax.set_xlim(minval-diff/10, maxval+diff/10)

                ax.grid()
                ax.xaxis.tick_top()
                if log:
                    ax.set_title('$\log_{10}$ '+VAR.var_labels[var], fontsize=11)
                else:
                    ax.set_title(VAR.var_labels[var], fontsize=11)
                ax.set_aspect('auto')

            elif back:

                ax.plot(Y, X, color=(0.35, 0.35, 0.45, 0.3), zorder=2, lw=VAR.linewidth/3)

    if verbose:
        CPRINT('Starting curve plot...', attrs='BLUE', end='\r')

    if not type(pathorprof) is xr.Dataset:
        prof = PRC(pathorprof)
    else:
        prof = pathorprof

    if not type(nprof) is list:

        if type(nprof) is np.ndarray:
            nprof = list(nprof)
        else:
            nprof = [nprof]

    for i, npr in enumerate(nprof):
        if npr < 0:
            nprof[i] = np.size(prof.N_PROF.values) + npr

    if dates is not None:

        if type(dates) is not list:

            if type(dates) is np.ndarray:
                dates = list(dates)
            else:
                dates = [dates]

        nprof = []

        for date in dates:

            date = FMT_date(date, 'dt', verbose=False)
            timz = FMT_date(prof.JULD, 'dt', verbose=False)
            nprof.append(np.argmin(np.abs(timz - date)))

            if np.nanmin(abs(timz - date)) > dt.timedelta(20):
                CPRINT('There is {} days between profile and input date.' .format(np.nanmin(abs(timz - date)).days),
                       attrs='YELLOW')

    if type(vars) is not list:
        vars = [vars]

    if ax is None:
        fig, ax = plt.subplots(1, len(vars)*len(nprof), sharey=True)
    else:
        fig = plt.gcf()

    fig.suptitle('Profiles of float {}'.format(VAR.floats_names[GET_wmo(prof)]))
    fig.subplots_adjust(bottom = 0.06, top = 0.85, left = 0.06, right = 0.97, wspace=0.04)

    vminexp = 9999.
    vmaxexp = -9999.

    if len(vars)*len(nprof) == 1:
        ax = [ax]

    for i in range(len(vars)*len(nprof)):

        aux(prof, vars[i%len(vars)], nprof[i//len(vars)], back, ax[i], color, fancy)
        ax[i].set_xlabel('{}'.format(FMT_date(prof.JULD.values[nprof[i//len(vars)]],
                                              'dt', verbose=False).strftime('%Y-%m-%d')))
        if vminexp > ax[i].get_xlim()[0]:
            vminexp = ax[i].get_xlim()[0]
        if vmaxexp < ax[i].get_xlim()[1]:
            vmaxexp = ax[i].get_xlim()[1]

    if vmin is not None:
        vminexp = vmin
    if vmax is not None:
        vmaxexp = vmax

    if len(vars) == 2:
        if vars[0] in vars[1] or (vmin is not None or vmax is not None):
            toscale = True
    elif len(vars) == 1:
        toscale = True

    if toscale:
        xlims = list(ax[0].get_xlim())
        for a in ax:
            if a.get_xlim()[0] < xlims[0]:
                xlims[0] = a.get_xlim()[0]
            if a.get_xlim()[1] > xlims[1]:
                xlims[1] = a.get_xlim()[1]

        for i in range(len(vars)*len(nprof)):
            ax[i].set_xlim(xlims)

    ax[0].set_ylim(dmax, 0.)
    ax[0].set_ylabel('Depth ($m$)')

    TINFO(ti, 4., 'Curves plot', verbose)

    return ax


def TS_xarr(XA, PR, TI, dmax = 250., vmin=None, vmax=None, out=5., extend='both', levels=11, datemin=None,
            datemax=None, scientific=True, ax=None, cmap='jet', cbarlab='', log=False, isolines=None,
            logisolines=False, cleaner=None, limiter=None, discrete=False, interpolate=True, title=None, verbose=True):
    '''
    Plots the time serie of an xarray.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param XA: The xarray with values to be displayed. (('N_PROF', 'N_LEVELS') shaped xarray)
    :param PR: The pressure array associated to XA. (('N_PROF', 'N_LEVELS') shaped xarray)
    :param TI: The time xarray. ((N_PROF,) shaped xarray)
    :param dmax: The maximum depth of the plot. (float, optional)
    :param vmin: The minimum value in the colormap. (float, optional)
    :param vmax: The maximum value in the colormap. (float, optional)
    :param extend: Which end of the colorbar to extend to a larger span. (str in ['neither', 'both', 'min', 'max'],
     default is 'both')
    :param levels: The isolines levels to use. If None, no isolines will be drawn. (1-D array-like)
    :param datemin: The lower bound in time. (str or datetime, optional)
    :param datemax: The upper bound in time. (str or datetime, optional)
    :param scientific: Will format the colorbar ticks. (boolean, default is True)
    :param ax: The matplotlib axes. (matplotlib axes, optional)
    :param cmap: The matplotlib colormap. (str, optional)
    :param cbarlab: The colorbar label. (str, optional)
    :param log: Linear or Log normalization colorbar. (boolean, default is True)
    :param isolines: The xarray of isolines. Have to be associated to the same pressure array than XA. (xarray,
     optional)
    :param logisolines: Whether to normalize the isolines logarithmically. (boolean, default is False)
    :param cleaner: The outliers cleaning tolerance as number of data standard deviation. (float, optional)
    :param limiter: Float limiting colorbar values to [mean - limiter*std, mean + limiter*std]. (float, optional)
    :param discrete: Whether to plot colors as discrete bins. (boolean, default is True)
    :param interpolate: Whether to interpolate linearly the depths dimension of the profiles. (boolean, default is True)
    :param title: The title. (str, optional)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''

    ti = t.time()

    # Creating necessary variables

    cmap = cm.get_cmap(cmap).copy()
    if ax is None:
        _, ax = plt.subplots()
        ax.set_position([0.08, 0.08, 0.8, 0.8])
    if dmax == 'max':
        dmax = np.nanmax(PR.values)
    nplots = 0

    # Aligning data and depths

    da = ALIGN(XA, PR, dmax)

    # Cleaning, interpolating and setting value limits

    if cleaner is not None:
        da = CLN_extremes(da, cleaner)

    if datemin is None or datemax is None:
        times = TI[~np.isnan(XA.values).all(axis=1)]
        if datemin is None:
            datemin = np.nanmin(times)
        if datemax is None:
            datemax = np.nanmax(times)

    if vmin is None:
        if limiter is not None and float(da.mean() - limiter * da.std()) > float(da.min()):
            vmin = float(da.mean() - limiter * da.std())
        else:
            vmin = CMP_lims(da.values, out)[0]

    if vmax is None:
        if limiter is not None and float(da.mean() + limiter * da.std()) < float(da.max()):
            vmax = float(da.mean() + limiter * da.std())
        else:
            vmax = CMP_lims(da.values, out)[1]

    ax.grid(False)

    if discrete:

        if log:

            X, Y = np.meshgrid(TI.values, da.DEPTH.values)
            war.filterwarnings('ignore', category=UserWarning)
            mappable = ax.contourf(X, Y, da.values, cmap=cmap, norm=colors.LogNorm(), extend=extend)

        else:

            levels = np.linspace(vmin, vmax, levels)
            X, Y = np.meshgrid(TI.values, da.DEPTH.values)
            mappable = ax.contourf(X, Y, da.values.T, levels=levels, cmap=cmap, extend=extend,
                                   norm=colors.Normalize(vmin=vmin, vmax=vmax))

    else:

        if log:

            X, Y = np.meshgrid(TI.values, da.DEPTH.values)
            war.filterwarnings('ignore', category=UserWarning)
            mappable = ax.pcolormesh(X, Y, da.values.T, cmap=cmap, norm=colors.LogNorm())

        else:

            levels = np.linspace(vmin, vmax, int(levels))
            X, Y = np.meshgrid(TI.values, da.DEPTH.values)
            mappable = ax.pcolormesh(X, Y, da.values.T, cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))

    ax.grid(True)
    nplots += 1

    # Colorbar

    cbar = CBAR(mappable, ax, cbarlab=cbarlab, extend=extend)

    # Handling isolines

    if isolines is not None:

        styles = ['solid', 'dashed', 'dashdot', 'dotted']

        if not type(isolines) is list:
            isolines = [isolines]
            logisolines = [logisolines]

        for i, iso in enumerate(isolines):

            if len(iso.values.shape) == 1:

                ax.plot(TI.values, iso.values, linewidth=VAR.linewidth, linestyle=styles[i%4], c='k')

            else:
                vminiso, vmaxiso = CMP_lims(iso.values, out=5.)
                TS_isolines(iso, PR, TI, ax=ax, dmax=dmax, vmin=vminiso, vmax=vmaxiso, linestyles=styles[i % 4],
                            levels=np.linspace(vminiso, vmaxiso, 5), datemin=datemin, datemax=datemax, c='k',
                            label=iso.name, log=logisolines[i%4], verbose=False)
                nplots += 1

    # Setting time limits and ticking axis

    ax.set_xlim((datemin, datemax))
    ax.set_ylim(0., dmax)
    TCK_timeax(ax)

    # Finalizing plot and printing time

    ax.set_ylabel('Depth (m)')
    if nplots % 2 == 1:
        ax.invert_yaxis()
    ax.grid()

    if title is not None:
        ax.set_title(title, pad=20)

    TINFO(ti, 1., 'Contour plot', verbose)

    return ax


def TS_isolines(IS, PR, TI, ax=None, dmax = 250., vmin=None, vmax=None, levels=11, datemin=None, datemax=None, c='k',
                linestyles='solid', label=None, log=False, cleaner=None, limiter=None, title=None, verbose=True):

    '''
    Plots the time serie of an xarray as isolines.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param IS: The xarray with values to be contoured. (('N_PROF', 'N_LEVELS') shaped xarray)
    :param PR: The pressure array associated to XA. (('N_PROF', 'N_LEVELS') shaped xarray)
    :param TI: The time xarray. ((N_PROF,) shaped xarray)
    :param ax: The matplotlib axes. (matplotlib axes, optional)
    :param dmax: The maximum depth of the plot. (float, optional)
    :param vmin: The minimum value in the colormap. (float, optional)
    :param vmax: The maximum value in the colormap. (float, optional)
    :param levels: The isolines levels to use. If None, no isolines will be drawn. (1-D array-like)
    :param datemin: The lower bound in time. (str or datetime, optional)
    :param datemax: The upper bound in time. (str or datetime, optional)
    :param c: The lines color. (matplotlib color, default is black)
    :param linestyles: The style of the lines. (str of ['solid', 'dashed', 'dashdot', 'dotted'], default if 'solid')
    :param label: The lines label. (str, optional)
    :param log: Linear or Log normalization. (boolean, default is True)
    :param cleaner:  Float erasing dataset values out of [mean - cleaner*std, mean + cleaner*std]. (float, optional)
    :param limiter: Float limiting colorbar values to [mean - limiter*std, mean + limiter*std]. (float, optional)
    :param title: The title. (str, optional)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''

    ti = t.time()

    # Aligning data and depths

    da = ALIGN(IS, PR, dmax)

    # Creating necessary variables

    if ax is None:
        _, ax = plt.subplots()
    if dmax == 'max':
        dmax = np.nanmax(PR.values)

    # Cleaning and setting value limits

    if cleaner is not None:
        da = CLN_extremes(da, cleaner)

    if vmin is None:
        if limiter is not None and float(da.mean() - limiter * da.std()) > float(da.min()):
            vmin = float(da.mean() - limiter * da.std())
        else:
            vmin = CMP_lims(da.values, out=0)[0]

    if vmax is None:
        if limiter is not None and float(da.mean() + limiter * da.std()) < float(da.max()):
            vmax = float(da.mean() + limiter * da.std())
        else:
            vmax = CMP_lims(da.values, out=0)[1]

    # Plot

    if log:

        X, Y = np.meshgrid(TI.values, da.DEPTH.values)
        war.filterwarnings('ignore', category=UserWarning)
        mappable = ax.contour(X, Y, da.values, norm=colors.LogNorm(), linewidths=VAR.linewidth, linestyles=linestyles,
                              colors=c)

    else:

        if type(levels) is int:
            levels = np.linspace(vmin, vmax, levels)

        X, Y = np.meshgrid(TI.values, da.DEPTH.values)
        mappable = ax.contour(X, Y, da.values, levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax),
                              linewidths=VAR.linewidth, linestyles=linestyles, colors=c)

    # Setting time limits and ticking axis

    if datemin is None or datemax is None:
        if datemin is None:
            datemin = np.datetime64(int(ax.get_xlim()[0]), 'D')
        if datemax is None:
            datemax = np.datetime64(int(ax.get_xlim()[1]), 'D')

    datemin, datemax = FMT_date(datemin, 'dt64', verbose=False), FMT_date(datemax, 'dt64', verbose=False)
    ax.set_xlim((datemin, datemax))
    ax.set_ylim(0., dmax)
    TCK_timeax(ax)

    # Handling legend

    if label is not None:

        proxy = plt.Line2D([-9999., -9999.], [-9999., -9999.], c=c, linewidth=VAR.linewidth, linestyle=linestyles)
        LGD(ax, proxy, label)

    # Finalizing plot and printing time

    ax.set_ylabel('Depth (m)')
    ax.invert_yaxis()
    ax.grid()

    if title is not None:
        ax.set_title(title, pad=20)


    TINFO(ti, 1., 'Contour plot', verbose)

    return ax


def TS_sprof(pathorprof, var, dmax = 250., vmin=None, vmax=None, out=3., extend='both', levels=11, datemin=None,
             datemax=None, t_res=10., scientific=False, ax=None, cmap=None, cbarlab=None, log=False, isolines=None,
             cleaner=None, limiter=None, QCs=4, title=None, discrete=False, global_lims=False, verbose=True,
             **kwargs):
    '''
    Plots the time serie of an ARGO Sprof variable.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. By default, opens the raw profile. (str, int or xarray)
    :param var: the variable to display. (str)
    :param dmax: The maximum depth of the plot. (float, optional)
    :param vmin: The minimum value in the colormap. (float, optional)
    :param vmax: The maximum value in the colormap. (float, optional)
    :param extend: Which end of the colorbar to extend to a larger span. (str in ['neither', 'both', 'min', 'max'],
     default is 'both')
    :param levels: The isolines levels to use. If None, no isolines will be drawn. (1-D array-like)
    :param datemin: The lower bound in time. (str or datetime, optional)
    :param datemax: The upper bound in time. (str or datetime, optional)
    :param scientific: Will format the colorbar ticks. (boolean, default is False)
    :param ax: The matplotlib axes. (matplotlib axes, optional)
    :param cmap: The matplotlib colormap. (str, optional)
    :param cbarlab: The colorbar label. (str, optional)
    :param log: Linear or Log normalization colorbar. (boolean, default is False)
    :param isolines: The variable to use as isolines. (str, optional)
    :param cleaner: Float erasing values outside of [mean - limiter*std, mean + limiter*std]. (float, optional)
    :param limiter: Float limiting colorbar values to [mean - limiter*std, mean + limiter*std]. (float, optional)
    :param title: The title. (str, optional)
    :param discrete: Whether to plot colors as discrete bins. (boolean, default is True)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''

    ti = t.time()

    if verbose:
        CPRINT('Starting variable plot...', attrs='BLUE')

    prof = RAW(pathorprof)

    # Kwargs have to take over regular arguments

    if 'kwargs' in kwargs.keys():

        if 'scientific' in kwargs['kwargs']:
            scientific = kwargs['kwargs']['scientific']
        if 'log' in kwargs['kwargs']:
            log = kwargs['kwargs']['log']
        if 'extend' in kwargs['kwargs']:
            extend = kwargs['kwargs']['extend']
        if 'cmap' in kwargs['kwargs']:
            cmap = kwargs['kwargs']['cmap']

    if title is None:
        title = '{} of float {}'.format(var, VAR.floats_names[GET_wmo(prof)])
    if cbarlab is None:
        cbarlab = VAR.var_labels[var]
    if cmap is None:
        cmap = VAR.var_cmaps[var]
    if global_lims:
        vmin, vmax = VAR.var_lims[var]

    if var[-9:] == '_ADJUSTED':

        xarr, pr, timz = prof[var], prof.PRES_ADJUSTED, prof.JULD

    else:

        xarr, pr, timz = prof[var], prof.PRES, prof.JULD

    # Subsampling

    dates = FMT_date(prof.JULD.values, 'dt', verbose=False)
    ddays = [np.round(td.days + td.seconds / 3600 / 24, 2) for td in (dates[1:] - dates[:-1])]
    i = 0
    k = 1
    indexes = [i]
    while i + k < np.size(prof.N_PROF.values)-1:
        while np.sum(ddays[i:i + k]) < t_res and i + k < np.size(prof.N_PROF.values)-1:
            k += 1
        if i + k < np.size(prof.N_PROF.values):
            i = i + k
            indexes.append(i)
        k = 1

    xarr, pr, timz = xarr.isel(N_PROF=indexes), pr.isel(N_PROF=indexes), timz.isel(N_PROF=indexes)

    ax = TS_xarr(xarr, pr, timz, dmax=dmax, vmin=vmin, vmax=vmax, out=out, extend=extend, levels=levels,
                 datemin=datemin, datemax=datemax, scientific=scientific, ax=ax, cmap=cmap, cbarlab=cbarlab,
                 log=log, cleaner=cleaner, limiter=limiter, title=title, discrete=discrete, verbose=False)

    nplots = 1

    if isolines is not None:

        styles = ['solid', 'dashed', 'dotted', 'dashdotted']

        if type(isolines) is not list:

            isolines_xarr = [prof[isolines].isel(N_PROF=indexes)]
            isolines = [isolines]

        else:

            isolines_xarr = [prof[iso].isel(N_PROF=indexes) for iso in isolines]

        logisolines = [VAR.var_kwargs[iso]['log'] for iso in isolines]

        for i, iso in enumerate(isolines_xarr):

            if len(iso.values.shape) > 1:

                TS_isolines(iso, pr, timz, ax=ax, log=logisolines[i], linestyles=styles[i % 4],
                            label=VAR.var_names[isolines[i]], verbose=False, c=VAR.var_kwargs[var]['isocol'])
                nplots += 1

            else:

                ax.plot(timz.values, iso.values, zorder=3., c=VAR.var_kwargs[var]['isocol'], linewidth=VAR.linewidth,
                        linestyle=styles[i%4])

                proxy = plt.Line2D([-9999., -9999.], [-9999., -9999.], c=VAR.var_kwargs[var]['isocol'],
                                   linewidth=VAR.linewidth, linestyle=styles[i%4])
                LGD(ax, proxy, VAR.var_names[isolines[i]])

    if nplots % 2 == 0:
        ax.invert_yaxis()

    TINFO(ti, 2., 'Variable plot', verbose)

    return ax


def OV_floatvars(pathorprof, vars, period, log=False, levels=8, dmax = 250, isolines=None, discrete=False,
                 QCs=4, zone_id='SP', verbose=True):
    '''
    Shows the given variables of given float.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param vars: the variables to display. (str)
    :param period: Either the period bounds as tuple or list or the year of interest as integer. (tuple, list or int)
    :param levels: The isolines levels to use. If None, no isolines will be drawn. (1-D array-like)
    :param dmax: The maximum depth of the plot. (float, optional)
    :param isolines: The variable to use as isolines. (str, optional)
    :param discrete: Whether to plot colors as discrete bins. (boolean, default is True)
    :param QCs: The QC codes to keep in the data. (int or list of int, default is (1, 2, 8))
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''

    ti = t.time()

    if verbose:
        CPRINT('Starting overview plot...', attrs='BLUE', end='\r')

    if type(period) is int:
        datemin = '{}-01-01'.format(period)
        datemax = '{}-12-31'.format(period)
    else:
        datemin, datemax = period[0], period[1]

    datemin, datemax = FMT_date(datemin, 'dt', verbose=False), FMT_date(datemax, 'dt', verbose=False)

    if type(pathorprof) is xr.Dataset:
        prof = pathorprof
    else:
        prof = PRC(pathorprof)

    wmo = int(prof.PLATFORM_NUMBER.values[0])

    nplots = len(vars) + 1
    yspace = 0.88/nplots
    fig, axes = plt.subplots(nplots, 1, sharex=True)
    if type(axes) is not np.ndarray:
        axes = [axes]

    fig.suptitle('Variables of {}'.format(VAR.floats_names[wmo]) +
                 '\n \\normalsize {} to {} '.format(FMT_date(datemin, 'dt', verbose=False).strftime('%b %d, %Y'),
                                               FMT_date(datemax, 'dt', verbose=False).strftime('%b %d, %Y')))

    if isolines is not None:
        if type(isolines) is not list:
            isolines = [isolines for _ in range(len(vars))]
    else:
        isolines = [None for _ in range(len(vars))]

    if verbose:
        CPRINT('Plotting float variables and isolines...', attrs='BLUE', end='\r')

    for i in range(len(vars)-1, -1, -1):

        var = vars[i]

        try:

            ax = axes[i+1]
            ax.set_position([0.07, 0.93-(i+2)*yspace, 0.75, yspace-0.02])
            vmin, vmax = CMP_lims(prof[var].values[prof.PRES.values<dmax], out=5.)
            TS_sprof(prof, var, ax=ax, dmax=dmax, datemin=datemin, datemax=datemax, isolines=isolines[i],
                     vmin=vmin, vmax=vmax, levels=levels, discrete=discrete, verbose=False,
                     kwargs=VAR.var_kwargs[var])
            ax.set_ylim(0., dmax)
            ax.invert_yaxis()
            ax.set_ylabel('Depth ($m$)')
            ax.set_title('')

        except IndexError as e:

            if verbose:
                CPRINT('There is no {} data in the current Sprof.'.format(var), attrs='YELLOW')

    # MEI

    file = 'Files/MEI/meiv2.data'
    activity = GET_activity(prof, var='CHLA')
    data = np.genfromtxt(file, delimiter=',')
    MEI = [data[i + 1, j + 1] for i in range(np.shape(data)[0] - 1) for j in range(np.shape(data)[1] - 1)]
    T = np.array([dt.datetime(int(y), int(m), 15) for y in data[1:, 0] for m in data[0, 1:]])
    MEI = np.where(np.abs(MEI) > 10., np.nan, MEI)
    MEI = np.where(T < activity[0]-(activity[1]-activity[0])/2, np.nan, MEI)
    MEI = np.where(T > activity[1]+(activity[1]-activity[0])/2, np.nan, MEI)
    obj1 = axes[0].plot(T, MEI, c='purple', linewidth=2.5, label='MEI')
    axes[0].plot([-1000, -999], [0, 0], label='Chla/$b_{bp}$ correlation', c='darkblue')
    axes[0].set_ylabel('MEI')
    # axes[0].grid()
    axes[0].set_ylim(-np.max(np.abs(axes[0].get_ylim())), np.max(np.abs(axes[0].get_ylim())))
    axes[0].legend(loc='lower left')
    axes[0].set_position([0.07, 0.93 - yspace, 0.75, yspace - 0.02])
    axb = axes[0].twinx()
    axb.set_ylabel('Chl/$b_{bp}$ corr')
    axb.set_position([0.07, 0.93 - yspace, 0.75, yspace - 0.02])
    axb.plot(prof.JULD.values, prof.CORR_CB.values, c='darkblue')

    TINFO(ti, 50., 'Overview plot of float'.format(wmo), verbose)

    return plt.gcf()


def TS_diag(pathorprof, ax=None, var=None, vmin=None, vmax=None, tmin=None, tmax=None, smin=None, smax=None,
            extend='both', facecolor=(0.7, 0.7, 0.9), background=True, verbose=True):
    '''
    Plots the TS diagram associated to an ARGO S-profile.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param ax: The matplotlib axes. (matplotlib axes, optional)
    :param var: The ARGO variable colouring the scatter plot points. (str, optional)
    :param vmin: The minimum value in the colormap. (float, optional)
    :param vmax: The maximum value in the colormap. (float, optional)
    :param tmin: The minimum temperature value. (float, optional)
    :param tmax: The maximum temperature value. (float, optional)
    :param smin: The minimum salinity value. (float, optional)
    :param smax: The maximum salinity value. (float, optional)
    :param extend: Which end of the colorbar to extend to a larger span. (str in ['neither', 'both', 'min', 'max'],
     default is 'both')
    :param facecolor: The background color. (matplotlib color, default is (0.9, 0.9, 0.95))
    :param background: Whether to display density isolines. (boolean, default is True)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''

    ti = t.time()

    def aux(prof, var, vmin, vmax, tmin, tmax, smin, smax, ax, extend, background):

        X, Y = prof.PSAL_ADJUSTED.values.flatten(), prof.CT.values.flatten()
        xlabel, ylabel = VAR.var_labels['PSAL'], VAR.var_labels['CT']

        indexes = np.arange(np.size(X))
        np.random.shuffle(indexes)
        X, Y = X[list(indexes)], Y[list(indexes)]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if var is None:
            ax.scatter(X, Y, s=7, c='k')
        else:
            C = prof[var].values.flatten()
            C = C[list(indexes)]
            mask = ~np.isnan(C)
            C = C[mask]
            if C.size > 1:
                X, Y = X[mask], Y[mask]
                if vmin is None:
                    vmin = CMP_lims(C, 5.)[0]
                if vmax is None:
                    vmax = CMP_lims(C, 5.)[1]
                mappable = ax.scatter(X, Y, c=C, s=7, vmin=vmin, vmax=vmax, cmap=VAR.var_cmaps[var])
                CBAR(mappable, ax=ax, cbarlab=VAR.var_labels[var], extend=extend, scientific=False)
            else:
                ax.scatter(X, Y, s=7, c='k')

        if smin is None:
            smin = ax.get_xlim()[0]
        if smax is None:
            smax = ax.get_xlim()[1]
        if smin is None:
            tmin = ax.get_ylim()[0]
        if smin is None:
            tmax = ax.get_ylim()[1]

        ax.set_xlim(smin, smax)
        ax.set_ylim(tmin, tmax)

        if background:

            smin, smax = ax.get_xlim()
            tmin, tmax = ax.get_ylim()

            X, Y = np.meshgrid(np.linspace(smin, smax, 100), np.linspace(tmin, tmax, 100))
            sigma = sw.density.sigma0(X, Y)
            levels = np.arange(18., 35., 1.)

            CS = ax.contour(X, Y, sigma, colors='k', levels=levels, linewidths=VAR.linewidth, zorder=1)
            ax.clabel(CS, levels=levels, inline=True, inline_spacing=30, fmt='%1.1f', fontsize=11)
            proxy = plt.plot([], [], c='k', linewidth=VAR.linewidth)
            LGD(ax, proxy,  [r'$\sigma_0$ ($kg.m^{-3}$)'])

    if verbose:
        CPRINT('Starting TS diagram plot...', attrs='BLUE', end='\r')

    prof = PRC(pathorprof)
    wmo = GET_wmo(prof)

    if ax is None:

        _, ax = plt.subplots()

    aux(prof, var, vmin, vmax, tmin, tmax, smin, smax, ax, extend, background)

    ax.set_facecolor(facecolor)
    ax.set_title('TS diagram for float \#{}'.format(wmo))

    TINFO(ti, 1., 'TS diagram', verbose)

    return ax


def INSPECT(pathorprof, var='CHLA_PRC', dmax=250., pause=1., cmap='twilight'):
    '''
    Reviews all profiles of a certain variable.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param var: The variable to review. (str, default is 'CHLA_PRC')
    :param dmax: The maximum depth for the plot. (float, default is 250.)
    :param pause: The pause in between images in seconds. (float, default is 1.)
    :param cmap: The colormap for the season, that is the background color. (matplotlib colormap, default is twilight)
    :return: None
    '''

    def F(date, cmap):

        def is_leap_year(year):
            return year % 4 == 0 and (not year % 100 == 0 or year % 400 == 0)

        N = len(cmap.colors)
        date = FMT_date(date, 'dt', verbose=False)
        day_of_year = date.timetuple().tm_yday
        ndays = 366 if is_leap_year(date.year) else 365
        color = cmap.colors[int((day_of_year-1) / ndays * N)] + [0.8]

        return color

    prof = PRC(pathorprof)
    wmo = GET_wmo(prof)
    X, Y, T = prof[var].values, prof.PRES.values, FMT_date(prof.JULD.values, 'dt', verbose=False)
    X = np.where(Y * VAR.hf > dmax, np.nan, X)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(cmap)

    vmin, vmax = np.nanpercentile(X, 0.3), np.nanpercentile(X, 99.7)

    ax.set_xlim(vmin - (vmax-vmin)/5, vmax + (vmax-vmin)/5)
    ax.set_ylim(dmax, 0.)
    trick = ax.scatter([-1000., -1000.], [-1000., -1000.], c=[0., 1.], cmap=cmap)
    cbar = plt.colorbar(trick)
    cbar.set_label('Month of the year')
    cbar.set_ticks([i / 12 for i in range(13)])
    cbar.set_ticklabels([dt.datetime(2000, i, 1).strftime('%B')[:3] for i in range(1, 13)] + ['Jan'])
    ax.set_xlabel(VAR.var_labels[var])
    ax.set_ylabel(VAR.var_labels['PRES'])

    for k in range(np.size(prof.N_PROF.values)):

        ax.set_title('Reviewing {} profiles of float {}.'.format(var, wmo) + '.' * (k%3) + '-' * (2 - k%3))
        ax.set_facecolor(F(T[k], cmap))
        ax.plot(X[k][~np.isnan(X[k])], Y[k][~np.isnan(X[k])], color='yellow', linewidth=2.)
        ax.text(vmax - 0.1 * (vmax-vmin)/10, 0.95 * dmax, str(T[k].year), fontsize=15)

        plt.pause(pause)

        for artist in ax.collections + ax.lines + ax.texts:
            artist.remove()

        if np.isnan((X[k:])).all():
            break



def INSPECT_CHLA(pathorprof, dmax=250., pause=1., cmap='twilight'):
    '''
    Reviews all processed chl-a profiles.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param dmax: The maximum depth for the plot. (float, default is 250.)
    :param pause: The pause in between images in seconds. (float, default is 1.)
    :param cmap: The colormap for the season, that is the background color. (matplotlib colormap, default is twilight)
    :return: None
    '''

    def F(date, cmap):

        def is_leap_year(year):
            return year % 4 == 0 and (not year % 100 == 0 or year % 400 == 0)

        N = len(cmap.colors)
        date = FMT_date(date, 'dt', verbose=False)
        day_of_year = date.timetuple().tm_yday
        ndays = 366 if is_leap_year(date.year) else 365
        color = cmap.colors[int((day_of_year - 1) / ndays * N)] + [0.8]

        return color

    prof = PRC(pathorprof)
    wmo = GET_wmo(prof)
    X, Xb, Y, T = prof.CHLA_PRC.values, prof.CHLA.values, prof.PRES.values,\
        FMT_date(prof.JULD.values, 'dt', verbose=False)
    X = np.where(Y * VAR.hf > dmax, np.nan, X)
    Xb = np.where(Y * VAR.hf > dmax, np.nan, Xb)
    fig, ax = plt.subplots()
    axb = ax.twinx()
    cmap = plt.get_cmap(cmap)

    vmin, vmax = np.nanpercentile(X, 0.3), np.nanpercentile(X, 99.7)
    vminb, vmaxb = np.nanpercentile(Xb, 0.3), np.nanpercentile(Xb, 99.7)

    ax.set_xlim(vmin - (vmax - vmin) / 5, vmax + (vmax - vmin) / 5)
    axb.set_xlim(vminb - (vmaxb - vminb) / 5, vmaxb + (vmaxb - vminb) / 5)
    ax.set_ylim(dmax, 0.)
    axb.set_ylim(dmax, 0.)
    trick = ax.scatter([-1000., -1000.], [-1000., -1000.], c=[0., 1.], cmap=cmap)
    cbar = plt.colorbar(trick)
    cbar.set_label('Month of the year')
    cbar.set_ticks([i / 12 for i in range(13)])
    cbar.set_ticklabels([dt.datetime(2000, i, 1).strftime('%B')[:3] for i in range(1, 13)] + ['Jan'])
    ax.set_xlabel(VAR.var_labels['CHLA_PRC'])
    ax.set_ylabel(VAR.var_labels['PRES'])

    for k in range(np.size(prof.N_PROF.values)):

        ax.set_title('Reviewing CHLA_PRC profiles of float {}'.format(wmo))
        ax.set_facecolor(F(T[k], cmap))

        ln1 = axb.plot(Xb[k][~np.isnan(Xb[k])], Y[k][~np.isnan(Xb[k])], color='orange', linewidth=2.)
        ln2 = ax.plot(X[k][~np.isnan(X[k])], Y[k][~np.isnan(X[k])], color='yellow', linewidth=2.)
        ax.text(vmax - 0.1 * (vmax - vmin) / 10, 0.95 * dmax, str(T[k].year), fontsize=15)

        if k == 0:
            LGD(ax, [ln1, ln2], ['Raw', 'Processed'])

        plt.pause(pause)

        for artist in ax.collections + ax.lines + ax.texts + axb.collections + axb.lines + axb.texts:
            artist.remove()

        if np.isnan((X[k:])).all():
            break


def COR_vars(pathorprof, vars, depth=None, pmax = 3, verbose=True):
    '''
    Plots variables correlation plots.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path or the ARGO profile. (str, int or xarray)
    :param vars: The list of variables to check correlation between. (list)
    :param depth: The depth at which to check variables correlation. If not given, checks all. (float, default is None)
    :param pmax: The maximum number of graphs grid columns. (int, default is 3)
    :param verbose: Wether to print info to the console. (bool, default is True)
    :return: The matplotlib axes.
    '''
    ti = t.time()

    if verbose:
        CPRINT('Calculating correlation...', attrs='BLUE', end='\r')

    prof = PRC(pathorprof)
    wmo = GET_wmo(prof)

    n = len(vars)
    magicnumber = int((n ** 2 - n) / 2)
    fig, axes = plt.subplots(magicnumber // pmax if magicnumber % pmax == 0 else
                             magicnumber // pmax + 1, min(pmax, magicnumber))
    fig.suptitle('Correlation in float {} parameters'.format(VAR.floats_names[wmo]))
    if type(axes) is np.ndarray:
        axes = axes.flatten()
    else:
        axes = [axes]
    i = 0
    p = 0

    while i < n - 1:

        j = 0

        while j < n - i - 1:

            addtolabX = ''
            addtolabY = ''

            X = prof[vars[i]].values
            Y = prof[vars[i + j + 1]].values

            if np.size(np.shape(X)) > 1:

                if depth is None:

                    nX = X.flatten()

                else:

                    addtolabX = ' at {:.0f}m'.format(depth)
                    nX = np.nan * np.zeros(np.size(prof.N_PROF.values))

                    for k in range(np.size(prof.N_PROF.values)):
                        if np.sum(~np.isnan(X[k])) > 10:
                            nX[k] = np.interp(depth, prof.PRES.values[k][~np.isnan(X[k])] * VAR.hf,
                                              X[k][~np.isnan(X[k])])

                    X = nX

            if np.size(np.shape(Y)) > 1:

                if depth is None:

                    nY = Y.flatten()

                else:

                    addtolabY = ' at {:.0f}m'.format(depth)
                    nY = np.nan * np.zeros(np.size(prof.N_PROF.values))

                    for k in range(np.size(prof.N_PROF.values)):
                        if np.sum(~np.isnan(Y[k])) > 10:
                            nY[k] = np.interp(depth, prof.PRES.values[k][~np.isnan(Y[k])] * VAR.hf,
                                              Y[k][~np.isnan(Y[k])])

                    Y = nY

            X, Y = X[~np.isnan(Y)], Y[~np.isnan(Y)]
            X, Y = X[~np.isnan(X)], Y[~np.isnan(X)]

            reg = stats.linregress(X, Y)
            r2 = reg.rvalue ** 2
            I, S = reg.intercept, reg.slope
            f = lambda x: S * x + I

            axes[p].scatter(X, Y, s=7, c='k')
            axes[p].plot(axes[p].get_xlim(),
                         [f(axes[p].get_xlim()[0]), f(axes[p].get_xlim()[1])],
                         c='gray', linewidth=0.5, linestyle='dashed')
            axes[p].set_xlabel(VAR.var_labels[vars[i]] + addtolabX)
            axes[p].set_ylabel(VAR.var_labels[vars[i + j + 1]] + addtolabY)
            axes[p].set_title('$r^2 = {:.2f}$'.format(r2))

            p += 1
            j += 1

        i += 1

    fig.tight_layout()

    TINFO(ti, 4., 'Plotted correlation plots', verbose)

    return axes