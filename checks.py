# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the module dedicated to processing checks. TST_ functions do the computation whereas CHK_ functions only
 check the derived variables in the processed Sprofiles.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''

from profiles import *
from classification import *
from plotargo import *


# Functions


def TST_despike(wmo, var='BBP700', n=3, indexes=None, thresh_up=0.95, thresh_down=None, smooth=VAR.Zsmooth):
    '''
    Tests the despiking algorythm on a profile variable.

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :param wmo: The test float wmo. (int)
    :param var: The test variable. (str, default is BBP700)
    :param n: The number of test windows. (int, default is 3)
    :param indexes: Specific profiles numbers. If not provided, tests randomly. (list of int, optional)
    :param thresh_up: The quantile reference to erase spikes. The thresh_up biggest residuals are being suppressed.
     (float, default is 0.95).
    :param thresh_down: Same as thresh_up but for negative spikes. (float, default is 1-thresh_up)
    :param smooth: The smoothing constant for FLT_mean. (float, default is VAR.Zsmooth)
    :return: None
    '''
    prof = RAW(wmo)

    if indexes is None:
        nvalues = np.sum(~np.isnan(np.where(prof.PRES.values<200., prof[var].values, np.nan).astype(float)), axis=1)
        indexes = np.random.choice(prof.N_PROF.values[nvalues>10], n)
    else:
        if type(indexes) is not list:
            indexes = [indexes]
    n = len(indexes)

    fig, ax = plt.subplots(1, n, sharey=True)
    if n == 1:
        ax = [ax]
    ax[0].set_ylabel('Depth ($m$)')
    ax[0].set_ylim(250., 0.)

    if thresh_down is None:
        thresh_down = 1 - thresh_up

    for i in range(n):

        k = indexes[i]

        x, y = prof[var].values[k], prof.PRES.values[k]
        x, y = x[~np.isnan(x)], y[~np.isnan(x)]
        xs = FLT_mean(x, y, smooth=smooth, verbose=False)

        spikes = x - FLT_mean(x, y, smooth=smooth)
        threshold_up = np.nanquantile(spikes, thresh_up)
        threshold_down = np.nanquantile(spikes, thresh_down)

        xds, yds = x[spikes<threshold_up], y[spikes<threshold_up]
        xds, yds = xds[spikes[spikes<threshold_up]>threshold_down], yds[spikes[spikes<threshold_up]>threshold_down]
        xdss = FLT_mean(xds, yds, smooth=smooth)

        xds = np.where(spikes<threshold_down, np.interp(y, yds, xdss), x)
        xds = np.where(spikes>threshold_up, np.interp(y, yds, xdss), xds)

        # ax[i].plot(spikes, y)
        # ax[i].plot([threshold_up, threshold_up], ax[i].get_ylim(), c='red', linestyle='dashed')
        # ax[i].plot([threshold_down, threshold_down], ax[i].get_ylim(), c='blue', linestyle='dashed')
        ax[i].plot(x, y)
        ax[i].plot(xds, y)

    fig.suptitle('Despiking {} of float {}'.format(var, FMT_wmo(wmo)))


def TST_mld(wmo, n=3, indexes=None):
    '''
    Tests the mld detection by threshold (0.03 and 0.125 in density and 0.2 in temperature) on floats profiles.

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :param wmo: The test float wmo. (int)
    :param n: The number of test windows. (int, default is 3)
    :param indexes: Specific profiles numbers. If not provided, tests randomly. (list of int, optional)
    :return: None
    '''

    prof = RAW(wmo)

    if indexes is None:
        nvalues = np.sum(~np.isnan(np.where(prof.PRES.values<200., prof.TEMP.values, np.nan).astype(float)), axis=1)
        indexes = np.random.choice(prof.N_PROF.values[nvalues>10], n)
    else:
        if type(indexes) is not list:
            indexes = [indexes]

    n = len(indexes)

    fig, ax = plt.subplots(1, n, sharey=True)
    if n == 1:
        ax = [ax]
    ax[0].set_ylabel('Depth ($m$)')
    ax[0].set_ylim(250., 0.)

    for k, i in enumerate(indexes):

        ax_d = ax[k].twiny()

        pres_d, pres_t, sal_i, temp_i = prof.PRES.values[i], prof.PRES.values[i], prof.PSAL.values[i], prof.TEMP.values[i]
        ct_i = sw.conversions.CT_from_t(sal_i, temp_i, pres_d)
        sig0_i = sw.density.sigma0(sal_i, ct_i)

        sig0_i, pres_d = sig0_i[~np.isnan(sig0_i)], pres_d[~np.isnan(sig0_i)]
        ct_i, pres_t = ct_i[~np.isnan(ct_i)], pres_t[~np.isnan(ct_i)]

        if np.size(pres_d) > 10 and np.min(pres_d) < 10. and np.max(pres_d) > 10.:

            ct_10m = np.interp(10, pres_t[~np.isnan(ct_i)], ct_i[~np.isnan(ct_i)])
            ct_thresh = ct_10m - 0.2
            ct_i, pres_t = ct_i[pres_t>10.], pres_t[pres_t>10.]
            mld_t = np.min(pres_t[ct_i < ct_thresh]) * VAR.hf
            qc_t = 1 - np.std(ct_i[pres_t < mld_t])/np.std(ct_i[pres_t<1.5*mld_t])

            sig0_10m = np.interp(10, pres_d[~np.isnan(sig0_i)], sig0_i[~np.isnan(sig0_i)])
            sig0_thresh125 = sig0_10m + 0.125
            sig0_thresh03 = sig0_10m + 0.03
            sig0_i, pres_d = sig0_i[pres_d>10.], pres_d[pres_d>10.]
            mld_d125 = np.min(pres_d[sig0_i > sig0_thresh125]) * VAR.hf
            mld_d03 = np.min(pres_d[sig0_i > sig0_thresh03]) * VAR.hf
            qc_d125 = 1 - np.std(sig0_i[pres_d<mld_d125])/np.std(sig0_i[pres_d<1.5*mld_d125])
            qc_d03 = 1 - np.std(sig0_i[pres_d<mld_d03])/np.std(sig0_i[pres_d<1.5*mld_d03])

            ln1 = ax[k].plot(sig0_i, pres_d*VAR.hf, c='darkblue', label='Density')
            ln2 = ax[k].plot([np.min(sig0_i), np.max(sig0_i)], [mld_d125, mld_d125],
                             label='$MLD_{{d125}}={:.1f}m$'.format(mld_d125), c='k')
            ln3 = ax[k].plot([np.min(sig0_i), np.max(sig0_i)], [mld_d03, mld_d03],
                             label='$MLD_{{d03}}={:.1f}m$'.format(mld_d03), c='k', linestyle='--')
            ln4 = ax_d.plot(ct_i, pres_t*VAR.hf, c='crimson', label='Temperature')
            ln5 = ax_d.plot([np.min(ct_i), np.max(ct_i)], [mld_t, mld_t], label='$MLD_t={:.1f}m$'.format(mld_t),
                            c='orange')

            ax[k].set_title('Profile \#{} :\n$QI_t: {:.2f}$ / $QI_{{d125}}: {:.2f}$, / $QI_{{d03}}: {:.2f}$'
                            .format(i, qc_t, qc_d125, qc_d03), fontsize=15)
            ax[k].set_xlabel('Density ($kg.m^{-3}$)')
            ax_d.set_xlabel('Temperature ($^{\circ}C$)')
            lns = ln1 + ln2 + ln3 + ln4 + ln5
            labs = [l.get_label() for l in lns]
            ax[k].legend(lns, labs, loc='lower left')

    fig.suptitle('Examples of MLD values from density ($d$) and temperature ($t$) for float \#{}'
                 .format(int(prof.PLATFORM_NUMBER.values[0])))
    fig.tight_layout()
    CPRINT('Tested MLD.', attrs='BLUE')


def TST_X11(wmo, var='DOWN_IRRADIANCE490', n=1):
    '''
    Tests the slope factor derivation algorythm of Xing et al., 2011.

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :param wmo: The test float wmo. (int)
    :param var: The wavelength of test. (str, DOWN_IRRADIANCE490, 412 or 380, recommanded is 490)
    :param n: The number of test windows. (int, default is 3)
    :return: None
    '''
    Kw, Chi, e = None, None, None

    if var[-3:] == '490':
        Kw = 0.0166
        Chi = 0.07242
        e = 0.68955
    elif var[-3:] == '412':
        Kw = 0.00812
        Chi = 0.12259
        e = 0.65175
    elif var[-3:] == '380':
        Kw = 0.0151
        Chi = 0.127
        e = 0.685

    prof = RAW(wmo)
    pres, irr, chl = prof.PRES.values, prof[var].values, prof.CHLA.values
    chl = CMP_chlfeatures(wmo, verbose=False)[0]

    for _ in range(n):

        i = np.random.randint(np.shape(pres)[0])

        P, I = pres[i], irr[i]
        P, I = P[~np.isnan(I)], I[~np.isnan(I)]

        if np.size(I) > 10:

            E0 = np.nanmax(I)
            P, I = P[I<E0/2], I[I<E0/2]
            P, I = P[I>E0/100], I[I>E0/100]

        while np.size(I) < 10:

            i = np.random.randint(np.shape(pres)[0])

            P, I = pres[i], irr[i]
            P, I = P[~np.isnan(I)], I[~np.isnan(I)]

            if np.size(I) > 10:

                E0 = np.nanmax(I)
                P, I = P[I < E0 / 2], I[I < E0 / 2]
                P, I = P[I > E0 / 100], I[I > E0 / 100]

        try:
            C = np.interp(P, pres[i][~np.isnan(chl[i])], chl[i][~np.isnan(chl[i])])
        except ValueError:
            raise Exception('No chlorophyll available.')

        indexes = np.arange(np.size(I))
        An = np.log(I) + P * Kw
        An, indexes = An[1:], indexes[1:]
        Cn = Chi * np.array([np.sum([(C[k]**e + C[k+1]**e) / 2 * (P[k+1] - P[k]) for k in range(l)])
                             for l in range(1, np.size(P))])
        N = np.size(Cn)

        # Applying criteria

        reginit = stats.linregress(Cn, An)
        s, o, r = reginit.slope, reginit.intercept, reginit.rvalue
        if s > 0:
            QI = 4
        else:
            QI = 1 if r ** 2 > 0.9 else 2 if r ** 2 > 0.8 else 3
        threshinit = (s*Cn + o) - 0.02*np.abs((s*Cn + o))

        An_init, Cn_init = An, Cn

        mask = An > threshinit

        Cn, An, indexes = Cn[mask], An[mask], indexes[mask]
        iter = 0

        while r**2 < 0.98 and iter < 5 and np.size(Cn) > 5:

            iter += 1
            old_r = r
            reg = stats.linregress(Cn, An)
            s, o, r = reg.slope, reg.intercept, reg.rvalue
            if r == old_r:
                break
            thresh = (s*Cn + o) - 0.02*np.abs((s*Cn + o))
            mask = An > thresh
            Cn, An, indexes = Cn[mask], An[mask], indexes[mask]

        rem = np.size(Cn)
        INT = np.nansum([(An_init[k] - (s*Cn_init[k]+o))**2 * (Cn_init[k] - Cn_init[k-1])
                        for k in range(1, np.size(An_init))]) / (np.nanmax(Cn_init) - np.nanmin(Cn_init)) /\
             (np.nanmax(An_init) - np.nanmin(An_init))
        percentclouds = (1 - np.size(Cn)/N)*100
        slope = np.exp(np.log(-s)/e)

        QI = 1 if INT <= 0.05 else 2 if INT <= 0.1 else 3 if INT <= 0.2 else 4
        if QI < 3 and rem <= 20:
            QI = 3
        if QI < 4 and rem <= 5:
            QI = 4
        if QI < 3 and percentclouds >= 90:
            QI = 3
        if QI < 4 and percentclouds >= 95:
            QI = 4
        if np.isnan(slope):
            QI = 4

        # Reconstructed points in irradiance profile
        indexesbar = []
        for k in range(N):
            if k not in indexes:
                indexesbar.append(k)
        indexesbar = np.array(indexesbar)
        reconstructed = np.nan * np.zeros(np.size(indexesbar))
        for l, k in enumerate(indexesbar):
            reconstructed[l] = o - P[k]*Kw + s*Chi*np.sum([(C[j]**e + C[j+1]**e) / 2 * (P[j+1] - P[j])
                                                           for j in range(k)])

        fig, ax = plt.subplots(1, 3)
        fig.suptitle('F{}P{} (X11 criteria)'.format(int(prof.PLATFORM_NUMBER.values[0]), i))

        ax[0].set_ylabel('$A_n$')
        ax[0].set_xlabel('$C_n$')
        ax[0].scatter(Cn_init, An_init, c='crimson', s=10, alpha=0.7, label='Rejected (clouds)')
        ax[0].plot(ax[0].get_xlim(), s*np.array(ax[0].get_xlim())+o, c='crimson',
                   label='Regression: {:.2f}'.format(slope))
        ax[0].set_title('$r^{{2}}$: {:.2f}, QI: {:.0f}'.format(r**2, QI))
        ax[0].scatter(Cn, An, c='b', label='Kept points')
        ax[0].legend()

        ax[1].set_ylim(200., 0.)
        ax[1].set_title('Irradiance profile: {:.0f}\% cloudy'.format(percentclouds))
        ax[1].set_ylabel('P (dbar)')
        ax[1].set_xlabel('$\\log(I)$')
        ax[1].scatter(np.log(irr[i][~np.isnan(irr[i])]), pres[i][~np.isnan(irr[i])], c='k', s=5, alpha=0.3, label='Raw')
        ax[1].scatter(np.log(I), P, c='crimson', s=10, alpha=0.5, label='Rejected (clouds)')
        ax[1].scatter(np.log(I)[indexes], P[indexes], c='b', label='Kept points')
        ax[1].scatter(reconstructed, P[indexesbar], c='purple', s=15, alpha=0.5, label='Reconstructed points')
        ax[1].scatter(np.exp(o), 0., marker = 'x', s=40, c='purple', label='$E_{O}$')
        ax[1].legend()

        ax[2].set_ylim(200., 0.)
        ax[2].set_ylabel('P (dbar)')
        ax[2].set_xlabel('$[Chl A]$')
        ax[2].set_title('Chlorophyll profile')
        ax[2].plot(chl[i][~np.isnan(chl[i])], pres[i][~np.isnan(chl[i])], c='crimson', label='Raw')
        ax[2].plot(np.exp(np.log(-s)/e)*chl[i][~np.isnan(chl[i])], pres[i][~np.isnan(chl[i])], c='g', label='Corrected')
        ax[2].legend()

        plt.tight_layout()


def CHK_mld(prc, n=3, dmax=170., verbose=True):

    TI = t.time()

    def CHK_one(prof, ax, axb, indexes, dmax):

        for k, i in enumerate(indexes):

            ax[k].set_title('Profile \#{}'.format(i), fontsize=15)
            ax[k].set_xlabel('Density ($kg.m^{-3}$)')
            axb[k].set_xlabel('Temperature ($^{\circ}C$)')

            pres, sal, temp, sig0 = prof.PRES.values[i], prof.PSAL.values[i], prof.TEMP.values[i], prof.SIG0.values[i]
            sal, temp = np.where(pres > dmax, np.nan, sal), np.where(pres > dmax, np.nan, temp)
            mld_d125 = prof.MLD_S125.values[i]
            mld_d03 = prof.MLD_S03.values[i]
            mld_t02 = prof.MLD_T02.values[i]
            qc_d125 = prof.MLD_S125_QI.values[i]
            qc_d03 = prof.MLD_S03_QI.values[i]
            qc_t02 = prof.MLD_T02_QI.values[i]

            ln1 = ax[k].plot(sig0[~np.isnan(sig0)], pres[~np.isnan(sig0)]*VAR.hf, c='darkblue', label='Density')
            ln2 = axb[k].plot(temp[~np.isnan(temp)], pres[~np.isnan(temp)]*VAR.hf, c='crimson', label='Temperature')

            xlims = np.nanmin(sig0) - np.nanstd(sig0)/4, np.nanmax(sig0) + np.nanstd(sig0)/4
            ax[k].set_xlim(xlims)
            xlimsb = np.nanmin(temp) - np.nanstd(temp)/4, np.nanmax(temp) + np.nanstd(temp)/4
            axb[k].set_xlim(xlimsb)

            ln4 = ax[k].plot(xlims, [mld_d03, mld_d03], label='$MLD_{{d03}} ({:.2f})$'
                             .format(qc_d03), c='darkmagenta')
            ln3 = ax[k].plot(xlims, [mld_d125, mld_d125], label='$MLD_{{d125}} ({:.2f})$'
                             .format(qc_d125), c='k', linestyle='dashed')
            ln5 = axb[k].plot(xlimsb, [mld_t02, mld_t02], label='$MLD_{{t02}} ({:.2f})$'
                             .format(qc_t02), c='k', linestyle='dotted')

            LGD(ax[k], ln1 + ln2 + ln3 + ln4 + ln5, [l.get_label() for l in (ln1 + ln2 + ln3 + ln4 + ln5)])


    prof = PRC(prc)

    fig, ax = plt.subplots(1, n, sharey=True)
    axb = [a.twiny() for a in ax]
    if n == 1:
        ax = [ax]
    ax[0].set_ylabel('Depth ($m$)')
    ax[0].set_ylim(dmax, 0.)
    fig.suptitle('Examples of MLD values from density ($d$) and temperature ($t$) for float \#{}'
                 .format(VAR.floats_names[GET_wmo(prof)]))

    while True:

        ti = t.time()

        nvalues = np.sum(~np.isnan(np.where(prof.PRES.values < 200., prof.TEMP.values, np.nan).astype(float)), axis=1)
        indexes = np.random.choice(np.arange((np.size(prof.N_PROF.values)))[nvalues > 10], n).astype(int)
        indexes = np.sort(indexes)

        CHK_one(prof, ax, axb, indexes, dmax)
        fig.tight_layout()
        CPRINT('Processing...', attrs='PURPLE', end='\r')
        plt.draw()
        plt.pause(3.)

        CPRINT('Check plot done ({}). Next? (y/n)'.format(FMT_secs(t.time() - ti)), attrs='PURPLE', end='\r')
        ans = INP_timeout(30.)
        if ans == 'y' or ans =='Y':
            for a in ax:
                for artist in a.collections + a.lines + a.texts:
                    artist.remove()
                a.get_legend().remove()
            for a in axb:
                for artist in a.collections + a.lines + a.texts:
                    artist.remove()
        else:
            break

    TINFO(TI, np.nan, 'Checked MLD', verbose)


def CHK_irrfit(prc, dmax=150., verbose=True):

    TI = t.time()

    def CHK_one(axes, index, dmax):

        axes[0].set_title('DOWNWELLING_PAR', fontsize=15)
        axes[1].set_title('DOWN_IRRADIANCE490', fontsize=15)
        axes[2].set_title('DOWN_IRRADIANCE412', fontsize=15)
        axes[3].set_title('DOWN_IRRADIANCE380', fontsize=15)

        pres, par, par_fit, par_cld, i490, i490_fit, i490_cld, i412, i412_fit, i412_cld, i380, i380_fit, i380_cld =\
            prof.PRES.values[index], prof.DOWNWELLING_PAR.values[index], prof.DOWNWELLING_PAR_FIT.values[index],\
                prof.DOWNWELLING_PAR_CLD.values[index], prof.DOWN_IRRADIANCE490.values[index], \
                prof.DOWN_IRRADIANCE490_FIT.values[index], prof.DOWN_IRRADIANCE490_CLD.values[index], \
                prof.DOWN_IRRADIANCE412.values[index], prof.DOWN_IRRADIANCE412_FIT.values[index], \
                prof.DOWN_IRRADIANCE412_CLD.values[index], prof.DOWN_IRRADIANCE380.values[index],\
                prof.DOWN_IRRADIANCE380_FIT.values[index], prof.DOWN_IRRADIANCE380_CLD.values[index]

        par, par_fit, par_cld = np.where(pres > dmax, np.nan, par), np.where(pres > dmax, np.nan, par_fit), \
            np.where(pres > dmax, np.nan, par_cld)
        i490, i490_fit, i490_cld = np.where(pres > dmax, np.nan, i490), np.where(pres > dmax, np.nan, i490_fit), \
            np.where(pres > dmax, np.nan, i490_cld)
        i412, i412_fit, i490_cld = np.where(pres > dmax, np.nan, i412), np.where(pres > dmax, np.nan, i412_fit), \
            np.where(pres > dmax, np.nan, i412_cld)
        i380, i380_fit, i490_cld = np.where(pres > dmax, np.nan, i380), np.where(pres > dmax, np.nan, i380_fit), \
            np.where(pres > dmax, np.nan, i380_cld)

        zeu = prof.ZEU.values[index]
        fg_par = prof.DOWNWELLING_PAR_FLG.values[index].squeeze()
        fg_490 = prof.DOWN_IRRADIANCE490_FLG.values[index].squeeze()
        fg_412 = prof.DOWN_IRRADIANCE412_FLG.values[index].squeeze()
        fg_380 = prof.DOWN_IRRADIANCE380_FLG.values[index].squeeze()

        axes[0].set_xlim(np.nanmin(par) - np.nanstd(par)/4, np.nanmax(par) + np.nanstd(par)/4)
        axes[1].set_xlim(np.nanmin(i490) - np.nanstd(i490)/4, np.nanmax(i490) + np.nanstd(i490)/4)
        axes[2].set_xlim(np.nanmin(i412) - np.nanstd(i412)/4, np.nanmax(i412) + np.nanstd(i412)/4)
        axes[3].set_xlim(np.nanmin(i380) - np.nanstd(i380)/4, np.nanmax(i380) + np.nanstd(i380)/4)

        ln1 = axes[0].plot(par[~np.isnan(par)], pres[~np.isnan(par)] * VAR.hf, c='darkblue',
                           label='Profile ({:.0f})'.format(fg_par))
        ln2 = axes[0].plot(par_fit[~np.isnan(par_fit)], pres[~np.isnan(par_fit)]*VAR.hf, c='crimson', label='Fit')
        ln3 = axes[0].plot(axes[0].get_xlim(), [zeu, zeu], label='$Z_{{eu}}$', c='darkmagenta', linestyle='dashed')
        sc = axes[0].scatter(par[par_cld.astype(bool)], pres[par_cld.astype(bool)], c='magenta', s=12, zorder=3,
                             label='Clouds')

        axes[1].plot(i490[~np.isnan(i490)], pres[~np.isnan(i490)] * VAR.hf, c='darkblue',
                     label='Profile ({:.0f})'.format(fg_490))
        axes[1].plot(i490_fit[~np.isnan(i490_fit)], pres[~np.isnan(i490_fit)]*VAR.hf, c='crimson', label='Fit')
        sc = axes[1].scatter(i490[i490_cld.astype(bool)], pres[i490_cld.astype(bool)], c='magenta', s=12, zorder=3,
                             label='Clouds')

        axes[2].plot(i412[~np.isnan(i412)], pres[~np.isnan(i412)] * VAR.hf, c='darkblue',
                     label='Profile ({:.0f})'.format(fg_412))
        axes[2].plot(i412_fit[~np.isnan(i412_fit)], pres[~np.isnan(i412_fit)]*VAR.hf, c='crimson', label='Fit')
        sc = axes[2].scatter(i412[i412_cld.astype(bool)], pres[i412_cld.astype(bool)], c='magenta', s=12, zorder=3,
                             label='Clouds')

        axes[3].plot(i380[~np.isnan(i380)], pres[~np.isnan(i380)] * VAR.hf, c='darkblue',
                     label='Profile ({:.0f})'.format(fg_380))
        axes[3].plot(i380_fit[~np.isnan(i380_fit)], pres[~np.isnan(i380_fit)]*VAR.hf, c='crimson', label='Fit')
        sc = axes[3].scatter(i380[i380_cld.astype(bool)], pres[i380_cld.astype(bool)], c='magenta', s=12, zorder=3,
                             label='Clouds')

        LGD(axes[3], ln1 + ln2 + ln3 + [sc], [l.get_label() for l in (ln1 + ln2 + ln3 + [sc])])

    prof = PRC(prc)

    fig, axes = plt.subplots(1, 4, sharey=True)
    axes[0].set_ylabel('Depth ($m$)')
    axes[0].set_ylim(dmax, 0.)
    fig.suptitle('Examples of irradiance profiles polynomial fits for float \#{} (spline degree = {})'
                 .format(VAR.floats_names[GET_wmo(prof)], VAR.irrpoly_order))

    while True:

        ti = t.time()

        nvalues = np.sum(~np.isnan(np.where(prof.PRES.values < 200., prof.DOWNWELLING_PAR.values, np.nan)
                                   .astype(float)), axis=1)
        index = np.random.choice(np.arange((np.size(prof.N_PROF.values)))[nvalues > 10], 1).astype(int)

        CHK_one(axes, index, dmax)
        fig.tight_layout()

        CPRINT('Processing...', attrs='PURPLE', end='\r')
        plt.draw()
        plt.pause(1.)

        CPRINT('Check plot done ({}). Next? (y/n)'.format(FMT_secs(t.time() - ti)), attrs='PURPLE', end='\r')
        ans = INP_timeout(30.)
        if ans == 'y' or ans =='Y':
            for a in axes:
                for artist in a.collections + a.lines + a.texts:
                    artist.remove()
            axes[3].get_legend().remove()
        else:
            break

    TINFO(TI, np.nan, 'Checked irradiance profiles', verbose)


def CHK_chlafit(prc, n=3, dmax=250., verbose=True):

    TI = t.time()

    def Fsgm(z, Ks, Z12s, s):
        return Ks / (1 + np.exp((z - Z12s) * s))

    def Fexp(z, Ke, Z12e):
        return Ke * np.exp(-np.log(2) * z / Z12e)

    def Fgss(z, Kg, Zm, sigg):
        return Kg * np.exp(-((z - Zm) / sigg) ** 2)

    def Fsum(z, Ke, Z12e, Km, Zm, sigg):
        return Fexp(z, Ke, Z12e) + Fgss(z, Km, Zm, sigg)

    def CHK_one(prof, pparams, ax, indexes, dmax):

        for k, i in enumerate(indexes):

            ax[k].set_xlabel('Chl-a ($mg.m^{-3}$)')

            pres, chl = prof.PRES.values[i], prof.CHLA_PRC.values[i]
            pres, chl = pres[pres < dmax], chl[pres < dmax]

            params = pparams.TYPE.values[i], pparams.KSGM.values[i], pparams.Z12SGM.values[i], pparams.SSGM.values[i], \
                pparams.KGSS.values[i], pparams.ZSCM.values[i], pparams.WSCM.values[i], pparams.KEXP.values[i], \
                pparams.Z12EXP.values[i], pparams.R2.values[i], pparams.SLOPE.values[i], pparams.INTER.values[i]
            ax[k].set_title('Profile \#{}\n$r^2={:.2f}$ / $slope={:.2f}$ / $intercept={:.2f}$'
                            .format(i, params[-3], params[-2], params[-1]), fontsize=13)

            ln1 = ax[k].plot(chl[~np.isnan(chl)], pres[~np.isnan(chl)]*VAR.hf, c='darkblue', linewidth=2.,
                             label='Chl-a', zorder=1)

            xlims = np.nanmin(chl) - np.nanstd(chl)/4, np.nanmax(chl) + np.nanstd(chl)/4
            ax[k].set_xlim(xlims)

            if params[0] == 1.:
                ln2 = ax[k].plot(Fsum(pres, params[7], params[8], params[4], params[5], params[6]), pres, linewidth=1.,
                                 c='magenta', label='SCM type fit', zorder=2)
                ln3 = ax[k].plot(xlims, [params[5], params[5]], linestyle='dashed', linewidth=1., label='SCM depth')
                ln4 = ax[k].plot([params[4], params[4]], [params[5] - params[6]/2, params[5] + params[6]/2],
                                 linestyle='dotted', linewidth=1., c='k', label='SCM width')
                ax[k].scatter([params[4], params[4]], [params[5] - params[6]/2, params[5] + params[6]/2], s=40,
                                    color='k')
                sc1 = ax[k].scatter([params[7]], [0.], s=40, color='g', label='$Chl_{surf}$')
                art = [sc1] + ln1 + ln2 + ln3 + ln4
                LGD(ax[k], art, [e.get_label() for e in art])
            elif params[0] == 2.:
                ln2 = ax[k].plot(Fsgm(pres, params[1], params[2], params[3]), pres, linewidth=1.,
                                 c='g', label='SCM type fit', zorder=2)
                sc1 = ax[k].scatter([params[1]], [0.], s=40, color='g', label='$Chl_{surf}$')
                sc2 = ax[k].scatter([params[1]/2, params[1]/2], [params[2], params[2]], linestyle='dashed',
                                    linewidth=1., label='$Z_{1/2}$', c='r')
                art = [sc1] + [sc2] + ln1 + ln2
                LGD(ax[k], art, [e.get_label() for e in art])
            else:
                pass

    prof = PRC(prc)
    PPARAMS = CLS_TYPE(prof)

    fig, ax = plt.subplots(1, n, sharey=True)
    if n == 1:
        ax = [ax]
    ax[0].set_ylabel('Depth ($m$)')
    ax[0].set_ylim(dmax, 0.)

    fig.suptitle('Examples of CHLA profiles fits for float \#{}'.format(VAR.floats_names[GET_wmo(prof)]))

    while True:

        ti = t.time()

        nvalues = np.sum(~np.isnan(np.where(prof.PRES.values < 200., prof.CHLA_PRC.values, np.nan).astype(float)),
                         axis=1)
        indexes = np.random.choice(np.arange((np.size(prof.N_PROF.values)))[nvalues > 10], n).astype(int)
        indexes = np.sort(indexes)

        CHK_one(prof, PPARAMS, ax, indexes, dmax)

        fig.tight_layout()
        CPRINT('Processing...', attrs='PURPLE', end='\r')
        plt.draw()
        plt.pause(3.)

        CPRINT('Check plot done ({}). Next? (y/n)'.format(FMT_secs(t.time() - ti)), attrs='PURPLE', end='\r')
        ans = INP_timeout(30.)
        if ans == 'y' or ans =='Y':
            for a in ax:
                if a is not None:
                    for artist in a.collections + a.lines + a.texts:
                        artist.remove()
                    if a.get_legend() is not None:
                        a.get_legend().remove()
        else:
            break

    TINFO(TI, np.nan, 'Checked chl-a fits', verbose)


def CHK_chla(prc, n=3, dmax=250., verbose=True):

    TI = t.time()

    def CHK_one(prof, ax, axb, i):

        pres, chl, chl_prc, par, par_f = prof.PRES.values[i], prof.CHLA.values[i], prof.CHLA_PRC.values[i], \
            prof.DOWNWELLING_PAR.values[i], prof.DOWNWELLING_PAR_FIT.values[i]
        pres, chl, chl_prc, par, par_f = pres[pres < dmax], chl[pres < dmax], chl_prc[pres < dmax], \
            par[pres < dmax], par_f[pres < dmax]
        iso15, zeu, mld = prof.ISO15.values[i], prof.ZEU.values[i], prof.MLD_S03.values[i]
        slope = np.median(prof.SLOPEF490.values[prof.DOWNWELLING_PAR_FLG.values < 3])
        sgm_share = chl_prc * (1-1/SGM(np.ones(np.size(par_f)), par_f))
        chl_prc_nq = chl_prc - sgm_share

        ln3 = ax.plot(chl[~np.isnan(chl)] * slope, pres[~np.isnan(chl)]*VAR.hf, c='darkblue', linewidth=1., alpha=0.8,
                         label='Raw chl-a / slope factor', zorder=2)
        ln2 = ax.plot(chl_prc[~np.isnan(chl_prc)], pres[~np.isnan(chl_prc)]*VAR.hf, c='darkblue', linewidth=2.,
                         label='Processed chl-a', zorder=2)
        ln1 = ax.plot(chl_prc_nq[~np.isnan(chl_prc_nq)], pres[~np.isnan(chl_prc_nq)]*VAR.hf, c='g', linewidth=2.,
                         label='Chl-a prc. - NPQ', zorder=2)
        ln6 = ax.plot(ax.get_xlim(), [mld, mld], c='k', linestyle='dashed', label='$MLD_{\\Delta\\sigma=0.03}$')

        xlims = np.nanmin(chl_prc) - np.nanstd(chl_prc)/4, np.nanmax(chl_prc) + np.nanstd(chl_prc)/4
        ax.set_xlim(xlims)

        ln5 = axb.plot(par[~np.isnan(par)], pres[~np.isnan(par)]*VAR.hf, c='orange', linewidth=1., alpha=0.8,
                         label='Raw PAR', zorder=1)
        ln4 = axb.plot(par_f[~np.isnan(par_f)], pres[~np.isnan(par_f)]*VAR.hf, c='orange', linewidth=2.,
                         label='Processed PAR', zorder=1)
        sc1 = axb.scatter(15., iso15, c='r', label='Isolume 15')
        sc2 = axb.scatter(par_f[0]/100, zeu, c='b', label='$Z_{eu}$')

        xlims = np.nanmin(par_f) - np.nanstd(par_f) / 4, np.nanmax(par_f) + np.nanstd(par_f) / 4
        axb.set_xlim(xlims)

        art = ln1 + ln2 + ln3 + ln4 + ln5 + ln6 + [sc1] + [sc2]
        LGD(ax, art, [e.get_label() for e in art])

    prof = PRC(prc)

    fig, ax = plt.subplots(1, n, sharey=True)
    if n == 1:
        ax = [ax]
    axb = [a.twiny() for a in ax]
    ax[0].set_ylabel('Depth ($m$)')
    ax[0].set_ylim(dmax, 0.)
    for a in ax:
        a.set_xlabel('Chl-a ($mg.m^{-3}$)')
    for a in axb:
        a.set_xlabel('PAR ($\mu \mathit{mol} E.m^{-2}.s^{-1}$)')

    fig.suptitle('Examples of CHLA profiles for float \#{}'.format(VAR.floats_names[GET_wmo(prof)]))

    while True:

        ti = t.time()

        CPRINT('Processing...', attrs='PURPLE', end='\r')

        nvalues = np.sum(~np.isnan(np.where(prof.PRES.values < 200., prof.CHLA_PRC.values, np.nan).astype(float)),
                         axis=1)
        indexes = np.random.choice(np.arange((np.size(prof.N_PROF.values)))[nvalues > 10], n).astype(int)
        indexes = np.sort(indexes)

        for k, i in enumerate(indexes):
            CHK_one(prof, ax[k], axb[k], i)

        fig.tight_layout()
        plt.draw()
        plt.pause(3.)

        CPRINT('Check plot done ({}). Next? (y/n)'.format(FMT_secs(t.time() - ti)), attrs='PURPLE', end='\r')
        ans = INP_timeout(30.)
        if ans == 'y' or ans =='Y':
            for a in list(ax) + list(axb):
                if a is not None:
                    for artist in a.collections + a.lines + a.texts:
                        artist.remove()
                    if a.get_legend() is not None:
                        a.get_legend().remove()
        else:
            break

    TINFO(TI, np.nan, 'Checked chl-a fits', verbose)