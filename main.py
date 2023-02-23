# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''Write your code here.'''

from plotargo import *
from classification import *
from maps import *
from fun import *

if __name__ == '__main__':

    time = t.time()
    print('Starting main...', end='\r')

    ##################### WRITE YOUR CODE #####################

    # prof = PRC(6636)
    #
    # fig, ax = plt.subplots()
    # ax.set_ylim(300., 0.)
    # axb = ax.twinx()
    # n = 30
    # nvalues = np.sum(~np.isnan(np.where(prof.PRES.values < 200., prof.DOWNWELLING_PAR.values, np.nan).astype(float)),
    #                  axis=1)
    # indexes = np.random.choice(np.arange((np.size(prof.N_PROF.values)))[nvalues > 10], n).astype(int)
    # indexes = np.sort(indexes)
    #
    # # Params
    # maxgap = 30.
    # r2min = 0.9
    # nptsmin = 30
    # dmaxfit = 150.
    #
    # # V0
    #
    # for i in indexes:
    #
    #     nimp, cloudy = False, False
    #     par, pres = prof.DOWNWELLING_PAR.values[i], prof.PRES.values[i]
    #     par, pres = par[~np.isnan(par)], pres[~np.isnan(par)]
    #     lpar = np.log10(par)
    #     ax.scatter(lpar, pres, c='gray', s=10, label='Raw profile')
    #
    #     par_up, pres_up = par[pres < dmaxfit], pres[pres < dmaxfit] # A changer en dark value
    #     lpar_up = np.log10(par_up)
    #     dlpar = D1(lpar, pres)
    #     dlpar_sm = FLT_mean(lpar, pres)
    #     gaps = pres_up[1:] - pres_up[:-1]
    #
    #     if np.size(lpar_up) > nptsmin and np.max(gaps) < maxgap:
    #
    #         lpar_up_sm = FLT_mean(lpar_up, pres_up, smooth=50.)
    #         res = lpar_up - lpar_up_sm
    #         ax.plot(lpar_up_sm, pres_up)
    #         axb.plot(dlpar_sm, pres)
    #         ax.scatter(lpar_up[res < -np.nanstd(res)], pres_up[res < -np.nanstd(res)], c='r')


            # reg = stats.linregress(pres_up, lpar_up)
            #
            # if reg.rvalue**2 > r2min:
            #
            #     ax.set_title('$r_1^2 = {:.2f}$: clean'.format(reg.rvalue ** 2))
            #     ax.plot(reg.slope * pres_up + reg.intercept, pres_up, c='g', linestyle='dashed', zorder=3,
            #             label='First fit')
            #     res = lpar_up - (reg.slope * pres_up + reg.intercept)
            #
            #     spl = UnivariateSpline(pres_up, lpar_up, w=1/np.abs(res), k=4, s=100., ext='zeros')
            #     newprof = spl(pres)
            #     newprof = np.where(newprof==0., np.nan, newprof)
            #
            #     ax.plot(newprof, pres, c='purple', linewidth=2., label='New profile')
            #
            # else:
            #
            #     res = lpar_up - (reg.slope * pres_up + reg.intercept)
            #     mask = res > 0. # or fixed value
            #     gaps = pres_up[mask][1:] - pres_up[mask][:-1]
            #     newreg = stats.linregress(pres_up[mask], lpar_up[mask])
            #
            #     if np.max(pres_up[mask]) - np.min(pres_up[mask]) < 50. or np.max(gaps) > 10. or\
            #             np.sum(mask) < 20 or newreg.rvalue**2 < r2min:
            #         nimp = True
            #     else:
            #         cloudy = True
            #         ax.scatter(lpar_up[mask], pres_up[mask], c='purple')
            #
            #     ax.set_title('$r_1^2 = {:.2f}$ vs. $r_2^2 = {:.2f}$ : {}'.format(reg.rvalue**2, newreg.rvalue**2,
            #                                                                      'Throw away' if nimp
            #                                                                      else 'cloudy' if cloudy
            #                                                                      else 'clean'))
            #     ax.plot(reg.slope * pres_up + reg.intercept, pres_up, c='k', linestyle='dashed', zorder=3,
            #             label='First fit')
            #     ax.plot(newreg.slope * pres_up + newreg.intercept, pres_up, c='crimson', linestyle='dashed', zorder=3,
            #             label='New fit')
            #
            # leg = ax.legend()
            # plt.pause(4.)

            # leg.remove()
            # for artist in ax.collections + ax.lines + ax.texts + axb.collections + axb.lines + axb.texts:
            #     artist.remove()

    # V1

    # for i in indexes:
    #
    #     nimp, cloudy = False, False
    #     par, pres = prof.DOWNWELLING_PAR.values[i], prof.PRES.values[i]
    #     par, pres = par[~np.isnan(par)], pres[~np.isnan(par)]
    #     lpar = np.log10(par)
    #     ax.scatter(lpar, pres, c='gray', s=10, label='Raw profile')
    #
    #     par_up, pres_up = par[pres < dmaxfit], pres[pres < dmaxfit] # A changer en dark value
    #     lpar_up = np.log10(par_up)
    #     gaps = pres_up[1:] - pres_up[:-1]
    #
    #     if np.size(lpar_up) > nptsmin and np.max(gaps) < maxgap:
    #
    #         reg = stats.linregress(pres_up, lpar_up)
    #
    #         if reg.rvalue**2 > r2min:
    #
    #             ax.set_title('$r_1^2 = {:.2f}$: clean'.format(reg.rvalue ** 2))
    #             ax.plot(reg.slope * pres_up + reg.intercept, pres_up, c='g', linestyle='dashed', zorder=3,
    #                     label='First fit')
    #             res = lpar_up - (reg.slope * pres_up + reg.intercept)
    #
    #             spl = UnivariateSpline(pres_up, lpar_up, w=1/np.abs(res), k=4, s=100., ext='zeros')
    #             newprof = spl(pres)
    #             newprof = np.where(newprof==0., np.nan, newprof)
    #
    #             ax.plot(newprof, pres, c='purple', linewidth=2., label='New profile')
    #
    #         else:
    #
    #             res = lpar_up - (reg.slope * pres_up + reg.intercept)
    #             mask = res > 0. # or fixed value
    #             gaps = pres_up[mask][1:] - pres_up[mask][:-1]
    #             newreg = stats.linregress(pres_up[mask], lpar_up[mask])
    #
    #             if np.max(pres_up[mask]) - np.min(pres_up[mask]) < 50. or np.max(gaps) > 10. or\
    #                     np.sum(mask) < 20 or newreg.rvalue**2 < r2min:
    #                 nimp = True
    #             else:
    #                 cloudy = True
    #                 ax.scatter(lpar_up[mask], pres_up[mask], c='purple')
    #
    #             ax.set_title('$r_1^2 = {:.2f}$ vs. $r_2^2 = {:.2f}$ : {}'.format(reg.rvalue**2, newreg.rvalue**2,
    #                                                                              'Throw away' if nimp
    #                                                                              else 'cloudy' if cloudy
    #                                                                              else 'clean'))
    #             ax.plot(reg.slope * pres_up + reg.intercept, pres_up, c='k', linestyle='dashed', zorder=3,
    #                     label='First fit')
    #             ax.plot(newreg.slope * pres_up + newreg.intercept, pres_up, c='crimson', linestyle='dashed', zorder=3,
    #                     label='New fit')
    #
    #         leg = ax.legend()
    #         plt.pause(4.)
    #
    #         leg.remove()
    #         for artist in ax.collections + ax.lines + ax.texts:
    #             artist.remove()

    # V2

    # r2min = 0.98
    # nptsmin = 30
    # dmaxfit = 200.
    # maxn = 5
    # dz = 20.
    #
    # for i in indexes:
    #
    #     nimp, cloudy = False, False
    #     par, pres = prof.DOWNWELLING_PAR.values[i], prof.PRES.values[i]
    #     par, pres = par[~np.isnan(par)], pres[~np.isnan(par)]
    #     par, pres = par[pres > 10.], pres[pres > 10.]
    #     lpar = np.log10(par)
    #     ax.scatter(lpar, pres, c='gray', s=10, label='Raw profile')
    #
    #     def F(x, a, b, xm, s):
    #
    #         return a*x + b #+ (x - xm)**2/s
    #
    #     fit = opt.curve_fit(F, xdata=pres, ydata=lpar, p0=(-3./200., 1.5, 70., 300.),
    #                         bounds=([-4./100., 0., 0., 1.], [-1./500., 4., 200., 10000.]))
    #     a, b, xm, s = fit[0]
    #
    #     ax.plot(F(pres, a, b, xm, s), pres)

        # if np.size(lpar) > nptsmin:
        #
        #     k = 0
        #     mask = np.ones(np.size(lpar)).astype(bool)
        #     knots = np.arange(np.min(pres[mask]) + 1, np.max(pres[mask]) - 1., dz)
        #     spl = LSQUnivariateSpline(pres[mask], lpar[mask], k=4, t=knots, ext='zeros')
        #     newprof = spl(pres)
        #     newprof = np.where(newprof == 0., np.nan, newprof)
        #
        #     ax.plot(newprof, pres, c='crimson', linewidth=2., label='First fit')
        #
        #     res = lpar - newprof
        #     r2 = 1 - np.sum(res ** 2) / np.sum((newprof - np.mean(newprof))**2)
        #
        #     while r2<r2min and k<maxn:
        #
        #         k += 1
        #         mask = res > np.percentile(res, 20.)
        #         spl = UnivariateSpline(pres[mask], lpar[mask], k=4, s=smooth / 4, ext='zeros')
        #         newprof = spl(pres)
        #         newprof = np.where(newprof == 0., np.nan, newprof)
        #         res = lpar - newprof
        #         r2 = 1 - np.nansum(res ** 2) / np.nansum((newprof - np.nanmean(newprof)) ** 2)
        #
        #     ax.plot(newprof, pres, c='purple', linewidth=2., label='Second fit')
        #     ax.scatter(lpar[mask], pres[mask], c='orange', s=10, label='Kept points')
        #     ax.set_title('$r^2 = {:.3f}$ / $N_{{iter}}={}$ / {}'.format(r2, k, 'clean' if r2> 0.98 else 'throw away'))
        #     leg = ax.legend()
        # plt.pause(4.)

        # leg.remove()
        # for artist in ax.collections + ax.lines + ax.texts:
        #     artist.remove()

    # fig, ax = plt.subplots()
    # ax.set_ylim(-0.05, 1.05)
    # axb = plt.gca().twinx()
    # thresh = 0.7
    #
    # for wmo in WMOS():
    #
    #     ax.set_title(VAR.floats_names[wmo])
    #     prof = RAW(wmo)
    #
    #     for _ in range(5):
    #
    #         i = np.random.randint(np.size(prof.N_PROF.values))
    #
    #         par, pres = prof.DOWNWELLING_PAR.values[i], prof.PRES.values[i]
    #         lpar, pres = np.log10(par), pres
    #         lpar, pres = lpar[pres < 500.], pres[pres < 500.]
    #
    #         if np.size(lpar[~np.isnan(lpar)]) > 10.: # else: throw away profile
    #
    #             axb.set_ylim(np.nanmin(lpar) - np.nanstd(lpar)/4, np.nanmax(lpar) + np.nanstd(lpar)/4)
    #             axb.plot(pres, lpar, c='k', alpha=0.3, linewidth=0.5)
    #             lpar = FLT_mean(lpar, pres, smooth=50., verbose=False)
    #             lpar, pres = lpar[~np.isnan(lpar)], pres[~np.isnan(lpar)]
    #             axb.plot(pres, lpar, c='k')
    #             reg = stats.linregress(pres, lpar)
    #
    #             if reg.slope < 0.: # else throw away profile
    #
    #                 rval = np.empty(shape=(np.size(pres),), dtype=float)
    #
    #                 for k in range(np.size(pres)):
    #
    #                     p = pres[k]
    #                     prestemp, lpartemp = pres[pres > p - 10.], lpar[pres > p - 10.]
    #                     prestemp, lpartemp = prestemp[prestemp < p + 5.], lpartemp[prestemp < p + 5.]
    #                     reg = stats.linregress(prestemp, lpartemp)
    #                     rval[k] = reg.rvalue ** 2
    #
    #                 rval = np.array(rval)
    #                 ax.plot(pres, rval, c='orange', linewidth=0.5)
    #
    #                 dark = np.min(pres[pres > 100.][rval[pres > 100.] < thresh]) \
    #                     if np.sum(rval[pres > 100.] < thresh) > 1 else np.nan # else : dark is deeper
    #
    #                 ax.plot([dark, dark], ax.get_ylim(), c='k', linestyle='dashed')
    #                 ax.plot([0.8*dark, 0.8*dark], ax.get_ylim(), c='purple', linestyle='dashed')
    #
    #             plt.pause(1.5)
    #             for artist in ax.collections + ax.lines + ax.texts + axb.collections + axb.lines + axb.texts:
    #                 artist.remove()


    # def CMP_dark(wmo):
    #
    #     ti = t.time()
    #     prof = RAW(wmo)
    #     N = np.size(prof.N_PROF.values)
    #     DARK = np.ones(N) * np.nan
    #
    #     thresh = 0.7
    #     dz = .5
    #
    #     for i in range(N):
    #
    #         print(i, end='\r')
    #
    #         par, pres = prof.DOWNWELLING_PAR.values[i], prof.PRES.values[i]
    #         lpar, pres = np.log10(par), pres
    #         lpar, pres = lpar[pres < 500.], pres[pres < 500.]
    #
    #         if np.size(lpar[~np.isnan(lpar)]) > 10.:
    #
    #             lpar = FLT_mean(lpar, pres, smooth=50., verbose=False)
    #             lpar = INT_1D(lpar, pres, np.arange(0., np.max(pres), dz)).squeeze()
    #             pres = np.arange(0., np.max(pres), dz)
    #             lpar, pres = lpar[~np.isnan(lpar)], pres[~np.isnan(lpar)]
    #             reg = stats.linregress(pres, lpar)
    #
    #             if reg.slope < 0.:
    #
    #                 rval = np.empty(shape=(np.size(pres),), dtype=float)
    #
    #                 for k in range(np.size(pres)):
    #
    #                     p = pres[k]
    #                     prestemp, lpartemp = pres[pres > p - 10.], lpar[pres > p - 10.]
    #                     prestemp, lpartemp = prestemp[prestemp < p + 5.], lpartemp[prestemp < p + 5.]
    #                     reg = stats.linregress(prestemp, lpartemp)
    #                     rval[k] = reg.rvalue ** 2
    #
    #                 rval = np.array(rval)
    #
    #                 DARK[i] = np.min(pres[pres > 100.][rval[pres > 100.] < thresh]) \
    #                     if np.sum(rval[pres > 100.] < thresh) > 1 else -1.
    #
    #     TINFO(ti, 60., 'Computed dark values', True)
    #
    #     return DARK


    #################### END OF YOUR CODE #####################

    print('\nMain execution: {}'.format(FMT_secs(t.time() - time)))

    INVOKE_BLINKY()