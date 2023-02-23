# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the module dedicated to computations on profiles. Allow computaion of profile local time (corrected from
longitude), physical features like conservative temperature, Brunt-Vaissala frequency or potential density, radiative
features like polynomial fits of downwelling irradiance profiles, clouds detection...etc.
'''

# Imports


from sort import *


# Functions


def CMP_localtimes(pathorprof, verbose=True):
    '''
    Computes the profiles local times (longitude correction).

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. (str, int or xarray)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The profiles local times. (numpy ndarray)
    '''

    def FMT_times(times):

        if type(times) is list or type(times) is np.ndarray:

            return np.array([FMT_times(time) for time in times])

        else:

            if times < 0.:
                while times < 0.:
                    times += 24.
            else:
                while times >= 24.:
                    times -= 24.

            return times

    time = t.time()

    prof = RAW(pathorprof)
    juld, lon = prof.JULD.values, prof.LONGITUDE.values

    dates = FMT_date(juld, 'dt', verbose=False)
    LOCALTIME = [date.hour + date.minute/60. + date.second/3600. for date in dates]
    corr = FMT_lon(lon)/360.*24.
    LOCALTIME = np.round(FMT_times(LOCALTIME+corr), 4)

    TINFO(time, 1., 'Local times calculation', verbose)

    return LOCALTIME


def CMP_physicalfeatures(pathorprof, verbose=True):
    '''
    Computes conservative temperature, Brunt-Vaisala frequency, density and mixed-layer depth
    (with its quality index).

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. (str, int or xarray)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: CT, BVF, SIG0, MLD and MLD_QI. (tuple of numpy ndarray)
    '''

    war.filterwarnings('ignore', category=RuntimeWarning)

    time = t.time()

    prof = RAW(pathorprof)
    pres, sal, tem = prof.PRES.values, prof.PSAL.values, prof.TEMP.values
    smooth = 10

    # Conservative temperature

    CT = sw.conversions.CT_from_t(sal, tem, pres)

    # Brunt Vaissala frequency

    bvf_temp = sw.stability.Nsquared(sal, CT, pres, axis=1)
    BVF = np.nan * np.zeros(np.shape(pres))
    for k in range(np.shape(pres)[0]):
        bvftemp = xr.DataArray(bvf_temp[0][k, ~np.isnan(bvf_temp[1][k])],
                               coords={'pres': bvf_temp[1][k, ~np.isnan(bvf_temp[1][k])]}, dims=['pres'])
        BVF[k, ~np.isnan(pres[k])] = bvftemp.interp(pres=pres[k, ~np.isnan(pres[k])]).values

    BVFMAX_DEPTH = np.nan * np.zeros(np.shape(pres)[0])
    for i in range(np.shape(pres)[0]):
        if not np.isnan(np.where(pres[i]<7., np.nan, BVF[i])).all():
            BVFMAX_DEPTH[i] = pres[i][np.nanargmax(np.where(pres[i]<7., np.nan, BVF[i]))] * VAR.hf

    # Density

    SIG0 = sw.density.sigma0(sal, CT)

    # MLD

    dct_thresh = 0.02
    dsig0_thresh = 0.01
    MLD_T02 = np.zeros(np.shape(pres)[0]) * np.nan
    MLD_T02_QI, MLD_S125, MLD_S125_QI, MLD_S03, MLD_S03_QI, MLD_DT, MLD_DT_QI, MLD_DS, MLD_DS_QI = MLD_T02.copy(),\
    MLD_T02.copy(), MLD_T02.copy(), MLD_T02.copy(), MLD_T02.copy(), MLD_T02.copy(), MLD_T02.copy(), MLD_T02.copy(), MLD_T02.copy()

    # MLD with thresholds

    for i in range(np.size(MLD_T02)):

        pres_s, pres_t, sal_i, temp_i, ct_i, sig0_i = pres[i], pres[i], sal[i], tem[i], CT[i], SIG0[i]
        sig0_i, pres_s = sig0_i[~np.isnan(sig0_i)], pres_s[~np.isnan(sig0_i)]
        ct_i, pres_t = ct_i[~np.isnan(ct_i)], pres_t[~np.isnan(ct_i)]

        if np.size(pres_t) > 20 and np.min(pres_t) < 10. and np.max(pres_t) > 10.:

            ct_10m = np.interp(10, pres_t[~np.isnan(ct_i)], ct_i[~np.isnan(ct_i)])
            ct_thresh = ct_10m - 0.2

            if np.min(ct_i[pres_t>10.]) < ct_thresh:

                ct_i, pres_t = ct_i[pres_t>10.], pres_t[pres_t>10.]
                mld_t = np.min(pres_t[ct_i < ct_thresh]) * VAR.hf
                qc_t = 1 - np.std(ct_i[pres_t < mld_t])/np.std(ct_i[pres_t<1.5*mld_t])
                MLD_T02[i] = mld_t
                MLD_T02_QI[i] = qc_t

        if np.size(pres_s) > 20 and np.min(pres_s) < 10. and np.max(pres_s) > 10.:

            sig0_10m = np.interp(10, pres_s[~np.isnan(sig0_i)], sig0_i[~np.isnan(sig0_i)])
            sig0_thresh125 = sig0_10m + 0.125
            sig0_thresh03 = sig0_10m + 0.03
            sig0_i, pres_s = sig0_i[pres_s > 10.], pres_s[pres_s > 10.]

            if np.max(sig0_i) > sig0_thresh03:
                mld_s03 = np.min(pres_s[sig0_i > sig0_thresh03]) * VAR.hf
                qc_s03 = 1 - np.std(sig0_i[pres_s<mld_s03])/np.std(sig0_i[pres_s<1.5*mld_s03])
                MLD_S03[i] = mld_s03
                MLD_S03_QI[i] = qc_s03

                if np.max(sig0_i) > sig0_thresh125:
                    mld_s125 = np.min(pres_s[sig0_i > sig0_thresh125]) * VAR.hf
                    qc_s125 = 1 - np.std(sig0_i[pres_s<mld_s125])/np.std(sig0_i[pres_s<1.5*mld_s125])
                    MLD_S125[i] = mld_s125
                    MLD_S125_QI[i] = qc_s125


    # MLD from derivatives

    for i in range(np.size(MLD_DT)):

        pres_s, pres_t, sal_i, temp_i, ct_i, sig0_i = pres[i], pres[i], sal[i], tem[i], CT[i], SIG0[i]
        sig0_i, pres_s = sig0_i[~np.isnan(sig0_i)], pres_s[~np.isnan(sig0_i)]
        ct_i, pres_t = ct_i[~np.isnan(ct_i)], pres_t[~np.isnan(ct_i)]
        dsig0_i = (sig0_i[1:] - sig0_i[:-1])/(pres_s[1:] - pres_s[:-1])
        dct_i = (ct_i[1:] - ct_i[:-1])/(pres_t[1:] - pres_t[:-1])
        pres_ds = (pres_s[1:] + pres_s[:-1]) / 2
        pres_dt = (pres_t[1:] + pres_t[:-1]) / 2

        if np.size(pres_t) > 20 and np.min(pres_t) < 10. and np.max(pres_t) > 10. and \
                np.min(dct_i[pres_dt>10.]) < -dct_thresh:

            ct_i, pres_t = ct_i[pres_t>10.], pres_t[pres_t>10.]
            dct_i, pres_dt = dct_i[pres_dt>10.], pres_dt[pres_dt>10.]
            mld_dt = np.min(pres_dt[dct_i < - dct_thresh]) * VAR.hf
            qc_dt = 1 - np.std(ct_i[pres_t<mld_dt])/np.std(ct_i[pres_t<1.5*mld_dt])

            MLD_DT[i] = mld_dt
            MLD_DT_QI[i] = qc_dt

        if np.size(pres_s) > 20 and np.min(pres_s) < 10. and np.max(pres_s) > 10. and \
                np.max(dsig0_i[pres_ds>10.]) > dsig0_thresh:

            sig0_i, pres_s = sig0_i[pres_s>10.], pres_s[pres_s>10.]
            dsig0_i, pres_ds = dsig0_i[pres_ds>10.], pres_ds[pres_ds>10.]

            mld_ds = np.min(pres_ds[dsig0_i > dsig0_thresh]) * VAR.hf
            qc_ds = 1 - np.std(sig0_i[pres_s<mld_ds])/np.std(sig0_i[pres_s<1.5*mld_ds])

            MLD_DS[i] = mld_ds
            MLD_DS_QI[i] = qc_ds

    TINFO(time, 4., 'Physical variables calculation', verbose)

    return CT, BVF, BVFMAX_DEPTH, SIG0, MLD_S125, MLD_S125_QI, MLD_S03, MLD_S03_QI, MLD_T02, MLD_T02_QI, MLD_DT,\
        MLD_DT_QI, MLD_DS, MLD_DS_QI


def CMP_irrfeatures(pathorprof, verbose=True):
    '''
    Computes the PAR related features. Contains profiles flag, clouds, fit. The flags correspond respectively to 1 for
    good profiles, 2 for probably good, 3 for probably bad and 4 for bad profiles. The fits are B splines of degree =
    irrpoly_order in globvars.py with nodes located where clouds density are the lowest along profile.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. (str, int or xarray)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: FIT_PAR, CLD_PAR, FLG_PAR, ISO15, ZEU, PAR_ZEU
    '''
    
    def FIT_irr(wmo, var, verbose=True):

        def inclouds(c, x):
            d = 0
            for k in range(1, len(x)):
                if c[k]:
                    d += x[k] - x[k - 1]
            return d

        def biggestcloud(c, x):
            d = 0
            k = 1
            prof = np.nan
            while k < len(x):
                dcur = 0
                while c[k]:
                    dcur += x[k] - x[k - 1]
                    k += 1
                if dcur > d:
                    d = dcur
                    prof = x[k]
                k += 1
            return d, prof

        def percentclouds(c, x):

            X, C = x[~np.isnan(c)], c[~np.isnan(c)]
            out = np.nan * np.zeros(np.size(X))
            f = interpolate.interp1d(X, C, kind='nearest')

            for i in range(np.size(X)):
                xmin, xmax = np.nanmax([np.nanmin(X), X[i] - 25]), np.nanmin([np.nanmax(X), X[i] + 25])
                c0 = f(xmin)
                xtemp, ctemp = X[X < xmax], C[X < xmax]
                xtemp, ctemp = xtemp[xtemp > xmin], ctemp[xtemp > xmin]
                xtemp, ctemp = np.hstack([xmin, xtemp, xmax]), np.hstack([c0, ctemp])
                dc = 0
                for k in range(np.size(xtemp) - 1):
                    if ctemp[k]:
                        dc += xtemp[k + 1] - xtemp[k]
                out[i] = dc / (xmax - xmin)

            out = np.interp(x, X, out, left=np.nan, right=np.nan)

            return out

        ti = t.time()
        war.filterwarnings('ignore', category=RuntimeWarning)

        prof = RAW(wmo)
        logs = open(VAR.logspath + 'IRR/{}_{}_logs.txt'.format(var, int(prof.PLATFORM_NUMBER.values[0])), 'w')

        P = prof.PRES.values
        nprof = np.shape(P)[0]
        CLD, FLG, FIT = np.nan * np.zeros(np.shape(P), dtype=bool), 4 * np.ones(np.shape(P)[0], dtype=int), \
                        np.nan * np.zeros(np.shape(P), dtype=float)
        cthresh = 0.1
        dz = 30

        IRR = prof[var].values

        for i in range(np.shape(P)[0]):

            if verbose:
                CPRINT('{} fitting: {:.0f}% done.'.format(var, i / (np.shape(P)[0] - 1) * 100) + LOADOTS(i), end='\r',
                       attrs='CYAN')

            try:

                # Computing envelope yd, clouds location and mean absolute errors

                x, y = P[i], np.log(IRR[i])
                mask1 = ~np.isnan(y)
                x, y = x[mask1], y[mask1]
                mask2 = x < np.min([np.nanmax(x), 350.])
                x, y = x[mask2], y[mask2]

                if np.size(x) < 30:
                    raise Exception('Not enough points')
                if np.max(x) - np.min(x) < 2 * dz:
                    raise Exception('Domain too small')
                if var == 'DOWNWELLING_PAR':
                    if np.max(np.exp(y)) - np.min(np.exp(y)) <= 50:
                        raise Exception('Too small value span')
                else:
                    if np.max(np.exp(y)) - np.min(np.exp(y)) <= 0.02:
                        raise Exception('Too small value span')

                top = x < np.nanmin(x) + dz / 2
                bot = x > np.nanmax(x) - dz / 2
                yd = FLT_forcedecrease(y, verbose=False)
                yd = np.interp(x, x[~np.isnan(yd)], yd[~np.isnan(yd)])
                yd = FLT_mean(yd, x, smooth=dz, verbose=False)
                resd = yd - y
                clouds = resd > cthresh
                clouds_pc = percentclouds(clouds, x)

                # Locating nodes where there is as few clouds as possible

                slices = np.arange(dz, np.nanmax(x) - dz, dz)

                if np.sum(~clouds) == 0:

                    nodes = slices

                else:

                    nodes = []
                    for depth in slices:

                        cc_masked = np.where(x > depth, clouds_pc, np.nan)
                        cc_masked = np.where(x < depth + dz, cc_masked, np.nan)

                        if not np.isnan(cc_masked).all():

                            icandidate = np.nanargmin(cc_masked)

                            if clouds_pc[icandidate] <= 0.8:
                                nodes.append(x[icandidate])

                    nodes = np.array(nodes)
                    nodes = nodes[nodes < np.nanmax(x) - dz]

                top = np.nanmin(x) + dz / 2
                bot = np.max(nodes) + dz

                # Flagging vars

                bgstcloud, bgstprof = biggestcloud(clouds, x)

                # Fitting

                spl = LSQUnivariateSpline(x[~clouds], y[~clouds], k=4, t=nodes)

                FIT[i] = spl(P[i])
                FIT[i] = np.where(P[i] < top, np.nan, FIT[i])
                FIT[i] = np.where(P[i] > bot, np.nan, FIT[i])
                if bgstcloud > 2 * dz:
                    FIT[i] = np.where(P[i] > bgstprof - bgstcloud, np.nan, FIT[i])
                FIT[i] = FLT_forcedecrease(FIT[i], verbose=False)

                interp = interpolate.interp1d(x, clouds, kind='nearest', fill_value='extrapolate')
                CLD[i] = interp(P[i])
                CLD[i] = np.where(mask1, CLD[i], np.nan)
                CLD[i][mask1] = np.where(mask2, CLD[i][mask1], np.nan)

                # Extrapolating to the surface

                fstart, pstart = FIT[i][~np.isnan(FIT[i])][0], P[i][~np.isnan(FIT[i])][0]
                slope = stats.linregress(P[i][~np.isnan(FIT[i])][P[i][~np.isnan(FIT[i])] < slices[0]],
                                         FIT[i][~np.isnan(FIT[i])][P[i][~np.isnan(FIT[i])] < slices[0]]).slope
                FIT[i] = np.where(P[i]<pstart, fstart + slope * (P[i] - pstart), FIT[i])

                # Fit is exponential of what we derived

                FIT[i] = np.exp(FIT[i])

                # Flagging

                MAE = np.nanmean(np.abs(resd[~np.isnan(clouds)]))
                MAE_nc = np.nanmean(np.abs(resd[~np.isnan(clouds)][~clouds[~np.isnan(clouds)]]))
                MAE_cl = np.nanmean(np.abs(resd[~np.isnan(clouds)][clouds[~np.isnan(clouds)]]))
                pc_clouds = inclouds(clouds[~np.isnan(clouds)], x[~np.isnan(clouds)]) \
                                / (np.nanmax(x[~np.isnan(clouds)]) - np.nanmin(x[~np.isnan(clouds)])) * 100

                flg = 1 if pc_clouds <= 20 else 2 if pc_clouds <= 40 else 3 if pc_clouds <= 60 else 4
                exc = False
                if flg >= 3 and MAE_cl <= 4 * cthresh:
                    exc = True
                    flg = 2
                    logs.write('P{:03d}: Overflagged because of clouds size\n'.format(i))
                if flg < 3 and MAE_cl >= 10 * cthresh:
                    exc = True
                    flg = 3
                    logs.write('P{:03d}: Underflagged because of clouds size\n'.format(i))
                if flg < 3 and bgstcloud > dz and bgstprof < np.min([np.max(x), 250]) * 0.8:
                    exc = True
                    flg = 3
                    logs.write('P{:03d}: Underflagged because of a big cloud\n'.format(i))

                if not exc:
                    logs.write('P{:03d}: Normal profile\n'.format(i))

                FLG[i] = flg

            except ValueError as e:

                logs.write('P{:03d}: Failed to find nodes\n'.format(i))

            except Exception as e:

                logs.write('P{:03d}: '.format(i) + str(e) + '\n')

        logs.close()

        TINFO(ti, nprof/14, '{} fit'.format(var), verbose)

        return CLD, FLG, FIT

    def CMP_ZEU(PAR_FIT, P):

        ZEU = np.nan * np.zeros(np.shape(PAR_FIT)[0])
        PAR_ZEU = np.nan * np.zeros(np.shape(PAR_FIT)[0])

        for i in range(np.shape(PAR_FIT)[0]):

            par, pres = PAR_FIT[i], P[i]
            par, pres = par[~np.isnan(pres)], pres[~np.isnan(pres)]
            par, pres = par[~np.isnan(par)], pres[~np.isnan(par)]

            if not np.size(par) == 0:

                PARsurf = np.nanmax(par)

                if np.size(pres[par < PARsurf / 100]) > 0:

                    Zeu_pres = np.min(pres[par < PARsurf / 100])
                    Zeu_par = PARsurf/100
                    ZEU[i] = Zeu_pres * VAR.hf
                    PAR_ZEU[i] = Zeu_par

        return ZEU, PAR_ZEU

    def CMP_ISO15(FIT_PAR, P):

        def DICHOTOMY_ISOLUME(par1d):

            thresh = 15
            mask = ~np.isnan(par1d)
            indices = np.arange(np.size(par1d))
            par = par1d.copy()[mask]
            imin, imax, i = 0, np.size(par) - 1, -1

            if np.size(par) > 2:

                if np.nanmin(par) < thresh and np.nanmax(par) > thresh:

                    while imax - imin > 2:

                        icur = int((imin + imax) // 2)

                        if par[icur] > thresh:

                            imin = icur

                        else:

                            imax = icur

                    if imax - imin == 2:

                        i = int((imin + imax) // 2)

                    else:

                        i = imin

            if i == -1:
                res = i
            else:
                res = indices[mask][i]

            return res

        nprof, _ = np.shape(P)
        ISO15 = np.nan * np.zeros(nprof)

        for i in range(nprof):

            ind = DICHOTOMY_ISOLUME(FIT_PAR[i])

            if ind > -1:

                ISO15[i] = P[i, ind] * VAR.hf

        return ISO15

    time = t.time()

    CLD_PAR, FLG_PAR, FIT_PAR = FIT_irr(pathorprof, 'DOWNWELLING_PAR', verbose=verbose)
    P = RAW(pathorprof).PRES.values
    ISO15 = CMP_ISO15(FIT_PAR, P)
    ZEU, PAR_ZEU = CMP_ZEU(FIT_PAR, P)

    TINFO(time, 2*np.shape(P)[0]/9, 'Irradiance features calculation', verbose)

    return FIT_PAR, CLD_PAR, FLG_PAR, ISO15, ZEU, PAR_ZEU


def CMP_bbpfeatures(pathorprof, verbose=True):
    '''
    Computes profiles smoothed backscattering at 700nm. Interpolates BBP data every 5m.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. (str, int or xarray)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The smoothed BBP700 array. (numpy array)
    '''

    if verbose:
        CPRINT('Despiking and smoothing B_bp:', attrs='BLUE')

    prof = RAW(pathorprof)
    BBP, pres = prof.BBP700.values.copy(), prof.PRES.values

    z = np.arange(0., 1001., 5.)
    z = np.vstack([z for _ in range(np.shape(pres)[0])])

    BBP = INT_1D(BBP, pres, z)
    BBP_ds = FLT_despike(BBP, z, verbose=True)

    ti = t.time()

    BBP_ds_sm = FLT_mean(BBP_ds, z, verbose=False)
    BBP_ds_sm = INT_1D(BBP_ds, z, pres)

    TINFO(ti, 1e-2 * BBP.shape[0], 'Smoothed B_bp.', verbose)

    return BBP_ds_sm


def CMP_chlfeatures(pathorprof, verbose=True):
    '''
    Computes the chlorophyll corrected from remaining dark values, dark value and dark depth.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. (str, int or xarray)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: CHLA_ZERO, VDARK, PDARK. (numpy ndarrays)
    '''

    time = t.time()

    prof = RAW(pathorprof)

    if verbose:
        CPRINT('Chl A features: 0% done.', end='\r', attrs='CYAN')

    N = np.shape(prof.PRES.values)[0]
    PDARK = np.nan * np.zeros(N)
    VDARK = np.nanmedian(np.where(prof.PRES.values > 600., prof.CHLA.values, np.nan))

    for i in range(N):

        if verbose:
            CPRINT('Chl A features: {:.0f}% done.'.format(i/(N-1)*100)+LOADOTS(i), end='\r', attrs='CYAN')

        pres, chl = prof.PRES.values[i], prof.CHLA.values[i]
        pres, chl = pres[~np.isnan(chl)], chl[~np.isnan(chl)]

        if np.size(chl) > 20 and np.max(pres) > 600.:

            noisestd = np.nanstd(chl[pres > 600.])
            count = np.array([np.sum(chl[k:]<VDARK+2*noisestd)/(np.size(chl)-k) for k in range(np.size(chl))])
            thresh = 0.95
            dark_i = -1

            while pres[dark_i] > 300.:

                dark_i = np.argmax(count > thresh)
                while chl[dark_i] > VDARK and dark_i < np.size(pres)-1:
                    dark_i += 1
                PDARK[i] = pres[dark_i]

                thresh -= 0.05

    CZERO = np.nan * np.zeros(np.shape(prof.CHLA.values))

    for i in range(N):

        if not np.isnan(PDARK[i]):

            CZERO[i] = prof.CHLA.values[i] - VDARK
            CZERO[i] = np.where(CZERO[i]<0, 0., CZERO[i])
            CZERO[i] = np.where(prof.PRES.values[i] > PDARK[i], 0., CZERO[i])

    TINFO(time, 4., 'Chl A features calculation', verbose)

    return CZERO, VDARK, PDARK


def CMP_X11slopef(pathorprof, var, verbose=True):
    '''
    Computes the slope factor using Xing et al., 2011, and a quality index. The quality index legend is 1
    for good profiles, 2 for probably good, 3 for probably bad and 4 for bad profiles.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. (str, int or xarray)
    :param var: The variable to use for the method: has to be DOWN_IRRADIANCE380, 412 or 490. (str)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: SLOPEF, QI. (numpy ndarrays)
    '''

    time = t.time()

    if var[-3:] == '490':   # Morel et al., 2007b : Examining the consistency of products derived from various ocean
                            # color sensors in open ocean (Case 1) waters in the perspective of a multi-sensor approach
        Kw = 0.0166
        Chi = 0.08253
        e = 0.62588

    elif var[-3:] == '412': # Morel et Maritorena, 2001.

        Kw = 0.00812
        Chi = 0.12259
        e = 0.65175

    elif var[-3:] == '380':

        Kw = 0.0151
        Chi = 0.127
        e = 0.685

    else:

        raise ValueError('The variable has to be irradiance at 380, 412 or 490nm.')

    prof = RAW(pathorprof)
    pres, irr = prof.PRES.values, prof[var].values

    if verbose:
        CPRINT('Slope factor with X11 method for {}: 0% done.'.format(var), end='\r', attrs='CYAN')

    chl = CMP_chlfeatures(pathorprof, verbose=False)[0]

    S, Q = np.nan * np.zeros(np.shape(pres)[0]), 4 * np.ones(np.shape(pres)[0])

    for i in range(np.shape(pres)[0]):

        if verbose:
            CPRINT('Slope factor with X11 method for {}: {:.0f}% done.' .format(var, i/(np.shape(pres)[0]-1)*100)
                  +LOADOTS(i), end='\r', attrs='CYAN')

        try:

            P, I = pres[i], irr[i]
            P, I = P[~np.isnan(I)], I[~np.isnan(I)]
            E0 = np.nanmax(I)
            P, I = P[I < E0 / 2], I[I < E0 / 2]
            P, I = P[I > E0 / 100], I[I > E0 / 100]
            C = np.interp(P, pres[i][~np.isnan(chl[i])], chl[i][~np.isnan(chl[i])])
            indexes = np.arange(np.size(I))

            An = np.log(I) + P * Kw
            An, indexes = An[1:], indexes[1:]
            Cn = Chi * np.array([np.sum([(C[k]**e + C[k+1]**e) / 2 * (P[k+1] - P[k]) for k in range(l)])
                                 for l in range(1, np.size(P))])
            N = np.size(Cn)

            reginit = stats.linregress(Cn, An)
            s, o, r = reginit.slope, reginit.intercept, reginit.rvalue

            An_init, Cn_init = An, Cn

            threshinit = (s*Cn + o) - 0.3
            mask = An > threshinit
            Cn, An, indexes = Cn[mask], An[mask], indexes[mask]
            iter = 0

            while r ** 2 < 0.98 and iter < 5 and np.size(Cn) > 5:

                iter += 1
                old_r = r
                reg = stats.linregress(Cn, An)
                s, o, r = reg.slope, reg.intercept, reg.rvalue
                if r == old_r:
                    break
                thresh = (s * Cn + o) - 0.02 * np.abs((s * Cn + o))
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

            S[i], Q[i] = slope, QI

        except ValueError:

            pass

    TINFO(time, 8., 'Slope factor calculation (X11 method, {})'.format(var), verbose)

    return S, Q


def CMP_newvars(pathorprof, verbose=True):
    '''
    Returns a dictionary containing new variables computed over an ARGO S_prof having light sensors.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. (str, int or xarray)
    :return: The dictionary with new variables. (xarrays dict)
    '''

    time = t.time()

    if verbose:
        CPRINT('\nComputing new variables:\n', attrs=['BLUE', 'ITALIC'])

    LTIME = CMP_localtimes(pathorprof, verbose=verbose)
    CT, BVF, BVFMAX_DEPTH, SIG0, MLD_S125, MLD_S125_QI, MLD_S03, MLD_S03_QI, MLD_T02, MLD_T02_QI, MLD_DT, \
    MLD_DT_QI, MLD_DS, MLD_DS_QI = CMP_physicalfeatures(pathorprof, verbose=verbose)
    FIT_PAR, CLD_PAR, FLG_PAR, ISO15, ZEU, PAR_ZEU = CMP_irrfeatures(pathorprof, verbose=verbose)
    BBP700_DS = CMP_bbpfeatures(pathorprof, verbose=verbose)
    CHLA_ZERO, VDARK, PDARK = CMP_chlfeatures(pathorprof, verbose=verbose)
    SLOPEF490, SLOPEF490_QI = CMP_X11slopef(pathorprof, var='DOWN_IRRADIANCE490', verbose=verbose)
    SLOPEF412, SLOPEF412_QI = CMP_X11slopef(pathorprof, var='DOWN_IRRADIANCE412', verbose=verbose)
    SLOPEF380, SLOPEF380_QI = CMP_X11slopef(pathorprof, var='DOWN_IRRADIANCE380', verbose=verbose)

    LTIME = xr.DataArray(LTIME, dims=['N_PROF'], name='LTIME')
    CT = xr.DataArray(CT, dims=['N_PROF', 'N_LEVELS'], name='CT')
    SIG0 = xr.DataArray(SIG0, dims=['N_PROF', 'N_LEVELS'], name='SIG0')
    BVF = xr.DataArray(BVF, dims=['N_PROF', 'N_LEVELS'], name='BVF')
    BVFMAX_DEPTH = xr.DataArray(BVFMAX_DEPTH, dims=['N_PROF'], name='BVFMAX_DEPTH')
    MLD_T02 = xr.DataArray(MLD_T02, dims=['N_PROF'], name='MLD_T02')
    MLD_T02_QI = xr.DataArray(MLD_T02_QI, dims=['N_PROF'], name='MLD_T02_QI')
    MLD_S125 = xr.DataArray(MLD_S125, dims=['N_PROF'], name='MLD_S125')
    MLD_S125_QI = xr.DataArray(MLD_S125_QI, dims=['N_PROF'], name='MLD_S125_QI')
    MLD_S03 = xr.DataArray(MLD_S03, dims=['N_PROF'], name='MLD_S03')
    MLD_S03_QI = xr.DataArray(MLD_S03_QI, dims=['N_PROF'], name='MLD_S03_QI')
    MLD_DT = xr.DataArray(MLD_DT, dims=['N_PROF'], name='MLD_DT')
    MLD_DT_QI = xr.DataArray(MLD_DT_QI, dims=['N_PROF'], name='MLD_DT_QI')
    MLD_DS = xr.DataArray(MLD_DS, dims=['N_PROF'], name='MLD')
    MLD_DS_QI = xr.DataArray(MLD_DS_QI, dims=['N_PROF'], name='MLD_QI')
    FIT_PAR = xr.DataArray(FIT_PAR, dims=['N_PROF', 'N_LEVELS'], name='FIT_PAR')
    CLD_PAR = xr.DataArray(CLD_PAR, dims=['N_PROF', 'N_LEVELS'], name='CLD_PAR')
    FLG_PAR = xr.DataArray(FLG_PAR, dims=['N_PROF'], name='FLG_PAR')
    ISO15 = xr.DataArray(ISO15, dims=['N_PROF'], name='ISO15')
    ZEU = xr.DataArray(ZEU, dims=['N_PROF'], name='ZEU')
    PAR_ZEU = xr.DataArray(PAR_ZEU, dims=['N_PROF'], name='PAR_ZEU')
    BBP700_DS = xr.DataArray(BBP700_DS, dims=['N_PROF', 'N_LEVELS'], name='BBP700_DS')
    CHLA_ZERO = xr.DataArray(CHLA_ZERO, dims=['N_PROF', 'N_LEVELS'], name='CHLA_ZERO')
    VDARK = xr.DataArray(np.array(VDARK), dims=[], name='CDARK_VALUE')
    PDARK = xr.DataArray(PDARK, dims=['N_PROF'], name='CDARK_DEPTH')
    SLOPEF490 = xr.DataArray(SLOPEF490, dims=['N_PROF'], name='SLOPEF490')
    SLOPEF490_QI = xr.DataArray(SLOPEF490_QI, dims=['N_PROF'], name='SLOPEF490_QI')
    SLOPEF412 = xr.DataArray(SLOPEF412, dims=['N_PROF'], name='SLOPEF412')
    SLOPEF412_QI = xr.DataArray(SLOPEF412_QI, dims=['N_PROF'], name='SLOPEF412_QI')
    SLOPEF380 = xr.DataArray(SLOPEF380, dims=['N_PROF'], name='SLOPEF380')
    SLOPEF380_QI = xr.DataArray(SLOPEF380_QI, dims=['N_PROF'], name='SLOPEF380_QI')

    newvars = {'LOCALTIME': LTIME,
               'CT': CT,
               'SIG0': SIG0,
               'BVF': BVF,
               'BVFMAX_DEPTH': BVFMAX_DEPTH,
               'MLD_T02': MLD_T02,
               'MLD_T02_QI': MLD_T02_QI,
               'MLD_S125': MLD_S125,
               'MLD_S125_QI': MLD_S125_QI,
               'MLD_S03': MLD_S03,
               'MLD_S03_QI': MLD_S03_QI,
               'MLD_DT': MLD_DT,
               'MLD_DT_QI': MLD_DT_QI,
               'MLD_DS': MLD_DS,
               'MLD_DS_QI': MLD_DS_QI,
               'DOWNWELLING_PAR_FIT': FIT_PAR,
               'DOWNWELLING_PAR_CLD': CLD_PAR,
               'DOWNWELLING_PAR_FLG': FLG_PAR,
               'ISO15': ISO15,
               'ZEU': ZEU,
               'PAR_ZEU': PAR_ZEU,
               'BBP700_DS': BBP700_DS,
               'CHLA_ZERO': CHLA_ZERO,
               'CDARK_VALUE': VDARK,
               'CDARK_DEPTH': PDARK,
               'SLOPEF490': SLOPEF490,
               'SLOPEF490_QI': SLOPEF490_QI,
               'SLOPEF412': SLOPEF412,
               'SLOPEF412_QI': SLOPEF412_QI,
               'SLOPEF380': SLOPEF380,
               'SLOPEF380_QI': SLOPEF380_QI
               }

    TINFO(time, np.size(RAW(pathorprof).N_PROF.values) * 0.9, 'Derivation of new variables (total)', verbose)

    return newvars

