# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the module dedicated to the computation of enhanced Sprof files.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''


# Imports


from processing import *


# Functions


def ADD_newvars(pathorprof, verbose=True):
    '''
    Adds new computable variables to the profile.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The profile(s) wmo. (int, path as str, xarray)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The profile(s) with new variables. (xarray or list of xarrays)
    '''

    time = t.time()

    prof = RAW(pathorprof)

    newvars = CMP_newvars(pathorprof, verbose=verbose)
    newprof = prof.copy().assign(newvars)
    newprof.attrs['history'] = 'Added new variables: {}'.format(dt.datetime.today().strftime('%Y-%m-%d'))

    return newprof


def ADD_prc(pathorprof, save=False, verbose=True):
    '''
    Processes new derivable variables using functions in module \'processing\' and adds it to your Sprof.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param save: Whether to save or not the new file to VAR.prcpath. (boolean, default is False)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The profile with processed variables. (xarray)
    '''

    def OPN(pathorprof, filter=VAR.basefilter, QCs=VAR.reject_QCs, datamode=VAR.datamode, verbose=True):
        '''
        Opens the profiles as WMOs, xarrays or filename. Filters with custom given filter (see filters.FLT function), QCs,
        datamode and adds newvars (can be as long as 5 minutes). If you will use the profile often, I recommand using
        ADD_prc with the parameter save=True and open it from your disk.

        ____

        Written by: T. Hermilly

        Contact: thomas.hermilly@ird.fr.

        Last update: 2023-02-09
        :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
         xarray or list)
        :param filter: The parameters you want to filter with and their values as: {'FILTER_PARAM': value}. Can be followed
         by '_MIN' or '_MAX', or be a list of numeric values if several values suit your needs. (optional, dictionary)
        :param QCs: The QC codes to reject in the ADJUSTED data. (int or list of int, default is reject_QCs in globvars.py)
        :param datamode: The parameter data modes to reject. (str or list of str in ['R', 'A', 'D'], default is datamode in
         globars.py)
        :param verbose: Whether to display information to the console. (boolean, default is True)
        :return: The extracted profiles. (xarray)
        '''

        if type(pathorprof) is list:

            if verbose:
                CPRINT('Opening profiles...', attrs='CYAN', end='\r')

            res = [OPN(p, filter=filter, QCs=QCs, datamode=datamode, verbose=False)
                   for p in pathorprof]

            TINFO(time, 0.8 * len(pathorprof) * np.sum([np.size(RAW(p).N_PROF.values) for p in pathorprof]),
                  'Opened profiles', verbose)

            return res

        else:

            ROOT()

            prof = FLT(ADD_newvars(FLT_datamode(FLT_qcs(FLT_greylist(RAW(pathorprof), QCs=QCs, reject=True,
                                                                     verbose=verbose), QCs=QCs, reject=True,
                                                        verbose=verbose), dm=datamode, verbose=verbose),
                                   verbose=verbose),
                       filter=filter, verbose=False)

            return prof


    def pCHLA_PRC(pres, iso15, mld, chl0, chl0_s, bbp_ds, bbp_ds_s, par_f, slope):

        CHLA_PRC = np.nan * np.zeros(np.shape(pres))

        for k in range(np.shape(CHLA_PRC)[0]):

            if not np.isnan(mld[k]) and not np.isnan(iso15[k]) and not np.isnan(chl0[k]).all() and \
                    not np.isnan(chl0_s[k]).all() and not np.isnan(bbp_ds_s[k]).all():

                # NPQ

                if mld[k] <= iso15[k]:

                    par_f[k][~np.isnan(chl0_s[k])] = np.interp(pres[k][~np.isnan(chl0_s[k])],
                                                               pres[k][~np.isnan(par_f[k])],
                                                               par_f[k][~np.isnan(par_f[k])])
                    chl_sgm_k = SGM(chl0_s[k][~np.isnan(chl0_s[k])], par_f[k][~np.isnan(chl0_s[k])])
                    chl_sgm_k = np.interp(pres[k], pres[k][~np.isnan(chl0_s[k])], chl_sgm_k)

                    CHLA_PRC[k] = np.where(pres[k] * VAR.hf < iso15[k], chl_sgm_k, chl0[k])

                    ratio = np.interp(mld[k], pres[k][~np.isnan(CHLA_PRC[k])] * VAR.hf,
                                      CHLA_PRC[k][~np.isnan(CHLA_PRC[k])]) / \
                            np.interp(mld[k], pres[k][~np.isnan(bbp_ds_s[k])] * VAR.hf,
                                      bbp_ds_s[k][~np.isnan(bbp_ds_s[k])])

                    CHLA_PRC[k] = np.where(pres[k] * VAR.hf <= mld[k], bbp_ds_s[k] * ratio, CHLA_PRC[k])

                else:

                    bbp_temp = np.interp(pres[k], pres[k][~np.isnan(bbp_ds[k])], bbp_ds[k][~np.isnan(bbp_ds[k])])
                    bbp_temp = np.where(pres[k] * VAR.hf < iso15[k], np.nan, bbp_temp)
                    bbp_temp = np.where(pres[k] * VAR.hf > mld[k], np.nan, bbp_temp)

                    chl0_temp = np.interp(pres[k], pres[k][~np.isnan(chl0[k])], chl0[k][~np.isnan(chl0[k])])
                    chl0_temp = np.where(pres[k] * VAR.hf < iso15[k], np.nan, chl0_temp)
                    chl0_temp = np.where(pres[k] * VAR.hf > mld[k], np.nan, chl0_temp)

                    ratio = np.nanmean(chl0_temp / bbp_temp)

                    if np.isnan(ratio):
                        ratio = np.interp(iso15[k], pres[k][~np.isnan(chl0[k])], chl0[k][~np.isnan(chl0[k])]) / \
                                np.interp(iso15[k], pres[k][~np.isnan(bbp_ds_s[k])],
                                          bbp_ds_s[k][~np.isnan(bbp_ds_s[k])])

                    CHLA_PRC[k] = np.where(pres[k] * VAR.hf < iso15[k], bbp_ds_s[k] * ratio, chl0[k])

        CHLA_PRC = slope * CHLA_PRC
        CHLA_PRC = INT_1D(CHLA_PRC, pres, pres)

        return CHLA_PRC


    def pSCM(chl_prc_s, pres, thresh=1.5):

        SCM = np.nan * np.zeros(np.shape(pres)[0])

        for k in range(np.shape(SCM)[0]):

            if not np.isnan(chl_prc_s[k]).all():
                if pres[k, np.nanargmax(chl_prc_s[k])] * VAR.hf > 10 and \
                    np.nanmax(chl_prc_s[k]) >= thresh * np.nanmean(chl_prc_s[k][pres[k] < 10.]):

                    SCM[k] = pres[k, np.nanargmax(chl_prc_s[k])] * VAR.hf

        return SCM


    def pSCM_features(chl_prc_s, pres, scm, cdark_depth):

        def f(x, A, mu, sig):
            return A * np.exp(-1 / 2 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

        scm_val, scm_depth, scm_width = np.nan * np.zeros(np.shape(pres)[0]), np.nan * np.zeros(np.shape(pres)[0]), \
                                        np.nan * np.zeros(np.shape(pres)[0])

        for k in range(np.shape(pres)[0]):

            if not np.isnan(scm[k]):

                chl, cdark_d, p = chl_prc_s[k], cdark_depth[k], pres[k]
                chl, p = chl[~np.isnan(chl)], p[~np.isnan(chl)]
                chl, p = chl[p < 400.], p[p < 400.]

                if np.size(chl) > 10:

                    chl_tilt = chl - chl[0] * (1 - p / (cdark_d / VAR.hf))
                    chl_tilt = np.where(p * VAR.hf > cdark_d, chl, chl_tilt)

                    res = opt.curve_fit(f, p, chl_tilt, p0=(10., 100., 20.), bounds=([1., 10., 3.],
                                                                                     [200., 300., 100.]))[0]

                    scm_val[k], scm_depth[k], scm_width[k] = f(res[1], res[0], res[1], res[2]) \
                                                             + np.interp(res[1], p, chl[0] *
                                                             (1 - p / (cdark_d / VAR.hf))), res[1], res[2]

        return scm_depth, scm_val, scm_width


    def pPAR_SCM(par_f, pres, scm):

        PAR_SCM = np.nan * np.zeros(np.shape(pres)[0])

        for k in range(np.shape(PAR_SCM)[0]):
            if not np.isnan(scm[k]) and np.sum(~np.isnan(par_f)) > 10:
                PAR_SCM[k] = np.interp(scm[k] / VAR.hf, pres[k][~np.isnan(par_f[k])], par_f[k][~np.isnan(par_f[k])])

        return PAR_SCM


    def pICHL(chl_prc_s, pres):

        iCHLA = np.nan * np.zeros(np.shape(pres))
        iCHLA_MAX = np.nan * np.zeros(np.shape(pres)[0])

        for k in range(np.shape(pres)[0]):

            if np.sum(~np.isnan(chl_prc_s[k])) > 10.:

                chl_i = np.interp(pres[k], pres[k][~np.isnan(chl_prc_s[k])], chl_prc_s[k][~np.isnan(chl_prc_s[k])])
                iCHLA[k, 0] = 0.

                l = 1

                while not (np.where(np.isnan(chl_i[l:]), 0., chl_i[l:]) == 0).all():
                    iCHLA[k, l] = iCHLA[k, l - 1] + (pres[k, l] - pres[k, l - 1]) * np.nanmean([chl_i[l - 1], chl_i[l]])
                    l += 1

                iCHLA_MAX[k] = iCHLA[k, l - 1]
                iCHLA[k, l:] = iCHLA[k, l - 1] * np.ones(np.shape(iCHLA)[1] - l)

        return iCHLA, iCHLA_MAX


    def pCORR_CB(chl_prc_s, bbp_ds_s, pres, zeu):

        CORR_CB = np.nan * np.zeros(np.shape(pres)[0])

        for i in range(np.shape(pres)[0]):

            if not np.isnan(zeu[i]):

                A, B, P = chl_prc_s[i], bbp_ds_s[i], pres[i]
                A, B, P = A[~np.isnan(A)], B[~np.isnan(A)], P[~np.isnan(A)]
                A, B, P = A[~np.isnan(B)], B[~np.isnan(B)], P[~np.isnan(B)]

                A, B = A[P < 2 * zeu[i] / VAR.hf], B[P < 2 * zeu[i] / VAR.hf]

                if not (np.size(A) + np.size(B) == 0):
                    reginit = stats.linregress(A, B)
                    CORR_CB[i] = reginit.rvalue

        return CORR_CB


    def pIPAR_ML(par_f, pres, mld):

        iPAR_ML = np.nan * np.zeros(np.shape(pres)[0])

        for i in range(np.shape(pres)[0]):

            parfiti, presi, mldi = par_f[i], pres[i], mld[i]
            parfiti, presi = parfiti[~np.isnan(parfiti)], presi[~np.isnan(parfiti)]
            parfiti, presi = parfiti[presi < mldi], presi[presi < mldi]
            parfiti, presi = parfiti[presi > 0.], presi[presi > 0.]

            if np.size(parfiti) > 1:

                f = interpolate.interp1d(presi, np.log(parfiti), fill_value='extrapolate')

                upperz = np.arange(0., np.nanmin(presi), 0.5)
                presi = np.hstack([upperz, presi])
                parfiti = np.hstack([np.array(np.exp(f(upperz))), parfiti])
                iPAR_ML[i] = np.sum([(parfiti[k] + parfiti[k + 1]) / 2 * (presi[k + 1] - presi[k])
                                     for k in range(np.size(presi) - 1)])
        return iPAR_ML

    time = t.time()

    prof = RAW(pathorprof)
    wmo = GET_wmo(prof)

    if verbose:
        CPRINT('Computing enhanced Sprof for {}:\n'.format(VAR.floats_names[wmo]), attrs=['BLUE', 'UNDERLINE'])
        CPRINT('Profiles filtering:\n'.format(VAR.floats_names[wmo]), attrs=['BLUE', 'ITALIC'])

    prof = OPN(wmo, verbose=verbose)

    pres = prof.PRES.values
    iso15 = prof.ISO15.values
    mld_s03 = prof.MLD_S03.values
    mld_s125 = prof.MLD_S125.values
    chl0 = prof.CHLA_ZERO.values
    chl0_s = FLT_mean(chl0, pres, verbose=False)
    cdark_depth = prof.CDARK_DEPTH.values
    bbp_ds = prof.BBP700_DS.values
    bbp_ds_s = FLT_mean(bbp_ds, pres, verbose=False)
    par_f = prof.DOWNWELLING_PAR_FIT.values
    slope = np.median(prof.SLOPEF490.values[prof.DOWNWELLING_PAR_FLG.values < 3]
                      [prof.SLOPEF490_QI.values[prof.DOWNWELLING_PAR_FLG.values < 3] < 3])
    zeu = prof.ZEU.values

    # Chlorophyll processing

    CHLA_PRC = pCHLA_PRC(pres, iso15, mld_s03, chl0, chl0_s, bbp_ds, bbp_ds_s, par_f, slope)
    chl_prc_s = FLT_mean(CHLA_PRC, pres, verbose=False)

    # SCM

    SCM = pSCM(chl_prc_s, pres)

    # SCM features (depths, values, widths)

    SCM_GDEPTH, SCM_GVAL, SCM_GWIDTH = pSCM_features(chl_prc_s, pres, SCM, cdark_depth)

    # PAR_SCM

    PAR_SCM = pPAR_SCM(par_f, pres, SCM)

    # Integrated CHLA

    iCHLA_PRC, iCHLA_MAX = pICHL(chl_prc_s, pres)

    # Correlation CHLA/BBP

    CORR_CB = pCORR_CB(chl_prc_s, bbp_ds_s, pres, zeu)

    # Integrated PAR in the ML

    iPAR_ML = pIPAR_ML(par_f, pres, mld_s125)

    # Converting to xarray

    SCM = xr.DataArray(SCM, dims=['N_PROF'], name='SCM')
    SCM_GDEPTH = xr.DataArray(SCM_GDEPTH, dims=['N_PROF'], name='SCM_GDEPTH')
    SCM_GVAL = xr.DataArray(SCM_GVAL, dims=['N_PROF'], name='SCM_GVAL')
    SCM_GWIDTH = xr.DataArray(SCM_GWIDTH, dims=['N_PROF'], name='SCM_GWIDTH')
    CHLA_PRC = xr.DataArray(CHLA_PRC, dims=['N_PROF', 'N_LEVELS'], name='CHLA_PRC')
    iCHLA_PRC = xr.DataArray(iCHLA_PRC, dims=['N_PROF', 'N_LEVELS'], name='iCHLA_PRC')
    iCHLA_MAX = xr.DataArray(iCHLA_MAX, dims=['N_PROF'], name='iCHLA_MAX')
    CORR_CB = xr.DataArray(CORR_CB, dims=['N_PROF'], name='CORR_CB')
    iPAR_ML = xr.DataArray(iPAR_ML, dims=['N_PROF'], name='iPAR_ML')
    PAR_SCM = xr.DataArray(PAR_SCM, dims=['N_PROF'], name='PAR_SCM')

    # Assigning

    processed = {'SCM': SCM,
                 'SCM_GDEPTH': SCM_GDEPTH,
                 'SCM_GVAL': SCM_GVAL,
                 'SCM_GWIDTH': SCM_GWIDTH,
                 'CHLA_PRC': CHLA_PRC,
                 'iCHLA_PRC': iCHLA_PRC,
                 'iCHLA_MAX': iCHLA_MAX,
                 'CORR_CB': CORR_CB,
                 'iPAR_ML': iPAR_ML,
                 'PAR_SCM': PAR_SCM}

    PRC = prof.assign(processed)
    PRC.attrs['history'] = 'Added new variables and processed: {}'.format(dt.datetime.today().strftime('%Y-%m-%d'))
    PRC.attrs['source'] = 'ArgoData package (thomas.hermilly@ird.fr)'

    # Saving

    if save:

        if verbose:
            CPRINT('Now saving...', attrs='CYAN', end='\r')

        name = '{}_processed.nc'.format(int(PRC.PLATFORM_NUMBER.values[0]), dt.datetime.today().strftime('%Y%m%d'))
        if name in os.listdir(VAR.prcpath):
            os.remove(VAR.prcpath+name)

        SAVE(PRC, name)

    TINFO(time, np.size(prof.N_PROF.values) * 0.8, '\nAdded processed variables', verbose)

    if verbose:
        CPRINT('\n')

    return PRC


def PRC(pathorprof, verbose=True):
    '''
    Opens processed Sprof. If not already under VAR.prcpath, processes it (can take more than a minute).

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The processed Sprof, as xarray. (xarray)
    '''

    ROOT()

    if type(pathorprof) is list:

        return [PRC(p) for p in pathorprof]

    try:

        if type(pathorprof) is str:
            prof = xr.open_dataset(VAR.prcpath + pathorprof)
        elif type(pathorprof) in [int, np.int, np.int64]:
            prof = xr.open_dataset(VAR.prcpath + '{}_processed.nc'.format(FMT_wmo(pathorprof)))
        else:
            prof = pathorprof

    except FileNotFoundError:

        CPRINT('Processed file not found. Do you want to try to process it now? (y/n)', attrs='BLUE', end='\r')
        ans = INP_timeout(20)

        if ans in ['Y', 'y', 'noinput']:

            CPRINT('Now processing...', attrs='BLUE', end='\r')
            prof = ADD_prc(RAW(pathorprof), verbose=False)

        else:

            CPRINT('Aborting.', attrs='YELLOW', end='\r')
            return None

    return prof


def CMP_stats(pathorprof, verbose=True):
    '''
    Returns a dictionary with all variable statistics on a profile, or a list of profiles. Contains
    number of values, averages, mins, max, var and std with associated key 'VAR_NVAL', 'VAR_MIN'...etc. If input is a
    list of profiles, returns a dictionary of dictionaries, with float WMOs as keys, and an 'all' key that gathers stats
    on all the profiles.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The statistics dictionary. (dictionary)
    '''

    time = t.time()

    CPRINT('Doing the stats...', attrs='CYAN', end='\r')

    def global_stats(dict):

        def flatten(list):
            return [item for sublist in list for item in sublist]

        wmos = list(dict.keys())
        keys = np.unique(flatten([list(dict[key].keys()) for key in dict.keys()]))

        global_dict = {}

        for key in keys:

            base = []
            for wmo in wmos:
                if key in dict[wmo].keys():
                    base.append((dict[wmo])[key])

            base_var = []
            for wmo in wmos:
                if key in dict[wmo].keys() and key[-4:] == '_VAR':
                    base_var.append(((dict[wmo])[key[:-4] + '_NVAL'] - 1) * (dict[wmo])[key])

            try:

                if key[-5:] == '_NVAL':
                    global_dict[key] = int(np.nansum(base))
                if key[-5:] == '_MEAN':
                    global_dict[key] = np.nanmean(base)
                elif key[-4:] == '_MIN':
                    global_dict[key] = np.nanmin(base)
                elif key[-4:] == '_MAX':
                    global_dict[key] = np.nanmax(base)
                elif key[-4:] == '_VAR':
                    nval = global_dict[key[:-4] + '_NVAL']
                    global_dict[key] = np.nansum(base_var) / (nval - 2)
                    global_dict[key[:-4] + '_STD'] = np.sqrt(global_dict[key])
                else:
                    if not (key in global_dict.keys()) and key[-4:] != '_STD':
                        CPRINT('You forgot a case, dumbass!', attrs='YELLOW')

            except KeyError:

                pass

        for key in global_dict.keys():
            if global_dict[key] == np.nan:
                global_dict.pop(key)

        return global_dict

    proflist = RAW(pathorprof)

    if type(proflist) is list:

        dict = {}

        for i, p in enumerate(proflist):

            dict[int(p.PLATFORM_NUMBER.values[0])] = CMP_stats(p, verbose=False)

        dict['all'] = global_stats(dict)

        TINFO(time, len(proflist)*10., 'Did the quick maths', verbose)

    else:

        prof = proflist
        biovars = GET_biovars(prof, verbose=False)
        dict = {}

        for var in prof.data_vars:

            if var in biovars or var in ['LONGITUDE', 'LATITUDE', 'TEMP', 'TEMP_ADJUSTED', 'PSAL', 'PSAL_ADJUSTED',
                                         'JULD', 'CT', 'MLD', 'SIG0', 'BVF', 'CT_ADJUSTED', 'MLD_ADJUSTED',
                                         'SIG0_ADJUSTED', 'BVF_ADJUSTED']:

                try:

                    nval = int(np.sum((~np.isnan(prof[var].values)).astype(int)))

                    if nval > 0:

                        if var == 'JULD':

                            dict[str(var)+'_NVAL'] = nval
                            dict[str(var)+'_MIN'] = np.nanmin(prof[var].values)
                            dict[str(var)+'_MAX'] = np.nanmax(prof[var].values)

                        else:

                            dict[str(var) + '_NVAL'] = nval
                            dict[str(var) + '_MEAN'] = np.nanmean(prof[var].values)
                            dict[str(var) + '_MIN'] = np.nanmin(prof[var].values)
                            dict[str(var) + '_MAX'] = np.nanmax(prof[var].values)
                            dict[str(var) + '_VAR'] = np.nanvar(prof[var].values)
                            dict[str(var) + '_STD'] = np.sqrt(dict[str(var) + '_VAR'])

                except Exception as e:

                    CPRINT(e, attrs='RED')

        TINFO(time, np.size(prof.N_PROF.values) * 1e-3, 'Did the quick maths', verbose)

    return dict


def GET_randomdate(pathorprof, n=1):
    '''
    Returns n random profile date(s), without repetitions.

    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param n: The number of dates you need. (int, default is 1)
    :return: The random dates. (datetime or list of datetime objects)
    '''

    prof = RAW(pathorprof)
    N = prof.N_PROF.values[-1]
    indexes = np.random.choice(np.arange(0., N, 1.), size=n, replace=False).astype(int)

    dates = FMT_date(prof.JULD.values[indexes], 'dt', verbose=False)
    dates = [dt.datetime(date.year, date.month, date.day) for date in dates]

    if len(dates) == 1:
        dates = dates[0]

    return dates


def GET_profnum(pathorprof, date, thresh=15., return_closest=False):
    '''
        Returns profile number at a certain date.

        ____

        Written by: T. Hermilly

        Contact: thomas.hermilly@ird.fr.

        Last update: 2023-02-09
        :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
         xarray or list)
        :param date: The date of interest. (datetime)
        :param thresh: The number of days maximum separating closest profile date and given date. (float, default is 15.)
        :param return_closest: Whether to overcome th thresh parameter. (bool, default is False)
        :param verbose: Whether to display information to the console. (boolean, default is True)
        :return: The N_PROF corresponding to the input date. (int)
        '''
    prof = RAW(pathorprof)

    dates = FMT_date(prof.JULD.values, 'dt', verbose=False)
    date = FMT_date(date, 'dt', verbose=False)

    if np.nanmin(np.abs(dates - date)) < dt.timedelta(days=thresh) or return_closest:
        argmin = np.nanargmin(np.abs(dates - date))
        return prof.N_PROF.values[argmin]
    else:
        war.warn('No profile has been detected within {:.0f} days of your date. If you need a date anyway, set param'
                 ' return_closest to True or extend thresh.'.format(thresh), stacklevel=2)
        return None


def GET_floatpos(pathorprof, date, thresh=35., return_closest=False):
    '''
    Returns float position at a certain date.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param date: The date of interest. (datetime)
    :param thresh: The number of days maximum separating closest profile date and given date. (float, default is 35.)
    :param return_closest: Whether to overcome thresh parameter. (bool, default is False)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The longitude and latitude of closest profile in time. (float, float)
    '''

    prof = RAW(pathorprof)
    date = FMT_date(date, 'dt', verbose=False)
    dates = FMT_date(prof.JULD.values, 'dt', verbose=False)
    i = np.argmin(np.abs(dates-date))
    lon, lat = prof.LONGITUDE.values[i], prof.LATITUDE.values[i]

    ndays = np.min(np.abs(dates-date))
    if ndays > dt.timedelta(thresh):
        CPRINT('There is {:.0f} days between your date and the closest float profile.'.format(ndays.days),
                      attrs='YELLOW')
        if not return_closest:
            return np.nan, np.nan

    return np.round(lon, 3), np.round(lat, 3)


def GET_activity(pathorprof, var='CHLA', verbose=True):
    '''
    Returns min and max date for float activity, regarding a specific variable.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param var: The variable of interest. (str, default is 'CHLA')
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return:
    '''
    ti = t.time()

    prof = RAW(pathorprof)

    datemin = FMT_date(np.nanmin(prof.JULD.values[~np.isnan(prof[var].values).all(axis=1)]), 'dt', verbose=False)
    datemax = FMT_date(np.nanmax(prof.JULD.values[~np.isnan(prof[var].values).all(axis=1)]), 'dt', verbose=False)

    TINFO(ti, 0.15, 'Determined activity period', verbose)

    return datemin, datemax


def GET_biovars(pathorprof, verbose=True):
    '''
    Returns the profile bgc variables list, if they are not only NaNs. If verbose is set to True, prints
    variables to the console. Legend: green is for variables available as 'ADJUSTED', blue is for variable for which at
    least one value is non NaN, orange is for variables that have everywhere a QC code '3' and red is for variables that
    have only a QC code '4'.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The list of variable names. (list of str)
    '''

    ti = t.time()

    if type(pathorprof) is list:

        res = [GET_biovars(RAW(prof), verbose=False) for prof in pathorprof]

        TINFO(ti, np.sum([np.size(RAW(prof).N_PROF.values) for prof in pathorprof]) * 0.02, 'Determined floats biovars',
              verbose)

        return res

    else:

        prof = RAW(pathorprof)

        biovars = ['CHLA',
                   'BBP700',
                   'CDOM',
                   'DOWNWELLING_PAR',
                   'DOWN_IRRADIANCE380',
                   'DOWN_IRRADIANCE412',
                   'DOWN_IRRADIANCE490',
                   'DOXY',
                   'NITRATE',
                   'PH_IN_SITU_TOTAL']

        biovars = biovars + [var + '_ADJUSTED' for var in biovars]

        profvars_colors = {}
        profvars = []

        for var in biovars:
            if var in prof.data_vars:
                if not (np.isnan(prof[var].values).all()):
                    profvars.append(var)
                    profvars_colors[var] = 'BLUE'

        if verbose:

            prof = FLT_qcs(prof, QCs=4, verbose=False)

            for var in profvars:
                if var+'_ADJUSTED' in profvars:
                    profvars_colors[var] = 'GREEN'

            for var in profvars:
                if (np.isnan(prof[var].values).all()):
                    profvars_colors[var] = 'DARK_CYAN'

            prof = FLT_qcs(prof, QCs=(3, 4), verbose=False)

            for var in profvars:
                if (np.isnan(prof[var].values).all()):
                    profvars_colors[var] = 'RED'

            for var in profvars:
                if not '_ADJUSTED' in var:
                    CPRINT(var, attrs=profvars_colors[var])

        TINFO(ti, np.size(prof.N_PROF.values) * 0.02, 'Determined float biological variables', verbose)

        return profvars


def CMP_pSCMthresh(pathorprof, thresh=1.5):
    '''
    Computes the annual SCM occurence percentage for a float, using a simple threshold criteria for SCM detection.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param thresh: The thresh paramater so that to detect a SCM, a profile needs to verify CHLmax > thresh * CHLsurf.
     (float, default is 1.5)
    :return: SCM occurence percentage. (float)
    '''

    prof = PRC(pathorprof)
    wmo = GET_wmo(prof)
    activity = GET_activity(prof)

    def pSCM(chl_prc_s, pres, thresh=thresh):

        SCM = np.nan * np.zeros(np.shape(pres)[0])

        for k in range(np.shape(SCM)[0]):

            if not np.isnan(chl_prc_s[k]).all():
                if pres[k, np.nanargmax(chl_prc_s[k])] * VAR.hf > 10 and \
                    np.nanmax(chl_prc_s[k]) >= thresh * np.nanmean(chl_prc_s[k][pres[k] < 10.]):

                    SCM[k] = pres[k, np.nanargmax(chl_prc_s[k])] * VAR.hf

        return SCM

    SCM = pSCM(FLT_mean(prof.CHLA_PRC.values, prof.PRES.values, verbose=False), prof.PRES.values)

    if activity[1] - activity[0] < dt.timedelta(days=365.):

        CPRINT('Float {} has less than a year of data. Impossible to compute percentage of SCM over a year.'
                      .format(VAR.floats_names[wmo]), attrs='YELLOW')

        return None

    else:

        date1 = activity[0]
        while date1 <= activity[1]:
            date1 += dt.timedelta(days=365.)
        date1 -= dt.timedelta(days=365.)

        date2 = activity[1]
        while date2 >= activity[0]:
            date2 -= dt.timedelta(days=365.)
        date2 += dt.timedelta(days=365.)

        temp1 = np.arange(0, (activity[1] - activity[0]).days, 5.)
        temp2 = np.array([float(e.days) for e in (FMT_date(prof.JULD.values, 'dt', verbose=False)
                          - FMT_date(prof.JULD.values[0], 'dt', verbose=False))])
        ti = np.array([activity[0] + dt.timedelta(days=e) for e in temp1])

        dcm = np.array([0. if np.isnan(e) else 1. for e in SCM])
        dcm = interpolate.interp1d(temp2, dcm, kind='nearest')
        dcm = dcm(temp1)

        # Compute proportion of DCMs in period 1 and 2, and average in the end

        pcDCM1 = np.sum(dcm[ti < date1])/np.size(dcm[ti < date1])
        pcDCM2 = np.sum(dcm[ti > date2])/np.size(dcm[ti > date2])
        pcDCM = np.mean([pcDCM1, pcDCM2])

        return np.round(100. * pcDCM, 2)