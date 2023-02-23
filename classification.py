# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the script used to classify profiles according to their shapes, and therefore characterize their bioregions.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''

# Imports

from profiles import *

# Functions


def CLS_TYPE(prc):
    '''
    Classifies profiles as sigmoid, gaussian or other, according to the fit quality to different functions.\n

    ____

    SIGMOID(z) = Ks / (1 + np.exp((z - Z12s) * s))\n
    GAUSSIAN(z) = Ke * np.exp(-np.log(2) * z / Z12e) + Kg * np.exp(-((z - Zm) / sigg) ** 2)

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param prc: The processed profile. (int or xarray)
    :return: Returns profiles type (str), best fit parameters Ks, Z12s, s, Ke, Z12e, Kg, Zm, sigg, fits r-squared
     values, slopes and intercepts (all numpy ndarrays).
    '''

    def Fsgm(z, Ks, Z12s, s):
        return Ks / (1 + np.exp((z - Z12s) * s))

    def Fexp(z, Ke, Z12e):
        return Ke * np.exp(-np.log(2) * z / Z12e)

    def Fgss(z, Kg, Zm, sigg):
        return Kg * np.exp(-((z - Zm) / sigg) ** 2)

    def Fsum(z, Ke, Z12e, Km, Zm, sigg):
        return Fexp(z, Ke, Z12e) + Fgss(z, Km, Zm, sigg)

    war.filterwarnings('ignore', category=RuntimeWarning)

    min_corr = 0.85
    prof = PRC(prc)

    type = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    Ks_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    Z12s_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    s_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    Ke_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    Z12e_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    Kg_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    Zm_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])
    sigg_a = np.nan * np.zeros(np.shape(prof.PRES.values)[0])

    R2, SLOPE, INTER = np.nan * np.zeros(np.shape(prof.PRES.values)[0]), \
        np.nan * np.zeros(np.shape(prof.PRES.values)[0]), np.nan * np.zeros(np.shape(prof.PRES.values)[0])

    for k in range(np.shape(prof.PRES.values)[0]):

        chla, pres = prof.CHLA_PRC.values[k], prof.PRES.values[k]
        chla, pres = chla[pres * VAR.hf < 400.], pres[pres * VAR.hf < 400.]
        chla = FLT_mean(chla, pres, verbose=False)
        chla, pres = chla[~np.isnan(chla)], pres[~np.isnan(chla)]

        if np.size(chla) > 10:

            rsg = opt.curve_fit(Fsgm, xdata=pres, ydata=chla, p0=(.3, 30., 0.2), bounds=([.1, 4., 0.], [3., 250., 1.]))
            regsg = stats.linregress(chla, Fsgm(pres, rsg[0][0], rsg[0][1], rsg[0][2]))
            rgs = opt.curve_fit(Fsum, xdata=pres, ydata=chla, p0=(.1, 20., 0.3, 100., 30.),
                                bounds=([0., 3., 0.05, 20., 5.], [1., 50., 1.5, 250., 100.]))
            reggs = stats.linregress(chla, Fsum(pres, rgs[0][0], rgs[0][1], rgs[0][2], rgs[0][3], rgs[0][4]))

            if regsg.rvalue ** 2 < min_corr and reggs.rvalue ** 2 < min_corr:

                type[k] = 0

            else:

                if np.argmax([regsg.rvalue ** 2, reggs.rvalue ** 2]) == 1 and\
                        reggs.rvalue ** 2 >= regsg.rvalue ** 2 + 0.05: # As the gaussian shape has 2 paramters more than
                    # the sgm, it has to have a r**2 at least more than 0.05 superior to the sigmoid one

                    type[k] = 1
                    Ke_a[k], Z12e_a[k], Kg_a[k], Zm_a[k], sigg_a[k] = rgs[0]
                    R2[k], SLOPE[k], INTER[k] = reggs.rvalue**2, reggs.slope, reggs.intercept

                else:

                    if regsg.rvalue ** 2 > min_corr:
                        type[k] = 2
                        Ks_a[k], Z12s_a[k], s_a[k] = rsg[0]
                        R2[k], SLOPE[k], INTER[k] = regsg.rvalue**2, regsg.slope, regsg.intercept
                    else:
                        type[k] = 0


    type = xr.DataArray(type, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF', 
                        attrs={'Info': '0 for unclassified, 1 for SCM, 2 for sigmoid.'})
    Ks_a = xr.DataArray(Ks_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF', 
                        attrs={'Info' : 'Intensity parameter of the sigmoid profile type.'})
    Z12s_a = xr.DataArray(Z12s_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF', 
                          attrs={'Info' : 'Half maximum value depth parameter of the sigmoid profile type.'})
    s_a = xr.DataArray(s_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF',
                       attrs={'Info' : 'Slope parameter of the sigmoid profile type.'})
    Ke_a = xr.DataArray(Ke_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF', 
                        attrs={'Info' : 'Intensity parameter of the exponential component of the SCM profile type'})
    Z12e_a = xr.DataArray(Z12e_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF',
                          attrs={'Info' : 'Half maximum value parameter of the exponential component of the SCM profile'
                                          ' type'})
    Kg_a = xr.DataArray(Kg_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF',
                        attrs={'Info' : 'Intensity parameter of the gaussian component of the SCM profile type'})
    Zm_a = xr.DataArray(Zm_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF',
                        attrs={'Info' : 'Maximum depth parameter of the gaussian component of the SCM profile type'})
    sigg_a = xr.DataArray(sigg_a, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF', 
                          attrs={'Info' : 'Width (standard deviation) parameter of the gaussian component of the SCM '
                                          'profile type'})
    R2 = xr.DataArray(R2, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF',
                      attrs={'Info' : 'R squared value of the profile fit.'})
    SLOPE = xr.DataArray(SLOPE, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF', 
                         attrs={'Info' : 'Slope value of the profile fit.'})
    INTER = xr.DataArray(INTER, coords={'N_PROF': prof.N_PROF.values}, dims='N_PROF', 
                         attrs={'Info' : 'Intercept value of the profile fit.'})

    PPARAMS = xr.Dataset(coords={'N_PROF': prof.N_PROF.values}).assign({'TYPE': type,
                                                                        'KSGM': Ks_a,
                                                                        'Z12SGM': Z12s_a,
                                                                        'SSGM': s_a,
                                                                        'KEXP': Ke_a,
                                                                        'Z12EXP': Z12e_a,
                                                                        'KGSS': Kg_a,
                                                                        'ZSCM': Zm_a,
                                                                        'WSCM': sigg_a,
                                                                        'R2': R2,
                                                                        'SLOPE': SLOPE,
                                                                        'INTER': INTER})

    return PPARAMS


def CMP_ptype(prc):
    '''
    Returns yearly occurence of sigmoid, gaussian and other types of profiles.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param prc: The processed profile. (int or xarray)
    :return: Yearly occurence of sigmoid, gaussian and other types of profiles. (floats)
    '''
    prof = PRC(prc)
    wmo = GET_wmo(prof)
    PPARAMS = CLS_TYPE(prof)
    activity = GET_activity(prof)

    if activity[1] - activity[0] < dt.timedelta(days=365.):

        print(colored('Float {} has less than a year of data. Impossible to compute percentage of SCM over a year.'
                      .format(VAR.floats_names[wmo]), 'yellow'))

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

        dcm = np.array([1. if e == 1. else 0. for e in PPARAMS.TYPE.values])
        dcm = interpolate.interp1d(temp2, dcm, kind='nearest')
        dcm = dcm(temp1)

        sgm = np.array([1. if e == 2. else 0. for e in PPARAMS.TYPE.values])
        sgm = interpolate.interp1d(temp2, sgm, kind='nearest')
        sgm = sgm(temp1)

        oth = np.array([1. if e == 0. else 0. for e in PPARAMS.TYPE.values])
        oth = interpolate.interp1d(temp2, oth, kind='nearest')
        oth = oth(temp1)

        # Compute proportion of DCMs in period 1 and 2, and average in the end

        pcDCM1 = np.sum(dcm[ti < date1]) / np.size(dcm[ti < date1])
        pcDCM2 = np.sum(dcm[ti > date2]) / np.size(dcm[ti > date2])
        pcDCM = np.mean([pcDCM1, pcDCM2])

        pcSGM1 = np.sum(sgm[ti < date1]) / np.size(sgm[ti < date1])
        pcSGM2 = np.sum(sgm[ti > date2]) / np.size(sgm[ti > date2])
        pcSGM = np.mean([pcSGM1, pcSGM2])

        pcOTH1 = np.sum(oth[ti < date1]) / np.size(oth[ti < date1])
        pcOTH2 = np.sum(oth[ti > date2]) / np.size(oth[ti > date2])
        pcOTH = np.mean([pcOTH1, pcOTH2])

        return np.round(100. * pcDCM, 2), np.round(100. * pcSGM, 2), np.round(100. * pcOTH, 2)