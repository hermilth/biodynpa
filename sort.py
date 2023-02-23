# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the script dedicated to profiles sorting. You can sort according to sensors on the Argo greylist, QCs,
datamodes, and custom filters.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''


# Imports


from filters import *


# Filters


def FLT_greylist(pathorprof, QCs=4, reject=True, verbose=True):
    '''
    Erases data in greylist (ARGO file reporting problems in sensors).

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param QCs: The QC codes to get rid of, or to keep in the data. (int or list of int, default is 4)
    :param reject: Rejects the given QCs. If set to False, keeps the given QCs. (boolean, default is True)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The filtered profiles. (xarray)
    '''

    time = t.time()

    try:

        greylist = open(VAR.indexpath+'ar_greylist_bgc_sp.txt')

    except FileNotFoundError:

        raise FileNotFoundError('Run DL_greylist() from downloads.py first, so that the greylist be in your files.')

    if type(QCs) is int:
        QCs = [QCs]
    QCs = list(QCs)

    if reject:
        QCs_reject = QCs
        QCs = []
        for i in range(10):
            if i not in QCs_reject:
                QCs.append(i)
    else:
        QCs_reject = []
        for k in range(10):
            if k not in QCs:
                QCs_reject.append(k)

    prof = RAW(pathorprof)
    wmo = int(prof.PLATFORM_NUMBER.values[0])

    greylist.readline()

    for l in greylist:
        a = CUT_line(l, ',')
        if int(a[0]) == wmo:
            if int(a[4]) in QCs_reject:

                datemin, datemax = a[2], a[3]
                datemin = dt.datetime.strptime(datemin, '%Y%m%d')
                try:
                    datemax = dt.datetime.strptime(datemax, '%Y%m%d')
                except ValueError:
                    datemax = dt.datetime.today()+dt.timedelta(1)

                var = a[1]

                times = FMT_date(prof.JULD.values, 'dt', verbose=False)
                erase = []

                for i, ti in enumerate(times):
                    if datemin < ti < datemax:
                        erase.append(i)

                data = prof[var].values
                data[erase] = np.nan * np.zeros(np.shape(data[erase]))
                newvar = xr.DataArray(data, dims=prof[var].dims, name=prof[var].name)

                prof.update({var: newvar})

    TINFO(time, 3., 'Erased data in greylist', verbose)

    return prof


def FLT_qcs(pathorprof, QCs=4, reject=True, verbose=True):
    '''
    Keeps only the input QCs in adjusted profiles data.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param QCs: The QC codes to get rid of, or to keep in the data. (int or list of int, default is 4)
    :param reject: Rejects the given QCs. If set to False, keeps the given QCs. (boolean, default is True)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The filtered profiles. (xarray)
    '''

    time = t.time()

    if type(QCs) is int:

        QCs = [QCs]

    QCs = list(QCs)

    if reject:

        QCs_reject = QCs
        QCs = []

        for i in range(10):
            if i not in QCs_reject:
                QCs.append(i)

    else:

        QCs_reject = []
        for k in range(10):
            if k not in QCs:
                QCs_reject.append(k)

    if type(pathorprof) is list:

        res = [FLT_qcs(p, QCs=QCs, reject=False, verbose=False) for p in pathorprof]

        TINFO(time, len(pathorprof)*np.sum([np.size(p.N_PROF.values)/300 for p in pathorprof]), 'Filtered unwanted QCs',
              verbose)

        return res

    else:

        prof = RAW(pathorprof)
        newprof = prof.copy(deep=True)

        if np.size(newprof.N_PROF) > 0:

            if QCs is not None:

                for var in newprof.data_vars:

                    # We consider here that if a pressure is not correct, the variables QCs will indicate it.

                    if var + '_QC' in newprof.data_vars and var not in ['JULD', 'PRES']:
                        for qcval in QCs_reject:
                            newprof.update({var: newprof[var].where(newprof[var + '_QC'] != str.encode(str(qcval)))})

        TINFO(time, np.size(prof.N_PROF.values)/100, 'Filtered unwanted QCs', verbose)

        return newprof


def FLT_datamode(pathorprof, dm='R', keep=False, verbose=True):
    '''
    Erases profiles according to the parameter datamode.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param dm: The datamode to keep or reject. (str in ['A', 'R', 'D'], default is 'R')
    :param keep: Whether to reject or to keep the given datamodes. (boolean, default is False)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The extracted profiles. (xarray)
    '''

    time = t.time()

    if dm is None:
        dm = []
    elif not type(dm) is list:
        dm = [dm]

    if keep:

        keep_dm = dm
        dm = []

        for e in ['R', 'A', 'D']:
            if e not in keep_dm:
                dm.append(e)

    dm = [bytes(e, 'utf-8') for e in dm]

    prof = RAW(pathorprof).copy()
    n_levels = np.size(prof.N_LEVELS.values)

    params = {}
    for i, var in enumerate(prof.PARAMETER.values[0, 0]):
        params[i] = str(var.strip())[2:-1]

    for i in params:

        var = params[i]

        var_dm = prof.PARAMETER_DATA_MODE.values[:, i]
        mask = np.array([e in dm for e in var_dm])
        mask = np.vstack([mask for _ in range(n_levels)]).T

        masked = prof[var].values.copy()
        masked = np.where(mask, np.nan, masked)

        newvar = xr.DataArray(masked, dims=['N_PROF', 'N_LEVELS'], attrs=prof[var].attrs, name=var)

        prof.update({var: newvar})

    TINFO(time, np.size(prof.N_PROF.values)/1500, 'Filtered unwanted datamodes', verbose)

    return prof


def FLT(pathorprof, filter, verbose=True):
    '''
    Erases values of profiles accordingly to a given filter (e.g. {'N_PROF': [30, 31, 32]} keeps oly
    profiles that have N_PROF equal to 30, 31 or 32).

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :param filter: The parameters you want to filter with and their values as: {'FILTER_PARAM': value}. Can be followed
     by "_MIN" or "_MAX", or even be a list for numeric values. (dictionary)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The extracted profiles. (xarray)
    '''

    time = t.time()

    def isin(a, b):
        return [x in a for x in b]

    # Opening file and initializing condition

    prof = RAW(pathorprof)
    cond = np.ones(np.size(prof.N_PROF.values), dtype=int)

    # Iterating filter

    for x in filter:

        # In case condition is on the JULD (time) variable

        if x[:4] == 'JULD':

            # Rounding dates to the day

            conversion = FMT_date(prof.JULD.values, 'dt', verbose=False)
            conversion = np.array([dt.datetime(da.year, da.month, da.day) for da in conversion])
            imposed_date = FMT_date(filter[x], 'dt', verbose=False)
            if not type(imposed_date) is np.ndarray:
                imposed_date = [imposed_date]
            imposed_date = np.array([dt.datetime(da.year, da.month, da.day) for da in imposed_date])

            if x[-4:] == '_MAX':
                cond = np.multiply(cond, conversion <= imposed_date)
            elif x[-4:] == '_MIN':
                cond = np.multiply(cond, conversion >= imposed_date)
            else:
                cond = np.multiply(cond, isin(imposed_date, conversion))

        # If the condition is on a numeric variable and has '_MIN' or '_MAX' added in the end of its name

        elif x[-4:] == '_MAX':
            cond = np.multiply(cond, prof[x[:-4]].values <= filter[x])
        elif x[-4:] == '_MIN':
            cond = np.multiply(cond, prof[x[:-4]].values >= filter[x])

        # Other cases (equality of numeric vars or non-numeric variables

        else:

            if type(filter[x]) is list:
                cond = np.multiply(cond, isin(filter[x], prof[x].values))
            else:
                cond = np.multiply(cond, prof[x].values == filter[x])

    # Indexes to be dropped are dropped

    drop = np.where(~(cond.astype(bool)), prof.N_PROF.values, np.nan)
    keep = np.where(cond.astype(bool), prof.N_PROF.values, np.nan)
    drop, keep = drop[~np.isnan(drop)], keep[~np.isnan(keep)]

    extracted_prof = prof.drop_sel(N_PROF=drop)
    extracted_prof.update({'N_PROF': keep})

    if verbose and np.size(extracted_prof.JULD.values) == 0:
        CPRINT('Extraction is empty.', attrs='YELLOW')

    TINFO(time, len(filter) * np.size(prof.N_PROF.values)/1500, 'Filtered profile', verbose)

    return extracted_prof
