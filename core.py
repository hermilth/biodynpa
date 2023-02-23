# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the module dedicated to basic functions. They are independant from all the other scripts.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''

import os
import sys
import signal
import shutil

import time as t
import datetime as dt
import xarray as xr
import numpy as np
import ftplib as flib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal as sig
import scipy.stats as stats
import scipy.interpolate as interpolate
import scipy.optimize as opt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings as war
import gsw as sw
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import globvars as VAR
import urllib as ulib

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
from textwrap import wrap


# MPL style


plt.rcParams.update(VAR.rcParams)


# Input timeout


def inputTimeOutHandler(signum, frame):
    raise VAR.InputTimedOut

signal.signal(signal.SIGALRM, inputTimeOutHandler)


def INP_timeout(timeout=0):
    '''
    Asks user for an entry with a time limit.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param timeout: The timeout in seconds. (float)
    :return: The user's input or 'noinput' if timeout has been reached. (str)
    '''

    timeout = int(timeout)
    unput = 'noinput'
    try:
        signal.alarm(timeout)
        unput = input()
        signal.alarm(0)
    except VAR.InputTimedOut:
        pass
    return unput


# Functions


def INIT():

    '''Run to initialize folders architecture.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    '''

    for path in [VAR.indexpath, VAR.chloropath, VAR.sprofpath, VAR.figpath, VAR.prcpath, VAR.logspath]:

        try:
            os.listdir(path)
        except FileNotFoundError:
            folders = CUT_line(path, '/')
            curpath = './'
            for i, folder in enumerate(folders):
                if folder != '':
                    if folder in os.listdir(curpath):
                        pass
                    else:
                        os.mkdir(curpath + folder)
                curpath += folder + '/'


def ROOT():
    '''
    Gets you back to your python project root.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :return: None.
    '''
    path = os.getcwd()
    i = 0
    while path[i:i + 8] != 'ArgoData' and i < len(path) - 8:
        i += 1
    c = 0
    for char in path[i + 8:]:
        if char == '/':
            c += 1
    for _ in range(c):
        os.chdir('..')


def SAVE(prof, name, folder=VAR.prcpath[:-1]):
    '''
    Saves an xarray profile to netcdf.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param prof: The Sprof as xarray. (xarray)
    :param name: The save name. (str)
    :param folder: The save folder. (str, default is VAR.prcpath)
    :return:
    '''

    def REFORMAT(prof):

        l = ['WMO_INST_TYPE', 'PARAMETER', 'FLOAT_SERIAL_NO', 'PLATFORM_NUMBER', 'STATION_PARAMETERS', 'PROJECT_NAME',
             'DATA_TYPE', 'PLATFORM_TYPE', 'PI_NAME', 'FIRMWARE_VERSION', 'POSITIONING_SYSTEM']

        for var in l:

            temp = prof[var].values

            shape = np.shape(temp)
            temp = temp.flatten()

            formatted_var = []

            for x in temp:

                x = x.decode('utf-8')

                if var in ['PROJECT_NAME', 'PI_NAME']:
                    newx = x + ' ' * (64 - len(x))
                elif var in ['PLATFORM_NUMBER', 'WMO_INST_TYPE']:
                    newx = x + ' '
                elif var in ['POSITIONING_SYSTEM']:
                    newx = x + ' ' * (8 - len(x))
                else:
                    newx = x + ' ' * (32 - len(x))

                newx = bytes(newx, 'utf-8')

                formatted_var.append(newx)

            prof[var].values = np.array(formatted_var, dtype=object).reshape(shape)

        return prof

    try:

        prof.to_netcdf(folder + '/' + name)

    except Exception:

        REFORMAT(prof).to_netcdf(folder + '/' + name)


def CUT_line(l, c):
    '''
     Cuts the string according to given separator c and returns a list of strings.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param l: the line. (str)
    :param c: the separator. (char)
    :return res: the list of strings separated by c in l. (list of str)
    '''
    
    res = []

    lastcut = -1
    for i, char in enumerate(l):
        if char == c:
            res.append(l[lastcut + 1:i])
            lastcut = i

    res.append(l[lastcut + 1:])

    return res


def WMOS(subset=None):
    '''
    Gives you the float WMOs in the Southern Pacific.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param subset: The subset of float to return WMOs from. None and 'l' are for floats with light sensors, 
     'n' for others. Give 'r' if you want a random WMO. (str of ['l', 'n', 'r'], optional)
    :return: The WMOs. (list of int)
    '''
    if not 'index_bgc_sp.txt' in os.listdir(VAR.indexpath):
        CPRINT('No index file found in \'{}\' folder!'.format(VAR.indexpath), attrs='YELLOW')

        return []

    if subset is not None:

        if subset in 'ess' or subset == 'essentials':
            return VAR.essentials
        elif subset == 'r':
            return VAR.essentials[np.random.randint(np.size(VAR.essentials))]

    file = open(VAR.indexpath+'index_bgc_sp.txt', 'r')
    l = file.readline()
    while not 'PAR' in l:
        l = file.readline()

    argo_light = CUT_line(l, ',')
    argo_light = [int(s[-8:]) for s in argo_light]

    l = file.readline()

    argo_other = CUT_line(l, ',')
    argo_other = [int(s[-8:]) for s in argo_other]

    if subset is None:

        return VAR.essentials

    elif subset == 'l':

        return argo_light

    elif subset == 'n':

        return argo_other

    elif subset in 'all':

        return argo_light+argo_other


def RAW(pathorprof):
    '''
    Opens the profiles as WMOs, xarrays or filename without altering them at all.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param pathorprof: The path, wmo or xarray of the profiles file. Can be a list of several of those. (str, int,
     xarray or list)
    :return: The Sprof as xarray. (xarray)
    '''
    if type(pathorprof) is list:

        return [RAW(p) for p in pathorprof]

    else:

        if type(pathorprof) is str:
            prof = xr.open_dataset(VAR.sprofpath + pathorprof)
        else:
            if type(pathorprof) in [int, np.int64]:
                pathorprof = FMT_wmo(pathorprof)
                prof = RAW('{}_Sprof.nc'.format(pathorprof))
            else:
                prof = pathorprof

        return prof


def GET_wmo(prof):
    '''
    Opens the profiles as WMOs, xarrays or filename without altering them at all.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param prof: The profile as xarray. (xarray)
    :return: The wmo. (int)
    '''

    return int(prof.PLATFORM_NUMBER.values[0])


def LOADOTS(k):
    '''
    The number of dots in the loading messages, as a function of iteration number.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param k: the iteration number. (int)
    :return: A string with a certain number of dots... (str)
    '''
    return '.' * (int(5 * np.log(k + 2) - 2 * np.log(k + 1) * np.sin((k + 1) / 5) - 2)//3)


def CPRINT(m, end=None, attrs=None, flush=False):
    '''
    Prints colored text in the console.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param m: The message to print in the console. (str)
    :param end: Analog to the python built-in print \'end\' parameter. (str)
    :param attrs: Font attributes (see VAR.font_attributes). (str or list of str)
    :param flush: Wether to wait for the line to complete before printing it. (bool, default is False)
    :return:
    '''

    if attrs is None:
        attrs = []
    if type(attrs)==str:
        attrs = [attrs]
    prefix = ''
    for e in attrs:
        prefix += VAR.font_attributes.__dict__[e]

    print(prefix + m + VAR.font_attributes.END, end=end, flush=flush)


def TINFO(time, secs, mess, verbose):
    '''Gives you time info and CPRINTs in yellow if the program is taking more than a given threshold.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param time: The start time. (float)
    :param secs: The expected maximum time. (float)
    :param mess: The message to CPRINT before the time info. (str)
    :param verbose: Whether to display information to the console. (boolean)
    :return: None'''

    if verbose:

        if t.time() - time > secs:
            print(VAR.font_attributes.YELLOW + mess + ': ' + VAR.font_attributes.RED + VAR.font_attributes.ITALIC +
                  VAR.font_attributes.UNDERLINE + FMT_secs(t.time() - time) + VAR.font_attributes.END)
        else:
            print(VAR.font_attributes.BLUE + mess + ': ' + VAR.font_attributes.GREEN + VAR.font_attributes.ITALIC +
                  VAR.font_attributes.UNDERLINE + FMT_secs(t.time() - time) + VAR.font_attributes.END)


def FMT_wmo(wmo, warn=True):
    '''
     Reformats your WMO number according to the S-profiles in your sprof folder.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param wmo: A sequence of digits that is in the sought WMO. (int or str)
    :param warn: Whether to warn the user about wmo confusion. (boolean, default is True)
    :return: The full WMO number. (int)
    '''
    if type(wmo) is list:

        return [FMT_wmo(w) for w in wmo]

    if not type(wmo) is str:
        wmo = str(wmo)

    res = wmo
    files = os.listdir(VAR.sprofpath)

    n = 0
    for file in files:
        if wmo in file:
            n += 1
            res = file[:7]

    res = int(res)

    if warn:
        if n == 0:
            CPRINT('No file corresponding to the given number in your {} folder.'.format(VAR.sprofpath), attrs='YELLOW')
        elif n > 1:
            CPRINT('Several files correspond your number in {} folder. Returning {}.'.format(VAR.sprofpath, res),
                   attrs='YELLOW')

    return res


def FMT_secs(s):
    '''
    Converts any number of seconds to an appropriate string.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param s: Number of seconds. (float)
    :return: The time formatted string. (str)
    '''

    if s < 1:

        return '{:.0f}ms'.format(s * 1000)

    elif s < 60:

        return '{:.2f}s'.format(s)

    elif s < 3600:

        m = int(s // 60)
        s = int(s % 60)
        return '{:02}m{:02}s'.format(m, s)

    else:

        h = int(s // 3600)
        m = int((s % 3600) // 60)
        s = int(s % 60)
        return '{:02}h{:02}m{:02}s'.format(h, m, s)


def FMT_date(date, datetype='dt', verbose=True):
    '''
     Returns a date in the given datetype.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

        :param date: Date or list of dates as 'YYY-MM-DD' or datetime object or numpy datetime64 object.
        :param datetype: The wanted output type. (str of ['str', 'dt64', 'dt'], default is 'dt')
        :param verbose: Whether to display information to the console. (boolean, default is True)
        :return: The formatted date(s).
    '''

    time = t.time()

    if type(date) is xr.DataArray:

        date = date.values

    if type(date) is list or type(date) is np.ndarray:

        res = np.array([FMT_date(da, datetype, verbose=False) for da in date])

        TINFO(time, 0.05, 'Formatted dates', verbose)

        return res

    else:

        if type(date) is str:

            datestr = date

            if 'T' in date:
                dateti = dt.datetime.strptime((np.datetime64(date, 's')).astype(str), '%Y-%m-%dT%H:%M:%S')
            else:
                dateti = dt.datetime.strptime(datestr, '%Y-%m-%d')

            dt64 = np.datetime64(datestr)

        elif type(date) is dt.datetime:

            if date.hour == 0 and date.minute == 0 and date.second == 0:
                datestr = '{0:04d}'.format(date.year) + '-{0:02d}'.format(date.month) + '-{0:02d}'.format(date.day)
            else:
                datestr = '{0:04d}'.format(date.year) + '-{0:02d}'.format(date.month) + '-{0:02d}'.format(date.day) \
                          + 'T' + '{0:02d}'.format(date.hour) + ':{0:02d}'.format(date.minute) + \
                          ':{0:02d}'.format(date.second)

            dateti = date
            dt64 = np.datetime64(datestr)

        elif type(date) is np.datetime64:

            if 'T' in date.astype(str):

                datestr = date.astype(str)[:19]
                dateti = dt.datetime.strptime(datestr, '%Y-%m-%dT%H:%M:%S')
                dt64 = np.datetime64(datestr)

            else:

                datestr = date.astype(str)
                dateti = dt.datetime.strptime(datestr, '%Y-%m-%d')
                dt64 = date

        else:

            war.warn('Cannot format date.', stacklevel=2)
            datestr, dateti, dt64 = None, None, None

        TINFO(time, 0.005, 'Formatted date', verbose)

        if datetype == 'dt':

            return dateti

        elif datetype == 'str':

            return datestr

        elif datetype == 'dt64':

            return dt64


def FMT_lon(longitude):
    '''
    Transforms longitude so as it is between 0 and 360.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param longitude: Longitude. (float or array)
    :return: The Longitude between 0 and 360. (float or array)
    '''

    if type(longitude) is np.ndarray:

        n = np.size(longitude)
        res = np.zeros((n,))

        for i in range(n):
            res[i] = FMT_lon(longitude[i])

        return res

    else:

        while longitude < 0.:
            longitude += 360.
        while longitude >= 360.:
            longitude -= 360.

        return longitude


def D1(AR, PR):

    '''
    Computes the first derivative of a 2D array, along dimension 1.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param AR: The array. (numpy array)
    :param PR: The coordinates. (numpy array)
    :return: The derivative of that array along given coordiantes. (numpy array)
    '''

    def aux(ar, pr):

        mask = ~np.isnan(ar)

        if np.sum(mask) > 3:

            x, y = pr[mask], ar[mask]
            diff = (y[2:] - y[:-2]) / (x[2:] - x[:-2])

            val0 = np.nan
            val_1 = np.nan

            diff = np.hstack([val0, diff, val_1])

            res = np.nan * np.zeros(np.size(mask))
            res[mask] = diff

        else:

            res = np.nan * np.zeros(np.size(ar))

        return res

    if np.size(np.shape(AR)) == 1:

        AR, PR = AR[np.newaxis], PR[np.newaxis]

    diff = np.nan * np.zeros(np.shape(AR))

    for i in range(np.shape(diff)[0]):

        diff[i] = aux(AR[i], PR[i])

    return diff.squeeze()


def INT_1D(AR, COO, NODES):
    '''
    Interpolates 2D arrays along dimension 1.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param AR: The array. (numpy array)
    :param COO: The coordinates. (numpy array)
    :param NODES: The new coordinates. It can be 1D (in case you want to always interpolate on the same coordinates) or
     2D. (numpy array)
    :return: The interpolated array. (numpy array)
    '''

    if np.size(np.shape(AR)) == 1:
        nprof = 1
        arr = AR.copy()[np.newaxis]
    else:
        nprof = np.shape(AR)[0]
        arr = AR.copy()

    if np.size(np.shape(COO)) == 1:
        coo = np.vstack([COO for _ in range(nprof)])
    else:
        coo = COO.copy()

    if not np.shape(arr) == np.shape(coo):
        raise Exception('Input shapes should match.')

    if np.size(np.shape(NODES)) == 1:
        nodes = np.vstack([NODES for _ in range(nprof)])
    else:
        nodes = NODES.copy()

    nlevels, nnodes = np.shape(arr)[1], np.shape(nodes)[1]
    res = np.nan * np.zeros((nprof, nnodes))

    for i in range(nprof):

        x, xp, yp = nodes[i][~np.isnan(nodes[i])], coo[i][~np.isnan(arr[i])], arr[i][~np.isnan(arr[i])]

        if np.size(xp) > 0 and np.size(x) > 0:

            res[i][~np.isnan(nodes[i])] = np.interp(x, xp, yp, left=np.nan, right=np.nan)

    return res


def ALIGN(AR, PR, dmax, resolution=500):
    '''
    Aligns pressure levels on unaligned profiles.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param AR: Numpy array with profiles along first axis. (numpy array)
    :param PR: Numpy array with profiles pressure data. (numpy array)
    :param dmax: Maximum depth of alignement. (float)
    :param resolution: Number of points between surface and dmax. (int, default is 500)
    :return da: A xarray DataArray with dimensions ('N_PROF', 'DEPTH'). (xarray DataArray)
    '''

    war.filterwarnings('ignore', category=FutureWarning)
    try:
        AR = AR.values
    except AttributeError:
        pass

    try:
        PR = PR.values
    except AttributeError:
        pass

    Y = np.linspace(0., dmax / VAR.hf, resolution)
    vals = INT_1D(AR, PR, Y)
    da = xr.DataArray(data=vals, dims=['N_PROF', 'DEPTH'], coords=dict(N_PROF=np.arange(np.shape(AR)[0]),
                                                                       DEPTH=Y * VAR.hf))
    da = da.interpolate_na('DEPTH')

    return da


def DENSITYF(y, res=20):
    '''
    Computes the density function of values in y.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param y: The points sample. (numpy array)
    :param res: The density function number of points between y min and max. (int, default is 20)
    :return: The x, f(x) couple associated to density function estimation. (couple of numpy arrays)
    '''

    def AND(t1, t2):
        res = np.array([1 if (t1[i] == 1 and t2[i] == 1) else 0 for i in range(np.size(t1))])
        return res

    x = np.linspace(np.nanmin(y), np.nanmax(y), res)
    count = [np.sum([AND(y > x[i], y < x[i+1])]) for i in range(np.size(x)-1)]
    prob = np.array(count)/np.size(y)

    return x, prob


def CMP_lims(vals, out=VAR.letout):
    '''
    Returns value limits, using quantiles values to let extremes out.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param vals: The data points as array. (array)
    :param out: The percentage of value to exclude on top and at the bottom of the dataset. (float)
    :return: The value bounds. (tuple)
    '''

    vals = vals.flatten()
    vals = vals[~np.isnan(vals)]

    if vals.size == 0:

        war.warn('No values to compute limits.', stacklevel=2)

        return None, None

    else:

        return tuple(np.quantile(vals, [out/100., 1.-out/100.]))


def CLN_extremes(arr, nstd=VAR.nstd_extremes):
    '''
    Erases extreme values in the array using a threshold proportional to standard deviation.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09

    :param A: The array to be cleaned. (xarray or numpy ndarray)
    :param nstd: The tolerance as number of standard deviation. (float, default is 5.)
    :return: The cleaned xarray. (xarray)
    '''

    try:
        data = arr.values
        vminerase, vmaxerase = np.nanmean(data) - np.nanstd(data) * nstd,  np.nanmean(data) + np.nanstd(data) * nstd
        A = arr.where(arr.values > vminerase, np.nan).copy()
        A = A.where(A.values < vmaxerase, np.nan)
    except AttributeError:
        data = arr
        vminerase, vmaxerase = np.nanmean(data) - np.nanstd(data) * nstd, np.nanmean(data) + np.nanstd(data) * nstd
        A = np.where(data < vminerase, np.nan, data)
        A = np.where(data > vmaxerase, np.nan, A)

    return A


def CBAR(mappable, ax, cbarlab='', extend='both', cbardir='v'):

    if cbardir[0]=='v' or cbardir[0]=='V':
        cax = plt.gcf().add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        cbardir = 'vertical'
    else:
        cax = plt.gcf().add_axes([ax.get_position().x0, ax.get_position().y0 - 0.05, ax.get_position().width, 0.03])
        cbardir = 'horizontal'

    cbar = plt.gcf().colorbar(mappable, ax=ax, cax=cax, label=cbarlab, extend=extend, orientation=cbardir)

    return cbar


def LGD(ax, proxy, label, loc='lower right', ncol=1):
    '''
    Adds a label and its proxy (i.e. Line2D, Rectangle...etc) to the axes legend.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param ax: The matplotlib axes. (matplotlib axes)
    :param proxy: The fake shape with adequate color for your label. (matplotlib object)
    :param label: The label to add in the legend. (str)
    :param loc: The location of the legend. (str of ['upper right', 'upper left', 'lower right', lower left'], default
     is 'lower right')

    :return: The matplotlib axes.
    '''

    if type(proxy) is list:

        if len(proxy) == 1:

            if type(label) is list:

                LGD(ax, proxy[0], label[0], loc=loc, ncol=ncol)

            else:

                LGD(ax, proxy[0], label, loc=loc, ncol=ncol)

        else:

            for i in range(len(proxy)):

                LGD(ax, proxy[i], label[i], loc=loc, ncol=ncol)

    else:

        legend = ax.get_legend()

        if legend is None:

            legend = ax.legend([proxy], [label], loc=loc)

        else:

            handles, texts = legend.legendHandles, [te._text for te in legend.texts]
            handles.append(proxy)
            texts.append(label)

            legend.remove()
            ax.legend(handles, texts, loc=loc, ncol=ncol)

        ax.get_legend().get_frame().set_facecolor((0.6, 0.6, 0.7, 0.5))

    return ax


def TCK_timeax(ax):
    '''
    Description: Nicely ticks your axis with dates.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param ax: The matplotlib axes. (matplotlib axes)
    :return: The matplotlib axes.
    '''

    datemin, datemax = np.datetime64(int(ax.get_xlim()[0]), 'D').astype(str), \
                       np.datetime64(int(ax.get_xlim()[1]), 'D').astype(str)
    datemin, datemax = FMT_date(str(datemin), 'dt', verbose=False), \
                       FMT_date(str(datemax), 'dt', verbose=False)

    if datemax - datemin < dt.timedelta(25.):

        ticks = [[[dt.datetime(year, month, 1), dt.datetime(year, month, 4), dt.datetime(year, month, 8),
                   dt.datetime(year, month, 12), dt.datetime(year, month, 16), dt.datetime(year, month, 20),
                   dt.datetime(year, month, 24), dt.datetime(year, month, 28)] for year in
                  range(datemin.year, datemax.year+1)] for month in range(1, 13)]
        ticks = np.array(ticks).flatten()
        xticks_labels = [x.strftime('%b, %d') for x in ticks]
        ticks = FMT_date(ticks, 'dt64', verbose=False)
        ticks = [tick.astype(int) for tick in ticks]

    elif datemax - datemin < dt.timedelta(40.):

        ticks = [[[dt.datetime(year, month, 1), dt.datetime(year, month, 8), dt.datetime(year, month, 16),
                   dt.datetime(year, month, 24)] for year in range(datemin.year, datemax.year+1)] for month in
                 range(1, 13)]
        ticks = np.array(ticks).flatten()
        xticks_labels = [x.strftime('%b, %d') for x in ticks]
        ticks = FMT_date(ticks, 'dt64', verbose=False)
        ticks = [tick.astype(int) for tick in ticks]

    elif datemax - datemin < dt.timedelta(70.):

        ticks = [[[dt.datetime(year, month, 1), dt.datetime(year, month, 16)] for year in
                  range(datemin.year, datemax.year + 1)] for month in range(1, 13)]
        ticks = np.array(ticks).flatten()
        xticks_labels = [x.strftime('%b, %d') for x in ticks]
        ticks = FMT_date(ticks, 'dt64', verbose=False)
        ticks = [tick.astype(int) for tick in ticks]

    elif datemax - datemin < dt.timedelta(200.):

        ticks = [[dt.datetime(year, month, 1) for year in range(datemin.year, datemax.year+1)] for month in
                 range(1, 13)]
        ticks = np.array(ticks).flatten()
        xticks_labels = [x.strftime('%b') for x in ticks]
        ticks = FMT_date(ticks, 'dt64', verbose=False)
        ticks = [tick.astype(int) for tick in ticks]

    elif datemax - datemin < dt.timedelta(400.):

        ticks = [[dt.datetime(year, 1, 1), dt.datetime(year, 4, 1), dt.datetime(year, 7, 1), dt.datetime(year, 10, 1)]
                 for year in range(datemin.year, datemax.year+1)]
        ticks = np.array(ticks).flatten()
        xticks_labels = [x.strftime('%b') for x in ticks]
        ticks = FMT_date(ticks, 'dt64', verbose=False)
        ticks = [tick.astype(int) for tick in ticks]

    elif datemax - datemin < dt.timedelta(800.):

        ticks = [[dt.datetime(year, 1, 1), dt.datetime(year, 7, 1)] for year in range(datemin.year, datemax.year+1)]
        ticks = np.array(ticks).flatten()
        xticks_labels = ['{}'.format(x.strftime('%b %Y')) for x in ticks]
        ticks = FMT_date(ticks, 'dt64', verbose=False)
        ticks = [tick.astype(int) for tick in ticks]

    else:

        ticks = [dt.datetime(year, 1, 1) for year in range(datemin.year, datemax.year+1)]
        xticks_labels = [x.strftime('%Y') for x in ticks]
        ticks = FMT_date(ticks, 'dt64', verbose=False)
        ticks = [tick.astype(int) for tick in ticks]

    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.set_xticklabels(xticks_labels)

    return ax


def SGM(FLUO, PAR, r=0.092, PARmid=261., e=2.2):
    '''
    The sigmoid model used in Xing et al., 2018 to correct Chlorophyll from non-photochemical quenching.
    :param FLUO: The fluorescence values. (numpy array)
    :param PAR: The PAR array. (numpy array)
    :param r: The r parameter value. (float, default is 0.092)
    :param PARmid: The PARmid parameter value. (float, default is 261.)
    :param e: The e parameter value. (float, default is 2.2)
    :return: The corrected fluorescence profile. (numpy array)
    '''

    CHL_C = FLUO / (r + (1 - r) / (1 + (PAR / PARmid) ** e))

    return CHL_C