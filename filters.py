# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the module dedicated to profiles vertical filtering. Contains median, mean, despiking, lowpass and
force-decrease filters.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''

# Imports


from core import *


# Filters


def FLT_med(var_array, coords, smooth=VAR.Zsmooth, interp=VAR.Zsmoothinterp, verbose=True):
    '''
    Median filters a 1D or 2D array.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param var_array: The array. (numpy 1D or 2D array)
    :param coords: The coordinates associated to the array. (numpy 1D or 2D array)
    :param smooth: The smoothing scale. (float, default is Zsmooth in globavars.py)
    :param interp: The interpolation resolution. (float, default is Zsmoothinterp in globavars.py)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The smoothed array. (numpy 1D or 2D array)'''

    war.filterwarnings('ignore', category=RuntimeWarning)

    ti = t.time()

    if np.size(np.shape(var_array)) == 1:

        arr = var_array.copy()[np.newaxis]
        coo = coords.copy()[np.newaxis]

    elif np.size(np.shape(var_array)) == 2 :

        arr = var_array.copy()
        coo = coords.copy()

    else:

        raise UserWarning('Dimension has to be 1 or 2 but is actually {}. '
                          'Returning None.'.format(np.size(np.shape(var_array))))

    if type(coords.flatten()[0]) == np.datetime64:

        coo = (coords-coords[0]).astype(float)/1e9/3600/24

    N = int(smooth // interp) + 1
    nodes = np.arange(0., np.nanmax(coo), interp)
    arr = INT_1D(arr, coo, nodes)
    extra = np.nan * np.zeros((np.shape(arr)[0], N//2))
    arr = np.hstack([extra, arr, extra])

    res = np.hstack([np.nanmedian(arr[:, i-N//2:i+N//2], axis=1)[np.newaxis].T
                     for i in range(N//2, np.shape(arr)[1]-N//2)])
    res = INT_1D(res, nodes, coo)

    TINFO(ti, res.shape[0]/60, 'Median filtered your array', verbose)

    return res.squeeze()


def FLT_mean(var_array, coords, smooth=VAR.Zsmooth, interp=VAR.Zsmoothinterp, verbose=True):
    '''
    Mean filters a 1D or 2D array.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param var_array: The array. (numpy 1D or 2D array)
    :param coords: The coordinates associated to the array. (numpy 1D or 2D array)
    :param smooth: The smoothing scale. (float, default is Zsmooth in globavars.py)
    :param interp: The interpolation resolution. (float, default is Zsmoothinterp in globavars.py)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The smoothed array. (numpy 1D or 2D array)'''

    war.filterwarnings('ignore', category=RuntimeWarning)

    ti = t.time()

    if not type(var_array) is np.ndarray:
        var_array = np.array(var_array)

    if np.size(np.shape(var_array)) == 1:

        arr = var_array.copy()[np.newaxis]
        coo = coords.copy()[np.newaxis]

    elif np.size(np.shape(var_array)) == 2 :

        arr = var_array.copy()
        coo = coords.copy()

    else:

        raise UserWarning('Dimension has to be 1 or 2 but is actually {}. '
                          'Returning None.'.format(np.size(np.shape(var_array))))

    if type(coords.flatten()[0]) == np.datetime64:

        coo = (coords-coords[0]).astype(float)/1e9/3600/24

    N = int(smooth // interp) + 1
    nodes = np.arange(0., np.nanmax(coo), interp)
    arr = INT_1D(arr, coo, nodes)
    extra = np.nan * np.zeros((np.shape(arr)[0], N//2))
    arr = np.hstack([extra, arr, extra])

    res = np.hstack([np.nanmean(arr[:, i-N//2:i+N//2], axis=1)[np.newaxis].T
                     for i in range(N//2, np.shape(arr)[1]-N//2)])
    res = INT_1D(res, nodes, coo)

    TINFO(ti, res.shape[0]/500, 'Mean filtered your array', verbose)

    return res.squeeze()


def FLT_despike(var_array, coords, thresh_up=95., thresh_down=None, smooth=VAR.Zsmooth, verbose=True):
    '''
    Filters 2D arrays erasing spikes on vertical profiles (along first dimension). Detects spikes by substracting the
    mean filtered profiles.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param var_array: The profiles array (profiles are along first dimension). (numpy array)
    :param coords: The coordinates array. (numpy array)
    :param thresh_up: Positive spikes detection threshold: erases if x - x_meanfiltered overcomes its own thresh_up
     percentile. (float between 0 and 100, default is 95)
    :param thresh_down: Same as thresh_up but for negative spikes. (float between 0 and 100, default is 100-thresh_up)
    :param smooth: The mean-filtered smoothing z characteristic value. (float, default is VAR.Zsmooth)
    :param verbose: Wether to print info to the console. (bool, default is True)
    :return: The despiked numpy array, with same coordinates as entry. (numpy array)
    '''

    ti = t.time()

    if thresh_down is None:
        thresh_down = 100. - thresh_up

    despiked = np.nan * np.zeros(np.shape(var_array))

    for i in range(np.shape(var_array)[0]):

        if verbose:
            CPRINT('Despiking: {:.02f}%'.format(i/(np.shape(var_array)[0]-1)*100) + LOADOTS(i), attrs='CYAN', end='\r')

        x, y = var_array[i], coords[i]
        mask = ~np.isnan(x)
        x, y = x[mask], y[mask]

        if np.size(x) > 10.:

            xs = FLT_mean(x, y, smooth=smooth, verbose=False)
            spikes = x - xs

            threshold_up = np.nanpercentile(spikes, thresh_up)
            threshold_down = np.nanpercentile(spikes, thresh_down)

            xds, yds = x[spikes < threshold_up], y[spikes < threshold_up]
            xds, yds = xds[spikes[spikes<threshold_up]>threshold_down], yds[spikes[spikes<threshold_up]>threshold_down]

            xdss = FLT_mean(xds, yds, smooth=smooth, verbose=False)
            xds = np.where(spikes < threshold_down, np.interp(y, yds, xdss), x)
            xds = np.where(spikes > threshold_up, np.interp(y, yds, xdss), xds)
            despiked[i][mask] = xds

    TINFO(ti, np.shape(var_array)[0]/2., 'Despiked signal', verbose)

    return despiked


def FLT_lowpass(var_array, coords, smooth=VAR.Zsmooth, interp=VAR.Zsmoothinterp, verbose=True):
    '''
    Low pass filters a 1D or 2D array.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param var_array: The array. (numpy 1D or 2D array)
    :param coords: The coordinates associated to the array. (numpy 1D or 2D array)
    :param smooth: The smoothing scale. (float, default is Zsmooth in globavars.py)
    :param interp: The interpolation resolution. (float, default is Zsmoothinterp in globavars.py)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The smoothed array. (numpy 1D or 2D array)
    '''

    def aux(arr1d, pres, d, interp):

        if np.sum(~np.isnan(arr1d)) > 20:

            nodes = np.arange(np.nanmin(pres[~np.isnan(arr1d)]), np.nanmax(pres[~np.isnan(arr1d)]), interp)
            x, y = pres[~np.isnan(arr1d)], arr1d[~np.isnan(arr1d)]
            regular = np.interp(nodes, x, y, left=np.nan, right=np.nan)
            N = np.size(regular)

            def butter_lowpass(cutoff, fs, order=2):
                return sig.butter(order, cutoff, fs=fs, btype='low', analog=False)

            def butter_lowpass_filter(data, cutoff, fs, order=2):
                coefs = butter_lowpass(cutoff, fs, order=order)
                y = sig.lfilter(coefs[0], coefs[1], data)
                return y

            freq = 1 / interp
            cutoff = 1 / d

            filtered = butter_lowpass_filter(regular, cutoff, freq)
            lstsq = np.sum((filtered - regular) ** 2) / N
            next_lstsq = np.sum((filtered[1:] - regular[:-1]) ** 2) / (N - 1)
            diffs = [lstsq, next_lstsq]
            for i in range(2, int(d / interp) + 1):
                lstsq = next_lstsq
                next_lstsq = np.sum((filtered[i:] - regular[:-i]) ** 2) / (N - i)
                diffs.append(next_lstsq)

            best = np.max([1, np.argmin(diffs)])

            res = np.interp(pres, nodes[:-best], filtered[best:], left=np.nan, right=np.nan)
            res = np.where(np.array([np.isnan(arr1d[i:]).all() for i in range(np.size(arr1d))]), np.nan, res)


        else:

            res = np.nan * np.zeros(np.size(arr1d))

        return res

    ti = t.time()

    if np.size(np.shape(var_array)) == 1:

        res = aux(var_array, coords, smooth, interp)
        res = np.where(coords < np.nanmin(coords) + smooth, np.nan, res)
        res = np.where(coords > np.nanmax(coords) - smooth, np.nan, res)

    elif np.size(np.shape(var_array)) == 2 :

        res = var_array.copy()

        for i in range(np.shape(res)[0]):

            res[i] = aux(var_array[i], coords[i], smooth, interp)
            res[i] = np.where(coords[i] < np.nanmin(coords[i]) + smooth, np.nan, res[i])
            res[i] = np.where(coords[i] > np.nanmax(coords[i]) - smooth, np.nan, res[i])

    else:

        raise UserWarning('Dimension has to be 1 or 2 but is actually {}. '
                          'Returning None.'.format(np.size(np.shape(var_array))))

    TINFO(ti, res.shape[0]/500, 'Lowpass filtered your profiles', verbose)

    return res


def FLT_forcedecrease(var_array, verbose=True):
    '''
    Removes increasing values of array.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param var_array: The array. (numpy 1D or 2D array)
    :return: The decreasing array. (numpy 1D or 2D array)
    '''

    def aux(arr1d):

        res = np.nan*np.zeros(np.size(arr1d))
        y = arr1d.copy()[~np.isnan(arr1d)]
        y = np.array([y[k] if y[k] == np.max(y[k:]) else np.nan for k in range(np.size(y))])
        res[~np.isnan(arr1d)] = y

        return res

    ti = t.time()

    if np.size(np.shape(var_array)) == 1:

        res = aux(var_array)

    elif np.size(np.shape(var_array)) == 2 :

        res = var_array.copy()

        for i in range(np.shape(res)[0]):

            res[i] = aux(res[i])

    else:

        res = None
        raise UserWarning('Dimension has to be 1 or 2 but is actually {}. '
                          'Returning None.'.format(np.size(np.shape(var_array))))

    TINFO(ti, 1e-3 * res.shape[0], 'Array is now decreasing', verbose)

    return res