# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the maps plot dedicated module.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''
import matplotlib.pyplot as plt

# Imports


from profiles import *


# Functions


def CHART(ax=None, zone_id=list(VAR.zones)[0], quality=None, grid=True, grsp=None, scale=False, scaleloc='right',
          bluemarble=False, figsize=VAR.rcParams['figure.figsize'], verbose=True):
    '''
    Description: Plots the backgroud map on a given ax and returns the axes. Doesn't automatically show without
    the use of another function as profloc or mapvariable. Use  if you just want the map.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param ax: The axes of the map plot. If None, creates new figure and gets its axes. (matplotlib axes, optional)
    :param zone_id: The zone of interest. (str in ['WO', 'PA', 'EP', 'WP', 'MA', 'FJ', 'EA'], default to 'SP')
    :param quality: The definition of the coastline. (str in ['10m', '50m', '110m'], optional)
    :param grid: Whether to print the meridians and parallels. (boolean, default to True)
    :param grsp: The grid spacing in degrees. It is identical for longitude and latitude. (float, optional)
    :param scale: Whether to print the map scale. (boolean, default to True)
    :param scaleloc: The location of the scale on the figure. (str of ['left', 'right'], default is 'left')
    :param bluemarble: chooses NASA's bluemarble image as background. Due to its resolution, only works nicely at global
     scale. (boolean, default to False)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes. (matplotlib axes)
    '''

    get_exponent = lambda x: np.floor(np.log10(x)).astype(int)

    ti = t.time()

    # Get chart limits from zone list

    def get_limits(Z):

        lonmin = Z[0] - Z[2] * 1000 * 360 / (VAR.Rt * 2 * np.pi * np.cos(Z[1] * np.pi / 180)) / 2
        lonmax = Z[0] + Z[2] * 1000 * 360 / (VAR.Rt * 2 * np.pi * np.cos(Z[1] * np.pi / 180)) / 2
        latmin = Z[1] - Z[3] * 1000 * 360 / (VAR.Rt * 2 * np.pi) / 2
        latmax = Z[1] + Z[3] * 1000 * 360 / (VAR.Rt * 2 * np.pi) / 2

        return lonmin, lonmax, latmin, latmax

    try:
        zone = VAR.zones[zone_id]
    except KeyError:
        CPRINT('Don\'t know the zone ID \'{}\'. Has to be one of the keys of zones dictionary, in globvars.'
               ' Replacing by \'SP\'', attrs='YELLOW')
        zone = VAR.zones['SP']

    limits = get_limits(zone)

    # Find adequate quality

    if quality is None:

        if zone[2] <= 1000.:
            quality = '10m'
        elif zone[2] <= 7000.:
            quality = '50m'
        else:
            quality = '110m'

    # Basemap draw

    if ax is None:

        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    else:

        fig = plt.gcf()

    try:

        ax.projection

    except AttributeError:

        CPRINT('Your axes do not have a geographic projection. Creating new axes.', attrs='YELLOW')

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    if bluemarble is False:

        ax.set_facecolor(VAR.oceancolor)

        land = cfeature.NaturalEarthFeature('physical', 'land', quality, edgecolor=(0., 0., 0.),
                                            facecolor=VAR.landcolor)
        ax.add_feature(land, zorder=3)

    else:

        img_extent = (-180, 180, -90, 90)
        img = plt.imread(VAR.indexpath + 'bluemarble.jpg')
        ax.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), zorder=0)

    # Parallels and meridians

    if grid:

        if grsp is None:

            def get_grsp(distkm, lat):

                lon = distkm * 1000 * 180 / (VAR.Rt * np.pi * np.cos(lat * np.pi / 180))
                serie = [0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 40.]

                i = 0
                while serie[i] < lon / 8. and i < len(serie) - 1:
                    i += 1

                return serie[i]

            grsp = get_grsp(zone[2], zone[1])

        meridians = np.array([k * grsp for k in range(-int(180. / grsp), int(180. / grsp) + 1)])
        parallels = np.array([k * grsp for k in range(-int(90. / grsp), int(90. / grsp) + 1)])

        roundto = -np.min([get_exponent(meridians[1] - meridians[0]), get_exponent(parallels[1] - parallels[0])]) - 1

        if bluemarble:
            ctemp = 'white'
        else:
            ctemp = 'black'
        ax.xaxis.tick_top()
        gl = ax.gridlines(xlocs=meridians, ylocs=parallels, zorder=4, linewidth=1., color=ctemp, alpha=0.5,
                          linestyle='dotted')
        roundto = np.max(
            [-np.min([get_exponent(meridians[1] - meridians[0]), get_exponent(parallels[1] - parallels[0])]) - 1, 0])

        ax.set_yticks(parallels)
        ax.set_xticks(meridians)

        if roundto == 0:

            xtickslab = (180 - meridians).astype(int)
            ytickslab = parallels.astype(int)

            xtickslab = ['${} ^{{\\circ}} W$'.format(e) if 0. < e < 180. else
                         '${} ^{{\\circ}} E$'.format(360 - e) if 180. < e < 360. else
                         '${} ^{{\\circ}}$'.format(e) for e in xtickslab]
            ytickslab = ['${} ^{{\\circ}} N$'.format(e) if 0. < e < 90. else
                         '${} ^{{\\circ}} S$'.format(-e) if -90. < e < 0. else
                         '${} ^{{\\circ}}$'.format(e) for e in ytickslab]
        else:

            xtickslab = (180 - meridians).astype(float)
            ytickslab = parallels.astype(float)

            xtickslab = ['${} ^{{\\circ}} W$'.format(np.round(e, roundto)) if 0. < e < 180. else
                         '${} ^{{\\circ}} E$'.format(np.round(360. - e, roundto)) if 180. < e < 360. else
                         '${} ^{{\\circ}}$'.format(np.round(e, roundto)) for e in xtickslab]
            ytickslab = ['${} ^{{\\circ}} N$'.format(np.round(e, roundto)) if 0. < e < 90. else
                         '${} ^{{\\circ}} S$'.format(-np.round(e, roundto)) if -90. < e < 0. else
                         '${} ^{{\\circ}}$'.format(np.round(e, roundto)) for e in ytickslab]

        ax.set_xticklabels(xtickslab)
        ax.set_yticklabels(ytickslab)

    ax.set_extent(limits, crs=ccrs.PlateCarree())

    # Scale

    if scale:

        width = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).width
        scalefontsize = width * fig.dpi / 150

        if zone[2] < 600.:
            scalesize = int((zone[2] / 6) / 10) * 10
        elif zone[2] < 6000.:
            scalesize = int((zone[2] / 6) / 100) * 100
        else:
            scalesize = int((zone[2] / 6) / 1000) * 1000

        width, height = ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0]
        rwidth, rheight = scalesize * width / zone[2], height / 150

        if scaleloc == 'left':

            xscale = ax.get_xlim()[0] + width / 20
            yheight = ax.get_ylim()[0] + height / 20
            coef = 1.8
            diff = rwidth / coef - rwidth / 2

            ax.add_patch(
                Rectangle((xscale, yheight - rheight), rwidth + 2 * diff, rheight * 10, color='white', zorder=3))
            ax.add_patch(
                Rectangle((xscale - rwidth / 2 + rwidth / coef, yheight), rwidth, rheight, color='black', zorder=3))
            ax.text(xscale + rwidth / coef, yheight + 3 * rheight, '{} km'.format(scalesize), ha='center', zorder=3,
                    fontsize=scalefontsize)

        else:

            xscale = ax.get_xlim()[1] - width / 20 - rwidth
            yheight = ax.get_ylim()[0] + height / 20
            coef = 1.8
            diff = rwidth / coef - rwidth / 2

            ax.add_patch(
                Rectangle((xscale, yheight - rheight), rwidth + 2 * diff, rheight * 10, color='white', zorder=3))
            ax.add_patch(
                Rectangle((xscale - rwidth / 2 + rwidth / coef, yheight), rwidth, rheight, color='black', zorder=3))
            ax.text(xscale + rwidth / coef, yheight + 3 * rheight, '{} km'.format(scalesize), ha='center', zorder=3,
                    fontsize=scalefontsize)

    TINFO(ti, 1., 'Charted your region', verbose)

    return ax


def PT_locs(ax=None, zone_id=list(VAR.zones)[0], c=VAR.placescolor):
    '''
    Displays your location names (see VAR.places) on a map.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param ax: The axes of the map plot. If None, creates new map. (matplotlib axes, optional)
    :param zone_id: The zone of interest. (str in VAR.zones, default is the first)
    :param c: The text color. (matplotlib color, optional)
    :return: The matplotlib axes. (matplotlib axes)
    '''

    if ax is None:
        ax = CHART(zone_id=zone_id)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xsp = xlim[1]-xlim[0]
    ysp = ylim[1]-ylim[0]

    for p in VAR.places:
        ax.text(VAR.places[p][0], VAR.places[p][1], r'\begin{center}' + p + r'\end{center}', zorder=2, fontsize=11,
                transform=ccrs.PlateCarree(), c=c)

    return ax


def PT_floats(wmo_list=None, datemin=None, datemax=None, ax=None, c=None, display_names=True, zone_id='SP',
              scaleloc='right', bluemarble=False, labelc='k', verbose=True):
    '''
    If axes not given, plots a map and shows locations of floats that exist in index file (see
    build_index() func). Else, just points float locations assuming your axes have an adequate projection.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param wmo_list: The floats WMOs. If None, takes all the floats that have PAR sensors. (list of int)
    :param datemin: The date at which to start trace display. (datetime or str 'YYYY-MM-DD', optional)
    :param datemax: The date at which to display position. (datetime or str 'YYYY-MM-DD', optional)
    :param ax: The axes of the map plot. If None, creates new map. (matplotlib axes, optional)
    :param c: The color of the shown floats. (matplotlib color, default is None)
    :param display_names: Whether to print WMOs on the map. (boolean, default is True)
    :param zone_id: The zone of interest. (str in ['WO', 'PA', 'EP', 'WP', 'MA', 'FJ', 'EA'], default to 'SP')
    :param scaleloc: The location of the scale on the figure. (str of ['left', 'right'], default is 'right')
    :param bluemarble: chooses NASA's bluemarble image as background. Due to its resolution, only works nicely at global
     scale. (boolean, default to False).
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes. (matplotlib axes)
    '''

    ti = t.time()

    def get_limits(Z):

        lonmin = Z[0] - Z[2] * 1000 * 360 / (VAR.Rt * 2 * np.pi * np.cos(Z[1] * np.pi / 180)) / 2
        lonmax = Z[0] + Z[2] * 1000 * 360 / (VAR.Rt * 2 * np.pi * np.cos(Z[1] * np.pi / 180)) / 2
        latmin = Z[1] - Z[3] * 1000 * 360 / (VAR.Rt * 2 * np.pi) / 2
        latmax = Z[1] + Z[3] * 1000 * 360 / (VAR.Rt * 2 * np.pi) / 2

        return lonmin, lonmax, latmin, latmax

    lims = get_limits(VAR.zones[zone_id])

    if datemin is None:
        datemin = dt.datetime(2000, 1, 1)
    else:
        datemin = FMT_date(datemin, 'dt', verbose=False)

    if datemax is None:
        datemax = dt.datetime.today()
    else:
        datemax = FMT_date(datemax, 'dt', verbose=False)

    if wmo_list is None:
        wmo_list = WMOS()
    elif not type(wmo_list) is list:
        wmo_list = [wmo_list]

    if ax is None:
        ax = CHART(zone_id=zone_id, scaleloc=scaleloc, bluemarble=bluemarble)
        ax.set_title('Last known position of floats on {}'.format(datemax.strftime('%b %d, %Y')))

    if c is None:
        floatcolors = True
    else:
        floatcolors = False

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xsp = xlim[1] - xlim[0]
    ysp = ylim[1] - ylim[0]

    for wmo in wmo_list:

        prof = PRC(wmo)
        lon, lat, tim = FMT_lon(prof.LONGITUDE.values), prof.LATITUDE.values, \
            FMT_date(prof.JULD.values, 'dt', verbose=False)

        if floatcolors:
            c = colors.to_rgba(VAR.floats_colors[FMT_wmo(wmo)])

        if np.size(lon) > 0:

            closest_date_ind = np.argmin(np.abs(tim - datemax))

            if verbose:
                if abs(tim[closest_date_ind] - datemax) > dt.timedelta(20.):
                    CPRINT('There is {} days between {} and closest location of float {}.'
                           .format(abs(tim[closest_date_ind] - datemax).days, datemax.strftime('%d %B, %Y'), wmo),
                           attrs='YELLOW')

            lon_end, lat_end = lon[closest_date_ind], lat[closest_date_ind]

            ax.plot(lon, lat, c=(1., 1., 1., 0.8), linewidth=2.5, zorder=2, transform=ccrs.PlateCarree())
            ax.plot(lon, lat, c=list(c)[:-1] + [0.8], linewidth=1.5, zorder=2, transform=ccrs.PlateCarree())
            ax.scatter(lon_end, lat_end, c='k', s=60, marker='o', zorder=3, transform=ccrs.PlateCarree())
            ax.scatter(lon_end, lat_end, c='white', s=30, marker='$*$', zorder=3, transform=ccrs.PlateCarree())

            if display_names:

                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                xsp = xlim[1] - xlim[0]
                ysp = ylim[1] - ylim[0]

                if lims[2] <= lat_end <= lims[3]:
                    ax.text(lon_end + xsp / 80, lat_end, '\\textbf{{{}}}'.format(VAR.floats_names[FMT_wmo(wmo)]),
                            zorder=3, fontsize=12, c=labelc, transform=ccrs.PlateCarree())

    TINFO(ti, 10., 'Pointed drifters', verbose)

    return ax


def MAP_contour(lon, lat, matrix, vmin=None, vmax=None, levels=6, lw=1.5, log=False, smooth=None, subsampling=True,
                roundto=2, ls=None, color='k', fs=8., inlinespacing=20., clab=False, clegend=None, title='',
                ax=None, zone_id='SP', quality=None, grid=True, grsp=None, scale=True, verbose=True):
    '''
    Plots the given scalar field contours. Parameters lon, lat and matrix should have same dimensions.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param lon: The longitude 1D-array. (numpy 1D array)
    :param lat: The latitude 1D-array. (numpy 1D array)
    :param matrix: The values array. (numpy 2D array)
    :param vmin: Minimum displayed value. (float, optional)
    :param vmax: Maximum displayed value. (float, optional)
    :param levels: The number of contour lines, or levels list. (int, list or np 1D array, default is 6)
    :param lw: The contours linewidth. (float, default is 1.5)
    :param log:
    :param roundto:
    :param ls:
    :param color:
    :param fs:
    :param inlinespacing:
    :param clab:
    :param clegend:
    :param title:
    :param ax:
    :param zone_id:
    :param quality:
    :param grid:
    :param grsp:
    :param scale:
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return:
    '''

    from matplotlib import ticker

    ti = t.time()

    # Creating axes if None are given

    if ax is None:
        ax = CHART(ax=None, zone_id=zone_id, quality=quality, grid=grid, grsp=grsp, scale=scale, bluemarble=False)
    ax.set_title(title)

    # Levels min and max

    if vmin is None or vmax is None and not log:
        if vmin is None:
            vmin = CMP_lims(matrix, out=2.)[0]
        if vmax is None:
            vmax = CMP_lims(matrix, out=2.)[1]

    # Reformat the handle 180th longitude degree

    i = 0
    for k in range(np.size(lon) - 1):
        if lon[k] * lon[k + 1] < 0:
            break
        else:
            i += 1
    i += 1
    lon = np.hstack([lon[i:], lon[:i] + 360.])
    matrix = np.hstack([matrix[:, i:], matrix[:, :i]])

    # Subsampling

    if subsampling:
        sslon, sslat = int(np.max([np.size(lon) / 300, 1])), int(np.max([np.size(lat) / 300, 1]))
        lon, lat, matrix = lon[::sslon], lat[::sslat], matrix[::sslat, ::sslon]

    # Smoothing

    if smooth is not None:

        war.filterwarnings('ignore', category=RuntimeWarning)
        newmat = np.nan * np.zeros(np.shape(matrix))
        for i in range(smooth, np.shape(matrix)[0] - smooth):

            CPRINT('Smoothing' + LOADOTS(i), attrs='BLUE', end='\r')

            for j in range(smooth, np.shape(matrix)[1] - smooth):
                newmat[i, j] = np.nanmean(matrix[i - smooth:i + smooth, j - smooth:j + smooth])

        matrix = newmat

    # Plotting contours

    levels = None if log else np.linspace(vmin, vmax, int(levels))
    cs = ax.contour(lon, lat, matrix, transform=ccrs.PlateCarree(), levels=levels, vmin=vmin, vmax=vmax, zorder=2,
                    linewidths=lw, colors=color, linestyles=ls, locator=ticker.LogLocator() if log else None)

    if clegend is not None:

        from matplotlib.lines import Line2D
        line = Line2D([0], [0], color=color, lw=lw, linestyle=ls)
        LGD(ax, [line], [clegend], loc='upper left')

    if clab:
        ax.clabel(cs, fmt='%1.{}f'.format(roundto), fontsize=fs, inline_spacing=inlinespacing)

    TINFO(ti, 10., 'Scalar field contour plot', verbose)

    return ax


def MAP_grid(lon, lat, matrix, vmin=None, vmax=None, cmap='jet', center=False, log=False, extend='both',
             ax=None, zone_id='SP', quality=None, grid=True, grsp=None, scale=True, subsampling=True, smooth=None,
             isolevels=6, isolines=None, clab=False, clegend=None, logisolines=False, cbarlab='', cbardir='vertical',
             title='', verbose=True):
    '''
    Plots the given scalar field using pcolormesh.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param lon: The longitude 1D-array. (numpy 1D array)
    :param lat: The latitude 1D-array. (numpy 1D array)
    :param matrix: The values array. (numpy 2D array)
    :param vmin: Minimum displayed value. (float, optional)
    :param vmax: Maximum displayed value. (float, optional)
    :param cmap: The colormap. (matplotlib cmap, default is jet)
    :param center: Wether to center the colorbar on 0. (bool, default is False)
    :param log: Wether to print values in log scale. (bool, default is False)
    :param extend: Which way to extend the colorbar. ('min', 'max' or 'both', default is both)
    :param ax: The axes on which to display. (matplotlib axes, optional)
    :param zone_id: The zone of interest. (str in ['WO', 'PA', 'EP', 'WP', 'MA', 'FJ', 'EA'], default to 'SP')
    :param quality: The definition of the coastline. (str in ['10m', '50m', '110m'], optional)
    :param grid: Whether to print the meridians and parallels. (boolean, default to True)
    :param grsp: The grid spacing in degrees. It is identical for longitude and latitude. (float, optional)
    :param scale: Whether to print the map scale. (boolean, default to True)
    :param subsampling: Wether to subsample input data to reasonable resolution. (bool, default is True)
    :param smooth: The number of values taken around a gridpoint to smooth the input values. (int, optional)
    :param isolines: The isolines array. (numpy 1D array, optional)
    :param isolevels: The number of isoline levels. (int, default is 6)
    :param clab: Wether to label the isolines. (bool, default is False)
    :param clegend: The name of the lines to display in the legend. (str, optional)
    :param logisolines: Wether to print isolines values in log scale. (bool, default is False)
    :param cbarlab: The colorbar label. (str, optional)
    :param cbardir: The colorbar direction. ('h' or 'v', default is vertical)
    :param title: The axes title. (str, optional)
    :param verbose: Whether to display information to the console. (boolean, default is True)
    :return: The matplotlib axes.
    '''

    ti = t.time()

    # Creating axes

    if ax is None:
        ax = CHART(ax=None, zone_id=zone_id, quality=quality, grid=grid, grsp=grsp, scale=scale, bluemarble=False)
    ax.set_title(title)

    # Colors min and max

    if vmin is None or vmax is None:
        if vmin is None:
            vmin = CMP_lims(matrix, out=2.)[0]
        if vmax is None:
            vmax = CMP_lims(matrix, out=2.)[1]

    if log and center:
        CPRINT('Cannot print log and centered values.', attrs='YELLOW')
        return ax

    if log and vmin <= 0.:
        CPRINT('Cannot print log negative or null values.', attrs='YELLOW')
        return ax

    if center:
        max = np.max([np.abs(vmin), np.abs(vmax)])
        vmin, vmax = -max, max

    # Reformat the handle 180th longitude degree

    i = 0
    for k in range(np.size(lon) - 1):
        if lon[k] * lon[k + 1] < 0:
            break
        else:
            i += 1
    i += 1
    lonf = np.hstack([lon[i:], lon[:i] + 360.])
    matrix = np.hstack([matrix[:, i:], matrix[:, :i]])

    # Subsampling

    if subsampling:

        sslon, sslat = int(np.max([np.size(lon) / 300, 1])), int(np.max([np.size(lat) / 300, 1]))
        lon, lonf, lat, matrix = lon[::sslon], lonf[::sslon], lat[::sslat], matrix[::sslat, ::sslon]
        if isolines is not None:
            isolines = isolines[::sslat, ::sslon]

    # Smoothing

    if smooth is not None:

        war.filterwarnings('ignore', category=RuntimeWarning)
        newmat = np.nan * np.zeros(np.shape(matrix))
        for i in range(smooth, np.shape(matrix)[0] - smooth):

            CPRINT('Smoothing' + LOADOTS(i), attrs='BLUE', end='\r')

            for j in range(smooth, np.shape(matrix)[1] - smooth):
                newmat[i, j] = np.nanmean(matrix[i - smooth:i + smooth, j - smooth:j + smooth])

        matrix = newmat

    # Plotting gridpoints

    mappable = ax.pcolormesh(lonf, lat, matrix, cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,
                             norm=colors.LogNorm() if log else None)

    # Colorbar

    cbar = CBAR(mappable, ax=ax, cbarlab=cbarlab, extend=extend, cbardir=cbardir)

    # Isolines

    if isolines is not None:
        MAP_contour(lon, lat, isolines, levels=isolevels, color='k', clegend=clegend, clab=clab, ax=ax, smooth=smooth,
                    log=logisolines, verbose=False)

    TINFO(ti, 10., 'Scalar field grid plot', verbose)

    return ax


def MAP_timetraj(wmo, zone_id='SP', jmin=None, jmax=None, cmap='jet', ax=None, c='k', legloc='upper right',
                 verbose=True):

    war.filterwarnings('ignore', category=UserWarning)
    ti = t.time()
    reftime = dt.datetime(2000, 1, 1)

    if type(wmo) not in [list, np.ndarray]:

        if ax is None:
            ax = CHART(zone_id=zone_id, scale=True, verbose=False)

        prof = PRC(wmo)
        wmo = GET_wmo(prof)
        timz = FMT_date(prof.JULD.values, 'dt', verbose=False) - reftime

        if jmin is None:
            jmin = np.nanmin(timz)
        elif type(jmin) is dt.datetime:
            jmin = FMT_date(jmin, 'dt', verbose=False)
        if jmax is None:
            jmax = np.nanmax(timz)
        elif type(jmax) is dt.datetime:
            jmax = FMT_date(jmax, 'dt', verbose=False)

        lon, lat = FMT_lon(prof.LONGITUDE.values), prof.LATITUDE.values

        ax.plot(lon, lat, c='k', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=2)
        ax.plot(lon, lat, c=c, linewidth=0.8, transform=ccrs.PlateCarree(), zorder=2)

        scatter = ax.scatter(lon, lat, c=[te.days for te in timz], vmin=jmin.days, vmax=jmax.days, cmap=cmap, s=15,
                             zorder=2, transform=ccrs.PlateCarree())
        cbar = CBAR(scatter, ax=ax, cbardir='v')
        cbar.ax.set_yticklabels(['${:02d}/{:02d}/{}$'.format(te.day, te.month, str(te.year)[-2:]) for te in
                                 [reftime + dt.timedelta(tee) for tee in cbar.ax.get_yticks()]])

    else:

        if ax is None:
            ax = CHART(zone_id=zone_id, scale=True, verbose=False)

        proxies, labels = [], []
        jmin_glob, jmax_glob = np.nan, np.nan

        for w in wmo:

            jmin_float = np.nanmin(FMT_date(PRC(w).JULD.values, 'dt', verbose=False) - reftime)
            jmax_float = np.nanmax(FMT_date(PRC(w).JULD.values, 'dt', verbose=False) - reftime)
            if type(jmin_glob) is float or jmin_float < jmin_glob:
                jmin_glob = jmin_float
            if type(jmax_glob) is float or jmax_float > jmax_glob:
                jmax_glob = jmax_float

        for i, w in enumerate(wmo):

            prof = PRC(w)
            w = GET_wmo(prof)
            lont, latt = prof.LONGITUDE.values[-1], prof.LATITUDE.values[-1]
            lon0, lat0 = prof.LONGITUDE.values[0], prof.LATITUDE.values[0]

            if i == 0:
                MAP_timetraj(w, zone_id=zone_id, jmin=jmin_glob, jmax=jmax_glob, ax=ax, c=VAR.floats_colors[w],
                             verbose=False)
            else:
                MAP_timetraj(w, zone_id=zone_id, jmin=jmin_glob, jmax=jmax_glob, ax=ax, c=VAR.floats_colors[w],
                             verbose=False)

            ax.scatter(lon0, lat0, color='k', s=110, zorder=3, transform=ccrs.PlateCarree())
            s = ax.scatter(lon0, lat0, color=VAR.floats_colors[w], s=35, zorder=3,
                           marker='d', transform=ccrs.PlateCarree(), label=w)
            ax.scatter(lont, latt, color='k', s=110, zorder=3, transform=ccrs.PlateCarree())
            p = ax.scatter(lont, latt, color=VAR.floats_colors[w], s=75, zorder=3,
                           marker='*', transform=ccrs.PlateCarree(), label=w)

            labels.append('{} start'.format(VAR.floats_names[w]))
            proxies.append(s)
            labels.append('{} last profile'.format(VAR.floats_names[w]))
            proxies.append(p)

        LGD(ax, proxies, labels, loc=legloc, ncol=2)

        namelist = ''
        for w in wmo:
            namelist += VAR.floats_names[FMT_wmo(w)] + ', '
        namelist = namelist[:-2]

        ax.set_title('Trajectory of floats {} and average MODIS chlorophyll-A (2015-2020)'.format(namelist))

    TINFO(ti, 4., 'Plotted floats trajectories', verbose)

    return ax

