# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''Bunch of lines that worked in the past.'''

import globvars as VAR
from profiles import *
from downloads import *
from plotargo import *
from fun import *

if __name__ == '__main__':

    time = t.time()

    print('Starting main...', end='\r')

    # _, ax = plt.subplots()
    # prof = openprof('Files/Sprof/{}_Sprof.nc'.format(wmo), filter=filter3)
    # ax.set_title('Profiles of adjusted CHLA')
    # viewprof(prof, 'CHLA_ADJUSTED', ax=ax, normalize='linear', cmap='YlGn')

    # prof = openprof('Files/Sprof/6901660_Sprof.nc')
    #
    # pnum = 21
    # zprofiles(prof,
    #           'BBP700_ADJUSTED',
    #           # [var+'_ADJUSTED' for var in get_biovars(prof)[:4]],
    #           pnum,
    #           )
    # proflocs(prof)
    # mapvariable(prof, 'CHLA', depth=80, s=5)

    # prof_list = openprof(WMOs)
    # prof = openprof(5906474)

    ### Test profdepths

    # depths = profdepths(prof_list)
    # plt.hist(depths, bins=50)

    ### Test histovar

    # histovar(prof_list, 'PH_IN_SITU_TOTAL',
    #          min = 7., max=8.5)
    # fig, ax = plt.gcf(), plt.gca()
    # fig.suptitle('PH_IN_SITU_TOTAL')
    # ax.set_xlabel(var_labels['PH_IN_SITU_TOTAL'])
    # ax.set_ylabel('$N_{points}$')
    # fig.tight_layout()

    # fig, ax = plt.subplots(2, 3)
    # minsmaxs = [[-100., 4000.], [-0.1, 2.], [-0.1, 3.]]
    #
    # for i, var in enumerate(['DOWNWELLING_PAR', 'DOWN_IRRADIANCE380', 'DOWN_IRRADIANCE412']):
    #     ax[0, i].set_title(var)
    #     histovar(prof_list, var, log=True, min = minsmaxs[i][0], max = minsmaxs[i][1], ax=ax[0, i])
    # for i, var in enumerate(['DOWN_IRRADIANCE490', 'LONGITUDE', 'LATITUDE']):
    #     if var == 'DOWN_IRRADIANCE490':
    #         ax[1, i].set_title(var)
    #         histovar(prof_list, var, log=True, min = -0.1, max = 3., ax=ax[1, i])
    #     else:
    #         ax[1, i].set_title(var)
    #         histovar(prof_list, var, ax=ax[1, i])
    #
    # plt.tight_layout()

    # stats = profstats(prof)
    # histovar(prof_list, 'PH_IN_SITU_TOTAL', min=7., max=8.5)

    ### Test mapvariable

    # mapvariable(prof_list, 'BBP700', depth=50, date=dt.datetime(2022, 2, 20), zone_id='SP')

    # extraction = openprof(prof_list, filter={'JULD_MIN': dt.datetime(2022, 1, 1),
    #                                          'JULD_MAX': dt.datetime(2023, 1, 1)})
    # lon, lat, c = profdensity(extraction, Rkm=1000., lonres=40, latres=40, return_count=True)
    # mappable = mapscalarf(lon, lat, c, cbar_label='$N_{1000km}$', vmin=0., vmax=100, bins=16)
    # fig = plt.gcf()
    # fig.suptitle('Number of profiles within 1000km in 2022')

    ### Test scalarcontour et profdensity

    # extract = openprof(prof_list, filter={'JULD_MIN': '2022-01-01',
    #                                       'JULD_MAX': '2022-04-01'})
    # lon, lat, count = profdensity(extract, Rkm=1000., lonres=40, latres=40, return_count=True)
    # scalarcontour(lon, lat, count, vmin=0, vmax=60, levels=12, cbar_label='$N_{1000km}/N_{profiles}$', extend='max')

    ### Animations of profdensity

    # Yearly

    # dates = [dt.datetime(y, 1, 1) for y in range(2015, 2023)]
    # extraction = openprof(prof_list, filter={'JULD_MIN': dates[0],
    #                                          'JULD_MAX': dates[-1]}, verbose=False)
    # lon, lat, c = profdensity(extraction, Rkm=1000., lonres=30, latres=20, return_count=True, verbose=True)
    # mappable = scalarcontour(lon, lat, c, vmin=0, extend='max', levels=20, cbar_label='$N_{profiles}$')
    # plt.gca().set_title('Number of profiles within 1000 km since 2015', pad=10)
    # plt.tight_layout()
    #
    # for i, y in enumerate(range(2015, 2022)):
    #     extraction = openprof(prof_list, filter={'JULD_MIN': dates[i],
    #                                              'JULD_MAX': dates[i+1]}, verbose=False)
    #     lon, lat, c = profdensity(extraction, Rkm=1000., lonres=30, latres=20, return_count=True, verbose=True)
    #     mappable = scalarcontour(lon, lat, c, vmin=0, vmax=150, extend='max', levels=30, cbar_label='$N_{profiles}$')
    #     plt.gcf().tight_layout()
    #     plt.gca().set_title('Number of profiles within 1000 km in {}'.format(y), pad=10)

    # Monthly

    # dates = []
    # for y in range(2015, 2023):
    #     for m in range(1, 13):
    #
    #         if dt.datetime(y, m, 1) > dt.datetime.today():
    #             break
    #         dates.append(dt.datetime(y, m, 1))
    #
    # dates = dates[2:]
    #
    # extraction = openprof(prof_list, filter={'JULD_MIN': dates[-3],
    #                                          'JULD_MAX': dates[-1]}, verbose=False)
    # lon, lat, c = profdensity(extraction, Rkm=1000., lonres=50, latres=35, return_count=True, verbose=False)
    # mappable = scalarcontour(lon, lat, c, vmin=0, vmax=40, extend='max', levels=10)
    #
    # serie = []
    # for i in range(len(dates)-1):
    #     print('{:.0f}% of dates processed.'.format(round(i/(len(dates)-1)*100)), end='\r')
    #     extraction = openprof(prof_list, filter={'JULD_MIN': dates[i],
    #                                              'JULD_MAX': dates[i+1]}, verbose=False)
    #     lon, lat, c = profdensity(extraction, Rkm=1000., lonres=50, latres=35, return_count=True, verbose=False)
    #     serie.append(c)
    #
    # mappable = scalarcontour(lon, lat, serie[0], cbar_label='$N_{1000km}$', vmin=0, vmax=40, extend='max', levels=30)
    #
    # fig = plt.gcf()
    # ax = plt.gca()
    #
    # def animate(i):
    #     ax.set_title('Number of profiles within 1000km on {}'.format(dates[i].strftime('%b %Y')), pad=10)
    #     scalarcontour(lon, lat, serie[i], cbar_label='$N_{1000km}$', vmin=0, vmax=40, levels=30, extend='max', ax=ax,
    #                   print_cbar=False, verbose=False)
    #
    # anim = animation.FuncAnimation(fig, animate, interval = 300, frames=len(serie)-1)

    ### Plottraj

    # ds = xr.open_dataset('Files/Chloro/20220105_d-ACRI-L4-CHL-MULTI_4KM-GLO-DT.nc')
    # lon, lat = ds.lon.values[::10], ds.lat.values[::10]
    # chl_surf = ds.CHL.values[-1,::10,::10]
    #
    # ax = chart(zone_id='WO')
    # scalargrid(lon, lat, chl_surf, ax=ax, vmin=0, vmax=0.6, cmap='jet', cbar_label='Surface chlorophyll '
    #                                                                                            '($mg.m^{-3}$)')
    #
    # ax = plottraj(prof_list)

    # context = chlsurf_context(prof, 'all dates', verbose=True, monthly=True, psw='usemystreetcredz')
    # strhist([c[0] for c in context])

    # context = chlsurf_context(prof, dt.datetime.today(), verbose=True, monthly=True, psw='usemystreetcredz')
    # ztprof(5905106, 'CHLA_ADJUSTED', levels=10)

    ### Chart and point

    # prof = openprof(5906214)

    # ax = plottraj(prof, zone_id='TS', c='purple')
    # point_drifters(prof, ax=ax)
    # point_locations(ax=ax)
    # plt.gcf().savefig('Files/Figures/{}_traj.png'.format(wmo))

    # dmax=500

    # _, axs = plt.subplots(1, 2)
    # VAR1, VAR2 = 'BBP700', 'CHLA'
    # ztprof(prof, VAR1, dmax=dmax, levels=20, ax=axs[0], erase_extreme=True)
    # ztprof(prof, VAR2, dmax=dmax, levels=20, vmin=0., ax=axs[1], erase_extreme=True)
    # plt.gcf().savefig('Files/Figures/{}_{}_{}.png'.format(wmo, VAR1, VAR2))

    # _, axs = plt.subplots(1, 2)
    # VAR1, VAR2 = 'TEMP', 'PSAL'
    # ztprof(prof, VAR1, dmax=dmax, levels=20, ax = axs[0])
    # ztprof(prof, VAR2, dmax=dmax, levels=20, ax = axs[1], erase_extreme=True)
    # plt.gcf().savefig('Files/Figures/{}_{}_{}.png'.format(wmo, VAR1, VAR2))

    # _, axs = plt.subplots(1, 2)
    # VAR1, VAR2 = 'DOWN_IRRADIANCE380', 'DOWN_IRRADIANCE412'
    # ztprof(prof, VAR1, dmax=150, normalize='log', ax = axs[1])
    # ztprof(prof, VAR2, dmax=150, normalize='log', ax = axs[0])
    # plt.gcf().savefig('Files/Figures/{}_{}_{}.png'.format(wmo, VAR1, VAR2))

    # _, axs = plt.subplots(1, 2)
    # VAR1, VAR2 = 'NITRATE', 'DOXY'
    # ztprof(prof, VAR1, dmax=dmax, levels=20, ax = axs[0])
    # ztprof(prof, VAR2, dmax=dmax, levels=20, ax = axs[1])

    # _, ax = plt.subplots()
    # ztprof(prof, 'PH_IN_SITU_TOTAL', dmax=dmax, levels=20, ax = ax)

    # _, ax = plt.subplots()
    # cbar = ztprof(prof, 'CDOM', dmax=200, normalize='linear', levels=20, ax=ax, extend='both', erase_extreme=True)

    ### Context

    # i = 7
    # wmo = WMOs[i]
    # prof = openprof(wmo)
    #
    # context = chlsurf_context(prof, date=dt.datetime.today(), monthly=True, psw='usemystreetcredz')
    # print(context)
    #
    # chl = xr.open_dataset('Files/Chloro/Monthly/20220301_m_20220331-ACRI-L4-CHL-MULTI_4KM-GLO-NRT.nc')
    # skip = 10
    # matrix = chl.CHL.values[-1, ::skip, ::skip]
    # lon, lat = chl.lon.values[::skip], chl.lat.values[::skip]
    # scalarcontour(lon, lat, matrix, vmin=0.0, levels=5, vmax=0.4, cmap=var_cmaps['CHLA'], zone_id='SP',
    #               title='Surface chlorophyll on March 2022', cbar_label=var_labels['CHLA'])
    #
    # point_locations(ax=plt.gca())
    # point_drifters(wmos('al'), ax=plt.gca(), names=False, c='crimson', s=20)
    # plottraj(prof, ax=plt.gca())
    # point_drifters(wmos=wmo, ax=plt.gca())

    # ax = point_drifters(wmos('al'), zone_id='SP_w', c='white', names=False, s=12, bluemarble=True)
    # ax = point_drifters(wmos('il'), zone_id='SP_w', ax=ax, c='pink', s=15, bluemarble=True)
    # ax= point_drifters(wmos('an'), zone_id='SP_w', ax=ax, c='lime', names=False, s=20, bluemarble=True)
    # ax = point_drifters(wmos('in'), zone_id='SP_w', c='red', names=False, s=20, bluemarble=True)
    # point_locations(ax=ax, c='white')
    # ax.set_title(r'\begin{Large} Last known position of floats \end{Large}', pad=20)

    # ax = plottraj(5906214, bluemarble=True, zone_id='SP', c='crimson')
    # point_drifters(5906214, ax=ax, c='white')

    # point_drifters(bluemarble=True)

    # fig = plt.figure()
    #
    # ax = fig.add_subplot(2, 2, 1, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', year=2018)
    # chart(ax=ax)
    # scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Year 2018')
    #
    # ax = fig.add_subplot(2, 2, 2, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', year=2019)
    # chart(ax=ax)
    # scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Year 2019')
    #
    # ax = fig.add_subplot(2, 2, 3, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', year=2020)
    # chart(ax=ax)
    # scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Year 2020')
    #
    # ax = fig.add_subplot(2, 2, 4, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', year=2021)
    # chart(ax=ax)
    # mappable = scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Year 2021')
    #
    # fig.subplots_adjust(left=0.05, right=0.85)
    # cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    # fig.colorbar(mappable, cax=cbar_ax, aspect=30).set_label('Count')
    # fig.suptitle('Profiles count within 1000km')

    # fig = plt.figure()
    #
    # ax = fig.add_subplot(2, 2, 1, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', month=12)
    # count_arr += profcount(ensemble='l', month=1)[2]
    # count_arr += profcount(ensemble='l', month=2)[2]
    # chart(ax=ax)
    # scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Winter')
    #
    # ax = fig.add_subplot(2, 2, 2, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', month=3)
    # count_arr += profcount(ensemble='l', month=4)[2]
    # count_arr += profcount(ensemble='l', month=5)[2]
    # chart(ax=ax)
    # scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Spring')
    #
    # ax = fig.add_subplot(2, 2, 3, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', month=6)
    # count_arr += profcount(ensemble='l', month=7)[2]
    # count_arr += profcount(ensemble='l', month=8)[2]
    # chart(ax=ax)
    # scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Summer')
    #
    # ax = fig.add_subplot(2, 2, 4, projection = ccrs.PlateCarree(central_longitude=180))
    # lon, lat, count_arr = profcount(ensemble='l', month=9)
    # count_arr += profcount(ensemble='l', month=10)[2]
    # count_arr += profcount(ensemble='l', month=11)[2]
    # chart(ax=ax)
    # mappable = scalarcontour(lon, lat, count_arr, vmin=0., vmax=280., levels=6, extend='max', ax=ax, print_cbar=False)
    # ax.set_title('Autumn')
    #
    # fig.subplots_adjust(left=0.05, right=0.85)
    # cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
    # fig.colorbar(mappable, cax=cbar_ax, aspect=30).set_label('Count')
    # fig.suptitle('Profiles count within 1000km')

    # year = 2020
    # wmo = 6902909
    #
    # prof = openprof(wmo, QCs=(0, 1, 2, 3, 5, 6, 7, 8, 9))
    # fig = plt.figure()
    # fig.suptitle('Variables of float \#{} in {}'.format(wmo, year))
    #
    # ax = fig.add_subplot(3, 2, 1, projection = ccrs.PlateCarree(central_longitude=180))
    # chart(ax=ax)
    # CHL_context(wmo, ax=ax, date='{}-12-01'.format(year), display_wmo=False, psw=VAR.psw)
    # ax.set_title('Chlorophyll on December {}'.format(year))
    #
    # ax = fig.add_subplot(3, 2, 2)
    # ZTplot_sprof(prof, 'RHO', ax=ax, levels=9, datemin='{}-01-01'.format(year), vmin=1023., vmax=1027.,
    #              datemax ='{}-12-31'.format(year), discrete=False)
    #
    # ax = fig.add_subplot(3, 2, 3)
    # ZTplot_sprof(prof, 'TEMP', ax=ax, levels=9, datemin='{}-01-01'.format(year), isolines='RHO', vmin=11., vmax=29.5,
    #              datemax ='{}-12-31'.format(year), discrete=False)
    #
    # ax = fig.add_subplot(3, 2, 4)
    # ZTplot_sprof(prof, 'PSAL', ax=ax, levels=9, datemin='{}-01-01'.format(year), isolines='RHO', vmin=35., vmax=36.5,
    #              datemax ='{}-12-31'.format(year), discrete=False)
    #
    # ax = fig.add_subplot(3, 2, 5)
    # ZTplot_sprof(prof, 'CHLA', ax=ax, levels=9, datemin='{}-01-01'.format(year), vmin=0., vmax=0.7, isolines='ISOLUME15',
    #              datemax ='{}-12-31'.format(year), discrete=False)
    #
    # ax = fig.add_subplot(3, 2, 6)
    # ZTplot_sprof(prof, 'DOWNWELLING_PAR', ax=ax, datemin='{}-01-01'.format(year), limiter=2.,
    #              datemax ='{}-12-31'.format(year), log=True, discrete=False)

    # ax = fig.add_subplot(4, 2, 7)
    # ZTplot_sprof(prof, 'DOXY', ax=ax, levels=9, datemin='{}-01-01'.format(year),
    #              datemax ='{}-12-31'.format(year), scientific=True, discrete=False)
    #
    # ax = fig.add_subplot(4, 2, 8)
    # ZTplot_sprof(prof, 'CDOM', ax=ax, levels=9, datemin='{}-01-01'.format(year),
    #              datemax ='{}-12-31'.format(year), scientific=True, discrete=False)

    ########## float_variables test

    # datelim = ('2019-05-01', '2020-05-01')
    # wmo = 6902909
    # vars = ['RHO', 'TEMP', 'PSAL', 'CHLA', 'BBP700', 'DOWNWELLING_PAR', 'DOXY']
    # isolines = [None, ['MLD', 'ISOLUME15'], 'RHO', 'RHO', 'MLD', 'MLD', 'MLD']
    # float_variables(wmo, vars, datelim, isolines=isolines, discrete=False)

    ######### Plotting vars to find adequate colors

    # var = 'CDOM'
    # fig, ax = plt.subplots(3, 2)
    #
    # for i, wmo in enumerate(wmos()[10:10+6]):
    #     print(wmo)
    #     try:
    #         ZTplot_sprof(wmo, var, ax = ax[i//2, i%2])
    #     except Exception:
    #         pass

    # data = xr.open_dataset(VAR.chloropath+'20190401_m_20190430-ACRI-L4-CHL-MULTI_4KM-GLO-REP.nc')
    # data = xr.open_dataset(VAR.chloropath+'20200101_m_20200131-ACRI-L4-CHL-MULTI_4KM-GLO-REP.nc')
    # data = xr.open_dataset(VAR.chloropath+'20210101_m_20210131-ACRI-L4-CHL-MULTI_4KM-GLO-REP.nc')
    # data = xr.open_dataset(VAR.chloropath+'20220101_m_20220131-ACRI-L4-CHL-MULTI_4KM-GLO-DT.nc')
    # chloro = data.CHL.values[0, ::10, ::10]
    # lon = data.lon.values[::10]
    # lat = data.lat.values[::10]
    #
    # lon = np.hstack([lon, 180])
    # chloro0 = chloro[:, 0]
    # chloro0 = chloro0[np.newaxis].T
    # chloro = np.hstack([chloro, chloro0])
    #
    # date = '2022-01-01'
    # ax = MAP_contour(lon, lat, chloro, cmap=VAR.var_cmaps['CHLA'], vmin=0.05, vmax=0.4, levels=8, zone_id='temp',
    #                  cbar_label='Satellite Chlorophyll ('+VAR.var_units['CHLA']+')')
    # PT_floats(WMOS('ess'), ax=ax, datemax=date, display_wmos=False)

    ############## Correlation MODIS / chl_surf

    # month = 2
    # deltat = 7.
    #
    # for month in range(1, 13):
    #
    #     date = dt.datetime(2020, month, 15)
    #     files = np.array(os.listdir('Files/MODIS/'))
    #     file = files[np.array([('2020{:02d}01'.format(month) in file) for file in files])][0]
    #     ds = xr.open_dataset('Files/MODIS/{}'.format(file))
    #     chl_surf = []
    #
    #     for wmo in WMOS():
    #
    #         try:
    #             prof = PRC(wmo)
    #             dates = FMT_date(prof.JULD.values, 'dt', verbose=False)
    #             dates_mask = np.abs(dates - date) < dt.timedelta(deltat)
    #
    #             if np.sum(dates_mask) > 0:
    #                 for i in range(np.sum(dates_mask)):
    #                     if not np.isnan(prof.CHLA_ADJUSTED.values[dates_mask][i]).all():
    #                         chl_surf.append([wmo, prof.LONGITUDE.values[dates_mask][i], prof.LATITUDE.values[dates_mask][i],
    #                                          prof.CHLA_ADJUSTED.values[dates_mask][i][~np.isnan(prof.CHLA_ADJUSTED.values[dates_mask][i])][0]])
    #         except Exception:
    #
    #             pass
    #
    #     chl_surf = np.array(chl_surf)
    #
    #     chl_modis = []
    #     lon, lat, chl = ds.lon.values, ds.lat.values, ds.chlor_a.values
    #     lon, lat = np.meshgrid(lon, lat)
    #     lon, lat, chl = lon.flatten(), lat.flatten(), chl.flatten()
    #     lon, lat, chl = lon[~np.isnan(chl)], lat[~np.isnan(chl)], chl[~np.isnan(chl)]
    #     interp = interpolate.NearestNDInterpolator(np.hstack([lon[np.newaxis].T, lat[np.newaxis].T]), chl)
    #
    #     for i in range(np.shape(chl_surf)[0]):
    #         chl_modis.append(interp(chl_surf[i][1], chl_surf[i][2]))
    #
    #     chl_modis = np.array(chl_modis)
    #
    #     for wmo in WMOS('ess'):
    #         if month == 1:
    #             plt.scatter(chl_modis[chl_surf[:, 0]==wmo], chl_surf[chl_surf[:, 0]==wmo, -1],
    #                         color=VAR.color_cycle[wmo], label=wmo)
    #         else:
    #             plt.scatter(chl_modis[chl_surf[:, 0] == wmo], chl_surf[chl_surf[:, 0] == wmo, -1],
    #                         color=VAR.color_cycle[wmo])
    #
    # plt.plot([-0.1, 0.5], [-0.1, 0.5], c='k')
    # plt.xlabel('Modis chlorophyll-A')
    # plt.ylabel('Floats Chlorophyll-A')
    # plt.grid()
    # plt.legend()
    # plt.gca().set_aspect(1.)
    # plt.xlim(-0.1, 0.5)
    # plt.ylim(-0.1, 0.5)
    # plt.tight_layout()

    ############### Diff CHLA_PRC and CHLA_ADJUSTED

    # prof = PRC(2701)
    # diff = prof.CHLA_PRC - prof.CHLA_ADJUSTED
    #
    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(prof.JULD.values, prof.MLD_T02_QI.values)
    # ax[0].set_ylabel('MLD QI')
    # ax[0].scatter(prof.JULD.values, prof.MLD_T02_QI.values, c='k', s=7)
    # ax[0].set_xticks([])
    # ZT_xarr(diff, prof.PRES, prof.JULD, ax=ax[1], cmap='seismic', vmin=-0.2, vmax=0.2, cbarlab='$\Delta chl (mg.m^{-3})$')
    # ax[0].set_xlim(ax[1].get_xlim())
    # fig.suptitle('MLD quality index and CHLA_PRC - CHLA_ADJUSTED (float \#{})'.format(int(prof.PLATFORM_NUMBER.values[0])))

    ############### GLODAPv2 salinity

    # depth = 250
    #
    # fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    # nc = xr.open_dataset('Files/Climatologies/GLODAPv2.salinity.nc')
    # lon, lat, mat, mat_err = nc.lon.values, nc.lat.values,\
    #                          nc.salinity.values[np.argmin(np.abs(nc.Pressure.values-depth))],\
    #                          nc.salinity_error.values[np.argmin(np.abs(nc.Pressure.values - depth))]
    # CHART(ax=axs[0])
    # CHART(ax=axs[1])
    # PT_floats(ax=axs[0], verbose=False)
    # PT_floats(ax=axs[1], verbose=False)
    # MAP_grid(lon, lat, mat, ax=axs[0], cmap=VAR.var_cmaps['PSAL'], cbar_label='Salinity ($ppm$)', extend='both', vmin=34.5, vmax=36.5)
    # MAP_grid(lon, lat, mat_err, ax=axs[1], cmap='jet', cbar_label='Salinity error ($ppm$)', extend='max', vmin=0., vmax=1.)
    # title = nc.Description
    # title = title[:np.argmax([e=='.' for e in title])]
    # fig.suptitle('\n'.join(wrap(title+' at {:.0f}dbar'.format(float(nc.Pressure[np.argmin(np.abs(nc.Pressure.values-depth))])))))
    # plt.subplots_adjust(hspace=0.3)
    # fig.savefig(VAR.figpath+'Climatologies/GLODAPv2.salinity.YR.{:.0f}m.png'.format(depth))
    #
    # year = 2019
    # nc = xr.open_dataset(VAR.indexpath + 'IPRC/IPRC_MLD_SP_{}.nc'.format(year))
    # ax = MAP_grid(nc.LON141_290.values, nc.LAT41_100.values, nc.MLD.values[0], vmin=20., vmax=80.,
    #               cbar_label='MLD ($m$)')
    # PT_floats(ax=ax, verbose=False)
    # ax.set_title('Annual mean of Argo MLD, 1$^{{\circ}}$x1$^{{\circ}}$ ({})'.format(year))
    # plt.gcf().savefig(VAR.figpath + '/Climatologies/argo_mld_yr_{}.png'.format(year))

    ############### GLODAPv2 salinity

    # fig, axs = plt.subplots(1, 2, sharey=True)
    # sal = xr.open_dataset('Files/Climatologies/GLODAPv2.salinity.nc')
    # temp = xr.open_dataset('Files/Climatologies/GLODAPv2.theta.nc')
    # axs[0].set_ylim(300., 0.)
    #
    # for i, wmo in enumerate(WMOS()):
    #
    #     pos = RAW(wmo).LONGITUDE.values[-1], RAW(wmo).LATITUDE.values[-1]
    #     lon_i, lat_i = np.argmin(np.abs(sal.lon.values - pos[0])), np.argmin(np.abs(sal.lat.values - pos[1]))
    #     sal_val = sal.salinity.values[:, lat_i, lon_i]
    #     temp_val = temp.theta.values[:, lat_i, lon_i]
    #     axs[0].plot(sal_val, sal.Pressure.values, c=VAR.color_cycle[wmo], linewidth=4., alpha=0.7, label=str(wmo), linestyle='-'
    #     if i%3==0 else '--' if i%3==1 else ':')
    #     axs[1].plot(temp_val, sal.Pressure.values, c=VAR.color_cycle[wmo], linewidth=4., alpha=0.7, label=str(wmo), linestyle='-'
    #     if i%3==0 else '--' if i%3==1 else ':')
    #
    # axs[0].legend()
    # axs[0].set_ylabel('Pressure ($dbar$)')
    # axs[0].set_xlabel('Salinity ($PSU$)')
    # axs[1].set_xlabel('Temperature ($^{\circ}C$)')
    # axs[0].set_title('Salinity profiles')
    # axs[1].set_title('Temperature profiles')
    # axs[1].set_xlim(10., 30.)
    # fig.suptitle('GLODAPV2 climatologies S and T profiles at float last known position')
    # plt.tight_layout()


    ############ Oxygen GLODAPv2


    # fig, ax = plt.subplots(2, 1, sharex=True)
    # var = 'oxygen'
    # unit = '$\mu mol.kg^{-1}$'
    # nc = xr.open_dataset('Files/Climatologies/GLODAPv2.{}.nc'.format(var))
    # heights_h, heights_z = [], []
    # n = -1
    # dmax = 500.
    # hscale = 500000
    #
    # for i, wmo in enumerate(WMOS()[:n]):
    #
    #     pos = RAW(wmo).LONGITUDE.values[-1], RAW(wmo).LATITUDE.values[-1]
    #     depth_i = np.argmin(np.abs(nc.Pressure.values - dmax))
    #     lon_i, lat_i = np.argmin(np.abs(nc.lon.values - pos[0])), np.argmin(np.abs(nc.lat.values - pos[1]))
    #     valz = nc[var].values[:depth_i, lat_i, lon_i]
    #     meshlon, meshlat = np.meshgrid(nc.lon.values, nc.lat.values)
    #     valh = nc[var].where(np.sqrt((2 * np.pi * VAR.Rt * np.cos(pos[1] * np.pi / 180) / 360 *
    #                                       (meshlon - pos[0])) ** 2 +
    #                                      (2*np.pi*VAR.Rt/360*(meshlat-pos[1])) ** 2) < hscale).values[:depth_i]
    #     heights_z.append(np.nanstd(valz)), heights_h.append(np.nanstd(valh))
    #     if np.isnan(nc[var].values[depth_i, lat_i, lon_i]):
    #         heights_z[-1] = np.nan
    #
    # bar_pos = (np.arange(len(WMOS()[:n])))
    # ax[0].bar(bar_pos, heights_h, color=list(VAR.color_cycle.values()))
    # ax[1].bar(bar_pos, heights_z, color=list(VAR.color_cycle.values()))
    # ax[1].set_xticks(bar_pos)
    # ax[1].set_xticklabels([str(e)[-4:] for e in WMOS()[:n]], fontsize=13)
    # ax[0].set_ylim(0., np.max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]]))
    # ax[1].set_ylim(0., np.max([ax[0].get_ylim()[1], ax[1].get_ylim()[1]]))
    # ax[0].grid(zorder=0)
    # ax[1].grid(zorder=0)
    # ax[0].set_ylabel('{} std$_h$ R = {:.0f}km ({})'.format(var, hscale/1000, unit), fontsize=14)
    # ax[1].set_ylabel('{} std$_z$ 0-{:.0f}m ({})'.format(var, dmax, unit), fontsize=14)
    # fig.suptitle('GLODAPV2 climatologies of {}: h and z standard deviation'.format(var))
    # fig.tight_layout()

    ############### GLODAPv2 climato

    # units = ['$PSU$', '$^{\circ}C$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$']
    #
    # for k, var in enumerate(['salinity', 'theta', 'phosphate', 'silicate', 'nitrate', 'oxygen']):
    #
    #     fig, ax = plt.subplots(1, 1, sharex=True)
    #     unit = units[k]
    #     nc = xr.open_dataset('Files/Climatologies/GLODAPv2.{}.nc'.format(var))
    #     heights_h, heights_z = [], []
    #     n = -1
    #     dmax = 400.
    #     hscale = 500000
    #
    #     for i, wmo in enumerate(WMOS()[:n]):
    #
    #         pos = RAW(wmo).LONGITUDE.values[-1], RAW(wmo).LATITUDE.values[-1]
    #         depth_i = np.argmin(np.abs(nc.Pressure.values - dmax)) + 1
    #         lon_i, lat_i = np.argmin(np.abs(nc.lon.values - pos[0])), np.argmin(np.abs(nc.lat.values - pos[1]))
    #         meshlon, meshlat = np.meshgrid(nc.lon.values, nc.lat.values)
    #         valh = nc[var].where(np.sqrt((2 * np.pi * VAR.Rt * np.cos(pos[1] * np.pi / 180) / 360 *
    #                                           (meshlon - pos[0])) ** 2 +
    #                                          (2*np.pi*VAR.Rt/360*(meshlat-pos[1])) ** 2) < hscale).values[:depth_i]
    #         # ndepth = [np.sum(~np.isnan(valh[l])) for l in range(depth_i)]
    #         # _, ax2 = plt.subplots()
    #         # ax2.scatter(nc.Pressure.values[:depth_i], ndepth)
    #         # ax2.set_title(str(wmo))
    #         valh = valh.flatten()
    #         valh = valh[~np.isnan(valh)]
    #         ax.boxplot(valh, positions=[i], widths=0.7, showfliers=False,
    #                       medianprops=dict(linewidth=3., color=VAR.color_cycle[wmo], zorder=0))
    #
    #     bar_pos = (np.arange(len(WMOS()[:n])))
    #     ax.set_xticks(bar_pos)
    #     ax.set_xticklabels([str(e)[-4:] for e in WMOS()[:n]], fontsize=13)
    #     ax.grid(zorder=0)
    #     ax.set_ylabel('{} ({})'.format(var, unit))
    #     fig.suptitle('GLODAPV2 climatologies of {}: std at floats positions (R={:.0f}, prof$_{{max}}$ = {:.0f})'
    #                  .format(var, hscale/1000, dmax))
    #     fig.tight_layout()
    #     fig.savefig(VAR.figpath+'Climatologies/glodapv2_{}_boxplot_{:.0f}km_{:.0f}m.png'.format(var, hscale/1000, dmax))

    ############# MODIS chloro satellite

    # nc2015 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20150101_20151231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2016 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20160101_20161231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2017 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20170101_20171231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2018 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20180101_20181231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2019 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20190101_20191231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2020 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20200101_20201231.L3m.YR.CHL.chlor_a.9km.nc')
    #
    # lon, lat = nc2015.lon.values, nc2015.lat.values
    # vals = np.mean(np.stack([nc2015.chlor_a.values, nc2016.chlor_a.values, nc2017.chlor_a.values, nc2018.chlor_a.values,
    #                          nc2019.chlor_a.values, nc2020.chlor_a.values], axis=0), axis=0)
    #
    # fig, ax = plt.subplots(1, 1, sharex=True)
    #
    # for i, wmo in enumerate(WMOS()):
    #
    #     prof = PRC(wmo)
    #     traj = np.vstack([prof.LONGITUDE.values, prof.LATITUDE.values])
    #
    #     boxvals = []
    #     for k in range(np.shape(traj)[1]):
    #         lon_i, lat_i = np.argmin(np.abs(lon - traj[0, k])), np.argmin(np.abs(lat - traj[1, k]))
    #         boxvals.append(vals[lat_i, lon_i])
    #
    #     boxvals = np.array(boxvals).flatten()
    #     boxvals = boxvals[~np.isnan(boxvals)]
    #     ax.boxplot(boxvals, positions=[i], widths=0.7, showfliers=False,
    #                medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    # bar_pos = (np.arange(len(WMOS())))
    # ax.set_xticks(bar_pos)
    # ax.set_xticklabels([str(e)[-4:] for e in WMOS()], fontsize=13)
    # ax.grid(zorder=0)
    # ax.set_xlabel('Float WMO')
    # ax.set_ylabel('Chlorophyll ($mg.m^{-3}$)')
    # fig.suptitle('MODIS sea surface chlorophyll 2015-2020 average along floats trajectories')
    # fig.tight_layout()
    #
    # mapax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
    # mapax.set_extent((-217.70397906323512, -68.29602093676488, -40.09551303398712, 3.0955130339871175))
    # CHART(ax=mapax)
    # MAP_grid(lon, lat, vals, ax=mapax, zone_id='SP', cmap='viridis', vmin=0., vmax=0.5,
    #          extend='max', cbar_label='MODIS chlorophyll ($mg.m^{-3}$)')
    # PT_floats(WMOS(), ax=mapax, bluemarble=True, c='red')
    # mapax.set_position([0.3, 0.5, 0.45, 0.35])
    # fig.axes[-1].set_position([0.41, 0.43, 0.23, 0.03])
    #
    # fig.savefig(VAR.figpath + 'Climatologies/MODIS_chl_2015to2020_average_boxplot.png')

    ############# MODIS chloro satellite

    # for year in range(2015, 2021):
    #
    #     files = os.listdir(VAR.indexpath+'MODIS')
    #     file = None
    #
    #     for f in files:
    #         if str(year) in f and 'chlor_a' in f:
    #             file = f
    #             break
    #
    #     nc = xr.open_dataset(VAR.indexpath+'MODIS/'+file)
    #
    #     fig, ax = plt.subplots(1, 1, sharex=True)
    #     hscale = 500000
    #
    #     for i, wmo in enumerate(WMOS()):
    #
    #         pos = GET_floatpos(wmo, dt.datetime(year, 7, 1))
    #
    #         if not np.isnan(pos[0]):
    #             lon_i, lat_i = np.argmin(np.abs(nc.lon.values - pos[0])), np.argmin(np.abs(nc.lat.values - pos[1]))
    #             meshlon, meshlat = np.meshgrid(nc.lon.values, nc.lat.values)
    #             valh = nc.chlor_a.where(np.sqrt((2 * np.pi * VAR.Rt * np.cos(pos[1] * np.pi / 180) / 360 *
    #                                      (meshlon - pos[0])) ** 2 +
    #                                     (2 * np.pi * VAR.Rt / 360 * (meshlat - pos[1])) ** 2) < hscale).values
    #             valh = valh.flatten()
    #             valh = valh[~np.isnan(valh)]
    #             ax.boxplot(valh, positions=[i], widths=0.7, showfliers=False,
    #                        medianprops=dict(linewidth=3., color=VAR.color_cycle[wmo], zorder=0))
    #
    #     bar_pos = (np.arange(len(WMOS())))
    #     ax.set_xticks(bar_pos)
    #     ax.set_xticklabels([str(e)[-4:] for e in WMOS()], fontsize=13)
    #     ax.set_ylim(-0.02, 0.53)
    #     ax.set_xlim(-1., len(WMOS()))
    #     ax.grid(zorder=0)
    #     ax.set_xlabel('Float WMO')
    #     ax.set_ylabel('Sea Surface Chlorophyll ($mg.m^{-3}$)')
    #     fig.suptitle('MODIS sea surface chlorophyll {} average around at floats positions ($R={:.0f}km$)'.format(year, hscale / 1000))
    #     fig.tight_layout()
    #
    #     mapax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
    #     mapax.set_extent((-217.70397906323512, -68.29602093676488, -40.09551303398712, 3.0955130339871175))
    #     MAP_grid(nc.lon.values, nc.lat.values, nc.chlor_a.values, ax=mapax, zone_id='SP', cmap='viridis', vmin=0., vmax=0.5,
    #              extend='max', cbar_label='MODIS chlorophyll ($mg.m^{-3}$)')
    #     PT_floats(WMOS(), ax=mapax, datemax=dt.datetime(year, 7, 1), bluemarble=True, c='red')
    #     mapax.set_position([0.4, 0.5, 0.45, 0.35])
    #     mapax.gridlines(draw_labels=True)
    #     # fig.axes[-1].set_position([0.51, 0.43, 0.23, 0.03])
    #     fig.savefig(VAR.figpath + 'Climatologies/MODIS_YR_chl_{}_boxplot_{:.0f}km_masked.png'.format(year, hscale / 1000))

    ########## Argo climato

    # nc = xr.open_dataset('Files/Argo/IPRC_S_SP_2015to2019.nc')
    #
    # for year in range(2015, 2020):
    #
    #     for depth in [0., 50., 100., 150., 200., 250., 300., 400., 500.]:
    #
    #         depth_i = np.argmin(np.abs(nc.LEV1_16.values-depth))
    #         ti = year-2015
    #         ax = MAP_grid(nc.LON131_300.values, nc.LAT31_100.values, nc.SALT.values[ti, depth_i], vmin=34.5, vmax=36.5,
    #                       cbar_label='S ($PSU$)', cmap=VAR.var_cmaps['PSAL'])
    #         PT_floats(ax=ax, verbose=False)
    #         ax.set_title('{} mean of Argo Salinity, 1$^{{\circ}}$x1$^{{\circ}}$, {:.0f}m'.format(year, depth))
    #         plt.gcf().savefig(VAR.figpath + '/Climatologies/argo_sal_yr_{}_{:.0f}m.png'.format(year, depth))


    ########## Argo Climato TS diagrams


    # sal = xr.open_dataset('Files/Argo/IPRC_S_SP_2015to2019.nc')
    # temp = xr.open_dataset('Files/Argo/IPRC_T_SP_2015to2019.nc')
    # depth, lon, lat, sal_val, temp_val = sal.LEV1_16.values, sal.LON131_300.values, sal.LAT31_100.values,\
    #                                      np.nanmean(sal.SALT.values, axis=0), np.nanmean(temp.PTEMP.values, axis=0)
    # X, Y = np.linspace(33., 37., 100), np.linspace(5., 35., 100)
    # X, Y = np.meshgrid(X, Y)
    # sigma0 = sw.density.sigma0(X, Y)
    #
    # sback, tback = [], []
    # tsdiaxes = []
    # tsdiags = []
    #
    # for i, wmo in enumerate(WMOS()):
    #
    #     prof = PRC(wmo)
    #     traj = np.vstack([prof.LONGITUDE.values, prof.LATITUDE.values])
    #     for k in range(np.shape(traj)[1]):
    #         if traj[0, k] < 0.:
    #             traj[0, k] += 360.
    #
    #     tval = []
    #     sval = []
    #     dval = []
    #     for k in range(np.shape(traj)[1]):
    #         lon_i, lat_i = np.argmin(np.abs(lon - traj[0, k])), np.argmin(np.abs(lat - traj[1, k]))
    #         tval.append(temp_val[:, lat_i, lon_i])
    #         sval.append(sal_val[:, lat_i, lon_i])
    #         dval.append(depth[:])
    #
    #     tback.append(tval[-1])
    #     sback.append(sval[-1])
    #
    #     tval, sval, dval = np.array(tval).flatten(), np.array(sval).flatten(), np.array(dval).flatten()
    #
    #     tsdiag, tsdiax = plt.subplots(1, 1)
    #     cont = tsdiax.contour(X, Y, sigma0, colors='k', levels=np.arange(20., 35., 0.5), zorder=0, linewidths=0.7,
    #                           linestyles='--')
    #     cont.clabel()
    #     scat = tsdiax.scatter(sval, tval, c=dval, cmap='viridis_r', vmin=0., vmax=350., label=str(wmo), s=200,
    #                           zorder=1, marker='v')
    #     tsdiag.colorbar(scat, extend='max', label='Depth (m)')
    #     tsdiax.set_aspect(1./10)
    #     tsdiax.set_xlim(34., 36.5)
    #     tsdiax.set_ylim(6., 31.)
    #     tsdiax.set_ylabel('Temperature ($^{\circ}C$)')
    #     tsdiax.set_xlabel('Salinity ($PSU$)')
    #     tsdiag.suptitle('T/S diagram along float \#{} trajectory from Argo gridded fields (2015-2019 average)'
    #                     .format(wmo))
    #     tsdiag.tight_layout()
    #
    #     tsdiaxes.append(tsdiax)
    #     tsdiags.append(tsdiag)
    #
    # sback, tback = np.array(sback).flatten(), np.array(tback).flatten()
    #
    # for d in tsdiaxes:
    #     d.scatter(sback, tback, color='k', alpha=0.1, s=40, zorder=0)
    # for i, d in enumerate(tsdiags):
    #     d.savefig('Files/Figures/Climatologies/TSdiag_ArgoClimato_{}.png'.format(VAR.essentials[i]))


    ########## Climato GLODAPv2 with errors

    # vars = ['salinity', 'theta', 'phosphate', 'silicate', 'nitrate', 'oxygen']
    # units = ['$PSU$', '$^{\circ}C$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$']
    # cmaps = [VAR.var_cmaps['PSAL'], VAR.var_cmaps['TEMP'], 'plasma', 'magma', VAR.var_cmaps['NITRATE'],
    #          VAR.var_cmaps['DOXY']]
    # bounds = [[34.5, 36.5], [13., 31.], [0., 2.], [0., 15.], [0., 20.], [150., 250.]]
    # err_bounds = [1., 5., 0.6, 15., 7.5, 50.]
    # extends = ['both', 'both', 'max', 'max', 'max', 'both']
    #
    # for i, var in enumerate(vars):
    #
    #     unit = units[i]
    #
    #     for depth in [0., 50., 100., 200., 500.]:
    #
    #         fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True,
    #                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    #         nc = xr.open_dataset('Files/Climatologies/GLODAPv2.{}.nc'.format(var))
    #         lon, lat, mat, mat_err = nc.lon.values, nc.lat.values,\
    #                                  nc[var].values[np.argmin(np.abs(nc.Pressure.values-depth))],\
    #                                  nc[var+'_error'].values[np.argmin(np.abs(nc.Pressure.values - depth))]
    #         CHART(ax=axs[0], verbose=False)
    #         CHART(ax=axs[1], verbose=False)
    #         PT_floats(ax=axs[0], verbose=False)
    #         PT_floats(ax=axs[1], verbose=False)
    #
    #         MAP_grid(lon, lat, mat, ax=axs[0], cmap=cmaps[i], cbar_label='{} ({})'.format(var, unit),
    #                  extend='both', vmin=bounds[i][0], vmax=bounds[i][1], verbose=False)
    #         MAP_grid(lon, lat, mat_err, ax=axs[1], cmap='jet', cbar_label='{} error ({})'.format(var, unit),
    #                  extend='max', vmin=0., vmax=err_bounds[i], verbose=False)
    #         title = nc.Description
    #         title = title[:np.argmax([e=='.' for e in title])]
    #         fig.suptitle('\n'.join(wrap(title+' at {:.0f}dbar'
    #                                     .format(float(nc.Pressure[np.argmin(np.abs(nc.Pressure.values-depth))])))))
    #         plt.subplots_adjust(hspace=0.3)
    #         fig.savefig(VAR.figpath+'Climatologies/GLODAPv2.{}.YR.{:.0f}m_err.png'.format(var, depth))


    ########## Climato GLODAPv2 without errors

    # vars = ['salinity', 'theta', 'phosphate', 'silicate', 'nitrate', 'oxygen']
    # units = ['$PSU$', '$^{\circ}C$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$']
    # cmaps = [VAR.var_cmaps['PSAL'], VAR.var_cmaps['TEMP'], 'plasma', 'magma', VAR.var_cmaps['NITRATE'],
    #          VAR.var_cmaps['DOXY']]
    # bounds = [[34.5, 36.5], [13., 31.], [0., 2.], [0., 15.], [0., 20.], [150., 250.]]
    # extends = ['both', 'both', 'max', 'max', 'max', 'both']
    #
    # for i, var in enumerate(vars):
    #
    #     unit = units[i]
    #
    #     for depth in [0., 50., 100., 200., 500.]:
    #
    #         fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    #         nc = xr.open_dataset('Files/Climatologies/GLODAPv2.{}.nc'.format(var))
    #         lon, lat, mat, mat_err = nc.lon.values, nc.lat.values,\
    #                                  nc[var].values[np.argmin(np.abs(nc.Pressure.values-depth))],\
    #                                  nc[var+'_error'].values[np.argmin(np.abs(nc.Pressure.values - depth))]
    #         CHART(ax=ax, verbose=False)
    #         PT_floats(ax=ax, verbose=False)
    #
    #         MAP_grid(lon, lat, mat, ax=ax, cmap=cmaps[i], cbar_label='{} ({})'.format(var, unit),
    #                  extend='both', vmin=bounds[i][0], vmax=bounds[i][1], verbose=False)
    #         title = nc.Description
    #         title = title[:np.argmax([e=='.' for e in title])]
    #         fig.suptitle('\n'.join(wrap(title+' at {:.0f}dbar'
    #                                     .format(float(nc.Pressure[np.argmin(np.abs(nc.Pressure.values-depth))])))))
    #         plt.subplots_adjust(hspace=0.3)
    #         fig.savefig(VAR.figpath+'Climatologies/GLODAPv2.{}.YR.{:.0f}m.png'.format(var, depth))

    ######### Float map chlorophyll

    # nc = xr.open_dataset('Files/MODIS/AQUA_MODIS.20180101_20181231.L3m.YR.CHL.chlor_a.9km.nc')
    # lon, lat = nc.lon.values, nc.lat.values
    # vals = nc.chlor_a.values
    # ax = CHART()
    # MAP_grid(lon, lat, vals, cmap='viridis', vmin=0., vmax=0.5, extend='max', ax=ax, cbar_label='SSC ($mg.m^{-3}$)')
    # PT_floats(WMOS(), ax=ax)

    ######### MLD comparison

    # fig, axs = plt.subplots(2, 4)
    # fig.suptitle('MLD$_t$ vs MLD$_s$')
    # axs = axs.flatten()
    # for i, wmo in enumerate([6645, 1656, 1687, 2701, 2909, 2907, 6636, 6635]):
    #     prof = PRC(wmo)
    #     axs[i].scatter(prof.MLD_T02.values, prof.MLD_S.values, s=7)
    #     xlims = axs[i].get_xlim()
    #     ylims = axs[i].get_ylim()
    #     axs[i].plot([0., 200.], [0., 200.], c='k', linewidth=0.7)
    #     axs[i].set_xlim(xlims)
    #     axs[i].set_ylim(ylims)
    #     axs[i].set_aspect(1.)
    #     axs[i].grid()
    #     axs[i].set_xlabel('MLD$_t$')
    #     axs[i].set_ylabel('MLD$_s$')
    #     axs[i].set_title('Float \#{}'.format(int(prof.PLATFORM_NUMBER.values[0])))
    # fig.tight_layout()

    ############ Float values hist

    # for k, var in enumerate(['TEMP', 'PSAL', 'SIG0', 'BVF', 'DOWNWELLING_PAR', 'CHLA', 'CHLA_ADJUSTED', 'CHLA_PRC',
    #                          'BBP700', 'CDOM','NITRATE', 'DOXY']):
    #
    #     fig, ax = plt.subplots(1, 1, sharex=True)
    #     unit = VAR.var_units[var]
    #
    #     for i, wmo in enumerate(WMOS()):
    #
    #         prc = PRC(wmo)
    #
    #         if var in list(prc.data_vars):
    #
    #             dmax = np.nanmean(prc.CDARK_DEPTH.values)
    #             vals = np.where(prc.PRES.values * VAR.hf > dmax, np.nan, prc[var].values)
    #             vals = vals[~np.isnan(vals)].flatten()
    #             ax.boxplot(vals, positions=[i], widths=0.7, showfliers=False,
    #                        medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    #     bar_pos = (np.arange(len(WMOS())))
    #     ax.set_xticks(bar_pos)
    #     ax.set_xticklabels([str(e)[-4:] for e in WMOS()], fontsize=13)
    #     ax.grid(zorder=0)
    #     ax.set_ylabel('{} ({})'.format(var, unit))
    #     fig.suptitle('Float {} values (prof$_{{max}} = Dark_{{mean}}$)'.format(var))
    #     fig.tight_layout()
    #     fig.savefig(VAR.figpath+'{}_floatvalues_boxplot.png'.format(var))

    ########### Argo climato salinity along floats traj

    # nc = xr.open_dataset('Files/Argo/IPRC_S_SP_2015to2019.nc')
    # lon, lat, depth = nc.LON131_300.values, nc.LAT31_100.values, nc.LEV1_16.values
    # vals = np.nanmean(nc.SALT.values, axis=0)
    #
    # fig, ax = plt.subplots(1, 1, sharex=True)
    #
    # for i, wmo in enumerate(WMOS()):
    #
    #     prof = PRC(wmo)
    #     traj = np.vstack([prof.LONGITUDE.values, prof.LATITUDE.values])
    #     for k in range(np.shape(traj)[1]):
    #         if traj[0, k] < 0.:
    #             traj[0, k] += 360.
    #     depth_i = np.argmin(np.abs(depth - np.nanmean(prof.CDARK_DEPTH.values))) + 1
    #
    #     boxvals = []
    #     for k in range(np.shape(traj)[1]):
    #         lon_i, lat_i = np.argmin(np.abs(lon - traj[0, k])), np.argmin(np.abs(lat - traj[1, k]))
    #         boxvals.append(vals[:depth_i, lat_i, lon_i])
    #
    #     boxvals = np.array(boxvals).flatten()
    #     boxvals = boxvals[~np.isnan(boxvals)]
    #     ax.boxplot(boxvals, positions=[i], widths=0.7, showfliers=False,
    #                medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    # bar_pos = (np.arange(len(WMOS())))
    # ax.set_xticks(bar_pos)
    # ax.set_xticklabels([str(e)[-4:] for e in WMOS()], fontsize=13)
    # ax.grid(zorder=0)
    # ax.set_xlabel('Float WMO')
    # ax.set_ylabel('Potential temperature ($^{\circ}C$)')
    # fig.suptitle('Argo Salinity 2015-2019 average along floats trajectories ($depth_{max}=CHL_{dark}$)')
    # fig.tight_layout()
    #
    # fig.savefig(VAR.figpath + 'Climatologies/ArgoClimato_2015to2020_salinity_boxplot.png')


    ############# Argo climato temperature map

    # nc = xr.open_dataset('Files/Argo/IPRC_T_SP_2015to2019.nc')
    # lon, lat = nc.LON131_300.values, nc.LAT31_100.values
    # vals = np.nanmean(nc.PTEMP.values, axis=0)
    #
    # for depth in [0., 50., 100., 200., 500.]:
    #
    #     cmap = VAR.var_cmaps['TEMP']
    #     vmin = 10.
    #     vmax = 30.
    #     extend = 'both'
    #     cbarlab = 'T ($^{\circ}C$)'
    #     title = 'Argo temperature climatology 2015-2019 (${:.0f}m$)'.format(depth)
    #
    #     depth_i = np.argmin(np.abs(nc.LEV1_16.values - depth))
    #     ax = CHART()
    #     MAP_grid(lon, lat, vals[depth_i], cmap=cmap, vmin=vmin, vmax=vmax, extend=extend, ax=ax, cbar_label=cbarlab)
    #     PT_floats(WMOS(), ax=ax)
    #     ax.set_title(title)
    #
    #     plt.gcf().savefig(VAR.figpath + 'Climatologies/ArgoClimato_temperature_2015to2019_{:.0f}.png'.format(depth))


    ########## GLODAP v2 boxplots masked

    # units = ['$PSU$', '$^{\circ}C$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$']
    # vars = ['salinity', 'theta', 'phosphate', 'silicate', 'nitrate', 'oxygen']
    #
    # for k, var in enumerate(vars):
    #
    #     unit = units[k]
    #     nc = xr.open_dataset('Files/Climatologies/GLODAPv2.{}.nc'.format(var))
    #     lon, lat, depth = nc.lon.values, nc.lat.values, nc.Pressure.values * VAR.hf
    #     vals = nc[var].values
    #     std = np.nanstd(vals)
    #     mask = nc[var + '_error'].values > std / 2
    #     vals = np.where(mask, np.nan, vals)
    #
    #     fig, ax = plt.subplots(1, 1, sharex=True)
    #
    #     for i, wmo in enumerate(WMOS()):
    #
    #         prof = PRC(wmo)
    #         traj = np.vstack([prof.LONGITUDE.values, prof.LATITUDE.values])
    #         depth_i = np.argmin(np.abs(depth - np.nanmean(prof.CDARK_DEPTH.values))) + 1
    #
    #         boxvals = []
    #         for k in range(np.shape(traj)[1]):
    #             lon_i, lat_i = np.argmin(np.abs(lon - traj[0, k])), np.argmin(np.abs(lat - traj[1, k]))
    #             boxvals.append(vals[:depth_i, lat_i, lon_i])
    #
    #         boxvals = np.array(boxvals).flatten()
    #         boxvals = boxvals[~np.isnan(boxvals)]
    #         ax.boxplot(boxvals, positions=[i], widths=0.7, showfliers=False,
    #                    medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    #     bar_pos = (np.arange(len(WMOS())))
    #     ax.set_xticks(bar_pos)
    #     ax.set_xticklabels([str(e)[-4:] for e in WMOS()], fontsize=13)
    #     ax.grid(zorder=0)
    #     ax.set_xlabel('Float WMO')
    #     ax.set_ylabel('{} ({})'.format(var, unit))
    #     fig.suptitle('Glodap {} along floats trajectories ($depth_{{max}}=CHL_{{dark}}$)'.format(var))
    #     fig.tight_layout()
    #
    #     fig.savefig(VAR.figpath + 'Climatologies/glodapv2_{}_boxplot_ontrajectory_untildark_masked.png'.format(var))


    ############## GLODAP v2 maps with mask

    # vars = ['salinity', 'theta', 'phosphate', 'silicate', 'nitrate', 'oxygen']
    # units = ['$PSU$', '$^{\circ}C$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$']
    # cmaps = [VAR.var_cmaps['PSAL'], VAR.var_cmaps['TEMP'], 'plasma', 'magma', VAR.var_cmaps['NITRATE'],
    #          VAR.var_cmaps['DOXY']]
    # bounds = [[34.5, 36.5], [13., 31.], [0., 2.], [0., 15.], [0., 20.], [150., 250.]]
    # err_bounds = [1., 5., 0.6, 15., 7.5, 50.]
    # extends = ['both', 'both', 'max', 'max', 'max', 'both']
    #
    # for i, var in enumerate(vars):
    #
    #     unit = units[i]
    #
    #     for depth in [0., 50., 100., 200., 500.]:
    #
    #         fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True,
    #                                 subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    #         nc = xr.open_dataset('Files/Climatologies/GLODAPv2.{}.nc'.format(var))
    #         lon, lat, mat, mat_err = nc.lon.values, nc.lat.values,\
    #                                  nc[var].values[np.argmin(np.abs(nc.Pressure.values-depth))],\
    #                                  nc[var+'_error'].values[np.argmin(np.abs(nc.Pressure.values - depth))]
    #
    #         std = np.nanstd(mat)
    #         mat = np.where(mat_err > std/2, np.nan, mat)
    #
    #         CHART(ax=axs[0], verbose=False)
    #         CHART(ax=axs[1], verbose=False)
    #         PT_floats(ax=axs[0], verbose=False)
    #         PT_floats(ax=axs[1], verbose=False)
    #
    #         MAP_grid(lon, lat, mat, ax=axs[0], cmap=cmaps[i], cbar_label='{} ({})'.format(var, unit),
    #                  extend='both', vmin=bounds[i][0], vmax=bounds[i][1], verbose=False)
    #         MAP_grid(lon, lat, mat_err, ax=axs[1], cmap='jet', cbar_label='{} error ({})'.format(var, unit),
    #                  extend='max', vmin=0., vmax=err_bounds[i], verbose=False)
    #         title = nc.Description
    #         title = title[:np.argmax([e=='.' for e in title])]
    #         fig.suptitle('\n'.join(wrap(title+' at {:.0f}dbar'
    #                                     .format(float(nc.Pressure[np.argmin(np.abs(nc.Pressure.values-depth))])))))
    #         plt.subplots_adjust(hspace=0.3)
    #         fig.savefig(VAR.figpath+'Climatologies/GLODAPv2.{}.YR.{:.0f}m_err_masked.png'.format(var, depth))


    ############## Boxplots float variables


    # for k, var in enumerate(['BVF']):
    #
    #     fig, ax = plt.subplots(1, 1, sharex=True)
    #     unit = VAR.var_units[var]
    #
    #     for i, wmo in enumerate(WMOS()):
    #
    #         prc = PRC(wmo)
    #
    #         if var in list(prc.data_vars):
    #
    #             dmax = np.nanmean(prc.CDARK_DEPTH.values)
    #             vals = np.where(prc.PRES.values * VAR.hf > dmax, np.nan, prc[var].values)
    #             vals = vals[~np.isnan(vals)].flatten()
    #             ax.boxplot(vals, positions=[i], widths=0.7, showfliers=False,
    #                        medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    #     bar_pos = (np.arange(len(WMOS())))
    #     ax.set_xticks(bar_pos)
    #     ax.set_xticklabels([str(e)[-4:] for e in WMOS()], fontsize=13)
    #     ax.grid(zorder=0)
    #     ax.set_ylabel('{} ({})'.format(var, unit))
    #     fig.suptitle('Float {} values (prof$_{{max}} = Dark_{{mean}}$)'.format(var))
    #     fig.tight_layout()
    #     # fig.savefig(VAR.figpath+'{}_floatvalues_boxplot.png'.format(var))


    ############ ArgoClimato MLD boxplots


    # nc = xr.open_dataset('Files/Argo/IPRC_MLD_SP_2015to2019.nc')
    # lon, lat = nc.LON141_295.values, nc.LAT31_100.values
    # vals = np.nanmean(nc.MLD.values, axis=0)
    #
    # fig, ax = plt.subplots(1, 1, sharex=True)
    #
    # for i, wmo in enumerate(WMOS()):
    #
    #     prof = PRC(wmo)
    #     traj = np.vstack([prof.LONGITUDE.values, prof.LATITUDE.values])
    #     for k in range(np.shape(traj)[1]):
    #         if traj[0, k] < 0.:
    #             traj[0, k] += 360.
    #
    #     boxvals = []
    #     for k in range(np.shape(traj)[1]):
    #         lon_i, lat_i = np.argmin(np.abs(lon - traj[0, k])), np.argmin(np.abs(lat - traj[1, k]))
    #         boxvals.append(vals[lat_i, lon_i])
    #
    #     boxvals = np.array(boxvals).flatten()
    #     boxvals = boxvals[~np.isnan(boxvals)]
    #     ax.boxplot(boxvals, positions=[i], widths=0.7, showfliers=False,
    #                medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    # bar_pos = (np.arange(len(WMOS())))
    # ax.set_xticks(bar_pos)
    # ax.set_xticklabels([str(e)[-4:] for e in WMOS()], fontsize=13)
    # ax.grid(zorder=0)
    # ax.set_xlabel('Float WMO')
    # ax.set_ylabel('MLD ($m$)')
    # fig.suptitle('Argo MLD 2015-2019 average along floats trajectories')
    # fig.tight_layout()
    #
    # fig.savefig(VAR.figpath + 'Climatologies/ArgoClimato_2015to2020_MLD_boxplot.png')



    ########### Glodap v2 boxplots at surface

    # units = ['$PSU$', '$^{\circ}C$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$', '$\mu mol.kg^{-1}$']
    # vars = ['salinity', 'theta', 'phosphate', 'silicate', 'nitrate', 'oxygen']
    # wmo_list = WMOS()
    #
    # for k, var in enumerate(vars):
    #
    #     unit = units[k]
    #     nc = xr.open_dataset('Files/Climatologies/GLODAPv2.{}.nc'.format(var))
    #     lon, lat = nc.lon.values, nc.lat.values
    #     vals = nc[var].values
    #     std = np.nanstd(vals)
    #     mask = nc[var + '_error'].values > std
    #     vals = np.where(mask, np.nan, vals)
    #
    #     fig, ax = plt.subplots(1, 1, sharex=True)
    #
    #     for i, wmo in enumerate(wmo_list):
    #
    #         prof = PRC(wmo)
    #         traj = np.vstack([prof.LONGITUDE.values, prof.LATITUDE.values])
    #
    #         boxvals = []
    #         for k in range(np.shape(traj)[1]):
    #             lon_i, lat_i = np.argmin(np.abs(lon - traj[0, k])), np.argmin(np.abs(lat - traj[1, k]))
    #             boxvals.append(vals[0, lat_i, lon_i])
    #
    #         boxvals = np.array(boxvals).flatten()
    #         boxvals = boxvals[~np.isnan(boxvals)]
    #         ax.boxplot(boxvals, positions=[i], widths=0.7, showfliers=False,
    #                    medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    #     bar_pos = (np.arange(len(wmo_list)))
    #     ax.set_xticks(bar_pos)
    #     ax.set_xticklabels([str(e)[-4:] for e in wmo_list], fontsize=13)
    #     ax.grid(zorder=0)
    #     ax.set_xlabel('Float WMO')
    #     ax.set_ylabel('{} ({})'.format(var, unit))
    #     fig.suptitle('Glodap {} along floats trajectories at surface'.format(var))
    #     fig.tight_layout()
    #
    #     fig.savefig(VAR.figpath + 'Climatologies/glodapv2_{}_boxplot_ontrajectory_surface_masked.png'.format(var))


    ########### ArgoClimato boxplots at surface


    # units = ['$PSU$', '$^{\circ}C$']
    # vars = ['SALT', 'PTEMP']
    # wmo_list = WMOS()
    #
    # for k, var in enumerate(vars):
    #
    #     unit = units[k]
    #     nc = xr.open_dataset('Files/Argo/IPRC_{}_SP_2015to2019.nc'.format(var))
    #     lon_name, lat_name = np.array(nc.variables)[['LON' in v for v in list(nc.variables)]][0],\
    #                          np.array(nc.variables)[['LAT' in v for v in list(nc.variables)]][0]
    #     lon, lat = nc[lon_name].values, nc[lat_name].values
    #     vals = np.nanmean(nc[var].values, axis=0)
    #
    #     fig, ax = plt.subplots(1, 1, sharex=True)
    #
    #     for i, wmo in enumerate(wmo_list):
    #
    #         prof = PRC(wmo)
    #         traj = np.vstack([prof.LONGITUDE.values, prof.LATITUDE.values])
    #         for k in range(np.shape(traj)[1]):
    #             if traj[0, k] < 0.:
    #                 traj[0, k] += 360.
    #
    #         boxvals = []
    #         for k in range(np.shape(traj)[1]):
    #             lon_i, lat_i = np.argmin(np.abs(lon - traj[0, k])), np.argmin(np.abs(lat - traj[1, k]))
    #             boxvals.append(vals[0, lat_i, lon_i])
    #
    #         boxvals = np.array(boxvals).flatten()
    #         boxvals = boxvals[~np.isnan(boxvals)]
    #         ax.boxplot(boxvals, positions=[i], widths=0.7, showfliers=False,
    #                    medianprops=dict(linewidth=3., color=VAR.cluster_color[wmo], zorder=0))
    #
    #     bar_pos = (np.arange(len(wmo_list)))
    #     ax.set_xticks(bar_pos)
    #     ax.set_xticklabels([str(e)[-4:] for e in wmo_list], fontsize=13)
    #     ax.grid(zorder=0)
    #     ax.set_xlabel('Float WMO')
    #     ax.set_ylabel('{} ({})'.format(var, unit))
    #     fig.suptitle('Argo {} along floats trajectories at surface'.format(var))
    #     fig.tight_layout()
    #
    #     fig.savefig(VAR.figpath + 'Climatologies/ArgoClimato_{}_boxplot_ontrajectory_surface_masked.png'.format(var))


    ########## BBP700 and CHLA_PRC correlation


    # prof = PRC(2701)
    # corr = []
    # for i in range(np.size(prof.N_PROF.values)):
    #     r = np.nan
    #     A, B = prof.CHLA_PRC.values[i], prof.BBP700.values[i]
    #     A, B = A[~np.isnan(A)], B[~np.isnan(A)]
    #     A, B = A[~np.isnan(B)], B[~np.isnan(B)]
    #     if not (np.size(A) + np.size(B) == 0):
    #         reginit = stats.linregress(A, B)
    #         r = reginit.rvalue
    #     corr.append(r)
    # plt.plot(prof.JULD.values, corr, label='Raw', c='k', linewidth=0.7)
    # plt.plot(prof.JULD.values, FLT_mean(corr, prof.JULD.values, smooth=50.), label='Smoothed (50d)', linewidth=2.,
    #          c='purple')
    # plt.grid()
    # plt.legend()
    # plt.xlabel('Date')
    # plt.ylabel('Correlation')
    # plt.suptitle('Float \#{} Chl-A/$b_{{bp700nm}}$ correlation'.format(int(prof.PLATFORM_NUMBER.values[0])))
    # plt.tight_layout()


    ########### Transects

    # vars = ['CHLA_PRC', 'BBP700', 'SIG0']
    # isolines = [['MLD_S03', 'MLD_S125', 'ISO15'], ['BVFMAX_DEPTH', 'ZEU'], ['SCM']]
    #
    # for wmo in WMOS():
    #
    #     prof = PRC(wmo)
    #     activity = GET_activity(prof, 'CHLA_PRC')
    #     OV_floatvars(PRC(wmo), vars, isolines=isolines, period=activity)
    #
    #     title = ''
    #     for var in vars:
    #         title += var+'_'
    #     title = title[:-1]
    #
    #     plt.gcf().savefig(VAR.figpath + str(wmo) + '/{}_{}.png'.format(title, wmo))

    ########### Wide Marquesas chl climato map

    # nc = xr.open_dataset('Files/MODIS/AQUA_MODIS.20180101_20181231.L3m.YR.CHL.chlor_a.9km.nc')
    # lon, lat = nc.lon.values, nc.lat.values
    # vals = nc.chlor_a.values
    # ax = CHART(zone_id='MA_w')
    # MAP_grid(lon, lat, vals, cmap='viridis', vmin=0., vmax=0.5, extend='max', ax=ax, cbar_label='SSC ($mg.m^{-3}$)')
    # PT_floats(WMOS(), ax=ax)
    # ax.set_title('Floats trajectories around Marquesas archipelago islands')
    #
    # plt.gcf().savefig('MA_w.png')

    ########### Vertical correlation of Chl-a and Bbp

    # for wmo in WMOS():
    #
    #     prof = PRC(wmo)
    #     Zeu = []
    #     corr = []
    #     for i in range(np.size(prof.N_PROF.values)):
    #
    #         r = np.nan
    #         Zeu_pres = np.nan
    #
    #         par, pres = prof.DOWNWELLING_PAR_FIT.values[i], prof.PRES.values[i]
    #         par, pres = par[~np.isnan(pres)], pres[~np.isnan(pres)]
    #         par, pres = par[~np.isnan(par)], pres[~np.isnan(par)]
    #
    #         if not (np.size(par) + np.size(pres) == 0):
    #
    #             PARsurf = np.nanmax(par)
    #
    #             if np.size(pres[par < PARsurf/100]) > 0:
    #
    #                 Zeu_pres = np.min(pres[par < PARsurf/100])
    #
    #                 A, B, pres = prof.CHLA_PRC.values[i], prof.BBP700.values[i], prof.PRES.values[i]
    #                 A, B, pres = A[~np.isnan(A)], B[~np.isnan(A)], pres[~np.isnan(A)]
    #                 A, B, pres = A[~np.isnan(B)], B[~np.isnan(B)], pres[~np.isnan(B)]
    #
    #                 A, B = A[pres < 2*Zeu_pres], B[pres < 2*Zeu_pres]
    #
    #                 if not (np.size(A) + np.size(B) == 0):
    #                     reginit = stats.linregress(A, B)
    #                     r = reginit.rvalue
    #
    #         Zeu.append(Zeu_pres)
    #         corr.append(r)
    #
    #     fig, ax = plt.subplots()
    #     ax2 = ax.twinx()
    #
    #     ax.plot(prof.JULD.values, corr, label='Raw', c=(0., 0., 0., 0.4), linewidth=0.7)
    #     ax.plot(prof.JULD.values, FLT_mean(corr, prof.JULD.values, smooth=50.), label='Smoothed (50d)', linewidth=2.,
    #             c='purple')
    #     xlim = ax.get_xlim()
    #     ax.plot(prof.JULD.values[:2], FLT_mean(corr, prof.JULD.values, smooth=50.)[:2], label='$Z_{eu}$', c='orange')
    #     ax2.plot(prof.JULD.values, Zeu, c='orange')
    #     ax.set_xlim(xlim)
    #     ax.grid()
    #     ax.legend(loc='upper left')
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('Correlation')
    #     ax2.set_ylabel('$Z_{eu}$ ($m$)')
    #     ax.set_title('{} Chl-A/$b_{{bp700nm}}$ vertical correlation (0-$Z_{{eu}}$)'.format(VAR.float_names[wmo]))
    #
    #     plt.tight_layout()
    #     fig.savefig(VAR.figpath +'{}/{}_CHL-A_Bbp_corr.png'.format(wmo, VAR.float_names[wmo]))

    ########## Coloured trajectories

    # for wmo in WMOS():
    #     zone_id = VAR.float_names[wmo]
    #     MAP_timetraj(wmo, zone_id)
    #     plt.gcf().savefig(VAR.figpath +'{}/{}_trajcolor.png'.format(wmo, VAR.float_names[wmo]))


    ########## Coloured trajectories by cluster


    # for zone_id in ['TS', 'CS', 'FJ_w', 'OL', 'MA_w']:
    #     cluster = zone_id[0]
    #     wmos = VAR.clusters[cluster]
    #     MAP_timetraj(wmos, zone_id=zone_id)
    #     plt.gcf().savefig(VAR.figpath + '{}_trajcolor.png'.format(cluster))


    ########## Timelines and MEI

    # for var in ['PSAL', 'TEMP', 'SIG0', 'CHLA', 'BBP700', 'DOWNWELLING_PAR', 'DOXY']:
    #     TL_var(var)
    #     plt.gcf().savefig(VAR.figpath+'{}_timeline_MEI.png'.format(var))

    ########## Map of floats with MODIS chlorophyll

    # nc2015 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20150101_20151231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2016 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20160101_20161231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2017 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20170101_20171231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2018 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20180101_20181231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2019 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20190101_20191231.L3m.YR.CHL.chlor_a.9km.nc')
    # nc2020 = xr.open_dataset('Files/MODIS/AQUA_MODIS.20200101_20201231.L3m.YR.CHL.chlor_a.9km.nc')
    #
    # lon, lat = nc2015.lon.values, nc2015.lat.values
    # vals = np.mean(np.stack([nc2015.chlor_a.values, nc2016.chlor_a.values, nc2017.chlor_a.values, nc2018.chlor_a.values,
    #                          nc2019.chlor_a.values, nc2020.chlor_a.values], axis=0), axis=0)
    #
    # ax = CHART()
    # MAP_grid(lon, lat, vals, vmin=0., vmax=0.25, cmap='viridis', extend='max', ax=ax,
    #          cbar_label='2015-2020 MODIS average chlorophyll-A ($mg.m^{-3}$)')
    # for c in VAR.clusters:
    #     PT_floats(VAR.clusters[c], ax=ax, c=VAR.cluster_colors[c])
    # ax.set_title('Last position of BGC Argo floats having PAR sensors of the South Pacific')
    # plt.gcf().savefig(VAR.figpath+'map.png')


    ####### Quality indexes comparison


    # QI125, QI03 = [], []
    # for wmo in WMOS():
    #     prof = PRC(wmo)
    #     QI125.append(np.nanmedian(prof.MLD_S125_QI.values))
    #     QI03.append(np.nanmedian(prof.MLD_S03_QI.values))
    # plt.scatter(np.arange(len(WMOS())), QI125, color='green', label='$QI_{125}$')
    # plt.scatter(np.arange(len(WMOS())), QI03, color='purple', label='$QI_{03}$')
    # plt.xlabel('Float names')
    # plt.ylabel('Quality indexes')
    # plt.gca().set_xticks(np.arange(len(WMOS())))
    # plt.gca().set_xticklabels([VAR.float_names[wmo] for wmo in WMOS()])
    # plt.grid()
    # plt.legend()
    # plt.title('Quality indexes for both MLD criteria ($\Delta \sigma = 0.03$ and $0.125 kg.m^{-3}$)')


    # Comparing with Argo Delayed Mode


    # wmo = FMT_wmo(1656)
    # prof = PRC(wmo)
    #
    # for i in range(1, np.size(prof.N_PROF.values)):
    #     try:
    #
    #         profdm = xr.open_dataset('Files/Dmode/{}/profiles/BD{}_{:03d}.nc'.format(wmo, wmo, i))
    #         date = FMT_date(profdm.JULD.values[0], 'dt', verbose=False)
    #         dates = FMT_date(prof.JULD.values, 'dt', verbose=False)
    #
    #         if np.min(np.abs(dates - date)) == dt.timedelta(0.):
    #
    #             k = np.argmin(np.abs(dates - date))
    #             rap = np.interp(np.max([prof.MLD_S125.values[k], prof.ISO15.values[k]]),
    #                             prof.PRES.values[k][~np.isnan(prof.CHLA_PRC.values[k])],
    #                             prof.CHLA_PRC.values[k][~np.isnan(prof.CHLA_PRC.values[k])])/\
    #                   np.interp(np.max([prof.MLD_S125.values[k], prof.ISO15.values[k]]),
    #                             profdm.PRES.values[~np.isnan(profdm.CHLA_ADJUSTED.values)],
    #                             profdm.CHLA_ADJUSTED.values[~np.isnan(profdm.CHLA_ADJUSTED.values)])
    #
    #             fig, ax = plt.subplots()
    #             ax.set_ylim(200., 0.)
    #             ax.set_title('Correction comparison (float {}, {})'
    #                          .format(VAR.float_names[wmo], date.strftime('%m/%d/%Y'))+
    #                          '\nSlope factors: {:.2f} and {:.2f} (resp. Argo DM and Processed)'
    #                          .format(1/float(prof.SLOPEF490.median())*rap, 1/float(prof.SLOPEF490.median())))
    #             ax.set_xlabel('Normalized Chlorophyll-A ($mg.m^{-3}$)')
    #             ax.set_ylabel('Pressure ($dbar$)')
    #
    #             ax.scatter(profdm.CHLA_ADJUSTED.values/np.nanmax(profdm.CHLA_ADJUSTED.values), profdm.PRES.values,
    #                        c='turquoise', s=15, label='Argo delayed mode')
    #             ax.scatter(prof.CHLA_PRC.values[k]/np.nanmax(prof.CHLA_PRC.values[k]), prof.PRES.values[k],
    #                        c='darkblue', s=25, label='ArgoData correction')
    #
    #             xlims = ax.get_xlim()
    #             ax.plot(xlims, [prof.MLD_S03.values[k], prof.MLD_S03.values[k]], c='k', linestyle='dashed',
    #                     label='MLD $\\Delta \\sigma = 0.03kg.m^{-3}$')
    #             ax.plot(xlims, [prof.MLD_S125.values[k], prof.MLD_S125.values[k]], c='k',
    #                     label='MLD $\\Delta \\sigma = 0.125kg.m^{-3}$')
    #             ax.plot(xlims, [prof.ISO15.values[k], prof.ISO15.values[k]], c='orange', linestyle='dashed',
    #                     label='Isolume$_{15}$')
    #
    #             ax.legend()
    #
    #             fig.savefig(VAR.figpath+'Comp_DM/{}_{}.png'.format(wmo, k))
    #
    #     except Exception as e:
    #         print(e)

    # Another way of plotting the transects

    # prof = PRC(2701)
    # chla = INT_1D(prof.CHLA_PRC.values, prof.PRES.values, np.linspace(0., 300., 500))
    # pres = np.linspace(0., 300., 500)
    # pres = np.vstack([pres for _ in range(np.shape(chla)[0])])
    # time = FMT_date(prof.JULD.values, 'dt', verbose=False)
    # time = time[np.newaxis].T
    # time = np.hstack([time for _ in range(np.shape(chla)[1])])
    # fig, ax = plt.subplots()
    # # ax.pcolormesh(time, pres, chla, vmin=0., vmax=0.5)
    # contour = ax.contourf(time, pres, chla, levels=np.linspace(0., .5, 21), cmap='jet', extend='max')
    # ax.set_ylim(300., 0.)
    # plt.plot(time[:, 0], prof.ISO15.values, c='white', linewidth=1.5)
    # plt.plot(time[:, 0], prof.ISO15.values, c='k', linewidth=1.)
    # plt.plot(time[:, 0], prof.MLD_S125.values, c='white', linewidth=1.5)
    # plt.plot(time[:, 0], prof.MLD_S125.values, c='k', linewidth=1., linestyle='dashed')
    # plt.colorbar(contour)


    # Boxplots for CHLA_PRC


    # chla_series = {}
    # depths = {}
    # for wmo in WMOS():
    #     serie = []
    #     prof = PRC(wmo)
    #     for k in range(np.size(prof.N_PROF.values)):
    #         if np.sum(~np.isnan(prof.CHLA_PRC.values[k])) > 10:
    #             # serie.append(np.nanmean(np.where(prof.PRES.values[k] < 5., prof.CHLA_PRC.values[k], np.nan)))
    #             serie.append(prof.CHLA_PRC.values[k][~np.isnan(prof.CHLA_PRC.values[k])][0])
    #     chla_series[wmo] = np.array(serie)[~np.isnan(serie)]
    #
    # fig, ax = plt.subplots()
    # for k, wmo in enumerate(chla_series):
    #     ax.boxplot(chla_series[wmo], positions=[k], widths=0.7, showfliers=False,
    #                medianprops=dict(linewidth=3., color=VAR.float_colors[wmo], zorder=0))
    #
    # ax.set_xticks(np.arange(np.size(WMOS())))
    # ax.set_xticklabels([VAR.float_names[w] for w in WMOS()], rotation=45)
    # ax.set_ylim(-0.05, 0.6)
    # ax.set_title('Surface CHLA over float lifetimes')
    # ax.set_ylabel(VAR.var_labels['CHLA_PRC'])
    # ax.set_xlabel('Float ID')


    # MODIS chla along floats trajectories boxplots


    # import pickle
    #
    # with open(VAR.indexpath + 'MODIS_ssc_series_along_floats_trajectories.pkl', 'rb') as f:
    #     series = pickle.load(f)
    #
    # depths = {}
    # for wmo in WMOS():
    #     serie = series[wmo]
    #
    # fig, ax = plt.subplots()
    # for k, wmo in enumerate(series):
    #     ax.boxplot(series[wmo], positions=[k], widths=0.7, showfliers=False,
    #                medianprops=dict(linewidth=3., color=VAR.float_colors[wmo], zorder=0))
    #
    # ax.set_xticks(np.arange(np.size(WMOS())))
    # ax.set_xticklabels([VAR.float_names[w] for w in WMOS()], rotation=45)
    # ax.set_ylim(-0.05, 0.6)
    # ax.set_title('MODIS surface CHLA along floats trajectories')
    # ax.set_ylabel('Chl-a ($mg.m^{-3}$)')
    # ax.set_xlabel('Float ID')
    # ax.grid()


    # MODIS chla along floats trajectories curve plots, and correlation plots


    # for wmo in WMOS():
    #
    #     prof = PRC(wmo)
    #
    #     # Float serie
    #     floatserie = np.nan * np.zeros(np.size(prof.N_PROF.values))
    #     for k in range(np.size(prof.N_PROF.values)):
    #         if np.sum(prof.PRES.values[k] * VAR.hf < prof.ZEU.values[k]/4.6) > 10:
    #             floatserie[k] = np.nanmean(prof.CHLA_PRC.values[k][prof.PRES.values[k] * VAR.hf <
    #                                                                prof.ZEU.values[k]/4.6])
    #
    #     # Sat serie
    #
    #     files = []
    #     for f in os.listdir(VAR.chloropath):
    #         if 'AQUA_MODIS.' in f and '.L3m.DAY.CHL.chlor_a.9km.nc' in f:
    #             files.append(f)
    #     mapdates = np.array([dt.datetime(int(f[11:15]), int(f[15:17]), int(f[17:19])) for f in files])
    #
    #     llt = np.array([[prof.LONGITUDE.values[k] if prof.LONGITUDE.values[k] >= 0. else prof.LONGITUDE.values[k] + 360.,
    #                      prof.LATITUDE.values[k], FMT_date(prof.JULD.values[k], 'dt', verbose=False)]
    #                     for k in range(np.size(prof.N_PROF.values))])
    #
    #     k = 0
    #     while not np.isnan(prof.CHLA_PRC.values[k:]).all():
    #         k += 1
    #     llt = llt[:k]
    #     floatserie = floatserie[:k]
    #
    #     satvals = np.nan * np.zeros(np.shape(llt)[0])
    #     dates = []
    #
    #     for k in range(len(llt)):
    #
    #         print(colored('Looking for satellite measurements along float {} trajectory'.format(VAR.float_names[wmo])
    #                       + LOADOTS(k), 'blue'), end='\r')
    #
    #         tol_lon = 10 * 9000 * 360 / (2 * np.pi * VAR.Rt * np.cos(llt[k, 1] * np.pi / 180))
    #         tol_lat = 10 * 9000 * 360 / (2 * np.pi * VAR.Rt)
    #         lonmin, lonmax = llt[k, 0] - tol_lon, llt[k, 0] + tol_lon
    #         latmin, latmax = llt[k, 1] - tol_lat, llt[k, 1] + tol_lat
    #
    #         map_i = np.argmin([np.abs(md - llt[k][2]).days for md in mapdates])
    #
    #         if np.abs(mapdates[map_i] - llt[k][2]).days < 10.:
    #
    #             map = xr.open_dataset(VAR.chloropath + files[map_i])
    #             lon, lat, chla = map.lon.values, map.lat.values, map.chlor_a.values
    #
    #             lon = np.where(lon < 0., lon + 360., lon)
    #             lon, lat = np.meshgrid(lon, lat)
    #             binned = np.vstack([lon.flatten(), lat.flatten(), chla.flatten()])[:, ~np.isnan(chla.flatten())]
    #
    #             binned = binned[:, binned[0] < lonmax]
    #             binned = binned[:, binned[0] > lonmin]
    #             binned = binned[:, binned[1] < latmax]
    #             binned = binned[:, binned[1] > latmin]
    #
    #             if np.size(binned) > 0:
    #
    #                 satvals[k] = interpolate.griddata(binned[:2].T, binned[2], [llt[k, :2]], method='nearest')
    #                 dates.append(mapdates[map_i])
    #
    #     tvals = llt[:, 2] - llt[0, 2]
    #     tvals = np.array([ti.days for ti in tvals])
    #     mask = ~np.isnan(satvals)
    #     satvals = np.interp(tvals, tvals[mask], satvals[mask])
    #
    #     T = np.array([dt.timedelta(float(ti)) + llt[0, 2] for ti in tvals])
    #
    #     fig, ax = plt.subplots()
    #     ax.plot(T[~np.isnan(floatserie)], floatserie[~np.isnan(floatserie)], c='k', linewidth=2., linestyle='dashed',
    #             label='Float surface Chl-a')
    #     ax.plot(T[~np.isnan(satvals)], satvals[~np.isnan(satvals)], c='orange')
    #     ax.scatter(T[mask], satvals[mask], c='darkblue', label='Found')
    #     ax.scatter(T[~mask], satvals[~mask], c='red', label='Interpolated (nearest)')
    #     ax.legend()
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('Chlorophyll ($mg.m^-{3}$)')
    #     ax.set_title('MODIS chl-a serie along float {} trajectory and {} mean chl-a 0m to optic depth'
    #                  .format(VAR.float_names[wmo], VAR.float_names[wmo]))
    #
    #     fig.savefig(VAR.figpath+'{}/MODIS_ssc_timeserie_along_{}.png'.format(wmo, VAR.float_names[wmo]))
    #
    #
    #     fig, ax = plt.subplots()
    #     reg = stats.linregress(floatserie[mask][~np.isnan(floatserie[mask])],
    #                            satvals[mask][~np.isnan(floatserie[mask])])
    #     r2, i, s = reg.rvalue**2, reg.intercept, reg.slope
    #     scatter = ax.scatter(floatserie[mask], satvals[mask], c=np.array([ti.month for ti in T])[mask], cmap='hsv')
    #     cbar = plt.colorbar(scatter)
    #     cbar.set_ticks(range(1, 13))
    #     cbar.set_ticklabels([dt.datetime(2000, i, 1).strftime('%B') for i in range(1, 13)])
    #     ax.plot(ax.get_xlim(), ax.get_xlim(), zorder=0, c=[0., 0., 0., 0.8], linestyle='dashed', linewidth=0.7,
    #             label='Identity')
    #     ax.plot(ax.get_xlim(), np.array(ax.get_xlim())*s + i, label='Regression', c='k')
    #     ax.legend()
    #     ax.grid()
    #     ax.set_aspect(1.)
    #     ax.set_xlabel('Float surface Chl-a ($mg.m^-{3}$)')
    #     ax.set_ylabel('MODIS surface Chl-a ($mg.m^-{3}$)')
    #     ax.set_title('Linear regression between MODIS and {} float Chl-a ($r^{{2}} = {:.2f}$)'
    #                  .format(VAR.float_names[wmo], r2))
    #
    #     fig.savefig(VAR.figpath+'{}/MODISvsFloat_regression_{}.png'.format(wmo, VAR.float_names[wmo]))


    # Checking day and night profiles


    # for wmo in WMOS():
    #
    #     prof = RAW(wmo)
    #     localtimes = CMP_localtime(prof, verbose=False)
    #     times = FMT_date(prof.JULD.values, 'dt', verbose=False)
    #     dtimes = times[1:] - times[:-1]
    #
    #     print(str(wmo)+' :')
    #
    #     for k in range(np.size(dtimes)):
    #
    #         if dt.timedelta(seconds=3600 * 4) < dtimes[k] < dt.timedelta(seconds=3600 * 14):
    #
    #             print('Profile #{}: {}/{}'.format(prof.N_PROF.values[k], localtimes[k], localtimes[k + 1]))
    #
    #             profprc = PRC(wmo)
    #
    #             if prof.N_PROF.values[k] in profprc.N_PROF.values or prof.N_PROF.values[k+1] in profprc.N_PROF.values:
    #
    #                 fig, ax = plt.subplots()
    #                 ax.set_ylim(300., 0.)
    #                 ax.set_title('Profile \#{}: {}/{}'.format(prof.N_PROF.values[k], localtimes[k], localtimes[k + 1]))
    #
    #                 try:
    #
    #                     i = int(profprc.N_PROF.values[profprc.N_PROF.values == prof.N_PROF.values[k]][0])
    #                     ax.plot(prof.CHLA.values[k+1][~np.isnan(prof.CHLA.values[k+1])]
    #                             * float(np.nanmedian(profprc.SLOPEF490.values)),
    #                             prof.PRES.values[k+1][~np.isnan(prof.CHLA.values[k+1])], c='green', label='Night')
    #
    #                 except Exception:
    #
    #                     i = int(profprc.N_PROF.values[profprc.N_PROF.values == prof.N_PROF.values[k+1]][0])
    #                     ax.plot(prof.CHLA.values[k][~np.isnan(prof.CHLA.values[k])]
    #                             * float(np.nanmedian(profprc.SLOPEF490.values)),
    #                             prof.PRES.values[k][~np.isnan(prof.CHLA.values[k])], c='darkblue', label='Night')
    #
    #                 ax.plot(profprc.CHLA_ZERO.values[i][~np.isnan(profprc.CHLA_ZERO.values[i])]
    #                         * float(np.nanmedian(profprc.SLOPEF490.values)),
    #                         profprc.PRES.values[i][~np.isnan(profprc.CHLA_ZERO.values[i])], c=[0., 0., 0., 0.7],
    #                         label='Raw day', linewidth=0.5)
    #                 ax.plot(profprc.CHLA_PRC.values[i][~np.isnan(profprc.CHLA_PRC.values[i])],
    #                         profprc.PRES.values[i][~np.isnan(profprc.CHLA_PRC.values[i])], c='k', label='Day')
    #                 ax.set_xlim(ax.get_xlim())
    #                 ax.plot(ax.get_xlim(), [profprc.MLD_S03.values[i], profprc.MLD_S03.values[i]], linestyle='dashed',
    #                         label='MLD_S03')
    #                 ax.plot(ax.get_xlim(), [profprc.ISO15.values[i], profprc.ISO15.values[i]], linestyle='dashed',
    #                         c='orange', label='Isolume 15')
    #                 axb = ax.twiny()
    #                 axb.plot(profprc.DOWNWELLING_PAR_FIT.values[i][~np.isnan(profprc.DOWNWELLING_PAR_FIT.values[i])],
    #                         profprc.PRES.values[i][~np.isnan(profprc.DOWNWELLING_PAR_FIT.values[i])],
    #                          c='purple', label='Downwelling PAR')
    #                 ax.plot([-1000, 1000], [5999, 5999], c='purple', label='Downwelling PAR')
    #                 ax.legend()


    # Histogram and scatter of percentage of SCM through a year with different SCM thresholds


    # for thresh in [1., 1.25, 1.5, 1.75, 2.]:
    #
    #     print(thresh)
    #
    #     fig, ax = plt.subplots(figsize=(15., 5.))
    #
    #     fig.suptitle('Percentage of SCM per year ($chla_{{max}} > {} \\times chla_{{surf}}$)'.format(thresh))
    #     ax.set_xlim(-1., len(WMOS()))
    #     ax.set_xticks(np.arange(0., len(WMOS())))
    #     ax.set_xticklabels([VAR.float_names[wmo] for wmo in WMOS()])
    #     ax.set_yticks(np.arange(0., 101., 10.))
    #     ax.set_xlabel('Float names')
    #     ax.set_ylabel('Yearly percentage (\%)')
    #     ax.grid(axis='y')
    #     width = 0.3
    #
    #     for k, wmo in enumerate(WMOS()):
    #
    #         pdcm = CMP_pSCM(wmo, thresh=thresh)
    #
    #         if pdcm is not None:
    #
    #             if k == 0.:
    #
    #                 ax.add_patch(Rectangle((k-width, 0), width, pdcm, facecolor='darkblue', label='SCM exists', zorder=3))
    #                 ax.add_patch(Rectangle((k, 0), width, 100-pdcm, facecolor='crimson', label='No SCM', zorder=3))
    #
    #             else:
    #
    #                 ax.add_patch(Rectangle((k - width, 0), width, pdcm, facecolor='darkblue', zorder=3))
    #                 ax.add_patch(Rectangle((k, 0), width, 100 - pdcm, facecolor='crimson', zorder=3))
    #
    #         else:
    #
    #             ax.text(k - width/2, 10., 'less than 1 year of data', fontsize=20., rotation=90.)
    #
    #     ax.legend(loc='upper left')
    #     fig.tight_layout()
    #     fig.savefig(VAR.figpath+'pcSCM_hist_thresh={}.png'.format(thresh))
    #
    #     fig, ax = plt.subplots(figsize=(4., 7.))
    #
    #     fig.suptitle('Percentage of SCM per year\n($chla_{{max}} > {} \\times chla_{{surf}}$)'.format(thresh))
    #     ax.set_xlim(-1., 1.)
    #     ax.set_xticks([])
    #     ax.set_yticks(np.arange(0., 101., 10.))
    #     ax.set_ylabel('Yearly percentage (\%)')
    #     done = []
    #
    #     for wmo in WMOS():
    #
    #         pdcm = CMP_pSCM(wmo, thresh=thresh)
    #
    #         if VAR.float_clusters[wmo] not in done:
    #
    #             done.append(VAR.float_clusters[wmo])
    #             ax.scatter(0., pdcm, color=list(mpl.colors.to_rgb(VAR.float_colors[wmo]))+[0.6], s=400, zorder=3,
    #                        label=VAR.float_clusters[wmo])
    #
    #         else:
    #
    #             ax.scatter(0., pdcm, color=list(mpl.colors.to_rgb(VAR.float_colors[wmo]))+[0.6], s=400, zorder=3)
    #
    #     ax.grid()
    #     ax.legend(labelspacing=2., borderpad=2.)
    #     fig.tight_layout()
    #     fig.savefig(VAR.figpath+'pcSCM_scat_thresh={}.png'.format(thresh))


    # Histogram and scatter of percentage of SCM through a year from shape classification


    # fig, ax = plt.subplots(figsize=(15., 5.))
    #
    # fig.suptitle('Percentage of SCM per year (Gaussian vs sigmoid classification)')
    # ax.set_xlim(-1., len(WMOS()))
    # ax.set_xticks(np.arange(0., len(WMOS())))
    # ax.set_xticklabels([VAR.float_names[wmo] for wmo in WMOS()])
    # ax.set_yticks(np.arange(0., 101., 10.))
    # ax.set_xlabel('Float names')
    # ax.set_ylabel('Yearly percentage (\%)')
    # ax.grid(axis='y')
    # width = 0.25
    #
    # for k, wmo in enumerate(WMOS()):
    #
    #     print(wmo)
    #     ptype = CMP_ptype(wmo)
    #
    #     if ptype is not None:
    #
    #         if k == 0.:
    #
    #             ax.add_patch(Rectangle((k - 1.5*width, 0.), width, ptype[0], facecolor='darkblue', label='SCM',
    #                                    zorder=3))
    #             ax.add_patch(Rectangle((k - 0.5*width, 0.), width, ptype[1], facecolor='crimson', label='SGM',
    #                                    zorder=3))
    #             ax.add_patch(Rectangle((k + 0.5*width, 0.), width, ptype[2], facecolor='g', label='Other',
    #                                    zorder=3))
    #
    #         else:
    #
    #             ax.add_patch(Rectangle((k - 1.5*width, 0.), width, ptype[0], facecolor='darkblue', zorder=3))
    #             ax.add_patch(Rectangle((k - 0.5*width, 0.), width, ptype[1], facecolor='crimson', zorder=3))
    #             ax.add_patch(Rectangle((k + 0.5*width, 0.), width, ptype[2], facecolor='g', zorder=3))
    #
    #     else:
    #
    #         ax.text(k - width/2, 10., 'less than 1 year of data', fontsize=20., rotation=90.)
    #
    # ax.legend(loc='upper left')
    # fig.tight_layout()
    # fig.savefig(VAR.figpath+'pcSCM_hist_thresh={}.png'.format(thresh))
    #
    # fig, ax = plt.subplots(figsize=(6., 7.))
    #
    # fig.suptitle('Percentage of SCM per year\n(Gaussian vs sigmoid classification)')
    # ax.set_xlim(-1., 1.)
    # ax.set_xticks([])
    # ax.set_yticks(np.arange(0., 101., 10.))
    # ax.set_ylabel('Yearly percentage (\%)')
    # done = []
    #
    # for wmo in WMOS():
    #
    #     print(wmo)
    #
    #     ptype = CMP_ptype(wmo)
    #
    #     if ptype is not None:
    #
    #         if VAR.float_clusters[wmo] not in done:
    #
    #             done.append(VAR.float_clusters[wmo])
    #             ax.scatter(0., ptype[0], color=list(mpl.colors.to_rgb(VAR.float_colors[wmo])) + [0.6], s=400, zorder=3,
    #                        label=VAR.float_clusters[wmo])
    #
    #         else:
    #
    #             ax.scatter(0., ptype[0], color=list(mpl.colors.to_rgb(VAR.float_colors[wmo])) + [0.6], s=400, zorder=3)
    #
    # ax.grid()
    # ax.legend(labelspacing=2., borderpad=2.)
    # fig.tight_layout()
    # fig.savefig(VAR.figpath+'pcSCM_scat_gss_vs_sgm_classif.png')


    # SCM occurences by seasons histogram


    # season = {3: 1, 4: 1, 5: 1, 6:2, 7:2, 8:2, 9:3, 10:3, 11: 3, 12:4, 1:4, 2: 4}
    # scol = {1: 'g', 2:'r', 3: 'goldenrod', 4: 'b'}
    # sname = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    # fig, ax = plt.subplots(figsize=(15., 5.))
    #
    # fig.suptitle('Percentage of SCM per season (Gaussian vs sigmoid classification)')
    # ax.set_xlim(-1., len(WMOS()))
    # ax.set_xticks(np.arange(0., len(WMOS())))
    # ax.set_xticklabels([VAR.float_names[wmo] for wmo in WMOS()])
    # ax.set_yticks(np.arange(0., 101., 10.))
    # ax.set_xlabel('Float names')
    # ax.set_ylabel('Yearly percentage (\%)')
    # ax.grid(axis='y')
    # width = 0.75
    #
    # for k, wmo in enumerate(WMOS()):
    #
    #     print(wmo)
    #
    #     prof = PRC(wmo)
    #     type = CLS_TYPE(prof).TYPE
    #     dates = FMT_date(prof.JULD.values, 'dt', verbose=False)
    #     seasons = np.array([season[day.month] for day in dates])
    #
    #     s0 = seasons[0]
    #     index_0 = 1
    #     while seasons[index_0] == s0 and index_0 < np.size(seasons):
    #         index_0 += 1
    #
    #     s_1 = seasons[-1]
    #     index_1 = -1
    #     while seasons[index_1] == s_1 and index_1 > - np.size(seasons) + 1:
    #         index_1 -= 1
    #
    #     if index_0 < np.size(seasons) + index_1:
    #
    #         type, seasons, dates = type[index_0:index_1], seasons[index_0:index_1], dates[index_0:index_1]
    #
    #         if 1 in seasons and 2 in seasons and 3 in seasons and 4 in seasons:
    #
    #             type, seasons, dates = type[~np.isnan(type)], seasons[~np.isnan(type)], dates[~np.isnan(type)]
    #
    #             tinterp = np.arange(0., (dates[-1]-dates[0]).days, 10.)
    #             interptype = interpolate.interp1d([(d-dates[0]).days for d in dates], type, kind='nearest')
    #             type, dates = interptype(tinterp), dates[0] + np.array([dt.timedelta(days=e) for e in tinterp])
    #             seasons = np.array([season[day.month] for day in dates])
    #
    #             if k == 0.:
    #
    #                 cum = 0.
    #
    #                 for s in range(1, 5):
    #
    #                     pscm = np.nansum(type[seasons==s]==1.)/np.size(type[seasons==s]) * 100. / 4
    #                     ax.add_patch(Rectangle((k - width/2, cum), width, pscm, facecolor=scol[s], zorder=3,
    #                                            label=sname[s]))
    #                     cum += pscm
    #
    #             else:
    #
    #                 cum = 0
    #
    #                 for s in range(1, 5):
    #
    #                     pscm = np.nansum(type[seasons==s]==1.)/np.size(type[seasons==s]) * 100. / 4
    #                     ax.add_patch(Rectangle((k - width/2, cum), width, pscm, facecolor=scol[s], zorder=3))
    #                     cum += pscm
    #
    # ax.legend(loc='upper left')
    # fig.tight_layout()
    # fig.savefig(VAR.figpath+'pcSCM_seasons.png')


    # SCM occurences by year histogram


    # color_cycle = [plt.get_cmap('gist_rainbow')(1. * i / (7 - 1)) for i in range(7)]
    # ycol = dict(zip(range(2016, 2023), color_cycle))
    #
    # fig, ax = plt.subplots(figsize=(15., 5.))
    # fig.suptitle('Percentage of SCM per year (Gaussian vs sigmoid classification)')
    # ax.set_xlim(-1., len(WMOS()))
    # ax.set_xticks(np.arange(0., len(WMOS())))
    # ax.set_xticklabels([VAR.float_names[wmo] for wmo in WMOS()])
    # ax.set_yticks(np.arange(0., 101., 10.))
    # ax.set_xlabel('Float names')
    # ax.set_ylabel('Yearly percentage (\%)')
    # ax.grid(axis='y')
    # width = 0.4
    # done = []
    #
    # for k, wmo in enumerate(WMOS()):
    #
    #     print(wmo)
    #
    #     prof = PRC(wmo)
    #     type = CLS_TYPE(prof).TYPE.values
    #     dates = FMT_date(prof.JULD.values, 'dt', verbose=False)
    #     years = np.array([day.year for day in dates])
    #
    #     index_0 = 0
    #     if not (dates[0].month == 1 and dates[0].days < 11.):
    #         y0 = years[0]
    #         while years[index_0] == y0 and index_0 < np.size(years):
    #             index_0 += 1
    #
    #     index_1 = -1
    #     if not (dates[-1].month == 12 and dates.day >= 20.):
    #         y_1 = years[-1]
    #         while years[index_1] == y_1 and -index_1 < np.size(years)-1:
    #             index_1 -= 1
    #
    #     if index_0 < np.size(years) + index_1:
    #
    #         type, dates = type[index_0:index_1], dates[index_0:index_1]
    #         type, dates = type[~np.isnan(type)], dates[~np.isnan(type)]
    #
    #         interptype = interpolate.interp1d([(d-dates[0]).days for d in dates], type, kind='nearest')
    #         tinterp = np.arange(0., (dates[-1]-dates[0]).days, 10.)
    #         type, dates = interptype(tinterp), dates[0] + np.array([dt.timedelta(days=e) for e in tinterp])
    #         years = np.array([day.year for day in dates])
    #         nyears = years[-1] - years[0] + 1
    #
    #         cum = 0.
    #
    #         for y in range(years[0], years[-1] + 1):
    #
    #             pscm = np.nansum(type[years==y]==1.)/np.size(type[years==y]) * 100. / nyears
    #             if not y in done:
    #                 ax.add_patch(Rectangle((k - width/2, cum), width, pscm, facecolor=ycol[y], edgecolor='k', zorder=3,
    #                                        linewidth=0.5, label=y))
    #                 done.append(y)
    #             else:
    #                 ax.add_patch(Rectangle((k - width/2, cum), width, pscm, facecolor=ycol[y], edgecolor='k', zorder=3,
    #                                        linewidth=0.5))
    #             cum += pscm
    #
    # ax.legend(loc='upper left')
    # fig.tight_layout()
    # fig.savefig(VAR.figpath+'pcSCM_years.png')


    # SCMs widths values and depths correlation


    # for wmo in WMOS():
    #
    # prof = PRC(wmo)
    # scmfeatures = prof.SCM_GDEPTH.values, prof.SCM_GVAL.values, prof.SCM_GWIDTH.values
    #     mask = ~np.isnan(V)
    #     D, V, W = D[mask], V[mask], W[mask]
    #     fig, ax = plt.subplots(1, 3, figsize=(15., 6.5))
    #     fig.suptitle('Correlation between SCMs depths, values and widths for float {}'.format(VAR.float_names[wmo]))
    #
    #     reg = stats.linregress(D, V)
    #     s, i, r2 = reg.slope, reg.intercept, reg.rvalue ** 2
    #     ax[0].scatter(D, V, c=[e.month for e in FMT_date(prof.JULD.values, 'dt', verbose=False)[mask]], cmap='twilight')
    #     ax[0].plot(ax[0].get_xlim(), s * np.array(ax[0].get_xlim()) + i, c='k',
    #                label='Reg slope: {:.4f}'.format(reg.slope))
    #     ax[0].set_title('SCMs depths vs. values: $r^{{2}} = {:.2f}$'.format(r2))
    #     ax[0].set_xlabel('SCMs depths')
    #     ax[0].set_ylabel('SCMs chl-a')
    #     ax[0].legend()
    #
    #     reg = stats.linregress(D, W)
    #     s, i, r2 = reg.slope, reg.intercept, reg.rvalue ** 2
    #     ax[1].scatter(D, W, c=[e.month for e in FMT_date(prof.JULD.values, 'dt', verbose=False)[mask]], cmap='twilight')
    #     ax[1].plot(ax[1].get_xlim(), s * np.array(ax[1].get_xlim()) + i, c='k',
    #                label='Reg slope: {:.2f}'.format(reg.slope))
    #     ax[1].set_title('SCMs depths vs. width: $r^{{2}} = {:.2f}$'.format(r2))
    #     ax[1].set_xlabel('SCMs depths')
    #     ax[1].set_ylabel('SCMs widths')
    #     ax[1].legend()
    #
    #     reg = stats.linregress(V, W)
    #     s, i, r2 = reg.slope, reg.intercept, reg.rvalue ** 2
    #     scat = ax[2].scatter(V, W, c=[e.month for e in FMT_date(prof.JULD.values, 'dt', verbose=False)[mask]],
    #                          cmap='twilight')
    #     ax[2].plot(ax[2].get_xlim(), s * np.array(ax[2].get_xlim()) + i, c='k',
    #                label='Reg slope: {:.2f}'.format(reg.slope))
    #     ax[2].set_title('SCMs values vs. width: $r^{{2}} = {:.2f}$'.format(r2))
    #     ax[2].set_xlabel('SCMs chl-a')
    #     ax[2].set_ylabel('SCMs widths')
    #     ax[2].legend()
    #
    #     cbar = plt.colorbar(scat, cax=fig.add_axes([ax[2].get_position().x1 + 0.01,
    #                                                 ax[2].get_position().y0, 0.02, ax[2].get_position().height]),
    #                         label='Month')
    #     cbar.set_ticks(np.arange(1, 13))
    #     cbar.set_ticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    #     fig.savefig(VAR.figpath + '{}/corr_scmfeatures_{}.png'.format(wmo, VAR.float_names[wmo]))

    # SCMs widths values and depths correlation for all

    # D, V, W, T = np.array([]), np.array([]), np.array([]), np.array([])
    #
    # for wmo in WMOS():
    #     prof = PRC(wmo)
    #     scmfeatures = prof.SCM_GDEPTH.values, prof.SCM_GVAL.values, prof.SCM_GWIDTH.values
    #     D, V, W, T = np.hstack([D, scmfeatures[0]]), np.hstack([V, scmfeatures[1]]), np.hstack([W, scmfeatures[2]]), \
    #         np.hstack([T, np.array([e.month for e in FMT_date(prof.JULD.values, 'dt', verbose=False)])])
    #
    # mask = ~np.isnan(V)
    # D, V, W, T = D[mask], V[mask], W[mask], T[mask]
    # fig, ax = plt.subplots(1, 3, figsize=(15., 6.5))
    # fig.suptitle('Correlation between SCMs depths, values and widths for all floats')
    #
    # reg = stats.linregress(D, V)
    # s, i, r2 = reg.slope, reg.intercept, reg.rvalue ** 2
    # ax[0].scatter(D, V, c=T, cmap='twilight')
    # ax[0].plot(ax[0].get_xlim(), s * np.array(ax[0].get_xlim()) + i, c='k', label='Reg slope: {:.4f}'.format(reg.slope))
    # ax[0].set_title('SCMs depths vs. values: $r^{{2}} = {:.2f}$'.format(r2))
    # ax[0].set_xlabel('SCMs depths')
    # ax[0].set_ylabel('SCMs chl-a')
    # ax[0].legend()
    #
    # reg = stats.linregress(D, W)
    # s, i, r2 = reg.slope, reg.intercept, reg.rvalue ** 2
    # ax[1].scatter(D, W, c=T, cmap='twilight')
    # ax[1].plot(ax[1].get_xlim(), s * np.array(ax[1].get_xlim()) + i, c='k', label='Reg slope: {:.2f}'.format(reg.slope))
    # ax[1].set_title('SCMs depths vs. width: $r^{{2}} = {:.2f}$'.format(r2))
    # ax[1].set_xlabel('SCMs depths')
    # ax[1].set_ylabel('SCMs widths')
    # ax[1].legend()
    #
    # reg = stats.linregress(V, W)
    # s, i, r2 = reg.slope, reg.intercept, reg.rvalue ** 2
    # scat = ax[2].scatter(V, W, c=T, cmap='twilight')
    # ax[2].plot(ax[2].get_xlim(), s * np.array(ax[2].get_xlim()) + i, c='k', label='Reg slope: {:.2f}'.format(reg.slope))
    # ax[2].set_title('SCMs values vs. width: $r^{{2}} = {:.2f}$'.format(r2))
    # ax[2].set_xlabel('SCMs chl-a')
    # ax[2].set_ylabel('SCMs widths')
    # ax[2].legend()
    #
    # cbar = plt.colorbar(scat, cax=fig.add_axes([ax[2].get_position().x1 + 0.01,
    #                                             ax[2].get_position().y0, 0.02, ax[2].get_position().height]),
    #                     label='Month')
    # cbar.set_ticks(np.arange(1, 13))
    # cbar.set_ticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    # fig.savefig(VAR.figpath + 'corr_scmfeatures.png')


    # Randomly test parameters for curve fit on CHLA_PRC

    #
    # def Fsgm(z, Ks, Z12, s):
    #
    #     return Ks/(1 + np.exp((z - Z12)*s))
    #
    # def Fgss(z, Kg, Zm, sig):
    #
    #     return Kg * np.exp(-((z - Zm) / sig) ** 2)
    #
    # def F(z, Ks, Z12, s, Km, Zm, sig):
    #
    #     return Fsgm(z, Ks, Z12, s) + Fgss(z, Km, Zm, sig)
    #
    # bounds = [(0., .5), (30., 70.), (0.02, 0.6), (0., 0.7), (20., 150.), (10., 70.)]
    # fig, ax = plt.subplots()
    # ax.set_ylim(300., 0.)
    # ax.set_xlim(-0.05, 1.)
    # z = np.linspace(0., 300., 200)
    #
    # for k in range(100):
    #
    #     Ks, Z12, s, Kg, Zm, sig = bounds[0][0] + bounds[0][1] * np.random.random(), \
    #                               bounds[1][0] + bounds[1][1] * np.random.random(), \
    #                               bounds[2][0] + bounds[2][1] * np.random.random(), \
    #                               bounds[3][0] + bounds[3][1] * np.random.random(), \
    #                               bounds[4][0] + bounds[4][1] * np.random.random(), \
    #                               bounds[5][0] + bounds[5][1] * np.random.random()
    #     y = F(z, Ks, Z12, s, Kg, Zm, sig)
    #     ys = Fsgm(z, Ks, Z12, s)
    #     yg = Fgss(z, Kg, Zm, sig)
    #
    #     ax.plot(y, z, c='k')
    #     ax.plot(ys, z, c='g')
    #     ax.plot(yg, z, c='r')
    #
    #     ax.plot(ax.get_xlim(), [Zm, Zm], c='r', linewidth=1., linestyle='dashed', label='$Z_m$ (gauss)')
    #     ax.plot([Kg, Kg], ax.get_ylim(), c='r', linewidth=1., linestyle='dotted', label='$K_g$ (gauss)')
    #     ax.plot(ax.get_xlim(), [Z12, Z12], c='g', linewidth=1., linestyle='dashed', label='$Z_{1/2}$ (sigm)')
    #     ax.plot([Ks, Ks], ax.get_ylim(), c='g', linewidth=1., linestyle='dotted', label='$K_s$ (sigm)')
    #
    #     if k == 0:
    #         ax.legend()
    #
    #     plt.pause(5.)
    #
    #     for artist in ax.collections + ax.lines + ax.texts:
    #         artist.remove()

    # Sigmoid + gaussian profile


    # def Fsgm(z, Ks, Z12, s):
    #
    #     return Ks/(1 + np.exp((z - Z12)*s))
    #
    # def Fgss(z, Kg, Zm, sig):
    #
    #     return Kg * np.exp(-((z - Zm) / sig) ** 2)
    #
    # def F(z, Ks, Z12, s, Km, Zm, sig):
    #
    #     return Fsgm(z, Ks, Z12, s) + Fgss(z, Km, Zm, sig)
    #
    # fig, ax = plt.subplots()
    # ax.set_ylim(300., 0.)
    # ax.set_xlim(-0.05, 1.)
    # ax_new = ax.twinx().twiny()
    # ax_new.set_xlim(-.05, .6)
    # ax_new.set_ylim(-.05, .6)
    # z = np.linspace(0., 300., 200)
    #
    # prof = PRC(2909)
    #
    # for i in range(100):
    #
    #     k = int(np.random.choice(np.arange(0., np.size(prof.N_PROF.values), 1.)))
    #
    #     chla, pres = prof.CHLA_PRC.values[k], prof.PRES.values[k]
    #     chla, pres = chla[~np.isnan(chla)], pres[~np.isnan(chla)]
    #     chla, pres = chla[pres * VAR.hf < 300.], pres[pres * VAR.hf < 300.]
    #
    #     if np.size(chla) > 10:
    #
    #         res = opt.curve_fit(F, xdata=pres, ydata=chla, p0=(.2, 30., 0.1, 0.3, 100., 30.),
    #                             bounds=([0., 5., 0., 0., 30., 4.], [3., 100., 1., 1.5, 250., 100.]))
    #         def Ftemp(z):
    #             return F(z, res[0][0], res[0][1], res[0][2], res[0][3], res[0][4], res[0][5])
    #
    #         ax.plot(chla, pres, c='k', label='Profile')
    #         ax.plot(Ftemp(z), z, c='crimson', linewidth=2., label='fit')
    #         ax.plot(Fsgm(z, res[0][0], res[0][1], res[0][2]), z, c='g', label='Sigmoid')
    #         ax.plot(Fgss(z, res[0][3], res[0][4], res[0][5]), z, c='b', label='Gaussian')
    #         ax.scatter(res[0][0], res[0][1], c='g', marker='x', label='($K_s$, $Z_{1/2}$)')
    #         ax.scatter(res[0][3], res[0][4], c='b', marker='x', label='($K_m$, $Z_m$)')
    #
    #         reg = stats.linregress(chla, Ftemp(pres))
    #         ax_new.scatter(chla, Ftemp(pres), c='k', alpha=0.6, s=7, label='chla vs fit: {:.4f}'.format(reg.rvalue**2))
    #         ax_new.plot(ax_new.get_xlim(), ax_new.get_xlim(), c='k', alpha=0.9, zorder=3, label='Identity')
    #         ax_new.plot(ax_new.get_xlim(), np.array(ax_new.get_xlim())*reg.slope + reg.intercept, c='k', linestyle='dashed',
    #                     alpha=0.9, label='Reg: {:.3f}x + {:.3f}'.format(reg.slope, reg.intercept))
    #
    #         leg1 = ax.legend(loc='lower right', title='Gauss' if res[0][3] > 1.5 * res[0][0] else 'Sigm')
    #         leg2 = ax_new.legend(loc='upper right')
    #
    #         plt.pause(5.)
    #
    #         leg1.remove()
    #         leg2.remove()
    #
    #         for artist in ax.collections + ax.lines + ax.texts:
    #             artist.remove()
    #
    #         for artist in ax_new.collections + ax_new.lines + ax_new.texts:
    #             artist.remove()