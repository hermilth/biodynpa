# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the script used to stock global variables.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''
import os

# Imports

import matplotlib.cm as cm
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Classes

class font_attributes:
    WHITE = ''
    BLACK = '\033[0;30m'
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    BROWN = '\033[0;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    DARKCYAN = '\033[36m'
    LIGHT_GRAY = '\033[0;37m'
    DARK_GRAY = '\033[1;30m'
    LIGHT_RED = '\033[1;31m'
    LIGHT_GREEN = '\033[1;32m'
    YELLOW = '\033[1;33m'
    LIGHT_BLUE = '\033[1;34m'
    LIGHT_PURPLE = '\033[1;35m'
    LIGHT_CYAN = '\033[1;36m'
    LIGHT_WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    FAINT = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    NEGATIVE = '\033[7m'
    CROSSED = '\033[9m'
    END = '\033[0m'

class InputTimedOut(Exception):
    pass

# Paths

indexpath = 'Files/'
chloropath = indexpath+'Chloro/'
sprofpath = indexpath+'Sprof/'
figpath = indexpath+'Figures/'
logspath = indexpath+'Logs/'
prcpath = indexpath+'Processed/'
meipath = indexpath+'MEI/'

# NASA appkey: you can get yours at https://oceandata.sci.gsfc.nasa.gov/appkey/

myappkey = 'ddccdcd28c616d486d9d461180f190585ecdf28e'

# Plot variables

mpl_ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
linewidth = 1.5
rcParams = {'backend': 'GTK4Cairo',
            'text.usetex': True,
            'text.latex.preamble': r'\usepackage{amsmath}',
            'font.family': 'serif',
            'font.serif': ['New Century Schoolbook'],
            'figure.figsize': [12. ,  8.],
            'figure.dpi': 90,
            'figure.subplot.top': 0.9,
            'figure.subplot.bottom': 0.15,
            'figure.subplot.left': 0.08,
            'figure.subplot.right': 0.88,
            'figure.titlesize': 20,
            'axes.titlesize': 14,
            'font.size': 13,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 10,
            'axes.xmargin': .08,
            'axes.ymargin': .08}
oceancolor = (0.2, 0.35, 0.75)
landcolor = (0.8, 0.5, 0.137)
placescolor = '#252525'

# WMOs of interest : file 'dataset_params.csv' to fill accordingly

with open(indexpath + 'dataset_params.csv') as f:
    dataset_params = np.array([[s for s in line[:-1].split(',')] for line in f.readlines()])
essentials = dataset_params[1:, 0].astype(int)
floats_colors = dict(zip(dataset_params[1:, 0].astype(int), dataset_params[1:, 1]))
floats_names = dict(zip(dataset_params[1:, 0].astype(int), dataset_params[1:, 4]))
clusters = np.unique(dataset_params[1:, 2])
clusters_colors = dict(zip(clusters, [dataset_params[1:, 3][dataset_params[1:, 2] == c][0] for c in clusters]))
clusters_floats = dict(zip(clusters, [dataset_params[1:, 0][dataset_params[1:, 2] == c].astype(int) for c in clusters]))

# Constants

Rt = 6.4e6
rho = 1025.
g = 9.81
hf = 10000 / rho / g
activate_fun = False
animtime = 2.5

# Zones definition [center_lon, center_lat, width_km, height_km]

zones = {'SP': [-143., -26., 15000., 6500.],                # South Pacific
         'WO': [0., 0., 70000, 17500],                      # World
         'PA': [-153., 0., 20000, 15000.],                  # Pacific
         'NP': [-170., 30., 15000., 7000.],                 # North Pacific
         'OL': [-157., -20., 3000., 1500.],                 # Ultra oligo zone
         'SP_w': [-143., -26., 16000., 8000.],              # South Pacific wide
         'EP': [-115., -26., 9000., 6000.],                 # Easter Pacific
         'WP': [-165., -26., 9000., 6000.],                 # Western Pacific
         'SO': [-144., -46., 12000., 8000.],                # Southern Ocean
         'CS': [156., -16., 3000., 2000.],                  # Coral Sea
         'TS': [160., -35., 3000., 1800.],                  # Tasman Sea
         'MA': [-140., -8.8, 600., 400.],                   # Marquesas
         'FJ': [178.4, -17.4, 2130., 1200.],                # Fiji islands
         'FJ_w': [178.4, -17.4, 3500., 2000.],              # Fiji islands wide
         'EA': [-109.3, -27.1, 880., 500.],                 # Easter island
         'MA_w': [-132., -9.3, 3400., 1300.],               # Marquesas wide
         'PE': [-85., -7., 2000., 1400.],                   # Peru upwelling
         'PE_w': [-85., -7., 4000., 2500.],                 # Peru upwelling wide
         'temp': [-140., -15., 4500., 2200],                # Temporary zone (for tests and figures)
         'M1': [-142., -9.3, 1100., 600.],                  # M1
         'M2': [-133., -9.3, 2400., 1000.],                 # M2
         'M3': [-132., -9.3, 2600., 1000.],                 # M3
         'M4': [-129., -9.3, 2800., 1000.],                 # M4
         'O1': [-162., -20., 4500., 2000.],                 # O1
         'O2': [-162., -20., 4500., 2000.],                 # O2
         'O3': [-162., -20., 4500., 2000.],                 # O3
         'F1': [178.4, -17.4, 3500., 2000.],                # F1
         'F2': [178.4, -17.4, 3500., 2000.],                # F2
         'F3': [178.4, -17.4, 3500., 2000.],                # F3
         'C1': [156., -16., 3000., 2000.],                  # C1
         'C2': [156., -16., 3000., 2000.],                  # C2
         'T1': [160., -35., 3000., 1800.],                  # T1
         'T2': [160., -35., 3000., 1800.]}                  # T2

zones_names = {'WO': 'World',
               'PA': 'Pacific Ocean',
               'NP': 'North Pacific',
               'SP': 'South Pacific',
               'CP': 'Central Pacific',
               'SP_w': 'South Pacific',
               'EP': 'Eastern South Pacific',
               'WP': 'Western South Pacific',
               'SO': 'Southern Ocean',
               'CS': 'Coral Sea',
               'TS': 'Tasman Sea',
               'MA': 'Marquesas Archipelago',
               'FJ': 'Fiji islands',
               'FJ_w': 'Fiji islands',
               'EA': 'Easter island',
               'MA_w': 'Marquesas Archipelago',
               'PE': 'Peru upwelling',
               'PE_w': 'Peru upwelling',
               'temp': 'Your area'}

places = {'Marquesas': [-139.5, -9.3],
          'Fidji islands': [181., -16.5],
          'Easter island': [-109.3, -27.1],
          'Tahiti': [-149., -17.],
          'New Caledonia': [168., -23.],
          r'\textit{Peru\\upwelling}': [-94., -3.],
          r'\textit{Coral sea}': [150., -17.],
          r'\textit{Tasman Sea}': [152., -42.],
          r'\textit{Southern Ocean}': [-145, -53.]}

# OPN base parameters, processed files parameter

reject_QCs = (4)
datamode = None
basefilter = {'DIRECTION': b'A', 'LOCALTIME_MIN': 9., 'LOCALTIME_MAX': 15.}

# Processing parameters

Tsmooth = 50.
Zsmooth = 10.
Zsmoothinterp = 0.2
irrpoly_order = 4
letout = 2.
nstd_extremes = 5.

# Password: for this to work, you have to store a txt file with your Copernicus Marine Service credentials just above
# the python project position in your folder architecture. On the first line should be written your id, and on the
# second line your password.

psw = 'usemystreetcredz'

# Names and units to use in plots

var_names = {'TEMP': 'Raw temperature',
             'TEMP_ADJUSTED': 'Adj. temperature',
             'PSAL': 'Raw salinity',
             'PSAL_ADJUSTED': 'Adj. salinity',
             'CHLA': 'Raw chl-a',
             'CHLA_ADJUSTED': 'Adj. chl-a',
             'CHLA_PRC': 'Processed chl-a',
             'iCHLA_PRC': 'Integrated chl-a',
             'iCHLA_MAX': 'Total int. chl-a',
             'PRES': 'Raw Pressure',
             'PRES_ADJUSTED': 'Adj. Pressure',
             'DOXY': 'Oxygen',
             'DOXY_ADJUSTED': 'Adj. oxygen',
             'BBP700': 'Raw b$_{bp700}$',
             'BBP700_DS': 'b$_{bp700}$ smoothed',
             'BBP700_ADJUSTED': 'Adj. $b_{bp700}$',
             'CDOM': 'Raw CDOM',
             'CDOM_ADJUSTED': 'Adj. CDOM',
             'DOWNWELLING_PAR': 'Raw PAR',
             'DOWNWELLING_PAR_ADJUSTED': 'Adj. PAR',
             'DOWN_IRRADIANCE380': 'Raw DW irr 380nm',
             'DOWN_IRRADIANCE380_ADJUSTED': 'Adj. DW irr',
             'DOWN_IRRADIANCE412': 'Raw DW irr 412nm',
             'DOWN_IRRADIANCE412_ADJUSTED': 'Adj. DW irr 412nm',
             'DOWN_IRRADIANCE490': 'Raw DW irr 490nm',
             'DOWN_IRRADIANCE490_ADJUSTED': 'Adj. DW irr 490nm',
             'DOWN_IRRADIANCE380_FIT': 'DW irr fit 380nm',
             'DOWN_IRRADIANCE412_FIT': 'DW irr fit 412nm',
             'DOWN_IRRADIANCE490_FIT': 'DW irr fit 490nm',
             'DOWNWELLING_PAR_FIT': 'DW PAR fit',
             'CLD': 'Clouds',
             'IRR_FLG': 'Irr flag',
             'ZEU' : 'Photic depth',
             'PAR_SCM' : 'PAR at SCM',
             'iPAR_ML': 'iPAR (ML)',
             'PH_IN_SITU_TOTAL': 'In-situ $pH$',
             'PH_IN_SITU_TOTAL_ADJUSTED': 'Adj. in-situ $pH$',
             'NITRATE': 'Nitrates',
             'NITRATE_ADJUSTED': 'Adj. nitrates',
             'LONGITUDE': 'Longitude',
             'LATITUDE': 'Latitude',
             'CT': 'Raw CT',
             'CT_ADJUSTED': 'Adj. CT',
             'MLD_T02': 'MLD ($\\Delta T$)',
             'MLD_S125': 'MLD ($\\Delta \\sigma_0=0.125$)',
             'MLD_S03': 'MLD ($\\Delta \\sigma_0=0.03$)',
             'MLD_DT': 'MLD ($T$ gradient)',
             'MLD_DS': 'MLD ($\\sigma_0$ gradient)',
             'MLD_T02_QI': 'MLD quality index',
             'MLD_S125_QI': 'MLD125 quality index',
             'MLD_S03_QI': 'MLD03 quality index',
             'MLD_DT_QI': 'MLD quality index',
             'MLD_DS_QI': 'MLD quality index',
             'SCM': 'SCM',
             'SCM_GDEPTH': 'SCM depth (gauss fit)',
             'SCM_GVAL': 'SCM chl-a (gauss fit)',
             'SCM_GWIDTH': 'SCM width',
             'ISO15': 'Isolume$_{15}$',
             'SIG0': 'Raw pot. density',
             'SIG0_ADJUSTED': 'Adj. pot. density',
             'BVF': 'Raw B.V. freq',
             'BVFMAX_DEPTH': '$BVF_{max}$ depth',
             'BVF_ADJUSTED': 'Adj. B.V. freq',
             'KD380': '$K_d_{380nm}$',
             'KD412': '$K_d_{412nm}$',
             'KD490': '$K_d_{490nm}$',
             'LOCALTIME': 'Local time',
             'CHLA_ZERO': 'Chl-a minus dark',
             'CDARK_VALUE': 'Chl-a dark value',
             'CDARK_DEPTH': 'Chl-a dark depth',
             'SLOPEF490': 'Slope factor 490nm',
             'SLOPEF490_QI': 'Slope factor 490nm QI',
             'SLOPEF412': 'Slope factor 490nm',
             'SLOPEF412_QI': 'Slope factor 412nm QI',
             'SLOPEF380': 'Slope factor 380nm',
             'SLOPEF380_QI': 'Slope factor 380nm QI'}

var_units = {'TEMP': '$^{\circ}C$',
             'TEMP_ADJUSTED': '$^{\circ}C$',
             'PSAL': '$\mathit{ppm}$',
             'PSAL_ADJUSTED': '$\mathit{ppm}$',
             'CHLA': '$mg.m^{-3}$',
             'CHLA_ADJUSTED': '$mg.m^{-3}$',
             'CHLA_PRC': '$mg.m^{-3}$',
             'iCHLA_PRC': '$mg.m^{-2}$',
             'iCHLA_MAX': '$mg.m^{-2}$',
             'PRES': '$\mathit{dbar}$',
             'PRES_ADJUSTED': '$\mathit{dbar}$',
             'DOXY': '$\mu \mathit{mol}.\mathit{kg}^{-1}$',
             'DOXY_ADJUSTED': '$\mu \mathit{mol}.\mathit{kg}^{-1}$',
             'BBP700': '$m^{-1}$',
             'BBP700_ADJUSTED': '$m^{-1}$',
             'BBP700_DS': '$m^{-1}$',
             'CDOM': '$ppb$',
             'CDOM_ADJUSTED': '$ppb$',
             'DOWNWELLING_PAR': '$\mu \mathit{mol} E.m^{-2}.s^{-1}$',
             'DOWNWELLING_PAR_ADJUSTED': '$\mu \mathit{mol} E.m^{-2}.s^{-1}$',
             'DOWN_IRRADIANCE380': '$W.m^{-2}.nm^{-1}$',
             'DOWN_IRRADIANCE380_ADJUSTED': '$W.m^{-2}.nm^{-1}$',
             'DOWN_IRRADIANCE412': '$W.m^{-2}.nm^{-1}$',
             'DOWN_IRRADIANCE412_ADJUSTED': '$W.m^{-2}.nm^{-1}$',
             'DOWN_IRRADIANCE490': '$W.m^{-2}.nm^{-1}$',
             'DOWN_IRRADIANCE490_ADJUSTED': '$W.m^{-2}.nm^{-1}$',
             'iPAR_ML': '$\mu \mathit{mol} E.m^{-1}.s^{-1}$',
             'PAR_SCM' : '$\mu \mathit{mol} E.m^{-2}.s^{-1}$',
             'PH_IN_SITU_TOTAL': '',
             'PH_IN_SITU_TOTAL_ADJUSTED': '',
             'NITRATE': '$\mu mol.kg^{-1}$',
             'NITRATE_ADJUSTED': '$\mu mol.kg^{-1}$',
             'LONGITUDE': '$deg$',
             'LATITUDE': '$deg$',
             'CT': '$^{\circ}C$',
             'CT_ADJUSTED': '$^{\circ}C$',
             'MLD_T02': '$m$',
             'MLD_S125': '$m$',
             'MLD_S03': '$m$',
             'MLD_DT': '$m$',
             'MLD_DS': '$m$',
             'MLD_ADJUSTED': '$m$',
             'SCM': '$m$',
             'SCM_GDEPTH': '$m$',
             'SCM_GVAL': '$mg.m^{-3}$',
             'SCM_GWIDTH': '$m$',
             'SCM_ADJUSTED': '$m$',
             'ISO15': '$m$',
             'ISO15_ADJUSTED': '$m$',
             'SIG0': '$kg.m^{-3}$',
             'SIG0_ADJUSTED': '$kg.m^{-3}$',
             'BVF': '$s^{-1}$',
             'BVF_ADJUSTED': '$s^{-1}$',
             'BVFMAX_DEPTH': '($m$)',
             'KD380': '$m^{-1}$',
             'KD412': '$m^{-1}$',
             'KD490': '$m^{-1}$',
             'DOWN_IRRADIANCE380_FIT': '$W.m^{-2}.nm^{-1}$',
             'DOWN_IRRADIANCE412_FIT': '$W.m^{-2}.nm^{-1}$',
             'DOWN_IRRADIANCE490_FIT': '$W.m^{-2}.nm^{-1}$',
             'LOCALTIME': '$\\mathit{Days}$',
             'ZEU': '$m$',
             'MLD_T02_QI': '',
             'MLD_S125_QI': '',
             'MLD_S03_QI': '',
             'MLD_DT_QI': '',
             'MLD_DS_QI': '',
             'IRR_FLG': '',
             'CLD': '',
             'DOWNWELLING_PAR_FIT': '$\mu \mathit{mol} E.m^{-2}.s^{-1}$',
             'CHLA_ZERO': '$mg.m^{-3}$',
             'CDARK_VALUE': '$mg.m^{-3}$',
             'CDARK_DEPTH': '$m$',
             'SLOPEF490': '',
             'SLOPEF490_QI': '',
             'SLOPEF412': '',
             'SLOPEF412_QI': '',
             'SLOPEF380': '',
             'SLOPEF380_QI': ''}

var_labels = dict(zip(var_names.keys(), [var_names[k] if len(var_units[k]) == 0 else
                                         '{}\n({})'.format(var_names[k], var_units[k]) for k in var_names.keys()]))

# Custom colormaps

def GET_cmap(*col):
    '''
    Description: Creates a colormap lineraly going through all the color arguments.\n
    ____\n
    Notes: This is the PhD work of T. Hermilly.\n
    Last update: 2022-05-23\n
    ____\n
    :param col: The colors composing the colormap. (matplotlib colors)
    :return: The matplotlib colormap.
    '''
    N = len(col)
    clist = [colors.to_rgba(c) for c in col]

    vals = np.ones((256, 4))

    for i in range(N-2):
        vals[i*256//(N-1):(i+1)*256//(N-1), 0] = np.linspace(clist[i][0], clist[i+1][0], (i+1)*256//(N-1)-i*256//(N-1))
        vals[i*256//(N-1):(i+1)*256//(N-1), 1] = np.linspace(clist[i][1], clist[i+1][1], (i+1)*256//(N-1)-i*256//(N-1))
        vals[i*256//(N-1):(i+1)*256//(N-1), 2] = np.linspace(clist[i][2], clist[i+1][2], (i+1)*256//(N-1)-i*256//(N-1))

    fill = 256 - (N-2)*256//(N-1)
    vals[-fill:, 0] = np.linspace(clist[-2][0], clist[-1][0], fill)
    vals[-fill:, 1] = np.linspace(clist[-2][1], clist[-1][1], fill)
    vals[-fill:, 2] = np.linspace(clist[-2][2], clist[-1][2], fill)

    return colors.ListedColormap(vals)

temp_cmap = GET_cmap((0.05, 0.05, 0.3), (0.1, 0.7, 0.9), (0.9, 0.95, 0.6), (0.9, 0.2, 0.2), (0.6, 0., 0.1))
psal_cmap = GET_cmap((0.05, 0.05, 0.3), (0.1, 0.7, 0.9), (1., 0.95, 0.9), (0.9, 0.45, 0.1), (0.6, 0., 0.1))
rho_cmap = GET_cmap((0., 0.2, 0.6), (0.1, 0.7, 0.9), (0.9, 0.95, 0.9), (0.8, 0.1, 0.1), (0.3, 0., 0.2))
par_cmap = GET_cmap((0.1, 0., 0.1), (0.3, 0., 0.4), (0.1, 0.2, 0.5), (0., 0.4, 0.6), (0.8, 0.8, 0.6), (1., 1., 0.9))
nitrate_cmap = GET_cmap((1., 1., 0.9), (1., 0.8, 0.4), (0.8, 0.5, 0.3), (0.6, 0.2, 0.1), (0.1, 0., 0.2))
oxy_cmap = GET_cmap((1., 1., 0.8), 'turquoise', 'darkblue', 'purple', (0.2, 0., 0.2))

# Cmaps to use in the plots

var_cmaps = {'TEMP': 'RdYlBu_r',
             'TEMP_ADJUSTED': 'RdYlBu_r',
             'PSAL': psal_cmap,
             'PSAL_ADJUSTED': psal_cmap,
             'CHLA': 'viridis',
             'CHLA_ADJUSTED': 'viridis',
             'CHLA_PRC': 'viridis',
             'iCHLA_PRC': 'viridis',
             'iCHLA_MAX': 'jet',
             'PRES': 'afmhot_r',
             'PRES_ADJUSTED': 'afmhot_r',
             'DOXY': oxy_cmap,
             'DOXY_ADJUSTED': oxy_cmap,
             'BBP700': 'magma_r',
             'BBP700_DS': 'magma_r',
             'BBP700_ADJUSTED': 'magma_r',
             'CDOM': 'Greens',
             'CDOM_ADJUSTED': 'Greens',
             'DOWNWELLING_PAR': par_cmap,
             'DOWNWELLING_PAR_ADJUSTED': par_cmap,
             'DOWN_IRRADIANCE380': 'magma',
             'DOWN_IRRADIANCE380_ADJUSTED': 'magma',
             'DOWN_IRRADIANCE412': 'magma',
             'DOWN_IRRADIANCE412_ADJUSTED': 'magma',
             'DOWN_IRRADIANCE490': 'magma',
             'DOWN_IRRADIANCE490_ADJUSTED': 'magma',
             'iPAR_ML': 'jet',
             'PAR_SCM': 'jet',
             'PH_IN_SITU_TOTAL': 'gist_rainbow',
             'PH_IN_SITU_TOTAL_ADJUSTED': 'gist_rainbow',
             'NITRATE': nitrate_cmap,
             'NITRATE_ADJUSTED': nitrate_cmap,
             'CT': 'RdYlBu_r',
             'CT_ADJUSTED': 'RdYlBu_r',
             'MLD_T02': 'afmhot_r',
             'MLD_S125': 'afmhot_r',
             'MLD_S03': 'afmhot_r',
             'MLD_DT': 'afmhot_r',
             'MLD_DS': 'afmhot_r',
             'SCM': 'afmhot_r',
             'SCM_GDEPTH': 'afmhot_r',
             'SCM_GVAL': 'afmhot_r',
             'SCM_GWIDTH': 'afmhot_r',
             'ISO15': 'inferno',
             'ISO15_ADJUSTED': 'inferno',
             'SIG0': rho_cmap,
             'SIG0_ADJUSTED': rho_cmap,
             'BVF': 'bwr',
             'BVFMAX_DEPTH': 'bwr',
             'BVF_ADJUSTED': 'bwr',
             'KD380': 'RdYlBu_r',
             'KD412': 'RdYlBu_r',
             'KD490': 'RdYlBu_r',
             'DOWN_IRRADIANCE380_FIT': 'RdYlBu_r',
             'DOWN_IRRADIANCE412_FIT': 'RdYlBu_r',
             'DOWN_IRRADIANCE490_FIT': 'RdYlBu_r',
             'LOCALTIME': 'RdYlBu_r',
             'ZEU': 'jet',
             'MLD_T02_QI': 'RdYlBu_r',
             'MLD_S125_QI': 'RdYlBu_r',
             'MLD_S03_QI': 'RdYlBu_r',
             'MLD_DT_QI': 'RdYlBu_r',
             'MLD_DS_QI': 'RdYlBu_r',
             'IRR_FLG': 'RdYlBu_r',
             'CLD': 'RdYlBu_r',
             'DOWNWELLING_PAR_FIT': 'RdYlBu_r',
             'CHLA_ZERO': 'RdYlBu_r',
             'CDARK_VALUE': 'RdYlBu_r',
             'CDARK_DEPTH': 'RdYlBu_r',
             'SLOPEF490': 'RdYlBu_r',
             'SLOPEF490_QI': 'RdYlBu_r',
             'SLOPEF412': 'RdYlBu_r',
             'SLOPEF412_QI': 'RdYlBu_r',
             'SLOPEF380': 'RdYlBu_r',
             'SLOPEF380_QI': 'RdYlBu_r'}

# Variables values limits

var_lims = {'TEMP': (15., 28.),
            'TEMP_ADJUSTED': (15., 28.),
            'PSAL': (34.5, 36.5),
            'PSAL_ADJUSTED': (34.5, 36.5),
            'CHLA': (0., 0.5),
            'CHLA_ADJUSTED': (0., 0.5),
            'CHLA_PRC': (0., 0.5),
            'iCHLA_PRC': (0., 50.),
            'iCHLA_MAX': (10., 50.),
            'PRES': (0., 1000.),
            'PRES_ADJUSTED': (0., 1000.),
            'DOXY': (150., 250.),
            'DOXY_ADJUSTED': (150., 250.),
            'BBP700': (0., 5.e-4),
            'BBP700_ADJUSTED': (0., 5.e-4),
            'BBP700_DS': (0., 5.e-4),
            'CDOM': (0., 2.),
            'CDOM_ADJUSTED': (1., 2.),
            'DOWNWELLING_PAR': (0., 3000.),
            'DOWNWELLING_PAR_ADJUSTED': (0., 3000.),
            'DOWN_IRRADIANCE380': (0., 1.),
            'DOWN_IRRADIANCE380_ADJUSTED': (0., 1.),
            'DOWN_IRRADIANCE412': (0., 1.),
            'DOWN_IRRADIANCE412_ADJUSTED': (0., 1.),
            'DOWN_IRRADIANCE490': (0., 1.),
            'DOWN_IRRADIANCE490_ADJUSTED': (0., 1.),
            'PH_IN_SITU_TOTAL': (7.81, 8.12),
            'PH_IN_SITU_TOTAL_ADJUSTED': (7.81, 8.12),
            'NITRATE': (0., 20.),
            'NITRATE_ADJUSTED': (0., 20.),
            'CT': (15., 28.),
            'CT_ADJUSTED': (15., 28.),
            'MLD_T02': (20., 46.),
            'MLD_S': (20., 46.),
            'MLD_DT': (20., 46.),
            'MLD_DS': (20., 46.),
            'SCM': (10., 46.),
            'SCM_GDEPTH': (20., 150.),
            'SCM_GVAL': (.2, 1.),
            'SCM_GWIDTH': (10., 50.),
            'ISO15': (80., 200.),
            'ISO15_ADJUSTED': (80., 200.),
            'SIG0': (22., 27.),
            'SIG0_ADJUSTED': (22., 27.),
            'BVF': (0., 5e-4),
            'BVF_ADJUSTED': (0., 5e-4),
            'BVFMAX_DEPTH': (10., 200.),
            'KD380': (0., 1.),
            'KD412': (0., 1.),
            'KD490': (0., 1.),
            'DOWN_IRRADIANCE380_FIT': (0., 1.),
            'DOWN_IRRADIANCE412_FIT': (0., 1.),
            'DOWN_IRRADIANCE490_FIT': (0., 1.),
            'LOCALTIME': (0., 24.),
            'ZEU': (50., 150.),
            'PAR_SCM': (200., 2000.),
            'MLD_T02_QI': (1, 4),
            'MLD_S_QI': (1, 4),
            'MLD_DT_QI': (1, 4),
            'MLD_DS_QI': (1, 4),
            'IRR_FLG': (1, 4),
            'CLD': (0, 1),
            'DOWNWELLING_PAR_FIT': (0., 3000.),
            'CHLA_ZERO': (0., 0.6),
            'CDARK_VALUE': (0., 0.05),
            'CDARK_DEPTH': (100., 400.),
            'SLOPEF490': (0.1, 2.),
            'SLOPEF490_QI': (1, 4),
            'SLOPEF412': (0.1, 2.),
            'SLOPEF412_QI': (1, 4),
            'SLOPEF380': (0.1, 2.),
            'SLOPEF380_QI': (1, 4)}

# Variable kwargs

var_kwargs = {'TEMP': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['TEMP'], 'isocol': 'k'},
              'TEMP_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both',
                                'cmap': var_cmaps['TEMP_ADJUSTED'], 'isocol': 'k'},
              'PSAL': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['PSAL'], 'isocol': 'k'},
              'PSAL_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['PSAL_ADJUSTED'],
                                'isocol': 'k'},
              'CHLA': {'scientific': False, 'log': False, 'extend': 'max', 'cmap': var_cmaps['CHLA'],
                       'isocol': 'white'},
              'CHLA_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'max', 'cmap': var_cmaps['CHLA_ADJUSTED'],
                                'isocol': 'white'},
              'CHLA_PRC': {'scientific': False, 'log': False, 'extend': 'max', 'cmap': var_cmaps['CHLA_ADJUSTED'],
                                'isocol': 'white'},
              'iCHLA_PRC': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['CHLA_ADJUSTED'],
                                'isocol': 'white'},
              'iCHLA_MAX': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['CHLA_ADJUSTED'],
                                'isocol': 'white'},
              'PRES': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['PRES'], 'isocol': 'k'},
              'PRES_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['PRES_ADJUSTED'],
                                'isocol': 'k'},
              'DOXY': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['DOXY'], 'isocol': 'k'},
              'DOXY_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['DOXY_ADJUSTED'],
                                'isocol': 'k'},
              'BBP700': {'scientific': True, 'log': False, 'extend': 'both', 'cmap': var_cmaps['BBP700'],
                         'isocol': 'white'},
              'BBP700_DS': {'scientific': True, 'log': False, 'extend': 'both', 'cmap': var_cmaps['BBP700_DS'],
                         'isocol': 'white'},
              'BBP700_ADJUSTED': {'scientific': True, 'log': False, 'extend': 'both',
                                  'cmap': var_cmaps['BBP700_ADJUSTED'], 'isocol': 'white'},
              'CDOM': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['CDOM'], 'isocol': 'k'},
              'CDOM_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['CDOM'],
                                'isocol': 'k'},
              'DOWNWELLING_PAR': {'scientific': True, 'log': True, 'extend': 'max',
                                  'cmap': var_cmaps['DOWNWELLING_PAR'], 'isocol': 'white'},
              'DOWNWELLING_PAR_ADJUSTED': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWNWELLING_PAR_ADJUSTED'], 'isocol': 'white'},
              'DOWN_IRRADIANCE380': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE380'], 'isocol': 'white'},
              'DOWN_IRRADIANCE380_ADJUSTED': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE380_ADJUSTED'], 'isocol': 'white'},
              'DOWN_IRRADIANCE412': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE412'], 'isocol': 'white'},
              'DOWN_IRRADIANCE412_ADJUSTED': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE412_ADJUSTED'], 'isocol': 'white'},
              'DOWN_IRRADIANCE490': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE490'], 'isocol': 'white'},
              'DOWN_IRRADIANCE490_ADJUSTED': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE490_ADJUSTED'], 'isocol': 'white'},
              'PH_IN_SITU_TOTAL': {'scientific': False, 'log': False, 'extend': 'both',
                                           'cmap': var_cmaps['PH_IN_SITU_TOTAL'], 'isocol': 'k'},
              'PH_IN_SITU_TOTAL_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both',
                                           'cmap': var_cmaps['PH_IN_SITU_TOTAL_ADJUSTED'], 'isocol': 'k'},
              'NITRATE': {'scientific': True, 'log': False, 'extend': 'both', 'cmap': var_cmaps['NITRATE'],
                          'isocol': 'k'},
              'NITRATE_ADJUSTED': {'scientific': True, 'log': False, 'extend': 'both',
                                   'cmap': var_cmaps['NITRATE_ADJUSTED'], 'isocol': 'k'},
              'CT': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['CT'], 'isocol': 'k'},
              'CT_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['CT_ADJUSTED'],
                              'isocol': 'k'},
              'MLD_T02': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['MLD_T02'],
                          'isocol': 'k'},
              'MLD_S125': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['MLD_S125'],
                           'isocol': 'k'},
              'MLD_S03': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['MLD_S03'],
                          'isocol': 'k'},
              'MLD_DT': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['MLD_DT'],
                         'isocol': 'k'},
              'MLD_DS': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['MLD_DS'],
                         'isocol': 'k'},
              'SCM': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['SCM'], 'isocol': 'k'},
              'SCM_GDEPTH': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['SCM'],
                             'isocol': 'k'},
              'SCM_GVAL': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['SCM'],
                           'isocol': 'k'},
              'SCM_GWIDTH': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['SCM'],
                             'isocol': 'k'},
              'ZEU': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['SCM'], 'isocol': 'k'},
              'ISO15': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['ISO15'],
                            'isocol': 'k'},
              'SIG0': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['SIG0'], 'isocol': 'k'},
              'SIG0_ADJUSTED': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['SIG0_ADJUSTED'],
                                'isocol': 'k'},
              'BVF': {'scientific': True, 'log': False, 'extend': 'max', 'cmap': var_cmaps['BVF'], 'isocol': 'k'},
              'BVFMAX_DEPTH': {'scientific': False, 'log': True, 'extend': 'both', 'cmap': var_cmaps['BVFMAX_DEPTH'],
                            'isocol': 'k'},
              'BVF_ADJUSTED': {'scientific': True, 'log': False, 'extend': 'max', 'cmap': var_cmaps['BVF_ADJUSTED'],
                               'isocol': 'k'},
              'KD380': {'log': False, },
              'KD412': {'log': False, },
              'KD490': {'log': False, },
              'DOWN_IRRADIANCE380_FIT': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE380_FIT'], 'isocol': 'white'},
              'DOWN_IRRADIANCE412_FIT': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE490_FIT'], 'isocol': 'white'},
              'DOWN_IRRADIANCE490_FIT': {'scientific': True, 'log': True, 'extend': 'max',
                                           'cmap': var_cmaps['DOWN_IRRADIANCE490_FIT'], 'isocol': 'white'},
              'iPAR_ML': {'scientific': False, 'log': False, 'extend': 'both', 'cmap': var_cmaps['iPAR_ML'],
                          'isocol': 'k'},
              'PAR_SCM': {'log': False, },
              'LOCALTIME': {'log': False, },
              'MLD_T02_QI': {'log': False, },
              'MLD_S125_QI': {'log': False, },
              'MLD_S03_QI': {'log': False, },
              'MLD_DT_QI': {'log': False, },
              'MLD_DS_QI': {'log': False, },
              'IRR_FLG': {'log': False, },
              'CLD': {'log': False, },
              'DOWNWELLING_PAR_FIT': {'log': False, },
              'CHLA_ZERO': {'log': False, },
              'CDARK_VALUE': {'log': False, },
              'CDARK_DEPTH': {'log': False, },
              'SLOPEF490': {'log': False, },
              'SLOPEF490_QI': {'log': False, },
              'SLOPEF412': {'log': False, },
              'SLOPEF412_QI': {'log': False, },
              'SLOPEF380': {'log': False, },
              'SLOPEF380_QI': {'log': False, }}