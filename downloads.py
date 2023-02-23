# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''This is the module dedicated to downloads.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''


# Imports


from profiles import *


# Functions


def DL_index(verbose=True):
    '''
    Creates a index file similarly formatted as the ARGO index file, but with a header containing floats only in
    the Pacific and with a different header, indicating what floats have PAR sensors and thoses which don't. Places
    it under your the path: indexpath (see global variables).

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :return: None
    '''

    def line_prepender(filename, line):
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)

    def get_wmo(l):

        for i in range(21, len(l) - 1):
            if l[i:i + 2] == 'SR' or l[i:i + 2] == 'SD':
                try:
                    wmo = int(l[i + 2:i + 9])

                    return wmo

                except ValueError:

                    wmo = np.nan

                    return wmo

    if 'argo_synthetic-profile_index.txt' in os.listdir(VAR.indexpath):
        os.remove(VAR.indexpath+'argo_synthetic-profile_index.txt')

    server = flib.FTP('ftp.ifremer.fr', 'anonymous', 'anonymous@ifremer.fr')
    server.encoding = 'utf-8'

    if verbose:
        CPRINT('Successfully connected to server.', attrs='BLUE', end='\r')

    server.cwd('~')
    server.cwd('ifremer/argo')

    file = open('argo_synthetic-profile_index.txt', 'wb')
    server.retrbinary(r'RETR argo_synthetic-profile_index.txt', file.write)

    server.quit()

    shutil.move('argo_synthetic-profile_index.txt', VAR.indexpath+'argo_synthetic-profile_index.txt')

    if verbose:
        CPRINT('Index file now under \'{}\'.'.format(VAR.indexpath), attrs='BLUE')

    if 'index_bgc_sp.txt' in os.listdir(VAR.indexpath):
        os.remove(VAR.indexpath+'index_bgc_sp.txt')

    file = open(VAR.indexpath+'argo_synthetic-profile_index.txt', 'r')
    index = open(VAR.indexpath+'index_bgc_sp.txt', 'w')

    head = []

    l = file.readline()
    while l[0] == '#':
        head.append(l)
        l = file.readline()

    head.append(l)

    argo_light = []
    argo_other = []

    for l in file:

        list = CUT_line(l, ',')

        if list[2] != '' and list[4] == 'P' and float(list[2]) < 5. and 'CHLA' in list[7]:

            wmo = get_wmo(l)

            if ('DOWNWELLING_PAR' in l) and not(wmo in argo_light):
                argo_light.append(wmo)
            elif not('DOWNWELLING_PAR' in l) and not(wmo in argo_other):
                argo_other.append(wmo)

    file.close()

    file = open(VAR.indexpath+'argo_synthetic-profile_index.txt', 'r')

    l = file.readline()
    while l[0] == '#':
        l = file.readline()

    for l in file:
        wmo = get_wmo(l)
        if wmo in argo_light or wmo in argo_other:
            index.write(l)

    file.close()
    index.close()

    line1 = '# Floats without PAR: '
    for wmo in argo_other:
        line1 += str(wmo) + ', '
    line1 = line1[:-2]

    line2 = '# Floats with PAR: '
    for wmo in argo_light:
        line2 += str(wmo) + ', '
    line2 = line2[:-2]

    head[0] = '# Title : South Pacific extracted synthetic-Profile directory file of the Argo Global Data Assembly ' \
              'Center'

    line_prepender(VAR.indexpath+'index_bgc_sp.txt', '#\n' + '#'+head[-1])
    line_prepender(VAR.indexpath+'index_bgc_sp.txt', line1)
    line_prepender(VAR.indexpath+'index_bgc_sp.txt', line2)
    line_prepender(VAR.indexpath+'index_bgc_sp.txt', '#\n')

    for l in reversed(head[:-1]):
        line_prepender(VAR.indexpath+'index_bgc_sp.txt', l)

    return None


def DL_greylist(verbose=True):
    '''
    Creates a index file similarly formatted as the ARGO index file, but with a header containing
    floats only in the Pacific and with a different header, indicating what floats have PAR sensors and thoses which
    don't. Places it under your the path: indexpath (see global variables).

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :return: None
    '''
    def line_prepender(filename, line):
        with open(filename, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(line.rstrip('\r\n') + '\n' + content)

    def get_wmo(l):

        for i in range(21, len(l) - 1):
            if l[i:i + 2] == 'SR' or l[i:i + 2] == 'SD':
                try:
                    wmo = int(l[i + 2:i + 9])

                    return wmo

                except ValueError:

                    wmo = np.nan

                    return wmo

    if 'ar_greylist.txt' in os.listdir(VAR.indexpath):
        os.remove(VAR.indexpath+'ar_greylist.txt')

    server = flib.FTP('ftp.ifremer.fr', 'anonymous', 'anonymous@ifremer.fr')
    server.encoding = 'utf-8'

    if verbose:
        CPRINT('Successfully connected to server.', attrs='BLUE', end='\r')

    server.cwd('~')
    server.cwd('ifremer/argo')

    file = open('ar_greylist.txt', 'wb')
    server.retrbinary(r'RETR ar_greylist.txt', file.write)

    server.quit()

    shutil.move('ar_greylist.txt', VAR.indexpath+'ar_greylist.txt')

    if verbose:
        CPRINT('Greylist file now under \'{}\'.'.format(VAR.indexpath), attrs='BLUE')

    if 'ar_greylist_bgc_sp.txt' in os.listdir(VAR.indexpath):
        os.remove(VAR.indexpath+'ar_greylist_bgc_sp.txt')

    file = open(VAR.indexpath+'ar_greylist.txt', 'r')
    greylist = open(VAR.indexpath+'ar_greylist_bgc_sp.txt', 'w')

    greylist.write(file.readline())
    wmos = WMOS('all')

    for l in file:
        a = CUT_line(l, ',')
        if int(a[0]) in wmos:
            greylist.write(l)

    return None


def GET_dacs(wmo_list, verbose=True):

    '''
    Seeks and return the dac of input WMOs.

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :param wmo_list: The WMO list. (list of int)
    :return: The list of dacs. (list of str)
    '''

    notalist = False

    if not(type(wmo_list) is list):
        notalist = True
        wmo_list = [wmo_list]

    server = flib.FTP('ftp.ifremer.fr', 'anonymous', 'anonymous@ifremer.fr')
    server.encoding = 'utf-8'

    if verbose:
         print('Successfully connected to server. Navigating through a sea of folders...', end='\r')

    server.cwd('~')
    server.cwd('ifremer/argo/dac')

    if notalist:
        if verbose:
            print('Serching for dac...', end='\r')
    else:
        if verbose:
            print('Serching for dacs...', end='\r')

    dacs = server.nlst()
    locs = (-np.ones(len(wmo_list), dtype=int)).astype(str)

    i = 0

    while '-1' in locs and i < len(dacs)-1:
        for dac in dacs:

            server.cwd(dac)

            if verbose:
                print(colored('Connected to dac {}...'.format(dac), 'blue'), end='\r')

            dac_wmos = server.nlst()
            found = [str(wmo) in dac_wmos for wmo in wmo_list]
            locs[found] = dac

            i += 1
            server.cwd('..')

    if notalist:

        if verbose:
            print(colored('File found. Quitting server.', 'blue'))
        server.quit()
        return locs[0]

    else:

        if verbose:
            print(colored('Files found. Quitting server.', 'blue'))
        server.quit()
        return locs


def DL_Sprof(wmo_list, dac=None, yestoall=False, notoall=False, verbose=True):
    '''
    Downloads the S-profiles of input WMOs, places it under your global variable of path: sprofpath.
    Will ask whether to redownload if the file is already in.

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :param wmo_list: The WMO list. (list of int)
    :param dac: The dacs of the profiles in the wmo_list param, in order. If None, this function uses find_dac to
     manage. (list of str, optional)
    :param yestoall: Systematically redownloads Sprof if set to True. (boolean, default is False)
    :param yestoall: Systematically not redownloads Sprof if set to True. (boolean, default is False)
    :return: The list of dacs. (list of str)
    '''

    if type(wmo_list) is int:
        wmo_list = [wmo_list]

    if dac is None:
        if verbose:
            CPRINT('Analysing wmos to find dacs...\n', attrs='CYAN')
        locs = GET_dacs(wmo_list, verbose=False)
    else:
        locs = [dac for _ in range(len(wmo_list))]
    dacs = np.unique(locs)

    if verbose:
        CPRINT('Connecting to ftp.ifremer.fr...', attrs='CYAN', end='\r')

    server = flib.FTP('ftp.ifremer.fr', 'anonymous', 'anonymous@ifremer.fr')
    server.encoding = 'utf-8'

    if verbose:
        CPRINT('Successfully connected to server. Navigating through a sea of folders...', end='\r', attrs='CYAN')

    server.cwd('~')
    server.cwd('ifremer/argo/dac')

    for dac in dacs:

        if dac != '-1':

            sub_wmo_list = list(np.array(wmo_list)[locs==dac])

            if verbose:
                CPRINT('Entering dac {}...'.format(dac), attrs='CYAN', end='\r')

            server.cwd(dac)

            if verbose:
                CPRINT('Currently working in /ifremer/argo/dac/{}.'.format(dac), attrs='CYAN')

            for wmo in sub_wmo_list:

                server.cwd('{}'.format(wmo))

                if not('{}_Sprof.nc'.format(wmo) in os.listdir(VAR.sprofpath)):

                    file = open('{}_Sprof.nc'.format(wmo), 'wb')

                    try:

                        if verbose:
                            CPRINT('Downloading {}_Sprof.nc...'.format(wmo), attrs='CYAN', end='\r')
                        time = t.time()
                        server.retrbinary(r'RETR {}_Sprof.nc'.format(wmo), file.write)
                        file.close()
                        if t.time()-time > 30.:
                            color = 'YELLOW'
                        else:
                            color='BLUE'

                        if verbose:
                            CPRINT('Retrieved {}_Sprof.nc: {}'.format(wmo, FMT_secs(t.time() - time)), attrs=color)

                    except Exception as e:

                        CPRINT('We\'ve got to cope with an exception: {}.'.format(e), attrs='CYAN')

                    shutil.move('{}_Sprof.nc'.format(wmo), VAR.sprofpath + '{}_Sprof.nc'.format(wmo))

                else:

                    tmax = 5

                    if not(yestoall) and not(notoall):

                        CPRINT('{}_Sprof.nc already in folder. Would you like to download it again? '
                              '{:.0f}s to answer [y/n]'.format(wmo, tmax), attrs='CYAN', end='\r')
                        answer = INP_timeout(timeout=tmax)

                        if answer == '':
                            if verbose:
                                CPRINT('No answer. Skipping {}_prof.nc.'.format(wmo), attrs='YELLOW')
                        elif not(answer in ['n', 'N', 'y', 'Y']):
                            if verbose:
                                CPRINT('Answer not recognized. Skipping {}_prof.nc.'.format(wmo), attrs='YELLOW')

                    elif yestoall:

                        answer='y'

                    else:

                        answer='n'

                    if answer == 'y' or answer == 'Y':

                        os.remove(VAR.sprofpath+'{}_Sprof.nc'.format(wmo))
                        file = open('{}_Sprof.nc'.format(wmo), 'wb')

                        try:

                            if verbose:
                                CPRINT('Downloading {}_Sprof.nc...'.format(wmo), end='\r', attrs='CYAN')
                            time = t.time()
                            server.retrbinary(r'RETR {}_Sprof.nc'.format(wmo), file.write)
                            file.close()
                            if t.time()-time > 30.:
                                color = 'YELLOW'
                            else:
                                color='BLUE'

                            if verbose:
                                CPRINT('Retrieved {}_Sprof.nc: {}'.format(wmo, FMT_secs(t.time() - time)), attrs=color)

                        except Exception as e:

                            CPRINT('We\'ve got to cope with an exception of type: {}.'.format(e), attrs='YELLOW')

                        shutil.move('{}_Sprof.nc'.format(wmo), VAR.sprofpath+'{}_Sprof.nc'.format(wmo))

                    elif answer == 'n' or answer == 'N':

                        if verbose:
                            CPRINT('Ok, skipping {}_Sprof.nc.'.format(wmo), attrs='BLUE')

                server.cwd('..')
            server.cwd('..')

        else:

            CPRINT('Some of your WMOs arent found on any dac available at ftp.ifremer.fr.', attrs='YELLOW')

    if verbose:
        CPRINT('All files downloaded. Quitting server.', attrs='BLUE')
    server.quit()


def RF_files(verbose=True):
    '''
    Refreshes the S profile files under sprofpath global variable of path.

    ____

    Author: T. Hermilly\n
    Contact: thomas.hermilly@ird.fr\n
    Last update: 2023-02-10
    :return: None
    '''
    def monthindex(str):

        list = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        res = False
        month = None

        for m in list:

            if m in str:
                res = True
                month = m

        i = np.nan
        if month is not None:
            i = 0
            while not(str[i:i+3]==month):
                i+=1

        return i

    def str2dt(str, year):

        dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9,
                'Oct': 10, 'Nov': 11, 'Dec': 12}

        return dt.datetime(year=year, month=dict[str[:3]], day=int(str[-2:]))

    files = os.listdir(VAR.sprofpath)
    wmo_list = [int(file[:-9]) for file in files]

    locs = GET_dacs(wmo_list, verbose=False)
    dacs = np.unique(locs)

    if len(files)>0:

        if verbose:
            CPRINT('Connecting to ftp.ifremer.fr...', attrs='CYAN', end='\r')

        server = flib.FTP('ftp.ifremer.fr', 'anonymous', 'anonymous@ifremer.fr')
        server.encoding = 'utf-8'

        if verbose:
            CPRINT('Successfully connected to server. Navigating through a sea of folders...', attrs='CYAN',  end='\r')

        server.cwd('~')
        server.cwd('ifremer/argo/dac')

        for dac in dacs:

            sub_wmo_list = list(np.array(wmo_list)[locs==dac])

            if verbose:
                CPRINT('Entering dac {}...'.format(dac), attrs='CYAN', end='\r')

            server.cwd(dac)

            if verbose:
                CPRINT('Currently working in /ifremer/argo/dac/{}.'.format(dac), attrs='BLUE')

            for wmo in sub_wmo_list:

                localage = round((os.times()[-1] - os.path.getctime(VAR.sprofpath+'{}_Sprof.nc'.format(wmo)))/3600/24, 1)

                server.cwd(str(wmo))
                lines = []
                server.retrlines('LIST', callback=lines.append)

                date = dt.date.today()
                for line in lines:
                    if 'Sprof.nc' in line:
                        ind = monthindex(line)
                        lessthan6months = line[ind+9]==':'
                        year = date.year

                        if not lessthan6months:
                            year = int(line[ind+8:ind+12])
                        elif date.month <=6:
                            if line[ind:ind+3] in ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                                year = date.year-1

                        date = str2dt(line[ind:ind+6], year)

                ftpage = (dt.datetime.today()-date).days

                if ftpage < localage:

                    f = open('{}_Sprof.nc'.format(wmo), 'wb')

                    try:

                        if verbose:
                            CPRINT('Downloading {}_Sprof.nc...'.format(wmo), attrs='CYAN', end='\r')
                        time = t.time()
                        server.retrbinary(r'RETR {}_Sprof.nc'.format(wmo), f.write)
                        f.close()

                        if t.time()-time > 30.:
                            color = 'yellow'
                        else:
                            color='blue'

                        if verbose:
                            CPRINT('Retrieved {}_Sprof.nc: {}'.format(wmo, FMT_secs(t.time() - time)), attrs='BLUE')

                    except Exception as e:

                        if verbose:
                            CPRINT('We\'ve got to cope with an exception of type: {}.'.format(e), attrs='YELLOW')

                    shutil.move('{}_Sprof.nc'.format(wmo), VAR.sprofpath+'{}_Sprof.nc'.format(wmo))

                else:

                    if verbose:
                        CPRINT('{}_Sprof.nc has no new version. FTP age: {:.0f} days / local file age: '
                          '{:.0f} days. '.format(wmo, ftpage, localage), attrs='BLUE')

                server.cwd('..')

            server.cwd('..')
            if verbose:
                CPRINT('Files from {} dac are all fresh now.'.format(dac), attrs='BLUE')

        if verbose:
            CPRINT('Files from all required dacs are fresh now. Quitting server.', attrs='BLUE')
        server.quit()

    else:

        if verbose:
            CPRINT('No files in directory. You should probably download the files first using dl_Sprof.',
                   attrs='YELLOW')


def DL_chloroday(start, appkey=VAR.myappkey, stop=dt.datetime.now(), step=10., verbose=True):

    '''
    Downloads the MODIS global CHL maps from start date to stop date with a step of step days. You can get your appkey
    at: https://oceandata.sci.gsfc.nasa.gov/appkey/
    :param start: The start date. (datetime or str 'YYYY-MM-DD')
    :param appkey: The appkey associated to your Nasa account. (str, default is my appkey :((( )
    :param stop: The stop date. (datetime or str 'YYYY-MM-DD', default is today)
    :param step: The time step in days. (float, default is 10.)
    :param verbose: Whether to print info to the console. (bool, default is True)
    :return: None
    '''

    start, stop = FMT_date(start, 'dt', verbose=False), FMT_date(stop, 'dt', verbose=False)
    date = start
    suffix = {'0': 'th', '1': 'st', '2': 'nd', '3': 'rd', '4': 'th',
              '5': 'th', '6': 'th', '7': 'th', '8': 'th', '9': 'th'}
    failed = []
    size = 0.
    k = 0

    while date <= stop:

        if not 'AQUA_MODIS.{:04d}{:02d}{:02d}.L3m.DAY.CHL.chlor_a.9km.nc'.format(date.year, date.month, date.day) \
                   in os.listdir(VAR.chloropath):

            try:

                if verbose:
                    CPRINT('Downloading MODIS L3 mapped chlorophyll on {}{}, {} (downloaded {}Mb already){}'
                          .format(date.strftime('%B %d'), suffix[date.strftime('%d')[-1]], date.strftime('%y'),
                                  int(size), LOADOTS(k)), attrs='CYAN', end='\r')

                ulib.request.urlretrieve('https://oceandata.sci.gsfc.nasa.gov/ob/getfile/AQUA_MODIS.{:04d}{:02d}{:02d}.'
                                         'L3m.DAY.CHL.chlor_a.9km.nc?appkey={}'
                                         .format(date.year, date.month, date.day, appkey),
                                         VAR.chloropath +'AQUA_MODIS.{:04d}{:02d}{:02d}.L3m.DAY.CHL.chlor_a.9km.nc'
                                         .format(date.year, date.month, date.day))
                
                size += os.path.getsize(VAR.chloropath +'AQUA_MODIS.{:04d}{:02d}{:02d}.L3m.DAY.CHL.chlor_a.9km.nc'
                                        .format(date.year, date.month, date.day)) / 1e6

            except ulib.error.HTTPError:

                failed.append(date)

        date += dt.timedelta(step)
        k += 1

    if verbose:

        date -= dt.timedelta(step)
        failed_str = ''
        for d in failed:
            failed_str += '{}{}, {} - '.format(d.strftime('%B %d'), suffix[d.strftime('%d')[-1]], d.strftime('%y'))
        if len(failed_str) > 3:
            failed_str = failed_str[:-3]
            CPRINT('Failed to find files at following dates: ' + failed_str + '.', attrs='YELLOW')

        CPRINT('Downloaded files until {}{}, {} ({}Mb downloaded)'.format(date.strftime('%B %d'),
                                                                          suffix[date.strftime('%d')[-1]],
                                                                          date.strftime('%y'), int(size)), attrs='BLUE')


def DL_MEI(verbose=True):
    '''
    Downloads monthly Multivariate Enso Index under VAR.meipath and formats it as a table with years as rows and months
    as columns.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :param verbose: Wether to print info to the console. (bool, default is True)
    :return: None
    '''

    ti = t.time()

    ulib.request.urlretrieve('https://psl.noaa.gov/enso/mei/data/meiv2.data', VAR.meipath + 'meiv2_temp.data')

    with open(VAR.meipath+'meiv2.data', 'w') as ff:

        ff.write('YR/M,     1,     2,     3,     4,     5,     6,     7,     8,     9,    10,    11,    12')

        with open(VAR.meipath+'meiv2_temp.data', 'r') as mei:

            mei.readline()

            for text in mei:

                line = np.array(CUT_line(text, ' '))
                line = line[line != '']

                if len(line) > 1:

                    if not text[0] == 'M':
                        st = ''
                        for e in line:
                            st +='\n{:.0f}'.format(float(e)) if float(e)>1000.\
                                else ',  {:.2f}'.format(float(e)) if float(e) > 0. \
                                else ', -9999' if float(e)<-100. \
                                else ', {:.2f}'.format(float(e)) if float(e) < 0. \
                                else ',  0.00'
                        ff.write(st)
                    else:
                        break

        ff.close()

    os.remove(VAR.meipath + 'meiv2_temp.data')

    TINFO(ti, 2., 'Downloaded MEI and saved under {}'.format(VAR.meipath), verbose)