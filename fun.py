# ~/miniconda3/envs/genenv/bin/python3.9
# -*- coding: utf-8 -*-

'''These are just personal functions. Should not work if you are not me.

____

Author: T. Hermilly\n
Contact: thomas.hermilly@ird.fr\n
Last update: 2023-02-10
'''

# Imports


from downloads import *


# Functions


def INVOKE_BLINKY(loading='Loading', end='Done.', animate = VAR.activate_fun, col='PURPLE'):
    '''
    Blinky is my spiritual master. Once one of the greatest with regards to oceanic sciences, he is now at his lowest,
    absolutely fucked up as hell. That is what happens when you power your efforts with crack at 7am and cocaine at
    noon, to submit abstracts or get reviewed on time. He survived though, and came back as a mentor, as his species is
    extremely resilient to low-quality drugs.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :return: None
    '''
    def speeding(time, N):

        coef = lambda x: 1 - np.sin(x / time * 3.4 * np.pi)/2.5

        times = [0.]
        for i in range(N + 1):
            times.append(times[-1] + time / N * coef(times[-1]))

        times = np.array(times)

        return times[1:] - times[:-1]

    strlist_start = [
    loading+'.~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~                                     \n'
    '                 oOOo    Oldy Blinky                        \n'
    '                 0OOo                         \n'
    '                  o0O       .\\\'|_.-                         \n'
    '                   oo     .\'  \'  /_                         \n'
    '                     o  .-"    -.   \'>   _                         \n'
    '                    .- -. -.    \'. /    / |__                         \n'
    '                   .-.--.-.       \' >  /   /                         \n'
    '                  (o( o( o )       \_."   <\                          \n'
    '                   \'-\'-\'\'-\'             )  |>                         \n'
    '                 (       _.-\'-.   ._\.__  _\                          \n'
    '                  \'----"/--.__.-) _-     \|                           \n',
    loading+'..~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~                                     \n'
    '   |O|          oOOo    Oldy Blinky                        \n'
    '                0OOo                         \n'
    '                 o0O       .\\\'|_.-                         \n'
    '                  oo     .\'  \'  /_                         \n'
    '                   o   .-"    -.   \'>                         \n'
    '                   .- -. -.    \'. /    /|_                         \n'
    '                  .-.--.-.       \' >  /  /                         \n'
    '                 (o( o( o )       \_."  <                         \n'
    '                  \'-\'-\'\'-\'            ) <                         \n'
    '                (       _.-\'-.   ._\.  _\                          \n'
    '                 \'----"/--.__.-) _-  \|                           \n',
    loading+'...~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-                                     \n'
    '   |G|         oOOo    Oldy Blinky                        \n'
    '   |O|         0OOo                         \n'
    '                o0O       .\\\'|_.-                         \n'
    '                oo      .\'  \'  /_                         \n'
    '                 o    .-"    -.   \'>                         \n'
    '                  .- -. -.    \'. /  <|\                          \n'
    '                 .-.--.-.       \' > /  |                         \n'
    '                (o( o( o )       \_."  <                         \n'
    '                 \'-\'-\'\'-\'           ) |                         \n'
    '               (       _.-\'-.   ._\. _\                          \n'
    '                \'----"/--.__.-) _- \|                           \n',
    loading+'.~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~                                     \n'
    '   |R|        oOOo    Oldy Blinky                        \n'
    '   |G|        0OOo                         \n'
    '   |O|         o0O       .\\\'|_.-                         \n'
    '               oo      .\'  \'  /_                         \n'
    '                 o   .-"    -.   \'>                         \n'
    '                 .- -. -.    \'. /    /|_                         \n'
    '                .-.--.-.       \' >  /  /                         \n'
    '               (o( o( o )       \_."  <                         \n'
    '                \'-\'-\'\'-\'            ) <                         \n'
    '              (       _.-\'-.   ._\.  _\                          \n'
    '               \'----"/--.__.-) _-  \|                           \n',
    loading+'..~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~                                     \n'
    '   |A|       oOOo    Oldy Blinky                        \n'
    '   |R|       0OOo                         \n'
    '   |G|        o0O       .\\\'|_.-                         \n'
    '   |O|         oo     .\'  \'  /_                         \n'
    '                 o  .-"    -.   \'>   _                         \n'
    '                .- -. -.    \'. /    / |__                         \n'
    '               .-.--.-.       \' >  /   /                         \n'
    '              (o( o( o )       \_."   <\                          \n'
    '               \'-\'-\'\'-\'             )  |>                         \n'
    '             (       _.-\'-.   ._\.__  _\                          \n'
    '              \'----"/--.__.-) _-     \|                           \n',
    loading+'...~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-                                     \n'
    '    |       oOOo    Oldy Blinky                        \n'
    '   |A|      0OOo                         \n'
    '   |R|       o0O       .\\\'|_.-                         \n'
    '   |G|        oo     .\'  \'  /_                         \n'
    '   |O|         o   .-"    -.   \'>   _                         \n'
    '               .- -. -.    \'. /    / |_                         \n'
    '              .-.--.-.       \' >  / l|                         \n'
    '             (o( o( o )       \_."   ||                          \n'
    '              \'-\'-\'\'-\'              |                         \n'
    '            (       _.-\'-.   ._\._ _\                          \n'
    '             \'----"/--.__.-) _-    \|                           \n',
    loading+'.~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~                                     \n'
    '    -      oOOo    Oldy Blinky                        \n'
    '    |      0OOo                         \n'
    '   |A|       o0O      .\\\'|_.-                         \n'
    '   |R|        oo    .\'  \'  /_                         \n'
    '   |G|       o    .-"    -.   \'>  .                         \n'
    '   |O|        .- -. -.    \'. /    /|                         \n'
    '             .-.--.-.       \' >  /_|                         \n'
    '            (o( o( o )       \_."  /                          \n'
    '             \'-\'-\'\'-\'            < |                         \n'
    '           (       _.-\'-.   ._\. _\                          \n'
    '            \'----"/--.__.-) _-   \|                           \n',
    loading+'..~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~                                     \n'
    '          oOOo    Oldy Blinky                        \n'
    '    -     0OOo                         \n'
    '    |       o0O      .\\\'|_.-                         \n'
    '   |A|       oo    .\'  \'  /_                         \n'
    '   |R|       o   .-"    -.   \'>   _                         \n'
    '   |G|       .- -. -.    \'. /    / |_                         \n'
    '   |O|      .-.--.-.       \' >  / l|                         \n'
    '           (o( o( o )       \_."   ||                          \n'
    '            \'-\'-\'\'-\'              |                         \n'
    '          (       _.-\'-.   ._\._ _\                          \n'
    '           \'----"/--.__.-) _-    \|                         \n'
               ]

    strlist = [
    loading+'...~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -     o0O       .\\\'|_.-                         \n'
    '    |      oo     .\'  \'  /_                         \n'
    '   |A|       o  .-"    -.   \'>   _                         \n'
    '   |R|      .- -. -.    \'. /    / |__                         \n'
    '   |G|     .-.--.-.       \' >  /   /                         \n'
    '   |O|    (o( o( o )       \_."   <\                          \n'
    '           \'-\'-\'\'-\'             )  |>                         \n'
    '         (       _.-\'-.   ._\.__  _\                          \n'
    '          \'----"/--.__.-) _-     \|                           \n',
    loading+'.~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -     o0O       .\\\'|_.-                         \n'
    '    |      oo     .\'  \'  /_                         \n'
    '   |A|      o   .-"    -.   \'>                         \n'
    '   |R|      .- -. -.    \'. /    /|_                         \n'
    '   |G|     .-.--.-.       \' >  /  /                         \n'
    '   |O|    (o( o( o )       \_."  <                         \n'
    '           \'-\'-\'\'-\'            ) <                         \n'
    '         (       _.-\'-.   ._\.  _\                          \n'
    '          \'----"/--.__.-) _-  \|                           \n',
    loading+'..~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -     o0O       .\\\'|_.-                         \n'
    '    |     oo      .\'  \'  /_                         \n'
    '   |A|     o    .-"    -.   \'>                         \n'
    '   |R|      .- -. -.    \'. /  <|\                          \n'
    '   |G|     .-.--.-.       \' > /  |                         \n'
    '   |O|    (o( o( o )       \_."  <                         \n'
    '           \'-\'-\'\'-\'           ) |                         \n'
    '         (       _.-\'-.   ._\. _\                          \n'
    '          \'----"/--.__.-) _- \|                           \n',
    loading+'...~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -     o0O       .\\\'|_.-                         \n'
    '    |     oo      .\'  \'  /_                         \n'
    '   |A|      o   .-"    -.   \'>                         \n'
    '   |R|      .- -. -.    \'. /    /|_                         \n'
    '   |G|     .-.--.-.       \' >  /  /                         \n'
    '   |O|    (o( o( o )       \_."  <                         \n'
    '           \'-\'-\'\'-\'            ) <                         \n'
    '         (       _.-\'-.   ._\.  _\                          \n'
    '          \'----"/--.__.-) _-  \|                           \n',
    loading+'.~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -     o0O       .\\\'|_.-                         \n'
    '    |      oo     .\'  \'  /_                         \n'
    '   |A|       o  .-"    -.   \'>   _                         \n'
    '   |R|      .- -. -.    \'. /    / |__                         \n'
    '   |G|     .-.--.-.       \' >  /   /                         \n'
    '   |O|    (-( -( - )       \_."   <\                          \n'
    '           \'-\'-\'\'-\'             )  |>                         \n'
    '         (       _.-\'-.   ._\.__  _\                          \n'
    '          \'----"/--.__.-) _-     \|                           \n',
    loading+'..~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -     o0O       .\\\'|_.-                         \n'
    '    |      oo     .\'  \'  /_                         \n'
    '   |A|      o   .-"    -.   \'>   _                         \n'
    '   |R|      .- -. -.    \'. /    / |_                         \n'
    '   |G|     .-.--.-.       \' >  / l|                         \n'
    '   |O|    (o( o( o )       \_."   ||                          \n'
    '           \'-\'-\'\'-\'              |                         \n'
    '         (       _.-\'-.   ._\._ _\                          \n'
    '          \'----"/--.__.-) _-    \|                           \n',
    loading+'..~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -      o0O      .\\\'|_.-                         \n'
    '    |       oo    .\'  \'  /_                         \n'
    '   |A|     o    .-"    -.   \'>  .                         \n'
    '   |R|      .- -. -.    \'. /    /|                         \n'
    '   |G|     .-.--.-.       \' >  /_|                         \n'
    '   |O|    (-( -( - )       \_."  /                          \n'
    '           \'-\'-\'\'-\'            < |                         \n'
    '         (       _.-\'-.   ._\. _\                          \n'
    '          \'----"/--.__.-) _-   \|                           \n',
    loading+'...~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-                                     \n'
    '         oOOo    Oldy Blinky                        \n'
    '         0OOo                         \n'
    '    -      o0O      .\\\'|_.-                         \n'
    '    |       oo    .\'  \'  /_                         \n'
    '   |A|      o   .-"    -.   \'>   _                         \n'
    '   |R|      .- -. -.    \'. /    / |_                         \n'
    '   |G|     .-.--.-.       \' >  / l|                         \n'
    '   |O|    (o( o( o )       \_."   ||                          \n'
    '           \'-\'-\'\'-\'              |                         \n'
    '         (       _.-\'-.   ._\._ _\                          \n'
    '          \'----"/--.__.-) _-    \|                         \n'
               ]

    if animate:
        time = VAR.animtime
    else:
        time = None

    wisdom = ['Work less, sleep more. Lazyness is the key.\n*almost dies from laughing*',
              'Your attitude, not your aptitude, will det... I forgot.\n*stupidness tears*',
              'Blame supervisors, not personal skills.\n*runs away without dignity*',
              'RaaawrrRR... BRrrWaAwrWarRww.\n*needs to stop crack*',
              'Learn to like your own company. You have no other LOL.\n*LOL fish face*',
              'Success is made of many lots of struggle.\n*Remembers his lack of success*',
              'Oceanography is like crack: transcending.\n*Drools acid*',
              'Can you pass me the blue crystals over there?\nTrust me, it\'s very important.',
              'Let me tell you why you should let go:\n*Sniffs 12 lines of cocaine*']

    magic_char = '\033[F'
    ret_depth = magic_char * 13

    if time is not None:

        CPRINT('\n#########\n', attrs=col)
        CPRINT('\n' * 12)

        N = 25
        times = speeding(time, N)

        for i in range(N):

            if i < 8:
                CPRINT('{}{}'.format(ret_depth, strlist_start[i % 8]), attrs=col, end='', flush=True)
            else:
                CPRINT('{}{}'.format(ret_depth, strlist[i % 8]), attrs=col, end='', flush=True)
            t.sleep(times[i])

    CPRINT('{}{}'.format(magic_char*13,'                                                                \n'+
                                end + '~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-~~-                                     \n'
                                '         oOOo    Oldy Blinky                        \n'
                                '         0OOo                         \n'
                                '    -      o0O      .\\\'|_.-                         \n'
                                '    |       oo    .\'  \'  /_                         \n'
                                '   |A|      o   .-"    -.   \'>   _                          \n'
                                '   |R|      .- -. -.    \'. /    / |__                         \n'
                                '   |G|     .-.--.-.       \' >  /   /                         \n'
                                '   |O|    (o( o( o )       \_."   <\                           \n'
                                '           \'-\'-\'\'-\'             )  |>                       \n'
                                '         (       _.-\'-.   ._\.__  _\                         \n'
                                '          \'----"/--.__.-) _-     \|                         \n'
                                ), end='', attrs=col)
    CPRINT('\n')
    CPRINT(wisdom[np.random.randint(len(wisdom))], attrs=[col, 'ITALIC'])


def ADD_TODO():
    '''
    Allows user to add stuff the to-do list.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :return: None
    '''

    ROOT()

    os.chdir('..')
    os.chdir('..')

    with open('todo.txt', 'r') as todo:
        lines = todo.readlines()
    cats = []
    for line in lines:
        if line[:3] == '###':
            cats.append(line[4:])
    categories = ''
    for cat in cats:
        categories += cat[:-3] + ', '
    categories = categories[:-2]

    CPRINT('More work is fine I guess! What\'s the task category?\n(Categories: {})'.format(categories), end='\r',
           attrs='BLUE')
    cat = INP_timeout(30)

    ncat = 1
    for categ in cats:
        if cat in categ:
            cat = categ
            break
        else:
            ncat += 1

    if ncat == len(cats)+1:
        CPRINT('There is no such category for the moment. I will put your task under \'Autres\'.', attrs='BLUE')
        cat = 'Autres'

    CPRINT('What\'s the deadline so? (DD/MM, n to skip, U for urgent)', attrs='BLUE', end='\r')
    dead = INP_timeout(30)
    if dead == 'U' or dead == 'u':
        pass
    elif len(dead) < 3:
        dead = '   '
    elif not dead[2] == '/':
        dead = '   '

    if dead != '   ' and not(dead == 'U' or dead == 'u'):
        CPRINT('How many days do you need for that? (DD, n to skip)', attrs='BLUE', end='\r')
        duration = INP_timeout(30)
        if duration == 'n':
            duration = ''
    else:
        duration = ''
    CPRINT('Now, describe the task.', attrs='BLUE', end='\r')
    task = INP_timeout(60)

    ind = 0
    for line in lines:
        if not('###' in line and cat in line):
            ind += 1
        else:
            ind += 1
            break

    newlines = lines[:ind+1]

    if dead == 'U' or dead == 'u':
        ftask = '- (U) ' + task
    elif dead[2] == '/':
        try:
            ftask = '- ({}-{:02d}) '.format(dead, int(duration)) + task
        except Exception:
            ftask = '- ({}) '.format(dead) + task
    else:
        ftask = '- ' + task

    newlines.append(ftask)
    newlines = newlines + lines[ind:]

    with open('todo.txt', 'w') as todo:
        todo.writelines(newlines)

    todo.close()
    os.chdir('Python/ArgoData/')

    CPRINT('Added task.', attrs='BLUE')


def TODO():
    '''
    Allows user to update the to-do list.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :return: None
    '''

    def get_wmo(l):

        for i in range(21, len(l) - 1):
            if l[i:i + 2] == 'SR' or l[i:i + 2] == 'SD':
                try:
                    wmo = int(l[i + 2:i + 9])

                    return wmo

                except ValueError:

                    wmo = np.nan

                    return wmo

    def print_todo():

        os.chdir('..')
        os.chdir('..')

        file = open('todo.txt', 'r')
        file.readline()
        file.readline()

        CPRINT('\n')

        i = 0
        for l in file:

            n = len(l)

            if n > 150:
                for i in range(n // 150, -1, -1):
                    l = l[:100 * (i + 1)] + '\n' + l[100 * (i + 1):]

            if l.rstrip() != '':
                if l[0] == '#':
                    col = 'BLUE'
                elif l[3] == 'U':
                    col = 'RED'
                else:
                    col = 'BLUE'
                    if l[5] == '/':

                        try:
                            thresh = int(l[9:11])
                        except Exception:
                            thresh = 2

                        try:
                            month, day = int(l[6:8]), int(l[3:5])
                            today = dt.datetime.today()
                            if dt.datetime(today.year, month, day) - today > dt.timedelta(100):
                                date = dt.datetime(today.year + 1, month, day)
                            else:
                                date = dt.datetime(today.year, month, day)
                            if date - today < dt.timedelta(thresh):
                                col = 'RED'
                            elif date - today < dt.timedelta(2 * thresh):
                                col = 'YELLOW'
                            else:
                                col = 'GREEN'
                        except Exception:
                            pass

                if l[0] == '-':
                    i += 1
                    CPRINT('({}) '.format(i) + l, attrs=col)
                else:
                    CPRINT(l, attrs=col)

        file.close()

        os.chdir('Python/ArgoData/')

    ROOT()
    got_here = False

    print_todo()

    os.chdir('..')
    os.chdir('..')

    try:

        CPRINT('\nDid you get something done? (y/n)', attrs='BLUE', end='\r')
        ans1 = INP_timeout(60)

        if ans1 == 'Y' or ans1 == 'y':

            CPRINT('Good! Tell me what points have you checked? (i, j, k...)', attrs='BLUE', end='\r')
            ans = INP_timeout(60)
            done = CUT_line(ans, ',')
            done = [int(d) for d in done]

            with open('todo.txt', 'r') as todo:
                lines = todo.readlines()

            i = 0
            delete = []
            for l in lines:
                if l[0] == '-':
                    i += 1
                    if i in done:
                        delete.append(l)

            with open('todo.txt', 'w') as todo:
                for l in lines:
                    d = False
                    for text in delete:
                        if l.strip('\n') == text[:-1]:
                            d = True
                    if not(d):
                        todo.write(l)

        else:

            CPRINT('Pff.', attrs='BLUE', end='\r')
            t.sleep(.2)
            CPRINT('Pff..', attrs='BLUE', end='\r')
            t.sleep(.4)
            CPRINT('Pff...', attrs='BLUE', end='\r')
            t.sleep(1.)
            CPRINT('Pff... Skipping work again I guess!', attrs='BLUE')
            t.sleep(.5)

        os.chdir('Python/ArgoData/')
        got_here = True
        ans2_has_been_yes = False

        CPRINT('Is there anything new you have to do? (y/n)', attrs='BLUE', end='\r')
        ans2 = INP_timeout(60)

        while ans2 == 'Y' or ans2 == 'y':

            ans2_has_been_yes = True
            ADD_TODO()
            CPRINT('Now, is there again something more? (y/n)', attrs='BLUE', end='\r')
            ans2 = INP_timeout(60)

        if ans1 == 'Y' or ans1 == 'y' or ans2_has_been_yes:

            CPRINT('Ok, your new todo-list is the following:', attrs='BLUE', end='\r')
            print_todo()

        else:

            CPRINT('Ok, chill it through then.', attrs='BLUE')
            t.sleep(.5)

        CPRINT('\n#########\n\nNow you\'re all done with the to-do list.', attrs='BLUE', end='\r')

    except Exception as e:

        if not got_here:
            os.chdir('Python/ArgoData/')

        CPRINT('\n#########\n\nAyyyy... Something went wrong: {}'.format(e), attrs='YELLOW', end='\r')


def MORNING():
    '''
    Keeps my dude up to date. Checks on the to-do list, checks for new ARGO in the Pacific, downloads new files and
    refreshes already existing files.

    ____

    Written by: T. Hermilly

    Contact: thomas.hermilly@ird.fr.

    Last update: 2023-02-09
    :return: None
    '''

    def get_wmo(l):

        for i in range(21, len(l) - 1):
            if l[i:i + 2] == 'SR' or l[i:i + 2] == 'SD':
                try:
                    wmo = int(l[i + 2:i + 9])

                    return wmo

                except ValueError:

                    wmo = np.nan

                    return wmo

    TODO()

    CPRINT('\n\n#########\n', attrs='BLUE')

    #########

    CPRINT('Downloading index...', attrs='BLUE', end='\r')
    DL_index(verbose=False)

    #########

    CPRINT('Refreshing Sprof files...', attrs='BLUE', end='\r')
    RF_files(verbose=False)

    #########

    cur_wmos = open(VAR.indexpath + 'current_wmos.txt', 'r')

    cur_wmos.readline()
    wmo_list_light = CUT_line(cur_wmos.readline(), ',')
    wmo_list_light = [int(wmo) for wmo in wmo_list_light]

    cur_wmos.readline()
    cur_wmos.readline()
    wmo_list_nolight = CUT_line(cur_wmos.readline(), ',')
    wmo_list_nolight = [int(wmo) for wmo in wmo_list_nolight]

    wmo_list = wmo_list_nolight + wmo_list_light

    new_light = []
    new_nolight = []

    file = open(VAR.indexpath + 'argo_synthetic-profile_index.txt', 'r')

    l = file.readline()
    while l[0] == '#':
        l = file.readline()

    for l in file:

        list = CUT_line(l, ',')

        if list[2] != '' and list[4] == 'P' and float(list[2]) < 5. and 'CHLA' in list[7]:

            wmo = get_wmo(l)

            if ('DOWNWELLING_PAR' in l) and not(wmo in wmo_list_light) and not(wmo in new_light):
                new_light.append(wmo)
            elif not('DOWNWELLING_PAR' in l) and not(wmo in wmo_list_nolight) and not(wmo in new_nolight):
                new_nolight.append(wmo)

    no_new_nolight = False
    no_new_light = False
    if len(new_nolight) > 0:
        CPRINT('New ARGO in the South Pacific!\n\nWMOs are:', attrs='BLUE')
        CPRINT(new_nolight, attrs='BLUE')
        CPRINT('\n')
    else:
        no_new_nolight = True

    if len(new_light) > 0:
        CPRINT('New BGC ARGO in the South Pacific!\n\nWMOs are:', 'yellow')
        CPRINT(new_light, 'yellow')
        CPRINT('\n')
    else:
        no_new_light = True

    if no_new_nolight and no_new_light:
        CPRINT('No new ARGO in the Pacific.', attrs='BLUE')

    file.close()
    cur_wmos.close()

    if not(no_new_light and no_new_nolight):

        CPRINT('Downloading new files...', attrs='BLUE')
        DL_Sprof(new_light+new_nolight, yestoall=True, verbose=False)

        cur_wmos = open(VAR.indexpath + 'current_wmos.txt', 'w')
        cur_wmos.write('Floats with PAR:\n')
        line_wmos = wmo_list_light + new_light
        line = ''
        for wmo in line_wmos:
            line += str(wmo) + ', '
        line = line[:-2]
        cur_wmos.write(line)

        cur_wmos.write('\n')
        cur_wmos.write('\n')

        cur_wmos.write('Floats with PAR:\n')
        line_wmos = wmo_list_nolight + new_nolight
        line = ''
        for wmo in line_wmos:
            line += str(wmo) + ', '
        line = line[:-2]
        cur_wmos.write(line)

        cur_wmos.close()

    INVOKE_BLINKY()
