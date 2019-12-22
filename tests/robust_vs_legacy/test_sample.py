plot = False


import sys
import phoebe
import numpy as np
import signal
from threading import Timer
import thread, time, sys
if plot:
    try:
        import matplotlib.pyplot as plot
    except ImportError:
        plot = False

from multiprocessing import Process
def run_with_limited_time(func, args, kwargs, time):
    """Runs a function with time limit

    :param func: The function to run
    :param args: The functions args, given as tuple
    :param kwargs: The functions keywords, given as dict
    :param time: The time limit in seconds
    :return: True if the function ended successfully. False if it was terminated.
    """
    p = Process(target=func, args=args, kwargs=kwargs)
    p.start()
    p.join(time)
    if p.is_alive():
        p.terminate()
        return False

    return True


def _draw_random_lin(min_, max_):
    return np.random.random()*(max_-min_)+min_

def _draw_random_lin_inv(min_, max_):
    val = _draw_random_lin(min_, max_)

    if np.random.random() < 0.5:
        return 1./val
    return val

def _draw_random_log(min_, max_):
    log = _draw_random_lin(np.log10(min_), np.log10(max_))
    return 10**log

def draw_period():
    return _draw_random_log(0.3,100)

def draw_sma():
    return _draw_random_log(1,100)

def draw_incl():
    return _draw_random_lin(0,360)

def draw_ecc():
    return _draw_random_lin(0.0, 1.0)

def draw_q():
    return _draw_random_lin_inv(0.1, 1)

def draw_longan():
    return _draw_random_lin(0,360)


def draw_rpole():
    return _draw_random_lin(0.1, 10.0)

def draw_teff():
    return _draw_random_lin(1000., 30000.)

def draw_syncpar():
    return _draw_random_lin_inv(0.1, 1)


def chi2(b, dataset, model1='phoebe1model', model2='phoebe2model'):

    ds = b.get_dataset(dataset) - b.get_dataset(dataset, method='*dep')
    if ds.method=='lc':
        depvar = 'fluxes'
    elif ds.method=='rv':
        depvar = 'rvs'
    else:
        raise NotImplementedError("chi2 doesn't support dataset method: '{}'".format(ds.method))

    chi2 = 0.0
    for comp in ds.components if len(ds.components) else [None]:
        if comp=='_default':
            continue
        # phoebe gives nans for RVs when a star is completely eclipsed, whereas
        # phoebe1 will give a value.  So let's use nansum to just ignore those
        # regions of the RV curve
        print("***", depvar, dataset, model1, model2, comp)
        chi2 += np.nansum((b.get_value(qualifier=depvar, dataset=dataset, model=model1, component=comp, context='model')\
            -b.get_value(qualifier=depvar, dataset=dataset, model=model2, component=comp, context='model'))**2)

    return chi2



if __name__ == '__main__':
    argv = sys.argv

    if len(argv) <= 1:
        print("USAGE: test_sample.py [results] [sample]")
        print("results: plot the results from the existing log file")
        print("sample: enter an infinite loop of sampling and appending to the log file (ctrl+c to exit)")

    if 'results' in argv:
        import matplotlib.pyplot as plt

        period, sma, incl, ecc, q, long_an,\
            pri_rpole, sec_rpole,\
            pri_teff, sec_teff,\
            pri_syncpar, sec_syncpar,\
            chi2lc, chi2rv = np.loadtxt('sample.results', unpack=True)

        params = {'period': period, 'sma': sma, 'incl': incl, 'ecc': ecc, 'q': q, 'long_an': long_an,\
                    'pri_rpole': pri_rpole, 'sec_rpole': sec_rpole,\
                    'pri_teff': pri_teff, 'sec_teff': sec_teff,\
                    'pri_syncpar': pri_syncpar, 'sec_syncpar': sec_syncpar}

        for name, chi2 in {'lc': chi2lc, 'rv': chi2rv}.items():
            plt.cla()
            plt.hist(chi2[chi2>=0])
            plt.xlabel(name)
            plt.show()

        for name, param in params.items():
            plt.cla()
            plt.plot(param, chi2lc, 'b.')
            plt.plot(param, chi2rv, 'r.')
            plt.xlabel(name)
            plt.ylabel('chi2')
            plt.show()



    if 'sample' in argv:
        f = open('sample.results', 'a')
        N = 501

        b = phoebe.Bundle.default_binary()
        # NOTE: we'll set the time arrays later
        b.add_dataset('LC', dataset='lc01')
        b.add_dataset('RV', dataset='rv01')


        b.add_compute(compute='phoebe', atm='extern_planckint')
        b.add_compute('legacy', compute='phoebe1', atm='blackbody')

        # TODO: eventually test over ld_coeffs - but right now ld seems to be broken vs legacy
        b.set_value_all('ld_coeffs', [0,0])

        i = 0
        while True:
            try:
                period = draw_period()
                b.set_value('period@binary', period)

                sma = draw_sma()
                b.set_value('sma@binary', sma)

                incl = draw_incl()
                b.set_value('incl@binary', incl)

                ecc = draw_ecc()
                b.set_value('ecc@binary', ecc)

                long_an = draw_longan()
                b.set_value('long_an@binary', long_an)

                q = draw_q()
                b.set_value('q@binary', q)

                pri_rpole = draw_rpole()
                b.set_value('rpole@primary', pri_rpole)

                sec_rpole = draw_rpole()
                b.set_value('rpole@secondary', sec_rpole)

                pri_teff = draw_teff()
                b.set_value('teff@primary', pri_teff)

                sec_teff = draw_teff()
                b.set_value('teff@secondary', sec_teff)

                pri_syncpar = draw_syncpar()
                b.set_value('syncpar@primary', pri_syncpar)

                sec_syncpar = draw_syncpar()
                b.set_value('syncpar@secondary', sec_syncpar)

                # TODO: abun, alb, gravb, ld_coeffs, passband, atm

                params = (period, sma, incl, ecc, q, long_an,\
                        pri_rpole, sec_rpole,\
                        pri_teff, sec_teff,\
                        pri_syncpar, sec_syncpar)
                params_vals = " ".join([str(p) for p in params])

                print("*** TRYING: {}".format(params))

                # we really want to sample from 0-1 in phase, so let's reset the time arrays
                b.set_value_all('time@dataset', np.linspace(0, period, N))

                print("*** PHOEBE 2")
                try:
                    b.run_compute('phoebe', model='phoebe2model')
                except ValueError:
                    phoebe2_passed = False
                    print("phoebe2 failed overflow checks, skipping")
                    continue
                else:
                    phoebe2_passed = True


                print("*** PHOEBE 1")
                # Let's skip overflow tests (within PHOEBE 2) on this first run
                # so that we will let PHOEBE 1 run forever and fail.  We do this
                # so that we can also test the failing cases vs each other
                phoebe1_passed = run_with_limited_time(b.run_compute,
                                                       ('phoebe1',),
                                                       {'model': 'phoebe1model',
                                                        'skip_checks': True},
                                                       time=10)
                if phoebe1_passed:
                    # this is horrendously hideous - the only way I could get the
                    # timeout to work means that we can't access the output, so
                    # b isn't updated.  But now we know that it won't timeout, so
                    # we can rerun compute
                    b.run_compute('phoebe1',
                                  model='phoebe1model',
                                  skip_checks=True)




                if phoebe1_passed != phoebe2_passed:
                    if phoebe1_passed:
                        # phoebe failed, phoebe 1 passed
                        chi2lc = -1
                        chi2rv = -1
                    else:
                        # phoebe passed, phoebe 1 failed
                        chi2lc = -2
                        chi2rv = -2
                elif phoebe1_passed and phoebe2_passed:
                    chi2lc = chi2(b, 'lc01')/N
                    # TODO: need to deal with NaNs in the phoebe RVs when occulted
                    # with proximity effects phoebe1 reverts to dynamical during these times
                    chi2rv = chi2(b, 'rv01')/N
                else:
                    # then both failed, perfect score!
                    # NOTE: this probably won't happen anymore since legacy
                    # is now segfaulting and so we're bailing once PHOEBE2
                    # fails overflow checks
                    chi2lc = 0.0
                    chi2rv = 0.0



                results = "{} {} {}".format(params_vals, chi2lc, chi2rv)
                print(results)
                f.write(results+"\n")
                i += 1

                print("completed {} iterations".format(i))

                if plot:
                    b.plot(show=True)

            except KeyboardInterrupt:
                break
            except ValueError, err:
                print(err)
                import pdb; pdb.set_trace()  # breakpoint d8e3dc28 //


        print("*** CLOSING LOG FILE ***")
        f.close()
