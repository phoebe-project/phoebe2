import numpy as np
import tempfile
import os

def pf_parse(pfoutname):
    try:
        pfx, pfy = np.loadtxt(pfoutname, unpack=True)
    except:
        pfx, pfy = np.loadtxt(pfoutname, delimiter=',', unpack=True)

    f = open(pfoutname, "r")
    flines = f.readlines()
    for i, x in enumerate(flines):
        if 'Weighted' in x:
            break
    i += 3
    pf_coeffs = []

    pf_coeffs.append(list(map(float, flines[i].split()[1:]))) #pf_coeffs[0] 1-4
    pf_coeffs.append(list(map(float, flines[i+1].split()[1:])))
    pf_coeffs.append(list(map(float, flines[i+2].split()[1:])))
    pf_coeffs.append(list(map(float, flines[i+3].split()[1:])))

    pshift = float(flines[i+5].split('(')[1].split(',')[0])
    # the shift needs to be applied to the phases array, but this will place it back into [-0.5, 0.5] range
    pfx -= pshift

    f.close()

    pf_knots = []
    for i in range(0,4):
        pf_knots.append(pf_coeffs[i].pop(0))

    return pfx, pfy, pf_coeffs, pf_knots, pshift

def pf_run (phases, fluxes, sigmas, order=2, iters=1000, vertices=200, chainlen=10, step=None, knots=None, coeffs_lines=[19,20,21,22], verbose=False):
    '''
    runs polyfit and returns fit and parameters
    kp.pf_run(phase, flux, weight, order, iters, vertices, step='auto', knots='auto', outfile="lcout.pf", coeffs_lines=[19,20,21,22])
    input: phase, flux, weight, order, iters, vertices, step ('auto' or float), knots ('auto' or float), outfile (optional), coeffs_lines (optional - if coefficients are found on lines other thant 19-22
    output: pfx, pfy, pf_knots, pf_coeffs
    note: currently only works for 4 breaks (should work for any order, but not tested)
    '''
    #prepare options
    options = "-o %d -i %d -n %d --chain-length %d" % (order, iters, vertices, chainlen)
    if step is None:
        options += " --find-steps"
    else:
        options += " -s %lf" % step
    if knots is None:
        options += " --find-knots"
    else:
        options += " -k %s" % str(knots).strip('[').strip(']') #to account for if passed in as array

    options += " --apply-pshift"

    #prepare polyfit input file
    pfin, pfinname = tempfile.mkstemp()
    pfout, pfoutname = tempfile.mkstemp()
    os.close(pfout) #polyfit writes to this, we'll open again later

    np.savetxt(pfinname, np.asarray([phases, fluxes, sigmas]).T)

    #run polyfit
    # print("polyfit {} {} > {}".format(options, pfinname, pfoutname))
    os.system ("polyfit {} {} > {}".format(options, pfinname, pfoutname))

    #parse output
    pfx, pfy, pf_coeffs, pf_knots, pshift = pf_parse(pfoutname)

    #cleanup
    os.remove(pfinname)
    os.remove(pfoutname)

    return pfx, pfy, pf_knots, pf_coeffs, pshift
