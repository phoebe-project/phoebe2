import phoebe as phb2

try:
    import phoebe_legacy as phb1
except ImportError:
    try:
        import phoebeBackend as phb1
    except ImportError:
        _has_phb1 = False
    else:
        _has_phb1 = True
else:
    _has_phb1 = True
import numpy as np
import os



def _legacy_test(filename='default.phoebe', verbose=True):

    #locate file
    dir = os.path.dirname(os.path.realpath(__file__))
    #load in phoebe parameter file
    params = np.loadtxt(os.path.join(dir, filename), dtype='str', delimiter = '=',
    converters = {0: lambda s: s.strip(), 1: lambda s: s.strip()})

    lcno = np.int(params[:,1][list(params[:,0]).index('phoebe_lcno')])
    rvno = np.int(params[:,1][list(params[:,0]).index('phoebe_rvno')])



    #load phoebe2 file
    if verbose:
        print(dir)
        print(os.path.join(dir, filename))
    b = phb2.Bundle.from_legacy(os.path.join(dir, filename), add_compute_legacy=True)
    b.rename_component('primary', 'cow')
    b.rename_component('secondary', 'pig')

    lcs = b.get_dataset(kind='lc').datasets
    rvs = b.get_dataset(kind='rv').datasets


    lc_id = []
    rv_id = []
    fluxes = []
    vels = []
    vels2 = []


    for x in range(lcno):
        # load file
        lc_add = '['+str(x+1)+']'
        id = params[:,1][list(params[:,0]).index('phoebe_lc_id'+lc_add)].strip('"')
        lc_id.append(id)

        datafile = params[:,1][list(params[:,0]).index('phoebe_lc_filename'+lc_add)].strip('"')

        if verbose: print(id)
        #load data file

        time_phb1, flux_phb1, err_phb1 = np.loadtxt(os.path.join(dir, datafile), unpack=True)

        # get third column type
        err_val = params[:,1][list(params[:,0]).index('phoebe_lc_indweight'+lc_add)].strip('"')

        # retrieve data from bundle
        time = b.filter(dataset=id, qualifier='times').get_value()
        flux = b.filter(dataset=id, qualifier='fluxes').get_value()
        sigma = b.filter(dataset=id, qualifier='sigmas').get_value()

        #compare to datafile
        if verbose: print("x={}, datafile={}, lcs[x]={}".format(x, datafile, lcs[x]))
        assert(np.all(time==time_phb1))
        if verbose: print("checking flux in "+str(lcs[x]))
        if verbose: print("x={}, datafile={}, lcs[x]={}".format(x, datafile, lcs[x]))
        assert(np.all(flux==flux_phb1))
        if verbose: print("checking sigma in "+str(lcs[x]))
        if err_val == 'Standard deviation':
            assert(np.all(sigma==err_phb1))
        else:
            val = np.sqrt(1/err_phb1)
            assert(np.allclose(sigma, val, atol=1e-7))

        #grab legacy ld_coeffs
        ld1x1 = params[:,1][list(params[:,0]).index('phoebe_ld_lcx1'+lc_add+'.VAL')]
        ld1y1 = params[:,1][list(params[:,0]).index('phoebe_ld_lcy1'+lc_add)]


        ld1x2 = params[:,1][list(params[:,0]).index('phoebe_ld_lcx2'+lc_add+'.VAL')]
        ld1y2 = params[:,1][list(params[:,0]).index('phoebe_ld_lcy2'+lc_add)]

        ld_coeffs1 = [float(ld1x1), float(ld1y1), float(ld1x2), float(ld1y2)]

        #grab phoebe2 ld_coeffs
        ldx1, ldy1 = b.filter(dataset=lcs[x], qualifier='ld_coeffs', component='cow').get_value()
        ldx1, ldy1 = b.filter(dataset=id, qualifier='ld_coeffs', component='cow').get_value()
        ldx2, ldy2 = b.filter(dataset=id, qualifier='ld_coeffs', component='pig').get_value()
        ld_coeffs2 = [ldx1, ldy1, ldx2, ldy2]
        if verbose:
            print("ld_coeffs1", ld_coeffs1)
            print("ld_coeffs2", ld_coeffs2)

        #compare
        if verbose: print("checking ld coeffs in "+str(lcs[x]))
        assert(np.all(ld_coeffs1==ld_coeffs2))
        #calculate lc

        fluxes.append(flux_phb1)

    prim = 0
    sec = 0

    for x in range(rvno):
        if verbose: print(x, rvno)
        rv_add = '['+str(x+1)+']'

        if verbose: print('rvs')

        err_val = params[:,1][list(params[:,0]).index('phoebe_rv_indweight'+rv_add)].strip('"')
        id = params[:,1][list(params[:,0]).index('phoebe_rv_id'+rv_add)].strip('"')
        comp = params[:,1][list(params[:,0]).index('phoebe_rv_dep'+rv_add)].strip('"').split(' ')[0].lower()

        rv_id.append(id)

        if verbose:
            print("id", id)
            print("comp",comp)

        if comp == 'primary':
            comp_name = 'cow'
        elif comp == 'secondary':
            comp_name = 'pig'
        a = int(x/2.)
        if verbose: print("loop iteration", a)


        datafile = params[:,1][list(params[:,0]).index('phoebe_rv_filename'+rv_add)].strip('"')

        time_phb1, rv_phb1, err_phb1 = np.loadtxt(os.path.join(dir, datafile), unpack=True)

        time = b.filter(dataset=id, qualifier='times', component=comp_name).get_value()
        rv = b.filter(dataset=id, qualifier='rvs', component=comp_name).get_value()
        sigma = b.filter(dataset=id, qualifier='sigmas', component=comp_name).get_value()

        if verbose: print("checking time in "+str(rvs[a]))
        assert(np.all(time==time_phb1))
        if verbose: print("checking rv in "+str(rvs[a]))
        if verbose: print("a={}, datafile={}, rvs[a]={}".format(a, datafile, rvs[a]))
        assert(np.all(rv==rv_phb1))
        sigma = b.filter(dataset=id, qualifier='sigmas', component=comp_name).get_value()
        if verbose: print("checking sigma in "+str(rvs[a]))

        if err_val == 'Standard deviation':
            assert(np.all(sigma==err_phb1))
        else:
            val = np.sqrt(1/err_phb1)
            assert(np.allclose(sigma, val, atol=1e-7))

        if comp_name == 'cow':
            rv_comp = '['+str(prim+1)+']'
            vels.append(rv_phb1)

            ld1x1 = params[:,1][list(params[:,0]).index('phoebe_ld_rvx1'+rv_comp)]
            ld1y1 = params[:,1][list(params[:,0]).index('phoebe_ld_rvy1'+rv_comp)]

            ld1x2 = params[:,1][list(params[:,0]).index('phoebe_ld_rvx2'+rv_comp)]
            ld1y2 = params[:,1][list(params[:,0]).index('phoebe_ld_rvy2'+rv_comp)]

            ld_coeffs1 = [float(ld1x1), float(ld1y1), float(ld1x2), float(ld1y2)]

            ldx1, ldy1 = b.filter(dataset=id, qualifier='ld_coeffs', component='cow').get_value()
            ldx2, ldy2 = b.filter(dataset=id, qualifier='ld_coeffs', component='pig').get_value()
            ld_coeffs2 = [ldx1, ldy1, ldx2, ldy2]
            if verbose:
                print("checking ld coeffs in primary "+str(rvs[a]))
            assert(np.all(ld_coeffs1==ld_coeffs2))
            prim = prim+1

        else:
            rv_comp = '['+str(sec+1)+']'
            vels2.append(rv_phb1)

            ld1x1 = params[:,1][list(params[:,0]).index('phoebe_ld_rvx1'+rv_comp)]
            ld1y1 = params[:,1][list(params[:,0]).index('phoebe_ld_rvy1'+rv_comp)]

            ld1x2 = params[:,1][list(params[:,0]).index('phoebe_ld_rvx2'+rv_comp)]
            ld1y2 = params[:,1][list(params[:,0]).index('phoebe_ld_rvy2'+rv_comp)]

            ld_coeffs1 = [float(ld1x1), float(ld1y1), float(ld1x2), float(ld1y2)]

            ldx1, ldy1 = b.filter(dataset=id, qualifier='ld_coeffs', component='cow').get_value()
            ldx2, ldy2 = b.filter(dataset=id, qualifier='ld_coeffs', component='pig').get_value()
            ld_coeffs2 = [ldx1, ldy1, ldx2, ldy2]
            if verbose:
                print("checking ld coeffs in secondary "+str(rvs[a]))
            assert(np.all(ld_coeffs1==ld_coeffs2))
            sec = sec+1

    if _has_phb1:
        b.run_compute(kind='legacy')

        for x in range(len(lcs)):
            lc2 = b.filter('fluxes', context='model', dataset=lc_id[x]).get_value()
            time = b.filter('times', context='model', dataset=lc_id[x]).get_value()
            if verbose: print("comparing lightcurve "+str(lcs[x]))

            assert(np.allclose(fluxes[x], lc2, atol=1e-5))

        for x in range(rvno):
            rv_add = '['+str(x+1)+']'
            if verbose: print(x, rvno)
            prim = 0
            sec = 0

            comp = params[:,1][list(params[:,0]).index('phoebe_rv_dep'+rv_add)].strip('"').split(' ')[0].lower()

            if verbose: print('comp', comp)

            if comp == 'primary':
                comp_name = 'cow'
            elif comp == 'secondary':
                comp_name = 'pig'

            rv2 = b.filter('rvs', component=comp_name, context='model').get_value()
            time = b.filter('times', component=comp_name, context='model').get_value()

            a = int(x/2.)
            if verbose: print("comp name", comp_name)

            if comp_name == 'cow':
                if verbose:
                    print("trying primary rv at "+str(rvs[a]))
                assert(np.allclose(vels[prim], rv2, atol=1e-5))
                prim= prim+1

            else:
                if verbose:
                    print("trying secondary rv at "+str(rvs[a]))
                assert(np.allclose(vels2[sec], rv2, atol=1e-5))
                sec = sec+1

    if verbose: print("comparing bolometric ld coeffs")

    ldxbol1 = params[:,1][list(params[:,0]).index('phoebe_ld_xbol1')]
    ldybol1 = params[:,1][list(params[:,0]).index('phoebe_ld_ybol1')]
    ldxbol2 = params[:,1][list(params[:,0]).index('phoebe_ld_xbol2')]
    ldybol2 = params[:,1][list(params[:,0]).index('phoebe_ld_ybol2')]

    ld1x2 = params[:,1][list(params[:,0]).index('phoebe_ld_rvx2'+rv_comp)]
    ld1y2 = params[:,1][list(params[:,0]).index('phoebe_ld_rvy2'+rv_comp)]
    ld_coeffs1 = [float(ldxbol1), float(ldybol1), float(ldxbol2), float(ldybol2)]

    ldx1, ldy1 = b.filter(qualifier='ld_coeffs_bol', component='cow').get_value()
    ldx2, ldy2 = b.filter(qualifier='ld_coeffs_bol', component='pig').get_value()

    ld_coeffs2 = [ldx1, ldy1, ldx2, ldy2]
    assert(np.all(ld_coeffs1==ld_coeffs2))



    return




def test_default(verbose=False):
    return _legacy_test('default.phoebe', verbose=verbose)

#def test_weighted(verbose=False):
#    return _legacy_test('weight.phoebe', verbose=verbose)

def test_contact(verbose=False):
    return _legacy_test('contact.phoebe', verbose=verbose)


if __name__ == '__main__':
    print('DEFAULT 1')
    test_default(verbose=True)
#    print('contact 1')
#    test_weighted(verbose=True)
    test_contact(verbose=True)
    print('DEFAULT 2')
    test_default(verbose=True)
