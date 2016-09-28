import phoebe as phb2
import phoebeBackend as phb1
import numpy as np
import matplotlib.pyplot as plt
import os

phoebe.devel_on()

def legacy_test(filename='default.phoebe'):

    # load phoebe 1 file

    dir = os.path.dirname(os.path.realpath(__file__))

    phb1.init()
    phb1.configure()
    phb1.open(os.path.join(dir, filename))

    #load phoebe2 file
    b = phb2.Bundle.from_legacy(os.path.join(dir, filename))

    # create time array and get datasets


    per = b['period@orbit'].value
#    time_rv = np.linspace(0, per, 4)
#    time_lc = np.linspace(0, per, 100)


    lcs = b.get_dataset(kind='lc').datasets
    lcs = lcs[::-1]
    rvs = b.get_dataset(kind='rv').datasets
    rvs = rvs[::-1]

    # phb2 compute

    fluxes = []
    vels = []
    vels2 = []

    for x in range(len(lcs)):
        # load file
        datafile = phb1.getpar('phoebe_lc_filename', x)
        data = np.loadtxt(os.path.join(dir, datafile))

        time = b.filter(dataset=lcs[x], qualifier='times').get_value()
        print "checking time in "+lcs[x]
        assert(np.all(time==data[:,0]))
        flux = b.filter(dataset=lcs[x], qualifier='fluxes').get_value()
        print "checking flux in "+lcs[x]
        assert(np.all(flux==data[:,1]))
        sigma = b.filter(dataset=lcs[x], qualifier='sigmas').get_value()
        print "checking sigma in "+lcs[x]
        assert(np.all(sigma==data[:,2]))
        #calculate lc
        flux, mesh = phb1.lc(tuple(data[:,0].tolist()), x, 1)
        fluxes.append(flux)

    rvno = phb1.getpar('phoebe_rvno')
    prim = 0
    sec = 0
    for x in range(rvno):
        component = phb1.getpar('phoebe_rv_dep', x).split(' ')[0].lower()
        a = int(x/2.)
        datafile = phb1.getpar('phoebe_rv_filename', x)
        data = np.loadtxt(os.path.join(dir, datafile))
        time = b.filter(dataset=rvs[a], qualifier='times', component=component).get_value()
        print "checking time in "+rvs[a]
        assert(np.all(time==data[:,0]))
        rv = b.filter(dataset=rvs[a], qualifier='rvs', component=component).get_value()
        print "checking rv in "+rvs[a]
        assert(np.all(rv==data[:,1]))
        sigma = b.filter(dataset=rvs[a], qualifier='sigmas', component=component).get_value()
        print "checking sigma in "+rvs[a]
        assert(np.all(sigma==data[:,2]))


        if component == 'primary':

            rv1 = np.array(phb1.rv1(tuple(data[:,0].tolist()), prim))
            vels.append(rv1)
            prim = prim+1
        else:
            rv2 = np.array(phb1.rv2(tuple(data[:,0].tolist()), sec))
            vels2.append(rv2)
            sec = sec+1

    b.run_compute(kind='legacy')
    for x in range(len(lcs)):
        lc2 = b.filter('fluxes', context='model', dataset=lcs[x]).get_value()
        time = b.filter('times', context='model', dataset=lcs[x]).get_value()
        print("comparing lightcurve "+str(lcs[x]))
        assert(np.allclose(fluxes[x], lc2, atol=1e-5))

    for x in range(rvno):
        prim = 0
        sec = 0
        component = phb1.getpar('phoebe_rv_dep', x).split(' ')[0].lower()
        rv2 = b.filter('rvs', component=component, context='model').get_value()
        time = b.filter('times', component=component, context='model').get_value()
        a = int(x/2.)
        if component == 'primary':
           print("trying primary rv at "+str(rvs[a]))
           assert(np.allclose(vels[prim], rv2, atol=1e-5))
           prim= prim+1

        else:
            print("trying secondary rv at "+str(rvs[a]))
            assert(np.allclose(vels2[sec], rv2, atol=1e-5))
            sec = sec+1

#            assert(np.all(vels2[x] == rv2))

#        if np.any((vels1[x]-rv2) != 0):
#            print("lightcurve "+str(lcs[x])+" failed")
#        else:
#            print("lightcurve "+str(lcs[x])+" passed")

    return

if __name__ == '__main__':

    logger= phb2.logger()
    filename = 'default.phoebe'
    legacy_test(filename)


