import argparse
import sys
import os
import phoebe
from phoebe import wd
from phoebe.parameters import tools
import matplotlib.pyplot as plt
import numpy as np

logger = phoebe.get_basic_logger()

if __name__=="__main__":
    
    #-- initialize the parser and subparsers
    parser = argparse.ArgumentParser(description='Run Wilson-Devinney.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #-- load defaults from file if available
    if sys.argv[1:] and os.path.isfile(sys.argv[-1]):
        try:
            ps,lc,rv = wd.lcin_to_ps(sys.argv[-1],version='wd2003')
        except TypeError:
            ps,lc,rv = wd.lcin_to_ps(sys.argv[-1],version='wdphoebe')
        
    #-- else, load defaults from parameters.definitions
    else:
        ps = phoebe.PS(frame='wd',context='root')
        lc = phoebe.PS(frame='wd',context='lc')
        rv = phoebe.PS(frame='wd',context='rv')
    
    #-- add positional arguments
    parser.add_argument('input_file', metavar='input_file', type=str, nargs='?',
                   default=None,
                   help='Wilson-Devinney input file')
    
    #-- add arguments to parser
    parser_ps = parser.add_argument_group('Root','System or global parameters')
    parser_lc = parser.add_argument_group('LC','Light curve parameters')
    parser_rv = parser.add_argument_group('RV','Radial velocity curve parameters')
    
    for iparser,ips in zip([parser_ps,parser_lc,parser_rv],[ps,lc,rv]):
        for jpar in ips:
            if ips.context == 'root':
                name = '--{}'.format(jpar)
            else:
                name = '--{}_{}'.format(ips.context,jpar)
            parameter = ips.get_parameter(jpar)
            
            help_message = parameter.get_description()
            if parameter.has_unit():
                help_message += ' [{}]'.format(parameter.get_unit())
            iparser.add_argument(name, default=ips[jpar], help=help_message)
    
    parser.add_argument('--do_phoebe', action='store_true')
    
    #-- parse the arguments
    args = vars(parser.parse_args())
    
    #-- get the input file
    input_file = args.pop('input_file')
    do_phoebe = args.pop('do_phoebe', False)
    
    #-- override the defaults
    for par in args:
        if par[:3]=='lc_' and par[3:] in lc:
            lc[par[3:]] = args[par]
        elif par[:3]=='rv_' and par[3:] in rv:
            rv[par[3:]] = args[par]
        else:
            ps[par] = args[par]
    
    #-- report the final parameters
    logger.info('\n'+str(ps))
    logger.info('\n'+str(lc))
    logger.info('\n'+str(rv))
    
    #-- compute the light curve
    curve,params = wd.lc(ps,request='curve',light_curve=lc,rv_curve=rv)
    
    #-- compare with phoebe if needed
    if do_phoebe:
        comp1,comp2,binary = wd.wd_to_phoebe(ps,lc,rv)
        star1,lcdep1,rvdep1 = comp1
        star2,lcdep2,rvdep2 = comp2
        crit_times = tools.critical_times(binary)
        mesh1 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching',delta=0.075,alg='c')
        mesh2 = phoebe.ParameterSet(frame='phoebe',context='mesh:marching',delta=0.075,alg='c')
        star1 = phoebe.BinaryRocheStar(star1,binary,mesh1,pbdep=[lcdep1,rvdep1])
        star2 = phoebe.BinaryRocheStar(star2,binary,mesh2,pbdep=[lcdep2,rvdep2])
        wd_bbag = phoebe.BodyBag([star1,star2])
        mpi = phoebe.ParameterSet('mpi', np=4)
        phoebe.universe.serialize(wd_bbag,filename='check.pars')
        phoebe.observe(wd_bbag, curve['indeps'], lc=True, rv=True, mpi=mpi)
                #extra_func=[phoebe.observatory.ef_binary_image],
                #extra_func_kwargs=[dict(select='teff',cmap=plt.cm.spectral)])
    
    #-- now do something with it
    plt.figure()
    plt.subplot(121)
    plt.plot(curve['indeps'],curve['lc']/curve['lc'].mean(),'ro-',lw=2,label='WD')
    plt.subplot(122)
    plt.plot(curve['indeps'],curve['rv1'],'ro-',lw=2,label='WD(a)')
    plt.plot(curve['indeps'],curve['rv2'],'ro--',lw=2,label='WD(b)')
    
    
    if do_phoebe:
        plt.subplot(121)
        lc = wd_bbag.get_synthetic(category='lc').asarray()
        plt.plot(lc['time'],lc['flux']/lc['flux'].mean(),'ko-')
        
        n = np.floor((crit_times - lc['time'][0])/ps['period'])
        plt.axvline(crit_times[0] - n[0]*ps['period'],color='g',lw=2,label='Periastron passage')
        plt.axvline(crit_times[1] - n[1]*ps['period'],color='c',lw=2,label='Superior conjunction')
        plt.axvline(crit_times[2] - n[2]*ps['period'],color='m',lw=2,label='Inferior conjunction')
        plt.legend(loc='best').get_frame().set_alpha(0.5)
        
        plt.subplot(122)
        rv = wd_bbag[0].get_synthetic(category='rv').asarray()
        plt.plot(rv['time'],rv['rv']*8.049861,'ko-',label='PH')
        rv = wd_bbag[1].get_synthetic(category='rv').asarray()
        plt.plot(rv['time'],rv['rv']*8.049861,'ko--',label='PH')
        
        n = np.floor((crit_times - lc['time'][0])/ps['period'])
        plt.axvline(crit_times[0] - n[0]*ps['period'],color='g',lw=2,label='Periastron passage')
        plt.axvline(crit_times[1] - n[1]*ps['period'],color='c',lw=2,label='Superior conjunction')
        plt.axvline(crit_times[2] - n[2]*ps['period'],color='m',lw=2,label='Inferior conjunction')
        plt.legend(loc='best').get_frame().set_alpha(0.5)
    
    plt.show()