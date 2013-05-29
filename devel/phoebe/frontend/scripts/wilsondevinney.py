import argparse
import sys
import os
import phoebe
from phoebe import wd
import matplotlib.pyplot as plt

logger = phoebe.get_basic_logger()

if __name__=="__main__":
    
    #-- initialize the parser and subparsers
    parser = argparse.ArgumentParser(description='Run Wilson-Devinney.',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #-- load defaults from file if available
    if sys.argv[1:] and os.path.isfile(sys.argv[-1]):
        ps,lc,rv = wd.lcin_to_ps(sys.argv[-1],version='wd2003')
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
    
    #-- parse the arguments
    args = vars(parser.parse_args())
    
    input_file = args.pop('input_file')
    
    for par in args:
        if par[:3]=='lc_' and par[3:] in lc:
            lc[par[3:]] = args[par]
        elif par[:3]=='rv_' and par[3:] in rv:
            rv[par[3:]] = args[par]
        else:
            ps[par] = args[par]
            
    curve,params = wd.lc(ps,request='curve',light_curve=lc,rv_curve=rv)
    
    print(ps)
    print(lc)
    print(rv)
    
    plt.figure()
    plt.subplot(121)
    plt.plot(curve['indeps'],curve['lc']/curve['lc'].mean(),'ro-',lw=2,label='WD')
    plt.subplot(122)
    plt.plot(curve['indeps'],curve['rv1'],'ro-',lw=2,label='WD(a)')
    plt.plot(curve['indeps'],curve['rv2'],'ro--',lw=2,label='WD(b)')
    
    plt.show()