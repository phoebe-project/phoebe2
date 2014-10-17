"""
Interface to the Wilson-Devinney code
=======================================

B{Example 1}: Generate a light curve with the WD code starting from an ``*.active``
WD parameter file.

Convert a .active parameter file to ParameterSet instances.
        
>>> pset,lcset,rvset = lcin_to_ps('../../../sandbox/pyphoebe/test_suite/wd_vs_pyphoebe/test01lcin.active',version='wd2003')
>>> lcset['phinc'] = 0.01
    
Calculate light curve and generate an image:

>>> curve,params = lc(pset,request='curve',light_curve=lcset,rv_curve=rvset)
>>> image,params = lc(pset,request='image',light_curve=lcset,rv_curve=rvset)


Make a plot: first subplot is the light curve, second plot the image in the
plane of the sky, third subplot are the radial velocity curves.

>>> p = pl.figure()
>>> p = pl.subplot(121)
>>> p = pl.plot(curve['indeps'],curve['lc'],'ko-')
>>> p = pl.subplot(222,aspect='equal')
>>> p = pl.plot(image['y'],image['z'],'ko',ms=1)
>>> p = pl.subplot(224)
>>> p = pl.plot(curve['indeps'],curve['rv1'],'ko-')
>>> p = pl.plot(curve['indeps'],curve['rv2'],'ro-')

]include figure]]images/wd_lc.png]

B{Example 2}: Generate a light curve with the WD code starting from default
parametersets:

>>> pset = pars.ParameterSet(frame='wd',context='root',ref='myroot')
>>> lcset = pars.ParameterSet(frame='wd',context='lc',ref='mylc')
>>> rvset = pars.ParameterSet(frame='wd',context='rv',ref='myrv')

Calculate light curve and generate an image:

>>> curve,params = lc(pset,request='curve',light_curve=lcset,rv_curve=rvset)
>>> image,params = lc(pset,request='image',light_curve=lcset,rv_curve=rvset)


Make a plot: first subplot is the light curve, second plot the image in the
plane of the sky, third subplot are the radial velocity curves.

>>> p = pl.figure()
>>> p = pl.subplot(121)
>>> p = pl.plot(curve['indeps'],curve['lc'],'ko-')
>>> p = pl.subplot(222,aspect='equal')
>>> p = pl.plot(image['y'],image['z'],'ko',ms=1)
>>> p = pl.subplot(224)
>>> p = pl.plot(curve['indeps'],curve['rv1'],'ko-')
>>> p = pl.plot(curve['indeps'],curve['rv2'],'ro-')

]include figure]]images/wd_lc3.png]

For completeness, here are all the parameters defined in the C{root} context,
and their units:

>>> print(pset)
    name mybinary                    --       wd Common name of the binary
   model 2                           --       wd Morphological constraints
    hjd0 55124.89703                HJD -     wd Origin of time
  period 22.1891087                   d -     wd Orbital period
    dpdt 0.0                        d/d -     wd First time derivative of period
  pshift 0.0776                      -- -     wd Phase shift
     sma 11.0104                   Rsol -     wd Semi-major axis
      rm 0.89747                     -- -     wd Mass ratio (secondary over primary)
    incl 87.866                     deg -     wd Inclination angle
     vga 0.0                       km/s -     wd Center-of-mass velocity
     ecc 0.28319                     -- -     wd Eccentricity
   omega 5.696919                   rad -     wd Initial argument of periastron for star 1
domegadt 0.0                      rad/s -     wd First time derivative of periastron
      f1 1.0                         -- -     wd Primary star synchronicity parameter
      f2 1.0                         -- -     wd Secondary star synchronicity parameter
   teff1 0.8105                  10000K -     wd Primary star effective temperature
   teff2 0.7299                  10000K -     wd Secondary star effective temperature
    pot1 7.3405                      -- -     wd Primary star surface potential
    pot2 7.58697                     -- -     wd Secondary star surface potential
    met1 0.0                         -- -     wd Primary star metallicity
    met2 0.0                         -- -     wd Secondary star metallicity
    alb1 1.0                         -- -     wd Primary star surface albedo
    alb2 0.864                       -- -     wd Secondary star surface albedo
    grb1 0.964                       -- -     wd Primary star gravity brightening
    grb2 0.809                       -- -     wd Secondary star gravity brightening
ld_model 2                           --       wd Limb darkening model
ld_xbol1 0.512                       --       wd Primary star bolometric LD coefficient x
ld_ybol1 0.0                         --       wd Primary star bolometric LD coefficient y
ld_xbol2 0.549                       --       wd Secondary star bolometric LD coefficient x
ld_ybol2 0.0                         --       wd Secondary star bolometric LD coefficient y
   mpage 1                           --       wd Output type of the WD code
    mref 1                           --       wd Reflection treatment
    nref 2                           --       wd Number of reflections
   icor1 1                           --       wd Turn on prox and ecl fx on prim RV
   icor2 1                           --       wd Turn on prox and ecl fx on secn RV
   stdev 0.0                         --       wd Synthetic noise standard deviation
   noise 3                           --       wd Noise scaling
    seed 100000001.0                 --       wd Seed for random generator
     ipb 0                           --       wd Compute second stars Luminosity
   ifat1 1                           --       wd Atmosphere approximation primary
   ifat2 1                           --       wd Atmosphere approximation secondary
      n1 70                          --       wd Grid size primary
      n2 70                          --       wd Grid size secondary
     the 0.0                         --       wd Semi-duration of primary eclipse in phase units
   mzero 0.0                         --       wd Zeropoint mag (vert shift of lc)
  factor 1.0                         --       wd Zeropoint flux (vert shift of lc)
     wla 0.592                       --       wd Reference wavelength (in microns)
  atmtab phoebe_atmcof.dat           --       wd Atmosphere table
  plttab phoebe_atmcofplanck.dat     --       wd Planck table
  ifsmv1 0                           --       wd Spots on star 1 co-rotate with the star
  ifsmv2 0                           --       wd Spots on star 2 co-rotate with the star
   xlat1 [300.0]                     --       wd Primary spots latitudes
  xlong1 [0.0]                       --       wd Primary spots longitudes
  radsp1 [1.0]                       --       wd Primary spots radii
 tempsp1 [1.0]                       --       wd Primary spots temperatures
   xlat2 [300.0]                     --       wd Secondary spots latitudes
  xlong2 [0.0]                       --       wd Secondary spots longitudes
  radsp2 [1.0]                       --       wd Secondary spots radii
 tempsp2 [1.0]                       --       wd Secondary spots temperatures
   label myroot                      --       wd Name of the system

In the C{lc} context:   
   
>>> print(lcset)
    filter 7        --       wd Filter name
indep_type 2        --       wd Independent modeling variable
     indep [ 0.]    --       wd Time/phase template or time/phase observations
   ld_lcx1 0.467    --       wd Primary star passband LD coefficient x
   ld_lcx2 0.502    --       wd Secondary star passband LD coefficient x
   ld_lcy1 0.0      --       wd Primary star passband LD coefficient y
   ld_lcy2 0.0      --       wd Secondary star passband LD coefficient y
       hla 8.11061  --       wd LC primary passband luminosity
       cla 4.4358   --       wd LC secondary passband luminosity
      opsf 0.0      --       wd Opacity frequency function
       el3 0.0      --       wd Third light contribution
 el3_units 1        --       wd Units of third light
    phnorm 0.25     --       wd Phase of normalisation
    jdstrt 0.0      --       wd Start Julian date
     jdend 1.0      --       wd End Julian date
     jdinc 0.1      --       wd Increment Julian date
    phstrt 0.0      --       wd Start Phase
     phend 1.0      --       wd End phase
     phinc 0.01     --       wd Increment phase
       ref mylc     --       wd Name of the observable
         
In the C{rv} context:

>>> print(rvset)
ld_rvx1 0.5   --       wd Primary RV passband LD coefficient x
ld_rvx2 0.5   --       wd Secondary RV passband LD coefficient x
ld_rvy1 0.5   --       wd Primary RV passband LD coefficient y
ld_rvy2 0.5   --       wd Secondary RV passband LD coefficient y
  vunit 1.0   --       wd Unit of radial velocity (km/s)
    ref myrv  --       wd Name of the observable
"""
#-- standard library
import os
import logging
import pickle
import copy
from collections import OrderedDict
#-- third party modules
import pylab as pl
import numpy as np
from scipy.optimize import nnls
#-- own modules
from phoebe.parameters import parameters as pars
from phoebe.parameters import definitions as defs
from phoebe.parameters import datasets
from phoebe.utils import utils
from phoebe.backend import processing
from phoebe.atmospheres import limbdark
try:
    from phoebe.wd import fwd
except ImportError:
    pass
    #print('fwd not compiled, please install')


logger = logging.getLogger('WD')


def lc(binary_parameter_set,request='curve',light_curve=None,rv_curve=None,filename=None):
    """
    Generate a light curve or image with the Wilson-Devinney code.
    
    >>> bps = pars.ParameterSet(frame="wd",context='root')
    
    If you're not really interested in comparing it with observations, you can
    just leave all the defaults. If no light curve parameters are attached to
    C{bps}, C{lc} will take the defaults and the user will be warned.
    
    >>> output1,params = lc(bps)
    >>> output2,params = lc(bps,request='image')
    
    Keys of the output are:

        - C{indeps}, C{lc}, C{rv1}, C{rv2} when C{request='curve'}
        - C{x}, C{y} when C{request='image'}
    
    On the other hand, you might want to give your own filter (or limb darkening
    parameters or whatever) and your own RV curve. In that case, you could do
    
    >>> mylc = pars.ParameterSet(definitions=defs.defs,frame='wd',context='lc')
    >>> output3,params = lc(bps,light_curve=mylc)
    
    >>> p = pl.figure()
    >>> p = pl.subplot(131)
    >>> p = pl.plot(output1['indeps'],output1['lc'],'ko-')
    >>> p = pl.plot(output3['indeps'],output3['lc'],'ko-')
    >>> p = pl.subplot(132)
    >>> p = pl.plot(output1['indeps'],output1['rv1'],'ko-')
    >>> p = pl.plot(output1['indeps'],output1['rv2'],'ro-')
    >>> p = pl.subplot(133,aspect='equal')
    >>> p = pl.plot(output2['y'],output2['z'],'ko',ms=1)
    
    ]include figure]]images/wd_lc2.png]
    
    The output dictionary C{params} contains:
    
        - C{L1}:     star 1 passband luminosity
        - C{L2}:     star 2 passband luminosity
        - C{M1}:     star 1 mass in solar masses
        - C{M2}:     star 2 mass in solar masses
        - C{R1}:     star 1 radius in solar radii
        - C{R2}:     star 2 radius in solar radii
        - C{Mbol1}:  star 1 absolute magnitude
        - C{Mbol2}:  star 2 absolute magnitude
        - C{logg1}:  star 1 log gravity
        - C{logg2}:  star 2 log gravity
        - C{SBR1}:   star 1 polar surface brightness
        - C{SBR2}:   star 2 polar surface brightness
        - C{phsv}:   star 1 potential
        - C{pcsv}:   star 2 potential
    
    @param binary_parameter_set: collection of all binary parameters
    @type binary_parameter_set: instance of BinarayParameterSet
    @param request: request the light curve and radial velocity curves, or an image ('curve' or 'image')
    @type request: string, one of 'curve' or 'image'
    @param light_curve: light curve parameterSet
    @type light_curve: ParameterSet
    @param rv_curve: radial velocity parameterSet
    @type rv_curve: ParameterSet
    @param filename: supply a filename if the old interface to WD is desired. An output file will be written and read in
    @type filename: str
    @return: synthesized light curve or image, and dictionary containing extra parameters
    @rtype: record array, dict
    """
    #-- we changed the definition of this keyword:
    if request == 'curve':
        request = 'lc'
    #-- the user can give a BinaryParameterSet or a dictionary
    if isinstance(binary_parameter_set,pars.ParameterSet):
        #-- convert the binary parameter set to the PYWD framework
        bps = binary_parameter_set.copy()
    
        #-- select the name of the light curve. If the light curve doesn't
        #   exist, warn the user and make a default one.
        if light_curve is None:
            #logger.warning("no light curve defined, adding default settings")
            light_curve = pars.ParameterSet(definitions=defs.defs,frame='wd',context='lc')
        #-- do the same for the radial velocities
        if rv_curve is None:
            #logger.warning("no rv curve defined, adding default settings")
            rv_curve = pars.ParameterSet(definitions=defs.defs,frame='wd',context='rv')
        #-- if not converted to wd, do it now
        if not 'wd' in bps.frame:
            bps.propagate('wd') #--> probably obsolete
    #-- the user can give a filename of a lcin file.
    elif isinstance(binary_parameter_set,str):
        bps,light_curve,rv_curve = pars.lcin_to_ps(binary_parameter_set)
    else:
        bps = binary_parameter_set.copy()

    #-- set some extra parameters for WD
    request_wd = ['lc','rv1','rv2','image'].index(request)+1
    
    #-- if request==4 (mesh), you cannot give a range of phases. The mesh
    #   will be computed at the 'phstrt' only.
    if request == 'image':
        bps['mpage'] = 'image'
        light_curve['indep'] = light_curve['indep'][:1]
    #-- these dummy variables are needed as input for wd's LC but are not used.
    if not 'jdstrt' in light_curve:
        light_curve.add(dict(qualifier='jdstrt',description='Start Julian date',    repr='%14.6f',cast_type=float,value=0,frame='wd',context='lc'))
        light_curve.add(dict(qualifier='jdend', description='End Julian date',      repr='%14.6f',cast_type=float,value=1.0,frame='wd',context='lc'))
        light_curve.add(dict(qualifier='jdinc', description='Increment Julian date',repr='%14.6f',cast_type=float,value=0.1,frame='wd',context='lc'))
    if not 'phstrt' in light_curve:
        light_curve.add(dict(qualifier='phstrt',description='Start Phase'          ,repr='%8.6f',cast_type=float,value=0,frame='wd',context='lc'))
        light_curve.add(dict(qualifier='phend', description='End phase',            repr='%8.6f',cast_type=float,value=1,frame='wd',context='lc'))
        light_curve.add(dict(qualifier='phinc', description='Increment phase',      repr='%8.6f',cast_type=float,value=0.01,frame='wd',context='lc'))
    
    if request in ['rv1', 'rv2']:
        indeps = rv_curve['indep']
    else:
        indeps = light_curve['indep']
    
    lc  = np.zeros_like(indeps)
    rv1 = np.zeros_like(indeps)
    rv2 = np.zeros_like(indeps)    
    vertno = len(indeps)
    skycoy = np.zeros(100000)
    skycoz = np.zeros(100000)
    params = np.zeros(14)
    
    #-- check the atm and plt tables
    if not os.path.isfile(bps['atmtab']):
        bps['atmtab'] = os.path.join(os.path.dirname(os.path.abspath(__file__)),bps['atmtab'])
    if not os.path.isfile(bps['plttab']):
        bps['plttab'] = os.path.join(os.path.dirname(os.path.abspath(__file__)),bps['plttab'])
    if not os.path.isfile(bps['atmtab']):
        raise IOError('No such file %s'%(bps['atmtab']))
    if not os.path.isfile(bps['plttab']):
        raise IOError('No such file %s'%(bps['plttab']))
    
    if filename is not None:
        #-- spots are not supported
        for par in ['xlat1','xlong1','radsp1','tempsp1','xlat2','xlong2','radsp2','tempsp2']:
            if not par in bps: # assume there is nothing given about the spots:
                bps.add(Parameter(par))
        #-- we can derive the number of spots from the given input
        if not 'nsp1' in bps:
            bps.add(dict(qualifier='nsp1',description='Number of spots on primary',repr='%d',cast_type=int,value=len(bps['xlat1'])))
        if not 'nsp2' in bps:
            bps.add(dict(qualifier='nsp2',description='Number of spots on secondary',repr='%d',cast_type=int,value=len(bps['xlat2'])))
                    
        #-- define the order of the parameters to pass to fwd.wrcli
        args1 = ['mpage','nref','mref','ifsmv1','ifsmv2','icor1','icor2','ld_model','indep_type','hjd0','period','dpdt','pshift','stdev','noise',
                 'seed','jdstrt','jdend','jdinc','phstrt','phend','phinc','phnorm','model','ipb','ifat1','ifat2','n1','n2','perr0','dperdt',
                 'the','vunit','ecc','sma','f1','f2','vga','incl','grb1','grb2','met1','teff1','teff2','alb1','alb2','pot1','pot2','rm',
                 'ld_xbol1','ld_xbol2','ld_ybol1','ld_ybol2','filter','hla','cla','ld_lcx1','ld_lcx2','ld_lcy1','ld_lcy2','el3','opsf','mzero',
                 'factor','wla','nsp1','xlat1','xlong1','radsp1','tempsp1','nsp2','xlat2','xlong2','radsp2','tempsp2']
        #-- now get the values of these parameters
        args1 = [filename] + [bps[key] for key in args1]
        
        #-- define the order of the parameters to pass to fwd.lc
        deps = np.zeros_like(indeps)
        args2 = [bps['atmtab'],bps['plttab'],filename,request_wd,vertno,light_curve['el3'],
                 indeps,deps,skycoy,skycoz,params]
        
        #-- call the relevant functions from WD: first write the parameter file,
        #   then run LC
        fwd.wrlci(*args1)
        indeps,deps,skycoy,skycoz,params = fwd.lc(*args2)
        if   request=='lc':    lc  = deps
        elif request=='rv1':   rv1 = deps
        elif request=='rv2':   rv2 = deps
    else:
        #-- set some variables which are necessary in the FORTRAN code, but which
        #   are derivable from the others.
        #xlat = np.hstack((bps['xlat1'], bps['xlat2']))
        #xlong = np.hstack((bps['xlong1'], bps['xlong2']))
        #radsp = np.hstack((bps['radsp1'], bps['radsp2']))
        #tempsp = np.hstack((bps['tempsp1'], bps['tempsp2']))
        
        #-- spots are not supported
        xlat,xlong,radsp,tempsp =  np.array([300.,  300.]), np.array([0.,  0.]), np.array([1.,  1.]), np.array([1.,  1.])
        if not 'nsp1' in bps:
            bps.add(dict(qualifier='nsp1',description='Number of spots on primary',repr='%d',cast_type=int,value=len(bps['xlat1'])))
        if not 'nsp2' in bps:
            bps.add(dict(qualifier='nsp2',description='Number of spots on secondary',repr='%d',cast_type=int,value=len(bps['xlat2'])))
        
        args = [bps['mpage'],bps['nref'],bps['mref'],bps['ifsmv1'],bps['ifsmv2'],bps['icor1'],
                bps['icor2'],bps['ld_model'],light_curve['indep_type'],bps['hjd0'],bps['period'],bps['dpdt'],
                bps['pshift'],bps['stdev'],bps['noise'],bps['seed'],light_curve['jdstrt'],light_curve['jdend'],
                light_curve['jdinc'],light_curve['phstrt'],light_curve['phend'],light_curve['phinc'],light_curve['phnorm'],bps['model'],
                bps['ipb'],bps['ifat1'],bps['ifat2'],bps['n1'],bps['n2'],bps['perr0'],
                bps['dperdt'],bps['the'],rv_curve['vunit'],bps['ecc'],bps['sma'],bps['f1'],
                bps['f2'],bps['vga'],bps['incl'],bps['grb1'],bps['grb2'],bps['met1'],
                bps['teff1'],bps['teff2'],bps['alb1'],bps['alb2'],bps['pot1'],bps['pot2'],
                bps['rm'],bps['ld_xbol1'],bps['ld_xbol2'],
                bps['ld_ybol1'],bps['ld_ybol2'],light_curve['filter'],
                light_curve['hla'],light_curve['cla'],light_curve['x1a'],light_curve['x2a'],light_curve['y1a'],light_curve['y2a'],
                light_curve['el3'],light_curve['opsf'],bps['mzero'],bps['factor'],bps['wla'],xlat,xlong,radsp,tempsp,
                bps['atmtab'],bps['plttab'],request_wd,vertno,light_curve['l3perc'],
                indeps,lc,rv1,rv2,skycoy,skycoz,params]
        #-- generate the light curve
        indeps,flux,rv1,rv2,skycoy,skycoz,params = fwd.lcnotemp(*args)
    #-- cast the parameters to a dictionary
    params = dict(L1=params[0],L2=params[1],M1=params[2],M2=params[3],
                  R1=params[4],R2=params[5],Mbol1=params[6],Mbol2=params[7],
                  logg1=params[8],logg2=params[9],SBR1=params[10],SBR2=params[11],
                  pot1=params[12],pot2=params[13])
    
    #-- we need to return different output for the image and lc requests
    if request=='image':
        #-- clean up skycoordinate array:
        keep = skycoy!=0
        skycoy = skycoy[keep]
        skycoz = skycoz[keep]
        return np.rec.fromarrays([skycoy,skycoz],names=['y','z']),params
    else:
        return np.rec.fromarrays([indeps,flux,rv1,rv2],names=['indeps','lc','rv1','rv2']),params

        
def lcin_to_ps(lcin,version='wdphoebe'):
    """
    Convert a WD lcin file to a ParameterSet
    
    Probably spots are not treated well, so you'd better not include them.
    
    @param lcin: name of the lcin file.
    @type lcin: str
    @param version: version of the LC input file
    @type version: str, one of 'wdphoebe' or 'wd2003'
    @return: ParameterSets with the values from lcin (root, lc and rv)
    @rtype: 3xParameterSet
    """
    #-- lcin formats:
    _formats = {}
    _formats['wdphoebe'] = [8*[2],\
                            [1,15,15,13,10,10,2,11],\
                            [14,15,13,12,12,12,12],\
                            [2,2,2,2,4,4,13,12,7,8],\
                            [6,13,10,10,10,9,7,7,7],\
                            [8,8,7,7,13,13,13,7,7,7,7],\
                            [3,10,10,7,7,7,7,8,10,8,8,9],\
                            [9,9,9,9],\
                            [5],\
                            [4],\
                            [1]]
    _formats['wd2003'] =  [8*[2],\
                            [1,15,15,13,10,10,2,11],\
                            [14,15,13,12,12,12,12],\
                            [2,2,2,2,3,3,13,12,7,8],\
                            [6,13,10,10,10,9,7,7,7],\
                            [8,8,7,7,13,13,13,7,7,7,7],\
                            [3,10,10,7,7,7,7,8,10,8,8,9],\
                            [9,9,9,9],\
                            [9,9,9,7,11,9,11,9,7],\
                            [5],\
                            [4],\
                            [1]] 
    #-- we want a default wd ParameterSet, but we add a light and radial
    #   velocity curve to the root
    ps = pars.ParameterSet(frame='wd',context='root')
    lc = pars.ParameterSet(frame='wd',context='lc')
    rv = pars.ParameterSet(frame='wd',context='rv')
    #-- these parameters are not listed in the parameter_definitions
    #lc.add(dict(qualifier='jdstrt',   description='Start Julian date',    repr='%14.6f',cast_type=float,value=0,frame='wd',context='lc'))
    #lc.add(dict(qualifier='jdend',    description='End Julian date',      repr='%14.6f',cast_type=float,value=1.0,frame='wd',context='lc'))
    #lc.add(dict(qualifier='jdinc',    description='Increment Julian date',repr='%14.6f',cast_type=float,value=0.1,frame='wd',context='lc'))
    #lc.add(dict(qualifier='phstrt',   description='Start Phase'          ,repr='%8.6f',cast_type=float,value=0,frame='wd',context='lc'))
    #lc.add(dict(qualifier='phend',    description='End phase',            repr='%8.6f',cast_type=float,value=1,frame='wd',context='lc'))
    #lc.add(dict(qualifier='phinc',    description='Increment phase',      repr='%8.6f',cast_type=float,value=0.01,frame='wd',context='lc'))
    
    #-- read in all the values from the lcin file. The values in a lcin file
    #   are defined
    params = []
    formats = _formats[version]
    with open(lcin,'r') as ff:
        for line,fmt in zip(ff.readlines(),formats):
            #-- sometimes people use the fortran 'D' exponential style, but
            #   it seems numpy can't handle that
            line = line.replace('d','e').replace('D','e')
            for ifmt in fmt:
                params.append(line[:ifmt])
                line = line[ifmt:]
    #-- except for the spots, they are given in this order:
    args = ['mpage','nref','mref','ifsmv1','ifsmv2','icor1','icor2','ld_model','indep_type','hjd0','period','dpdt','pshift','stdev','noise',
            'seed','jdstrt','jdend','jdinc','phstrt','phend','phinc','phnorm','model','ipb','ifat1','ifat2','n1','n2','perr0','dperdt',
            'the','vunit','ecc','sma','f1','f2','vga','incl','grb1','grb2','met1','teff1','teff2','alb1','alb2','pot1','pot2','rm',
            'ld_xbol1','ld_xbol2','ld_ybol1','ld_ybol2','filter','hla','cla','ld_lcx1','ld_lcx2','ld_lcy1','ld_lcy2','el3','opsf','mzero',
            'factor','wla']
    #-- overwrite the default values with the ones given in the input file
    for arg,par in zip(args,params):
        if arg in ps:
            ps[arg] = par
        if arg in lc:
            lc[arg] = par
        if arg in rv:
            rv[arg] = par
    #-- fake the presence of spots
    #new_args = ['nsp1','xlat1','xlong1','radsp1','tempsp1','nsp2','xlat2','xlong2','radsp2','tempsp2']
    new_args = ['xlat1','xlong1','radsp1','tempsp1','xlat2','xlong2','radsp2','tempsp2']
    for par in new_args:
        if not par in ps: # assume there is nothing given about the spots:
            ps.add(Parameter(par))
    #-- maybe the jdstrt or phstrt were given instead of directly the independent
    #   variables as an array. If so, translate them to an array
    if lc['indep_type']==1: # phase
        prefix = 'jd'
    else: # time
        prefix = 'ph'
    lc['indep'] = np.arange(lc[prefix+'strt'],lc[prefix+'end']+lc[prefix+'inc'],lc[prefix+'inc'])
    #-- a *very* complicated constraint to assure that C{indep} is synchronized with
    #   the 'strt' and 'end' and 'inc' of phases or julian days.
    lc.add_constraint('{indep} = '+'eval("np.arange({{{{{{prefix}}strt}}}},{{{{{{prefix}}end}}}},{{{{{{prefix}}inc}}}})".format(prefix=[None,"jd","ph"][{indep_type}]).format(**self))')
    #lc.add_constraint("{indep} = np.arange({%sstrt},{%send},{%sinc})"%(prefix,prefix,prefix))
    #-- check the atm and plt tables
    if not os.path.isfile(ps['atmtab']):
        ps['atmtab'] = os.path.join(os.path.dirname(os.path.abspath(__file__)),ps['atmtab'])
    if not os.path.isfile(ps['plttab']):
        ps['plttab'] = os.path.join(os.path.dirname(os.path.abspath(__file__)),ps['plttab'])
        
    
    #-- that's it!
    return ps,lc,rv

def ps_to_lcin(ps,lc,rv,lcin='lcin.active'):
    """
    Convert a ParameterSet to an lcin file.
    
    @param ps: ParameterSet to convert
    @type ps: ParameterSet
    @param lc: ParameterSet to convert
    @type lc: ParameterSet
    @param rv: ParameterSet to convert
    @type rv: ParameterSet
    @param lcin: name of the lcin file to write
    @type lcin: str
    """
    ps = ps+lc+rv
    args = ['mpage','nref','mref','ifsmv1','ifsmv2','icor1','icor2','ld_model','indep_type','hjd0','period','dpdt','pshift','stdev','noise',
            'seed','jdstrt','jdend','jdinc','phstrt','phend','phinc','phnorm','model','ipb','ifat1','ifat2','n1','n2','perr0','dperdt',
            'the','vunit','ecc','sma','f1','f2','vga','incl','grb1','grb2','met1','teff1','teff2','alb1','alb2','pot1','pot2','rm',
            'ld_xbol1','ld_xbol2','ld_ybol1','ld_ybol2','filter','hla','cla','ld_lcx1','ld_lcx2','ld_lcy1','ld_lcy2','el3','opsf','mzero',
            'factor','wla','nsp1','xlat1','xlong1','radsp1','tempsp1','nsp2','xlat2','xlong2','radsp2','tempsp2']
    #-- we can derive the number of spots from the given input
    if not 'nsp1' in ps:
        ps.add(dict(qualifier='nsp1',description='Number of spots on primary',repr='%d',cast_type=int,value=len(ps['xlat1'])))
    if not 'nsp2' in ps:
        ps.add(dict(qualifier='nsp2',description='Number of spots on secondary',repr='%d',cast_type=int,value=len(ps['xlat2'])))
    #-- prepare the arguments to write to an lcin file. These are all the
    #   previously defined arguments, plus the name of the lcin file.
    args_to_wrlci = tuple([lcin]+[ps[arg] for arg in args])
    fwd.wrlci(*args_to_wrlci)            


def wd_to_phoebe(ps_wd,lc,rv,ignore_errors=True):
    """
    Convert parameters from pyWD to phoebe.
    
    @param ps_wd: parameterSet with frame C{pywd} context C{root}
    @type ps_wd: parameterSet
    @param lc: parameterSet with frame C{pywd} context C{lc}
    @type lc: parameterSet
    @param rv: parameterSet with frame C{pywd} context {rv}
    @type rv: parameterSet
    @return: three parameterSets: (phoebe/component, phoebe/component, phoebe/orbit)
    @rtype: 3 x parameterSet
    """
    #-- initialise some parameterset. Copy the observables, to make sure they
    #   carry the same label
    orbit = pars.ParameterSet(frame='phoebe',context='orbit',add_constraints=True)
    body1 = pars.ParameterSet(frame='phoebe',context='component',add_constraints=True)
    body2 = pars.ParameterSet(frame='phoebe',context='component',add_constraints=True)
    lcdep1 = pars.ParameterSet(frame='phoebe',context='lcdep',add_constraints=True)
    lcdep2 = lcdep1.copy()
    rvdep1 = pars.ParameterSet(frame='phoebe',context='rvdep',add_constraints=True)
    rvdep2 = pars.ParameterSet(frame='phoebe',context='rvdep',add_constraints=True)
    
    #-- not implemented: met, mpage, mref, nref, icor, stdev,
    #   noise, seed, ipb, ifat, n1, n2, the, mzero, factor, wla, atmtab, model
    #   plttab, ifsmv, xlat, xlong, radsp, tempsp
    translate2binary = dict(name='label',hjd0='t0',period='period',dpdt='dpdt',
        pshift='phshift',sma='sma',rm='q',incl='incl',ecc='ecc',
        omega='per0',domegadt='dperdt')
    translate2body = dict(f='syncpar',teff='teff',pot='pot',alb='alb',grb='gravb')
    translate2lc = dict(ld_model='ld_func',filter='passband')
    translate2rv = dict()
    
    mypsets = [orbit,
               body1,body2,
               lcdep1,lcdep2,
               rvdep1,rvdep2]
    mytrnsls = [translate2binary,
                translate2body,translate2body,
                translate2lc,translate2lc,
                translate2rv,translate2rv]
    
    for pset in [ps_wd,lc,rv]:
        for par in pset:
            try:
                myval = pset.get_value_with_unit(par)
            except ValueError:
                myval = pset.get_value(par)
            
            param = pset.get_parameter(par)
            
            if hasattr(param,'choices') and hasattr(param,'cast_type'):
                if param.cast_type=='indexf':
                    myval = param.choices[myval-1]
                if param.cast_type=='index':
                    myval = param.choices[myval]
                
            for i,(mypset,trnsl) in enumerate(zip(mypsets,mytrnsls)):
                if par in trnsl:
                    new_par = trnsl[par]
                    mypset[new_par] = myval
                elif (i%2)==1 and par[-1]=='1' and par[:-1] in trnsl:
                    new_par = trnsl[par[:-1]]
                    mypset[new_par] = myval
                elif (i%2)==0 and par[-1]=='2' and par[:-1] in trnsl:
                    new_par = trnsl[par[:-1]]
                    mypset[new_par] = myval
                

    #-- limb darkening laws need manual tweaking
    lcdep1['ld_coeffs'] = [lc['ld_lcx1'],lc['ld_lcy1']]
    lcdep2['ld_coeffs'] = [lc['ld_lcx2'],lc['ld_lcy2']]
    lcdep1['boosting'] = False
    lcdep2['boosting'] = False
    rvdep1['ld_coeffs'] = [rv['ld_rvx1'],rv['ld_rvy1']]
    rvdep2['ld_coeffs'] = [rv['ld_rvx2'],rv['ld_rvy2']]
    
    body1['atm'] = (ps_wd['ifat1']==0) and 'blackbody' or 'kurucz'
    body2['atm'] = (ps_wd['ifat2']==0) and 'blackbody' or 'kurucz'
    body1['ld_coeffs'] = [ps_wd['ld_xbol1'],ps_wd['ld_ybol1']]
    body2['ld_coeffs'] = [ps_wd['ld_xbol2'],ps_wd['ld_ybol2']]
    lcdep1['atm'] = (ps_wd['ifat1']==0) and 'blackbody' or 'kurucz'
    lcdep2['atm'] = (ps_wd['ifat2']==0) and 'blackbody' or 'kurucz'
    rvdep1['atm'] = (ps_wd['ifat1']==0) and 'blackbody' or 'kurucz'
    rvdep2['atm'] = (ps_wd['ifat2']==0) and 'blackbody' or 'kurucz'
    
    #-- reflection?
    body1['alb'] = 1-body1['alb']
    body2['alb'] = 1-body2['alb']
    lcdep1['alb'] = 1 - lcdep1['alb']
    lcdep2['alb'] = 1 - lcdep2['alb']
    rvdep1['alb'] = 1 - rvdep1['alb']
    rvdep2['alb'] = 1 - rvdep2['alb']
    body1['irradiator'] = True
    body2['irradiator'] = True
    
    body1['ld_func'] = 'logarithmic' if ps_wd['ld_model']==2 else 'linear'
    body2['ld_func'] = 'logarithmic' if ps_wd['ld_model']==2 else 'linear'
    lcdep1['ld_func'] = 'logarithmic' if ps_wd['ld_model']==2 else 'linear'
    lcdep2['ld_func'] = 'logarithmic' if ps_wd['ld_model']==2 else 'linear'
    rvdep1['ld_func'] = 'logarithmic' if ps_wd['ld_model']==2 else 'linear'
    rvdep2['ld_func'] = 'logarithmic' if ps_wd['ld_model']==2 else 'linear'
    
    #-- passband luminosities and third light
    if ps_wd['ipb'] == 0:
        lcdep1['pblum'] = -1.0
        lcdep2['pblum'] = -1.0
    else:
        lcdep1['pblum'] = lc['hla']
        lcdep2['pblum'] = lc['cla']
    
    lcdep1['l3'] = lc['el3']
    lcdep2['l3'] = 0.
    
    #-- set explicit labels; this guarantees the same label for consecutive
    #   calls, otherwise the labels are random strings
    body1['label'] = 'star1'
    body2['label'] = 'star2'
    
    orbit['c1label'] = body1['label']
    orbit['c2label'] = body2['label']
    
    comp1 = body1,lcdep1,rvdep1
    comp2 = body2,lcdep2,rvdep2
    
    #-- convert from t0 (superior conjunction) to t0 (periastron passage)
    #t_supconj = orbit['t0']
    #phshift = orbit['phshift']
    #P = orbit['period']
    #per0 = orbit.get_value('per0','rad')
    #t0 = t_supconj + (phshift - 0.25 + per0/(2*np.pi))*P
    #orbit['t0'] = t0
    #orbit['t0type'] = 'periastron passage'
    
    
    orbit['t0type'] = 'superior conjunction'
    #-- gamma velocity needs to be corrected
    globals = pars.ParameterSet('position', vgamma=ps_wd.get_value('vga','km/s'))
    #orbit['vgamma'] = orbit['vgamma']*100.
    
    return comp1,comp2,orbit, globals


class BodyEmulator(object):
    """
    Wrap parameterSets for WD in a Body type class.
    
    This enables the fitters to use the WD code to do the actual fitting.
    """
    def __init__(self,pset,lcset=None,rvset=None,obs=None):
        """
        Initialize a Body.
        
        Only thing you need to do is pass the root parameterset, those
        for a light curve and/or radial velocity set, and the observations.
        """
        self.params = OrderedDict()
        self.params['pbdep'] = OrderedDict()
        self.params['obs'] = OrderedDict()
        self.params['syn'] = OrderedDict()
        
        self.params['root'] = pset
        if lcset is not None:
            ref = lcset['ref']
            self.params['pbdep']['lcdep'] = OrderedDict()
            self.params['pbdep']['lcdep'][ref] = lcset
            self.params['syn']['lcsyn'] = OrderedDict()
            self.params['syn']['lcsyn'][ref] = datasets.DataSet(context='lcsyn',ref=ref)
        if rvset is not None:
            ref = rvset['ref']
            self.params['pbdep']['rvdep'] = OrderedDict()
            self.params['pbdep']['rvdep'][ref+'1'] = rvset
            self.params['pbdep']['rvdep'][ref+'2'] = rvset.copy()
            self.params['syn']['rvsyn'] = OrderedDict()
            self.params['syn']['rvsyn'][ref+'1'] = datasets.DataSet(context='rvsyn',ref=ref)
            self.params['syn']['rvsyn'][ref+'2'] = datasets.DataSet(context='rvsyn',ref=ref)
            
        if obs is not None:
            for iobs in obs:
                if iobs.context[:2]=='lc':
                    if not 'lcobs' in self.params['obs']:
                        self.params['obs']['lcobs'] = OrderedDict()
                    self.params['obs']['lcobs'][iobs['ref']] = iobs
                    self.params['pbdep']['lcdep'][iobs['ref']]['indep'] = iobs['time']
                elif iobs.context[:2]=='rv':
                    if not 'rvobs' in self.params['obs']:
                        self.params['obs']['rvobs'] = OrderedDict()
                    self.params['obs']['rvobs'][iobs['ref']] = iobs
                    self.params['pbdep']['rvdep'][iobs['ref']]['indep'] = iobs['time']
                    
        self._preprocessing = []
        self._postprocessing = []
        
            
    
    def reset(self):
        """
        Resetting doesn't do anything.
        """
        pass
    
    def clear_synthetic(self):
        """
        Clear the body from all calculated results.
        """
        result_sets = dict(lcsyn=datasets.LCDataSet,
                       rvsyn=datasets.RVDataSet)
        if hasattr(self,'params') and 'syn' in self.params:
            for pbdeptype in self.params['syn']:
                for ref in self.params['syn'][pbdeptype]:
                    old = self.params['syn'][pbdeptype][ref]
                    new = result_sets[pbdeptype](context=old.context,ref=old['ref'])
                    self.params['syn'][pbdeptype][ref] = new
    
    def walk_all(self):
        for level1 in self.params.values():
            for level2 in level1.values():
                yield level2
    
    def walk(self):
        walk = utils.traverse(self.params,list_types=(list,tuple),dict_types=(dict,))
        for parset in walk:
            yield parset
    
    def add_preprocess(self,func,*args,**kwargs):
        """
        Add a preprocess to the Body.
        
        The list of preprocessing functions are executed before set_time is
        called.
        
        @param func: name of a processing function in backend.processes
        @type func: str
        """
        self._preprocessing.append((func,args,kwargs))

    def add_postprocess(self,func,*args,**kwargs):
        """
        Add a postprocess to the Body.
        
        @param func: name of a processing function in backend.processes
        @type func: str
        """
        self._postprocessing.append((func,args,kwargs))
    
    def preprocess(self,time):
        """
        Run the preprocessors.
        
        @param time: time to which the Body will be set
        @type time: float or None
        """
        for func,arg,kwargs in self._preprocessing:
            getattr(processing,func)(self,time,*arg,**kwargs)
    
    def postprocess(self,time):
        """
        Run the postprocessors.
        """
        for func,arg,kwargs in self._postprocessing:
            getattr(processing,func)(self,time,*args,**kwargs)
            
    def compute(self,*args,**kwargs):
        #self.preprocess(0.)
        
        
        if 'lcdep' in self.params['pbdep']:
            for ref in self.params['pbdep']['lcdep'].keys():
                lcset = self.params['pbdep']['lcdep'][ref]
                root = self.params['root']
                try:
                    ld_model = root.get_parameter('ld_model').get_choices()[root['ld_model']-1]
                    #-- fix limbdarkening:
                    atm_kwargs1 = dict(teff=self.params['root'].request_value('teff1','K'),logg=4.0)
                    atm_kwargs2 = dict(teff=self.params['root'].request_value('teff2','K'),logg=4.0)
                    basename = '{}_{}{:02.0f}_{}_equidist_r_leastsq_teff_logg.fits'.format('kurucz','p',0,ld_model)        
                    atm = '/home/pieterd/software/phoebe-code/devel/phoebe/atmospheres/tables/ld_coeffs/'+basename
                    passband = lcset.get_parameter('filter').get_choices()[lcset['filter']-1].upper()
                    coeffs1 = limbdark.interp_ld_coeffs(atm, passband, atm_kwargs=atm_kwargs1)
                    coeffs2 = limbdark.interp_ld_coeffs(atm, passband, atm_kwargs=atm_kwargs2)
                    lcset['ld_lcx1'] = coeffs1[0]
                    lcset['ld_lcy1'] = coeffs1[1] if len(coeffs1.ravel())==3 else 0.0
                    lcset['ld_lcx2'] = coeffs2[0]
                    lcset['ld_lcy2'] = coeffs2[1] if len(coeffs2.ravel())==3 else 0.0
                except:
                    print("Failed deriving LD for some reason --> debug!")
                    pass
                
                curve,params = lc(self.params['root'],request='lc',light_curve=lcset)
                self.params['syn']['lcsyn'][ref]['time'] = curve['indeps']
                self.params['syn']['lcsyn'][ref]['flux'] = curve['lc']
                self.out = params
                
        lcset_dummy = lcset.copy()
        if 'rvdep' in self.params['pbdep']:
            refs = self.params['pbdep']['rvdep'].keys()
            for ref in refs:
                rvset = self.params['pbdep']['rvdep'][ref]
                lcset_dummy['indep'] = rvset['indep']
                curve1,params1 = lc(self.params['root'],request='lc',
                                    light_curve=lcset_dummy,rv_curve=rvset)
                self.params['syn']['rvsyn'][ref]['time'] = curve1['indeps']
                #self.params['syn']['rvsyn'][ref]['rv'] = phoebe.convert('km/s','Rsol/d',curve1['rv'+ref[-1]])
                self.params['syn']['rvsyn'][ref]['rv'] = curve1['rv'+ref[-1]]
        
        
        
        #-- passband luminosity and third light:
        pblum_par = self.params['obs']['lcobs'].values()[0].get_parameter('pblum')
        l3_par = self.params['obs']['lcobs'].values()[0].get_parameter('l3')
        pblum = pblum_par.get_adjust() and not pblum_par.has_prior()
        l3 = l3_par.get_adjust() and not l3_par.has_prior()
        
        model = np.array(self.params['syn']['lcsyn'].values()[0]['flux'])
        obs = self.params['obs']['lcobs'].values()[0]['flux']
        sigma = self.params['obs']['lcobs'].values()[0]['sigma']
        
        if pblum and not l3:
            pblum = np.average(obs/model,weights=1./sigma**2)
            l3 = 0.
        #   only offset
        elif not pblum and l3:
            pblum = 1.
            l3 = np.average(obs-model,weights=1./sigma**2)
        #   scaling factor and offset
        elif pblum and l3:
            A = np.column_stack([model.ravel(),np.ones(len(model.ravel()))])
            pblum,l3 = nnls(A,obs.ravel())[0]
        
        if pblum is False:
            pblum = 1.0
        if l3 is False:
            l3 = 0.0
            
        logger.info("pblum = {}, l3 = {}".format(pblum,l3))
        pblum_par.set_value(pblum)
        l3_par.set_value(l3)
        
        #self.postprocess(0.)
        
    def get_model(self):
        
        model = np.array(self.params['syn']['lcsyn'].values()[0]['flux'])
        
        mu = self.params['obs']['lcobs'].values()[0]['flux']
        sigma = self.params['obs']['lcobs'].values()[0]['sigma']
        observations = self.params['obs']['lcobs'].values()[0]
        
        pblum = observations['pblum'] if ('pblum' in observations) else 1.0
        l3 = observations['l3'] if ('l3' in observations) else 0.0
        
        logger.info("pblum = {}, l3 = {}".format(pblum,l3))
        
        model = model*pblum+l3
        
        if 'rvobs' in self.params['obs']:
            
            rv_model = np.hstack([val['rv'] for val in self.params['syn']['rvsyn'].values()])
            rv_mu = np.hstack([val['rv'] for val in self.params['obs']['rvobs'].values()])
            rv_sigma = np.hstack([val['sigma'] for val in self.params['obs']['rvobs'].values()])
            
            model = np.hstack([model, rv_model])
            sigma = np.hstack([sigma, rv_sigma])
            mu = np.hstack([mu, rv_mu])
            
        return mu,sigma,model
    
    def get_data(self):
        mu = self.params['obs']['lcobs'].values()[0]['flux']
        sigma = self.params['obs']['lcobs'].values()[0]['sigma']
        return mu,sigma
    
    def get_logp(self):
        mu,sigma,model = self.get_model()
        term1 = - 0.5*np.log(2*np.pi*sigma**2)
        term2 = - (mu-model)**2/(2.*sigma**2)
        logp = (term1 + term2).sum()
        chi2 = -2*term2.sum()
        N = len(mu)
        return logp,chi2,N
    
    def get_synthetic(self,category='lc',ref=0):
        if category=='lc':
            if isinstance(ref,str):
                return self.params['syn']['lcsyn'][ref]
            else:
                return self.params['syn']['lcsyn'].values()[ref]
        elif category=='rv':
            if isinstance(ref,str):
                return self.params['syn']['rvsyn'][ref]
            else:
                return self.params['syn']['rvsyn'].values()[ref]
    
    def get_obs(self,category='lc',ref=0):
        if category=='lc':
            if isinstance(ref,str):
                return self.params['obs']['lcobs'][ref]
            else:
                return self.params['obs']['lcobs'].values()[ref]
        elif category=='rv':
            if isinstance(ref,str):
                return self.params['obs']['rvobs'][ref]
            else:
                return self.params['obs']['rvobs'].values()[ref]
    
    def set_values_from_priors(self):
        raise NotImplementedError
    
    def remove_mesh(self):
        """
        Removing the mesh also doesn't do anything.
        """
        pass
    
    def save(self,filename):
        """
        Save this Body to a pickle.
        """
        ff = open(filename,'w')
        pickle.dump(self,ff)
        ff.close()  
        logger.info('Saved model to file {} (pickle)'.format(filename))
    
    def copy(self):
        return copy.deepcopy(self)
    
if __name__=="__main__":
    import doctest
    doctest.testmod()
    pl.show()
