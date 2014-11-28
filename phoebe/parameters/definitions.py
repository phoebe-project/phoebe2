"""
Standard definitions of parameters and their default values.
"""
import glob 
import numpy as np
import os

#  /* **********************   Model parameters   ************************** */


#-- WD
defs  = [dict(qualifier="name" ,description="Common name of the binary",    repr="%s",cast_type=str,value="mybinary",frame=["wd"],alias=['phoebe_name'],context='root'),
         dict(qualifier="model",description="Morphological constraints",    repr="%s",choices=["X-ray binary",
                                                                                                 "Unconstrained binary system",
                                                                                                 "Overcontact binary of the W UMa type",
                                                                                                 "Detached binary",
                                                                                                 "Overcontact binary not in thermal contact",
                                                                                                 "Semi-detached binary, primary star fills Roche lobe",
                                                                                                 "Semi-detached binary, secondary star fills Roche lobe",
                                                                                                 "Double contact binary"],
                                                                            long_description=("X-ray binary: potential value is computed from q, F2, e, omega, i and the (eclipse duration). "
                                                                                               "Unconstrained: Component luminosity is not required to be consistent with temperatures. "
                                                                                               "Overcontact W Uma: Seven constraints. "
                                                                                               "Detached binary: Consistent luminosity-temperatures (can be severed if ipb=1). "
                                                                                               ""),
                                                                                                 cast_type='indexm',value="Unconstrained binary system",
                                                                                                 frame=["wd"],alias=['mode'],context='root')]                                                                                                 

#  /* **********************   System parameters   ************************* */

#-- WD
defs += [dict(qualifier="hjd0",  description="Origin of time",                            repr= "%f", llim=-1E10, ulim=  1E10, step= 0.0001, adjust=False, cast_type=float, unit='HJD',  value= 55124.89703,frame=["wd"],alias=['T0','phoebe_hjd0'],context='root'),
         dict(qualifier="period",description="Orbital period",                            repr= "%f", llim=  0.0, ulim=  1E10, step= 0.0001, adjust=False, cast_type=float, unit='d',    value= 22.1891087 ,frame=["wd"],alias=['p','phoebe_period'],context='root'),
         dict(qualifier="dpdt",  description="First time derivative of period",           repr= "%f", llim= -1.0, ulim=   1.0, step=   1E-6, adjust=False, cast_type=float, unit='d/d',  value= 0.0        ,frame=["wd"],alias=['phoebe_dpdt'],context='root'),
         dict(qualifier="pshift",description="Phase shift",                               repr= "%f",     llim= -0.5, ulim=   0.5, step=   0.01, adjust=False, cast_type=float,              value= 0.0776     ,frame=["wd"],alias=['phoebe_pshift'],context='root'),
         dict(qualifier="sma",   description="Semi-major axis",                           repr= "%f",     llim=  0.0, ulim=  1E10, step=   0.01, adjust=False, cast_type=float, unit='Rsol', value=11.0104     ,frame=["wd"],alias=['a','phoebe_sma'],context=['root'],prior=dict(distribution='uniform',lower=0,upper=1e10)),
         dict(qualifier="rm",    description="Mass ratio (secondary over primary)",       repr= "%f",     llim=  0.0, ulim=  1E10, step=   0.01, adjust=False, cast_type=float,              value=0.89747     ,frame=["wd"],alias=['q','phoebe_rm'],context='root'),
         dict(qualifier="incl",  description="Inclination angle",                               repr= "%f",     llim=  0.0, ulim= 180.0, step=   0.01, adjust=False, cast_type=float, unit='deg',  value=87.866      ,frame=["wd"],alias=['i','phoebe_incl'],context=['root'],prior=dict(distribution='uniform',lower=-180,upper=180)),
         dict(qualifier="vga",   description="Center-of-mass velocity",                   repr= "%f",     llim= -1E3, ulim=   1E3, step=    1.0, adjust=False, cast_type=float, unit='km/s', value= 0.0        ,frame=["wd"],alias=['gamma','phoebe_vga'],context='root'),
         dict(qualifier='ecc'  , description='Eccentricity'                             , repr= '%f',     llim=  0.0, ulim=   0.99,step=   0.01, adjust=False, cast_type=float,              value=0.28319     ,frame=["wd"],alias=['e','phoebe_ecc'],context='root'),
         dict(qualifier='omega', description='Initial argument of periastron for star 1', repr= '%f',     llim= -2*np.pi, ulim=   2*np.pi,step=   0.01, adjust=False, cast_type=float, unit='rad',  value=5.696919    ,frame=["wd"],alias=['perr0'],context='root'),
         dict(qualifier='domegadt',description='First time derivative of periastron'    , repr='%f',      llim=  0.0, ulim=    1.0,step=   0.01, adjust=False, cast_type=float, unit='rad/d',value=0,           frame=["wd"],alias=['dperdt'],context='root')]

#  /* ********************   Component parameters   ************************ */

defs += [dict(qualifier='f1',    description="Primary star synchronicity parameter"      ,repr='%f',llim=  0.0, ulim=   10.0,step=   0.01, adjust=False,cast_type=float,value=1.,frame=["wd"],alias=['phoebe_f1'],context='root'), 
         dict(qualifier='f2',    description="Secondary star synchronicity parameter"    ,repr='%f',llim=  0.0, ulim=   10.0,step=   0.01, adjust=False,cast_type=float,value=1.,frame=["wd"],alias=['phoebe_f2'],context='root'), 
         dict(qualifier='teff1', description="Primary star effective temperature"        ,repr='%f',llim=  0.2 ,ulim=      5,step=   0.01, adjust=False,cast_type=float,value=0.8105,unit='10000K',frame=["wd"],alias=[],context='root'),
         dict(qualifier='teff2', description="Secondary star effective temperature"      ,repr='%f',llim=  0.001 ,ulim=      5,step=   0.01, adjust=False,cast_type=float,value=0.7299,unit='10000K',frame=["wd"],alias=[],context='root'),
         dict(qualifier='pot1',  description="Primary star surface potential"            ,repr='%f',llim=  0.0, ulim=  1000.0,step=   0.01, adjust=False,cast_type=float,value=7.34050,frame=["wd"],alias=['phoebe_pot1'],context='root'),
         dict(qualifier='pot2',  description="Secondary star surface potential"          ,repr='%f',llim=  0.0, ulim=  1000.0,step=   0.01, adjust=False,cast_type=float,value=7.58697,frame=["wd"],alias=['phoebe_pot2'],context='root'),
         dict(qualifier='met1',  description="Primary star metallicity"                  ,repr='%f',llim=  0.0, ulim=    1.0,step=   0.01, adjust=False,cast_type=float,value=0.0,frame=["wd"],alias=['phoebe_met1'],context='root'),
         dict(qualifier='met2',  description="Secondary star metallicity"                ,repr='%f',llim=  0.0, ulim=    1.0,step=   0.01, adjust=False,cast_type=float,value=0.0,frame=["wd"],alias=['phoebe_met2'],context='root'),
         dict(qualifier='alb1',  description="Primary star surface albedo"               ,repr='%f',llim=  0.0, ulim=    1.5,step=   0.01, adjust=False,cast_type=float,value=1.0,frame=["wd"],alias=['phoebe_alb1'],context='root'),
         dict(qualifier='alb2',  description="Secondary star surface albedo"             ,repr='%f',llim=  0.0, ulim=    1.5,step=   0.01, adjust=False,cast_type=float,value=0.864,frame=["wd"],alias=['phoebe_alb2'],context='root'),
         dict(qualifier='grb1',  description="Primary star gravity brightening"          ,repr='%f',llim=  0.0, ulim=    1.5,step=   0.01, adjust=False,cast_type=float,value=0.964,frame=["wd"],alias=['phoebe_grb1','gr1'],context='root'),
         dict(qualifier='grb2',  description="Secondary star gravity brightening"        ,repr='%f',llim=  0.0, ulim=    1.5,step=   0.01, adjust=False,cast_type=float,value=0.809,frame=["wd"],alias=['phoebe_grb2','gr2'],context='root')]

#  /* ****************   Light/RV curve dependent parameters   ************ */

defs += [dict(qualifier='filter',description='Filter name',choices=['STROMGREN.U','stromgren.v','stromgren.b','stromgren.y',
                                                                    'johnson.U','johnson.B','JOHNSON.V','johnson.R','johnson.I','johnson.J','johnson.K','johnson.L','johnson.M','johnson.N',
                                                                    'bessell.RC','bessell.IC',
                                                                    'kallrath.230','kallrath.250','kallrath.270','kallrath.290','kallrath.310','kallrath.330',
                                                                    'tycho.B','tycho.V','hipparcos.hp','COROT.EXO','COROT.SIS','JOHNSON.H',
                                                                    'GENEVA.U','GENEVA.B','GENEVA.B1','GENEVA.B2','GENEVA.V','GENEVA.V1','GENEVA.G',
                                                                    'KEPLER.V','SDSS.U'],repr='%s',cast_type='indexf',value='johnson.V',frame=["wd"],context='lc',alias=['phoebe_lc_filter']),
         dict(qualifier="indep_type",    description="Independent modeling variable",repr="%s",choices=['time (hjd)','phase'],cast_type='indexf',value="phase",frame=["wd"],alias=['jdphs','phoebe_lc_indep'],context=['lc','rv']),
         dict(qualifier="indep",         description="Time/phase template or time/phase observations",repr="%s",cast_type=np.array,value=np.arange(0,1.005,0.01),frame=["wd"],context=['lc','rv']),
         dict(qualifier="ld_model", description="Limb darkening model",                      choices=['linear','logarithmic','square root law'],repr= "%s",cast_type='indexf',value="logarithmic",frame=["wd"],alias=['ld','phoebe_ld_model'],context='root'),
         dict(qualifier="ld_xbol1", description="Primary star bolometric LD coefficient x",  repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.512,frame=["wd"],alias=['phoebe_ld_xbol1'],context='root'),
         dict(qualifier="ld_ybol1", description="Primary star bolometric LD coefficient y",  repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.0,frame=["wd"],alias=['phoebe_ld_ybol1'],context='root'),
         dict(qualifier="ld_xbol2", description="Secondary star bolometric LD coefficient x",repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.549,frame=["wd"],alias=['phoebe_ld_xbol2'],context='root'),
         dict(qualifier="ld_ybol2", description="Secondary star bolometric LD coefficient y",repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.0,frame=["wd"],alias=['phoebe_ld_ybol2'],context='root'),
         dict(qualifier="ld_lcx1",  description="Primary star passband LD coefficient x",    repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.467,frame=["wd",'jktebop'],alias=['x1a','phoebe_ld_lcx1'],context='lc'),
         dict(qualifier="ld_lcx2",  description="Secondary star passband LD coefficient x",  repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.502,frame=["wd",'jktebop'],alias=['x2a','phoebe_ld_lcx2'],context='lc'),
         dict(qualifier="ld_lcy1",  description="Primary star passband LD coefficient y",    repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.0,frame=["wd",'jktebop'],alias=['y1a','phoebe_ld_lcy1'],context='lc'),
         dict(qualifier="ld_lcy2",  description="Secondary star passband LD coefficient y",  repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.0,frame=["wd",'jktebop'],alias=['y2a','phoebe_ld_lcy2'],context='lc'),
         dict(qualifier="ld_rvx1",  description="Primary RV passband LD coefficient x",      repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.5,frame=["wd"],context='rv'),
         dict(qualifier="ld_rvx2",  description="Secondary RV passband LD coefficient x",    repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.5,frame=["wd"],context='rv'),
         dict(qualifier="ld_rvy1",  description="Primary RV passband LD coefficient y",      repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.5,frame=["wd"],context='rv'),
         dict(qualifier="ld_rvy2",  description="Secondary RV passband LD coefficient y",    repr= "%f",llim=-10.0, ulim=10.0,step= 0.01,cast_type=float,value=0.5,frame=["wd"],context='rv'),
         dict(qualifier="hla" ,     description="LC primary passband luminosity"  ,repr='%f',cast_type=float,llim=0,ulim=1e10,step=0.01,value=8.11061,frame=["wd"],alias=['phoebe_hla'],context='lc'),
         dict(qualifier="cla" ,     description="LC secondary passband luminosity",repr='%f',cast_type=float,llim=0,ulim=1e10,step=0.01,value=4.43580,frame=["wd"],alias=['phoebe_cla'],context='lc'),
         dict(qualifier="opsf",     description="Opacity frequency function",      repr="%f",cast_type=float,llim=0.0,ulim=1E10,step=0.01,value=0.0,frame=["wd"],alias=['phoebe_opsf'],context='lc'),
         dict(qualifier="el3",      description="Third light contribution",repr="%f",cast_type=float,llim=0.0,ulim=1E10,step=0.01,value=0.0,frame=["wd"],alias=['phoebe_el3'],context='lc'),
         dict(qualifier='el3_units',description="Units of third light",choices=['total light','percentage'],repr='%s',cast_type='indexf',value='total light',frame=["wd"],alias=['l3perc','phoebe_el3_units'],context='lc'),
         dict(qualifier='phnorm',   description='Phase of normalisation',repr='%8.6f',cast_type=float,value=0.25,frame=["wd"],context='lc'),
         dict(qualifier='jdstrt',   description='Start Julian date',    repr='%14.6f',cast_type=float,value=0,frame=["wd"],context='lc'),
         dict(qualifier='jdend',    description='End Julian date',      repr='%14.6f',cast_type=float,value=1.0,frame=["wd"],context='lc'),
         dict(qualifier='jdinc',    description='Increment Julian date',repr='%14.6f',cast_type=float,value=0.1,frame=["wd"],context='lc'),
         dict(qualifier='phstrt',   description='Start Phase'          ,repr='%8.6f',cast_type=float,value=0,frame=["wd"],context='lc'),
         dict(qualifier='phend',    description='End phase',            repr='%8.6f',cast_type=float,value=1,frame=["wd"],context='lc'),
         dict(qualifier='phinc',    description='Increment phase',      repr='%8.6f',cast_type=float,value=0.01,frame="wd",context='lc'),
         #dict(qualifier='data',     description='Filename containing the data',cast_type='filename2data',value=None,frame=['main',"wd"],context='lc'),
         ]

#  /* *********   Values specifically for Wilson-Devinney   **************** */

defs += [dict(qualifier='mpage', description="Output type of the WD code", choices=['lightcurve','rvcurve','lineprofile','starradii','image'],repr="%s",cast_type='indexf',value='lightcurve',frame=["wd"],context='root'),
         dict(qualifier='mref',  description="Reflection treatment",       choices=['simple','detailed'],                                     repr="%s",cast_type='indexf',value='simple',frame=["wd"],context='root'),
         dict(qualifier='nref',  description='Number of reflections',      repr='%d',cast_type=int,value=2,frame=["wd"],alias=['phoebe_reffect_reflections'],context='root'),
         dict(qualifier='icor1', description='Turn on prox and ecl fx on prim RV',  repr='%d',cast_type=int,value=True,frame=["wd"],alias=['phoebe_proximity_rv1_switch'],context='root'),
         dict(qualifier='icor2', description='Turn on prox and ecl fx on secn RV',repr='%d',cast_type=int,value=True,frame=["wd"],alias=['phoebe_proximity_rv2_switch'],context='root'),
         dict(qualifier='stdev', description='Synthetic noise standard deviation',repr='%f',cast_type=float,value=0,frame=["wd"],context='root'),
         dict(qualifier='noise', description='Noise scaling',              choices=['proportional','square root','independent'],repr='%s',cast_type='indexf',value='independent',frame=["wd"],context='root'),
         dict(qualifier='seed',  description='Seed for random generator', repr='%f',cast_type=float,value=100000001.,frame=["wd"],alias=['phoebe_synscatter_seed'],context='root'),
         dict(qualifier='ipb',   description='Compute second stars Luminosity'                ,repr='%d',cast_type=int,value=False,frame=["wd"],context='root'),
         dict(qualifier='ifat1', description='Atmosphere approximation primary'               ,repr='%s',choices=['blackbody','kurucz'],cast_type='index',value='kurucz',frame=["wd"],alias=['phoebe_atm1_switch'],context='root'),
         dict(qualifier='ifat2', description='Atmosphere approximation secondary'             ,repr='%s',choices=['blackbody','kurucz'],cast_type='index',value='kurucz',frame=["wd"],alias=['phoebe_atm2_switch'],context='root'),
         dict(qualifier='n1',    description='Grid size primary'                              ,repr='%d',cast_type=int,value=70,frame=["wd"],alias=['phoebe_grid_finesize1'],context='root'),
         dict(qualifier='n2',    description='Grid size secondary'                            ,repr='%d',cast_type=int,value=70,frame=["wd"],alias=['phoebe_grid_finesize2'],context='root'),
         dict(qualifier='the',   description='Semi-duration of primary eclipse in phase units',repr='%f',cast_type=float,value=0,frame=["wd"],context='root'),
         dict(qualifier='vunit', description='Unit of radial velocity (km/s)',repr='%f',cast_type=float,value=1.,frame=["wd"],context='rv'),
         dict(qualifier='mzero',  description='Zeropoint mag (vert shift of lc)',repr='%f',cast_type=float,value=0,frame=["wd"],context='root'),
         dict(qualifier='factor', description='Zeropoint flux (vert shift of lc)',repr='%f',cast_type=float,value=1.,frame=["wd"],context='root'),
         dict(qualifier='wla',    description='Reference wavelength (in microns)',repr='%f',cast_type=float,value=0.59200,frame=["wd"],context='root'),
         dict(qualifier='atmtab', description='Atmosphere table',                        repr='%s', cast_type=str, value='phoebe_atmcof.dat',      frame=["wd"],context='root'),
         dict(qualifier='plttab', description='Planck table',                            repr='%s', cast_type=str, value='phoebe_atmcofplanck.dat',frame=["wd"],context='root')
         ]

#  /* *********************   SPOT PARAMETERS      ************************* */

defs += [dict(qualifier='ifsmv1', description='Spots on star 1 co-rotate with the star', repr='%d', cast_type=int,   value=0,     frame=["wd"],alias=['phoebe_spots_corotate1'],context='root'),
         dict(qualifier='ifsmv2', description='Spots on star 2 co-rotate with the star', repr='%d', cast_type=int,   value=0,     frame=["wd"],alias=['phoebe_spots_corotate2'],context='root'),
         dict(qualifier='xlat1',  description='Primary spots latitudes'                , repr='%s', cast_type='list',value=[300.],frame=["wd"],alias=['wd_spots_lat1'],context='root'),
         dict(qualifier='xlong1', description='Primary spots longitudes'               , repr='%s', cast_type='list',value=[0.],  frame=["wd"],alias=['wd_spots_long1'],context='root'),
         dict(qualifier='radsp1', description='Primary spots radii'                    , repr='%s', cast_type='list',value=[1.],  frame=["wd"],alias=['wd_spots_rad1'],context='root'),
         dict(qualifier='tempsp1',description='Primary spots temperatures'             , repr='%s', cast_type='list',value=[1.],  frame=["wd"],alias=['wd_spots_temp1'],context='root'),
         dict(qualifier='xlat2',  description='Secondary spots latitudes'              , repr='%s', cast_type='list',value=[300.],frame=["wd"],alias=['wd_spots_lat2'],context='root'),
         dict(qualifier='xlong2', description='Secondary spots longitudes'             , repr='%s', cast_type='list',value=[0.],  frame=["wd"],alias=['wd_spots_long2'],context='root'),
         dict(qualifier='radsp2', description='Secondary spots radii'                  , repr='%s', cast_type='list',value=[1.],  frame=["wd"],alias=['wd_spots_rad2'],context='root'),
         dict(qualifier='tempsp2',description='Secondary spots temperatures'           , repr='%s', cast_type='list',value=[1.],  frame=["wd"],alias=['wd_spots_temp2'],context='root'),
         ]
                  

# /* **********************   Values specifically for spheres *************** */

defs +=[dict(qualifier='teff', description="Effective temperature"        ,repr='%.0f',llim=  0,ulim=  1e20,step=   100., adjust=False,cast_type=float,value=5777.,unit='K',frame=["phoebe"],alias=[],context='star'),
        dict(qualifier='radius', description='Radius',repr='%f', cast_type=float,   value=1., unit='Rsol', adjust=False,frame=["phoebe"],context='star'),
        dict(qualifier='mass', description='Stellar mass',repr='%g',cast_type=float,value=1., unit='Msol', adjust=False,frame=["phoebe"],context='star'),
        dict(qualifier='atm',    description='Atmosphere model',long_description=("Atmosphere models can be given in three ways: (1) using a short alias (e.g. 'kurucz') in which "
                                                                                  "case the FITS-table's filename will be derived from the alias, the limb darkening function and "
                                                                                  "other information like reddening, boosting etc... (2) using a relative filename, in which case "
                                                                                  "the file will be looked up in the ld_coeffs directory in src/phoebe/atmospheres/tables/ld_coeffs/"
                                                                                  " or (3) via an absolute filename"), repr='%s',cast_type=str,value='blackbody',frame=["phoebe"],context=['lcdep','rvdep','ifdep','spdep','pldep','amdep']),
        dict(qualifier='atm',    description='Bolometric Atmosphere model',long_description=("The bolometric atmosphere table is used to look up the bolometric intensities, which are "
                                                                                             "typically used for heating processes. Atmosphere models can be given in three ways: (1) using a short alias (e.g. 'kurucz') in which "
                                                                                             "case the FITS-table's filename will be derived from the alias, the limb darkening function and "
                                                                                             "other information like reddening, boosting etc... (2) using a relative filename, in which case "
                                                                                             "the file will be looked up in the ld_coeffs directory in src/phoebe/atmospheres/tables/ld_coeffs/"   
                                                                                             " or (3) via an absolute filename"),repr='%s',cast_type=str,value='blackbody',frame=["phoebe"],context=['star','component','accretion_disk']),
        dict(qualifier='rotperiod', description='Polar rotation period',repr='%f',cast_type=float,value=22.,adjust=False,frame=["phoebe"],unit='d',context='star'),
        dict(qualifier='diffrot', description='(Eq - Polar) rotation period (<0 is solar-like)',repr='%f',cast_type=float,value=0.,adjust=False,frame=["phoebe"],unit='d',context='star'),
        dict(qualifier='gravb',  description='Bolometric gravity brightening',repr='%f',cast_type=float,value=1.,llim=0,ulim=1.5,step=0.05,adjust=False,alias=['grb'],frame=["phoebe"],context='star'),
        dict(qualifier='gravblaw',description='Gravity brightening law',repr='%s',cast_type='choose',choices=['zeipel','espinosa','claret'],value='zeipel',frame=['phoebe'],context=['star','component']),
        dict(qualifier='incl',   description='Inclination angle', unit='deg',repr='%f',llim=-180,ulim=180,step=0.01,adjust=False,cast_type=float,value=90.,frame=["phoebe"],context=['star','accretion_disk']),
        dict(qualifier='long',   description='Orientation on the sky (East of North)', repr='%f',llim=-360., ulim=   360.,step=   0.01, adjust=False, cast_type=float, unit='deg',  value=0.,frame=["phoebe"],context=['star','accretion_disk']),
        dict(qualifier='t0',   description='Origin of time', repr='%f',adjust=False, cast_type=float, unit='JD',  value=0.,frame=["phoebe"],context=['star','accretion_disk']),
        #dict(qualifier='distance',description='Distance to the star',repr='%f',cast_type=float,value=10.,adjust=False,unit='pc',frame=['phoebe'],context='star'),
        dict(qualifier='shape', description='Shape of surface',repr='%s',cast_type='choose',choices=['equipot','sphere'],value='equipot',frame=["phoebe"],context='star'),
        #dict(qualifier='vgamma', description='Systemic velocity',repr='%f',llim=-1e6,ulim=1e6,step=0.1,adjust=False,cast_type=float,value=0.,unit='km/s',alias=['vga'],frame=["phoebe"],context='star'),
        ]

#  /* ********************* PHOEBE ********************************* */
#   BINARY CONTEXT
defs += [dict(qualifier='dpdt',   description='Period change',unit='s/yr',repr='%f',llim=-1e8,ulim=1e8,step=1e-5,adjust=False,cast_type=float,value=0,frame=["phoebe"],context='orbit'),
         dict(qualifier='dperdt', description='Periastron change',unit='deg/yr',repr='%f',llim=-1000,ulim=1000,step=1e-5,adjust=False,cast_type=float,value=0,frame=["phoebe"],context='orbit'),
         dict(qualifier='ecc',    description='Eccentricity',repr='%f',llim=0,ulim=1.,step=0.01,adjust=False,cast_type=float,value=0.,frame=["phoebe"],context='orbit'),
         dict(qualifier='t0',     description='Zeropoint date',unit='JD',repr='%f',llim=-2e10,ulim=2e10,step=0.001,adjust=False,cast_type=float,value=0.,alias=['hjd0'],frame=["phoebe"],context='orbit'),
         dict(qualifier='t0type', description='Interpretation of zeropoint date', repr='%s', cast_type='choose', choices=['periastron passage','superior conjunction'], value='periastron passage',frame=["phoebe"],context='orbit'),
         dict(qualifier='incl',   description='Orbital inclination angle',\
                                  long_description=("The inclination angle of the orbit is defined such that an angle of 90 degrees"
                                                   " means an edge-on system, i.e. the components eclipse each other. An angle of "
                                                   "0 degrees means face-on, the components orbit in the plane of the sky. An angle "
                                                   "of 45 degrees means that when t0type is superior conjunction, the time equals "
                                                   "the time zeropoint (t0), and the longitude of the ascending node (long_an) is zero, "
                                                   " then the secondary will be in front and below the primary"),\
                                                       unit='deg',repr='%f',llim=-180,ulim=180,step=0.01,adjust=False,cast_type=float,value=90.,frame=["phoebe"],context='orbit'),
         dict(qualifier='label',  description='Name of the system',repr='%s',cast_type='make_label',value='',frame=["phoebe","wd"],context=['orbit','root']),
         dict(qualifier='period', description='Period of the system',repr='%f',unit='d',llim=0,ulim=1e10,step=0.01,adjust=False,cast_type=float,value=3.,frame=["phoebe"],context='orbit'),
         dict(qualifier='per0',   description='Periastron',repr='%f',unit='deg',llim=-360,ulim=360,step=0.01,adjust=False,cast_type=float,value=90.,frame=["phoebe"],context='orbit'),
         dict(qualifier='phshift',description='Phase shift',repr='%f',llim=-1,ulim=+1,step=0.001,adjust=False,cast_type=float,value=0,frame=["phoebe"],context='orbit'),
         dict(qualifier='q',      description='Mass ratio',repr='%f',llim=0,ulim=1e6,step=0.0001,adjust=False,cast_type=float,value=1.,alias=['rm'],frame=["phoebe"],context='orbit'),
         dict(qualifier='sma',    description='Semi major axis',unit='Rsol',repr='%f',llim=0,ulim=1e10,step=0.0001,adjust=False,value=8.,cast_type=float,frame=["phoebe"],context='orbit'),
         dict(qualifier='long_an',   description='Longitude of ascending node', repr='%f',llim=  0.0, ulim=   360,step=   0.01, adjust=False, cast_type=float, unit='deg',  value=0.,frame=["phoebe"],context='orbit'),
         dict(qualifier='c1label', description='ParameterSet connected to the primary component',repr='%s',value=None,cast_type=str, frame=["phoebe"],context='orbit'),
         dict(qualifier='c2label', description='ParameterSet connected to the secondary component',repr='%s',value=None,cast_type=str, frame=["phoebe"],context='orbit'),
         ]

#    BODY CONTEXT
defs += [dict(qualifier='alb',    description='Bolometric albedo (1-alb heating, alb reflected)',          repr='%f',cast_type=float,value=0.,llim=0,ulim=5,step=0.05,adjust=False,frame=["phoebe"],alias=['albedo'],context=['component','star','accretion_disk']),
         dict(qualifier='redist',description='Global redist par (1-redist) local heating, redist global heating',
              long_description="During reflection computations, a 'redist' fraction of the incoming light will be used to heat the entire object, while a fraction '1-redist' will be used to locally heat the object. If you want complete heat redistribution, set redist=1. If you only want local heating, set redist=0. Note that the incoming light is dependent on the value of the albedo ('alb') parameter. If alb=0, the 'redist' parameter has no effect since all the light will be reflected, and nothing will be used for heating. In summary, a fraction '1-alb' of the incoming light is reflected, a fraction 'redist*alb' is used for global heating, and a fraction '(1-redist)*alb' for local heating.",
              repr='%f',cast_type=float,value=0.,llim=0,ulim=1,step=0.05,adjust=False,frame=["phoebe"],context=['component','star','accretion_disk']),
         dict(qualifier='redisth',description='Horizontal redist par (redisth/redist) horizontally spread',
              long_description=("During reflection computations, a 'redisth' "
                                "fraction of the incoming light that is used "
                                "for heating (i.e. a 'redisth*redist' fraction of the "
                                "incoming light) will be used to heat the "
                                "object horizontally, as to mimick horizontal "
                                "winds going around the object."),
              repr='%f',cast_type=float,value=0.,llim=0,ulim=1,step=0.05,adjust=False,frame=["phoebe"],context=['component','star','accretion_disk']),
         dict(qualifier='syncpar',description='Synchronicity parameter',repr='%f',cast_type=float,value=1.,llim=0,ulim=10000.,step=0.01,adjust=False,alias=['f'],frame=["phoebe"],context='component'),
         dict(qualifier='gravb',  description='Bolometric gravity brightening',repr='%f',cast_type=float,value=1.0,llim=0,ulim=1.5,step=0.05,adjust=False,alias=['grb'],frame=["phoebe"],context='component'),
         dict(qualifier='pot',    description="Roche potential value",repr='%f',cast_type=float,value=4.,llim=0,ulim=1e10,step=0.01,adjust=False,frame=["phoebe"],context='component'),
         dict(qualifier='teff',   description='Mean effective temperature',repr='%.0f',cast_type=float,unit='K',value=10000.,llim=0.,ulim=1e20,step=1,adjust=False,frame=["phoebe"],context='component'),         
         dict(qualifier='morphology',   description='Binary type (unconstrained, detached...)',repr='%s',cast_type='choose',choices=['unconstrained','detached','semi-detached','overcontact'],value='unconstrained',frame=["phoebe"],context='component'),         
         #dict(qualifier='distance',description='Distance to the binary system',repr='%f',cast_type=float,value=10.,unit='pc',adjust=False,frame=['phoebe'],context='orbit'),
         dict(qualifier='irradiator',description='Treat body as irradiator of other objects',repr='',cast_type='make_bool',value=True,frame=['phoebe'],context=['component','star','accretion_disk']),
         dict(qualifier='abun',description='Metallicity',repr='%f',cast_type=float,value=0.,frame=['phoebe'],context=['component','star']),
         dict(qualifier='label',  description='Name of the body',repr='%s',cast_type='make_label',value='',frame=["phoebe"],context=['component','star','accretion_disk']),
        ]

#    INTERSTELLAR REDDENING
defs += [dict(qualifier='law',       description='Interstellar reddening law',repr='%s',cast_type='choose',choices=['chiar2006','fitzpatrick1999','fitzpatrick2004','donnel1994','cardelli1989','seaton1979'],value='fitzpatrick2004',frame=["phoebe"],context=['reddening:interstellar']),
         dict(qualifier='extinction',description='Passband extinction',repr='%f', unit='mag', cast_type=float,value=0,adjust=False,frame=["phoebe"],context=['reddening:interstellar']),
         dict(qualifier='passband',  description='Reference bandpass for extinction parameter',repr='%s',cast_type=str,value='JOHNSON.V',frame=["phoebe"],context=['reddening:interstellar']),
         dict(qualifier='Rv',        description='Total-to-selective extinction',repr='%f',cast_type=float,value=3.1,adjust=False,frame=["phoebe"],context=['reddening:interstellar']),
        ]

#    CIRCULAR SPOT CONTEXT
defs += [dict(qualifier='long',      description='Spot longitude at T0',           repr='%f',cast_type=float,value= 0.,unit='deg',llim=0,ulim=360.,step=1.0,adjust=False,frame=["phoebe"],context='circ_spot'),
         dict(qualifier='colat',     description='Spot colatitude at T0 (CHECK!)',            repr='%f',cast_type=float,value=90.,unit='deg',llim=0,ulim=180.,step=1.0,adjust=False,frame=["phoebe"],context='circ_spot'),
         dict(qualifier='angrad',    description='Spot angular radius',unit='deg',repr='%f',cast_type=float,value=5.,llim=0,ulim=90.,step=.01,adjust=False,frame=["phoebe"],context='circ_spot'),
         dict(qualifier='teffratio', description='Relative temperature difference in spot',repr='%f',cast_type=float,value=0.9,llim=0,ulim=10,step=0.05,adjust=False,frame=["phoebe"],context='circ_spot'),
         dict(qualifier='abunratio', description='Relative log(abundance) difference in spot',repr='%f',cast_type=float,value=0.0,llim=-5,ulim=5,step=0.01,adjust=False,frame=["phoebe"],context='circ_spot'),
         dict(qualifier='subdiv_num',description="Number of times to subdivide spot",repr='%d',cast_type=int,value=3,adjust=False,frame=['phoebe'],context='circ_spot'),
         dict(qualifier='t0',        description="Spot time zeropoint",repr='%f',cast_type=float,unit='JD',value=0.,adjust=False,frame=['phoebe'],context='circ_spot'),
        ]        

defs += [dict(qualifier='delta',    description='Stepsize for mesh generation via marching method',repr='%f',cast_type=float,value=0.1,
              long_description='The stepsize is approximately equal to the size of an edge of a typical triangle in the mesh, expressed in stellar radii. Beware that halving the stepsize, quadruples the number of mesh points. Also for very deformed objects, the number of surface elements can increase drastically (because the surface area is larger.',
              frame=['phoebe'],context=['mesh:marching']),
         dict(qualifier='maxpoints',description='Maximum number of triangles for marching method',repr='%d',cast_type=int,value=100000,frame=['phoebe'],context=['mesh:marching']),
         dict(qualifier='gridsize', description='Number of meshpoints for WD style discretization',repr='%d',cast_type=int,value=90,frame=['phoebe'],context=['mesh:wd']),
         dict(qualifier='alg', description='Select type of algorithm',repr='%s',cast_type='choose',choices=['c','python'],value='c',frame=['phoebe'],context=['mesh:marching']),
         dict(qualifier='radial', description='Number of meshpoints in the radial direction',repr='%d',cast_type=int,value=20,frame=['phoebe'],context=['mesh:disk']),
         dict(qualifier='longit', description='Number of meshpoints in the longitudinal direction',repr='%d',cast_type=int,value=50,frame=['phoebe'],context=['mesh:disk']),
         #dict(qualifier='style',    description='Discretization style',repr='%s',cast_type='choose',choices=['marching','cmarching','wd'],value='marching',frame=['phoebe'],context=['mesh'])
         ]        
        
#    DATA contexts
defs += [dict(qualifier='ld_func', description='Limb darkening model',repr='%s',cast_type='choose',choices=['uniform','linear','logarithmic', 'quadratic', 'square_root','power', 'claret', 'hillen', 'prsa'],value='uniform',frame=["phoebe"],context=['lcdep','amdep','rvdep']),
         dict(qualifier='ld_func', description='Bolometric limb darkening model',repr='%s',cast_type='choose',choices=['uniform','linear','logarithmic', 'quadratic', 'square_root','power','claret', 'hillen', 'prsa'],value='uniform',frame=["phoebe"],context=['component','star','accretion_disk']),
         dict(qualifier='ld_coeffs',       description='Limb darkening coefficients',long_description=("Limb darkening coefficients can be given in four ways: (1) using a short alias (e.g. 'kurucz') in which "
                                                                                  "case the FITS-table's filename will be derived from the alias, the limb darkening function and "
                                                                                  "other information like reddening, boosting etc... (2) using a relative filename, in which case "
                                                                                  "the file will be looked up in the ld_coeffs directory in src/phoebe/atmospheres/tables/ld_coeffs/"
                                                                                  " (3) via an absolute filename or (4) via a list of user-specified floats. In the latter case, "
                                                                                  "you need to have as many coefficients as the 'ld_func' requires"),repr='%s',value=[1.],cast_type='return_string_or_list',frame=["phoebe"],context=['lcdep','amdep']),
         dict(qualifier='ld_coeffs',       description='Bolometric limb darkening coefficients',long_description=("Bolometric limb darkening coefficients can be given in four ways: (1) using a short alias (e.g. 'kurucz') in which "
                                                                                  "case the FITS-table's filename will be derived from the alias, the limb darkening function and "
                                                                                  "other information like reddening, boosting etc... (2) using a relative filename, in which case "
                                                                                  "the file will be looked up in the ld_coeffs directory in src/phoebe/atmospheres/tables/ld_coeffs/"
                                                                                  " (3) via an absolute filename or (4) via a list of user-specified floats. In the latter case, "
                                                                                  "you need to have as many coefficients as the 'ld_func' requires"),repr='%s',value=[1.],cast_type='return_string_or_list',frame=["phoebe"],context=['component','star','accretion_disk']),
         dict(qualifier='passband', description='Photometric passband',repr='%s',cast_type='make_upper',value='JOHNSON.V',frame=["phoebe"],context=['lcdep','amdep','sidep', 'analytical:binary']),
         dict(qualifier='pblum',    description='Passband luminosity',repr='%f',cast_type=float,value=-1.0,adjust=False,frame=["phoebe"],context=['lcdep','amdep','spdep','ifdep','pldep']),
         dict(qualifier='l3',       description='Third light',repr='%f',cast_type=float,value=0.,adjust=False,frame=["phoebe"],context=['lcdep','amdep','spdep','ifdep','pldep']),
         dict(qualifier='alb',      description="Passband Bond's albedo , alb=0 is no reflection",
                               long_description=("The passband albedo sets the "
                                                 "fraction of reflected versus incoming light within the passband. Thus, "
                                                 "grey scattering means all the passband albedos are equal. A passband albedo "
                                                 "of unity means no redistribution of wavelengths. A passband albedo exceeding "
                                                 "unity means that light from other wavelengths is redistributed inside the "
                                                 "current passband. Passband albedo cannot be negative, you cannot reflect less "
                                                 "light than no light. Passband albedo of zero means that within that passband,"
                                                 "all the light is absorbed."), repr='%f',cast_type=float,value=0.,llim=0,ulim=5,step=0.01,adjust=False,frame=["phoebe"],context=['lcdep','amdep','rvdep','ifdep','spdep','pldep']),
         dict(qualifier='method',   description='Method for calculation of total intensity',repr='%s',cast_type='choose',choices=['analytical','numerical'],value='numerical',frame=["phoebe"],context='lcdep'),
         dict(qualifier='label',    description='Name of the observable',repr='%s',cast_type='make_label',value='',hidden=True,frame=["phoebe"],context=['puls','circ_spot']),
         dict(qualifier='ref',    description='Name of the observable',repr='%s',cast_type=str,value='',frame=["wd"],context=['lc','rv']),
         dict(qualifier='ref',      description='Name of the observable',repr='%s',cast_type=str,value='',frame=["phoebe"],hidden=True,context=['lcdep','amdep','rvdep','ifdep','spdep','pldep','etvdep','sidep']),
         dict(qualifier='boosting',  description='Take photometric doppler shifts into account',repr='',value=True,cast_type='make_bool',frame=['phoebe'],context=['lcdep','amdep','ifdep','spdep','pldep','rvdep']),
         dict(qualifier='scattering',  description='Scattering phase function',repr='',value='isotropic',cast_type='choose',choices=['isotropic','henyey','henyey2', 'rayleigh','hapke'], frame=['phoebe'],context=['lcdep','amdep','ifdep','spdep','pldep','rvdep']),
         dict(qualifier='time',     description='Timepoint LC',repr='%s',unit='JD',value=[],frame=["phoebe"],context='lcsyn'),
         dict(qualifier='phase',     description='Phasepoint LC',repr='%s',unit='cy',value=[],frame=["phoebe"],context='lcsyn'),
         dict(qualifier='flux',   description='Calculated flux',repr='%s',value=[],unit='W/m3',frame=["phoebe"],context='lcsyn'),
         dict(qualifier='samprate',   description='Sampling rate to apply',repr='%s',value=[],frame=["phoebe"],context=['lcsyn','rvsyn']),
         dict(qualifier='used_samprate',   description='Applied sampling rate',repr='%s',value=[],frame=["phoebe"],context=['lcsyn','rvsyn']),
         #dict(qualifier='used_exptime',   description='Applied exposure time',repr='%s',value=[],frame=["phoebe"],context='lcsyn'),
         dict(qualifier='filename', description='Name of the file containing the data',repr='%s',cast_type=str,value='',frame=['phoebe'],context=['lcobs','spobs','rvobs','ifobs','plobs','etvobs','amobs','siobs','lcsyn','ifsyn','rvsyn','spsyn','ifobs','ifsyn','plsyn','etvsyn','amsyn','sisyn']),
         dict(qualifier='ref',    description='Name of the data structure',repr='%s',cast_type=str,value='',hidden=False,frame=["phoebe"],context=['lcobs','rvobs','spobs','ifobs','etvobs','psdep','lcsyn','spsyn','amsyn','rvsyn','ifsyn','plsyn','plobs','etvsyn','amobs','orbsyn']),
         #dict(qualifier='scale',    description='Linear scaling constant',repr='%f',cast_type=float,value=1.0,frame=["phoebe"],context=['lcsyn','spsyn','amsyn','rvsyn','ifsyn','plsyn','etvsyn','orbsyn']),
         #dict(qualifier='offset',    description='Offset constant',repr='%f',cast_type=float,value=0.0,frame=["phoebe"],context=['lcsyn','spsyn','amsyn','rvsyn','ifsyn','plsyn','etvsyn','orbsyn']),
         dict(qualifier='time',     description='Timepoints of the data',repr='%s',cast_type=np.array,value=[],unit='JD',frame=['phoebe'],context=['lcobs','spobs','rvobs','ifobs','plobs','etvobs','amobs','siobs']),
         dict(qualifier='phase',     description='Phasepoints of the data',repr='%s',cast_type=np.array,value=[],unit='cy',frame=['phoebe'],context=['lcobs','spobs','rvobs','ifobs','plobs','etvobs']),
         dict(qualifier='flux',   description='Observed signal',repr='%s',cast_type=np.array,value=[],unit='W/m3',frame=["phoebe"],context='lcobs'),
         dict(qualifier='sigma',  description='Data sigma',repr='%s',cast_type=np.array,value=[],unit='W/m3',frame=['phoebe'],context=['lcobs','lcsyn']),
         dict(qualifier='sigma',  description='Data sigma',repr='%s',cast_type=np.array,value=[],unit='km/s',frame=['phoebe'],context=['rvobs','rvsyn']),
         dict(qualifier='sigma',  description='Data sigma',repr='%s',cast_type=np.array,value=[],unit='d',frame=['phoebe'],context=['etvobs','etvsyn']),
         dict(qualifier='flag',    description='Signal flag',repr='%s',cast_type=np.array,value=[],frame=["phoebe"],context=['lcobs','rvobs']),
         dict(qualifier='weight',    description='Signal weight',repr='%s',cast_type=np.array,value=[],frame=["phoebe"],context=['lcobs','rvobs']),
         dict(qualifier='exptime',    description='Signal exposure time',repr='%s',cast_type=np.array,value=[],unit='s', frame=["phoebe"],context=['lcobs','rvobs']),
         dict(qualifier='samprate',    description='Signal sampling rate',repr='%s',cast_type=np.array,value=[],frame=["phoebe"],context=['lcobs','rvobs']),
         
         #dict(qualifier='fittransfo',    description='Transform variable in fit',repr='%s',cast_type=str,value='linear',frame=["phoebe"],context=['lcobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','flux','sigma','flag','weight'],cast_type='return_list_of_strings',hidden=True,frame=["phoebe"],context=['lcobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','flux','samprate','used_samprate'],cast_type='return_list_of_strings',frame=["phoebe"],context=['lcsyn']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','rv','sigma'],frame=["phoebe"],context=['rvobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','etv','sigma'],frame=["phoebe"],context=['etvobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','etv'],frame=["phoebe"],context=['etvsyn']),
         dict(qualifier='etv',  description='Eclipse timing variations (eclipse_time - time)',repr='%s',cast_type=np.array,value=[],unit='d',frame=["phoebe"],context=['etvobs','etvsyn']),
         dict(qualifier='eclipse_time', description='Observed eclipse time',repr='%s', cast_type=np.array,value=[],unit='d',frame=["phoebe"],context=['etvobs','etvsyn']),
         dict(qualifier='time', description='Timepoint ETV',repr='%s',value=[],unit='JD',frame=["phoebe"],context='etvsyn'),
         dict(qualifier='phase',     description='Phasepoint ETV',repr='%s',unit='cy',value=[],frame=["phoebe"],context='etvsyn'),
         dict(qualifier='rv',   description='Radial velocities',repr='%s',cast_type=np.array,value=[],unit='km/s',frame=["phoebe"],context='rvobs'),
         dict(qualifier='rv',   description='Radial velocities',repr='%s',value=[],unit='km/s',frame=["phoebe"],context='rvsyn'),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','rv'],frame=["phoebe"],context=['rvsyn']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['wavelength','time','flux','continuum'],cast_type='return_list_of_strings',frame=["phoebe"],context=['spobs','spsyn']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['wavelength','time','flux','V','Q','U','continuum'],cast_type='return_list_of_strings',frame=["phoebe"],context=['plobs','plsyn']),
         dict(qualifier='user_columns', description='Column names given by user', repr='%s', write_protected=True,value=None, hidden=True, cast_type='return_self', frame=['phoebe'], context=['lcobs','rvobs','ifobs','spobs','plobs','etvobs']),
         dict(qualifier='user_components', description='Component names given by user', repr='%s', write_protected=True,value=None, hidden=True, cast_type='return_self', frame=['phoebe'], context=['lcobs','rvobs', 'ifobs','spobs','plobs','etvobs']),
         dict(qualifier='user_dtypes', description='Data types given by user', repr='%s', write_protected=True,value=None, hidden=True, cast_type='return_self', frame=['phoebe'], context=['lcobs','rvobs', 'ifobs','spobs','plobs','etvobs']),
         dict(qualifier='user_units', description='Units given by user', repr='%s', write_protected=True, value=None, hidden=True, cast_type='return_self', frame=['phoebe'], context=['lcobs','rvobs', 'ifobs','spobs','plobs','etvobs']),
        ]

# Orbsyn context
defs += [dict(qualifier='bary_time', description='Barycentric times', repr='%s', value=[], context='orbsyn', frame='phoebe'),
         dict(qualifier='prop_time', description='Proper times', repr='%s', value=[], context='orbsyn', frame='phoebe'),
         dict(qualifier='position', description='Position of the object in the orbit', repr='%s', value=[], context='orbsyn', frame='phoebe'),
         dict(qualifier='velocity', description='Velocity of the object in the orbit', repr='%s', value=[], context='orbsyn', frame='phoebe'),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['bary_time', 'prop_time'],cast_type='return_list_of_strings',frame=["phoebe"],context=['orbsyn']),
        ]

defs += [dict(qualifier='wavelength',description='Wavelengths of observed spectrum',repr='%s',value=[],unit='nm',frame=["phoebe"],context='spobs'),
         dict(qualifier='continuum',  description='Continuum intensity of the spectrum',repr='%s',value=[],frame=["phoebe"],context='spobs'),
         dict(qualifier='flux',  description='Flux of the spectrum',repr='%s',value=[],frame=["phoebe"],context='spobs'),
         dict(qualifier='sigma',  description='Noise in the spectrum',repr='%s',value=[],frame=["phoebe"],context=['spobs','spsyn']),
         dict(qualifier='snr',  description='Signal to noise of the spectrum',repr='%s',value=[],frame=["phoebe"],context='spobs'),
         dict(qualifier='offset',       description='Linear scaling constant to match obs with syn',repr='%f',cast_type=float,value=0.,adjust=False,frame=["phoebe"],context=['lcobs','spobs','ifobs','plobs','etvobs']),
         dict(qualifier='scale',    description='Linear scaling factor to match obs with syn',repr='%f',cast_type=float,value=1.0,adjust=False,frame=["phoebe"],context=['lcobs','spobs','ifobs','plobs']),
         dict(qualifier='vgamma_offset',       description='Offset in systemic velocity',repr='%f',cast_type=float,value=0.,adjust=False,frame=["phoebe"],context=['rvobs'],alias=['offset']),
         dict(qualifier='statweight',    description='Statistical weight in overall fitting',repr='%f',cast_type=float,value=1.0,adjust=False,frame=["phoebe"],context=['lcobs','spobs','ifobs','plobs','rvobs']),
         ]        

defs += [dict(qualifier='wavelength',description='Wavelengths of calculated spectrum',repr='%s',value=[],unit='nm',frame=["phoebe"],context='plobs'),
         dict(qualifier='continuum',  description='Continuum intensity of the spectrum',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='flux',  description='Observed Stokes I profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='sigma',  description='Noise in the Stokes I profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='V',  description='Observed Stokes V profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='sigma_V',  description='Noise in the Stokes V profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='Q',  description='Observed Stokes Q profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='sigma_Q',  description='Noise in the Stokes Q profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='U',  description='Observed Stokes U profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='sigma_U',  description='Noise in the Stokes U profile',repr='%s',value=[],frame=["phoebe"],context='plobs'),
         dict(qualifier='V',  description='Calculated Stokes V profile',repr='%s',value=[],frame=["phoebe"],context='plsyn'),
         dict(qualifier='Q',  description='Calculated Stokes Q profile',repr='%s',value=[],frame=["phoebe"],context='plsyn'),
         dict(qualifier='U',  description='Calculated Stokes U profile',repr='%s',value=[],frame=["phoebe"],context='plsyn'),
         ]
        
defs += [dict(qualifier='ld_coeffs',description='Limb darkening coefficients',repr='%s',cast_type='return_string_or_list',value=[1.],frame=["phoebe"],context='rvdep'),
         dict(qualifier='passband', description='Photometric passband',repr='%s',value='JOHNSON.V',cast_type='make_upper',frame=["phoebe"],context='rvdep'),
         dict(qualifier='method',   description='Method for calculation of radial velocity',repr='%s',cast_type='choose',choices=['flux-weighted','dynamical'],value='flux-weighted',frame=["phoebe"],context='rvdep'),
         dict(qualifier='time',     description='Timepoint',unit='JD',repr='%s',value=[],frame=["phoebe"],context=['rvsyn','pssyn','amsyn','sisyn']),
         dict(qualifier='phase',     description='Phasepoint',unit='cy',repr='%s',value=[],frame=["phoebe"],context=['rvsyn','pssyn','amsyn','sisyn']),
        ]        

defs += [dict(qualifier='ld_func', description='Limb darkening model',repr='%s',cast_type='choose',choices=['uniform','linear','logarithmic', 'quadratic', 'square_root','claret'],value='uniform',frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='ld_coeffs',description='Limb darkening coefficients',repr='%s',cast_type='return_string_or_list',value=[1.],frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='passband', description='Photometric passband',repr='%s',value='JOHNSON.V',cast_type='make_upper',frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='method',   description='Method for calculation of spectrum',repr='%s',cast_type='choose',choices=['analytical','numerical'],value='numerical',frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='weak_field', description='Type of approximation to use (weak field or not)',repr='',cast_type='make_bool',value=False,frame=['phoebe'],context='pldep'),
         dict(qualifier='glande',   description='Lande factor',repr='%f',cast_type=float,value=1.2,frame=["phoebe"],context=['pldep']),
         dict(qualifier='R',        description='Resolving power lambda/Dlambda (or c/Deltav)',repr='%g',cast_type=float,value=400000.,frame=["phoebe"],context=['spobs','plobs']),
         dict(qualifier='R_input',  description='Resolving power lambda/Dlambda (or c/Deltav) of the model',repr='%g',cast_type=float,value=0.,frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='vmacro',   description='Analytical macroturbulent velocity',repr='%g',unit='km/s',cast_type=float,value=0.,adjust=False,frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='vmicro',   description='Microturbulent velocity', long_description="Currently only used when profile=gauss.", repr='%g',unit='km/s',cast_type=float,value=5.0,adjust=False,frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='depth',   description='Depth of Gaussian profile', long_description="Currently only used when profile=gauss.", repr='%g',cast_type=float,value=0.4,adjust=False,frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='alphaT',   description='Temperature dependence of depth of profile', repr='%g',cast_type=float,value=0.0,adjust=False,frame=["phoebe"],context=['spdep','pldep']),

         dict(qualifier='vgamma_offset', description='Systemic velocity',long_description="A negative systemic velocity means the synthetic profile will be blue-shifted (towards shorter wavelengths). Note that the observed profile is never changed, all modifications are done on the synthetic spectra only.", repr='%f',llim=-1e6,ulim=1e6,step=0.1,adjust=False,cast_type=float,value=0.,unit='km/s',alias=['vga'],frame=["phoebe"],context=['spobs','plobs']),
         dict(qualifier='profile',  description='Line profile source (gridname or "gauss")',repr='%s',cast_type=str,value='gauss',frame=["phoebe"],context=['spdep','pldep']),
         dict(qualifier='time',     description='Timepoint',repr='%s',value=[],frame=["phoebe"],context=['spsyn','plsyn']),
         dict(qualifier='wavelength',description='Wavelengths of calculated spectrum',repr='%s',value=[],unit='nm',frame=["phoebe"],context=['spsyn','plsyn']),
         dict(qualifier='flux',      description='Intensity of calculated spectrum',repr='%s',value=[],unit='W/m3',frame=["phoebe"],context=['spsyn','plsyn']),
         dict(qualifier='continuum', description='Continuum of calculated spectrum',repr='%s',value=[],unit='W/m3',frame=["phoebe"],context=['spsyn','plsyn']),
         dict(qualifier='stokesV',   description='Stokes V profile',repr='%s',value=[],frame=["phoebe"],context=['plsyn']),
        ]

defs += [dict(qualifier='ld_func',   description='Limb darkening model',repr='%s',cast_type=str,value='uniform',frame=["phoebe"],context='ifdep'),
         dict(qualifier='ld_coeffs',         description='Limb darkening coefficients',repr='%s',cast_type='return_string_or_list',value=[1.],frame=["phoebe"],context='ifdep'),
         dict(qualifier='passband',   description='Photometric passband',repr='%s',cast_type='make_upper',value='JOHNSON.V',frame=["phoebe"],context='ifdep'),
         dict(qualifier='bandwidth_smearing',   description='Type of bandwidth smearing to use',
                        long_description=("(1) 'off' means the image will be treated monochromatically (2) 'complex' means "
                                          "that the image will be assumed to be the same in all wavelengths, but an average "
                                          "of the image, weighted with the passband response function, will be made with "
                                          "different spatial frequencies (3) 'power' is like the previous option, but "
                                          "the average of the power is taken (4) 'detailed' means that separate images are made "
                                          "in different subdivisions of the passband, and finally combined. The resolution "
                                          "for options (2,3,4) are set via the parameter bandwidth_subdiv."),
                        repr='%s',cast_type='choose',value='off',
                                      choices=['off','complex','power','detailed'], frame=["phoebe"],context='ifdep'),
         dict(qualifier='bandwidth_subdiv',   description='Resolution of the bandwidth smearing',
                         long_description=("No bandwidth smearing (monochromatic) is set with bandwidth_subdiv=0"),repr='%d',cast_type=int,value=10, frame=["phoebe"],context='ifdep'),
         #dict(qualifier='baseline',   description='Length of the baseline',repr='%f',value=0.,unit='m',frame=["phoebe"],context=['ifobs','ifsyn']),
         #dict(qualifier='posangle',   description='Position angle of the baseline',repr='%f',value=0.,unit='deg',frame=["phoebe"],context=['ifobs','ifsyn']),
         #dict(qualifier='freq',  description='Cyclic frequency',repr='%s',value=[],unit='cy/arcsec',frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='ucoord',   description='U-coordinate',repr='%s',value=[],unit='m',frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='vcoord',   description='V-coordinate',repr='%s',value=[],unit='m',frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='eff_wave',   description='Effective wavelength',repr='%s',value=[],unit='AA',frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='vis2', description='Squared Visibility',repr='%s',value=[],frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='sigma_vis2', description='Error on squared visibility',repr='%s',value=[],frame=["phoebe"],context=['ifobs']),
         dict(qualifier='vphase',   description='Phase of visibility',repr='%s',value=[],unit='rad', frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='sigma_vphase',   description='Error on phase of visibility',repr='%s',value=[],frame=["phoebe"],context=['ifobs']),
         dict(qualifier='total_flux',   description='Total flux',repr='%s',value=[],frame=["phoebe"],context=['ifsyn']),
         dict(qualifier='ucoord_2',   description='U-coordinate of second baseline',repr='%s',value=[],unit='m',frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='vcoord_2',   description='V-coordinate of second baseline',repr='%s',value=[],unit='m',frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='vis2_2', description='Squared Visibility of second baseline',repr='%s',value=[],frame=["phoebe"],context=['ifsyn']),
         dict(qualifier='vphase_2',   description='Phase of visibility of second baseline',repr='%s',value=[],unit='rad', frame=["phoebe"],context=['ifsyn']),         
         dict(qualifier='total_flux_2',   description='Total flux of second baseline',repr='%s',value=[],frame=["phoebe"],context=['ifsyn']),
         dict(qualifier='vis2_3', description='Squared Visibility of third baseline',repr='%s',value=[],frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='sigma_vis2_3', description='Error on squared visibility of third baseline',repr='%s',value=[],frame=["phoebe"],context=['ifobs']),
         dict(qualifier='vphase_3',   description='Phase of visibility',repr='%s',value=[],unit='rad', frame=["phoebe"],context=['ifsyn']),
         dict(qualifier='triple_ampl',   description='Triple amplitude',repr='%s',value=[],unit='rad', frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='sigma_triple_ampl',   description='Error on triple amplitude',repr='%s',value=[],frame=["phoebe"],context=['ifobs']),
         dict(qualifier='closure_phase',   description='Closure phase',repr='%s',value=[],unit='rad', frame=["phoebe"],context=['ifobs','ifsyn']),
         dict(qualifier='sigma_closure_phase',   description='Error on closure phase',repr='%s',value=[],frame=["phoebe"],context=['ifobs']),
         dict(qualifier='total_flux_3',   description='Total flux of third baseline',repr='%s',value=[],frame=["phoebe"],context=['ifsyn']),
         dict(qualifier='images',   description='Basename for files of original baseline plots',cast_type=str,repr='%s',value='',frame=["phoebe"],context=['ifobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','ucoord','vcoord','vis2','sigma_vis2','vphase','sigma_vphase',
                                                                                       'ucoord_2','vcoord_2', 'triple_ampl', 'sigma_triple_ampl',
                                                                                       'closure_phase', 'sigma_closure_phase'],cast_type='return_list_of_strings',frame=["phoebe"],context=['ifobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','ucoord','vcoord','vis2','vphase',
                                                                                       'ucoord_2','vcoord_2', 'vis2_2', 'vphase_2',
                                                                                       'vis2_3', 'closure_phase'],cast_type='return_list_of_strings',frame=["phoebe"],context=['ifsyn']),
         dict(qualifier='time',     description='Timepoint',repr='%s',value=[],unit='JD',frame=["phoebe"],context='ifsyn'),
        ]

defs += [dict(qualifier='eclx', description='Ecliptic Cartesian x-coordinates of observer', repr='%s', cast_type=np.array, value=[], unit='au',frame='phoebe', context='amobs'),
         dict(qualifier='ecly', description='Ecliptic Cartesian y-coordinates of observer', repr='%s', cast_type=np.array, value=[], unit='au',frame='phoebe', context='amobs'),
         dict(qualifier='eclz', description='Ecliptic Cartesian z-coordinates of observer', repr='%s', cast_type=np.array, value=[], unit='au',frame='phoebe', context='amobs'),
         dict(qualifier='time_offset',     description='Zeropoint date offset for observations',unit='JD',repr='%f',llim=-2e10,ulim=2e10,step=0.001,adjust=False,cast_type=float,value=0.,alias=['hjd0'],frame=["phoebe"],context=['amobs', 'lcobs', 'rvobs', 'spobs', 'plobs', 'ifobs']),
         dict(qualifier='delta_ra', description='Right ascension offset coordinate', repr='%s', value=[],frame=['phoebe'], context=['amsyn','amobs']),
         dict(qualifier='delta_dec', description='Declination offset coordinate', repr='%s', value=[],frame=['phoebe'], context=['amsyn','amobs']),
         dict(qualifier='plx_lambda', description='Longitude of parallax circle in ecliptic coordinates', repr='%s', value=[],frame=['phoebe'], context=['amsyn']),
         dict(qualifier='plx_beta', description='Latitude of parallax circle in ecliptic coordinates', repr='%s', value=[],frame=['phoebe'], context=['amsyn']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','delta_ra','delta_dec','eclx','ecly','eclz'],cast_type='return_list_of_strings',frame=["phoebe"],context=['amobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','delta_ra','delta_dec','plx_lambda','plx_beta'],cast_type='return_list_of_strings',frame=["phoebe"],context=['amsyn']),
         ]
         
defs += [dict(qualifier='pos_angle', description='Position angle (East of North?)', repr='%s', unit='deg', value=[],frame=['phoebe'], context=['sisyn','siobs']),
         dict(qualifier='sigma_pos_angle', description='Uncertainty on position angle', repr='%s', unit='deg', value=[],frame=['phoebe'], context=['sisyn','siobs']),
         dict(qualifier='sep', description='Separation', repr='%s', unit='as', value=[],frame=['phoebe'], context=['sisyn','siobs']),
         dict(qualifier='sigma_sep', description='Uncertainty on separation', repr='%s', unit='as', value=[],frame=['phoebe'], context=['sisyn','siobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','pos_angle','sigma_pos_angle', 'sep', 'sigma_sep'],cast_type='return_list_of_strings',frame=["phoebe"],context=['siobs']),
         dict(qualifier='columns',  description='Data columns',repr='%s',value=['time','pos_angle','separation'],cast_type='return_list_of_strings',frame=["phoebe"],context=['sisyn']),
         ]


defs += [dict(qualifier='coordinates',description="Location of geometrical barycenter",cast_type=np.array,repr='%s',value=[0.,0.,0.],unit='Rsol',frame=["phoebe"],context=['point_source']),
         dict(qualifier='photocenter',description="Location of passband photocenter",cast_type=np.array,repr='%s',value=[0.,0.,0.],unit='Rsol',frame=["phoebe"],context=['point_source']),
         dict(qualifier='velocity',   description="Velocity of the body",repr='%s',cast_type=np.array,value=[0.,0.,0.],unit='km/s',frame=["phoebe"],context=['point_source']),
         dict(qualifier='distance',   description="Distance to the body",repr='%f',cast_type=float,value=0,unit='pc',frame=["phoebe"],context=['point_source']),
         dict(qualifier='radius',     description='Mean radius',repr='%g',value=0.,unit='Rsol',frame=["phoebe"],context=['point_source']),
         dict(qualifier='mass',       description="Mass of the body as a point source",repr='%g',value=0.,unit='Msol',frame=["phoebe"],context=['point_source']),
         dict(qualifier='teff',       description="passband mean temperature",repr='%g',value=0.,unit='K',frame=["phoebe"],context=['point_source']),
         dict(qualifier='surfgrav', description="passband mean surface gravity",repr='%g',value=0.,unit='[cm/s2]',frame=["phoebe"],context=['point_source']),
         dict(qualifier='intensity', description="passband mean intensity",repr='%g',value=0.,unit='W/m3',frame=["phoebe"],context=['point_source']),
        ]
        
defs += [dict(qualifier='coordinates',description="Location of the body's geometrical barycenter",repr='%s',value=[],unit='Rsol',frame=["phoebe"],context=['psdep']),
         dict(qualifier='photocenter',description="Location of the body's passband photocenter",repr='%s',value=[],unit='Rsol',frame=["phoebe"],context=['psdep']),
         dict(qualifier='velocity',   description="Velocity of the body",repr='%s',value=[],unit='Rsol/d',frame=["phoebe"],context=['psdep']),
         dict(qualifier='mass',       description="Mass of the body as a point source",repr='%s',value=[],unit='Msol',frame=["phoebe"],context=['psdep']),
         dict(qualifier='teff',       description="passband mean temperature",repr='%s',value=[],unit='K',frame=["phoebe"],context=['psdep']),
         dict(qualifier='surfgrav', description="passband mean surface gravity",repr='%s',value=[],unit='m/s2',frame=["phoebe"],context=['psdep']),
         dict(qualifier='intensity', description="passband mean intensity",repr='%s',value=[],unit='W/m3',frame=["phoebe"],context=['psdep']),
        ]

#    PULSATION contexts
defs += [dict(qualifier='freq',     description='Pulsation frequency',repr='%f',cast_type=float,value=1.,unit='cy/d',frame=["phoebe"],adjust=False,context='puls'),
         dict(qualifier='phase',    description='Pulsation phase',repr='%f',cast_type=float,value=0.,unit='cy',adjust=False,frame=["phoebe"],context='puls'),
         dict(qualifier='ampl',     description='Pulsation amplitude (fractional radius)',repr='%f',cast_type=float,adjust=False,value=0.00,frame=["phoebe"],context='puls'),
         dict(qualifier='l',        description='Degree of the mode',repr='%d',cast_type=int,value=0,frame=["phoebe"],context='puls'),
         dict(qualifier='m',        description='Azimuthal order of the mode',repr='%d',cast_type=int,value=0,frame=["phoebe"],context='puls'),
         dict(qualifier='k',        description='Horizontal/vertical displacement',repr='%f',adjust=False, cast_type=float,value=0.00,frame=["phoebe"],context='puls'),
         dict(qualifier='t0',       description='Zeropoint time for pulsational ephemeris',repr='%f',adjust=False, cast_type=float,value=0.0,frame=["phoebe"],context='puls'),
         dict(qualifier='dfdt',     description='Linear frequency shift',repr='%f',adjust=False, cast_type=float,value=0.0,unit='cy/d2',frame=["phoebe"],context='puls'),
         dict(qualifier='ledoux_coeff',description='Ledoux Cln',repr='%f',cast_type=float,value=0.,
         long_description="The Ledoux coefficients determines the rotation splitting of frequencies beyond the standard geometrical splitting. If the coefficients is zero, there is only geometrical splitting. Else, higher order effects can be taken into account.",frame=['phoebe'],context='puls'),
         dict(qualifier='amplteff',description='Amplitude of temperature perturbation',repr='%g',cast_type=float,value=0.0,adjust=False,frame=["phoebe"],context='puls'),
         dict(qualifier='phaseteff',description='Phase of temperature perturbation',repr='%g',cast_type=float,value=0.,adjust=False,frame=["phoebe"],context='puls'),
         dict(qualifier='amplgrav',description='Amplitude of gravity perturbation',repr='%g',cast_type=float,value=0.0,adjust=False,frame=["phoebe"],context='puls'),
         dict(qualifier='phasegrav',description='Phase of gravity perturbation',repr='%g',cast_type=float,value=0.0,adjust=False,frame=["phoebe"],context='puls'),
         dict(qualifier='incl',     description='Angle between rotation and pulsation axis',unit='deg',repr='%f',llim=0,ulim=360,step=0.01,adjust=False,cast_type=float,value=0.,frame=["phoebe"],context='puls'),
         dict(qualifier='phaseincl',description='Phase angle of obliquity',repr='%g',cast_type=float,value=0.0, unit='deg', adjust=False,frame=["phoebe"],context='puls'),
         dict(qualifier='trad_coeffs',  description='B vector for traditional approximation',repr='%s',cast_type=np.array,value=[],frame=['phoebe'],context='puls'),
         dict(qualifier='scheme',   description='Type of approximation for description of pulsations',repr='%s',cast_type='choose',choices=['nonrotating','coriolis','traditional approximation'],value='nonrotating',frame=["phoebe"],context='puls'),
        ]

#    MAGNETIC FIELD contexts

defs += [dict(qualifier='Bpolar',     description='Polar magnetic field strength',repr='%f',cast_type=float,adjust=False,value=1.,unit='G',frame=["phoebe"],context='magnetic_field:dipole'),
         dict(qualifier='beta',       description='Magnetic field angle wrt rotation axis',repr='%f',cast_type=float,adjust=False,value=0.,unit='deg',frame=["phoebe"],context='magnetic_field:dipole'),
         dict(qualifier='phi0',      description='Phase angle of magnetic field',repr='%f',cast_type=float,adjust=False,value=90.,unit='deg',frame=["phoebe"],context='magnetic_field:dipole'),
         dict(qualifier='Bpolar',     description='Polar magnetic field strength',repr='%f',cast_type=float,adjust=False,value=1.,unit='G',frame=["phoebe"],context='magnetic_field:quadrupole'),
         dict(qualifier='beta1',       description='First magnetic moment angle wrt rotation axis',repr='%f',cast_type=float,adjust=False,value=0.,unit='deg',frame=["phoebe"],context='magnetic_field:quadrupole'),
         dict(qualifier='phi01',      description='Phase angle of first magnetic moment',repr='%f',cast_type=float,adjust=False,value=90.,unit='deg',frame=["phoebe"],context='magnetic_field:quadrupole'),
         dict(qualifier='beta2',       description='Second magnetic moment angle wrt rotation axis',repr='%f',cast_type=float,adjust=False,value=0.,unit='deg',frame=["phoebe"],context='magnetic_field:quadrupole'),
         dict(qualifier='phi02',      description='Phase angle of Second magnetic moment',repr='%f',cast_type=float,adjust=False,value=90.,unit='deg',frame=["phoebe"],context='magnetic_field:quadrupole'),
        ]

# VELOCITY FIELD contexts
defs += [dict(qualifier='vmacro_rad', description='Radial macroturbulence component', cast_type=float, repr='%f',adjust=False,value=0.0,unit='km/s',frame=['phoebe'], context='velocity_field:turb'),
         dict(qualifier='vmacro_tan', description='Tangentional macroturbulence component', cast_type=float, repr='%f',adjust=False,value=0.0,unit='km/s',frame=['phoebe'], context='velocity_field:turb'),
         ]

defs += [dict(qualifier='bottom', description='Relative depth of meridional cell', cast_type=float, repr='%f',adjust=False,value=0.5,frame=['phoebe'], context='velocity_field:meri'),
         dict(qualifier='location', description='Relative location of stellar surface wrt circulation', cast_type=float, repr='%f',adjust=False,value=1.0,frame=['phoebe'], context='velocity_field:meri'),
         dict(qualifier='vmeri_ampl', description='Amplitude of meridional velocity', cast_type=float, repr='%f',adjust=False,value=0.002,unit='km/s',frame=['phoebe'], context='velocity_field:meri'),
         dict(qualifier='penetration_depth', description='Penetration depth', cast_type=float, repr='%f',adjust=False,value=0.2,frame=['phoebe'], context='velocity_field:meri'),
         dict(qualifier='latitude', description='Latitude of start of circulation', cast_type=float, repr='%f',adjust=False,value=0.0,unit='deg', frame=['phoebe'], context='velocity_field:meri'),
         ]

defs += [dict(qualifier='cells', description='Number of granulation cells', cast_type=int, repr='%d',adjust=False,value=100,frame=['phoebe'], context='granulation'),
         #dict(qualifier='seed', description='Random generator seed', cast_type=int, repr='%d',value=1111,frame=['phoebe'], context='granulation'),
         dict(qualifier='vgran_ampl', description='Amplitude of granulation velocity', cast_type=float, repr='%f',adjust=False,value=2.0,unit='km/s',frame=['phoebe'], context='granulation'),
         dict(qualifier='vgran_angle', description='Maximum perturbation on rad. comp. velocity', cast_type=float, repr='%f',adjust=False,value=-1,unit='deg',frame=['phoebe'], context='granulation'),
         dict(qualifier='teff_ampl', description='Amplitude of effective temperature variation', cast_type=float, repr='%f',adjust=False,value=10.0,unit='K',frame=['phoebe'], context='granulation'),
         dict(qualifier='pattern', description='Type of pattern to apply',
              long_description=('The granulation pattern is determined by the metric used to add the '
                                'Worley noise. "f1" corresponds to Gaussian bubbles on the surface '
                                'while "f2-f1" resembles granulation cells.'), cast_type='choose', choices=['f1', 'f2', 'f2-f1', 'f1-f2'], repr='%s',value='f2-f1',frame=['phoebe'], context='granulation'),
         ]

# SCATTERING contexts
defs += [dict(qualifier='asymmetry', description='Scattering asymmetry (negative = backwards, positive = forwards)', cast_type=float, repr='%f', llim=-1, ulim=1, adjust=False,value=0.0,frame=['phoebe'], context='scattering:henyey'),
         dict(qualifier='asymmetry1', description='Forward scattering strength (>0)', cast_type=float, repr='%f',adjust=False,value=0.8, llim=0, ulim=1, frame=['phoebe'], context='scattering:henyey2'),
         dict(qualifier='asymmetry2', description='Backward scattering strength (<0)', cast_type=float, repr='%f',adjust=False,value=-0.38, llim=-1, ulim=0, frame=['phoebe'], context='scattering:henyey2'),
         dict(qualifier='scat_ratio', description='Ratio between forward and backward scattering', cast_type=float, repr='%f',adjust=False,value=0.9, llim=0, ulim=1, frame=['phoebe'], context='scattering:henyey2'),
         dict(qualifier='asymmetry', description='Scattering asymmetry (negative = backwards, positive = forwards)', cast_type=float, repr='%f',adjust=False,value=0.0,frame=['phoebe'], context='scattering:hapke'),
         dict(qualifier='hot_spot_ampl', description='Amplitude of the hot spot', cast_type=float, repr='%f',adjust=False,value=0.0,frame=['phoebe'], context='scattering:hapke'),
         dict(qualifier='hot_spot_width', description='Width of the hot spot', cast_type=float, repr='%f',adjust=False,value=0.0,frame=['phoebe'], context='scattering:hapke'),
         ]

#    Accretion disk contexts        
defs += [dict(qualifier='dmdt',     description='Mass transfer rate',repr='%f',cast_type=float,value=1e-4,unit='Msol/yr',frame=["phoebe"],context='accretion_disk'),
         dict(qualifier='mass',     description='Host star mass',repr='%f',cast_type=float,value=1.,unit='Msol',frame=["phoebe"],context='accretion_disk'),
         dict(qualifier='rin',      description='Inner radius of disk',repr='%f',cast_type=float,value=1.,unit='Rsol',frame=["phoebe"],context='accretion_disk'),
         dict(qualifier='rout',     description='Outer radius of disk',repr='%f',cast_type=float,value=20.,unit='Rsol',frame=["phoebe"],context='accretion_disk'),
         dict(qualifier='height',   description='height of disk',repr='%f',cast_type=float,value=1e-2,unit='Rsol',frame=["phoebe"],context='accretion_disk'),
         dict(qualifier='b',        description='Host star rotation parameter',repr='%f',cast_type=float,value=1.,llim=0,ulim=1,frame=["phoebe"],context='accretion_disk'),
         #dict(qualifier='distance', description='Distance to the disk',repr='%f',cast_type=float,value=10.,adjust=False,unit='pc',frame=['phoebe'],context='accretion_disk'),
        ]

# Analytical binary model
defs += [dict(qualifier='flux_cont', description='Continuum flux level', repr='%f', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='flux_night', description='Night side flux level', repr='%f', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='alb_geom', description='Geometrical albedo', repr='%f', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='mass1', description='Primary mass', repr='%f', unit='Msol', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='mass2', description='Secondary mass', repr='%f', unit='Msol', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='teff1', description='Primary effective temperature', repr='%f', unit='K', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='teff2', description='Secondary effective temperature', repr='%f', unit='K', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='abun1', description='Primary metallicity', repr='%f', cast_type=float, value=0.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='abun2', description='Secondary metallicity', repr='%f', cast_type=float, value=0.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='radius1', description='Primary radius', unit='Rsol', repr='%f', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='radius2_rel', description='Relative secondary radius (units of radius1)', repr='%f', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='alpha_d1', description='Doppler boosting', repr='%s', cast_type='return_self', value='kurucz', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='ld_linear1', description='Linear limb darkening parameter', repr='%s', cast_type='return_self', value='kurucz', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='ld_func1', description='Primary limb darkening function', repr='%s', cast_type=str, value='linear', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='ld_func2', description='Secondary limb darkening function', repr='%s', cast_type=str, value='linear', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='gravb1', description='Gravity brightening parameter', repr='%s', cast_type='return_self', value='kurucz', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='c01', description='Primary first LD coefficient', repr='%s', cast_type='return_self', value='kurucz', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='c11', description='Primary second LD coefficient', repr='%s', cast_type='return_self', value='kurucz', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='c21', description='Primary third LD coefficient', repr='%s', cast_type='return_self', value='kurucz', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='c31', description='Primary fourth LD coefficient', repr='%s', cast_type='return_self', value='kurucz', frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='period', description='Orbital period', repr='%f', cast_type=float, unit='d', value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='incl', description='Orbital inclination', repr='%f', cast_type=float, unit='deg', value=1.0, frame=['phoebe'], context='analytical:binary'),
         dict(qualifier='sma_rel', description='Relative semi-major axis (units of radius1)', repr='%f', cast_type=float, value=1.0, frame=['phoebe'], context='analytical:binary'),
        ]

#    Fitting contexts       
defs += [dict(qualifier='iters',     description='Number of iterations',repr='%d',cast_type=int,value=1000,frame=["phoebe"],context='fitting:pymc'),
         dict(qualifier='burn',     description='Burn parameter',repr='%d',cast_type=int,value=0,frame=["phoebe"],context='fitting:pymc'),
         dict(qualifier='thin',     description='Thinning parameter',repr='%d',cast_type=int,value=1,frame=["phoebe"],context='fitting:pymc'),
         dict(qualifier='incremental',description='Add results to previously computed chain file',repr='',cast_type='make_bool',value=False,frame=['phoebe'],context='fitting:pymc'),
         dict(qualifier='label',    description='Fit run name',repr='%s',cast_type='make_label',value='',frame=["phoebe"],context='fitting:pymc'),
        ]

defs += [dict(qualifier='iters',    description='Number of iterations',repr='%d',cast_type=int,value=1000,frame=["phoebe"],context='fitting:emcee'),
         dict(qualifier='burn',     description='Burn-in parameter',repr='%d',cast_type=int,value=0,frame=["phoebe"],context='fitting:emcee'),
         dict(qualifier='thin',     description='Thinning parameter',repr='%d',cast_type=int,value=1,frame=["phoebe"],context='fitting:emcee'),
         dict(qualifier='walkers',  description='Number of walkers',
                                    long_description=("Walkers are the members of the ensemble. They are "
                                                      "almost like separate Metropolis-Hastings chains but "
                                                      "the proposal distribution for a given walker depends "
                                                      "on the positions of all the other walkers in the "
                                                      "ensemble. See Goodman & Weare (2010) for more details. "
                                                      "The best technique seems to be to start in a small ball "
                                                      "around the a priori preferred position. Don't worry, "
                                                      "the walkers quickly branch out and explore the rest of the space."),repr='%d',cast_type='require_even',value=6,frame=["phoebe"],context='fitting:emcee'),
         dict(qualifier='init_from',     description='Initialize walkers from priors, posteriors or previous run', cast_type='choose', choices=['prior', 'posterior', 'previous_run'], value='prior', repr='%s', frame=["phoebe"],context='fitting:emcee'),
         dict(qualifier='incremental',description='Add results to previously computed chain file',repr='',cast_type='make_bool',value=False,frame=['phoebe'],context='fitting:emcee'),
         dict(qualifier='acc_frac',description='Acceptance fraction',repr='%f',cast_type=float,value=0.0,frame=['phoebe'],context='fitting:emcee'),
         dict(qualifier='label',    description='Fit run name',repr='%s',cast_type='make_label',value='',frame=["phoebe"],context='fitting:emcee'),
         dict(qualifier='computelabel', description='Label of the compute params to use',repr='%s',cast_type=str,value='preview',frame=["phoebe"],context='fitting:emcee'),
         dict(qualifier='mpilabel', description='Label of the MPI params to use or blank for None (trumps those in the compute PS)',repr='%s',cast_type=str,value='None',frame=["phoebe"],context='fitting:emcee'),
        ]
        
defs += [dict(qualifier='method',    description='Nonlinear fitting method',repr='%s',cast_type='choose',value='leastsq',choices=['leastsq','nelder','lbfgsb','anneal','powell','cg','newton','cobyla','slsqp'],frame=["phoebe"],context='fitting:lmfit'),
         dict(qualifier='iters',     description='Number of iterations',long_description='If iters=n, then n number of iterations will be done. With `init_from="prior"`, you can randomize the starting point for the fit.',repr='%d',cast_type=int,value=1,frame=["phoebe"],context=['fitting:lmfit','fitting:lmfit:nelder', 'fitting:lmfit:leastsq']),
         dict(qualifier='init_from', description='Randomly draw the initial position from the priors or keep system parameters', repr='%s',cast_type='choose', choices=['system','prior','posterior'], value='system', frame=['phoebe'], context=['fitting:lmfit','fitting:lmfit:nelder', 'fitting:lmfit:leastsq']),
         dict(qualifier='label',     description='Fit run name',repr='%s',cast_type='make_label',value='',frame=["phoebe"],context=['fitting:lmfit','fitting:lmfit:nelder', 'fitting:lmfit:leastsq']),
         dict(qualifier='computelabel', description='Label of the compute params to use',repr='%s',cast_type=str,value='preview',frame=["phoebe"],context=['fitting:lmfit','fitting:lmfit:nelder', 'fitting:lmfit:leastsq']),
         dict(qualifier='mpilabel', description='Label of the MPI params to use or blank for None (trumps those in compute PS)',repr='%s',cast_type=str,value='None',frame=["phoebe"],context=['fitting:lmfit', 'fitting:lmfit:nelder', 'fitting:lmfit:leastsq']),
         dict(qualifier='compute_ci',description='Compute detailed confidence intervals',long_description="The F-test is used to compare the null model, which is the best fit we have found, with an alternate model, where one of the parameters is fixed to a specific value. The value is changed until the difference between chi2_start and chi2_final can't be explained by the loss of a degree of freedom within a certain confidence.",repr='',cast_type='make_bool',value=False,frame=["phoebe"],context=['fitting:lmfit','fitting:lmfit:nelder', 'fitting:lmfit:leastsq']),
         dict(qualifier='bounded',   description='Include boundaries in fit',
                                     long_description=('This float allows you to constrain the fit '
                                                       'parameters to a certain interval. The interval '
                                                       'is given by the "get_limits" method of the '
                                                       'Distribution class (i.e. the prior). Setting this '
                                                       'parameter to 3.5 will expand the prior range 3.5 times. '
                                                       'The MINPACK-1 implementation '
                                                       'used in scipy.optimize.leastsq for the Levenberg-Marquardt '
                                                       'algorithm does not explicitly support bounds on parameters, '
                                                       'and expects to be able to fully explore the available range '
                                                       'of values for any Parameter. Simply placing hard constraints '
                                                       '(that is, resetting the value when it exceeds the desired '
                                                       'bounds) prevents the algorithm from determining the partial '
                                                       'derivatives, and leads to unstable results. '
                                                       'Instead of placing such hard constraints, bounded parameters '
                                                       'are mathematically transformed using the formulation devised '
                                                       '(and documented) for MINUIT. This is implemented following '
                                                       '(and borrowing heavily from) the leastsqbound from J. J. '
                                                       'Helmus. Parameter values are mapped from internally used, '
                                                       'freely variable values P_internal to bounded parameters '
                                                       'P_bounded. Tests show that uncertainties estimated for '
                                                       'bounded parameters are quite reasonable. Of course, if '
                                                       'the best fit value is very close to a boundary, the '
                                                       'derivative estimated uncertainty and correlations for that '
                                                       'parameter may not be reliable. The MINUIT documentation '
                                                       'recommends caution in using bounds. Setting bounds can certainly '
                                                       'increase the number of function evaluations (and so '
                                                       'computation time), and in some cases may cause some '
                                                       'instabilities, as the range of acceptable parameter values '
                                                       'is not fully explored. On the other hand, prelminary tests '
                                                       'suggest that using max and min to set clearly outlandish '
                                                       'bounds does not greatly affect performance or results.'),repr='',cast_type=float,value=True,frame=["phoebe"],context=['fitting:lmfit','fitting:lmfit:nelder', 'fitting:lmfit:leastsq']),
        dict(qualifier='xtol',     description='Relative error in parameter values acceptable for convergence',repr='%f',cast_type=float,value=0.0001,frame=["phoebe"],context=['fitting:lmfit:nelder']),
        dict(qualifier='ftol',     description='Relative error in model acceptable for convergence',repr='%f',cast_type=float,value=0.0001,frame=["phoebe"],context=['fitting:lmfit:nelder']),
        dict(qualifier='maxfun',     description='Maximum number of function evaluations to make',repr='%s',cast_type='return_none_or_float',value=None,frame=["phoebe"],context=['fitting:lmfit:nelder']),
        dict(qualifier='xtol',     description='Relative error desired in the approximate solution',repr='%f',cast_type=float,value=1.49012e-8,frame=["phoebe"],context=['fitting:lmfit:leastsq']),
        dict(qualifier='ftol',     description='Relative error desired in the sum of squares',repr='%f',cast_type=float,value=1.49012e-8,frame=["phoebe"],context=['fitting:lmfit:leastsq']),
        dict(qualifier='gtol',     description='Orthogonality desired between the function vector and the columns of the Jacobian',repr='%f',cast_type=float,value=0.0,frame=["phoebe"],context=['fitting:lmfit:leastsq']),
        dict(qualifier='epsfcn',     description='Step length for the numerical estimation of the Jacobian',
                                long_description=("A suitable step length for the forward-difference approximation"
                                                  " of the Jacobian. If epsfcn is less than the machine precision, "
                                                  "it is assumed that the relative errors in the function are of the"
                                                  " order of the machine precision."), repr='%f',cast_type=float,value=0.001,frame=["phoebe"],context=['fitting:lmfit:leastsq']),
        dict(qualifier='maxfev',     description='Maximum number of function evaluations to make',
                                long_description=("The maximum number of calls to the function, If None or zero, then"
                                                  " 100*(N+1) is the maximum where N is the number of free parameters"), repr='%s',cast_type='return_none_or_float',value=None,frame=["phoebe"],context=['fitting:lmfit:leastsq']),
        ]
        

defs += [dict(qualifier='label',     description='Fit run name',repr='%s',cast_type='make_label',value='',frame=["phoebe"],context='fitting:minuit'),
         dict(qualifier='bounded',   description='Include boundaries in fit',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='fitting:minuit'),
        ]
        
defs += [dict(qualifier='label',     description='Fit run name',repr='%s',cast_type='make_label',value='',frame=["phoebe"],context='fitting:grid'),
         dict(qualifier='sampling',  description='Number of points to sample per parameter for non-bin priors',repr='%d',cast_type=int,value=5,frame=["phoebe"],context='fitting:grid'),
         dict(qualifier='iterate',   description='Type of iteration: list or product of priors',repr='%s',cast_type='choose',
                                     long_description=("Determines the type of iterations. Suppose the prior on parameter 'a' is [0,1,2] and "
                                                       "on parameter 'b' it is [7,8,9]. Then, if iterate=list, the fitting routines will "
                                                       "run over 3 grid points with coordinates (a,b) in [(0,7), (1,8), (2,9)]. If iterate=product, "
                                                       "all combinations will be tried out, that is (a,b) is one of [(0,7), (0,8), (0,9), (1,7), "
                                                       "(1,8), (1,9), (2,7), (2,8), (2,9)]."),
                                     choices=['product','list'],value='product',frame=['phoebe'],context='fitting:grid'),
        ]

#    MPI and computation context
defs += [dict(qualifier='label',                description='label for the MPI options',repr='%s',cast_type='make_label',value='default_mpi',frame=["phoebe"],context=['mpi','mpi:torque']),
         dict(qualifier='np',       description='Number of nodes',repr='%d',cast_type=int,value=4,frame=["phoebe"],context='mpi'),
         dict(qualifier='hostfile',     description='hostfile',repr='%s',cast_type=str,value='',frame=["phoebe"],context='mpi'),
         dict(qualifier='byslot',     description='byslot',repr='',cast_type='make_bool',value=False,frame=["phoebe"],context='mpi'),
         dict(qualifier='python',   description='Python executable',repr='%s',cast_type=str,value='python',frame=["phoebe"],context='mpi'),
         dict(qualifier='directory', description='Directory for temporary files', cast_type=str, value='',frame=['phoebe'],context='mpi'),
        ]

defs += [dict(qualifier='np',       description='Number of nodes',repr='%d',cast_type=int,value=4,frame=["phoebe"],context='mpi:slurm'),
         dict(qualifier='hostfile',     description='hostfile',repr='%s',cast_type=str,value='',frame=["phoebe"],context='mpi:slurm'),
         dict(qualifier='byslot',     description='byslot',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='mpi:slurm'),
         dict(qualifier='python',   description='Python executable',repr='%s',cast_type=str,value='python',frame=["phoebe"],context='mpi:slurm'),
         dict(qualifier='time', description='Maximum time of one process', cast_type=float, value=0, unit='min',frame=['phoebe'],context='mpi:slurm'),
         dict(qualifier='memory', description='Maximum amount of memory', cast_type=float, value=0, unit='MB', frame=['phoebe'],context='mpi:slurm'),
         dict(qualifier='partition', description='SLURM partition to commit job to', cast_type=str, value='',frame=['phoebe'],context='mpi:slurm'),
         dict(qualifier='directory', description='Directory for temporary files', cast_type=str, value='',frame=['phoebe'],context='mpi:slurm'),
        ]

defs += [dict(qualifier='nodes',       description='Node specification string',repr='%d',cast_type=str,value='48:big',frame=["phoebe"],context='mpi:torque'),
         dict(qualifier='jobname',       description='Jobname',repr='%s',cast_type=str,value='Phoebe2',frame=["phoebe"],context='mpi:torque'),
         dict(qualifier='time', description='Maximum time of one process', cast_type=float, value=60, unit='min',frame=['phoebe'],context='mpi:torque'),
         dict(qualifier='memory', description='Maximum amount of memory', cast_type=float, value=0, unit='MB', frame=['phoebe'],context='mpi:torque'),
         dict(qualifier='email', description='Email to which job alerts should be sent', cast_type=str, value='', frame=['phoebe'],context='mpi:torque'),
         dict(qualifier='alerts', description='Job alerts to be mailed (b)egin, (e)nd, (a)bort (or (n)one)', choices=['n', 'b', 'e', 'a', 'be', 'ba', 'ea', 'bea'],\
              cast_type='choose', value='a', frame=['phoebe'], context='mpi:torque'),
         dict(qualifier='python',   description='Python executable',repr='%s',cast_type=str,value='python',frame=["phoebe"],context='mpi:torque'),
         dict(qualifier='mpirun',   description='Mpirun executable',repr='%s',cast_type=str,value='mpirun',frame=["phoebe"],context='mpi:torque'),
         
         ]
        
#    Plotting context
defs += [dict(qualifier='ref',        description='identifier for the axes options',repr='%s',cast_type=str,value='',frame=["phoebe"],context='plotting:figure'),
         dict(qualifier='axesrefs',    description='list of axes refs to plot on the figure',repr='%s',cast_type=list,value=[],frame=["phoebe"],context='plotting:figure'),
         dict(qualifier='axeslocs',    description='list of locations (one for each in axes_refs, tuple or string) to plot the figure',repr='%s',cast_type=list,value=[],frame=["phoebe"],context='plotting:figure'),
         dict(qualifier='axessharex',        description="list of refs of other axes already existing in the figure", repr='%s',cast_type=list,value=[],frame=["phoebe"],context='plotting:figure'),
         dict(qualifier='axessharey',        description="list of refs of other axes already existing in the figure", repr='%s',cast_type=list,value=[],frame=["phoebe"],context='plotting:figure'),

        ]

defs += [dict(qualifier='ref',        description='identifier for the axes options',repr='%s',cast_type=str,value='',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='plotrefs',    description='list of axes refs to plot on the figure',repr='%s',cast_type=list,value=[],frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='location',     description="location in the figure for this axis", repr='%s',cast_type=str,value='auto',frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='active',        description="whether to draw this axes", repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='category',     description='what type of plot is this',choices=['lc','rv','sp','etv'],cast_type='choose',value='lc',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='phased',       description="objref of the orbit to use for phasing, or False", repr='%s', cast_type=str, value='False',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='title',        description="title of the plot", repr='%s',cast_type=str,value='_auto_',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='xunit',        description="unit to plot on the xaxis", repr='%s',cast_type=str,value='_auto_',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='yunit',        description="unit to plot on the yaxis", repr='%s',cast_type=str,value='_auto_',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='xlabel',        description="label on the xaxis", repr='%s',cast_type=str,value='_auto_',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='ylabel',        description="label on the yaxis", repr='%s',cast_type=str,value='_auto_',frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='xlim',        description="limits on the xaxis", repr='%s',cast_type='return_string_or_list',value=(None,None),frame=["phoebe"],context='plotting:axes'),
         dict(qualifier='ylim',        description="limits on the yaxis", repr='%s',cast_type='return_string_or_list',value=(None,None),frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='xticks',        description="list of floats to draw xticks", repr='%s',cast_type=list,value=['_auto_'],frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='yticks',        description="list of floats to draw yticks", repr='%s',cast_type=list,value=['_auto_'],frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='xticklabels',        description="list of strings for labels on xticks", repr='%s',cast_type=list,value=['_auto_'],frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='yticklabels',        description="list of strings for labels on yticks", repr='%s',cast_type=list,value=['_auto_'],frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='sharex',        description="ref of another axes already existing in the figure", repr='%s',cast_type=str,value='_auto_',frame=["phoebe"],context='plotting:axes'),
         #~ dict(qualifier='sharey',        description="ref of another axes already existing in the figure", repr='%s',cast_type=str,value='_auto_',frame=["phoebe"],context='plotting:axes'),
        ]
        
defs += [dict(qualifier='ref',        description='identifier for the plotting options',repr='%s',cast_type=str,value='',frame=["phoebe"],context='plotting:plot'),
         dict(qualifier='func',      description='the processing function (either a marshaled function or the name of a function in phoebe.frontend.plotting)',repr='%s',cast_type=str,value='',frame=["phoebe"],context=["plotting:plot"]),
        ]

# Server context
defs += [dict(qualifier='label', description='label for the server',repr='%s',cast_type='make_label',value='',frame=["phoebe"],context='server'),
         dict(qualifier='username', description='(optional) username for the server (ssh [-i identity_file] [username@]host)',repr='%s',cast_type=str,value='',frame=["phoebe"],context='server'),
         dict(qualifier='host', description='hostname for the server (ssh [-i identity_file] [username@]host), or None if local',repr='%s',cast_type=str,value='',frame=["phoebe"],context='server'),
         dict(qualifier='identity_file', description='(optional) identity file for the server (ssh [-i identity_file] [username@]host)',repr='%s',cast_type=str,value='',frame=["phoebe"],context='server'),
         dict(qualifier='server_dir', description='location on server to copy files and run script',repr='%s',cast_type=str,value='',frame=["phoebe"],context='server'),
         dict(qualifier='server_script', description='location on the server of a script to run (ie. to setup a virtual environment) before running phoebe',repr='%s',cast_type=str,value='',frame=["phoebe"],context='server'),
         dict(qualifier='mount_dir', description='local mounted directory to host:server_dir',repr='%s',cast_type=str,value='',frame=["phoebe"],context='server'),
         ]
         
# Logger context
defs += [dict(qualifier='label', description='label',repr='%s',cast_type='make_label',value='default_logger',frame=["phoebe"],context='logger'),
         dict(qualifier='style',    description='logger style',repr='%s',cast_type='choose',value='default',choices=['default','grandpa','minimal','trace'],frame=["phoebe"],context='logger'),
         dict(qualifier='clevel',   description='print to consolve this level and above',repr='%s',cast_type='choose',value='WARNING',choices=['INFO','DEBUG','WARNING'],frame=["phoebe"],context='logger'),
         dict(qualifier='flevel',   description='print to file this level and above',repr='%s',cast_type='choose',value='DEBUG',choices=['INFO','DEBUG','WARNING'],frame=["phoebe"],context='logger'),
         dict(qualifier='filename', description='log file to print messages with level flevel and above',repr='%s',cast_type=str,value='None',frame=["phoebe"],context='logger'),
         dict(qualifier='filemode', description='mode to open log file',repr='%s',cast_type='choose',value='w',choices=['a','w'],frame=["phoebe"],context='logger'),
         ]

# GUI context
defs += [dict(qualifier='label', description='label',repr='%s',cast_type='make_label',value='default_gui',frame=["phoebe"],context='gui'),
         dict(qualifier='panel_system', description='show system panel on startup',repr='',cast_type='make_bool',value=False,frame=["phoebe"],context='gui'),
         dict(qualifier='panel_params', description='show parameters panel on startup',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='gui'),
         dict(qualifier='panel_fitting', description='show fitting panel on startup',repr='',cast_type='make_bool',value=False,frame=["phoebe"],context='gui'),
         dict(qualifier='panel_versions', description='show versions panel on startup',repr='',cast_type='make_bool',value=False,frame=["phoebe"],context='gui'),
         dict(qualifier='panel_datasets', description='show datasets/plotting panel on startup',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='gui'),
         dict(qualifier='panel_python', description='show python console panel on startup',repr='',cast_type='make_bool',value=False,frame=["phoebe"],context='gui'),
         dict(qualifier='pyinterp_tutsys', description='show system messages in the console',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='gui'),
         dict(qualifier='pyinterp_tutplots', description='show plotting messages in the console',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='gui'),
         dict(qualifier='pyinterp_tutsettings', description='show settings messages in the console',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='gui'),
         dict(qualifier='pyinterp_thread_on', description='use threading in the python console',repr='',cast_type='make_bool',value=True,frame=["phoebe"],context='gui'),
         dict(qualifier='pyinterp_startup_custom', description='custom startup script to run on gui load',repr='%s',cast_type=str,value='import numpy as np',frame=["phoebe"],context='gui'),
         ]
         
         
# Compute context
defs += [dict(qualifier='label',                description='label for the compute options',repr='%s',cast_type='make_label',value='compute',frame=["phoebe"],context='compute'),
         dict(qualifier='time',                 description='Compute observables of system at these times',repr='%s',value='auto',frame=["phoebe"],cast_type='return_string_or_list',context='compute'),
         dict(qualifier='refs',                 description='Compute observables of system at these times',repr='%s',value='auto',frame=["phoebe"],cast_type='return_string_or_list',context='compute'),
         dict(qualifier='types',                description='Compute observables of system at these times',repr='%s',value='auto',frame=["phoebe"],cast_type='return_string_or_list',context='compute'),
         dict(qualifier='samprate',             description='Compute observables of system with these sampling rates',repr='%s',value='auto',frame=["phoebe"],cast_type='return_string_or_list',context='compute'),
         dict(qualifier='mesh_rescale',         description='Scaling factor for mesh densities (<1 decrease, >1 increase)',repr='%s',cast_type=float,value=1.0,frame=['phoebe'],context='compute'),
         dict(qualifier='heating',              description='Allow irradiators to heat other Bodies',repr='',cast_type='make_bool',value=True,frame=['phoebe'],context='compute'),
         dict(qualifier='refl',                 description='Allow irradiated Bodies to reflect light',repr='',cast_type='make_bool',value=True,frame=['phoebe'],context='compute'),
         dict(qualifier='refl_num',             description='Number of reflections',repr='%d',cast_type=int,value=1,frame=['phoebe'],context='compute'),
         dict(qualifier='ltt',                  description='Correct for light time travel effects',repr='',cast_type='make_bool',value=False,frame=['phoebe'],context='compute'),
         dict(qualifier='subdiv_alg',           description='Subdivision algorithm',repr='%s',cast_type='choose',value='edge',choices=['edge'],frame=["phoebe"],context='compute'),
         dict(qualifier='subdiv_num',           description='Number of subdivisions',repr='%d',cast_type=int,value=3,frame=["phoebe"],context='compute'),
         dict(qualifier='eclipse_alg',          description='Type of eclipse algorithm',choices=['auto','full','convex','only_horizon','binary','graham','none'],
                                                 long_description=("'only_horizon': labels the triangles wich a surface normal directed away from the observer as visible, otherwise they are labeled as hidden // "
                                                                   "'convex': uses QHull in conjunction with Delaunay triangulation to detect eclipsed triangles and label the ones which are partially visible // "
                                                                   "'graham': uses Graham scan in conjunction with binary search trees to detect eclipsed triangles and label the ones which are partially visible //"
                                                                   "'binary': uses only_horizon outside of predicted eclipses, graham inside eclipses"
                                                                   "'auto': let (a) God decide which algorithm to use. I have no idea what it does."),cast_type='choose',value='graham',frame=['phoebe'],context='compute'),
         dict(qualifier='boosting_alg',          description='Type of boosting algorithm',choices=['none','simple','local','full'],
                                                 long_description=("'none': no boosting correction // "
                                                                   "'simple': global increase/decrease of projected intensity according to mean stellar parameters // "
                                                                   "'local': local increase/decrease of projected intensity according to local stellar parameters //"
                                                                   "'full': adjust local intensity and limb darkening coefficients according to local stellar parameters"),
                                                 cast_type='choose',value='none',frame=['phoebe'],context='compute'),
         dict(qualifier='irradiation_alg',          description='Type of irradiation algorithm',choices=['full', 'point_source'],
                                                 long_description=("'full': complete irradiation calculation"
                                                                   "'point_source':  approximate irradiator as point source"),
                                                                   cast_type='choose',value='point_source',frame=['phoebe'],context='compute'),
         
        dict(qualifier='mpilabel', description='Label of the MPI params to use or blank for None',repr='%s',cast_type=str,value='None',frame=["phoebe"],context='compute'),
        ] 

# Globals context
defs += [dict(qualifier='ra', description='Right ascension', repr='%s', value=0.0, llim=-np.inf, ulim=np.inf, unit='deg', cast_type='return_equatorial_ra', frame=['phoebe'], context=['position']),
         dict(qualifier='dec', description='Declination', repr='%s', value=0.0, llim=-np.inf, ulim=np.inf, unit='deg', cast_type='return_equatorial_dec', frame=['phoebe'], context=['position']),
         dict(qualifier='epoch', description='Epoch of coordinates', repr='%s', value='J2000', cast_type=str, frame=['phoebe'], context=['position']),
         dict(qualifier='pmra', description='Proper motion in right ascension', repr='%s', value=0.0, unit='mas/yr', cast_type=float, frame=['phoebe'], context=['position']),
         dict(qualifier='pmdec', description='Proper motion in declination', repr='%s', value=0.0, unit='mas/yr', cast_type=float, frame=['phoebe'], context=['position']),
         dict(qualifier='distance',description='Distance to the object',repr='%f',cast_type=float,value=10.,adjust=False,unit='pc',frame=['phoebe'],context='position'),
         dict(qualifier='vgamma', description='Systemic velocity',repr='%f',llim=-1e6,ulim=1e6,step=0.1,adjust=False,cast_type=float,value=0.,unit='km/s',alias=['vga'],frame=["phoebe"],context='position'),         
        ]

        
#  /* ********************* DERIVABLE QUANTITIES ********************************* */
defs += [dict(qualifier='tdyn',   description='Dynamical timescale',repr='%f',cast_type=float,value=0,frame=['phoebe'],context='derived'),
         dict(qualifier='ttherm', description='Thermal timescale',  repr='%f',cast_type=float,value=0,frame=['phoebe'],context='derived'),
        ]

# "complicated" relations and constraints

#rels = [dict(qualifier='asini', description='Projected system semi-major axis', repr='%f', value=10., unit='Rsol', cast_type=float, connections=['sma', 'incl'], frame=['phoebe'], context='orbit'),
        #dict(qualifier='mass', description='Component mass', repr='%f', value=1., unit='Msol', cast_type=float, connections=['period@orbit', 'q@orbit', 'sma@orbit'], frame=['phoebe'], context='component'),
        #dict(qualifier='mass', description='Component mass', repr='%f', value=1., unit='Msol', cast_type=float, connections=[], frame=['phoebe'], context='component'),
        #dict(qualifier='teff_ratio', description='Effective temperature ratio between components', repr='%f', value=1., cast_type=float, connections=['teff@primary','teff@secondary'], frame=['phoebe'], context='binarybag')]

rels = {'binary':
            {'radius':dict(in_level_as='pot', qualifier='radius',
                           description='Component polar radius',repr='%f',
                           cast_type=float, unit='Rsol', adjust=False,
                           frame=["phoebe"], context='component'),
             'mass':dict(in_level_as='pot', qualifier='mass',
                           description='Component dynamical mass',repr='%f',
                           cast_type=float, unit='Msol', adjust=False,
                           frame=["phoebe"], context='component'),
             'asini':dict(in_level_as='sma', qualifier='asini',
                          description='Projected system semi-major axis',
                          repr='%f', cast_type=float, unit='Rsol', adjust=False,
                          frame=["phoebe"], context='orbit'),
             'vsini':dict(in_level_as='pot', qualifier='vsini',
                          description='Component projected equatorial velocity',
                          repr='%f', cast_type=float, unit='km/s', adjust=False,
                          frame=["phoebe"], context='component'),
             'logg':dict(in_level_as='pot', qualifier='logg',
                          description='Component logarithmic surface gravity',
                          repr='%f', cast_type=float, unit='[cm/s2]', adjust=False,
                          frame=["phoebe"], context='component'),
             'teffratio':dict(in_level_as='__system__', qualifier='teffratio',
                          description='Effective temperature ratio (secondary/primary)',
                          repr='%f', cast_type=float, adjust=False,
                          frame=["phoebe"], context='extra')},
        }


# simple constraints
constraints = {'phoebe':{}}        
constraints['phoebe']['orbit'] = ['{sma1} = {sma} / (1.0 + 1.0/{q})',
                         '{sma2} = {sma} / (1.0 + {q})',
                         '{totalmass} = 4*pi**2 * {sma}**3 / {period}**2 / constants.GG',
                         '{mass1} = 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + {q})',
                         '{mass2} = 4*pi**2 * {sma}**3 / {period}**2 / constants.GG / (1.0 + 1.0/{q})',
                         '{asini} = {sma} * sin({incl})',
                         '{com} = {q}/(1.0+{q})*{sma}',
                         '{q1} = {q}',
                         '{q2} = 1.0/{q}',
                         #'{circum} = 4*{sma1}*special.ellipk({ecc})/{period}',
                         ]
constraints['phoebe']['star'] = ['{surfgrav} = constants.GG*{mass}/{radius}**2',
#                                 '{angdiam} = 2*{radius}/{distance}',
                        ]

