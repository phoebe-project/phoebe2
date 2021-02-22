import numpy as np
import phoebe as phb
import os.path
import sys
import logging

try:
    import phoebe_legacy as phb1
except ImportError:
    try:
        import phoebeBackend as phb1
    except ImportError:
        _use_phb1 = False
    else:
        _use_phb1 = True
else:
    _use_phb1 = True


from io import IOBase as _IOBase

from phoebe import conf
from phoebe import list_passbands as _list_passbands
from phoebe.distortions import roche
# from phoebe.constraints.builtin import t0_ref_to_supconj

import libphoebe
logger = logging.getLogger("IO")
logger.addHandler(logging.NullHandler())

_skip_filter_checks = {'check_default': False, 'check_visible': False}

_default_passband_map_1to2 = {'TESS:default': 'TESS:T', 'Tycho:BT': 'Tycho:B', 'Tycho:VT': 'Tycho:V'}
_default_passband_map_2to1 = {v:k for k,v in _default_passband_map_1to2.items()}


def _is_file(obj):
    return isinstance(obj, _IOBase) or obj.__class__.__name__ in ['FileStorage']

"""
Dictionaries of parameters for conversion between phoebe1 and phoebe 2

"""

_1to2par = {'ld_model':'ld_func',
            'bol':'ld_coeffs_bol',
            'rvcoff': 'ld_coeffs',
            'lccoff':'ld_coeffs',
            'active': 'enabled',
#            'model': 'morphology',
            'filter': 'passband',
            'hjd0':'t0_ref',
            'period': 'period',  # sidereal
            'dpdt': 'dpdt',
            'sma':'sma',
            'rm': 'q',
            'incl': 'incl',
            'pot':'requiv',
            'met':'abun',
            'f': 'syncpar',
            'alb': 'irrad_frac_refl_bol',
            'grb':'gravb_bol',
            'ecc': 'ecc',
            'perr0':'per0',
            'dperdt': 'dperdt',
            'hla': 'pblum',
            'cla': 'pblum',
            'el3': 'l3',
            'el3frac':'l3_frac',
            'el3_units':'l3_mode',
            'reflections':'refl_num',
            'finesize': 'gridsize',
            'vga': 'vgamma',
            'teff':'teff',
            'msc1':'msc1',
            'msc2':'msc2',
            'ie':'ie',
            'atm': 'atm',
            'flux':'fluxes',
            'vel':'rvs',
            'sigmarv':'sigmas',
            'sigmalc':'sigmas',
            'time':'times',
            'longitude':'long',
            'radius': 'radius',
            'tempfactor':'relteff',
            'colatitude':'colat',
            'cadence': 'exptime',
            }
#            'rate':'rate'}

#TODO: add back proximity_rv maybe?
#TODO: add back 'excess': 'extinction',

_2to1par = {v:k for k,v in _1to2par.items()}

_units1 = {'incl': 'deg',
           'period': 'd',
           'dpdt': 'd/d',
           'sma': 'Rsun',
           'vga':'km/s',
           'teff': 'K',
           'perr0': 'rad',
           'dperdt': 'rad/d', # could be deg/d depending on config setting in legacy
           'flux':'W/m2',
           'sigmalc': 'W/m2',
           'sigmarv': 'km/s',
           'vel':'km/s',
           'time':'d',
           'exptime':'s'}

_parsect = {'t0':'component',
            'period':'component',
            'dpdt':'component',
            'pshift':'component',
            'sma':'component',
            'rm': 'component',
            'incl':'component',
            'perr0':'component',
            'dperdt':'component',
            'hla': 'component',
            'cla':'component',
       #     'el3':'component',
            'reffect':'compute',
            'reflections':'compute',
            'finegrid':'mesh',
            'vga':'system',
            'msc1_switch': 'compute',
            'msc2_switch': 'compute',
            'ie_switch':'compute',
            'proximity_rv1':'compute',
            'proximity_rv2': 'compute'}


#_bool1to2 = {1:True, 0:False}

_bool2to1 = {True:1, False:0}

_bool1to2 = {1:True, 0:False}

"""
ld_legacy -

"""

def ld_to_phoebe(pn, d, rvdep=None, dataid=None, law=None):
    if 'bol' in pn:
        d['context'] = 'component'
        pnew = 'bol'
    elif 'rv' in pn:
        d['context'] = 'dataset'
        d['dataset'] = dataid
        pnew = 'rvcoff'
    elif 'lc' in pn:
        d['context'] = 'dataset'
        d['dataset'] = dataid
        pnew = 'lccoff'
    else:
        pnew = 'ld_model'
    if 'x' in pn:
        d['index'] = 0
    elif 'y' in pn and law != 'Linear cosine law':
        d['index'] = 1

    return [pnew, d]
"""
ret_dict - finds the phoebe 2 twig associated with a given parameter

pname:  name of the parameter

dataid: name of the dataset

rvdep = determines whether rv belongs to primary or secondary

"""


def ret_dict(pname, val, dataid=None, rvdep=None, comid=None):
#    pname = pname.split('[')[0]
    pieces = pname.split('_')
    pnew = pieces[-1]
    d = {}
    #on the very rare occasion phoebe 1 has a separate unit parameters
    if pnew == 'units':
        pnew = pieces[-2]+'_'+pieces[-1]

    if pnew == 'switch':
       pnew = pieces[1]
       if pnew  == 'proximity':
           pnew = pnew+'_'+pieces[2]
           d['context'] = 'compute'
# component specific parameters end with 1 or 2 in phoebe1
    if pnew[-1] == '1':
        d['component'] = 'primary'
#        d.setdefault('context', 'component')
        d['context'] = 'component'
        pnew = pnew[:-1]

    elif pnew[-1] == '2':
        d['component'] = 'secondary'
#        d.setdefault('context', 'component')
        d['context'] = 'component'
        pnew = pnew[:-1]

# even though gridsize belongs to each component it is located in the compute parameter set


    if pieces[1] == 'lc':
        d['dataset'] = dataid
        d['context'] = 'dataset'

    elif pieces[1] == 'rv':
        d['component'] = rvdep
        d['dataset'] = dataid
        d['context'] = 'dataset'

    elif pieces[1] == 'ld':
        pnew, d  = ld_to_phoebe(pnew, d, rvdep, dataid)

    elif pieces[1] == 'spots':
        d['component'] = rvdep
        d['context'] = 'feature'
        d['feature'] = dataid

    if pnew == 'hla':
        d['component'] = 'primary'
    elif pnew == 'cla':
        d['component'] = 'secondary'

# two different radius parameters, only include spots
#    print "this is pnew", pnew
    if pnew == 'radius' and pieces[1] != 'spots':
        pnew = None
        d = {}

    if pnew in _1to2par:
        try:
            d.setdefault('context', _parsect[pnew])
        except:
            try:
                d['context']
            except:
                d['context'] = None

        if pnew in ['atm', 'model', 'cindex', 'finesize', 'reffect', 'reflections']:
            d['context']  ='compute'

        if 'proximity' in pnew:
            d['context'] = 'compute'

        if d['context'] == 'component':

            d.setdefault('component', 'binary')

#        if d['context'] == 'compute':
#            d.setdefault('compute', comid)

        if pnew == 'reflections':
            d['value'] = int(val)#+1

        else:
            d['value'] = val

        d['qualifier'] = _1to2par[pnew]

        if pnew in _units1:
            d['unit'] = _units1[pnew]

    else:
        d ={}
        logger.info("Parameter "+str(pname)+" has no Phoebe 2 counterpart")


    return pnew, d

def load_lc_data(filename, indep, dep, indweight=None, mzero=None, bundle=None, dir='./'):

    """
    load dictionary with lc data
    """
    if dir is None:
        logger.warning("to load referenced data files, pass filename as string instead of file object")
        return {}

    if '/' in filename:
        path, filename = os.path.split(filename)
    else:
        # TODO: this needs to change to be directory of the .phoebe file
        path = dir

    load_file = os.path.join(path, filename)
    try:
        lcdata = np.loadtxt(load_file)
    except IOError:
        logger.warning("Could not load data file referenced at {}. Dataset will be empty.".format(load_file))
        return {}
    ncol = len(lcdata[0])

    #check if there are enough columns for errors
    if ncol >= 3:
        sigma = True
        # convert standard weight to standard deviation

        if indweight == 'Standard weight':
            err = np.sqrt(1/lcdata[:,2])
            lcdata[:,2] = err
            logger.warning('Standard weight has been converted to Standard deviation.')
    else:
        sigma = False
        logger.warning('A sigma column was mentioned in the .phoebe file but is not present in the lc data file')

    #if phase convert to time
    if indep == 'Phase':
        logger.warning("Phoebe 2 doesn't accept phases, converting to time with respect to the given ephemeris")
        times = bundle.to_time(lcdata[:,0])
        lcdata[:,0] = times

    if dep == 'Magnitude':
        mag = lcdata[:,1]
        flux = 10**(-0.4*(mag-mzero))

        if sigma == True:
            mag_err = lcdata[:,2]
            flux_err = np.abs(10**(-0.4*((mag+mag_err)-mzero)) - flux)
            lcdata[:,2] = flux_err

        lcdata[:,1] = flux
    d = {}
    d['phoebe_lc_time'] = lcdata[:,0]
    d['phoebe_lc_flux'] = lcdata[:,1]

    if sigma == True:
        d['phoebe_lc_sigmalc'] = lcdata[:,2]

    return d

def load_rv_data(filename, indep, dep, indweight=None, dir='./'):

    """
    load dictionary with rv data.
    """
    if dir is None:
        logger.warning("to load referenced data files, pass filename as string instead of file object")
        return {}


    if '/' in filename:
        path, filename = os.path.split(filename)
    else:
        path = dir

    load_file = os.path.join(path, filename)
    try:
        rvdata = np.loadtxt(load_file)
    except IOError:
        logger.warning("Could not load data file referenced at {}. Dataset will be empty.".format(load_file))
        return {}

    d ={}
    d['phoebe_rv_time'] = rvdata[:,0]
    d['phoebe_rv_vel'] = rvdata[:,1]
    ncol = len(rvdata[0])

    if indweight=="Standard deviation":

        if ncol >= 3:
            d['phoebe_rv_sigmarv'] = rvdata[:,2]
        else:
            logger.warning('A sigma column is mentioned in the .phoebe file but is not present in the rv data file')
    elif indweight =="Standard weight":
                if ncol >= 3:
                    sigma = np.sqrt(1/rvdata[:,2])
                    d['phoebe_rv_sigmarv'] = sigma
                    logger.warning('Standard weight has been converted to Standard deviation.')
    else:
        logger.warning('Phoebe 2 currently only supports standard deviaton')

    return d

def det_dataset(eb, passband, dataid, comp, time):

    """
    Since RV datasets can have values related to each component in phoebe2, but are component specific in phoebe1
    , it is important to determine which dataset to add parameters to. This function will do that.
    eb - bundle
    rvpt - relevant phoebe 1 parameters

    """
    rvs = eb.get_dataset(kind='rv', **_skip_filter_checks).datasets
    #first check to see if there are currently in RV datasets
    if dataid == 'Undefined':
        dataid = None
#    if len(rvs) == 0:
    #if there isn't we add one the easy part

    try:
        eb._check_label(dataid)

        rv_dataset = eb.add_dataset('rv', dataset=dataid, times=[], **_skip_filter_checks)

    except ValueError:

        logger.warning("The name picked for the radial velocity curve is forbidden. Applying default name instead")
        rv_dataset = eb.add_dataset('rv', times=[], **_skip_filter_checks)

#     else:
#     #now we have to determine if we add to an existing dataset or make a new one
#         rvs = eb.get_dataset(kind='rv').datasets
#         found = False
#         #set the component of the companion
#
#         if comp == 'primary':
#             comp_o = 'primary'
#         else:
#             comp_o = 'secondary'
#         for x in rvs:
#             test_dataset = eb.get_dataset(x, check_visible=False)
#
#
#             if len(test_dataset.get_value(qualifier='rvs', component=comp_o, check_visible=False)) == 0:                #so at least it has an empty spot now check against filter and length
# #               removing reference to time_o. If there are no rvs there should be no times
# #                time_o = test_dataset.get_value('times', component=comp_o)
#                 passband_o = test_dataset.get_value('passband')
#
# #                if np.all(time_o == time) and (passband == passband_o):
#                 if (passband == passband_o):
#                     rv_dataset = test_dataset
#                     found = True
#
#         if not found:
#             try:
#                 eb._check_label(dataid)
#
#                 rv_dataset = eb.add_dataset('rv', dataset=dataid, times=[])
#
#             except ValueError:
#
#                 logger.warning("The name picked for the lightcurve is forbidden. Applying default name instead")
#                 rv_dataset = eb.add_dataset('rv', times=[])

    return rv_dataset


"""
Load a phoebe legacy file complete with all the bells and whistles

filename - a .phoebe file (from phoebe 1)

"""

def load_legacy(filename, add_compute_legacy=True, add_compute_phoebe=True,
                ignore_errors=False, passband_map={}):
    conf_interactive_checks_state = conf.interactive_checks
    conf_interactive_constraints_state = conf.interactive_constraints
    conf.interactive_off(suppress_warning=True)


    if _is_file(filename):
        f = filename
        legacy_file_dir = None

    elif isinstance(filename, str):
        filename = os.path.expanduser(filename)
        legacy_file_dir = os.path.dirname(filename)

        logger.debug("importing from {}".format(filename))
        f = open(filename, 'r')

    else:
        raise TypeError("filename must be string or file object, got {}".format(type(filename)))

# load the phoebe file

#    params = np.loadtxt(filename, dtype='str', delimiter = '=')
    params = np.loadtxt(filename, dtype='str', delimiter = '=',
    converters = {0: lambda s: s.strip(), 1: lambda s: s.strip()})

    morphology = params[:,1][list(params[:,0]).index('phoebe_model')]


# load an empty legacy bundle and initialize obvious parameter sets
    if 'Overcontact' in morphology:
#        raise NotImplementedError
        contact_binary= True
        semi_detached = False
        eb = phb.Bundle.default_binary(contact_binary=True)
    elif 'Semi-detached' in morphology:
        semi_detached = True
        contact_binary = False
        eb = phb.Bundle.default_binary()
    else:
        semi_detached = False
        contact_binary = False
        eb = phb.Bundle.default_binary()
#    comid = []
#    if add_compute_phoebe == True:
    #    comid.append('phoebe01')
#        eb.add_compute('phoebe')#, compute=comid[0])
    if add_compute_legacy == True:
    #    comid.append('lega1')
        eb.add_compute('legacy')#, compute=comid[-1])


#basic filter on parameters that make no sense in phoebe 2
    ind = [list(params[:,0]).index(s) for s in params[:,0] if not ".ADJ" in s and not ".MIN" in s and not ".MAX" in s and not ".STEP" in s and not "gui_" in s]
    params = params[ind]

# determine number of lcs and rvs
    rvno = np.int(params[:,1][list(params[:,0]).index('phoebe_rvno')])
    lcno = np.int(params[:,1][list(params[:,0]).index('phoebe_lcno')])
# and spots
    spotno = np.int(params[:,1][list(params[:,0]).index('phoebe_spots_no')])

# delete parameters that have already been accounted for and find lc and rv parameters

    params = np.delete(params, [list(params[:,0]).index('phoebe_lcno'), list(params[:,0]).index('phoebe_rvno')], axis=0)


# check to see if reflection is on

    ref_effect = np.int(params[:,1][list(params[:,0]).index('phoebe_reffect_switch')])

    if ref_effect == 0:

        params[:,1][list(params[:,0]).index('phoebe_reffect_reflections')] = 0
        logger.warning('Phoebe Legacy reflection effect switch is set to false so refl_num is being set to 0.')

    if not add_compute_legacy:
        params = np.delete(params, [list(params[:,0]).index('phoebe_reffect_reflections'), list(params[:,0]).index('phoebe_ie_switch'),
                list(params[:,0]).index('phoebe_grid_finesize1'), list(params[:,0]).index('phoebe_grid_finesize2')], axis=0)

    if 'Overcontact' in morphology:
        params = np.delete(params, [list(params[:,0]).index('phoebe_pot2.VAL')], axis=0)
        if 'UMa'in morphology:
            params[:,1][list(params[:,0]).index('phoebe_teff2.VAL')] = params[:,1][list(params[:,0]).index('phoebe_teff1.VAL')]
            params[:,1][list(params[:,0]).index('phoebe_grb2.VAL')] = params[:,1][list(params[:,0]).index('phoebe_grb1.VAL')]
            params[:,1][list(params[:,0]).index('phoebe_alb2.VAL')] = params[:,1][list(params[:,0]).index('phoebe_alb1.VAL')]
    elif 'Semi-detached' in morphology:

        if "primary" in morphology:
            params = np.delete(params, [list(params[:,0]).index('phoebe_pot1.VAL')], axis=0)
        elif "secondary" in morphology:
            params = np.delete(params, [list(params[:,0]).index('phoebe_pot2.VAL')], axis=0)

    else:
        #none of the other constraints are useful
        pass

#pull out global values for fti
    try:
        fti = _bool1to2[int(params[:,1][list(params[:,0]).index('phoebe_cadence_switch')])]
        fti_exp = params[:,1][list(params[:,0]).index('phoebe_cadence')]
        fti_ovs = params[:,1][list(params[:,0]).index('phoebe_cadence_rate')]
        fti_ts = params[:,1][list(params[:,0]).index('phoebe_cadence_timestamp')].strip('"')
        params = np.delete(params, [list(params[:,0]).index('phoebe_cadence'), list(params[:,0]).index('phoebe_cadence_switch'), list(params[:,0]).index('phoebe_cadence_rate'), list(params[:,0]).index('phoebe_cadence_timestamp')], axis=0)

    except:

        fti = False
        fti_exp = '1766'
        fti_ovs = '1'
        fti_ts = 'Mid-exposure'
#    params =  np.delete(params, [list(params[:,0]).index('phoebe_cadence'), list(params[:,0]).index('phoebe_cadence_switch')], axis=0)#, list(params[:,0]).index('phoebe_cadence_rate'), list(params[:,0]).index('phoebe_cadence_timestamp')], axis=0)

#    fti_type = params[:,1][list(params[:,0]).index('phoebe_cadenc_rate')]
# create mzero and grab it if it exists
    mzero = None
    if 'phoebe_mnorm' in params:
        mzero = np.float(params[:,1][list(params[:,0]).index('phoebe_mnorm')])
# determine if luminosities are decoupled and set pblum_mode accordingly
    try:
        decoupled_luminosity = np.int(params[:,1][list(params[:,0]).index('phoebe_usecla_switch')])
    except:
        pass
#    if decoupled_luminosity == 0:
#        eb.set_value(qualifier='pblum_mode', value='component-coupled')
#    else:
#        eb.set_value(qualifier='pblum_mode', value='decoupled')

#Determin LD law

    ldlaw = params[:,1][list(params[:,0]).index('phoebe_ld_model')].strip('"')
# FORCE hla and cla to follow conventions so the parser doesn't freak out.
    for x in range(1,lcno+1):
        hlain = list(params[:,0]).index('phoebe_hla['+str(x)+'].VAL')
#        clain = list(params[:,0]).index('phoebe_cla['+str(x)+'].VAL')
        params[:,0][hlain] = 'phoebe_lc_hla1['+str(x)+'].VAL'
#        params[:,0][clain] = 'phoebe_lc_cla2['+str(x)+'].VAL'
        hla = np.float(params[:,1][hlain]) #pull for possible conversion of l3
#        cla = np.float(params[:,1][clain]) #pull for possible conversion of l3

#        if contact_binary:
#            params = np.delete(params, [list(params[:,0]).index('phoebe_lc_cla2['+str(x)+'].VAL')], axis=0)

#and split into lc and rv and spot parameters

    lcin = [list(params[:,0]).index(s) for s in params[:,0] if "lc" in s]
    rvin = [list(params[:,0]).index(s) for s in params[:,0] if "rv" in s and not "proximity" in s]
    spotin = [list(params[:,0]).index(s) for s in params[:,0] if "spots" in s]
    lcpars = params[lcin]
    rvpars = params[rvin]
    spotpars = params[spotin]
    lcin.extend(rvin)
    lcin.extend(spotin)
    params = np.delete(params, lcin, axis=0)

# grab third light unit which is not lightcurve dependent in phoebe legacy and remove from params
    l3_units = params[:,1][list(params[:,0]).index('phoebe_el3_units')].strip('"')
    params = np.delete(params, [list(params[:,0]).index('phoebe_el3_units')], axis=0)

#load orbital/stellar parameters

#we do this here because we may need it to convert phases to times
#
#    ecc = np.float(params[:,1][list(params[:,0]).index('phoebe_ecc.VAL')])
#    perr0 = np.float(params[:,1][list(params[:,0]).index('phoebe_perr0.VAL')])
#    period = np.float(params[:,1][list(params[:,0]).index('phoebe_ecc.VAL')])
#    t0_ref = np.float(params[:,1][list(params[:,0]).index('phoebe_hjd0.VAL')])

# create datasets and fill with the correct parameters
    for x in range(len(params)):

        pname = params[:,0][x]
        pname = pname.split('.')[0]

        val = params[:,1][x].strip('"')
        pnew, d = ret_dict(pname, val)

        if pnew == 'ld_model':
            ldlaws_1to2= {'Linear cosine law': 'linear', 'Logarithmic law': 'logarithmic', 'Square root law': 'square_root'}
            if val == 'Linear cosine law':
                logger.warning('Linear cosine law is not currently supported. Converting to linear instead')
            d['value'] = ldlaws_1to2[val]#val[0].lower()+val[1::]

            # since ld_coeffs is dataset specific make sure there is at least one dataset
#            if lcno != 0 or rvno != 0:
#                eb.set_value_all(check_visible=False, **d)
            #now change to take care of bolometric values
            d['qualifier'] = d['qualifier']+'_bol'
        if pnew == 'pot':

            if contact_binary:
                eb.flip_constraint('pot', component='contact_envelope', solve_for='requiv@primary', check_nan=False)
                d['component'] = 'contact_envelope'
                d['context'] = 'component'
                d['qualifier'] = 'pot'

            else:
                d['kind'] = 'star'
                d['qualifier'] = 'requiv'
                d.pop('value') #remove qualifier from dictionary to avoid conflicts in the future

                comp_no = ['', 'primary', 'secondary'].index(d['component'])

                q_in = list(params[:,0]).index('phoebe_rm.VAL')
                q = np.float(params[:,1][q_in])
                F_in = list(params[:,0]).index('phoebe_f{}.VAL'.format(comp_no))
                F = np.float(params[:,1][F_in])
                a_in = list(params[:,0]).index('phoebe_sma.VAL')
                a = np.float(params[:,1][a_in])
                e_in = list(params[:,0]).index('phoebe_ecc.VAL')
                e = np.float(params[:,1][e_in])
                delta = 1-e # defined at periastron

                d['value'] = roche.pot_to_requiv(float(val), a, q, F, delta, component=comp_no)
                d['kind'] = None

                d['context'] = 'component'

    # change t0_ref and set hjd0
        if pnew == 'hjd0':

            d.pop('qualifier') #avoiding possible conflicts
            d.pop('value') #avoiding possible conflicts
            #
            #

            eb.flip_constraint(solve_for='t0_supconj', constraint_func='t0_ref_supconj', **d)

    #        elif pnew == 'filter':
    #       make sure t0 accounts for any phase shift present in phoebe 1
            try:
                #if being reimported after phoebe2 save this parameter won't exist
                pshift_in = list(params[:,0]).index('phoebe_pshift.VAL')
                pshift = np.float(params[:,1][pshift_in])
            except:
                pshift = 0.0

            period_in = list(params[:,0]).index('phoebe_period.VAL')
            period = np.float(params[:,1][period_in])

            t0 = float(val)+pshift*period
    #       new
            d['value'] = t0
            d['qualifier'] = 't0_ref'
             # write method for this

    #        elif pnew == 'excess':
                     # requires two parameters that phoebe 1 doesn't have access to Rv and extinction
        # elif pnew == 'alb':
            # val = 1.-float(val)
            # d['value'] = val
        elif pnew == 'atm':
            val = int(val)

            if val == 0:
                d['value'] = 'extern_planckint'
            if val == 1:
                d['value'] = 'extern_atmx'
            logger.warning('If you would like to use phoebe 1 atmospheres, you must add this manually')
            if add_compute_legacy:
                d['kind'] = 'legacy'
                eb.set_value(check_visible=False, **d)

            d['kind'] = 'phoebe'
            d['value'] = 'ck2004'
#            atm_choices = eb.get_compute('detailed').get_parameter('atm', component='primary').choices
#            if d['value'] not in atm_choices:
                #TODO FIND appropriate default
#                d['value'] = 'atmcof'

        elif pnew == 'finesize':
                    # set gridsize
            d['value'] = val
            eb.set_value_all(check_visible=False, **d)
            # change parameter and value to ntriangles
            val = N_to_Ntriangles(int(np.float(val)))
            d['qualifier'] = 'ntriangles'
            d['value'] = val
#        elif pnew == 'refl_num':
        if len(d) > 0:
            try:
                eb.set_value_all(check_visible=False, **d)
            except Exception as err:
                raise Exception("could not set_value_all({}).  Original error: {}".format(d, str(err)))
    if semi_detached:
        if 'primary' in morphology:
            eb.add_constraint('semidetached', component='primary')
        elif 'secondary' in morphology:
            eb.add_constraint('semidetached', component='secondary')





    #make sure constraints have been applied
    eb.run_delayed_constraints()


# First LC
    # grab third light unit which is not lightcurve dependent in phoebe legacy and remove from params
#    l3_units = params[:,1][list(params[:,0]).index('phoebe_el3_units')].strip('"')
#    params = np.delete(params, [list(params[:,0]).index('phoebe_el3_units')], axis=0)

    for x in range(1,lcno+1):

        #list of parameters related to current dataset

        lcint = [list(lcpars[:,0]).index(s) for s in lcpars[:,0] if "["+str(x)+"]" in s]
        lcpt = lcpars[lcint]



        #determine whether individual fti parameter exists and add them if not
 #       print 'phoebe_lc_cadence_switch['+str(x)+']', int(lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_switch['+str(x)+']')])
 #       fti_ts_ind = lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_timestamp['+str(x)+']')].strip('"')

 #       print fti_ts_ind

#        if fti_ts_ind != 'Mid-exposure':
#            print "i shouldn't go here"

#            logger.warning('Phoebe 2 only uses Mid-Exposure for calculating finite exposure times.')


        try:
            fti_ind = _bool1to2[int(lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_switch['+str(x)+']')])]
            fti_ts_ind = lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_timestamp['+str(x)+']')].strip('"')

            if fti_ts_ind != 'Mid-exposure':
                logger.warning('Phoebe 2 only uses Mid-Exposure for calculating finite exposure times.')

        except:

            logger.warning('Your .phoebe file was created using a version of phoebe which does not support dataset dependent finite integration time parameters')
            fti_val = _bool2to1[fti]
            ftia = np.array(['phoebe_lc_cadence_switch['+str(x)+']', fti_val])
            fti_expa = np.array(['phoebe_lc_cadence['+str(x)+']', fti_exp])
            fti_ovsa = np.array(['phoebe_lc_cadence_rate['+str(x)+']', fti_ovs])
            #fti_tsa = np.array(['phoebe_lc_cadence_timestamp['+str(x)+']', fti_ts])
            lcpt = np.vstack((lcpt,ftia,fti_expa,fti_ovsa))#, fti_tsa))

            fti_ind = False


#        if not fti_ind:

#        if not fti_ind:

#            if fti_ts != 'Mid-exposure':
#                logger.warning('Phoebe 2 only uses Mid-Exposure times for calculating finite exposure times.')
#            if fti:
#                lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_rate['+str(x)+']')] = fti_ovs

#            else:
#                lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_rate['+str(x)+']')] = '0'


#            lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence['+str(x)+']')] = fti_exp

#            lcpt[:,1][list(lcpt[:,0]).index('phoebe_lc_cadence_rate['+str(x)+']')] = fti_ovs






#STARTS HERE
        lc_dict = {}

        for y in range(len(lcpt)):
            parameter = lcpt[y][0].split('[')[0]
            lc_dict[parameter] = lcpt[:,1][y].strip('"')

        #add third light
        l3 = np.float(params[:,1][list(params[:,0]).index('phoebe_el3['+str(x)+'].VAL')])


#        if params[:,1][list(params[:,0]).index('phoebe_el3_units')].strip('"') == 'Total light':
#            logger.warning('l3 as a percentage of total light is currently not supported in phoebe 2')
#            l3=0
#            l3 = l3/(4.0*np.pi)*(hla+cla)/(1.0-l3)
        lc_dict['phoebe_lc_el3_units'] = l3_units
        #since l3 has two possible parameters in phoebe2 adjust phoebe legacy accordingly
        if l3_units == 'Total light':
            lc_dict['phoebe_lc_el3frac'] = l3
        else:
            lc_dict['phoebe_lc_el3'] = l3

    # Determine the correct dataset to open
    # create rv data dictionary
#        indweight = lc_dict['phoebe_lc_indweight']

#        if indweight == 'Unavailable':
#            indweight = None

        #make sure filename parameter exists and if not add it

        try:
            lc_dict['phoebe_lc_filename']
        except:
            lc_dict['phoebe_lc_filename'] = 'Undefined'



        if mzero != None and lc_dict['phoebe_lc_filename'] != 'Undefined':

            indweight = lc_dict['phoebe_lc_indweight']

            if indweight == 'Unavailable':
                indweight = None

            data_dict = load_lc_data(filename=lc_dict['phoebe_lc_filename'],  indep=lc_dict['phoebe_lc_indep'], dep=lc_dict['phoebe_lc_dep'], indweight=indweight, mzero=mzero, dir=legacy_file_dir, bundle=eb)

            lc_dict.update(data_dict)

    # get dataid and load
#datain = list(lcpt[:,0]).index('phoebe_lc_id['+str(x)+']')
        dataid = lc_dict['phoebe_lc_id']
        del lc_dict['phoebe_lc_id']

        if dataid == 'Undefined':

            lc_dataset = eb.add_dataset('lc')

        else:

            try:
                eb._check_label(dataid)

                lc_dataset = eb.add_dataset('lc', dataset=dataid)

            except ValueError:

                logger.warning("The name picked for the lightcurve is forbidden. Applying default name instead")
                lc_dataset = eb.add_dataset('lc')

    # get name of new dataset

        dataid = lc_dataset.dataset

    #enable dataset
        enabled = lc_dict['phoebe_lc_active']
        del lc_dict['phoebe_lc_active']

        d ={'qualifier':'enabled', 'dataset':dataid, 'value':enabled}
        eb.set_value_all(check_visible=False, **d)

        # disable interpolating ld coefficients
        eb.set_value_all(qualifier='ld_mode', dataset=dataid, value='manual', **_skip_filter_checks)
        eb.set_value_all(qualifier='ld_mode_bol', value='manual', **_skip_filter_checks)

    #set pblum reference

        if decoupled_luminosity == 0:
            eb.set_value(qualifier='pblum_mode', dataset=dataid, value='component-coupled', **_skip_filter_checks)
        else:
            eb.set_value(qualifier='pblum_mode', dataset=dataid, value='decoupled', **_skip_filter_checks)

    #set ldlaw

        ldlaws_1to2= {'Linear cosine law': 'linear', 'Logarithmic law': 'logarithmic', 'Square root law': 'square_root'}
        if ldlaw == 'Linear cosine law':
            logger.warning('Linear cosine law is not currently supported. Converting to linear instead')

        value = ldlaws_1to2[ldlaw]#val[0].lower()+val[1::]

        # since ld_coeffs is dataset specific make sure there is at least one dataset
        #    if lcno != 0 or rvno != 0:
        d ={'qualifier':'ld_func', 'dataset':dataid, 'value':value}
        eb.set_value_all(check_visible=False, **d)



    #get available passbands

        choices = _list_passbands()

    #create parameter dictionary

    # cycle through all parameters. separate out data parameters and model parameters. add model parameters

        for k in lc_dict:

            # deal with fti
            if 'cadence' in k:

                pieces = k.split('_')

                if pieces[-1] == 'switch':
                    pnew = 'fti_method'
                    bool_fti = {0:'none', 1:'oversample'}
                    val = int(lc_dict[k])
                    value = bool_fti[val]
                    d ={'qualifier': pnew, 'dataset':dataid, 'context':'compute', 'value':value}

                if pieces[-1] == 'rate':
                    pnew = 'fti_oversample'
                    value = int(lc_dict[k])
                    d ={'qualifier': pnew, 'dataset':dataid, 'context':'compute', 'value':value}

                if pieces[-1] == 'timestamp':
                    pass
                if pieces[-1] == 'cadence':
                    pnew, d = ret_dict(k, lc_dict[k], dataid=dataid)

            else:
                pnew, d = ret_dict(k, lc_dict[k], dataid=dataid)

#            print d
        # as long as the parameter exists add it
            if len(d) > 0:

                if d['qualifier'] == 'passband':
                    d['value'] = passband_map.get(d['value'], _default_passband_map_1to2.get(d['value'], d['value']))

                    if d['value'] not in choices:
                        if ignore_errors:
                            logger.warning("no match for passband='{}', defaulting to 'Johnson:V'".format(d['value']))
                            d['value'] = 'Johnson:V'
                        else:
                            raise ValueError("no match for passband='{}'.  Pass ignore_errors to default to 'Johnson:V' (and ignore any future errors) with a warning in the logger.  Or pass passband_map as a dictionary.".format(d['value']))

                if d['qualifier'] == 'l3_mode':
                    choice_dict = {'Flux':'flux', 'Total light':'fraction'}
                    val = choice_dict[d['value']]
                    d['value'] = val

#                    d['component'] = 'contact_envelope'

                try:

                    eb.set_value_all(check_visible=False, **d)
                except ValueError as exc:
                    raise ValueError(exc.args[0] + " ({})".format(d))

#Now rvs
    rvs = eb.get_dataset(kind='rv', **_skip_filter_checks).datasets
    for x in range(1,rvno+1):

    #list of parameters related to current dataset
        rvint = [list(rvpars[:,0]).index(s) for s in rvpars[:,0] if "["+str(x)+"]" in s]
        rvpt = rvpars[rvint]

    #determine whether to use global or individual fti parameters
        try:
            fti_ind = _bool1to2[int(rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence_switch['+str(x)+']')])]
            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'), axis=0)
            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence_rate['+str(x)+']'), axis=0)
        except:
            logger.warning('finite integration time is not currently supported for RV datasets in Phoebe 2')

    #    if fti_ind:
    #        if fti:
    #            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'))
            #    rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']')] = fti_exp

    #        else:
    #            rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'))
            #    rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']')] = 0.0

    #        rvpt = np.delete(rvpt, list(rvpt[:,0]).index('phoebe_rv_cadence['+str(x)+']'))
    #        rvpt[:,1][list(rvpt[:,0]).index('phoebe_rv_cadence_rate['+str(x)+']')] = fti_ovs


#create rv dictionary
        rv_dict = {}
        for x in range(len(rvpt)):
#    parameter = rvpt[x][0].split('_')[-1].split('[')[0]
            parameter = rvpt[x][0].split('[')[0]

            rv_dict[parameter] = rvpt[:,1][x].strip('"')
    # grab some parameters we'll need

        passband = rv_dict['phoebe_rv_filter']
        time = []
        dataid = rv_dict['phoebe_rv_id']
        del rv_dict['phoebe_rv_id']

        comp = rv_dict['phoebe_rv_dep'].split(' ')[0].lower()

    # create rv data dictionary
#        indweight = rv_dict['phoebe_rv_indweight']

#        if indweight == 'Unavailable':
#            indweight = None

        try:
            rv_dict['phoebe_rv_filename']
        except:
            rv_dict['phoebe_rv_filename'] = 'Undefined'


        if rv_dict['phoebe_rv_filename'] != 'Undefined':
            indweight = rv_dict['phoebe_rv_indweight']

            if indweight == 'Unavailable':
                indweight = None

            data_dict = load_rv_data(filename=rv_dict['phoebe_rv_filename'], indep=rv_dict['phoebe_rv_indep'], dep=rv_dict['phoebe_rv_dep'], indweight=indweight, dir=legacy_file_dir)

            rv_dict.update(data_dict)

            time = rv_dict.get('phoebe_rv_time', [])
        #

        rv_dataset = det_dataset(eb, passband, dataid, comp, time)
        dataid = rv_dataset.dataset
        #enable dataset
        enabled = rv_dict['phoebe_rv_active']
        del rv_dict['phoebe_rv_active']

        d ={'qualifier':'enabled', 'dataset':dataid, 'value':enabled}

        eb.set_value_all(check_visible=False, **d)

        # disable interpolating ld coefficients
        eb.set_value_all(qualifier='ld_mode', dataset=dataid, value='manual', **_skip_filter_checks)


    #set ldlaw

        ldlaws_1to2= {'Linear cosine law': 'linear', 'Logarithmic law': 'logarithmic', 'Square root law': 'square_root'}
        if ldlaw == 'Linear cosine law':
            logger.warning('Linear cosine law is not currently supported. Converting to linear instead')

        value = ldlaws_1to2[ldlaw]#val[0].lower()+val[1::]

        # since ld_coeffs is dataset specific make sure there is at least one dataset
        #    if lcno != 0 or rvno != 0:
        d ={'qualifier':'ld_func', 'dataset':dataid, 'value':value}
        eb.set_value_all(check_visible=False, **d)


    #get available passbands and set

        choices = rv_dataset.get_parameter(qualifier='passband', **_skip_filter_checks).choices
        pnew, d = ret_dict('phoebe_rv_filter', rv_dict['phoebe_rv_filter'], dataid=dataid)
        if d['qualifier'] == 'passband' and d['value'] not in choices:
                d['value'] = 'Johnson:V'
        eb.set_value_all(check_visible=False, **d)
        del rv_dict['phoebe_rv_filter']
# now go through parameters and input the results into phoebe2

        for k  in rv_dict:

            pnew, d = ret_dict(k, rv_dict[k], rvdep = comp, dataid=dataid)

            if len(d) > 0:
                eb.set_value_all(check_visible=False, **d)

# And finally spots

    spot_unit =  spotpars[:,1][list(spotpars[:,0]).index('phoebe_spots_units')].strip('"').lower()[:-1]

    for x in range(1,spotno+1):

        spotin = [list(spotpars[:,0]).index(s) for s in spotpars[:,0] if "["+str(x)+"]" in s]
        spotpt = spotpars[spotin]
        source =  np.int(spotpt[:,1][list(spotpt[:,0]).index('phoebe_spots_source['+str(x)+']')])
        spotpt = np.delete(spotpt, list(spotpt[:,0]).index('phoebe_spots_source['+str(x)+']'), axis=0)
        enabled = np.int(spotpt[:,1][list(spotpt[:,0]).index('phoebe_spots_active_switch['+str(x)+']')])
        spotpt = np.delete(spotpt, list(spotpt[:,0]).index('phoebe_spots_active_switch['+str(x)+']'), axis=0)

        if source == 1:
            component = 'primary'
        elif source == 2:
            component = 'secondary'
 #       elif source == 0:
 #           component = 'secondary'
        else:
            raise ValueError("spot component not specified and cannot be added")

#   create spot

        spot = eb.add_feature('spot', component=component)

        #TODO check this tomorrow
        dataid = spot.features[0]

        #enable or disable spot
        val = _bool1to2[enabled]
        eb.set_value_all(qualifier='enabled', value=val, feature=dataid, **_skip_filter_checks)
#   add spot parameters

        for k in range(len(spotpt)):
            param = spotpt[:,0][k].split('[')[0]
            value = spotpt[:,1][k]
            # print "param", param
            pnew, d = ret_dict(param, value, rvdep = component, dataid=dataid)
            if len(d) > 0:
                if d['qualifier'] != 'relteff':
                    d['unit'] = spot_unit

                eb.set_value_all(check_visible=False, **d)

    # reset any flipped constraints
    eb.flip_constraint(constraint_func='t0_ref_supconj', solve_for='t0_ref')


    #change t0_system to equal t0_ref
    t0_ref = eb.get_value(qualifier='t0_ref', context='component', unit='d', **_skip_filter_checks)
    eb.set_value(qualifier='t0', context='system', value=t0_ref, unit='d', **_skip_filter_checks)

    if contact_binary:
        eb.flip_constraint('requiv@primary', solve_for='pot@contact_envelope')
    if 'Linear' in ldlaw:

        ldcos = eb.filter(qualifier='ld_coeffs', **_skip_filter_checks).to_list()
        ldcosbol = eb.filter(qualifier='ld_coeffs_bol', **_skip_filter_checks).to_list()
        for x in range(len(ldcos)):
            val = ldcos[x].value[0]
            ldcos[x].set_value(np.array([val]))

        for x in range(len(ldcosbol)):

            val = ldcosbol[x].value[0]
            ldcosbol[x].set_value(np.array([val]))
    if conf_interactive_constraints_state:
        eb.run_delayed_constraints()
        conf.interactive_constraints_on()
    if conf_interactive_checks_state:
        conf.interactive_checks_on()

    # turn on relevant switches like heating. If
    return eb

"""

Return phoebe1 parameter value in the right units (or without units)
eb -phoebe 2 bundle
d - dictionary with parameter, context, dataset etc.

"""

def par_value(param, index=None, **kwargs):
# build a dictionary

    d={}
    d['qualifier'] = param.qualifier
    d['component'] = param.component
    d['dataset'] = param.dataset
    d['compute'] = param.compute
    d['kind'] = param.kind

# Determine what type of parameter you have and find it's value
    if isinstance(param, phb.parameters.FloatParameter) and not isinstance(param, phb.parameters.FloatArrayParameter):
        ptype = 'float'
    # since it's a float it must have units. Therefore return value in correct units
        pnew = _2to1par[d['qualifier']]
        try:
            unit = _units1[pnew]
        except:
            unit = None

        val = param.get_quantity(unit=unit, **kwargs).value
        # if d['qualifier'] == 'alb':
            # val = [1.0-val]

        # else:
            # val = [val]
        if param.qualifier == 'requiv':
            # NOTE: for the parent orbit, we can assume a single orbit if we've gotten this far
            # NOTE: mapping between the qualifier is handled by the 2to1 dictionary
            b = param._bundle
            comp_no = b.hierarchy.get_primary_or_secondary(param.component, return_ind=True)

            sma = b.get_value(qualifier='sma', kind='orbit', context='component', unit='solRad', **_skip_filter_checks)

            q = b.get_value(qualifier='q', kind='orbit', context='component', **_skip_filter_checks)
            q = roche.q_for_component(q, component=comp_no)
            F = b.get_value(qualifier='syncpar', component=param.component, context='component', **_skip_filter_checks)
            e = b.get_value(qualifier='ecc', kind='orbit', context='component')
            delta = 1-e # at periastron
            s = np.array([0,0,1]).astype(float) # aligned case, we would already have thrown an error if misaligned

            val = roche.requiv_to_pot(val, sma, q, F, delta, s, component=comp_no)
        elif param.qualifier == 'period':
            # find period_sidereal(t0_ref) (currently is period_sidereal(t0@system))
            b = param._bundle
            logger.debug("period_sidereal(t0@system): {}".format(val))
            dpdt = b.get_value(qualifier='dpdt', component=param.component, context='component', unit='d/d', **_skip_filter_checks)
            t0_system = b.get_value(qualifier='t0', context='system', unit='d')
            # NOTE: t0_ref already accounts for BOTH dpdt and dperdt
            t0_ref = b.get_value(qualifier='t0_ref', component=param.component, context='component', unit='d', **_skip_filter_checks)
            val += dpdt * (t0_ref - t0_system)
            logger.debug("period_sidereal(t0_ref, dpdt={} d/d): {}".format(dpdt, val))

        elif param.qualifier == 'per0':
            # find per0(t0_ref) (currently is per0(t0@system))
            b = param._bundle
            dperdt = b.get_value(qualifier='dperdt', component=param.component, context='component', unit='rad/d', **_skip_filter_checks)
            t0_system = b.get_value(qualifier='t0', context='system', unit='d', **_skip_filter_checks)
            t0_ref = b.get_value(qualifier='t0_ref', component=param.component, context='component', unit='d', **_skip_filter_checks)
            logger.debug("per0(t0@system): {}".format(val))
            val += dperdt * (t0_ref - t0_system)
            val = val % (2*np.pi)
            logger.debug("per0(t0_ref, dperdt={} rad/d): {}".format(dperdt, val))

        val = [val]

    elif isinstance(param, phb.parameters.ChoiceParameter):
        ptype = 'choice'
        val = [param.get_value(**kwargs)]
        if d['qualifier'] == 'atm':
        # in phoebe one this is a boolean parameter because you have the choice of either kurucz or blackbody

            ptype='boolean'

        elif d['qualifier'] == 'ld_func' or d['qualifier'] == 'ld_func_bol':

            ldlaws_2to1= {'linear':'Linear cosine law', 'logarithmic':'Logarithmic law', 'square_root':'Square root law'}
            val = ldlaws_2to1[val[0]]
            val = ['"'+str(val)+'"']

        elif d['qualifier'] == 'passband':
            val = [_default_passband_map_2to1.get(val[0], val[0])]

    elif isinstance(param, phb.parameters.BoolParameter):

        ptype = 'boolean'
        val = [_bool2to1[param.get_value(**kwargs)]]

    elif isinstance(param, phb.parameters.IntParameter):
        ptype = 'int'
        val = [param.get_value(**kwargs)]

        # if d['qualifier'] == 'refl_num':

        #    val = [param.get_value()-1]
            # val = [param.get_value()]

        # else:
            # val = [param.get_value()]

    elif isinstance(param, phb.parameters.FloatArrayParameter):
        # val1 = param.get_value()[0]
        # val2 = param.get_value()[1]
        # val = [val1, val2]

        val = param.get_value(**kwargs).tolist()

        ptype='array'
        if len(val) == 1:
            val.append(0.0)

    else:
        ptype = 'unknown'
        val = [param.get_value(**kwargs)]

    return [val, ptype]

"""

Return phoebe1 parameter name from phoebe 2 info


"""

def ret_ldparname(param, comp_int=None, dtype=None, dnum=None, ptype=None, index=None):
    if 'bol' in param:

        if ptype=='array':
            pnew1 = 'xbol'
            pnew2 = 'ybol'
            pnew = [pnew1, pnew2]
        else:
            return ['phoebe_ld_model']
    else:
        if ptype == 'array':
            pnew1 = str(dtype)+'x'
            pnew2 = str(dtype)+'y'
            pnew = [pnew1, pnew2]
        else:
            return ['phoebe_ld_model']

    if comp_int == 1:
        pnew = [x + '1' for x in pnew]

    elif comp_int == 2:
        pnew = [x + '2' for x in pnew]

    if dnum != None:
        dset = '['+str(dnum)+'].VAL'
    else:
        dset = ''
    return ['phoebe_ld_'+x+dset for x in pnew]

def ret_parname(param, comp_int=None, dtype=None, dnum=None, ptype=None, index=None):

# separate lds from everything because they suck

    if 'ld' in param:

        pname = ret_ldparname(param, comp_int=comp_int, dtype=dtype, dnum=dnum, ptype=ptype, index=index)
    else:
    # first determine name of parameters and whether it is associated with a com

        if comp_int == 1:

            if param == 'pblum':
                pnew = 'hla'

            elif param in ['enabled','statweight','l3','passband']:
                pnew = _2to1par.get(param, param)

            else:
                pnew = _2to1par.get(param, param)+'1'

        elif comp_int == 2:

            if param == 'pblum':
                pnew = 'cla'

            elif param in ['enabled','statweight','l3','passband']:
                pnew = _2to1par.get(param, param)


            else:
                pnew = _2to1par.get(param, param)+'2'
        else:

            pnew = _2to1par.get(param, param)

    # one parameter doesn't follow the rules, so i'm correcting for it
            if pnew == 'reflections':
                pnew = 'reffect_reflections'
    #determine the context of the parameter and the number of the dataset

        if dtype != None:
            if param != 'l3':
                dtype = dtype+'_'
            else:
                dtype = ''
        else:
            dtype = ''

        if dnum != None:

            dset = '['+str(dnum)+']'
        else:
            dset = ''
    # determine the determinant of the parameter based on parameter type
        # print "check", param, ptype, dtype

        if ptype == 'float':

            if dtype=='spots_':
                det = ''
            else:
                det = '.VAL'

  #      elif ptype == 'boolean' and dtype=='':
  #          print("inside", param, ptype, dtype)
  #          det = '_switch'

        elif ptype == 'boolean':
            if dtype=='':
                det='_switch'

            elif dtype=='spots_':
                det = '_switch'+dset
                dset = ''
            else:
                det=''
        else:
            det = ''

        pname = ['phoebe_'+str(dtype)+str(pnew)+dset+det]

    return pname

"""
create dictionary of parameters from phoebe bundle
for loading into phoebe legacy
bundle -
compute - compute parameters

"""
def pass_to_legacy(eb, compute=None, **kwargs):


    #run checks which must pass before allowing use of function
    #l3_modes must all be the same
    if kwargs.get('disable_l3', False):
        l3_mode_force_flux = True
    else:
        l3_modes = [p.value for p in eb.filter(qualifier='l3_mode', **_skip_filter_checks).to_list()]
        if len(list(set(l3_modes))) > 1:
            logger.warning("legacy does not natively support mixed values of l3_mode, so all will be converted to 'flux' before passing to PHOEBE legacy.")
            l3_mode_force_flux = True
            eb.compute_l3s(compute=compute, set_value=True)
        else:
            l3_mode_force_flux = False

    eb.run_delayed_constraints()


    # check to make sure you have exactly two stars and exactly one orbit
    stars = eb.hierarchy.get_stars()
    orbits = eb.hierarchy.get_orbits()
    primary, secondary = stars

    if len(stars) != 2 or len(orbits) != 1:
        raise ValueError("PHOEBE 1 only supports binaries. Either provide a different system or edit the hierarchy.")
# check for contact_binary

    contact_binary = eb.hierarchy.is_contact_binary(primary)

    if not contact_binary:
        # contacts are always aligned, for detached systems we need to check
        # to make sure they are aligned.
        for star in stars:
            if eb.hierarchy.is_misaligned():
                raise ValueError("PHOEBE 1 only supports aligned systems.  Edit pitch and yaw to be aligned or use another backend")

    # Did you pass a compute parameter set?
    if compute is not None:
        # print("compute", compute)
        computeps = eb.get_compute(compute=compute)
#        computeps = eb.get_compute(compute=compute, kind='legacy', check_visible=False)

    #Find Compute Parameter Set
    else:
        ncompute = len(eb.filter(context='compute', kind='legacy', **_skip_filter_checks).computes)
        if ncompute == 1:
            computeps = eb.get_compute(kind='legacy', **_skip_filter_checks)
            compute = computeps.compute

        elif ncompute == 0:
            raise ValueError('Your bundle must contain a "legacy" compute parameter set in order to export to a legacy file.')

        else:
            raise ValueError('Your bundle contains '+str(ncompute)+' parameter sets. You must specify one to use.')


    # TODO: can we somehow merge these instead of needing to re-mesh between?

    # handle any limb-darkening interpolation
    eb.compute_ld_coeffs(compute=compute, set_value=True, **{k:v for k,v in kwargs.items() if k in computeps.qualifiers})

    # TODO: remove this check once https://github.com/phoebe-project/phoebe1/issues/4 is closed
    for pblum_param in eb.filter(qualifier='pblum', unit='W', **_skip_filter_checks).to_list():
        if pblum_param.get_value() >= 1e4:
            raise ValueError("PHOEBE legacy cannot handle pblum values larger than 1e4")

# check for semi_detached
    semi_detached = None #keep track of which component is in semidetached
    #handle two semi_detached stars
    requiv_primary_constraint = eb.get_parameter(qualifier='requiv', component=primary, context='component', **_skip_filter_checks).is_constraint
    requiv_secondary_constraint = eb.get_parameter(qualifier='requiv', component=secondary, context='component', **_skip_filter_checks).is_constraint
    if requiv_primary_constraint and requiv_primary_constraint.constraint_func == 'semidetached':
        semi_detached = 'primary'
    if requiv_secondary_constraint and requiv_secondary_constraint.constraint_func == 'semidetached':
        if semi_detached:
            logger.warning('Phoebe 1 does not support double Roche lobe overflow system. Defaulting to Primary star only.')
        else:
            semi_detached = 'secondary'

#  catch all the datasets
    # define datasets


    lcs = eb.get_dataset(kind='lc', **_skip_filter_checks).datasets
    rvs = eb.get_dataset(kind='rv', **_skip_filter_checks).datasets
    # only spots have an enabled parameter in legacy compute options
    spots = eb.filter(qualifier='enabled', compute=compute, **_skip_filter_checks).features

    #create dictionary to store parameters
    legacy_dict = {}
    #make lists to put results with important things already added

    legacy_dict['phoebe_rvno'] = len(rvs)*2
    legacy_dict['phoebe_spots_no'] = len(spots)
    legacy_dict['phoebe_lcno'] = len(lcs)
#    parnames = ['phoebe_rvno', 'phoebe_spots_no', 'phoebe_lcno']
#    parvals = [len(rvs)*2, len(spots), len(lcs)]
#    types = ['int', 'int']

    #Force the independent variable to be time
    legacy_dict['phoebe_indep'] = "Time (HJD)"
#    parnames.append('phoebe_indep')
#    parvals.append('"Time (HJD)"')
#    types.append('choice')
    # add l3_mode
    choice_dict = {'flux':'Flux', 'fraction':'Total light'}
    if len(lcs) > 0:
        if l3_mode_force_flux:
            l3_mode = 'flux'
        else:
            l3_mode = eb.filter(qualifier='l3_mode', **_skip_filter_checks)[0].value
        legacy_dict['phoebe_el3_units'] = '"'+choice_dict[l3_mode]+'"'
 #       parnames.append('phoebe_el3_units')
 #       parvals.append('"'+choice_dict[l3_mode]+'"')
 #       types.append('choice')
    else:
        legacy_dict['phoebe_el3_units'] = '"'+choice_dict['flux']+'"'
#        parnames.append('phoebe_el3_units')
#        parvals.append('"'+choice_dict['flux']+'"')
#        types.append('choice')
# add limb darkening law first because it exists many places in phoebe2



    ldlaws = set([p.get_value() for p in eb.filter(qualifier='ld_func', **_skip_filter_checks).to_list()])

    ldlaws_bol = set([p.get_value() for p in eb.filter(qualifier='ld_func_bol', **_skip_filter_checks).to_list()])


    #no else
    if len(ldlaws) == 1:

        #check values
        if list(ldlaws)[0] not in ['linear', 'logarithmic', 'square_root']:
            raise ValueError(list(ldlaws)[0]+" is not an acceptable value for phoebe 1. Accepted options are 'linear', 'logarithmic' or 'square_root'")
        #define choices
        if ldlaws != ldlaws_bol:
            logger.warning('ld_func_bol does not match ld_func. ld_func will be chosen')

        # BERT: why the filter(...)[0] here?  Why not use get_parameter? (and same below for ld_func_bol)
        param = eb.filter(qualifier='ld_func', component=primary, **_skip_filter_checks)[0]
        val, ptype = par_value(param)
        pname = ret_parname(param.qualifier)
        #load to array
        for y in range(len(pname)):
            legacy_dict[pname[y]] = val[y]
#        parnames.extend(pname)
#        parvals.extend(val)
#        types.append(ptype)

    elif len(set(ldlaws)) > 1:
        raise ValueError("Phoebe 1 takes only one limb darkening law.")

    else:
        if list(ldlaws_bol)[0] not in ['linear', 'logarithmic', 'square_root']:
            raise ValueError(list(ldlaws)[0]+" is not an acceptable value for phoebe 1. Accepted options are 'linear', 'logarithmic' or 'square_root'")

        param = eb.filter(qualifier='ld_func_bol', component=primary)[0]
        val, ptype = par_value(param)
        pname = ret_parname(param.qualifier)
        #load to array
        for y in range(len(pname)):
            legacy_dict[pname[y]] = val[y]


#        parnames.extend(pname)
#        parvals.extend(val)
#        types.append(ptype)
#        raise ValueError("You have not defined a valid limb darkening law.")




    if len(lcs) != 0:
        pblum_mode = eb.get_value(dataset=lcs[0], qualifier='pblum_mode', **_skip_filter_checks)
        if pblum_mode == 'decoupled':
            decouple_luminosity = '1'

            if contact_binary:
                raise ValueError("contact binaries in legacy do not support decoupled pblums")

        elif pblum_mode == 'component-coupled':
            decouple_luminosity = '0'
        elif pblum_mode == 'dataset-scaled':
            decouple_luminosity = '0'
        else:
            # then we'll rely on the values from compute_pblums and pass luminosities for both objecs
            decouple_luminosity = '1'
        legacy_dict['phoebe_usecla_switch'] = decouple_luminosity
#        parnames.append('phoebe_usecla_switch')
#        parvals.append(decouple_luminosity)
#        types.append('boolean')

    prpars = eb.filter(component=primary, context='component', check_visible=True)
    secpars = eb.filter(component=secondary, context='component', check_visible=True)
    if contact_binary:
        #note system
        legacy_dict['phoebe_model'] = '"Overcontact binary not in thermal contact"'
        #parnames.append('phoebe_model')
        #parvals.append('"Overcontact binary not in thermal contact"')
        #types.append('choice')

        comp_int = 1
        envelope = eb.hierarchy.get_envelope_of(primary)
        val = [eb.get_value(qualifier='pot', component=envelope, context='component')]
        ptype = 'float'
        # note here that phoebe1 assigns this to the primary, not envelope
        pname = ret_parname('pot', comp_int=comp_int, ptype=ptype)
        for y in range(len(pname)):
            legacy_dict[pname[y]] = val[y]
        #parnames.extend(pname)
        #parvals.extend(val)
    elif semi_detached:
        legacy_dict['phoebe_model'] = '"Semi-detached binary, '+semi_detached+' star fills Roche lobe'
        #parnames.append('phoebe_model')
        #parvals.append('"Semi-detached binary, '+semi_detached+' star fills Roche lobe')
        #types.append('choice')

#   pblum
        # TODO BERT: need to deal with multiple datasets

    else:
        legacy_dict['phoebe_model'] = '"Detached binary"'
        #parnames.append('phoebe_model')
        #parvals.append('"Detached binary"')
        #types.append('choice')

    for param in prpars.to_list():
        if contact_binary and param.qualifier == 'requiv':
            continue

        comp_int = 1

        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['sma', 'period', 'incl','enabled','statweight','l3'] or param.component == '_default':
                param = None


        except:
            logger.warning(param.twig+' has no phoebe 1 corollary')
            param=None
        if param != None:
            val, ptype = par_value(param)


            pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)
            # print val, ptype, pname
            for y in range(len(pname)):
                if pname[y] not in legacy_dict.keys():

                    legacy_dict[pname[y]] = val[y]
                #parnames.extend(pname)
                #parvals.extend(val)
                #if ptype == 'array':
                #    types.append(ptype)
                #    types.append(ptype)
                #else:
                #    types.append(ptype)

    for param in secpars.to_list():
        if contact_binary and param.qualifier == 'requiv':
            continue

        comp_int = 2
        # make sure this parameter exists in phoebe 1
#        param = eb.get_parameter(secpars[x], component= 'secondary')
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['sma', 'period', 'incl', 'ld_func', 'ld_func_bol'] or param.component == '_default':
                param = None

        except:
            logger.warning(param.twig+' has no phoebe 1 corollary')
            param = None

# get rid of confusing parameters like sma and period which only exist for orbits in phoebe 1

        if param != None:

            val, ptype = par_value(param)
            # if param.qualifier == 'irrad_frac_refl_bol':
                # val = [1-float(val[0])]
            pname = ret_parname(param.qualifier, comp_int=comp_int, ptype=ptype)
            for y in range(len(pname)):
                if pname[y] not in legacy_dict.keys():

                    legacy_dict[pname[y]] = val[y]

                #parnames.extend(pname)
                #parvals.extend(val)
                #if ptype == 'array':
                #    types.append(ptype)
                #    types.append(ptype)
                #else:
                #    types.append(ptype)


# loop through lcs

    for x in range(len(lcs)):
        quals = eb.filter(dataset=lcs[x], context=['dataset', 'compute'], check_visible=False)
        #pull out l3_mode
        quals = quals.exclude('l3_mode')

        #phoebe 2 is ALWAYS times so pass time as the ind variable
        legacy_dict['phoebe_lc_indep['+str(x+1)+']'] = 'Time (HJD)'
        legacy_dict['phoebe_lc_dep['+str(x+1)+']'] = 'Flux'
        legacy_dict['phoebe_lc_id['+str(x+1)+']'] = lcs[x]
        #parnames.append('phoebe_lc_indep['+str(x+1)+']')
        #parvals.append('Time (HJD)')
        #types.append('choice')
        #parnames.append('phoebe_lc_dep['+str(x+1)+']')
        #parvals.append('Flux')
        #types.append('choice')
        #parnames.append('phoebe_lc_id['+str(x+1)+']')
        #parvals.append(lcs[x])
        #types.append('choice')



        for param in quals.to_list():

            if param.component == primary:
                comp_int = 1
            elif param.component == secondary:
                comp_int = 2

            else:
                comp_int = None

            try:
                pnew = _2to1par[param.qualifier]
                if param.qualifier in [ 'alb', 'fluxes', 'sigmas', 'times'] or param.component == '_default':

                    param = None
            except:

                logger.warning(param.twig+' has no phoebe 1 corollary')
                param = None

            if param != None:

                val, ptype = par_value(param)

                if param.qualifier == 'pblum':
                    if contact_binary:
                        if comp_int == 2:
                            # TODO: this is again assuming the the secondary is coupled to the primary
                            continue

                        pname = ret_parname(param.qualifier, comp_int=comp_int, dnum=x+1, ptype=ptype)
                    else:
                        pname = ret_parname(param.qualifier, comp_int=comp_int, dnum=x+1, ptype=ptype)

                elif param.qualifier == 'exptime':

                    logger.warning("Finite integration Time is not fully supported and will be turned off by legacy wrapper before computation")
                #    pname = ['phoebe_cadence_switch']
                #    val = ['0']
                    #ptype='boolean'

                elif param.qualifier == 'l3_frac':
                    if param.is_visible and not l3_mode_force_flux:
                        pname = ret_parname('l3', comp_int=comp_int, dtype='lc', dnum = x+1, ptype=ptype)
                    else:
                        continue

                elif param.qualifier == 'l3':
                    if param.is_visible or l3_mode_force_flux:
                        pname = ret_parname('l3', comp_int=comp_int, dtype='lc', dnum = x+1, ptype=ptype)
                        if kwargs.get('disable_l3', False):
                            val = [0.0 for p in pname]
                    else:
                        continue

                else:

                    pname = ret_parname(param.qualifier, comp_int=comp_int, dtype='lc', dnum = x+1, ptype=ptype)

                for y in range(len(pname)):
                    if pname[y] not in legacy_dict.keys():
                        legacy_dict[pname[y]] = val[y]

                    #parnames.extend(pname)
                    #parvals.extend(val)
                    #if ptype == 'array':
                    #    types.append(ptype)
                    #    types.append(ptype)
                    #else:
                    #    types.append(ptype)


#loop through rvs
#if there is more than one rv...try this
    # set curve number
    num = 1
    for y in range(len(rvs)):

        #get rv qualifiers
        quals = eb.filter(dataset=rvs[y], context=['dataset', 'compute'], **_skip_filter_checks)

        #cycle through components
        comps = quals.filter(qualifier='times', check_visible=True).components
        # comps = eb.hierarchy.get_stars()

        rv_type = {primary:{'curve' : '"Primary RV"', 'comp_int' : 1} , \
        secondary: {'curve':'"Secondary RV"', 'comp_int':2}}

        for i in range(len(comps)):
            legacy_dict['phoebe_rv_indep['+str(num)+']'] = 'Time (HJD)'
            legacy_dict['phoebe_rv_dep['+str(num)+']']  = rv_type[comps[i]]['curve']
            legacy_dict['phoebe_rv_id['+str(num)+']'] = rvs[y]
        #    parnames.append('phoebe_rv_indep['+str(num)+']')
        #    parvals.append('Time (HJD)')
        #    types.append('choice')
            #dependent variable
        #    parnames.append('phoebe_rv_dep['+str(num)+']')
        #    parvals.append(rv_type[comps[i]]['curve'])
        #    types.append('choice')
        #    parnames.append('phoebe_rv_id['+str(num)+']')
        #    parvals.append(rvs[y])

            for param in quals.to_list():


                if param.component == comps[i] or param.component == None:
                    try:
                        pnew = _2to1par[param.qualifier]
                        if param.qualifier in ['ld_func', 'rvs', 'times', 'sigmas'] or param.component == '_default':
                            param = None

                    except:
                        logger.warning(param.twig+' has no phoebe 1 corollary')
                        param = None

                    if param != None:

                        try:

                            comp_int = rv_type[param.component]['comp_int']

                        except:
                            comp_int = None

                        val, ptype = par_value(param)
                        pname = ret_parname(param.qualifier, comp_int = comp_int, dtype='rv', dnum = num, ptype=ptype)

    # if is tries to append a value that already exists...stop that from happening
                for pi in range(len(pname)):
                    if pname[pi] not in legacy_dict.keys():
                        legacy_dict[pname[pi]] = val[pi]

                        #    parnames.extend(pname)
                        #    parvals.extend(val)
                        #    if ptype == 'array':
                        #        types.append(ptype)
                        #        types.append(ptype)
                        #    else:
                        #        types.append(ptype)
            num = num+1

#spots
    legacy_dict['phoebe_spots_units'] = '"Degrees"'
#    parnames.append('phoebe_spots_units')
#    parvals.append('"Degrees"')
#    types.append('choice')
    for y in range(len(spots)):
        #specify component
        source = eb.get_feature(spots[y], **_skip_filter_checks).component
        if source == 'primary':
            source_val=1
        if source == 'secondary':
            source_val=2
        pname = 'phoebe_spots_source['+str(y+1)+']'
        val = source_val
        legacy_dict[pname] = val
        #get qualifiers
        quals = eb.filter(feature=spots[y], context='feature', check_visible=True)
        quals = quals + eb.filter(feature=spots[y], context='compute', compute=compute, check_visible=True)

        for param in quals.to_list():

            if param.component == primary:
                comp_int = 1
            elif param.component == secondary:
                comp_int = 2
            else:
                comp_int = None

            try:
                pnew = _2to1par[param.qualifier]

            except:
                param = None

            if param != None:

                val, ptype = par_value(param)
                pname = ret_parname(param.qualifier, comp_int=None, dtype='spots', dnum = y+1, ptype=ptype)

                for z in range(len(pname)):
                    legacy_dict[pname[z]] = val[z]
            #    parnames.extend(pname)
            #    parvals.extend(val)


#loop through the orbit

    oquals = eb.get_orbit(orbits[0])

    for param in oquals.to_list():
        try:
            pnew = _2to1par[param.qualifier]
            if param.qualifier in ['ld_func']:
                    param = None

        except:

            logger.warning(param.twig+' has no phoebe 1 corollary')
            param = None
        if param != None:
            val, ptype = par_value(param)



            pname = ret_parname(param.qualifier,ptype=ptype)
            for y in range(len(pname)):
                legacy_dict[pname[y]] = val[y]


#loop through LEGACY compute parameter set

    for param in computeps.to_list():
        if param.component == primary:
            comp_int = 1
        elif param.component == secondary:
            comp_int = 2
        else:
            comp_int = None



        if param.qualifier == 'refl_num':
            if param.get_value(**kwargs) == 0:
                #Legacy phoebe will calculate reflection no matter what.
                # Turn off reflection switch but keep albedos
                logger.warning('To completely remove irradiation effects in PHOEBE Legacy irrad_frac_refl_bol must be set to zero for both components')
                pname = 'phoebe_reffect_switch'
                val = '0'
                for y in range(len(pname)):
                    legacy_dict[pname] = val



            else:
                pname = 'phoebe_reffect_switch'
                val = '1'
                ptype = 'boolean'
                for y in range(len(pname)):
                    legacy_dict[pname] = val

        if param.qualifier == 'irrad_method':
            #to completely turn of irradiation in phoebe1 albedos must be zero
            if param.get_value(**kwargs) == 'none':
                 legacy_dict['phoebe_alb1.VAL'] = 0.0
                 legacy_dict['phoebe_alb2.VAL'] = 0.0


        try:

            pnew = _2to1par[param.qualifier]

            if param.qualifier in ['ld_func'] or param.dataset:
                param = None
            elif param.component == '_default':
                param = None
            if pnew == 'active':
                param = None
        except:

            logger.warning(param.twig+' has no phoebe 1 corollary')
            param = None

        if param != None:
            val, ptype = par_value(param, **kwargs)
            if param.qualifier == 'gridsize':
                pname = ret_parname(param.qualifier, comp_int = comp_int, dtype='grid', ptype=ptype)
            elif param.qualifier =='atm':
                atmval = {'extern_atmx':1, 'extern_planckint':0}
                pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)

                val = str(atmval[val[0]])
            else:
                pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)

            if pname[0] not in legacy_dict.keys():

                legacy_dict[pname[0]] = val[0]


    sysquals = eb.filter(context='system')

    for param in sysquals.to_list():
        if param.component == primary:
            comp_int = 1
        elif param.component == secondary:
            comp_int = 2
        else:
            comp_int = None

        try:
            pnew = _2to1par[param.qualifier]
        except:
            logger.warning(param.twig+' has no phoebe 1 corollary')
            param = None

        if param != None:

            val, ptype = par_value(param)

            pname = ret_parname(param.qualifier, comp_int = comp_int, ptype=ptype)
            if pname[0] not in legacy_dict.keys():
                legacy_dict[pname[0]] = val[0]


# Phoebe1 has certain parameters that do not have phoebe 2 corollaries. If you did
#not load a phoebe1 compute parameter set these must be set to defaults
    kinds = []

    for x in eb.computes:
        kinds.append(eb[x].meta['kind'])
    if 'legacy' not in kinds:
        #phoebe 1 defaults
        legacy_dict['phoebe_reffect_reflections'] = '1'
        legacy_dict['phoebe_reffect_switch'] = '0'
        legacy_dict['phoebe_ie_switch'] = '0'




    return legacy_dict





"""
import parameters to legacy
params - dictionary of parameters

"""



def import_to_legacy(params):
    #initiate phoebe legacy python wrapper
    if not _use_phb1:
        raise ImportError("phoebeBackend for phoebe legacy not found")

    #number of datasets must be loaded first
    rvno = params['phoebe_rvno']
    lcno = params['phoebe_lcno']
    spno = params['phoebe_spots_no']

    phb1.setpar('phoebe_rvno', rvno, 0)
    phb1.setpar('phoebe_lcno', lcno, 0)
    phb1.setpar('phoebe_spots_no', spno, 0)

    params.pop('phoebe_rvno')
    params.pop('phoebe_lcno')
    params.pop('phoebe_spots_no')


    keys = params.keys()

    for key in keys:
        value = params[key]
        #remove VAL from key
        if len(key.split('.')) == 2:
                key = key.split('.')[0]

        #make sure value is proper data type
        if type(value) == str:
            if '"' in value:
                value = value.strip('"')
            elif '.' in value:
                value = float(value)
            else:
                try:
                    value = int(value)
                except:
                    pass
        #normal or dataset param?
        if '[' in key: #dataset
            num = int(key[-2])-1
            param = key[:-3]

            try:
                phb1.setpar(param, value, num)
            except:
                print(param+', '+str(value)+', '+str(num)+' didnt get added')
        else:
            param = key
            try:
                phb1.setpar(param, value, 0)
            except:
                print(param+', '+str(value)+' didnt get added')
    return

def write_legacy_file(params, filename):
    keys = params.keys()
    f = open(filename, 'w+')
    f.write('# Phoebe 1 file created from phoebe 2 bundle. Some functionality may be lost\n')

    for key in keys:
        param = key
        value = params[param]
        f.write(str(param)+' = '+str(value)+'\n')

    return


def N_to_Ntriangles(N):
    """
    @N: WD style gridsize

    Converts WD style grid size @N to the number of triangles on the
    surface.

    Returns: number of triangles.
    """

    theta = np.array([np.pi/2*(k-0.5)/N for k in range(1, N+1)])
    phi = np.array([[np.pi*(l-0.5)/Mk for l in range(1, Mk+1)] for Mk in np.array(1 + 1.3*N*np.sin(theta), dtype=int)])
    Ntri = 2*np.array([len(p) for p in phi]).sum()

    return Ntri
