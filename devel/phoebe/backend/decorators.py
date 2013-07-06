"""
Decorators specifically for backend purposes.
"""
import functools
import inspect
import cPickle
import tempfile
import subprocess
import os
from phoebe.parameters import parameters
from phoebe.utils import decoratorsmod


def parse_ref(fctn):
    """
    Expand the general ref keyword in a function to a list of valid references.
    
    If 'ref' is given, make sure it returns a list of refs.
    
    C{ref} can be any of:
        - '__bol': only bolometric
        - 'all': bolometric and all observables
        - 'alldep': all observables
        - 'alllcdep': all observables of type lc
        - 'myref': the observable with ref 'myref'
        - ['ref1','ref2']: observables matching these refs
        
    """
    @functools.wraps(fctn)
    def parse(self, *args, **kwargs):
        """
        Expand the ref keyword to valid ref strings.
        
        Possibilities:
            - None: only bolometric
            - '__bol': only bolometric
            - 'all': bolometric and all observables
            - 'alldep': all observables
            - 'alllcdep': all observables of type lc
            - 'myref': the observable with ref 'myref'
            - ['ref1','ref2']: observables matching these refs
        """
        # Take the default from the function that is decorated using the
        # "inspect" standard library module
        fctn_args = inspect.getargspec(fctn)
        if 'ref' in fctn_args.args:
            Nargs = len(fctn_args.args)
            index = fctn_args.args.index('ref')-Nargs
            default = fctn_args.defaults[index]
            
        # If no default was given there, the "ueber"-default is 'all'.
        else:
            default = 'all'    
        
        # Now retrieve the ref or return the default. The trailing-underscore
        # variable will be used to collect all the strings of valid references.
        ref = kwargs.pop('ref',default)
        ref_ = []
        
        # If the ref is allready a list, check if the ref is in this Body, if
        # it's not, we remove it!
        if isinstance(ref, (list, tuple)):
            
            # These are the refs that are given:
            refs_given = set(list(ref))
            
            # These are the refs that are available:
            refs_avail = []
            for ps in self.walk():
                if not ps.context[-3:] == 'dep':
                    continue
                refs_avail.append(ps['ref'])
            refs_avail = set(refs_avail + ['__bol']) # (bolometric one is also available)
            
            # So these are the refs that are given and available
            kwargs['ref'] = list(refs_given & refs_avail)
        
        # If the ref is None, it means we need to take the bolometric one
        elif ref is None or ref == '__bol':
            kwargs['ref'] = ['__bol']
        
        # Else there are a number of possibilities
        else:
            
            # If ref is all, we also take the bolometric one
            if ref == 'all':
                ref_.append('__bol')
                
            # Let's walk over all deps and retrieve necessary references.
            for ps in self.walk():
                if not ps.context[-3:] == 'dep':
                    continue
                
                # If ref is all or alldep, take them all! Also add it if the
                # ref matches
                if ref == 'all' or ref == 'alldep' or ref == ps['ref']:
                    ref_.append(ps['ref'])
                
                # If ref is alllcdep, take only the lc ones
                elif ref[:3] == 'all' and ps.context[:2] == ref[3:5]:
                    ref_.append(ps['ref'])
            
            # Make sure not to take duplicates
            kwargs['ref'] = list(set(ref_))
            
        return fctn(self, *args, **kwargs)
    
    return parse




def merge_synthetic(list_of_bodies):
    """
    Merge synthetic data of a list of bodies.
    
    Evidently, the bodies need to represent the same system.
    
    Could be useful for MPI stuff.
    
    @param list_of_bodies: a list of bodies for which the results need to be merged
    @type list_of_bodies: list of Bodies
    @return: the first body but with the results of all the others added
    @rtype: Body
    """
    iterators = [body.walk_type(type='syn') for body in list_of_bodies]
    for iteration in zip(*iterators):
        for ps in iteration[1:]:
            if ps.context[-3:]=='syn':
                for key in ps:
                    if 'columns' in ps and not key in ps['columns']:
                        continue
                    value = ps[key]
                    if isinstance(value,list):
                        iteration[0][key] += value
    return list_of_bodies[0]









def mpirun(fctn):
    """
    Use MPIRUN to execute a function on different nodes.
    
    Restrictions on the calling signature of the function C{fctn}::
    
        fctn(system,*args,**kwargs)
        
    Restrictions on the type of args and kwargs::
        
        They need to be pickleable.
    
    Restrictions on kwargs::
        
        The C{params} ParameterSet needs to be given.
    
    The decorator adds the new keyword argument C{mpi}, which is either
    C{None} (no mpi) or a ParameterSet with the details on how to run MPI.
    
    This decorator pickles the C{system}, the C{args} and C{kwargs} in
    separate C{NamedTemporaryFiles}. Next, in each different thread the
    module C{MPI/mpirun.py} is called from a terminal. The command-line
    interface requires 4 positional arguments::
    
        python mpirun.py func system.pck args.pck kwargs.pck
    
    where C{func} is the name of the function to be called, that needs to be
    defined inside C{observatory} (a C{getattr(observatory,func)} is done).
    
    After each run, the synthetic data are collected and finally merged.
    """
    @functools.wraps(fctn)
    def do_run(system,*args,**kwargs):
        mpirun = kwargs.pop('mpi',None)
        if mpirun is None:
            return fctn(system,*args,**kwargs)
        else:
            params = kwargs.pop('params',None)
            if params is None:
                try:
                    params = parameters.ParameterSet(context='compute',**kwargs)
                except:
                    print("You cannot give extra_func with MPI")
                    raise
            kwargs['params'] = params
            if mpirun is True:
                mpirun = parameters.ParameterSet(context='mpi')
            try:
                #-- pickle args and kwargs in NamedTemporaryFiles, which we will
                #   delete afterwards
                direc = os.getcwd()
                sys_file = tempfile.NamedTemporaryFile(delete=False,dir=direc)
                cPickle.dump(system,sys_file)
                sys_file.close()
                args_file = tempfile.NamedTemporaryFile(delete=False,dir=direc)
                cPickle.dump(args,args_file)
                args_file.close()
                kwargs_file = tempfile.NamedTemporaryFile(delete=False,dir=direc)
                cPickle.dump(kwargs,kwargs_file)
                kwargs_file.close()
                #-- construct mpirun command
                np = mpirun['np']
                byslot = ' --byslot' if mpirun['byslot'] else ''
                python = mpirun['python']
                hostfile = ' --hostfile {}'.format(mpirun['hostfile']) if mpirun['hostfile'] else ''
                mpirun_loc = os.path.abspath(__file__)
                mpirun_loc = os.path.split(os.path.split(mpirun_loc)[0])[0]
                mpirun_loc = os.path.join(mpirun_loc,'backend','mpirun.py')
                #-- run mpi    
                cmd = 'mpirun -np {np}{hostfile}{byslot} {python} {mpirun_loc} {fctn.__name__} {sys_file.name} {args_file.name} {kwargs_file.name}'.format(**locals())
                flag = subprocess.check_call(cmd,shell=True)
                # Load the results from the function from the pickle file
                with open(sys_file.name,'r') as ff:
                    results = cPickle.load(ff)
                # Merge the original system with the results from the function
                merge_synthetic([system,results])
            finally:
                # Clean up pickle files that are lying around
                if os.path.isfile(sys_file.name):
                    os.unlink(sys_file.name)
                if os.path.isfile(args_file.name):
                    os.unlink(args_file.name)
                if os.path.isfile(kwargs_file.name):
                    os.unlink(kwargs_file.name)
            system.compute_pblum_or_l3()
    return do_run
    
#def mpirun_emcee(fctn):
    #"""
    #Use MPIRUN to execute a emcee on different nodes.
    #"""
    #@functools.wraps(fctn)
    #def do_run(system,params,mpi,fitparams,pool=None):
        #mpirun = mpi
        ##mpirun = True
        #if mpirun is None:
            #return fctn(system,params,mpi,fitparams,pool=pool)
        #else:
            #if mpirun is True:
                #mpirun = parameters.ParameterSet(context='mpi')
            #try:
                ##-- pickle parameters in NamedTemporaryFiles, which we will
                ##   delete afterwards
                #direc = os.getcwd()
                #sys_file = tempfile.NamedTemporaryFile(delete=False,dir=direc)
                #cPickle.dump(system,sys_file)
                #sys_file.close()
                #params_file = tempfile.NamedTemporaryFile(delete=False,dir=direc)
                #cPickle.dump(params,params_file)
                #params_file.close()
                #mpi_file = tempfile.NamedTemporaryFile(delete=False,dir=direc)
                #cPickle.dump(mpi,mpi_file)
                #mpi_file.close()
                #fitparams_file = tempfile.NamedTemporaryFile(delete=False,dir=direc)
                #cPickle.dump(params,fitparams_file)
                #fitparams_file.close()
                ##-- construct mpirun command
                #np = mpirun['np']
                #byslot = ''#' --byslot' if mpirun['byslot'] else ''
                #python = mpirun['python']
                #hostfile = ' --hostfile {}'.format(mpirun['hostfile']) if mpirun['hostfile'] else ''
                #mpirun_loc = os.path.abspath(__file__)
                #mpirun_loc = os.path.split(os.path.split(mpirun_loc)[0])[0]
                #mpirun_loc = os.path.join(mpirun_loc,'backend','fitting.py')
                ##-- run mpi
                #cmd = 'mpirun -np {np}{hostfile}{byslot} {python} {mpirun_loc} {fctn.__name__} {sys_file.name} {params_file.name} {mpi_file.name} {fitparams_file.name}'.format(**locals())
                #print cmd
                #flag = subprocess.check_call(cmd,shell=True)
                ## Load the results from the function from the pickle file
                #with open(fitparams_file.name,'r') as ff:
                    #fitparams = cPickle.load(ff)
            #finally:
                ## Clean up pickle files that are lying around
                #if os.path.isfile(sys_file.name):
                    #os.unlink(sys_file.name)
                #if os.path.isfile(params_file.name):
                    #os.unlink(params_file.name)
                #if os.path.isfile(mpi_file.name):
                    #os.unlink(mpi_file.name)
                #if os.path.isfile(fitparams_file.name):
                    #os.unlink(fitparams_file.name)
            #return fitparams
    #return do_run