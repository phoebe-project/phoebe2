"""
Various tools and utilities.
"""
import logging
import sys
import os
import inspect
import webbrowser
import numpy as np


#{ Loggers

class MyFormatter(logging.Formatter):
    width = 10

    def format(self, record):
        max_filename_width = self.width - 3 - len(str(record.lineno))
        filename = record.filename
        if len(record.filename) > max_filename_width:
            filename = record.filename[:max_filename_width]
        a = "%s:%s" % (filename, record.lineno)
        return "[%s] %s" % (a.ljust(self.width), record.msg)


def arguments():
    """
    Return arguments from the current function.
    
    Returns tuple containing dictionary of calling function's named arguments
    and a list of calling function's unnamed positional arguments.
    """
    posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    return args, posargs

def which(cmd):
    """
    Locate a command.
    
    Emulates "which" command from Linux terminal.
    
    @param cmd: command to run
    @type cmd: str
    @return: absolute path to command
    @rtype: str
    """
    paths = os.path.expandvars('$PATH').split(':')
    for path in paths:
        attempt = os.path.join(path, cmd)
        if os.path.isfile(attempt):
            break
    else:
        raise ValueError("Command '{}' not found".format(cmd))
    return attempt


def get_basic_logger(style="default",clevel='WARNING',
                             flevel='DEBUG',filename=None,filemode='w'):
    """
    Return a basic logger via a log file and/or terminal.
    
    Example 1: log only to the console, accepting levels "INFO" and above
    
    >>> logger = utils.get_basic_logger()
    
    Example 2: log only to the console, accepting levels "DEBUG" and above
    
    >>> logger = utils.get_basic_logger(clevel='DEBUG')
    
    Example 3: log only to a file, accepting levels "DEBUG" and above
    
    >>> logger = utils.get_basic_logger(clevel=None,filename='mylog.log')
    
    Example 4: log only to a file, accepting levels "INFO" and above
    
    >>> logger = utils.get_basic_logger(clevel=None,flevel='INFO',filename='mylog.log')
    
    Example 5: log to the terminal (INFO and above) and file (DEBUG and above)
    
    >>> logger = utils.get_basic_logger(filename='mylog.log')
    
    The different logging styles are:
    
    C{style='default'}::
    
        Wed, 13 Feb 2013 08:47 root INFO     Some information
    
    C{style='grandpa'}::
    
        # INFO    Some information
    
    C{style='minimal'}::
    
        Some information
    
    @param style: logger style
    @type style: str, one of 'default','grandpa','minimal'
    """
    name = ""
    #-- define formats
    if style=='default':
        format  = '%(asctime)s %(name)-12s %(levelname)-7s %(message)s'
        datefmt = '%a, %d %b %Y %H:%M'
    elif style=='grandpa':
        format  = '# %(levelname)-7s %(message)s'
        datefmt = '%a, %d %b %Y %H:%M'
    elif style=='minimal':
        format = ''
        datefmt = '%a, %d %b %Y %H:%M'
        
    if style=='trace':
        formatter = MyFormatter()
    else:
        formatter = logging.Formatter(fmt=format,datefmt=datefmt)
    
    
    if clevel: clevel = logging.__dict__[clevel.upper()]
    if flevel: flevel = logging.__dict__[flevel.upper()]
    
    #-- set up basic configuration.
    #   The basicConfig sets up one default logger. If you give a filename, it's
    #   a FileHandler, otherwise a StreamHandler.
    #-- If we want console and filename, first set up a basic FileHandler, then
    #   add terminal StreamHandler
    if filename is not None:
        if flevel is None:
            level = clevel
        else:
            level = flevel
        logging.basicConfig(level=level,
                            format=format,datefmt=datefmt,
                            filename=filename,filemode=filemode)
        fh = logging.FileHandler(filename)
        fh.setLevel(flevel)
        fh.setFormatter(formatter)
        logging.getLogger(name).addHandler(fh)
    if filename is not None and clevel:    
        # define a Handler which writes INFO messages or higher to the sys.stderr
        ch = logging.StreamHandler()
        ch.setLevel(clevel)
        # tell the handler to use this format
        ch.setFormatter(formatter)
        logging.getLogger(name).addHandler(ch)
    #-- If we only want a console:
    else:
        logging.basicConfig(level=clevel,format=format,datefmt=datefmt,
                            filename=filename,filemode=filemode)        
    #-- fix filename logging
    if filename is not None:
        logging.getLogger(name).handlers[0].level = flevel
    return logging.getLogger(name)

def pop_streamhandlers(logger):
    #-- remove streamhandlers -- they don't have a baseFilename
    to_remove = [handler for handler in logger.handlers if not hasattr(handler,'baseFilename')]
    logger.handlers = [handler for handler in logger.handlers if hasattr(handler,'baseFilename')]
    return to_remove

def pop_filehandlers(logger):
    #-- remove filehandlers -- they have a baseFilename
    to_remove = [handler for handler in logger.handlers if hasattr(handler,'baseFilename')]
    logger.handlers = [handler for handler in logger.handlers if not hasattr(handler,'baseFilename')]
    return to_remove

def add_filehandler(logger,style="default",flevel='DEBUG',
                   filename=None,filemode='w'):
    name = ""
    #-- define formats
    if style=='default':
        format  = '%(asctime)s %(name)-12s %(levelname)-7s %(message)s'
        datefmt = '%a, %d %b %Y %H:%M'
    elif style=='grandpa':
        format  = '# %(levelname)-7s %(message)s'
        datefmt = '%a, %d %b %Y %H:%M'
    elif style=='minimal':
        format = ''
        datefmt = '%a, %d %b %Y %H:%M'
    flevel = logging.__dict__[flevel.upper()]
    logging.basicConfig(level=flevel,
                        format=format,datefmt=datefmt,
                        filename=filename,filemode=filemode)
    fh = logging.FileHandler(filename)
    fh.setLevel(flevel)
    formatter = logging.Formatter(fmt=format,datefmt=datefmt)
    fh.setFormatter(formatter)
    logging.getLogger(name).addHandler(fh)
    logging.getLogger(name).handlers[-1].level = flevel
    return logging.getLogger(name)

def help(obj):
    """
    Open the HTML help pages for a certain function or class instance.
    """
    try:
        myclass = obj
        modfile = inspect.getabsfile(obj)
    except TypeError:
        myclass = obj.__class__
        modfile = inspect.getabsfile(obj.__class__)
    
    path = os.path.splitext(modfile[modfile.rfind('phoebe'):])[0]
    
    url_base = "http://www.phoebe-project.org/2.0/docs/"
    url_module = ".".join(path.split(os.sep)[:-1]) + '.html'
    url_loc = ".".join(path.split(os.sep)) + '.' + myclass.__name__
    
    url = os.path.join(url_base, url_module + '#' + url_loc)
    webbrowser.open(url)

#}



#{ Common tools

   
def traverse(o, list_types=(list, tuple),dict_types=(dict,)):
    """
    Walk down nested iterables.
    
    List-type iterables are traversed as::
        
        for value in list:
            yield value
    
    Dictionary-type iterables are traversed as::
    
        for value in dict:
            yield dict[value]
            
    @param o: object to traverse
    @type o: iterable
    @param list_types: list of classes to treat as lists
    @type list_types: list of classes
    @param dict_types: list of classes to treat as dictionaries
    @type dict_types: list of classes
    """
    #-- in the case of a list:
    if isinstance(o, list_types):
        for value in o:
            for subvalue in traverse(value,list_types=list_types,dict_types=dict_types):
                yield subvalue
    #-- in the case of a dictionary:
    elif isinstance(o, dict_types):
        for value in o:
            for subvalue in traverse(o[value],list_types=list_types,dict_types=dict_types):
                yield subvalue
    #-- otherwise we don't have to traverse
    else:
        yield o


def traverse_memory(o, memory=None, list_types=(list, tuple),dict_types=(dict,),
                    get_label=(),get_key=(),get_context=(),skip=(),
                    parset_types=(),container_types=()):
    """
    Walk down nested iterables.
    
    List-type iterables are traversed as::
        
        for value in list:
            yield value
    
    Dictionary-type iterables are traversed as::
    
        for value in dict:
            yield dict[value]
            
    @param o: object to traverse
    @type o: iterable
    @param list_types: list of classes to treat as lists
    @type list_types: list of classes
    @param dict_types: list of classes to treat as dictionaries
    @type dict_types: list of classes
    """
    if memory is None:
        memory = ['root']
    #-- in the case of a list:
    if isinstance(o, list_types):
        for value in o:
            if isinstance(value,skip): continue
            yield value,memory+[value]
            for subvalue,mem in traverse_memory(value,memory=memory+[value],
                                                list_types=list_types,
                                                dict_types=dict_types,
                                                parset_types=parset_types):
                if isinstance(subvalue,skip): continue
                yield subvalue,mem
    #-- in the case of a dictionary:
    elif isinstance(o, dict_types):
        for value in o:
            if isinstance(value,skip): continue
            yield value,memory+[value]
            for subvalue,mem in traverse_memory(o[value],memory=memory+[value],
                                                list_types=list_types,
                                                dict_types=dict_types,
                                                parset_types=parset_types):
                if isinstance(subvalue,skip): continue
                yield subvalue,mem
    #-- in the case of a parameterSet:
    elif isinstance(o, parset_types):
        for value in o:
            value = o.get_parameter(value)
            yield value,memory+[value]
    #-- in the case of a Container
    elif isinstance(o, container_types):
        for key in o.sections.keys():
            value = o.sections[key]
            yield value, memory+[value]
    #-- otherwise we don't have to traverse
    else:
        yield o,memory

            

    
def deriv(x,y):
    """
    3 point Lagrangian differentiation.
    
    Returns z = dy/dx
    
    Example usage:
    
    >>> X = np.array([ 0.1, 0.3, 0.4, 0.7, 0.9])
    >>> Y = np.array([ 1.2, 2.3, 3.2, 4.4, 6.6])
    >>> deriv(X,Y)
    array([  3.16666667,   7.83333333,   7.75      ,   8.2       ,  13.8       ])
    """
    #-- body derivation
    x0_x1 = np.roll(x,1)-x
    x1_x2 = x-np.roll(x,-1)
    x0_x2 = np.roll(x,1)-np.roll(x,-1)
    derivee = np.roll(y,1)*x1_x2/(x0_x1*x0_x2)
    derivee = derivee + y*(1/x1_x2-1/x0_x1)
    derivee = derivee - np.roll(y,-1)*x0_x1/(x0_x2*x1_x2)
    #-- edges
    derivee[0]=y[0]*(1./x0_x1[1]+1./x0_x2[1])
    derivee[0]=derivee[0]-y[1]*x0_x2[1]/(x0_x1[1]*x1_x2[1])
    derivee[0]=derivee[0]+y[2]*x0_x1[1]/(x0_x2[1]*x1_x2[1])
    nm3=len(x)-3
    nm2=len(x)-2
    nm1=len(x)-1
    derivee[nm1]=-y[nm3]*x1_x2[nm2]/(x0_x1[nm2]*x0_x2[nm2])
    derivee[nm1]=derivee[nm1]+y[nm2]*x0_x2[nm2]/(x0_x1[nm2]*x1_x2[nm2])
    derivee[nm1]=derivee[nm1]-y[nm1]*(1./x0_x2[nm2]+1./x1_x2[nm2])
    
    return derivee



def phasediagram(time, y, period, t0=0.0, repeat=0, sort=True,
                 return_sortarray=False):
    """
    Construct a phase diagram.
    
    @param y: a list of arrays to phase up.
    @type y: list of numpy arrays
    """
    
    phase = ((time-t0) % period) / period
    
    if sort:
        sa = np.argsort(phase)
        y = [iy[sa] for iy in y]
        phase = phase[sa]
    
    # Better remove this?
    if repeat:
        phase = np.hstack([phase + i*period for i in range(repeat+1)])
        y = [np.hstack([iy]*(repeat+1)) for iy in y]
    
    if return_sortarray:
        return phase, y, sa
    else:
        return phase, y



def mad(a, axis=None):
    """
    Compute *Median Absolute Deviation* of an array along given axis.
    
    From https://informatique-python.readthedocs.org/en/latest/Exercices/mad.html
    """

    med = np.median(a, axis=axis)                # Median along given axis
    if axis is None:
        umed = med                              # med is a scalar
    else:
        umed = np.expand_dims(med, axis)         # Bring back the vanished axis
    mad = np.median(np.abs(a - umed), axis=axis) # MAD along given axis

    return mad
#}
