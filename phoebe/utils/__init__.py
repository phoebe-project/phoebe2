import logging

import sys

import numpy as np

_skip_filter_checks = {'check_default': False, 'check_visible': False}

import logging
logger = logging.getLogger("UTILS")
logger.addHandler(logging.NullHandler())

def _bytes(s):
    return bytes(s, 'utf-8')

def get_basic_logger(clevel='WARNING',flevel='DEBUG',
                     style="default",filename=None,filemode='w'):
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

def parse_json(pairs):
    """
    modified from:
    https://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-from-json#34796078

    pass this to the object_pairs_hook kwarg of json.load/loads
    """
    def _string(item):
        if isinstance(item, bytes):
            # return item.decode('utf-8')
            return _bytes(item)
        else:
            return item

    new_pairs = []
    for key, value in pairs:
        key = _string(key)

        if isinstance(value, dict):
            value = parse_json(value.items())
        elif isinstance(value, list):
            value = [_string(v) for v in value]
        else:
            value = _string(value)

        new_pairs.append((key, value))
    return dict(new_pairs)

def phase_mask_inds(phases, mask_phases):
    def _individual_mask(phases, mask):
        # move mask onto range (-0.5, 0.5)
        def _map(m):
            if m < -0.5:
                return _map(m+1)
            if m > 0.5:
                return _map(m-1)
            return m

        mask = [_map(m) for m in mask]

        does_wrap = mask[0] > mask[1]

        if does_wrap:
            return np.logical_or(phases > mask[0], phases < mask[1])
        else:
            return np.logical_and(phases >= mask[0], phases <= mask[1])

    if mask_phases is None:
        return np.isfinite(phases)

    masks = [_individual_mask(phases, m) for m in mask_phases]
    if len(masks) == 0:
        inds = np.isfinite(phases)
    elif len(masks) == 1:
        inds = masks[0]
    else:
        inds = np.logical_or(*masks)

    return inds


def _get_masked_times(b, dataset, mask_phases, mask_t0, return_times_phases=False):
    # concatenate for the case of datasets (like RVs) with times in multiple components
    times = np.unique(np.concatenate([time_param.get_value() for time_param in b.filter(qualifier='times', dataset=dataset, **_skip_filter_checks).to_list()]))
    phases = b.to_phases(times, t0=mask_t0)
    masked_times = times[phase_mask_inds(phases, mask_phases)]
    if return_times_phases:
        return masked_times, times, phases
    return masked_times

def _get_masked_compute_times(b, dataset, mask_phases, mask_t0, is_time_dependent, times=None, phases=None):
    # for compute_times/phases we can't just mask because we need to make
    # sure we "surround" each of the observation datapoints
    if times is None:
        times = np.unique(np.concatenate([time_param.get_value() for time_param in b.filter(qualifier='times', dataset=dataset, unit='d', **_skip_filter_checks).to_list()]))
    if phases is None:
        phases = b.to_phases(times, t0=mask_t0)

    compute_times = b.get_value(qualifier='compute_times', dataset=dataset, context='dataset', unit='d', **_skip_filter_checks)

    if mask_phases is None:
        return compute_times

    compute_phases = b.to_phases(compute_times, t0=mask_t0)

    indices = []

    def _phase_diff(ph1, ph2):
        # need to account for phase-wrapping when finding the nearest two points
        return min([abs(ph1-ph2), abs(ph1+1-ph2), abs(ph1-1-ph2)])

    if is_time_dependent:
        times_masked = times[phase_mask_inds(phases, mask_phases)]

        for tm in times_masked:
            indices += list(abs(compute_times-tm).argsort()[:2])
    else:
        phases_masked = phases[phase_mask_inds(phases, mask_phases)]

        for phm in phases_masked:
            indices += list(np.array([_phase_diff(cph, phm) for cph in compute_phases]).argsort()[:2])

    return compute_times[list(set(indices))]
