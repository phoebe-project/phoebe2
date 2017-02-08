import logging

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