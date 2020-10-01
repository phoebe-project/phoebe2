# coding: utf-8
"""
NOTE: this is a modified version of Schwimmbad 0.3.0 (https://github.com/adrn/schwimmbad)
written by Adrian Price-Whelan (and contributors listed below).  It has been
stripped down and modified for the needs within PHOEBE and redistributed under
the original MIT license.

Contributions by:
- Peter K. G. Williams
- Júlio Hoffimann Mendes
- Dan Foreman-Mackey
"""

# Standard library
import sys
import logging
log = logging.getLogger(__name__)
_VERBOSE = 5

from .multipool import MultiPool
from .mpi import MPIPool
