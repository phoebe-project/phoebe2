########################################################################
#                                                                      #
#    Author of Makefile.am: Andrej Prsa <andrej.prsa@villanova.edu>    #
#                                                                      #
########################################################################
#                                                                      #
# This program is free software; you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation; either version 2, or (at your option)  #
# any later version.                                                   #
#                                                                      #
# This program is distributed in the hope that it will be useful,      #
# but WITHOUT ANY WARRANTY; without even the implied warranty of       #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        #
# GNU General Public License for more details.                         #
#                                                                      #
# You should have received a copy of the GNU General Public License    #
# along with this program; if not, write to the Free Software          #
# Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.            #
#                                                                      #
########################################################################

lib_LTLIBRARIES = libscripter.la
bin_PROGRAMS = phoebe-scripter

phoebe_scripter_SOURCES = main.c

libscripter_la_SOURCES =                                                \
    phoebe_scripter.grm.y             phoebe_scripter.lng.l             \
    phoebe_scripter_arithmetics.c     phoebe_scripter_arithmetics.h     \
    phoebe_scripter_ast.c             phoebe_scripter_ast.h             \
    phoebe_scripter_commands.c        phoebe_scripter_commands.h        \
    phoebe_scripter_configuration.c   phoebe_scripter_configuration.h   \
    phoebe_scripter_core.c            phoebe_scripter_core.h            \
    phoebe_scripter_directives.c      phoebe_scripter_directives.h      \
    phoebe_scripter_error_handling.c  phoebe_scripter_error_handling.h  \
    phoebe_scripter_io.c              phoebe_scripter_io.h              \
    phoebe_scripter_plotting.c        phoebe_scripter_plotting.h        \
    phoebe_scripter_types.c           phoebe_scripter_types.h           

BUILT_SOURCES = phoebe_scripter.lng.h phoebe_scripter.grm.h

EXTRA_DIST = phoebe_scripter.lng.h

phoebe_scripter.lng.h:
	$(LEX) phoebe_scripter.lng.l

AM_YFLAGS = -d
AM_CFLAGS = -Wall $(EXTRA_WARNING_COMPILER_FLAG)

phoebe_scripter_LDADD = -lscripter

library_includedir=$(includedir)/phoebe
library_include_HEADERS = phoebe_scripter.h                \
                          phoebe_scripter.lng.h            \
                          phoebe_scripter_arithmetics.h    \
                          phoebe_scripter_ast.h            \
                          phoebe_scripter_commands.h       \
                          phoebe_scripter_configuration.h  \
                          phoebe_scripter_core.h           \
                          phoebe_scripter_directives.h     \
                          phoebe_scripter_error_handling.h \
                          phoebe_scripter_io.h             \
                          phoebe_scripter_plotting.h       \
                          phoebe_scripter_types.h
