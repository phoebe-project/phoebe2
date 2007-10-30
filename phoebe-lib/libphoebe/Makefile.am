########################################################################
#                                                                      #
#                          PHOEBE 0.30 library                         #
#                                                                      #
########################################################################
#                                                                      #
#    Author of Makefile.am: Andrej Prsa <andrej.prsa@fmf.uni-lj.si>    #
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

lib_LTLIBRARIES = libphoebe.la

libphoebe_la_SOURCES =                                                  \
    phoebe_accessories.c              phoebe_accessories.h              \
    phoebe_allocations.c              phoebe_allocations.h              \
    phoebe_base.c                     phoebe_base.h                     \
    phoebe_build_config.h             phoebe_global.h                   \
    phoebe_calculations.c             phoebe_calculations.h             \
    phoebe_configuration.c            phoebe_configuration.h            \
    phoebe_constraints.c              phoebe_constraints.h              \
    phoebe_constraints.lex.c          phoebe_constraints.lex.h          \
    phoebe_constraints.tab.c          phoebe_constraints.tab.h          \
    phoebe_data.c                     phoebe_data.h                     \
    phoebe_error_handling.c           phoebe_error_handling.h           \
    phoebe_fitting.c                  phoebe_fitting.h                  \
    phoebe_fortran_interface.c        phoebe_fortran_interface.h        \
    phoebe_ld.c                       phoebe_ld.h                       \
    phoebe_nms.c                      phoebe_nms.h                      \
    phoebe_parameters.c               phoebe_parameters.h               \
    phoebe_radiation.c                phoebe_radiation.h                \
    phoebe_roche.c                    phoebe_roche.h                    \
    phoebe_spectra.c                  phoebe_spectra.h                  \
    phoebe_types.c                    phoebe_types.h                    \
                                      phoebe.h

BUILT_SOURCES =                                                         \
	phoebe_constraints.lex.c          phoebe_constraints.lex.h      \
	phoebe_constraints.tab.c          phoebe_constraints.tab.h

EXTRA_DIST = \
    phoebe_constraints.lex            phoebe_constraints.y

phoebe_constraints.tab.c: phoebe_constraints.y
	$(YACC) -d -p pc phoebe_constraints.y
phoebe_constraints.lex.c: phoebe_constraints.lex
	$(LEX) phoebe_constraints.lex

libphoebe_la_LDFLAGS = -release $(PACKAGE_VERSION)
libphoebe_la_FFLAGS  = -ff2c
libphoebe_la_CFLAGS  = -Wall

libphoebe_la_LIBADD  = ../libwd/libwd.la

library_includedir=$(includedir)/phoebe
library_include_HEADERS = phoebe.h                   \
                          phoebe_accessories.h       \
                          phoebe_allocations.h       \
                          phoebe_base.h              \
                          phoebe_calculations.h      \
                          phoebe_configuration.h     \
                          phoebe_constraints.h       \
                          phoebe_data.h              \
                          phoebe_error_handling.h    \
                          phoebe_fitting.h           \
                          phoebe_fortran_interface.h \
                          phoebe_global.h            \
                          phoebe_ld.h                \
                          phoebe_nms.h               \
                          phoebe_parameters.h        \
                          phoebe_roche.h             \
                          phoebe_spectra.h           \
                          phoebe_types.h
