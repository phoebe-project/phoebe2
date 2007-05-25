########################################################################
#                                                                      #
#                               WD library                             #
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

lib_LTLIBRARIES = libwd.la

libwd_la_SOURCES = \
    $(BUILT_SOURCES)

SOURCES = \
    atmx.f      bbl.f       binnum.f    cloud.f     conjph.f    dgmprd.f       \
    dminv.f     dura.f      ellone.f    fourls.f    gabs.f      jdph.f         \
    kepler.f    lcr.f       legendre.f  light.f     linpro.f    lum.f          \
    lump.f      mlrg.f      modlog.f    nekmin.f    olump.f     planckint.f    \
    rangau.f    ranuni.f    rddata.f    ring.f      romq.f      sincos.f       \
    spot.f      square.f    surfas.f    volume.f    wrhead.f    wrdata.f       \
    wrfoot.f    lc.sub.f

BUILT_SOURCES = \
    atmx.c      bbl.c       binnum.c    cloud.c     conjph.c    dgmprd.c       \
    dminv.c     dura.c      ellone.c    fourls.c    gabs.c      jdph.c         \
    kepler.c    lcr.c       legendre.c  light.c     linpro.c    lum.c          \
    lump.c      mlrg.c      modlog.c    nekmin.c    olump.c     planckint.c    \
    rangau.c    ranuni.c    rddata.c    ring.c      romq.c      sincos.c       \
    spot.c      square.c    surfas.c    volume.c    wrhead.c    wrdata.c       \
    wrfoot.c    lc.sub.c

EXTRA_DIST = \
	$(SOURCES)

$(libwd_la_SOURCES):
	f2c $(SOURCES)

CLEANFILES = \
	$(BUILT_SOURCES)

libwd_la_LDFLAGS = -release 2007-05-24
#libwd_la_FFLAGS  =
#libwd_la_CFLAGS  =
libwd_la_LIBADD = -lg2c

library_includedir=$(includedir)/phoebe
library_include_HEADERS = wd.h
