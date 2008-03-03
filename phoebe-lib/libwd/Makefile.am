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
    atmx.f      bbl.f       binnum.f    cloud.f     conjph.f    dgmprd.f       \
    dminv.f     dura.f      ellone.f    fourls.f    gabs.f      jdph.f         \
    kepler.f    lcr.f       legendre.f  light.f     linpro.f    lum.f          \
    lump.f      mlrg.f      modlog.f    nekmin.f    olump.f     planckint.f    \
    rangau.f    ranuni.f    rddata.f    ring.f      romq.f      sincos.f       \
    spot.f      square.f    surfas.f    volume.f    wrhead.f    wrdata.f       \
    wrfoot.f    wrlci.f     wrdci.f     lc.sub.f    dc.sub.f

#EXTRA_DIST =

#CLEANFILES = 

libwd_la_LDFLAGS = -release 2007-08-15
#libwd_la_FFLAGS  =
#libwd_la_CFLAGS  =

library_includedir=$(includedir)/phoebe
library_include_HEADERS = wd.h
