########################################################################
#                                                                      #
#                          PHOEBE 0.31 library                         #
#                                                                      #
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

SUBDIRS = libwd libphoebe ptf wd docs/reference

# Enable gtk-doc during make distcheck:
DISTCHECK_CONFIGURE_FLAGS=--enable-gtk-doc

ACLOCAL_AMFLAGS=-I autoconfig

EXTRA_DIST = autoconfig/compile                      \
             autoconfig/config.guess                 \
             autoconfig/config.sub                   \
             autoconfig/depcomp                      \
             autoconfig/install-sh                   \
             autoconfig/ltmain.sh                    \
             autoconfig/missing
