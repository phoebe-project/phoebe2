########################################################################
#                                                                      #
#                             PHOEBE SVN GUI                           #
#                                                                      #
########################################################################
#                                                                      #
#    Author of Makefile.am: Andrej Prsa <andrej.prsa@villanova.edu>    #
#                                                                      #
########################################################################
#                                            Last revision: 2009-03-15 #
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

SUBDIRS = glade pixmaps src

mimedir = $(datarootdir)/mime/packages
mime_DATA = phoebe-gui.xml

mimeinfodir = $(datarootdir)/mime-info
mimeinfo_DATA = phoebe-gui.mime

desktopdir = $(datarootdir)/applications
desktop_DATA = phoebe-gui.desktop

EXTRA_DIST = autoconfig/depcomp                      \
             autoconfig/install-sh                   \
             autoconfig/missing                      \
             phoebe-gui.xml                          \
             phoebe-gui.mime                         \
             phoebe-gui.desktop                      \
             src/version.h

#install-data-hook:
#	if test "x@have_update_mime_database@" = "xyes"; then \
#	update-mime-database $(datarootdir)/mime; \
#	fi
#	if test "x@have_update_desktop_database@" = "xyes"; then \
#	update-desktop-database; \
#	fi

