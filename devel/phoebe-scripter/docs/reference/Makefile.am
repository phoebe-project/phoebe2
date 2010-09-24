########################################################################
#                                                                      #
#                  PHOEBE scripter Reference Manual                    #
#                                                                      #
########################################################################
#                                                                      #
#    Author of Makefile.am: Andrej Prsa <andrej.prsa@villanova.edu>    #
#                                                                      #
#                                            Last revision: 2008-06-24 #
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

scripterdocdir = $(datarootdir)/phoebe_scripter/docs/reference
scripterdoc_DATA = api.xml         \
                   intro.xml       \
		   general.xml     \
		   identifiers.xml

if XSLTPROC_FOUND
scripterdoc_DATA += api.html
endif

BUILT_SOURCES = api.html
CLEANFILES    = api.html

# configure determines stylesheet location relative to top of src dir -
# if this doc directory changes this will need to be changed as well
DOCBOOK_STYLESHEET = ../../@DOCBOOK_STYLESHEET@
api.html:
	if test ! "x@XSLTPROC@" = "x"; then \
		xsltproc -o api.html $(DOCBOOK_STYLESHEET) $(top_srcdir)/docs/reference/api.xml; \
	fi

EXTRA_DIST = $(scripterdoc_DATA)

