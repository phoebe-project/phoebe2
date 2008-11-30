########################################################################
#                                                                      #
#                          PHOEBE SVN plugins                          #
#                                                                      #
########################################################################
#                                                                      #
#   Author of configure.ac: Andrej Prsa <andrej.prsa@villanova.edu>    #
#                                                                      #
#                               * * *                                  #
#                                                                      #
#  When you change configure.ac, don't forget to run auto(re)conf for  #
#                     changes to take effect!                          #
#                                                                      #
#                               * * *                                  #
#                                            Last revision: 2008-09-22 #
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

# This configure.ac will work with autoconf 2.50 or later:
  AC_PREREQ(2.50)

# Initialize configure.ac for autoconf:
  AC_INIT(phoebe-plugins, [svn], http://phoebe.fiz.uni-lj.si)

# Create a dedicated header file phoebe_scripter_build_config.h that will
# contain all #define statements that correspond to whatever ./configure
# found:
  AC_CONFIG_HEADERS([phoebe_plugins_build_config.h])

# Define the release name, date and type:
  build_date=`date`

  AC_DEFINE(
	PHOEBE_PLUGINS_RELEASE_NAME,
	[["PHOEBE plugins svn"]],
	[PHOEBE release name])
  AC_DEFINE_UNQUOTED(
	PHOEBE_PLUGINS_RELEASE_DATE,
	[["(built on $build_date)"]],
	[PHOEBE release date])

# Initialize the location of the sources and auto* stuff:
  AC_CONFIG_SRCDIR([polyfit/polyfit.c])
  AC_CONFIG_AUX_DIR(autoconfig)
  AC_CONFIG_MACRO_DIR(autoconfig)

# Initialize configure.ac for automake:
  AM_INIT_AUTOMAKE([-Wall -Werror gnu])
  AM_MAINTAINER_MODE

# Libtool:
  AC_PROG_LIBTOOL

# Check for system programs:

  # C compiler:
    AC_PROG_CC

    if test -z $CC; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs an ANSI-compliant C compiler to build."
      echo "Please install it and run the ./configure script again."
      echo ""
      exit
    fi

# Check if PHOEBE library is installed:

  AC_CHECK_LIB(phoebe, phoebe_init, phoebe_check=1, phoebe_check=0)

    if test $phoebe_check -eq 0; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE scripter needs PHOEBE library installed to be built."
      echo "Remember, the scripter is only a driver for the library!"
      echo ""
     exit
    fi

# Check if PHOEBE scripter is installed:

  # Make program:
    AC_PROG_MAKE_SET

  # Install program:
    AC_PROG_INSTALL

# Optional debugging support:
  debugstate="no"
  AC_ARG_ENABLE(debug,
    [  --enable-debug          Use verbose debugging mode (default=no)],
    [ac_enable_debug=yes], [ac_enable_debug=no])
  if test $ac_enable_debug = "yes"; then
  AC_DEFINE(PHOEBE_DEBUG_SUPPORT, 1, Define if verbose debugging is required:)
  debugstate="yes"
  fi

# Check for C library features:
  AC_CHECK_HEADERS([stdlib.h string.h unistd.h])
  AC_HEADER_STDBOOL
  AC_HEADER_STDC

  AC_TYPE_SIZE_T

  AC_C_CONST
  AC_FUNC_MALLOC
  AC_FUNC_REALLOC
  AC_FUNC_VPRINTF

  AC_CHECK_FUNCS([getcwd memset])
  AC_CHECK_FUNCS([strchr strdup])

# Finally, write everything to Makefiles:
  AC_CONFIG_FILES([
                  Makefile
                  polyfit/Makefile
                  ])

# Conclude:
  AC_OUTPUT

# Write out the configuration summary:
  echo ""
  echo "PHOEBE plugins configuration complete."
  echo ""
  echo "Configuration summary for $PACKAGE_STRING:"
  echo "-----------------------------------------------------------------"
  echo "C compiler used:                  $CC"
  echo "Maintainer mode (build sources):  $USE_MAINTAINER_MODE"
  echo "Top-level install directory:      $prefix"
  echo "Verbose debugging messages:       $debugstate"
  echo ""

