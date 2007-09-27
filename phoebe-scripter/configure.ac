########################################################################
#                                                                      #
#                         PHOEBE 0.30RC1 scripter                      #
#                                                                      #
########################################################################
#                                                                      #
#   Author of configure.ac: Andrej Prsa <andrej.prsa@fmf.uni-lj.si>    #
#                                                                      #
#                               * * *                                  #
#                                                                      #
#  When you change configure.ac, don't forget to run auto(re)conf for  #
#                     changes to take effect!                          #
#                                                                      #
#                               * * *                                  #
#                                            Last revision: 2007-09-26 #
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
  AC_INIT(phoebe_scripter, 0.30, http://phoebe.fiz.uni-lj.si)

# Create a dedicated header file phoebe_scripter_build_config.h that will
# contain all #define statements that correspond to whatever ./configure
# found:
  AC_CONFIG_HEADERS([src/phoebe_scripter_build_config.h])

# Define the release name, date and type:
  AC_DEFINE(
	PHOEBE_SCRIPTER_RELEASE_NAME,
	[["PHOEBE scripter 0.30RC1 (release candidate)"]],
	[PHOEBE release name])
  AC_DEFINE(
	PHOEBE_SCRIPTER_RELEASE_DATE,
	[["Oct 15, 2007"]],
	[PHOEBE release date])

# Initialize the location of the sources:
  AC_CONFIG_SRCDIR([src/main.c])

# Initialize the location of the autoconf macros:
  AC_CONFIG_AUX_DIR(autoconfig)

# Initialize configure.ac for automake:
  AM_INIT_AUTOMAKE(phoebe,0.30RC1)
  AM_MAINTAINER_MODE

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

  # Lexer (needed for PHOEBE constraints); we require flex 2.5.33 or later:
    AC_PROG_LEX
    if test "$LEX" != flex; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs the GNU flex 2.5.33 or later to build."
      echo "Please install it and run ./configure script again."
      echo ""
      exit
    fi

    # Check for 2.5.33 version:
    if ! flex --version | grep 2.5.[[3-9]][[0-9]] > /dev/null 2>&1; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs GNU flex 2.5.33 or later to build."
      echo "Please install it and run ./configure script again."
      echo ""
      exit
    fi

  # Parser:
  # (here we don't call AC_PROG_YACC because we don't want -y attached to bison)
    AC_CHECK_PROGS(YACC, 'bison' byacc yacc, :)

    if test "$YACC" == :; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs a yacc-compliant grammar parser to build."
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

# GNUPlot support:
  AC_ARG_ENABLE(gnuplot,
      [  --enable-gnuplot        Enable gnuplot for drawing graphs (default=yes)],
      [ac_enable_gnuplot=yes], [ac_enable_gnuplot=yes])
  AC_DEFUN([AC_PROG_GNUPLOT], [AC_CHECK_PROG(GNUPLOT,gnuplot,yes)])
  AC_PROG_GNUPLOT
  if test "$GNUPLOT" == yes; then
    AC_DEFINE(PHOEBE_GNUPLOT_SUPPORT, 1, Define if GNUPlot is supported:)
    gnuplot="gnuplot"
  fi

# Concatenate all supporting plotting programs into $plotters:
  plotters=$gnuplot
  if test x$plotters = x; then
    plotters="none"
  fi

# Readline support for scripter line editing capability:

  AC_ARG_ENABLE(readline,
  [  --enable-readline       Use GNU readline (default=yes)],
  [ac_enable_readline=$enableval], [ac_enable_readline=yes])

  readline_check_passed=1
  AC_CHECK_LIB(readline, main, , readline_check_passed=0)

  readlinesupport="yes"
  if test $readline_check_passed -eq 0; then
  readlinesupport="no"
  fi

  if test $ac_enable_readline = "no"; then
    readlinesupport="disabled"
    AC_DEFINE(PHOEBE_READLINE_DISABLED, 1, Define if GNU readline should be disabled:)
  fi
    
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
  AC_CHECK_HEADERS([inttypes.h limits.h stdlib.h string.h unistd.h])
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
                  help/Makefile
                  src/Makefile
                  ])

# Conclude:
  AC_OUTPUT

# Write out the configuration summary:
  echo ""
  echo "PHOEBE scripter configuration complete."
  echo ""
  echo "Configuration summary for $PACKAGE_STRING:"
  echo "-----------------------------------------------------------------"
  echo "C compiler used:                  $CC"
  echo "Grammar lexer used:               $LEX"
  echo "Grammar parser used:              $YACC"
  echo "Top-level install directory:      $prefix"
  echo "Scripter binary goes to:          $prefix/bin"
  echo "Scripter help files go to:        $prefix/share/phoebe_scripter"
  echo "Supported plotting packages:      $plotters"
  echo "GNU readline library present:     $readlinesupport"
  echo "Verbose debugging messages:       $debugstate"
  echo ""
