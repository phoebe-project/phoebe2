########################################################################
#                                                                      #
#                          PHOEBE 0.30 library                         #
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
#                                            Last revision: 2007-09-14 #
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
  AC_INIT(PHOEBE,[0.30],[andrej.prsa@fmf.uni-lj.si],[phoebe])

# Create a dedicated header file phoebe_build_config.h that will contain
# all #define statements that correspond to whatever ./configure found:
  AC_CONFIG_HEADERS([libphoebe/phoebe_build_config.h])

# Initialize the location of the sources:
  AC_CONFIG_SRCDIR([libphoebe/phoebe_base.c])

# Initialize the location of the autoconf macros:
  AC_CONFIG_AUX_DIR(autoconfig)

# Initialize the location of local autoconf macros:
  AC_CONFIG_MACRO_DIR(autoconfig)

# Initialize configure.ac for automake:
  AM_INIT_AUTOMAKE(phoebe,0.30)
  AM_MAINTAINER_MODE

# Get the system type:
  AC_CANONICAL_BUILD

# Define the release name, date and type:
  AC_DEFINE(
	PHOEBE_RELEASE_NAME,
	[["PHOEBE 0.30RC1 (release candidate)"]],
	[PHOEBE release name])
  AC_DEFINE(
	PHOEBE_RELEASE_DATE,
	[["Oct 15, 2007"]],
	[PHOEBE release date])

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

  # FORTRAN compiler:
    AC_PROG_F77

    if test -z $F77; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs an F77-compliant FORTRAN compiler to build."
      echo "Please install it and run the ./configure script again."
      echo ""
      exit
    fi

  # Libtool:
    AC_PROG_LIBTOOL

  # Make program:
    AC_PROG_MAKE_SET

  # Install program:
    AC_PROG_INSTALL

  # f2c program:
    AC_CHECK_PROG(f2c_check_passed,f2c,yes,no)
    if test "x$f2c_check_passed" = "xno"; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs a FORTRAN-to-C converter f2c to build."
      echo "Please install it and run ./configure script again."
      echo ""
      exit
    fi

  # g2c library:
    g2c_check_passed=yes
    AC_CHECK_LIB(g2c,main,,g2c_check_passed=no)
    if test "x$g2c_check_passed" = "xno"; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs a libg2c library (part of g77) to build."
      echo "Please install it and run ./configure script again."
      echo ""
      exit
    fi

  # Lexer (needed for PHOEBE constraints); we require flex 2.5.33 or later:
    AC_PROG_LEX
    if test "$LEX" != flex -a "$LEX" != flex-new; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs the GNU flex 2.5.33 or later to build."
      echo "Please install it and run ./configure script again."
      echo ""
      exit
    fi

    # Check for 2.5.33 version:
    if ! $LEX --version | grep 2.5.[[3-9]][[0-9]] > /dev/null 2>&1; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs GNU flex 2.5.33 or later to build."
      echo "Please install it and run ./configure script again."
      echo ""
      exit
    fi

  # Parser (needed for PHOEBE constraints):
  # (here we don't call AC_PROG_YACC because we don't want -y attached to bison)
    AC_CHECK_PROGS(YACC, 'bison' byacc yacc, :)

    if test "$YACC" == :; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs a yacc-compliant grammar parser to build."
      echo "Please install it and run ./configure script again."
      echo ""
      exit
    fi

  # System type:
    AC_DEFINE_UNQUOTED(PHOEBE_SYSTEM_TYPE,"$build",[System type])

  # GSL support:
    AC_ARG_ENABLE(gsl,
        [  --enable-gsl            Use GNU Scientific Library (default=yes)],
        [ac_enable_gsl=no], [ac_enable_gsl=yes])
    gsl_check_passed=1
    AC_CHECK_LIB(gslcblas, main, , gsl_check_passed=0)
    AC_CHECK_LIB(gsl,      main, , gsl_check_passed=0)
    gslsupport="yes"
    if test $gsl_check_passed -eq 0; then
      gslsupport="no"
    fi
    if test $ac_enable_gsl = "no"; then
      gslsupport="no"
      gslsupport="disabled"
      AC_DEFINE(PHOEBE_GSL_DISABLED, 1, Define if GSL should be disabled:)
    fi

  # gtk-doc support; sorry for the silly formatting, gtkdocize requires it:
    gtkdocstate="no"
GTK_DOC_CHECK(1.0)
    if test "$enable_gtk_doc" == "yes"; then
      gtkdocstate="yes"
    fi

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
  AC_HEADER_DIRENT
  AC_HEADER_STDC
  AC_CHECK_HEADERS([locale.h stddef.h stdlib.h string.h unistd.h])
  AC_HEADER_STDBOOL
  AC_C_CONST
  AC_TYPE_SIZE_T
  AC_FUNC_CLOSEDIR_VOID
  AC_FUNC_MALLOC
  AC_FUNC_REALLOC
  AC_FUNC_STAT
  AC_FUNC_VPRINTF
  AC_CHECK_FUNCS([getcwd mkdir mkfifo pow setlocale sqrt strchr strdup strstr])

# Check for FORTRAN library features:

  # Make sure FORTRAN internal libraries will be available for linking:
  AC_F77_LIBRARY_LDFLAGS

  # Define a switch to tell FORTRAN what architecture it is compiling for:
  AC_DEFINE(f2cFortran, 1, [This is for Linux x86 architectures:])

# Check for OS features:

  # Are symbolic links allowed:
    AC_PROG_LN_S

# Finally, write everything to Makefiles:
  AC_CONFIG_FILES([
                  Makefile
                  libwd/Makefile
                  libphoebe/Makefile
                  ptf/Makefile
                  wd/Makefile
                  docs/reference/Makefile
                  ])

# Conclude:
  AC_OUTPUT

# Write out the configuration summary:
  echo ""
  echo "PHOEBE library configuration complete."
  echo ""
  echo "Configuration summary for $PACKAGE_STRING:"
  echo "-----------------------------------------------------------"
  echo "System type:                      $build"
  echo "C compiler used:                  $CC"
  echo "FORTRAN compiler used:            $F77"
  echo "Grammar lexer used:               $LEX"
  echo "Grammar parser used:              $YACC"
  echo "PHOEBE library directory:         $prefix/lib"
  echo "PHOEBE include directory:         $prefix/include/phoebe"
  echo "GNU Scientific Library support:   $gslsupport"
  echo "Build API documentation:          $gtkdocstate"
  echo "Verbose debugging messages:       $debugstate"
  echo ""
