########################################################################
#                                                                      #
#                          PHOEBE SVN library                          #
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
#                                            Last revision: 2008-06-11 #
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
  AC_INIT([phoebe-lib],[svn],[phoebe-devel@lists.sourceforge.net])

# Create a dedicated header file phoebe_build_config.h that will contain
# all #define statements that correspond to whatever ./configure found:
  AC_CONFIG_HEADERS([libphoebe/phoebe_build_config.h])

# Initialize the location of the sources:
  AC_CONFIG_SRCDIR([libphoebe/phoebe_base.c])

# Initialize the location of the autoconf macros:
  AC_CONFIG_AUX_DIR(autoconfig)

# Initialize the location of local autoconf macros:
  AC_CONFIG_MACRO_DIR(autoconfig)

# Pass the following options to automake and enable maintainer mode:
  AM_INIT_AUTOMAKE([-Wall -Werror gnu])
  AM_MAINTAINER_MODE

# Get the system type:
  AC_CANONICAL_BUILD

# Define the release name, date and type:
  AC_DEFINE(
	PHOEBE_RELEASE_NAME,
	[["PHOEBE svn"]],
	[PHOEBE release name])
  AC_DEFINE(
	PHOEBE_RELEASE_DATE,
	[["(snapshot)"]],
	[PHOEBE release date])

# Define the default installation directory:
  if test "x$prefix" = "xNONE"; then
    AC_DEFINE_UNQUOTED(
		PHOEBE_TOP_DIR,
		["/usr/local"],
		[Top-level install directory (as set by ./configure)])
  else
    AC_DEFINE_UNQUOTED(
		PHOEBE_TOP_DIR,
		["$prefix"],
		[Top-level install directory (as set by ./configure)])
  fi

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

  # Fortran compiler:
    AC_PROG_F77([gfortran g77])

    if test -z $F77; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE needs a F77 compliant compiler (gfortran) to build."
      echo "Please install it and run the ./configure script again."
      echo ""
      exit
    fi

  # Get $FLIBS for mixing Fortran with C:
  # (see http://www.gnu.org/software/automake/manual/automake.html#Mixing-Fortran-77-With-C-and-C_002b_002b)
    AC_F77_LIBRARY_LDFLAGS

  # Libtool:
    AC_PROG_LIBTOOL

  # Make program:
    AC_PROG_MAKE_SET

  # Install program:
    AC_PROG_INSTALL

  # Lexer (needed for PHOEBE constraints); we require flex 2.5.33 or later:
    AC_PROG_LEX
    if $LEX --version | grep 2.5.[[3-9]][[0-9]] > /dev/null 2>&1; then
      flex_check_passed="yes"
    fi

  # Check for existence of all flexed sources:
    AC_SUBST([flex_built_sources],["phoebe_constraints.lng.c phoebe_constraints.lng.h"])

    flex_missing=""
    AC_MSG_CHECKING([for flexed PHOEBE sources])
    for f in $flex_built_sources; do
      test -f libphoebe/$f && found=yes || found=no
      test "x$found" = "xno" && flex_missing="$flex_missing $f"
    done

    test -n "$flex_check_passed" && test -n "$flex_missing" && AC_MSG_RESULT([no, but may be built])
    test -z "$flex_check_passed" && test -n "$flex_missing" && AC_MSG_RESULT([no])

    test -z "$flex_check_passed" && test -n "$flex_missing" && AC_MSG_ERROR([
    built source file(s) missing in libphoebe/ directory:
      ${flex_missing}

    The flex program is required to build these source files, but it does
    not appear to be installed.
    ])

    test -z "$flex_missing" && AC_MSG_RESULT([yes])

  # Parser (needed for PHOEBE constraints):
    AC_PROG_YACC
    if $YACC --version | grep bison > /dev/null 2>&1; then
      bison_check_passed="yes"
    fi

  # Check for existence of all bisoned sources:
    AC_SUBST([bison_built_sources],["phoebe_constraints.grm.c phoebe_constraints.grm.h"])

    bison_missing=""
    AC_MSG_CHECKING([for bisoned PHOEBE sources])
    for f in $bison_built_sources; do
      test -f libphoebe/$f && found=yes || found=no
      test "x$found" = "xno" && bison_missing="$bison_missing $f"
    done

    test -n "$bison_check_passed" && test -n "$bison_missing" && AC_MSG_RESULT([no, but may be built])
    test -z "$bison_check_passed" && test -n "$bison_missing" && AC_MSG_RESULT([no])

    test -z "$bison_check_passed" && test -n "$bison_missing" && AC_MSG_ERROR([
    built source file(s) missing in libphoebe/ directory:
      ${bison_missing}

    The bison program is required to build these source files, but it does
    not appear to be installed.
    ])

    test -z "$bison_missing" && AC_MSG_RESULT([yes])

  # System type:
    AC_DEFINE_UNQUOTED(PHOEBE_SYSTEM_TYPE,"$build",[System type])

  # Math library:
    AC_CHECK_LIB([m],[cos])

  # Functions in linux libc, but not windows libc:
    AC_CHECK_FUNCS([geteuid getegid])
    AC_CHECK_HEADERS([ftw.h])

    ACX_PTHREAD([pthreadsupport="yes"],[pthreadsupport="no"])

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

# Recent versions of gcc have a -Wextra option for more verbose warnings
  extra_warn="-Wextra"
  echo -n 'checking for $(CC) -Wextra... '
  AC_LANG_CONFTEST([/* test for -Wextra */ int main(){}])
  if gcc $extra_warn conftest.c -o -; then
    echo yes
  else
    echo no
    extra_warn=""
  fi
  
  AC_DEFINE(EXTRA_WARNING_COMPILER_FLAG, $extra_warn, [Set -Wextra if the c compiler recognizes it:])

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
  echo "Fortran compiler used:            $F77"
  echo "PHOEBE library directory:         $prefix/lib"
  echo "PHOEBE include directory:         $prefix/include/phoebe"
  echo "POSIX threading support:          $pthreadsupport"
  echo "GNU Scientific Library support:   $gslsupport"
  echo "Build API documentation:          $gtkdocstate"
  echo "Maintainer mode (build sources):  $USE_MAINTAINER_MODE"
  echo "Verbose debugging messages:       $debugstate"
  echo ""

if test -z "$flex_check_passed" || test -z "$bison_check_passed"; then
  echo "Notice(s):"
  echo ""
fi

if test -z "$flex_check_passed"; then
  echo "Flex 2.5.33 or later does not appear to be installed; it is required"
  echo "to rebuild PHOEBE lexical source files that are distributed in this"
  echo "tarball. If you modify the flex sources in libphoebe/ directory, you"
  echo "will be unable to build PHOEBE without first installing flex."
  echo ""
fi

if test -z "$bison_check_passed"; then
  echo "Bison 1.9 or later does not appear to be installed; it is required"
  echo "to rebuild PHOEBE grammar source files that are distributed in this"
  echo "tarball. If you modify the bison sources in libphoebe/ directory, you"
  echo "will be unable to build PHOEBE without first installing bison."
  echo ""
fi
