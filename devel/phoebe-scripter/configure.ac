########################################################################
#                                                                      #
#                           PHOEBE SVN scripter                        #
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
#                                            Last revision: 2009-05-02 #
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
  AC_INIT(phoebe-scripter, [svn], http://phoebe.fiz.uni-lj.si)

# Create a dedicated header file phoebe_scripter_build_config.h that will
# contain all #define statements that correspond to whatever ./configure
# found:
  AC_CONFIG_HEADERS([src/phoebe_scripter_build_config.h])

# Define the release name, date and type:
  build_date=`date`

  AC_DEFINE(
	PHOEBE_SCRIPTER_RELEASE_NAME,
	[["PHOEBE scripter svn"]],
	[PHOEBE release name])
  AC_DEFINE_UNQUOTED(
	PHOEBE_SCRIPTER_RELEASE_DATE,
	[["(built on $build_date)"]],
	[PHOEBE release date])

# Initialize the location of the sources and auto* stuff:
  AC_CONFIG_SRCDIR([src/main.c])
  AC_CONFIG_AUX_DIR(autoconfig)
  AC_CONFIG_MACRO_DIR(autoconfig)

# Initialize configure.ac for automake:
  AM_INIT_AUTOMAKE([-Wall -Werror gnu])
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
    if $LEX --version | grep 2.5.[[3-9]][[0-9]] > /dev/null 2>&1; then
      flex_check_passed="yes"
    fi

  # Check for existence of all flexed sources:
    AC_SUBST([flex_built_sources],["phoebe_scripter.lng.c phoebe_scripter.lng.h"])

    flex_missing=""
    AC_MSG_CHECKING([for flexed PHOEBE scripter sources])
    for f in $flex_built_sources; do
      test -f src/$f && found=yes || found=no
      test "x$found" = "xno" && flex_missing="$flex_missing $f"
    done

    test -n "$flex_check_passed" && test -n "$flex_missing" && AC_MSG_RESULT([no, but may be built])
    test -z "$flex_check_passed" && test -n "$flex_missing" && AC_MSG_RESULT([no])

    test -z "$flex_check_passed" && test -n "$flex_missing" && AC_MSG_ERROR([
    built source file(s) missing in src/ directory:
      ${flex_missing}

    The flex program is required to build these source files, but it does
    not appear to be installed.
  ])

  test -z "$flex_missing" && AC_MSG_RESULT([yes])

  # Parser:
    AC_PROG_YACC
    if $YACC --version | grep bison > /dev/null 2>&1; then
      bison_check_passed="yes"
    fi

  # Check for existence of all bisoned sources:
    AC_SUBST([bison_built_sources],["phoebe_scripter.grm.c phoebe_scripter.grm.h"])

    bison_missing=""
    AC_MSG_CHECKING([for bisoned PHOEBE scripter sources])
    for f in $bison_built_sources; do
      test -f src/$f && found=yes || found=no
      test "x$found" = "xno" && bison_missing="$bison_missing $f"
    done

    test -n "$bison_check_passed" && test -n "$bison_missing" && AC_MSG_RESULT([no, but may be built])
    test -z "$bison_check_passed" && test -n "$bison_missing" && AC_MSG_RESULT([no])

    test -z "$bison_check_passed" && test -n "$bison_missing" && AC_MSG_ERROR([
    built source file(s) missing in src/ directory:
      ${bison_missing}

    The bison program is required to build these source files, but it does
    not appear to be installed.
    ])

    test -z "$bison_missing" && AC_MSG_RESULT([yes])

# Check if math library is installed:
  AC_CHECK_LIB(m, sin)

# Check if ltdl is installed:
  AC_CHECK_LIB(ltdl, lt_dlinit, , ltdl_missing=1)
    if test "x$ltdl_missing" = "x1"; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE scripter needs libtool's ltdl to be installed."
      echo "Please install it and rerun ./configure."
      echo ""
     exit
    fi

# Check if PHOEBE library is installed:

  AC_CHECK_LIB(phoebe, phoebe_init,, phoebe_missing=1)

    if test "x$phoebe_missing" = "x1"; then
      echo ""
      echo "       *** PHOEBE configuration failure ***"
      echo ""
      echo "PHOEBE scripter needs PHOEBE library installed to be built."
      echo "Please install it and rerun ./configure."
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

  if test $ac_enable_readline = "yes"; then
    # termcap is a requirement for readline, check for it first
    readline_check_passed=1
    readlinesupport="no"
    AC_CHECK_LIB(termcap, tgetstr, , readline_check_passed=0)
    if test $readline_check_passed -eq 0; then
      echo "NO LIB"
      readlinesupport="no"
    else 
      AC_CHECK_LIB(readline, rl_insert_text, , readline_check_passed=0)
      if test $readline_check_passed -eq 0; then
        readlinesupport="no"
      else      
        readlinesupport="yes"
      fi
    fi
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

# Recent versions of gcc have a -Wextra option for more verbose warnings
  extra_warn="-Wextra"
  echo -n 'checking for $(CC) -Wextra... '
  AC_LANG_CONFTEST([/* test for -Wextra */ int main(){}])
  if gcc $extra_warn conftest.c -o - ; then
    echo yes
  else
    echo no
    extra_warn=""
  fi
  
  AC_DEFINE(EXTRA_WARNING_COMPILER_FLAG, $extra_warn, [Set -Wextra if the c compiler recognizes it:])

# Optional docbook support:
# AC_PROG_XSLTPROC
# docbookstate="no"
#
# AC_MSG_CHECKING([docbook.xsl])
# handle --with-docbook-xsl=PATH option if invoked
# AC_ARG_WITH([docbook-xsl],
#             [AS_HELP_STRING([--with-docbook-xsl],
#               [Specify full path of docbook.xsl, or "no" to disable])],
#             [docbook_stylesheet=$with_docbook_xsl],
#             [])
#
# by default use docbook.xsl distributed with phoebe-scripter
# if test "x$docbook_stylesheet" = "x" ; then
#   docbookstate="yes"
#   docbook_stylesheet="$srcdir/docs/reference/xsl-stylesheets/xhtml/docbook.xsl"
# fi
#
# check for the docbook.xsl file
# if test  "x$docbook_stylesheet" != "xno"; then
#   if test -e  $docbook_stylesheet; then
#     docbookstate="yes" # docbook.xsl has been found
#
#   else # explicit path passed in but not found.
#     AC_MSG_WARN([docbook_stylesheet set to "$docbook_stylesheet" but file not found.  Skipping documentation.])
#   fi
#   else
#     docbook_stylesheet=""
# fi
#
# AC_SUBST([DOCBOOK_STYLESHEET], [$docbook_stylesheet])
# AM_CONDITIONAL([XSLTPROC_FOUND], [test "x$docbookstate" = "xyes"])
# AC_MSG_RESULT([$docbookstate]);

# Optional plugin support:
  AC_LIBTOOL_DLOPEN
  AC_PROG_LIBTOOL

# Check for C library features:
  AC_CHECK_HEADERS([inttypes.h limits.h stdlib.h string.h unistd.h])
  AC_HEADER_STDBOOL
  AC_HEADER_STDC

  AC_TYPE_SIZE_T

  AC_C_CONST
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
  echo "Maintainer mode (build sources):  $USE_MAINTAINER_MODE"
  echo "Top-level install directory:      $prefix"
  echo "Scripter binary goes to:          $prefix/bin"
  echo "Scripter help files go to:        $prefix/share/phoebe_scripter"
  echo "Plugins go to:                    $prefix/lib"
  echo "Supported plotting packages:      $plotters"
  echo "GNU readline library support:     $readlinesupport"
# echo "Build API documentation:          $docbookstate"
# echo "  Docbook XSL:                    $docbook_stylesheet"
  echo "Verbose debugging messages:       $debugstate"
  echo ""

if test -z "$flex_check_passed" || test -z "$bison_check_passed"; then
  echo "Notice:"
  echo ""
fi

if test -z "$flex_check_passed"; then
  echo "Flex 2.5.33 or later does not appear to be installed; it is required"
  echo "to rebuild PHOEBE lexical source files that are distributed in this"
  echo "tarball. If you modify the flex sources in the src/ directory, you"
  echo "will be unable to build PHOEBE without first installing flex."
  echo ""
fi

if test -z "$bison_check_passed"; then
  echo "Bison 1.9 or later does not appear to be installed; it is required"
  echo "to rebuild PHOEBE grammar source files that are distributed in this"
  echo "tarball. If you modify the bison sources in the src/ directory, you"
  echo "will be unable to build PHOEBE without first installing bison."
  echo ""
fi
