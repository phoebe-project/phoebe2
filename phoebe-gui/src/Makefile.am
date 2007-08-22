INCLUDES = \
	-DPACKAGE_DATA_DIR=\""$(datadir)"\" \
	-DPACKAGE_LOCALE_DIR=\""$(prefix)/$(DATADIRNAME)/locale"\" \
	@PACKAGE_CFLAGS@ -I/usr/include/libglade-2.0

bin_PROGRAMS = phoebe_gui

phoebe_gui_SOURCES =                               \
	phoebe_gui_main.h       phoebe_gui_main.c       \
	phoebe_gui_callbacks.h  phoebe_gui_callbacks.c  \
	phoebe_gui_types.h	    phoebe_gui_types.c     \
	phoebe_gui_base.h	    phoebe_gui_base.c       	\
	phoebe_gui_treeviews.h  phoebe_gui_treeviews.c	\
	phoebe_gui_accessories.h  phoebe_gui_accessories.c	\
	phoebe_gui_global.h

phoebe_gui_CFLAGS = -Wall -g
phoebe_gui_LDADD = @PACKAGE_LIBS@ -lphoebe
phoebe_gui_LDFLAGS = -export-dynamic

EXTRA_DIST = phoebe.glade
