AM_CPPFLAGS = \
	@GTK_CFLAGS@ @LIBGLADE_CFLAGS@ \
	-DPACKAGE_DATA_DIR=\""$(datadir)"\" \
	-DPACKAGE_LOCALE_DIR=\""$(prefix)/$(DATADIRNAME)/locale"\"

bin_PROGRAMS = phoebe-gui

phoebe_gui_SOURCES =                                                    \
	phoebe_gui_main.h               phoebe_gui_main.c               \
	phoebe_gui_callbacks.h          phoebe_gui_callbacks.c          \
	phoebe_gui_types.h              phoebe_gui_types.c              \
	phoebe_gui_base.h               phoebe_gui_base.c               \
	phoebe_gui_treeviews.h          phoebe_gui_treeviews.c          \
	phoebe_gui_accessories.h        phoebe_gui_accessories.c        \
	phoebe_gui_plotting.h           phoebe_gui_plotting.c           \
	phoebe_gui_error_handling.h     phoebe_gui_error_handling.c     \
	phoebe_gui_global.h

AM_CFLAGS = -Wall $(EXTRA_WARNING_COMPILER_FLAG) -g \
	-DGLADE_XML_DIR=\"$(prefix)/share/phoebe/gui/glade\" \
	-DGLADE_PIXMAP_DIR=\"$(prefix)/share/phoebe/gui/pixmaps\"

phoebe_gui_LDADD = @GTK_LIBS@ @LIBGLADE_LIBS@
phoebe_gui_LDFLAGS = -export-dynamic
