#!/usr/bin/python

import gtk
#import matplotlib as mpl
#mpl.use ('GTKAgg')

#import os
#import numpy as np
#import matplotlib.pyplot as mplplt
#from matplotlib.axes import Axes as mplaxes
#from matplotlib.figure import Figure as mplfig
#from matplotlib.widgets import RectangleSelector as selector
#from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as mpl2gtk

import phoebeBackend as backend
from phoebe_callbacks import *
from phoebe_options import *

class PhoebeParameter:
	def __init__(self, qualifier):
		prop = backend.parameter(qualifier)
		if prop == None:
			return
		
		self.qualifier   = prop[0]
		self.description = prop[1]
		self.kind        = prop[2]
		self.format      = prop[3]
		self.minval      = prop[4]
		self.maxval      = prop[5]
		self.step        = prop[6]
		self.value       = prop[7]
		
		return

	def __str__(self):
		info = "\n"
		info += "Description:    %s\n" % self.description
		info += "Qualifier:      %s\n" % self.qualifier
		info += "Value:          %s\n" % self.value
		return info

class PhoebeGUI:
	def __init__(self):
		builder = gtk.Builder ()
		builder.add_from_file ("glade/phoebe_py.glade")
		self.window = builder.get_object ("phoebe_window")
		self.window.show ()
		builder.connect_signals (self)
		
		phoebe_add_options (builder)
		
		gtk.main ()
	
	# CALLBACKS:
	
	def on_phoebe_quit_event (self, widget, event=None, data=None):
		return on_phoebe_quit()

	
if __name__ == "__main__":
	backend.init()
	backend.configure()

	PhoebeName = PhoebeParameter ("phoebe_lc_filename")
	print PhoebeName

	phoebe = PhoebeGUI()
