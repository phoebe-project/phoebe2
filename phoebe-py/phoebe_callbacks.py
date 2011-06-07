import gtk

from phoebe_io import *

def on_phoebe_quit():
	dialog = phoebe_question ("By quitting PHOEBE, all changes will be lost. Continue?")
	response = dialog.run()
	if response == gtk.RESPONSE_OK:
		dialog.destroy()
		gtk.main_quit()
	
	dialog.destroy()
	return True
