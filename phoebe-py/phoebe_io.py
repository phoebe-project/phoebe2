import gtk

def phoebe_question (message):
	"""
	Creates a question dialog.
	"""
	return gtk.MessageDialog (flags=gtk.DIALOG_MODAL, type=gtk.MESSAGE_QUESTION, buttons=gtk.BUTTONS_OK_CANCEL, message_format=message)
