import gtk

def phoebe_add_options (builder):
	# PHOEBE model:
	model = builder.get_object ("phoebe_model_options")
	options = [
		"X-ray binary",
		"Unconstrained binary system",
		"Overcontact binary of the W UMa type",
		"Detached binary",
		"Overcontact binary not in thermal contact",
		"Semi-detached binary, primary star fills Roche lobe",
		"Semi-detached binary, secondary star fills Roche lobe",
		"Double contact binary"
	]

	for i in range(0, len(options)):
		row = model.append()
		model.set_value (row, 0, options[i])
	
	builder.get_object("phoebe_data_star_model_combobox").set_active (3)

	return
