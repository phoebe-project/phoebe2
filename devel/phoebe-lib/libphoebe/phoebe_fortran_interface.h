#ifndef PHOEBE_FORTRAN_INTERFACE_H
	#define PHOEBE_FORTRAN_INTERFACE_H 1

#include "phoebe_global.h"

int create_lci_file (char filename[], WD_LCI_parameters param);
int create_dci_file (char *filename, void *pars);

#endif
