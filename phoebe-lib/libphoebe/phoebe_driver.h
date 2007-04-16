#ifndef PHOEBE_DRIVER_H
	#define PHOEBE_DRIVER_H 1

int wd_sincos (int star, int N, int N1, double *snth, double *csth, double *snfi, double *csfi, int *mmsave);
int wd_binnum (double *abunarr, int abundim, double abunval, int *abunidx);

#endif
