#include <stdlib.h>

#include "phoebe_build_config.h"
#include "cfortran.h"

#include "phoebe_driver.h"
#include "phoebe_error_handling.h"

#define WD_SINCOS(KOMP,N,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE) \
        CCALLSFSUB8(SINCOS,sincos,INT,INT,INT,DOUBLEV,DOUBLEV,DOUBLEV,DOUBLEV,INTV,KOMP,N,N1,SNTH,CSTH,SNFI,CSFI,MMSAVE)

#define WD_BINNUM(ABUNARR,ABUNDIM,ABUNVAL,ABUNIDX) \
        CCALLSFSUB4(BINNUM,binnum,DOUBLEV,INT,DOUBLE,INTV,ABUNARR,ABUNDIM,ABUNVAL,ABUNIDX)

int wd_sincos (int star, int N, int N1, double *snth, double *csth, double *snfi, double *csfi, int *mmsave)
{
	snth   = calloc (260,  sizeof (*snth));
	csth   = calloc (260,  sizeof (*csth));
	snfi   = calloc (6400, sizeof (*snfi));
	csfi   = calloc (6400, sizeof (*csfi));
	mmsave = calloc (124,  sizeof (*mmsave));

	WD_SINCOS (star, N, N1, snth, csth, snfi, csfi, mmsave);
	return SUCCESS;
}

int wd_binnum (double *abunarr, int abundim, double abunval, int *abunidx)
{
	WD_BINNUM (abunarr, abundim, abunval, abunidx);
	return SUCCESS;
}
