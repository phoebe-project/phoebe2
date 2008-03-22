#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

#include "phoebe_accessories.h"
#include "phoebe_calculations.h"
#include "phoebe_error_handling.h"
#include "phoebe_global.h"
#include "phoebe_ld.h"
#include "phoebe_parameters.h"
#include "phoebe_types.h"

LD_table *PHOEBE_ld_table;

int intern_compare_ints (const void *a, const void *b)
{
	const int *v1 = (const int *) a;
	const int *v2 = (const int *) b;

	return *v1  - *v2;
}

LD_table *phoebe_ld_table_vh1993_load (char *dir)
{
	/**
	 * phoebe_ld_table_vh1993_load:
	 * @dir: directory that contains Van Hamme (1993) LD tables
	 *
	 * Scans all files in the passed directory dir and extracts LD triplets (T,
	 * log g, M/H) from table headers. It then creates a 3D matrix what holds
	 * all of these elements so that LD[M][T][lg] is structured by indices and
	 * all non-existing nodes padded with nans. For optimized lookup this
	 * matrix has a border of nans in all three directions - so that the lookup
	 * function does not have to check for matrix boundaries.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int status;
	LD_table *LD;

	DIR *dirlist;
	struct dirent *file;

	FILE *in;

	char line[255];
	char LDfile[255];

	int i, j, k, l;

	int na = -1001; /* n/a field in the LD matrix */

	int T;
	double lg, M;
	
	/* LD readout structure: */
	int recno = 1;
	struct {
		char    *filename;
		long int pos;
		int      T;
		double   lg;
		double   M;
	} *rec;

	int counter = 0;

	status = phoebe_open_directory (&dirlist, dir);
	if (status != SUCCESS) {
		phoebe_lib_error ("failed to open %s for reading.\n", dir);
		return NULL;
	}

	rec = phoebe_malloc (sizeof (*rec));

	while ( (file = readdir (dirlist)) ) {
		sprintf (LDfile, "%s/%s", dir, file->d_name);

		/* Skip directories and 'ld_availability.data' file: */
		if (phoebe_filename_is_directory (LDfile)) continue;
		if (strcmp (file->d_name, "ld_availability.data") == 0) continue;

		in = fopen (LDfile, "r");

		if (!in) {
			phoebe_lib_error ("failed to open %s for reading, skipping.\n", file->d_name);
			continue;
		}

		while (!feof (in)) {
			fgets (line, 255, in);
			if (sscanf (line, " Teff = %d K, log g = %lf, [M/H] = %lf", &T, &lg, &M) == 3) {
				rec[counter].filename = strdup (LDfile);
				rec[counter].pos      = ftell (in);
				rec[counter].T        = T;
				rec[counter].lg       = lg;
				rec[counter].M        = M;
				counter++;
				if (counter >= recno) {
					rec = phoebe_realloc (rec, 2*recno*sizeof (*rec));
					recno *= 2;
				}
			}
		}
		fclose (in);
	}

	if (!counter) {
		/* No lines have been read */
		free(rec);
		return NULL;
	}

	phoebe_close_directory (&dirlist);

	LD = phoebe_malloc (sizeof (*LD));

	LD->Tnodes = phoebe_array_new (TYPE_INT_ARRAY);
	phoebe_array_alloc (LD->Tnodes, 2);
	LD->Tnodes->val.iarray[0] = LD->Tnodes->val.iarray[1] = na;

	LD->Mnodes = phoebe_array_new (TYPE_INT_ARRAY);
	phoebe_array_alloc (LD->Mnodes, 2);
	LD->Mnodes->val.iarray[0] = LD->Mnodes->val.iarray[1] = na;

	LD->lgnodes = phoebe_array_new (TYPE_INT_ARRAY);
	phoebe_array_alloc (LD->lgnodes, 2);
	LD->lgnodes->val.iarray[0] = LD->lgnodes->val.iarray[1] = na;

	for (i = 0; i < counter; i++) {
		for (j = 1; j < LD->Tnodes->dim; j++)
			if (LD->Tnodes->val.iarray[j] == rec[i].T)
				break;
		if (j == LD->Tnodes->dim) {
			phoebe_array_realloc (LD->Tnodes, LD->Tnodes->dim+1);
			LD->Tnodes->val.iarray[j-1]   = rec[i].T;
			LD->Tnodes->val.iarray[j] = na;
		}

		for (j = 1; j < LD->Mnodes->dim; j++)
			if (LD->Mnodes->val.iarray[j] == (int) (10.0*rec[i].M))
				break;
		if (j == LD->Mnodes->dim) {
			phoebe_array_realloc (LD->Mnodes, LD->Mnodes->dim+1);
			LD->Mnodes->val.iarray[j-1]   = (int) (10.0*rec[i].M);
			LD->Mnodes->val.iarray[j] = na;
		}

		for (j = 1; j < LD->lgnodes->dim; j++)
			if (LD->lgnodes->val.iarray[j] == (int) (10.0*rec[i].lg))
				break;
		if (j == LD->lgnodes->dim) {
			phoebe_array_realloc (LD->lgnodes, LD->lgnodes->dim+1);
			LD->lgnodes->val.iarray[j-1]   = (int) (10.0*rec[i].lg);
			LD->lgnodes->val.iarray[j] = na;
		}
	}

	/* Sort the node lists: */
	qsort (&(LD-> Tnodes->val.iarray[1]), LD-> Tnodes->dim-2, sizeof(LD-> Tnodes->val.iarray[1]), intern_compare_ints);
	qsort (&(LD-> Mnodes->val.iarray[1]), LD-> Mnodes->dim-2, sizeof(LD-> Mnodes->val.iarray[1]), intern_compare_ints);
	qsort (&(LD->lgnodes->val.iarray[1]), LD->lgnodes->dim-2, sizeof(LD->lgnodes->val.iarray[1]), intern_compare_ints);

/*
	printf ("Temperatures:\n");
	for (i = 0; i < LD->Tnodes->dim; i++)
		printf ("%d ", LD->Tnodes->val.iarray[i]);
	printf ("\n");

	printf ("Metallicities:\n");
	for (i = 0; i < LD->Mnodes->dim; i++)
		printf ("%d ", LD->Mnodes->val.iarray[i]);
	printf ("\n");

	printf ("Gravities:\n");
	for (i = 0; i < LD->lgnodes->dim; i++)
		printf ("%d ", LD->lgnodes->val.iarray[i]);
	printf ("\n");
*/

	LD->table = phoebe_malloc (LD->Mnodes->dim * sizeof (*(LD->table)));
	for (i = 0; i < LD->Mnodes->dim; i++) {
		LD->table[i] = phoebe_malloc (LD->Tnodes->dim * sizeof (**(LD->table)));
		for (j = 0; j < LD->Tnodes->dim; j++) {
			LD->table[i][j] = phoebe_malloc (LD->lgnodes->dim * sizeof (***(LD->table)));
			for (k = 0; k < LD->lgnodes->dim; k++) {
				LD->table[i][j][k].fn = NULL;
				LD->table[i][j][k].pos = -1;
			}
		}
	}

	for (l = 0; l < counter; l++) {
		for (i = 1; i < LD->Mnodes->dim; i++)
			if (LD->Mnodes->val.iarray[i] == (int) (10.0*rec[l].M))
				break;
		for (j = 1; j < LD->Tnodes->dim; j++)
			if (LD->Tnodes->val.iarray[j] == rec[l].T)
				break;
		for (k = 1; k < LD->lgnodes->dim; k++)
			if (LD->lgnodes->val.iarray[k] == (int) (10.0*rec[l].lg))
				break;

		LD->table[i][j][k].fn  = strdup (rec[l].filename);
		LD->table[i][j][k].pos = rec[l].pos;

		free (rec[l].filename);
	}
	free (rec);

	return LD;
}

int phoebe_ld_table_free (LD_table *LD)
{
	/**
	 * phoebe_ld_table_free:
	 * @LD: limb darkening table
	 *
	 * Frees the contents of the limb darkening table @LD. It is safe to call
	 * this function on the unallocated table, so there is no need to check
	 * for LD presence.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	int i, j, k;

	if (!LD)
		return SUCCESS;

	for (i = 0; i < LD->Mnodes->dim; i++) {
		for (j = 0; j < LD->Tnodes->dim; j++) {
			for (k = 0; k < LD->lgnodes->dim; k++)
				free (LD->table[i][j][k].fn);
			free (LD->table[i][j]);
		}
		free (LD->table[i]);
	}
	free (LD->table);

	phoebe_array_free (LD-> Mnodes);
	phoebe_array_free (LD-> Tnodes);
	phoebe_array_free (LD->lgnodes);

	free (LD);

	return SUCCESS;
}

LDLaw phoebe_ld_model_type (const char *ldlaw)
{
	/*
	 * This function makes a conversion from LD model string to LD enumerator.
	 */

	if (strcmp (ldlaw, "Linear cosine law") == 0) return LD_LAW_LINEAR;
	if (strcmp (ldlaw, "Logarithmic law"  ) == 0) return LD_LAW_LOG;
	if (strcmp (ldlaw, "Square root law"  ) == 0) return LD_LAW_SQRT;
	return LD_LAW_INVALID;
}

/*
PHOEBE_passband phoebe_passband_id_from_string (const char *passband)
{
	if (strcmp (passband,  "Bolometric") == 0) return BOLOMETRIC;
	if (strcmp (passband,   "350nm (u)") == 0) return STROEMGREN_U;
	if (strcmp (passband,   "411nm (v)") == 0) return STROEMGREN_V;
	if (strcmp (passband,   "467nm (b)") == 0) return STROEMGREN_B;
	if (strcmp (passband,   "547nm (y)") == 0) return STROEMGREN_Y;
	if (strcmp (passband,   "360nm (U)") == 0) return JOHNSON_U;
	if (strcmp (passband,   "440nm (B)") == 0) return JOHNSON_B;
	if (strcmp (passband,   "550nm (V)") == 0) return JOHNSON_V;
	if (strcmp (passband,   "700nm (R)") == 0) return JOHNSON_R;
	if (strcmp (passband,   "900nm (I)") == 0) return JOHNSON_I;
	if (strcmp (passband,  "1250nm (J)") == 0) return JOHNSON_J;
	if (strcmp (passband,  "2200nm (K)") == 0) return JOHNSON_K;
	if (strcmp (passband,  "3400nm (L)") == 0) return JOHNSON_L;
	if (strcmp (passband,  "5000nm (M)") == 0) return JOHNSON_M;
	if (strcmp (passband, "10200nm (N)") == 0) return JOHNSON_N;
	if (strcmp (passband,  "647nm (Rc)") == 0) return COUSINS_R;
	if (strcmp (passband,  "786nm (Ic)") == 0) return COUSINS_I;
	if (strcmp (passband,  "505nm (Hp)") == 0) return HIPPARCOS;
	if (strcmp (passband,  "419nm (Bt)") == 0) return TYCHO_B;
	if (strcmp (passband,  "523nm (Vt)") == 0) return TYCHO_V;
	
	return PASSBAND_INVALID;
}
*/

char *phoebe_ld_get_vh1993_passband_name (PHOEBE_passband *passband)
{
	/*
	 * This function is a bridge between PHOEBE passbands and Walter Van
	 * Hamme (1993) markup. It takes a pointer to PHOEBE passband and, if
	 * an equivalent exists in the tables, it returns its name. If the match
	 * is not found, NULL is returned.
	 */

	if (strcmp (passband->set, "Stromgren") == 0 && strcmp (passband->name, "u") == 0) return "u";
	if (strcmp (passband->set, "Stromgren") == 0 && strcmp (passband->name, "v") == 0) return "v";
	if (strcmp (passband->set, "Stromgren") == 0 && strcmp (passband->name, "b") == 0) return "b";
	if (strcmp (passband->set, "Stromgren") == 0 && strcmp (passband->name, "y") == 0) return "y";

	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "U") == 0) return "U";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "B") == 0) return "B";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "V") == 0) return "V";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "R") == 0) return "R";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "I") == 0) return "I";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "J") == 0) return "J";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "K") == 0) return "K";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "L") == 0) return "L";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "M") == 0) return "M";
	if (strcmp (passband->set,   "Johnson") == 0 && strcmp (passband->name, "N") == 0) return "N";

	if (strcmp (passband->set,   "Cousins") == 0 && strcmp (passband->name, "R") == 0) return "R";
	if (strcmp (passband->set,   "Cousins") == 0 && strcmp (passband->name, "I") == 0) return "I";

	if (strcmp (passband->set, "Hipparcos") == 0 && strcmp (passband->name, "Hp") == 0) return "HIP";
	if (strcmp (passband->set, "Hipparcos") == 0 && strcmp (passband->name, "BT") == 0) return "TyB";
	if (strcmp (passband->set, "Hipparcos") == 0 && strcmp (passband->name, "VT") == 0) return "TyV";

	if (strcmp (passband->set, "Bolometric") == 0) return "bolo";

	return NULL;
/*
	Still missing:
		case BLOCK_230:    return "230";
		case BLOCK_250:    return "250";
		case BLOCK_270:    return "270";
		case BLOCK_290:    return "290";
		case BLOCK_310:    return "310";
		case BLOCK_330:    return "330";
		default:           return NULL;
*/
}

int intern_get_ld_node (const char *fn, long int pos, LDLaw ldlaw, PHOEBE_passband *passband, double *x0, double *y0)
{
	/*
	 * This is an internal wrapper to get the LD coefficients from a file.
	 */

	FILE *in;
	char line[255], pass[10];
	double linx, sqrtx, sqrty, logx, logy;

	/* Read out the values of coefficients: */
	phoebe_debug ("  opening %s:\n", fn);
	in = fopen (fn, "r");
	if (!in) {
		phoebe_lib_error ("LD table %s not found, aborting.\n", fn);
		return ERROR_LD_TABLES_MISSING;
	}
	fseek (in, pos, SEEK_SET);
	while (TRUE) {
		fgets (line, 255, in);
		if (sscanf (line, " %s %lf (%*f) %lf %lf (%*f) %lf %lf (%*f)", pass, &linx, &logx, &logy, &sqrtx, &sqrty) == 6) {
			if (strcmp (pass, phoebe_ld_get_vh1993_passband_name (passband)) == 0) break;
		}
	}
	fclose (in);

	switch (ldlaw) {
		case LD_LAW_LINEAR: *x0 = linx;               break;
		case LD_LAW_SQRT:   *x0 = sqrtx; *y0 = sqrty; break;
		case LD_LAW_LOG:    *x0 = logx;  *y0 = logy;  break;
		default:
			phoebe_lib_error ("exception handler invoked in intern_get_ld_node (). Please report this!\n");
			return ERROR_LD_LAW_INVALID;
	}

	phoebe_debug ("  %s[%ld]:  %s: %lf, %lf, %lf, %lf, %lf\n", fn, pos, pass, linx, logx, logy, sqrtx, sqrty);

	return SUCCESS;
}

int phoebe_ld_get_coefficients (LDLaw ldlaw, PHOEBE_passband *passband, double M, double T, double lg, double *x, double *y)
{
	/*
	 *  This function queries the LD coefficients database using the setup
	 *  stored in LDTable structure.
	 */

	LD_table *LD = PHOEBE_ld_table;
	int i, j, k, l;

	/* Interpolation structures: */
	double pars[3], lo[3], hi[3];
	union {
		double *d;
		PHOEBE_vector **vec;
	} fv;

	phoebe_debug ("entering phoebe_get_ld_coefficients () function.\n");

	phoebe_debug ("  checking whether LD tables are present...\n");
	if (!LD)
		return ERROR_LD_TABLES_MISSING;

	phoebe_debug ("  checking whether LD law is sane...\n");
	if (ldlaw == LD_LAW_INVALID)
		return ERROR_LD_LAW_INVALID;

	phoebe_debug ("  checking whether passband definition is sane...\n");
	if (!passband)
		return ERROR_PASSBAND_INVALID;

	/* Get node array indices: */
	for (i = 1; i < LD->Mnodes->dim; i++)
		if (10.0*M < LD->Mnodes->val.iarray[i])
			break;

	for (j = 1; j < LD->Tnodes->dim; j++)
		if (T < LD->Tnodes->val.iarray[j])
			break;

	for (k = 1; k < LD->lgnodes->dim; k++)
		if (10.0*lg < LD->lgnodes->val.iarray[k])
			break;

	if  (
		!LD->table[i-1][j-1][k-1].fn || !LD->table[i-1][j-1][k].fn ||
		!LD->table[i-1][j  ][k-1].fn || !LD->table[i-1][j  ][k].fn ||
		!LD->table[i  ][j-1][k-1].fn || !LD->table[i  ][j-1][k].fn ||
		!LD->table[i  ][j  ][k-1].fn || !LD->table[i  ][j  ][k].fn
		)
		return ERROR_LD_PARAMS_OUT_OF_RANGE;

	/* Set the interpolation nodes: */
	pars[0] = 10.0*M;
	  lo[0] = (double) LD->Mnodes->val.iarray[i-1];
	  hi[0] = (double) LD->Mnodes->val.iarray[i];

	pars[1] = T;
	  lo[1] = (double) LD->Tnodes->val.iarray[j-1];
	  hi[1] = (double) LD->Tnodes->val.iarray[j];

	pars[2] = 10.0*lg;
	  lo[2] = (double) LD->lgnodes->val.iarray[k-1];
	  hi[2] = (double) LD->lgnodes->val.iarray[k];

	phoebe_debug ("  metallicity: %2.2lf < %2.2lf < %2.2lf\n", lo[0], 10.0* M, hi[0]);
	phoebe_debug ("  temperature: %2.2lf < %2.2lf < %2.2lf\n", lo[1],       T, hi[1]);
	phoebe_debug ("  gravity:     %2.2lf < %2.2lf < %2.2lf\n", lo[2], 10.0*lg, hi[2]);

	if (ldlaw == LD_LAW_LINEAR) {
		fv.d = phoebe_malloc (8 * sizeof (*(fv.d)));

		/* Read out the values of coefficients: */
		intern_get_ld_node (LD->table[i-1][j-1][k-1].fn, LD->table[i-1][j-1][k-1].pos, ldlaw, passband, &fv.d[0], NULL);
		intern_get_ld_node (LD->table[ i ][j-1][k-1].fn, LD->table[ i ][j-1][k-1].pos, ldlaw, passband, &fv.d[1], NULL);
		intern_get_ld_node (LD->table[i-1][ j ][k-1].fn, LD->table[i-1][ j ][k-1].pos, ldlaw, passband, &fv.d[2], NULL);
		intern_get_ld_node (LD->table[ i ][ j ][k-1].fn, LD->table[ i ][ j ][k-1].pos, ldlaw, passband, &fv.d[3], NULL);
		intern_get_ld_node (LD->table[i-1][j-1][ k ].fn, LD->table[i-1][j-1][ k ].pos, ldlaw, passband, &fv.d[4], NULL);
		intern_get_ld_node (LD->table[ i ][j-1][ k ].fn, LD->table[ i ][j-1][ k ].pos, ldlaw, passband, &fv.d[5], NULL);
		intern_get_ld_node (LD->table[i-1][ j ][ k ].fn, LD->table[i-1][ j ][ k ].pos, ldlaw, passband, &fv.d[6], NULL);
		intern_get_ld_node (LD->table[ i ][ j ][ k ].fn, LD->table[ i ][ j ][ k ].pos, ldlaw, passband, &fv.d[7], NULL);

		/* Do the interpolation: */
		phoebe_interpolate (3, pars, lo, hi, TYPE_DOUBLE, fv.d);

		/* Assign the return value and free the array: */
		*x = fv.d[0]; *y = 0;
		phoebe_debug ("  LD coefficient: %lf\n", *x);
		free (fv.d);
	}
	else {
		fv.vec = phoebe_malloc (8 * sizeof (*(fv.vec)));
		for (l = 0; l < 8; l++) {
			fv.vec[l] = phoebe_vector_new ();
			phoebe_vector_alloc (fv.vec[l], 2);
		}

		/* Read out the values of coefficients: */
		intern_get_ld_node (LD->table[i-1][j-1][k-1].fn, LD->table[i-1][j-1][k-1].pos, ldlaw, passband, &(fv.vec[0]->val[0]), &(fv.vec[0]->val[1]));
		intern_get_ld_node (LD->table[ i ][j-1][k-1].fn, LD->table[ i ][j-1][k-1].pos, ldlaw, passband, &(fv.vec[1]->val[0]), &(fv.vec[1]->val[1]));
		intern_get_ld_node (LD->table[i-1][ j ][k-1].fn, LD->table[i-1][ j ][k-1].pos, ldlaw, passband, &(fv.vec[2]->val[0]), &(fv.vec[2]->val[1]));
		intern_get_ld_node (LD->table[ i ][ j ][k-1].fn, LD->table[ i ][ j ][k-1].pos, ldlaw, passband, &(fv.vec[3]->val[0]), &(fv.vec[3]->val[1]));
		intern_get_ld_node (LD->table[i-1][j-1][ k ].fn, LD->table[i-1][j-1][ k ].pos, ldlaw, passband, &(fv.vec[4]->val[0]), &(fv.vec[4]->val[1]));
		intern_get_ld_node (LD->table[ i ][j-1][ k ].fn, LD->table[ i ][j-1][ k ].pos, ldlaw, passband, &(fv.vec[5]->val[0]), &(fv.vec[5]->val[1]));
		intern_get_ld_node (LD->table[i-1][ j ][ k ].fn, LD->table[i-1][ j ][ k ].pos, ldlaw, passband, &(fv.vec[6]->val[0]), &(fv.vec[6]->val[1]));
		intern_get_ld_node (LD->table[ i ][ j ][ k ].fn, LD->table[ i ][ j ][ k ].pos, ldlaw, passband, &(fv.vec[7]->val[0]), &(fv.vec[7]->val[1]));

		/* Do the interpolation: */
		phoebe_interpolate (3, pars, lo, hi, TYPE_DOUBLE_ARRAY, fv.vec);

		/* Assign the return value and free the array: */
		*x = fv.vec[0]->val[0]; *y = fv.vec[0]->val[1];
		for (l = 0; l < 8; l++)
			phoebe_vector_free (fv.vec[l]);
		free (fv.vec);
		phoebe_debug ("  LD coefficients: %lf, %lf\n", *x, *y);
	}

	return SUCCESS;
}
