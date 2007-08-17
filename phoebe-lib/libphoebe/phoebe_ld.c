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

int      PHOEBE_ld_table_size;
LDtable *PHOEBE_ld_table;

LDelem *phoebe_ld_elem_new (double M, int T, double lg)
{
	LDelem *elem = phoebe_malloc (sizeof (*elem));
	elem->M  = M;
	elem->T  = T;
	elem->lg = lg;
	return elem;
}

int phoebe_ld_table_free ()
{
	int i;
	
	for (i = 0; i < PHOEBE_ld_table_size; i++) {
		free (PHOEBE_ld_table[i].elem);
		free (PHOEBE_ld_table[i].filename);
	}
	free (PHOEBE_ld_table);

	return SUCCESS;
}

int read_in_ld_nodes (char *dir)
{
	/*
	 * This function scans all files in the passed directory dir and extracts
	 * LD triplets (T, log g, M/H) from table headers. The structure to hold
	 * these data is somewhat complicated, that's why this function is fairly
	 * long. Yet the benefit of it is that the access speed to data files is
	 * maximized while PHOEBE is running. This table could alternatively be
	 * hashed in an external file (as was done in PHOEBE 0.2x), yet I don't
	 * think waiting for 7/10th of a second during startup is that bad. I'm
	 * pretty confident this scan time can be furtherly reduced, yet I don't
	 * see a good reason why this would be necessary.
	 */

	int status;
	
	DIR *dirlist;
	struct dirent *file;

	FILE *in;

	char line[255];
	char LDfile[255];

	int i, j;

	int T;
	double lg, M;
	
	/* The structure above will be filled at readout to the following array:  */
	int    recno = 1;
	LDrecord *rec = phoebe_malloc (sizeof (*rec));
	LDrecord swapper;

	int counter = 0;
	
	status = phoebe_open_directory (&dirlist, dir);
	if (status != SUCCESS)
		return status;

	while ( (file = readdir (dirlist)) ) {
		sprintf (LDfile, "%s/%s", dir, file->d_name);

		/* Skip directories and 'ld_availability.data' file:                  */
		if (filename_is_directory (LDfile)) continue;
		if (strcmp (file->d_name, "ld_availability.data") == 0) continue;

		in = fopen (LDfile, "r");

		if (!in) {
			phoebe_lib_error ("file %s is invalid, skipping.\n", file->d_name);
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

	status = phoebe_close_directory (&dirlist);
	if (status != SUCCESS)
		return status;

	PHOEBE_ld_table_size = counter;

	/* To this point all available records have been read, along with their   */
	/* parent filenames and positions therein. We shall now sort these values */
	/* by M, then by T and finally by log g:                                  */

	for (i = 0; i < counter-1; i++) {
		for (j = i+1; j < counter; j++) {
			if (rec[i].M > rec[j].M) {
				swapper.M  = rec[j].M;
				swapper.lg = rec[j].lg;
				swapper.T  = rec[j].T;
				swapper.pos = rec[j].pos;
				swapper.filename = rec[j].filename;
			
				rec[j].M        = rec[i].M;
				rec[j].lg       = rec[i].lg;
				rec[j].T        = rec[i].T;
				rec[j].pos      = rec[i].pos;
				rec[j].filename = rec[i].filename;

				rec[i].M        = swapper.M; 
				rec[i].lg       = swapper.lg; 
				rec[i].T        = swapper.T; 
				rec[i].pos      = swapper.pos; 
				rec[i].filename = swapper.filename; 
			}
			else if (rec[i].M == rec[j].M) {
				if (rec[i].T > rec[j].T) {
					swapper.M  = rec[j].M;
					swapper.lg = rec[j].lg;
					swapper.T  = rec[j].T;
					swapper.pos = rec[j].pos;
					swapper.filename = rec[j].filename;
			
					rec[j].M        = rec[i].M;
					rec[j].lg       = rec[i].lg;
					rec[j].T        = rec[i].T;
					rec[j].pos      = rec[i].pos;
					rec[j].filename = rec[i].filename;

					rec[i].M        = swapper.M; 
					rec[i].lg       = swapper.lg; 
					rec[i].T        = swapper.T; 
					rec[i].pos      = swapper.pos; 
					rec[i].filename = swapper.filename; 
				}
				else if (rec[i].T == rec[j].T) {
					if (rec[i].lg > rec[j].lg) {
						swapper.M  = rec[j].M;
						swapper.lg = rec[j].lg;
						swapper.T  = rec[j].T;
						swapper.pos = rec[j].pos;
						swapper.filename = rec[j].filename;
			
						rec[j].M        = rec[i].M;
						rec[j].lg       = rec[i].lg;
						rec[j].T        = rec[i].T;
						rec[j].pos      = rec[i].pos;
						rec[j].filename = rec[i].filename;

						rec[i].M        = swapper.M; 
						rec[i].lg       = swapper.lg; 
						rec[i].T        = swapper.T; 
						rec[i].pos      = swapper.pos; 
						rec[i].filename = swapper.filename; 
					}
				}
			}
		}
	}

	/* A final step: let's populate the LD table with elements and free the   */
	/* temporary readout structure rec:                                       */

	PHOEBE_ld_table = phoebe_malloc (counter * sizeof (*PHOEBE_ld_table));
	for (i = 0; i < counter; i++) {
		PHOEBE_ld_table[i].elem     = phoebe_ld_elem_new (rec[i].M, rec[i].T, rec[i].lg);
		PHOEBE_ld_table[i].filename = strdup (rec[i].filename);
		PHOEBE_ld_table[i].filepos  = rec[i].pos;
		PHOEBE_ld_table[i].Mnext    = NULL;
		PHOEBE_ld_table[i].Mprev    = NULL;
		PHOEBE_ld_table[i].Tnext    = NULL;
		PHOEBE_ld_table[i].Tprev    = NULL;
		PHOEBE_ld_table[i].lgnext   = NULL;
		PHOEBE_ld_table[i].lgprev   = NULL;
		
		free (rec[i].filename);
	}
	free (rec);

	for (i = 0; i < counter-1; i++) {
		if (PHOEBE_ld_table[i].elem->lg < PHOEBE_ld_table[i+1].elem->lg) {
			PHOEBE_ld_table[i].lgnext   = &PHOEBE_ld_table[i+1];
			PHOEBE_ld_table[i+1].lgprev = &PHOEBE_ld_table[i];
		}

		j = i+1;
		while (j < counter && PHOEBE_ld_table[i].elem->T == PHOEBE_ld_table[j].elem->T) j++;
		if (j == counter) continue;

		for (; j < counter; j++) {
			if (PHOEBE_ld_table[i].elem->M == PHOEBE_ld_table[j].elem->M && PHOEBE_ld_table[i].elem->lg == PHOEBE_ld_table[j].elem->lg) {
				PHOEBE_ld_table[i].Tnext = &PHOEBE_ld_table[j];
				PHOEBE_ld_table[j].Tprev = &PHOEBE_ld_table[i];
				break;
			}
		}
		if (j == counter) continue;

		while (j < counter && PHOEBE_ld_table[i].elem->M == PHOEBE_ld_table[j].elem->M) j++;
		if (j == counter) continue;

		for (; j < counter; j++) {
			if (PHOEBE_ld_table[i].elem->T == PHOEBE_ld_table[j].elem->T && PHOEBE_ld_table[i].elem->lg == PHOEBE_ld_table[j].elem->lg) {
				PHOEBE_ld_table[i].Mnext = &PHOEBE_ld_table[j];
				PHOEBE_ld_table[j].Mprev = &PHOEBE_ld_table[i];
				break;
			}
		}
		if (j == counter) continue;
	}

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

	return NULL;
/*
	Still missing:
		case BOLOMETRIC:   return "bolo";
		case BLOCK_230:    return "230";
		case BLOCK_250:    return "250";
		case BLOCK_270:    return "270";
		case BLOCK_290:    return "290";
		case BLOCK_310:    return "310";
		case BLOCK_330:    return "330";
		default:           return NULL;
*/
}

int intern_get_ld_node (LDtable *table, LDLaw ldlaw, PHOEBE_passband *passband, double *x0, double *y0)
{
	/*
	 * This is an internal wrapper to get the LD coefficients from a file.
	 */

	FILE *in;
	char line[255], pass[10];
	double linx, sqrtx, sqrty, logx, logy;

	/* Read out the values of coefficients:                                   */
	phoebe_debug ("  opening %s:\n", table->filename);
	in = fopen (table->filename, "r");
	if (!in) {
		phoebe_lib_error ("LD table %s not found, aborting.\n", table->filename);
		return ERROR_LD_TABLES_MISSING;
	}
	fseek (in, table->filepos, SEEK_SET);
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

	phoebe_debug ("    M = %lf, T = %d, lg = %lf\n", table->elem->M, table->elem->T, table->elem->lg);
	phoebe_debug ("    %s: %lf, %lf, %lf, %lf, %lf\n", pass, linx, logx, logy, sqrtx, sqrty);

	return SUCCESS;
}

int phoebe_get_ld_coefficients (LDLaw ldlaw, PHOEBE_passband *passband, double M, int T, double lg, double *x, double *y)
{
	/*
	 *  This function queries the LD coefficients database using the setup
	 *  stored in LDTable structure.
	 */

	LDtable *table = &PHOEBE_ld_table[0];

	/* Interpolation structures: */
	double pars[3], lo[3], hi[3];
	union {
		double *d;
		PHOEBE_vector **vec;
	} fv;
	int i;

	phoebe_debug ("entering phoebe_get_ld_coefficients () function.\n");

	phoebe_debug ("  checking whether LD tables are present...\n");
	if (!table)
		return ERROR_LD_TABLES_MISSING;

	phoebe_debug ("  checking whether LD law is sane...\n");
	if (ldlaw == LD_LAW_INVALID)
		return ERROR_LD_LAW_INVALID;

	phoebe_debug ("  checking whether passband definition is sane...\n");
	if (!passband)
		return ERROR_PASSBAND_INVALID;

	phoebe_debug ("  checking whether parameters are out of range...\n");
	if  (  M  < table->elem->M
		|| T  < table->elem->T
		|| lg < table->elem->lg
		)
		return ERROR_LD_PARAMS_OUT_OF_RANGE;

	while ( M >= table->elem->M ) {
		table = table->Mnext;
		if (!table)
			return ERROR_LD_PARAMS_OUT_OF_RANGE;
	}

	while ( lg >= table->elem->lg ) {
		table = table->lgnext;
		if (!table)
			return ERROR_LD_PARAMS_OUT_OF_RANGE;
	}

	while ( T >= table->elem->T ) {
		table = table->Tnext;
		if (!table)
			return ERROR_LD_PARAMS_OUT_OF_RANGE;
	}

	phoebe_debug ("  everything ok, continuing to interpolation.\n");
	table = table->Mprev->Tprev->lgprev;

	/* Set the interpolation nodes: */
	pars[0] = T;                   pars[1] = lg;                    pars[2] = M;
	  lo[0] = table->elem->T;        lo[1] = table->elem->lg;         lo[2] = table->elem->M;
	  hi[0] = table->Tnext->elem->T; hi[1] = table->lgnext->elem->lg; hi[2] = table->Mnext->elem->M;

	if (ldlaw == LD_LAW_LINEAR) {
		fv.d = phoebe_malloc (8 * sizeof (*(fv.d)));

		/* Read out the values of coefficients:                                   */
		intern_get_ld_node (table, ldlaw, passband, &fv.d[0], NULL);
		intern_get_ld_node (table->Tnext, ldlaw, passband, &fv.d[1], NULL);
		intern_get_ld_node (table->lgnext, ldlaw, passband, &fv.d[2], NULL);
		intern_get_ld_node (table->Tnext->lgnext, ldlaw, passband, &fv.d[3], NULL);
		intern_get_ld_node (table->Mnext, ldlaw, passband, &fv.d[4], NULL);
		intern_get_ld_node (table->Tnext->Mnext, ldlaw, passband, &fv.d[5], NULL);
		intern_get_ld_node (table->lgnext->Mnext, ldlaw, passband, &fv.d[6], NULL);
		intern_get_ld_node (table->Tnext->lgnext->Mnext, ldlaw, passband, &fv.d[7], NULL);

		/* Do the interpolation: */
		phoebe_interpolate (3, pars, lo, hi, TYPE_DOUBLE, fv.d);

		/* Assign the return value and free the array:                        */
		*x = fv.d[0];
		free (fv.d);
	}
	else {
		fv.vec = phoebe_malloc (8 * sizeof (*(fv.vec)));
		for (i = 0; i < 8; i++) {
			fv.vec[i] = phoebe_vector_new ();
			phoebe_vector_alloc (fv.vec[i], 2);
		}

		/* Read out the values of coefficients:                                   */
		intern_get_ld_node (table, ldlaw, passband, &fv.vec[0]->val[0], &fv.vec[0]->val[1]);
		intern_get_ld_node (table->Tnext, ldlaw, passband, &fv.vec[1]->val[0], &fv.vec[1]->val[1]);
		intern_get_ld_node (table->lgnext, ldlaw, passband, &fv.vec[2]->val[0], &fv.vec[2]->val[1]);
		intern_get_ld_node (table->Tnext->lgnext, ldlaw, passband, &fv.vec[3]->val[0], &fv.vec[3]->val[1]);
		intern_get_ld_node (table->Mnext, ldlaw, passband, &fv.vec[4]->val[0], &fv.vec[4]->val[1]);
		intern_get_ld_node (table->Tnext->Mnext, ldlaw, passband, &fv.vec[5]->val[0], &fv.vec[5]->val[1]);
		intern_get_ld_node (table->lgnext->Mnext, ldlaw, passband, &fv.vec[6]->val[0], &fv.vec[6]->val[1]);
		intern_get_ld_node (table->Tnext->lgnext->Mnext, ldlaw, passband, &fv.vec[7]->val[0], &fv.vec[7]->val[1]);

		/* Do the interpolation: */
		phoebe_interpolate (3, pars, lo, hi, TYPE_DOUBLE_ARRAY, fv.vec);

		/* Assign the return value and free the array:                        */
		*x = fv.vec[0]->val[0]; *y = fv.vec[0]->val[1];
		for (i = 0; i < 8; i++)
			phoebe_vector_free (fv.vec[i]);
	}

	return SUCCESS;
}
