#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

#include "phoebe_accessories.h"
#include "phoebe_calculations.h"
#include "phoebe_configuration.h"
#include "phoebe_data.h"
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

PHOEBE_array *intern_find_unique_elements (int *input, int size)
{
	int i, unique = 0, na = -1001;
	PHOEBE_array *output;

	/* Copy the input: */
	int *copy = phoebe_malloc (size * sizeof (*copy));
	for (i = 0; i < size; i++)
		copy[i] = input[i];

	/* Sort the copied array: */
	qsort (copy, size, sizeof (*copy), intern_compare_ints);

	/* Count unique values in those arrays: */
	i = 1;
	while (1) {
		while (i < size-1 && copy[i] == copy[i-1]) i++;
		if (i == size) break;
		unique++;
		copy[unique] = copy[i];
		i++;
	}

	/* Allocate the output array and copy unique values: */
	output = phoebe_array_new (TYPE_INT_ARRAY);
	phoebe_array_alloc (output, unique+2);
	for (i = 1; i <= unique; i++)
		output->val.iarray[i] = copy[i-1];
	output->val.iarray[0] = output->val.iarray[i] = na;

	free (copy);

	return output;
}

PHOEBE_ld *phoebe_ld_new ()
{
	/**
	 * phoebe_ld_new:
	 *
	 * Initializes a new limb darkening (LD) table.
	 *
	 * Returns: a pointer to the new #PHOEBE_ld structure.
	 */

	PHOEBE_ld *table = phoebe_malloc (sizeof (*table));

	table->set      = NULL;
	table->name     = NULL;
	table->reftable = NULL;
	
	table->lin_x    = NULL;
	table->log_x    = NULL;
	table->log_y    = NULL;
	table->sqrt_x   = NULL;
	table->sqrt_y   = NULL;

	return table;
}

int phoebe_ld_alloc (PHOEBE_ld *table, int dim)
{
	/**
	 * phoebe_ld_alloc:
	 * @table: initialized limb darkening table
	 * @dim:   table dimension
	 *
	 * Allocates memory for the limb darkening table.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!table)
		return ERROR_LD_TABLE_NOT_INITIALIZED;

	if (table->lin_x)
		return ERROR_LD_TABLE_ALREADY_ALLOCATED;

	if (dim <= 0)
		return ERROR_LD_TABLE_INVALID_DIMENSION;

	table->lin_x  = phoebe_vector_new (); phoebe_vector_alloc (table->lin_x, dim);
	table->log_x  = phoebe_vector_new (); phoebe_vector_alloc (table->log_x, dim);
	table->log_y  = phoebe_vector_new (); phoebe_vector_alloc (table->log_y, dim);
	table->sqrt_x = phoebe_vector_new (); phoebe_vector_alloc (table->sqrt_x, dim);
	table->sqrt_y = phoebe_vector_new (); phoebe_vector_alloc (table->sqrt_y, dim);

	return SUCCESS;
}

int phoebe_ld_realloc (PHOEBE_ld *table, int dim)
{
	/**
	 * phoebe_ld_realloc:
	 * @table: initialized or already allocated limb darkening table
	 * @dim:   table dimension
	 *
	 * Reallocates memory for the limb darkening table.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!table)
		return ERROR_LD_TABLE_NOT_INITIALIZED;

	if (dim <= 0)
		return ERROR_LD_TABLE_INVALID_DIMENSION;

	phoebe_vector_realloc (table->lin_x, dim);
	phoebe_vector_realloc (table->log_x, dim);
	phoebe_vector_realloc (table->log_y, dim);
	phoebe_vector_realloc (table->sqrt_x, dim);
	phoebe_vector_realloc (table->sqrt_y, dim);

	return SUCCESS;
}

int phoebe_ld_free (PHOEBE_ld *table)
{
	/**
	 * phoebe_ld_free:
	 * @table: limb darkening table to be freed
	 *
	 * Frees memory of the limb darkening @table.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	if (!table)
		return SUCCESS;

	if (table->set)      free (table->set);
	if (table->name)     free (table->name);
	if (table->reftable) free (table->reftable);

	if (table->lin_x)    phoebe_vector_free (table->lin_x);
	if (table->log_x)    phoebe_vector_free (table->log_x);
	if (table->log_y)    phoebe_vector_free (table->log_y);
	if (table->sqrt_x)   phoebe_vector_free (table->sqrt_x);
	if (table->sqrt_y)   phoebe_vector_free (table->sqrt_y);

	free (table);

	return SUCCESS;
}

PHOEBE_ld *phoebe_ld_new_from_file (const char *filename)
{
	/**
	 * phoebe_ld_new_from_file:
	 * @filename: input file
	 *
	 * Reads in limb darkening coefficients. The table must have 5 columns:
	 * linear cosine coefficient, log coefficients, and sqrt coefficients.
	 * The @filename can either be absolute or relative; if relative, its
	 * location is presumed to be one of the following:
	 *
	 * 1) current working directory,
	 * 2) limb darkening directory (as stored in the global #PHOEBE_LD_DIR
	 *    variable).
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	PHOEBE_ld *table;
	FILE *ld_file;

	char *full_filename = NULL;

	char *lddir;
	char line[255], keyword[255];
	char *ptr;

	int i = 0;

	/* If the absolute path is given, use it as is or bail out. */
	if (filename[0] == '/') {
		if (!(ld_file = fopen (filename, "r")))
			return NULL;
	}

	/* If a relative path is given, try the current working directory; if it
	 * is still not found, try the limb darkening directory. If still it can't
	 * be found, bail out.
	 */
	else {
		ld_file = fopen (filename, "r");

		if (!ld_file) {
			phoebe_config_entry_get ("PHOEBE_LD_DIR", &lddir);
			full_filename = phoebe_concatenate_strings (lddir, "/", filename, NULL);
			ld_file = fopen (full_filename, "r");

			if (!ld_file) return NULL;
		}
	}

	/* By now either the file exists or NULL has been returned. */

	/* Allocate memory for the table; we'll start with 3800 since there are
	 * typically 3800 records in the table, and increase this if the size
	 * is exceeded.
	 */

	table = phoebe_ld_new ();
	phoebe_ld_alloc (table, 3800);

	/* Parse the header: */

	while (fgets (line, 255, ld_file)) {
		line[strlen(line)-1] = '\0';
		if (strchr (line, '#')) {
			/* This can be either be a comment or a header entry. */
			if (sscanf (line, "# %s", &(keyword[0])) != 1) continue;
			if (strcmp (keyword, "PASS_SET") == 0) {
				ptr = line + strlen (keyword) + 2;      /* +2 because of "# " */
				while (*ptr == ' ' || *ptr == '\t') ptr++;
				phoebe_debug ("PASS_SET encountered: %s\n", ptr);
				table->set = strdup (ptr);
				continue;
			}
			if (strcmp (keyword, "PASSBAND") == 0) {
				ptr = line + strlen (keyword) + 2;      /* +2 because of "# " */
				while (*ptr == ' ' || *ptr == '\t') ptr++;
				phoebe_debug ("PASSBAND encountered: %s\n", ptr);
				table->name = strdup (ptr);
				continue;
			}
			if (strcmp (keyword, "REFTABLE") == 0) {
				ptr = line + strlen (keyword) + 2;      /* +2 because of "# " */
				while (*ptr == ' ' || *ptr == '\t') ptr++;
				phoebe_debug ("REFTABLE encountered: %s\n", ptr);
				table->reftable = strdup (ptr);
				continue;
			}

			/* It's just a comment, ignore it. */
			continue;
		}
		else {
			/* Do we need to increase the table size? */
			if (i == table->log_x->dim)
				phoebe_ld_realloc (table, 2*i);

			/* Read out 5-column data only: */
			if (sscanf (line, "%lf %lf %lf %lf %lf", &table->lin_x->val[i],
						&table->log_x->val[i], &table->log_y->val[i],
						&table->sqrt_x->val[i], &table->sqrt_y->val[i]) != 5)
				continue;

			i++;
		}
	}

	fclose (ld_file);

	if (i == 0) {
		phoebe_ld_free (table);
		table = NULL;
	}

	if (full_filename)
		free (full_filename);

	return table;
}

int phoebe_ld_attach (PHOEBE_ld *table)
{
	/**
	 * phoebe_ld_attach:
	 * @table: populated limb darkening table
	 *
	 * Attaches the passed limb darkening @table to its respective passband.
	 * The passband is determined from the limb darkening table keywords
	 * PASS_SET and PASSBAND -- these have to match passband definitions
	 * exactly. See #PHOEBE_passband and #PHOEBE_ld structures for more
	 * information on identifying passbands.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	char *filter;
	PHOEBE_passband *passband;

	if (!table)
		return ERROR_LD_TABLE_NOT_INITIALIZED;
	if (!table->set || !table->name)
		return ERROR_LD_TABLE_PASSBAND_NOT_SPECIFIED;

	filter = phoebe_concatenate_strings (table->set, ":", table->name, NULL);
	passband = phoebe_passband_lookup (filter);

	if (!passband) {
		phoebe_lib_warning ("loading LD tables for passband %s failed.\n", filter);
		free (filter);
		return ERROR_LD_TABLE_PASSBAND_NOT_FOUND;
	}

	passband->ld = table;
	free (filter);

	return SUCCESS;
}

int phoebe_ld_attach_all (char *dir)
{
	/**
	 * phoebe_ld_attach_all:
	 * @dir: directory where limb darkening coefficients are stored
	 *
	 * Opens the directory @dir, scans all files in that directory and
	 * reads in all found LD tables. It attaches these tables to their
	 * respective passbands.
	 *
	 * This function must be called after phoebe_read_in_passbands(),
	 * otherwise attaching will fail.
	 *
	 * Returns: #PHOEBE_error_code.
	 */

	DIR *ld_dir;
	struct dirent *ld_file;
	char filename[255];

	int status;

	PHOEBE_ld *table;

	status = phoebe_open_directory (&ld_dir, dir);
	if (status != SUCCESS)
		return status;

	while ( (ld_file = readdir (ld_dir)) ) {
		sprintf (filename, "%s/%s", dir, ld_file->d_name);

		if (phoebe_filename_is_directory (filename)) continue;

		table = phoebe_ld_new_from_file (filename);
		if (!table)
			phoebe_debug ("File %s skipped.\n", filename);
		else
			phoebe_ld_attach (table);
	}

	phoebe_close_directory (&ld_dir);

	return SUCCESS;
}

LD_table *phoebe_ld_table_intern_load (char *model_list)
{
	/**
	 * phoebe_ld_table_intern_load:
	 * @model_list: filename of the reference table
	 *
	 * Loads the reference table for the internal PHOEBE database of limb
	 * darkening coefficients. The table must contain 3 columns: temperature,
	 * gravity and metallicity of the nodes. All values in the table must be
	 * integers, where gravity and metallicity are multiplied by 10. I.e., the
	 * entry for log(g)=4.5 is 45, the entry for [M/H]=-0.5 is -05.
	 *
	 * The reference table contains three #PHOEBE_array's: Mnodes, Tnodes and
	 * lgnodes. These arrays contain all nodes, in addition to -1001 for the
	 * first and last element of the array, which simplifies the bounds check
	 * of the lookup function phoebe_ld_get_coefficients().
	 *
	 * The table itself is a 3D matrix with dimensions of the three
	 * #PHOEBE_arrays mentioned above. The 'pos' fields are padded with -1
	 * if the node is not in the database, or the consecutive entry number in
	 * the reference table if it is. Since the LD tables are read to memory
	 * (in contrast to Van Hamme LD tables), the 'fn' field is always set to
	 * #NULL.
	 *
	 * Returns: #LD_table.
	 */

	int i, j, k, v, matched;
	LD_table *LD;
	FILE *models = fopen (model_list, "r");

	/* There are 3800 models by default, and we want to avoid unnecessary
	 * realloc-ing. That is why the arrays will be allocated with recsize
	 * and reallocated only if recsize is exceeded.
	 */

	int recsize = 3801, count;
	int *recT, *reclg, *recM;

	if (!models) {
		phoebe_lib_error ("failed to open %s for reading.\n", model_list);
		return NULL;
	}

	/* Read in the nodes: */

	recT  = phoebe_malloc (recsize * sizeof(*recT));
	reclg = phoebe_malloc (recsize * sizeof(*reclg));
	recM  = phoebe_malloc (recsize * sizeof(*recM));

	i = 0;
	while (1) {
		if (i == recsize) {
			phoebe_lib_warning ("reallocating LD arrays -- so that you know.\n");
			recsize *= 2;
			recT  = phoebe_realloc (recT, recsize * sizeof(*recT));
			reclg = phoebe_realloc (recT, recsize * sizeof(*reclg));
			recM  = phoebe_realloc (recT, recsize * sizeof(*recM));
		}

		matched = fscanf (models, "%d %d %d\n", &recT[i], &reclg[i], &recM[i]);

		if (feof (models) || matched != 3)
			break;

		i++;
	}
	fclose (models);
	count = i+1;

	phoebe_debug ("%d models read in.\n", count);

	LD = phoebe_malloc (sizeof (*LD));
	LD->Tnodes  = intern_find_unique_elements (recT,  count);
	LD->lgnodes = intern_find_unique_elements (reclg, count);
	LD->Mnodes  = intern_find_unique_elements (recM,  count);

	phoebe_debug ("There are %d temperature nodes, %d log(g) nodes, and %d metallicity nodes.\n", LD->Tnodes->dim-2, LD->lgnodes->dim-2, LD->Mnodes->dim-2);

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

	for (v = 0; v < count; v++) {
		for (i = 1;  recM[v] != LD->Mnodes->val.iarray[i];  i++) ;
		for (j = 1;  recT[v] != LD->Tnodes->val.iarray[j];  j++) ;
		for (k = 1; reclg[v] != LD->lgnodes->val.iarray[k]; k++) ;
		LD->table[i][j][k].pos = v;
	}

	free (recT); free (reclg); free (recM);

	return LD;
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

		while (fgets (line, 255, in)) {
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

LD_model phoebe_ld_model_type (const char *ldlaw)
{
	/**
	 * phoebe_ld_model_type:
	 * @ldlaw: name of the limb darkening model
	 *
	 * Converts common name of the limb darkening model to the LD enumerator.
	 * See #LD_model enumerator for a list of choices.
	 *
	 * Returns: enumerated #LD_model value.
	 */

	if (strcmp (ldlaw, "Linear cosine law") == 0) return LD_LAW_LINEAR;
	if (strcmp (ldlaw, "Logarithmic law"  ) == 0) return LD_LAW_LOG;
	if (strcmp (ldlaw, "Square root law"  ) == 0) return LD_LAW_SQRT;
	return LD_LAW_INVALID;
}

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

int intern_get_ld_node (const char *fn, long int pos, LD_model ldlaw, PHOEBE_passband *passband, double *x0, double *y0, bool ld_intern)
{
	/*
	 * This is an internal wrapper to get the LD coefficients from a file.
	 */

	FILE *in;
	char line[255], pass[10];
	double linx, sqrtx, sqrty, logx, logy;

	if (ld_intern == 1) {
		if (!passband->ld)
			return ERROR_LD_TABLE_PASSBAND_NOT_FOUND;

		switch (ldlaw) {
			case LD_LAW_LINEAR:
				*x0 = passband->ld->lin_x->val[pos];
			break;
			case LD_LAW_LOG:
				*x0 = passband->ld->log_x->val[pos];
				*y0 = passband->ld->log_y->val[pos];
			break;
			case LD_LAW_SQRT:
				*x0 = passband->ld->sqrt_x->val[pos];
				*y0 = passband->ld->sqrt_y->val[pos];
			break;
			default:
				return ERROR_LD_LAW_INVALID;
		}
	}
	else {
		/* Read out the values of coefficients: */
		phoebe_debug ("  opening %s:\n", fn);
		in = fopen (fn, "r");
		if (!in) {
			phoebe_lib_error ("LD table %s not found, aborting.\n", fn);
			return ERROR_LD_TABLES_MISSING;
		}
		fseek (in, pos, SEEK_SET);
		while (fgets (line, 255, in))
			if (sscanf (line, " %s %lf (%*f) %lf %lf (%*f) %lf %lf (%*f)", pass, &linx, &logx, &logy, &sqrtx, &sqrty) == 6)
				if (strcmp (pass, phoebe_ld_get_vh1993_passband_name (passband)) == 0) break;
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
	}

	return SUCCESS;
}

int phoebe_ld_get_coefficients (LD_model ldlaw, PHOEBE_passband *passband, double M, double T, double lg, double *x, double *y)
{
	/*
	 *  This function queries the LD coefficients database using the setup
	 *  stored in LD_table structure.
	 */

	LD_table *LD = PHOEBE_ld_table;
	int i, j, k, l;
	bool ld_intern;

	/* Interpolation structures: */
	double pars[3], lo[3], hi[3];
	union {
		double *d;
		PHOEBE_vector **vec;
	} fv;

	phoebe_debug ("entering phoebe_get_ld_coefficients () function.\n");

	phoebe_config_entry_get ("PHOEBE_LD_INTERN", &ld_intern);

	phoebe_debug ("  checking whether LD tables are present...\n");
	if (!LD)
		return ERROR_LD_TABLES_MISSING;

	phoebe_debug ("  checking whether LD law is sane...\n");
	if (ldlaw == LD_LAW_INVALID)
		return ERROR_LD_LAW_INVALID;

	phoebe_debug ("  checking whether passband definition is sane...\n");
	if (!passband)
		return ERROR_PASSBAND_INVALID;

	phoebe_debug ("  checking if a corresponding LD table exists...\n");
	if (ld_intern && !passband->ld)
		return ERROR_LD_TABLES_MISSING;

	phoebe_debug ("  all checks are fine, proceeding.\n");
	
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
		LD->table[i-1][j-1][k-1].pos == -1 || LD->table[i-1][j-1][k].pos == -1 ||
		LD->table[i-1][j  ][k-1].pos == -1 || LD->table[i-1][j  ][k].pos == -1 ||
		LD->table[i  ][j-1][k-1].pos == -1 || LD->table[i  ][j-1][k].pos == -1 ||
		LD->table[i  ][j  ][k-1].pos == -1 || LD->table[i  ][j  ][k].pos == -1
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

		intern_get_ld_node (LD->table[i-1][j-1][k-1].fn, LD->table[i-1][j-1][k-1].pos, ldlaw, passband, &fv.d[0], NULL, ld_intern);
		intern_get_ld_node (LD->table[ i ][j-1][k-1].fn, LD->table[ i ][j-1][k-1].pos, ldlaw, passband, &fv.d[1], NULL, ld_intern);
		intern_get_ld_node (LD->table[i-1][ j ][k-1].fn, LD->table[i-1][ j ][k-1].pos, ldlaw, passband, &fv.d[2], NULL, ld_intern);
		intern_get_ld_node (LD->table[ i ][ j ][k-1].fn, LD->table[ i ][ j ][k-1].pos, ldlaw, passband, &fv.d[3], NULL, ld_intern);
		intern_get_ld_node (LD->table[i-1][j-1][ k ].fn, LD->table[i-1][j-1][ k ].pos, ldlaw, passband, &fv.d[4], NULL, ld_intern);
		intern_get_ld_node (LD->table[ i ][j-1][ k ].fn, LD->table[ i ][j-1][ k ].pos, ldlaw, passband, &fv.d[5], NULL, ld_intern);
		intern_get_ld_node (LD->table[i-1][ j ][ k ].fn, LD->table[i-1][ j ][ k ].pos, ldlaw, passband, &fv.d[6], NULL, ld_intern);
		intern_get_ld_node (LD->table[ i ][ j ][ k ].fn, LD->table[ i ][ j ][ k ].pos, ldlaw, passband, &fv.d[7], NULL, ld_intern);

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
		intern_get_ld_node (LD->table[i-1][j-1][k-1].fn, LD->table[i-1][j-1][k-1].pos, ldlaw, passband, &(fv.vec[0]->val[0]), &(fv.vec[0]->val[1]), ld_intern);
		intern_get_ld_node (LD->table[ i ][j-1][k-1].fn, LD->table[ i ][j-1][k-1].pos, ldlaw, passband, &(fv.vec[1]->val[0]), &(fv.vec[1]->val[1]), ld_intern);
		intern_get_ld_node (LD->table[i-1][ j ][k-1].fn, LD->table[i-1][ j ][k-1].pos, ldlaw, passband, &(fv.vec[2]->val[0]), &(fv.vec[2]->val[1]), ld_intern);
		intern_get_ld_node (LD->table[ i ][ j ][k-1].fn, LD->table[ i ][ j ][k-1].pos, ldlaw, passband, &(fv.vec[3]->val[0]), &(fv.vec[3]->val[1]), ld_intern);
		intern_get_ld_node (LD->table[i-1][j-1][ k ].fn, LD->table[i-1][j-1][ k ].pos, ldlaw, passband, &(fv.vec[4]->val[0]), &(fv.vec[4]->val[1]), ld_intern);
		intern_get_ld_node (LD->table[ i ][j-1][ k ].fn, LD->table[ i ][j-1][ k ].pos, ldlaw, passband, &(fv.vec[5]->val[0]), &(fv.vec[5]->val[1]), ld_intern);
		intern_get_ld_node (LD->table[i-1][ j ][ k ].fn, LD->table[i-1][ j ][ k ].pos, ldlaw, passband, &(fv.vec[6]->val[0]), &(fv.vec[6]->val[1]), ld_intern);
		intern_get_ld_node (LD->table[ i ][ j ][ k ].fn, LD->table[ i ][ j ][ k ].pos, ldlaw, passband, &(fv.vec[7]->val[0]), &(fv.vec[7]->val[1]), ld_intern);

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
