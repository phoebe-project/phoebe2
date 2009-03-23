#ifndef PHOEBE_ACCESSORIES_H
	#define PHOEBE_ACCESSORIES_H 1

#include <stdio.h>
#include <dirent.h>

#include "phoebe_types.h"

int   phoebe_open_directory                   (DIR **dirlist, const char *dirname);
int   phoebe_close_directory                  (DIR **dir);

bool  phoebe_filename_exists                  (const char *filename);
bool  phoebe_filename_has_write_permissions   (const char *filename);
bool  phoebe_filename_has_read_permissions    (const char *filename);
bool  phoebe_filename_has_execute_permissions (const char *filename);
bool  phoebe_filename_has_full_permissions    (const char *filename);
bool  phoebe_filename_is_directory            (const char *filename);
bool  phoebe_filename_is_regular_file         (const char *filename);

char *phoebe_get_current_working_directory    ();
char *phoebe_resolve_relative_filename        (char *filename);
int   phoebe_list_directory_contents          (char *dir);
char *phoebe_concatenate_strings              (const char *str, ...);
char *phoebe_clean_data_line                  (char *line);

bool  atob                                    (char *str);
char *phoebe_strdup                           (const char *s);
char *phoebe_readline                         (FILE *stream);
char *phoebe_create_temp_filename             (char *templ);

long int phoebe_seed ();

#endif
