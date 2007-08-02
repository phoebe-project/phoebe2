#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "phoebe_accessories.h"
#include "phoebe_allocations.h"
#include "phoebe_build_config.h"
#include "phoebe_calculations.h"
#include "phoebe_error_handling.h"
#include "phoebe_types.h"

int phoebe_open_directory (DIR **dir, const char *dirname)
{
	int error;

	*dir = opendir (dirname);
	error = errno;

	if (!(*dir)) {
		switch (error) {
			case EACCES:  return ERROR_DIRECTORY_PERMISSION_DENIED;
			case EMFILE:  return ERROR_DIRECTORY_TOO_MANY_FILE_DESCRIPTORS;
			case ENFILE:  return ERROR_DIRECTORY_TOO_MANY_OPEN_FILES;
			case ENOENT:  return ERROR_DIRECTORY_NOT_FOUND;
			case ENOMEM:  return ERROR_DIRECTORY_INSUFFICIENT_MEMORY;
			case ENOTDIR: return ERROR_DIRECTORY_NOT_A_DIRECTORY;
			default:      return ERROR_DIRECTORY_UNKNOWN_ERROR;
		}
	}
	return SUCCESS;
}

int phoebe_close_directory (DIR **dir)
{
	int error;

	int status = closedir (*dir);
	error = errno;

	if (status != 0) {
		switch (error) {
			case EBADF:   return ERROR_DIRECTORY_BAD_FILE_DESCRIPTOR;
			default:      return ERROR_DIRECTORY_UNKNOWN_ERROR;
		}
	}
	return SUCCESS;
}

bool filename_exists (const char *filename)
{
	/*
	 * This function checks whether a file exists.
	 */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0) return TRUE;

	return FALSE;
}

bool filename_has_write_permissions (const char *filename)
	{
	/* This function checks whether the supplied filename has write permissions */
	/* for the effective user ID.                                               */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0)
		{
		if ( ( (log.st_uid == geteuid ()) && (log.st_mode & S_IWUSR) ) ||
			   ( (log.st_gid == getegid ()) && (log.st_mode & S_IWGRP) ) ||
			   ( (log.st_mode & S_IWOTH) ) )
			return TRUE;

		return FALSE;
		}

	return FALSE;
	}

bool filename_has_read_permissions (const char *filename)
	{
	/* This function checks whether the supplied filename has read permissions  */
	/* for the effective user ID.                                               */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0)
		{
		if ( ( (log.st_uid == geteuid ()) && (log.st_mode & S_IRUSR) ) ||
			   ( (log.st_gid == getegid ()) && (log.st_mode & S_IRGRP) ) ||
			   ( (log.st_mode & S_IROTH) ) )
			return TRUE;
		
		return FALSE;
		}

	return FALSE;
	}

bool filename_has_execute_permissions (const char *filename)
	{
	/* This function checks whether the supplied filename has execute permissi- */
	/* ons for the effective user ID.                                           */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0)
		{
		if ( ( (log.st_uid == geteuid ()) && (log.st_mode & S_IXUSR) ) ||
			   ( (log.st_gid == getegid ()) && (log.st_mode & S_IXGRP) ) ||
			   ( (log.st_mode & S_IXOTH) ) )
			return TRUE;

		return FALSE;
		}

	return FALSE;
	}

bool filename_has_full_permissions (const char *filename)
	{
	/* This function checks whether the supplied filename has full permissions  */
	/* for the effective user ID.                                               */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0)
		{
		if ( ( (log.st_uid == geteuid ()) && (log.st_mode & S_IRUSR) && (log.st_mode & S_IWUSR) && (log.st_mode & S_IXUSR) ) ||
			   ( (log.st_gid == getegid ()) && (log.st_mode & S_IRGRP) && (log.st_mode & S_IWGRP) && (log.st_mode & S_IXGRP) ) ||
			   ( (log.st_mode & S_IROTH) && (log.st_mode & S_IWOTH) && (log.st_mode & S_IXOTH) ) )
			return TRUE;

		return FALSE;
		}

	return FALSE;
	}

bool filename_is_directory (const char *filename)
{
	/*
	 * This function checks whether the supplied filename is a directory.
	 */

	struct stat log;
	int check;

	if (!filename) return FALSE;

	check = stat (filename, &log);
	if (check == 0) {
		if (S_ISDIR (log.st_mode)) return TRUE;
		return FALSE;
	}

	return FALSE;
}

bool filename_is_regular_file (const char *filename)
	{
	/* This function checks whether the supplied filename is a regular file.    */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0)
		{
		if (S_ISREG (log.st_mode)) return TRUE;
		return FALSE;
		}

	return FALSE;
	}

char *get_current_working_directory ()
{
	/*
	 * This function gets the current working directory that is used to resolve
	 * relative filenames. It allocates space itself, the user has to free it.
	 */

	size_t size = 100;

	while (1) {
		char *buffer = phoebe_malloc (size);
		if (getcwd (buffer, size) == buffer)
			return buffer;
		free (buffer);
		if (errno != ERANGE)
			return NULL;
		size *= 2;
	}
}

char *resolve_relative_filename (char *filename)
{
	/* 
	 * This function takes a relative filename and resolves it to absolute; it
	 * allocates the memory for the string itself, the user has to free it.
	 */

	char *cwd;
	char *abs;

	/* Is the filename already absolute?                                        */
	if (filename[0] == '/') return filename;

	cwd = get_current_working_directory ();
	abs = phoebe_malloc (strlen (cwd) + strlen (filename) + 2); /* for / and \0 */
	sprintf (abs, "%s/%s", cwd, filename);
	free (cwd);

	return abs;
}

int list_directory_contents (char *dir)
{
	DIR *directory;
	struct dirent *direntry;
	
	directory = opendir (dir);
	if (directory)
		{
		while ( (direntry = readdir (directory)) )
			puts (direntry->d_name);
		closedir (directory);
		}
	else
		phoebe_lib_error ("Error reading current directory contents.\n");

	return SUCCESS;
}

char *concatenate_strings (const char *str, ...)
{
	/*
	 * This function concatenates all passed strings into one string. The last
	 * argument to be passed must be null. Note that strcpy () function also
	 * copies the \0 character.
	 */

	va_list args;
	char *out = phoebe_malloc (strlen(str)+1);
	char *s;
	char *change;

	strcpy (out, str);

	va_start (args, str);
	while ( (s = va_arg (args, char *)) )  {
		out = phoebe_realloc (out, strlen (out) + strlen (s) + 1);
		change = &out[strlen(out)];
		strcpy (change, s);
	}

	va_end (args);
	return out;
}

bool atob (char *str)
{
	/*
	 * This is a natural extension to C's ato* family of functions that
	 * converts a string str to a boolean and returns it.
	 */

	if (!str) return FALSE;
	if (str[0] == 'y' || str[0] == 'Y' || strcmp (str, "TRUE") == 0 || strcmp (str, "1") == 0) return TRUE;
	return FALSE;
}

char *phoebe_strdup (const char *s)
{
	/*
	 * The strdup () function is used excessively throughout PHOEBE sources.
	 * Although virtually every single modern C library defines it in the
	 * string.h header file, we supply here a copy just to be sure for the
	 * remaining C libraries.
	 */

	size_t len = strlen (s) + 1;
	void *new = phoebe_malloc (len);
	return (char *) memcpy (new, s, len);
}
