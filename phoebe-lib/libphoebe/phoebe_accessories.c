#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "phoebe_accessories.h"
#include "phoebe_build_config.h"
#include "phoebe_calculations.h"
#include "phoebe_configuration.h"
#include "phoebe_error_handling.h"
#include "phoebe_types.h"

/**
 * SECTION:phoebe_accessories
 * @title: PHOEBE accessories
 * @short_description: functions that facilitate common type manipulation
 *
 * These are the functions that facilitate common type manipulation.
 * They mostly pertain to I/O and string handling.
 */

int phoebe_open_directory (DIR **dir, const char *dirname)
{
	/**
	 * phoebe_open_directory:
	 * @dir: placeholder for the directory stream pointer
	 * @dirname: path to the directory
	 * 
	 * This is a wrapper to opendir() for opening the directory @dirname
	 * gracefully. In case opendir() is successful, a pointer to the directory
	 * stream @dir is set and %SUCCESS is returned. Otherwise @dir is set to
	 * %NULL and error code is returned.
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

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
	/**
	 * phoebe_close_directory:
	 * @dir: pointer to the directory stream
	 * 
	 * This is a wrapper to closedir() for closing the directory stream
	 * @dir gracefully. In case closedir() is successful, %SUCCESS is returned;
	 * otherwise error code is returned.
	 * 
	 * Returns: #PHOEBE_error_code.
	 */

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

bool phoebe_filename_exists (const char *filename)
{
	/**
	 * phoebe_filename_exists:
	 * @filename: path to the file to be checked for existence
	 *
	 * Checks for existence of the file @filename. Returns %YES if it exists
	 * and %NO if it does not exist.
	 * 
	 * Returns: #bool.
	 */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0) return TRUE;

	return FALSE;
}

bool phoebe_filename_has_write_permissions (const char *filename)
{
	/**
	 * phoebe_filename_has_write_permissions:
	 * @filename: path to the file to be checked for permissions
	 *
	 * Checks for write permissions of the file @filename. Returns %YES if it
	 * has write permissions and %NO if it does not.
	 * 
	 * Returns: #bool.
	 */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);

	/* On linux we can use geteuid() and getegid() to check for permissions: */

#if defined HAVE_GETEUID && defined HAVE_GETGUID
	if (check == 0) {
		if (( (log.st_uid == geteuid ()) && (log.st_mode & S_IWUSR) ) ||
			( (log.st_gid == getegid ()) && (log.st_mode & S_IWGRP) ) ||
			( (log.st_mode & S_IWOTH) ))
			return TRUE;

		return FALSE;
	}

	/* On windows, however, there is no counterpart, so it suffices to do: */

#else
	if (check == 0 && (log.st_mode & S_IWRITE) )
		return TRUE;
#endif

	return FALSE;
}

bool phoebe_filename_has_read_permissions (const char *filename)
{
	/**
	 * phoebe_filename_has_read_permissions:
	 * @filename: path to the file to be checked for permissions
	 *
	 * Checks for read permissions of the file @filename. Returns %YES if it
	 * has read permissions and %NO if it does not.
	 * 
	 * Returns: #bool.
	 */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);

	/* On linux we can use geteuid() and getegid() to check for permissions: */

#if defined HAVE_GETEUID && defined HAVE_GETGUID
	if (check == 0) {
		if (( (log.st_uid == geteuid ()) && (log.st_mode & S_IRUSR) ) ||
			( (log.st_gid == getegid ()) && (log.st_mode & S_IRGRP) ) ||
			( (log.st_mode & S_IROTH) ))
			return TRUE;

		return FALSE;
	}

	/* On windows, however, there is no counterpart, so it suffices to do: */

#else
	if (check == 0 && (log.st_mode & S_IREAD) )
		return TRUE;
#endif

	return FALSE;
}

bool phoebe_filename_has_execute_permissions (const char *filename)
{
	/**
	 * phoebe_filename_has_execute_permissions:
	 * @filename: path to the file to be checked for permissions
	 *
	 * Checks for execute permissions of the file @filename. Returns %YES if it
	 * has execute permissions and %NO if it does not.
	 * 
	 * Returns: #bool.
	 */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);

	/* On linux we can use geteuid() and getegid() to check for permissions: */

#if defined HAVE_GETEUID && defined HAVE_GETGUID
	if (check == 0) {
		if (( (log.st_uid == geteuid ()) && (log.st_mode & S_IXUSR) ) ||
			( (log.st_gid == getegid ()) && (log.st_mode & S_IXGRP) ) ||
			( (log.st_mode & S_IXOTH) ))
			return TRUE;

		return FALSE;
	}

	/* On windows, however, there is no counterpart, so it suffices to do: */

#else
	if (check == 0 && (log.st_mode & S_IEXEC) )
		return TRUE;
#endif

	return FALSE;
}

bool phoebe_filename_has_full_permissions (const char *filename)
{
	/**
	 * phoebe_filename_has_full_permissions:
	 * @filename: path to the file to be checked for permissions
	 *
	 * Checks for full (read, write, execute) permissions of the file @filename.
	 * Returns %YES if it has full permissions and %NO if it does not.
	 * 
	 * Returns: #bool.
	 */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);

	/* On linux we can use geteuid() and getegid() to check for permissions: */

#if defined HAVE_GETEUID && defined HAVE_GETGUID
	if (check == 0) {
		if (( (log.st_uid == geteuid ()) && (log.st_mode & S_IRUSR) && (log.st_mode & S_IWUSR) && (log.st_mode & S_IXUSR) ) ||
			( (log.st_gid == getegid ()) && (log.st_mode & S_IRGRP) && (log.st_mode & S_IWGRP) && (log.st_mode & S_IXGRP) ) ||
			( (log.st_mode & S_IROTH) && (log.st_mode & S_IWOTH) && (log.st_mode & S_IXOTH) ) )
			return TRUE;

		return FALSE;
	}

	/* On windows, however, there is no counterpart, so it suffices to do: */

#else
	if (check == 0 && (log.st_mode & S_IREAD) && (log.st_mode & S_IWRITE) && (log.st_mode & S_IEXEC))
		return TRUE;
#endif

	return FALSE;
}

bool phoebe_filename_is_directory (const char *filename)
{
	/**
	 * phoebe_filename_is_directory:
	 * @filename: path to the file to be checked for type
	 *
	 * Checks whether the file @filename is a directory. Returns %YES if it
	 * is a directory and %NO if it is not.
	 * 
	 * Returns: #bool.
	 */

	struct stat log;
	int check;

	if (!filename) return FALSE;

	check = stat (filename, &log);
	if (check == 0) {
		if (log.st_mode & S_IFDIR) return TRUE;
		return FALSE;
	}

	return FALSE;
}

bool phoebe_filename_is_regular_file (const char *filename)
{
	/**
	 * phoebe_filename_is_regular_file:
	 * @filename: path to the file to be checked for type
	 *
	 * Checks whether the file @filename is a regular file. Returns %YES if it
	 * is a regular file and %NO if it is not.
	 * 
	 * Returns: #bool.
	 */

	struct stat log;
	int check;

	if (filename == NULL) return FALSE;

	check = stat (filename, &log);
	if (check == 0) {
		if (log.st_mode & S_IFREG) return TRUE;
		return FALSE;
	}

	return FALSE;
}

char *phoebe_get_current_working_directory ()
{
	/**
	 * phoebe_get_current_working_directory:
	 *
	 * Queries the system for the current working directory and returns it.
	 * This function allocates the memory to hold the path; the calling
	 * function has to free the allocated memory after it is done using it.
	 * 
	 * Returns: path to the current working directory.
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

char *phoebe_resolve_relative_filename (char *filename)
{
	/**
	 * phoebe_resolve_relative_filename:
	 * @filename: relative path to be resolved (made absolute)
	 *
	 * This function takes a relative filename and resolves it to absolute; it
	 * allocates the memory for the string itself, the calling function has to
	 * free it.
	 * 
	 * Returns: #bool.
	 */

	char *cwd;
	char *abs;

	/* Is the filename already absolute?                                        */
	if (filename[0] == '/') return filename;

	cwd = phoebe_get_current_working_directory ();
	if (!cwd) {
		phoebe_lib_error ("cannot get a path to the current directory.");
		return NULL;
	}
	abs = phoebe_malloc (strlen (cwd) + strlen (filename) + 2); /* for / and \0 */
	sprintf (abs, "%s/%s", cwd, filename);
	free (cwd);

	return abs;
}

int phoebe_list_directory_contents (char *dir)
{
	/**
	 * phoebe_list_directory_contents:
	 * @dir: path to the directory to be listed
	 *
	 * This function lists the contents of the directory @dir on screen. It
	 * is used mostly for debugging purposes.
	 * 
	 * Returns: #bool.
	 */

	DIR *directory;
	struct dirent *direntry;
	
	directory = opendir (dir);
	if (directory) {
		while ( (direntry = readdir (directory)) )
			puts (direntry->d_name);
		closedir (directory);
	}
	else
		phoebe_lib_error ("Error reading current directory contents.\n");

	return SUCCESS;
}

char *phoebe_concatenate_strings (const char *str, ...)
{
	/**
	 * phoebe_concatenate_strings:
	 * @str: first string to be concatenated
	 * @...: a %NULL-terminated list of strings to be concatenated
	 *
	 * Concatenates all passed strings into one string. The last argument to
	 * be passed must be %NULL. The user should free the returned string when
	 * it is no longer necessary.
	 * 
	 * Returns: a newly allocated, concatenated string.
	 */

	va_list args;
	char *out = phoebe_malloc (strlen(str)+1);
	char *s;
	char *change;

	strcpy (out, str);

	va_start (args, str);
	while ( (s = va_arg (args, char *)) ) {
		out = phoebe_realloc (out, strlen (out) + strlen (s) + 1);
		change = &out[strlen(out)];
		strcpy (change, s);
	}

	va_end (args);
	return out;
}

char *phoebe_clean_data_line (char *line)
{
	/**
	 * phoebe_clean_data_line:
	 * @line: input string to be cleaned
	 *
	 * Takes a string, cleans it of all comment delimeters, spaces, tabs and
	 * newlines, and copies the contents to a newly allocated string. The
	 * original string is not modified in any way.
	 *
	 * Returns: clean string.
	 */

	/* This function has been thorougly tested. */

	char *value, *start, *stop;

	if (!line)
		return NULL;

	if (strlen(line) == 0)
		return NULL;

	start = line;
	while (*start != '\0' && (*start == '\n' || *start == '\t' || *start == ' ' || *start == 13))
		start++;

	if ((stop = strchr (line, '#'))) {
		if (stop > start)
			stop--;
	}
	else
		stop = &line[strlen(line)];

	while (stop != start && (*stop == '\n' || *stop == '\t' || *stop == ' ' || *stop == 13))
		stop--;

	phoebe_debug ("start: %c; stop: %c; length: %ld\n", *start, *stop, stop-start+1);
	if (start == stop)
		return NULL;

	value = phoebe_malloc ((stop-start+2)*sizeof(*value));
	memcpy (value, start, (stop-start+1)*sizeof(*value));
	value[stop-start+1] = '\0';

	return value;
}

bool atob (char *str)
{
	/**
	 * atob:
	 * @str: string to be converted to boolean
	 *
	 * This is a natural extension to C's ato* family of functions that
	 * converts a string @str to a boolean.
	 * 
	 * Returns: #bool.
	 */

	if (!str) return FALSE;
	if (str[0] == 'y' || str[0] == 'Y' || strcmp (str, "TRUE") == 0 || strcmp (str, "1") == 0) return TRUE;
	return FALSE;
}

char *phoebe_strdup (const char *s)
{
	/**
	 * phoebe_strdup:
	 * @s: string to be duplicated
	 * 
	 * The strdup() function is used excessively throughout PHOEBE sources.
	 * Although virtually every single modern C library defines it in the
	 * string.h header file, we supply here a copy just to be sure for the
	 * remaining C libraries.
	 *
	 * Returns: a duplicate of string @s.
	 */

	size_t len = strlen (s) + 1;
	void *dup = phoebe_malloc (len);
	return (char *) memcpy (dup, s, len);
}

char *phoebe_readline (FILE *stream)
{
	/**
	 * phoebe_readline:
	 * @stream: file stream from which to read a line
	 *
	 * Reads a line from the file stream @stream. The memory to hold the line
	 * is allocated dynamically, so there is no limit to the length of the
	 * line. It also means that the user must not allocate the string before
	 * the call to this function, and that the memory should be freed after
	 * its use.
	 *
	 * Returns: a string containing the read line.
	 */

	int len = 256;
	char *line, *cont;

	line = phoebe_malloc (len * sizeof (*line));
	if (!fgets (line, len, stream)) return NULL;
	cont = &(line[0]);

	/* The following part takes care of lines longer than 256 characters: */
	while (!strchr (cont, '\n') && !strchr (cont, EOF)) {
		len *= 2;
		line = phoebe_realloc (line, len * sizeof (*line));
		cont = &(line[len/2-1]);
		if (!fgets (cont, len/2+1, stream)) break;
		if (feof (stream)) break;
	}

	return line;
}

char *phoebe_create_temp_filename (char *templ)
{
	/**
	 * phoebe_create_temp_filename:
	 * @templ: filename template; it must end with XXXXXX
	 *
	 * Creates a unique temporary filename in the directory #PHOEBE_TEMP_DIR.
	 * If a unique filename cannot be found, or if @templ is invalid, #NULL
	 * is returned. The calling function should free the returned string.
	 *
	 * Returns: string with a unique filename.
	 */

	int fd = -1;
	char *tmpdir, *tmpfname, *check;

	phoebe_config_entry_get ("PHOEBE_TEMP_DIR", &tmpdir);
	tmpfname = phoebe_concatenate_strings (tmpdir, "/", templ, NULL);

#ifdef __MINGW32__
	check = mktemp (tmpfname);
	if (!check || strlen (check) == 0) {
		free (tmpfname);
		return NULL;
	};
	FILE *f;
	if ((f = fopen(tmpfname, "w")) == -1) {
		free (tmpfname);
		return NULL;
	};
	fclose(f);
#else
	if ((fd = mkstemp (tmpfname)) == -1) {
		free (tmpfname);
		return NULL;
	};

	close(fd);
#endif
	return tmpfname;
}

long int phoebe_seed ()
{
	/**
	 * phoebe_seed:
	 *
	 * Computes a random seed for the RNG that will be different for every
	 * phoebe invocation. This way running phoebe many times a second (i.e.
	 * for cluster operations) won't result in the same seed. This seeding
	 * snippet takes the clock state, time state and process ID, shuffles
	 * them and returns a pseudorandom seed.
	 *
	 * Returns: random seed.
	 */

	long int a, b, c;

	a = clock ();
	b = time (0);
	c = getpid ();

    a=a-b;  a=a-c;  a=a^(c >> 13);
    b=b-c;  b=b-a;  b=b^(a << 8);
    c=c-a;  c=c-b;  c=c^(b >> 13);
    a=a-b;  a=a-c;  a=a^(c >> 12);
    b=b-c;  b=b-a;  b=b^(a << 16);
    c=c-a;  c=c-b;  c=c^(b >> 5);
    a=a-b;  a=a-c;  a=a^(c >> 3);
    b=b-c;  b=b-a;  b=b^(a << 10);
    c=c-a;  c=c-b;  c=c^(b >> 15);

	return c;
}
