#ifndef PHOEBE_SPOTS_H
	#define PHOEBE_SPOTS_H 1

typedef struct PHOEBE_spot {
	bool   active;
	bool   tba;
	int    src;
	double lat;
	double long;
	double rad;
	double temp;
} PHOEBE_spot;

typedef struct PHOEBE_spot_list {
	PHOEBE_spot *spot;
	struct PHOEBE_spot_list *next;
} PHOEBE_spot_list;

extern PHOEBE_spot_list *PHOEBE_spots;

PHOEBE_spot *phoebe_spot_new ();
PHOEBE_spot *phoebe_spot_add (bool active, bool tba, int src, double lat, double long, double rad, double temp);

int phoebe_spot_set_active (PHOEBE_spot *spot, bool   active);
int phoebe_spot_set_tba    (PHOEBE_spot *spot, bool   tba);
int phoebe_spot_set_src    (PHOEBE_spot *spot, int    src);
int phoebe_spot_set_lat    (PHOEBE_spot *spot, double lat);
int phoebe_spot_set_long   (PHOEBE_spot *spot, double long);
int phoebe_spot_set_rad    (PHOEBE_spot *spot, double rad);
int phoebe_spot_set_temp   (PHOEBE_spot *spot, double temp);

int phoebe_spot_get_active (PHOEBE_spot *spot, bool   *active);
int phoebe_spot_get_tba    (PHOEBE_spot *spot, bool   *tba);
int phoebe_spot_get_src    (PHOEBE_spot *spot, int    *src);
int phoebe_spot_get_lat    (PHOEBE_spot *spot, double *lat);
int phoebe_spot_get_long   (PHOEBE_spot *spot, double *long);
int phoebe_spot_get_rad    (PHOEBE_spot *spot, double *rad);
int phoebe_spot_get_temp   (PHOEBE_spot *spot, double *temp);

int phoebe_spot_free       (PHOEBE_spot *spot);

phoebe_spot_pass_to_wd     (PHOEBE_spot *spot, int no);

#endif
