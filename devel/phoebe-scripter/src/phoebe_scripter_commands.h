#ifndef PHOEBE_SCRIPTER_COMMANDS_H
	#define PHOEBE_SCRIPTER_COMMANDS_H 1

#include "phoebe_scripter_ast.h"

int scripter_register_all_commands ();

scripter_ast_value scripter_open_parameter_file           (scripter_ast_list *args);
scripter_ast_value scripter_save_parameter_file           (scripter_ast_list *args);
scripter_ast_value scripter_set_parameter_value           (scripter_ast_list *args);
scripter_ast_value scripter_set_parameter_limits          (scripter_ast_list *args);
scripter_ast_value scripter_set_parameter_step            (scripter_ast_list *args);
scripter_ast_value scripter_mark_for_adjustment           (scripter_ast_list *args);
scripter_ast_value scripter_add_constraint                (scripter_ast_list *args);
scripter_ast_value scripter_create_wd_lci_file            (scripter_ast_list *args);
scripter_ast_value scripter_plot_lc_using_gnuplot         (scripter_ast_list *args);
scripter_ast_value scripter_plot_rv_using_gnuplot         (scripter_ast_list *args);
scripter_ast_value scripter_plot_spectrum_using_gnuplot   (scripter_ast_list *args);
scripter_ast_value scripter_plot_eb_using_gnuplot         (scripter_ast_list *args);
scripter_ast_value scripter_plot_using_gnuplot            (scripter_ast_list *args);
scripter_ast_value scripter_set_spectra_repository        (scripter_ast_list *args);
scripter_ast_value scripter_array                         (scripter_ast_list *args);
scripter_ast_value scripter_curve                         (scripter_ast_list *args);
scripter_ast_value scripter_column                        (scripter_ast_list *args);
scripter_ast_value scripter_spectrum                      (scripter_ast_list *args);
scripter_ast_value scripter_format                        (scripter_ast_list *args);
scripter_ast_value scripter_substr                        (scripter_ast_list *args);
scripter_ast_value scripter_defined                       (scripter_ast_list *args);
scripter_ast_value scripter_get_parameter_value           (scripter_ast_list *args);
scripter_ast_value scripter_minimize_using_nms            (scripter_ast_list *args);
scripter_ast_value scripter_minimize_using_dc             (scripter_ast_list *args);
scripter_ast_value scripter_adopt_minimizer_results       (scripter_ast_list *args);
scripter_ast_value scripter_compute_light_levels          (scripter_ast_list *args);
scripter_ast_value scripter_prompt                        (scripter_ast_list *args);
scripter_ast_value scripter_transform_hjd_to_phase        (scripter_ast_list *args);
scripter_ast_value scripter_transform_flux_to_magnitude   (scripter_ast_list *args);
scripter_ast_value scripter_compute_lc                    (scripter_ast_list *args);
scripter_ast_value scripter_compute_critical_potentials   (scripter_ast_list *args);
scripter_ast_value scripter_set_lc_properties             (scripter_ast_list *args);
scripter_ast_value scripter_compute_rv                    (scripter_ast_list *args);
scripter_ast_value scripter_compute_mesh                  (scripter_ast_list *args);
scripter_ast_value scripter_compute_chi2                  (scripter_ast_list *args);
scripter_ast_value scripter_get_ld_coefficients           (scripter_ast_list *args);
scripter_ast_value scripter_set_spectrum_properties       (scripter_ast_list *args);
scripter_ast_value scripter_get_spectrum_from_repository  (scripter_ast_list *args);
scripter_ast_value scripter_get_spectrum_from_file        (scripter_ast_list *args);
scripter_ast_value scripter_apply_doppler_shift           (scripter_ast_list *args);
scripter_ast_value scripter_apply_rotational_broadening   (scripter_ast_list *args);
scripter_ast_value scripter_merge_spectra                 (scripter_ast_list *args);
scripter_ast_value scripter_multiply_spectra              (scripter_ast_list *args);
scripter_ast_value scripter_broaden_spectrum              (scripter_ast_list *args);
scripter_ast_value scripter_crop_spectrum                 (scripter_ast_list *args);
scripter_ast_value scripter_resample_spectrum             (scripter_ast_list *args);
scripter_ast_value scripter_integrate_spectrum            (scripter_ast_list *args);

scripter_ast_value scripter_compute_perr0_phase           (scripter_ast_list *args);

#endif
