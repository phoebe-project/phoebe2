from phoebe.parameters import *

try:
    import sklearn as _sklearn
    from sklearn.gaussian_process import GaussianProcessRegressor
except ImportError:
    logger.warning("scikit-learn not installed: only required for gaussian processes")
    _use_sklearn = False
else:
    _use_sklearn = True

try:
    import celerite2 as _celerite2
except ImportError:
    logger.warning("celerite2 not installed: only required for gaussian processes")
    _use_celerite2 = False
else:
    _use_celerite2 = True

__all__ = ['handle_gaussian_processes']

_skip_filter_checks = {'check_default': False, 'check_visible': False}

def handle_gaussian_processes(b, model, model_ps, enabled_features, computeparams):

    for ds in model_ps.datasets:
        gp_sklearn_features = b.filter(feature=enabled_features, dataset=ds, kind='gp_sklearn', **_skip_filter_checks).features
        gp_celerite2_features = b.filter(feature=enabled_features, dataset=ds, kind='gp_celerite2', **_skip_filter_checks).features

        if len(gp_sklearn_features)!=0 or len(gp_celerite2_features)!=0:
            # we'll loop over components (for RVs or LPs, for example)
            # get the data we need to fit the GP model
            ds_ps = b.get_dataset(dataset=ds, **_skip_filter_checks)
            xqualifier = {'lp': 'wavelength'}.get(ds_ps.kind, 'times')
            yqualifier = {'lp': 'flux_densities', 'rv': 'rvs', 'lc': 'fluxes'}.get(ds_ps.kind)
            yerrqualifier = {'lp': 'wavelength'}.get(ds_ps.kind, 'sigmas')

            _exclude_phases_enabled = computeparams.get_value(qualifier='gp_exclude_phases_enabled', dataset=ds, **_skip_filter_checks)

            if ds_ps.kind in ['lc']:
                ds_comps = [None]
            else:
                ds_comps = ds_ps.filter(qualifier=xqualifier, check_visible=True).components
            for ds_comp in ds_comps:
                ds_x = ds_ps.get_value(qualifier=xqualifier, component=ds_comp, **_skip_filter_checks)
                model_x = model_ps.get_value(qualifier=xqualifier, dataset=ds, component=ds_comp, **_skip_filter_checks)
                ds_sigmas = ds_ps.get_value(qualifier=yerrqualifier, component=ds_comp, **_skip_filter_checks)
                # ds_sigmas = ds_ps.get_value(qualifier='sigmas', component=ds_comp, **_skip_filter_checks)
                # TODO: do we need to inflate sigmas by lnf?
                if not len(ds_x):
                    # should have been caught by run_checks_compute
                    raise ValueError("gaussian_process requires dataset observations (cannot be synthetic only).  Add observations to dataset='{}' or disable feature={}".format(ds, gp_features))

            residuals, model_y_dstimes = b.calculate_residuals(model=model,
                                            dataset=ds,
                                            component=ds_comp,
                                            return_interp_model=True,
                                            as_quantity=False,
                                            consider_gaussian_process=False)
            model_y = model_ps.get_quantity(qualifier=yqualifier, dataset=ds, component=ds_comp, **_skip_filter_checks)

            gp_kernels = []
            alg_operations = []

            def _load_gps(gp_kernel_classes, gp_features, ds):

                for gp in gp_features:
                    gp_ps = b.filter(feature=gp, context='feature', **_skip_filter_checks)
                    kind = gp_ps.get_value(qualifier='kernel', **_skip_filter_checks)

                    kwargs = {p.qualifier: p.value for p in gp_ps.exclude(qualifier=['kernel', 'enabled']).to_list() if p.is_visible}
                    # TODO: replace this with getting the parameter from compute options
                    if _exclude_phases_enabled:
                        exclude_phase_ranges = computeparams.get_value(qualifier='gp_exclude_phases', dataset=ds, **_skip_filter_checks)
                    else:
                        exclude_phase_ranges = []

                    alg_operations.append(kwargs.pop('alg_operation'))
                    gp_kernels.append(gp_kernel_classes.get(kind)(**kwargs))

                    gp_kernel = gp_kernels[0]
                    for i in range(1, len(gp_kernels)):
                        if alg_operations[i] == 'product':
                            gp_kernel *= gp_kernels[i]
                            # print(gp_kernel)
                        else:
                            gp_kernel += gp_kernels[i]
                            # print(gp_kernel)


                    if len(exclude_phase_ranges) != 0:
                        # get t0, period and exclude_phases
                        ephem = b.get_ephemeris(component='binary', period='period', t0='t0_supconj')
                        t0 = ephem.get('t0', 0.0)
                        period = ephem.get('period', 1.0)

                        phases = np.array(exclude_phase_ranges)

                        # determine extent of data wrt t0
                        i0 = int((t0 - min(ds_x))/period)-1
                        i1 = int((max(ds_x-t0))/period)+1

                        x_new = ds_x
                        residuals_new = residuals
                        sigmas_new = ds_sigmas
                        for i in range(i0,i1+1,1):
                            for j in range(phases.shape[0]):
                                condition = (x_new < t0+(i+phases[j][0])*period) | (x_new > t0+(i+phases[j][1])*period)
                                x_new = x_new[condition]
                                residuals_new = residuals_new[condition]
                                sigmas_new = sigmas_new[condition]

                        gp_x = x_new
                        gp_y = residuals_new
                        gp_yerr = sigmas_new

                    else:
                        gp_x = ds_x
                        gp_y = residuals
                        gp_yerr = ds_sigmas

                return gp_kernel, gp_x, gp_y, gp_yerr

            if len(gp_sklearn_features) > 0:
                gp_kernel_classes = {'constant': _sklearn.gaussian_process.kernels.ConstantKernel,
                                    'white': _sklearn.gaussian_process.kernels.WhiteKernel,
                                    'rbf': _sklearn.gaussian_process.kernels.RBF,
                                    'matern': _sklearn.gaussian_process.kernels.Matern,
                                    'rational_quadratic': _sklearn.gaussian_process.kernels.RationalQuadratic,
                                    'exp_sine_squared': _sklearn.gaussian_process.kernels.ExpSineSquared,
                                    'dot_product': _sklearn.gaussian_process.kernels.DotProduct}

                gp_kernel, gp_x, gp_y, gp_yerr = _load_gps(gp_kernel_classes, gp_sklearn_features, ds)

                gp_regressor = GaussianProcessRegressor(kernel=gp_kernel)
                gp_regressor.fit(gp_x.reshape(-1,1), gp_y)

                # NOTE: .predict can also be called directly to the model times if we want to avoid interpolation altogether
                gp_y = gp_regressor.predict(ds_x.reshape(-1,1), return_std=False)

            if len(gp_celerite2_features) > 0:
                gp_kernel_classes = {'sho': _celerite2.terms.SHOTerm,
                                    'rotation': _celerite2.terms.RotationTerm,
                                    'matern32': _celerite2.terms.Matern32Term
                                    }
                gp_kernel, gp_x, gp_y, gp_yerr = _load_gps(gp_kernel_classes, gp_celerite2_features, ds)

                gp = _celerite2.GaussianProcess(gp_kernel, mean=0.0)
                gp.compute(gp_x, yerr=gp_yerr)
                gp_y = gp.predict(gp_y, t=ds_x, return_var=False)


            # store just the GP component in the model PS as well
            gp_param = FloatArrayParameter(qualifier='gps', value=gp_y, default_unit=model_y.unit, readonly=True, description='GP contribution to the model {}'.format(yqualifier))
            y_nogp_param = FloatArrayParameter(qualifier='{}_nogps'.format(yqualifier), value=model_y_dstimes, default_unit=model_y.unit, readonly=True, description='{} before adding gps'.format(yqualifier))
            if len(ds_x) != len(model_x) or not np.all(ds_x == model_x):
                logger.warning("model for dataset='{}' resampled at dataset times when adding GPs".format(ds))
                model_ps.set_value(qualifier=xqualifier, dataset=ds, component=ds_comp, value=ds_x, ignore_readonly=True, **_skip_filter_checks)

            b._attach_params([gp_param, y_nogp_param], dataset=ds, check_copy_for=False, **metawargs)

            # update the model to include the GP contribution
            model_ps.set_value(qualifier=yqualifier, value=model_y_dstimes+gp_y, dataset=ds, component=ds_comp, ignore_readonly=True, **_skip_filter_checks)

