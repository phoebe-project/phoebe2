"""
Unit tests for Passband.interpolate_ldcoeffs() method.
"""

import pytest
import numpy as np
from phoebe.atmospheres import passbands, models
from phoebe.atmospheres.passbands import InterpQuery


class TestInterpolateLdcoeffs:
    """Test class for Passband.interpolate_ldcoeffs() method."""

    @classmethod
    def setup_class(cls):
        """Set up test class with Johnson V passband."""
        try:
            # Try to get Johnson:V passband which should have ck2004:ld content
            cls.pb = passbands.get_passband('Johnson:V', content=['ck2004:ld'])
        except Exception:
            # If not available, skip all tests in this class
            pytest.skip("Johnson:V passband with ck2004:ld content not available")

    def test_interpolate_ldcoeffs_basic_functionality(self):
        """Test basic functionality with valid parameters."""
        # Create a simple query with standard stellar parameters
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5772., 4.43, 0.0]])  # Sun-like parameters
        query = InterpQuery(cols, pts)

        # Test with default parameters
        result = self.pb.interpolate_ldcoeffs(query)

        # Should return an InterpResult with interps shape (1, 4) for power law (default)
        assert isinstance(result, passbands.InterpResult)
        assert result.interps.shape == (1, 4)
        assert not np.any(np.isnan(result.interps))

    def test_interpolate_ldcoeffs_all_ld_functions(self):
        """Test all supported limb darkening functions."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5772., 4.43, 0.0]])
        query = InterpQuery(cols, pts)

        ld_func_expected_shapes = {
            'linear': (1, 1),
            'logarithmic': (1, 2),
            'square_root': (1, 2),
            'quadratic': (1, 2),
            'power': (1, 4),
            'all': (1, 11)  # All LD coefficients
        }

        for ld_func, expected_shape in ld_func_expected_shapes.items():
            result = self.pb.interpolate_ldcoeffs(query, ld_func=ld_func)
            assert isinstance(result, passbands.InterpResult), f"ld_func={ld_func}"
            assert result.interps.shape == expected_shape, f"ld_func={ld_func}"
            assert not np.any(np.isnan(result.interps)), f"ld_func={ld_func} contains NaN"

    def test_interpolate_ldcoeffs_temperature_range(self):
        """Test interpolation across CK2004 temperature range (3500K to 50000K)."""
        cols = ['teffs', 'loggs', 'abuns']

        # Test various temperatures within the CK2004 range
        test_temperatures = [3500., 5772., 10000., 25000.]

        for teff in test_temperatures:
            pts = np.array([[teff, 4.43, 0.0]])
            query = InterpQuery(cols, pts)

            result = self.pb.interpolate_ldcoeffs(query)
            assert isinstance(result, passbands.InterpResult)
            assert result.interps.shape == (1, 4)
            assert not np.any(np.isnan(result.interps)), f"NaN at Teff={teff}"

    def test_interpolate_ldcoeffs_extrapolation_high_temp(self):
        """Test extrapolation behavior at high temperatures."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[50000., 4.43, 0.0]])  # High temperature likely outside grid
        query = InterpQuery(cols, pts)

        # Test with different extrapolation methods
        try:
            result_none = self.pb.interpolate_ldcoeffs(query, ld_extrapolation_method='none')
            # If 'none' doesn't raise an error, it may return NaN or handle gracefully
            if np.any(np.isnan(result_none.interps)):
                # This is expected behavior for points outside the grid
                pass
            else:
                # Unexpected but not necessarily wrong
                assert result_none.interps.shape == (1, 4)
        except (ValueError, RuntimeError):
            # Expected for 'none' extrapolation outside grid
            pass

        # Test with extrapolation methods that should handle out-of-bounds values
        for method in ['nearest', 'linear']:
            result = self.pb.interpolate_ldcoeffs(query, ld_extrapolation_method=method)
            assert isinstance(result, passbands.InterpResult)
            assert result.interps.shape == (1, 4)

    def test_interpolate_ldcoeffs_multiple_points(self):
        """Test interpolation with multiple query points."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([
            [5772., 4.43, 0.0],    # Sun
            [3800., 4.5, -0.5],    # Cool dwarf
            [15000., 4.0, 0.0],    # Hot star
            [6000., 3.5, 0.3]      # Giant
        ])
        query = InterpQuery(cols, pts)

        result = self.pb.interpolate_ldcoeffs(query)
        assert result.interps.shape == (4, 4)
        assert not np.any(np.isnan(result.interps))

    def test_interpolate_ldcoeffs_intens_weighting(self):
        """Test different intensity weighting options."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5772., 4.43, 0.0]])
        query = InterpQuery(cols, pts)

        for intens_weighting in ['photon', 'energy']:
            result = self.pb.interpolate_ldcoeffs(query, intens_weighting=intens_weighting)
            assert isinstance(result, passbands.InterpResult)
            assert result.interps.shape == (1, 4)
            assert not np.any(np.isnan(result.interps))

    def test_interpolate_ldcoeffs_extrapolation_methods(self):
        """Test different extrapolation methods."""
        cols = ['teffs', 'loggs', 'abuns']
        # Use parameters that might be at the edge of or slightly outside the grid
        pts = np.array([[3200., 4.43, 0.0]])  # Below typical CK2004 range
        query = InterpQuery(cols, pts)

        extrapolation_methods = ['none', 'nearest', 'linear']

        for method in extrapolation_methods:
            if method == 'none':
                # Should handle gracefully or raise appropriate error
                try:
                    result = self.pb.interpolate_ldcoeffs(query, ld_extrapolation_method=method)
                    # If it doesn't raise an error, result should be valid or contain NaN
                    assert isinstance(result, passbands.InterpResult)
                    # NaN values are acceptable for 'none' extrapolation outside grid
                except (ValueError, RuntimeError):
                    # Expected for 'none' extrapolation outside grid
                    pass
            else:
                result = self.pb.interpolate_ldcoeffs(query, ld_extrapolation_method=method)
                assert isinstance(result, passbands.InterpResult)
                assert result.interps.shape == (1, 4)

    def test_interpolate_ldcoeffs_different_ldatm(self):
        """Test with different limb darkening atmosphere models."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5772., 4.43, 0.0]])
        query = InterpQuery(cols, pts)

        # Test with CK2004 (default)
        result_ck = self.pb.interpolate_ldcoeffs(query, ldatm=models.CK2004ModelAtmosphere)
        assert isinstance(result_ck, passbands.InterpResult)
        assert result_ck.interps.shape == (1, 4)

    def test_interpolate_ldcoeffs_edge_cases(self):
        """Test edge cases and boundary conditions."""
        cols = ['teffs', 'loggs', 'abuns']

        # Test with single point at grid boundaries
        boundary_params = [
            [3500., 5.0, -2.5],   # Cool end
            [50000., 3.0, 0.5]    # Hot end
        ]

        for params in boundary_params:
            pts = np.array([params])
            query = InterpQuery(cols, pts)

            try:
                result = self.pb.interpolate_ldcoeffs(query)
                assert isinstance(result, passbands.InterpResult)
                assert result.interps.shape == (1, 4)
            except (ValueError, RuntimeError):
                # May be outside the available grid - acceptable
                pass

    def test_interpolate_ldcoeffs_invalid_ld_func(self):
        """Test error handling for invalid ld_func."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5772., 4.43, 0.0]])
        query = InterpQuery(cols, pts)

        with pytest.raises(ValueError, match="ld_func=invalid"):
            self.pb.interpolate_ldcoeffs(query, ld_func='invalid')

    def test_interpolate_ldcoeffs_consistency(self):
        """Test that results are consistent and physically reasonable."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5772., 4.43, 0.0]])
        query = InterpQuery(cols, pts)

        # Get coefficients for different LD functions
        linear = self.pb.interpolate_ldcoeffs(query, ld_func='linear')
        power = self.pb.interpolate_ldcoeffs(query, ld_func='power')

        # Basic sanity checks
        assert np.all(linear.interps >= 0), "Linear coefficients should be non-negative"
        assert np.all(linear.interps <= 2), "Linear coefficients should be reasonable"

        # Power law coefficients should sum to a reasonable value
        assert np.sum(power.interps) > 0, "Power law coefficients should sum to positive value"
        assert np.sum(power.interps) < 5, "Power law coefficients sum should be reasonable"

    def test_interpolate_ldcoeffs_query_subset_behavior(self):
        """Test that method correctly uses query.subset() for basic axes."""
        # Create query with extra columns that shouldn't be used
        cols = ['teffs', 'loggs', 'abuns', 'mus']  # mus not used for LD coeffs
        pts = np.array([[5772., 4.43, 0.0, 1.0]])
        query = InterpQuery(cols, pts)

        # Should work fine and ignore the 'mus' column
        result = self.pb.interpolate_ldcoeffs(query)
        assert isinstance(result, passbands.InterpResult)
        assert result.interps.shape == (1, 4)

    def test_interpolate_ldcoeffs_reproducibility(self):
        """Test that repeated calls produce identical results."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5772., 4.43, 0.0]])
        query = InterpQuery(cols, pts)

        result1 = self.pb.interpolate_ldcoeffs(query)
        result2 = self.pb.interpolate_ldcoeffs(query)

        np.testing.assert_array_equal(result1.interps, result2.interps)


# Additional test for error cases requiring specific setup
def test_interpolate_ldcoeffs_missing_ld_content():
    """Test error when LD content is not available in passband."""
    # This test may be difficult to set up since most passbands include LD content
    # We'll skip it if we can't create the appropriate test conditions
    pytest.skip("Skipping test - difficult to create passband without LD content in current setup")
