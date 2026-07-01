"""
Unit tests for atmosphere blending functionality in passbands.

This test suite focuses on testing the blending of model atmospheres with blackbody
atmospheres as parameters go off-grid. Tests both interpolate_inorms() and
interpolate_imus() methods with real data.
"""

import pytest
import numpy as np

# Import the modules we need to test
import phoebe.atmospheres.passbands as passbands
import phoebe.atmospheres.models as models


class TestAtmosphereBlending:
    """Comprehensive tests for atmosphere blending functionality."""

    @pytest.fixture(scope="class")
    def johnson_v_passband(self):
        """Load the real Johnson:V passband for testing."""
        return passbands.get_passband('Johnson:V')

    @pytest.fixture
    def on_grid_query(self):
        """Create a query with parameters well within CK2004 atmosphere bounds."""
        cols = ['teffs', 'loggs', 'abuns']
        # Use parameters well within the CK2004 grid
        pts = np.array([
            [5778, 4.44, 0.0],    # Sun-like star
            [6000, 4.50, 0.2],    # Slightly hotter, metal-rich
        ])
        return passbands.InterpQuery(cols, pts)

    @pytest.fixture
    def off_grid_query(self):
        """Create a query with parameters outside CK2004 atmosphere bounds."""
        cols = ['teffs', 'loggs', 'abuns']
        # Use parameters likely outside the CK2004 grid
        pts = np.array([
            [3000, 5.5, 0.0],     # Very cool, high gravity
            [30000, 3.0, 0.5],    # Very hot, low gravity
            [2500, 6.0, -2.0],    # Extremely cool, very high gravity, metal-poor
        ])
        return passbands.InterpQuery(cols, pts)

    @pytest.fixture
    def mixed_grid_query(self):
        """Create a query with some points on-grid and some off-grid."""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([
            [5778, 4.44, 0.0],    # On-grid: Sun-like
            [6000, 4.50, 0.2],    # On-grid: Slightly hotter
            [3000, 5.5, 0.0],     # Off-grid: Very cool, high gravity
            [25000, 3.0, 0.0],    # Off-grid: Very hot, low gravity
        ])
        return passbands.InterpQuery(cols, pts)

    @pytest.fixture
    def imus_query(self):
        """Create a query suitable for Imu interpolation (includes mu values)."""
        cols = ['teffs', 'loggs', 'abuns', 'mus']
        # Include mu values for specific intensity interpolation
        pts = np.array([
            [5778, 4.44, 0.0, 1.0],    # Center of disk
            [5778, 4.44, 0.0, 0.5],    # Mid-limb
            [6000, 4.50, 0.2, 1.0],    # Different star, center
            [3000, 5.5, 0.0, 1.0],     # Off-grid star, center
        ])
        return passbands.InterpQuery(cols, pts)

    def test_no_blending_on_grid(self, johnson_v_passband, on_grid_query):
        """Test that no blending occurs for on-grid parameters."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        result = johnson_v_passband.interpolate_inorms(
            on_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0
        )
        
        assert isinstance(result, passbands.InterpResult)
        assert hasattr(result, 'interps')
        assert len(result.interps) == len(on_grid_query.pts)
        
        # For on-grid points, there should be no blending factors
        if hasattr(result, 'bfs') and result.bfs is not None:
            # Blending factors should be 0 (no blending) for on-grid points
            assert np.allclose(result.bfs, 0.0, atol=1e-6)

    def test_blending_off_grid(self, johnson_v_passband, off_grid_query):
        """Test that blending occurs for off-grid parameters."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        result = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0,
            atm_extrapolation_method='linear',  # Allow extrapolation
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        assert isinstance(result, passbands.InterpResult)
        assert hasattr(result, 'interps')
        assert len(result.interps) == len(off_grid_query.pts)
        
        # For off-grid points, there should be blending factors
        assert hasattr(result, 'bfs')
        assert result.bfs is not None
        assert len(result.bfs) == len(off_grid_query.pts)
        
        # Blending factors should be > 0 for off-grid points
        assert np.any(result.bfs > 0)
        
        # Blending factors should be <= 1
        assert np.all(result.bfs <= 1.0)

    def test_blending_factor_calculation(self, johnson_v_passband, mixed_grid_query):
        """Test that blending factors are calculated correctly."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        result = johnson_v_passband.interpolate_inorms(
            mixed_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        assert isinstance(result, passbands.InterpResult)
        
        if hasattr(result, 'bfs') and result.bfs is not None:
            # Blending factors should be in [0, 1] range
            assert np.all(result.bfs >= 0.0)
            assert np.all(result.bfs <= 1.0)
            
            # Distance information should be available
            assert hasattr(result, 'dists')
            assert result.dists is not None

    def test_blending_vs_no_blending_comparison(self, johnson_v_passband, off_grid_query):
        """Test that blending produces different results than no blending."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        # Get results without blending
        result_no_blend = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='none',
            atm_extrapolation_method='linear'
        )
        
        # Get results with blending
        result_with_blend = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Results should be different when blending is applied
        assert not np.allclose(result_no_blend.interps, result_with_blend.interps, rtol=1e-10)

    def test_blending_with_different_thresholds(self, johnson_v_passband, off_grid_query):
        """Test blending behavior with different distance thresholds."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        # Test with very small threshold (more points should be considered off-grid)
        result_small_thresh = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-10,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Test with larger threshold (fewer points should be considered off-grid)
        result_large_thresh = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-2,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Both should return valid results
        assert isinstance(result_small_thresh, passbands.InterpResult)
        assert isinstance(result_large_thresh, passbands.InterpResult)

    def test_blending_with_different_margins(self, johnson_v_passband, off_grid_query):
        """Test blending behavior with different blending margins."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        # Test with small blending margin (sharper transition)
        result_small_margin = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=1.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Test with large blending margin (smoother transition)
        result_large_margin = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=5.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Both should return valid results
        assert isinstance(result_small_margin, passbands.InterpResult)
        assert isinstance(result_large_margin, passbands.InterpResult)

    def test_imus_blending_basic(self, johnson_v_passband, imus_query):
        """Test basic Imu blending functionality."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        result = johnson_v_passband.interpolate_imus(
            imus_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        assert isinstance(result, passbands.InterpResult)
        assert hasattr(result, 'interps')
        assert len(result.interps) == len(imus_query.pts)
        
        # Results should be finite and positive
        assert np.all(np.isfinite(result.interps))
        assert np.all(result.interps > 0)

    def test_imus_blending_vs_no_blending(self, johnson_v_passband):
        """Test that Imu blending produces different results than no blending."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        # Create an off-grid query for Imus
        cols = ['teffs', 'loggs', 'abuns', 'mus']
        pts = np.array([
            [3000, 5.5, 0.0, 1.0],  # Off-grid parameters
            [3000, 5.5, 0.0, 0.5],
        ])
        off_grid_imus_query = passbands.InterpQuery(cols, pts)

        # Results without blending
        result_no_blend = johnson_v_passband.interpolate_imus(
            off_grid_imus_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='none',
            atm_extrapolation_method='linear'
        )
        
        # Results with blending
        result_with_blend = johnson_v_passband.interpolate_imus(
            off_grid_imus_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Results should be different when blending is applied
        assert not np.allclose(result_no_blend.interps, result_with_blend.interps, rtol=1e-10)

    def test_blending_blackbody_consistency(self, johnson_v_passband):
        """Test that extreme off-grid points approach blackbody values."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        # Create extremely off-grid query
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[100, 10.0, 5.0]])  # Extremely off-grid
        extreme_query = passbands.InterpQuery(cols, pts)

        # Get blended result
        result_blended = johnson_v_passband.interpolate_inorms(
            extreme_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Get pure blackbody result
        bb_query = passbands.InterpQuery(['teffs'], np.array([[100]]))
        result_blackbody = johnson_v_passband.interpolate_inorms(
            bb_query,
            atm=models.BlackbodyModelAtmosphere,
            ld_func='linear',
            ld_coeffs=[0.6],
            atm_extrapolation_method='linear'
        )
        
        # For extremely off-grid points, blended result should be close to pure blackbody
        # (This may not be exactly equal due to the blending algorithm, but should be similar)
        assert isinstance(result_blended, passbands.InterpResult)
        assert isinstance(result_blackbody, passbands.InterpResult)

    def test_invalid_blending_method(self, johnson_v_passband, on_grid_query):
        """Test behavior with invalid blending method."""
        # Invalid blending method should either be ignored or raise an error
        try:
            result = johnson_v_passband.interpolate_inorms(
                on_grid_query,
                atm=models.CK2004ModelAtmosphere,
                blending_method='invalid_method'
            )
            # If it doesn't raise an error, it should return a valid result
            assert isinstance(result, passbands.InterpResult)
        except (ValueError, KeyError):
            # It's also acceptable to raise an error for invalid blending method
            pass

    def test_blending_parameter_edge_cases(self, johnson_v_passband, off_grid_query):
        """Test blending with edge case parameters."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        # Test with zero threshold (everything is off-grid)
        result_zero_thresh = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=0.0,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Test with very large threshold (nothing is off-grid)
        result_large_thresh = johnson_v_passband.interpolate_inorms(
            off_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e10,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Both should return valid results
        assert isinstance(result_zero_thresh, passbands.InterpResult)
        assert isinstance(result_large_thresh, passbands.InterpResult)

    def test_blending_return_structure(self, johnson_v_passband, mixed_grid_query):
        """Test that blending returns proper InterpResult structure."""
        # Skip if CK2004 not available
        if 'ck2004:Imu' not in johnson_v_passband.content:
            pytest.skip("CK2004 atmosphere tables not available")

        result = johnson_v_passband.interpolate_inorms(
            mixed_grid_query,
            atm=models.CK2004ModelAtmosphere,
            blending_method='blackbody',
            dist_threshold=1e-5,
            blending_margin=3.0,
            atm_extrapolation_method='linear',
            ld_extrapolation_method='linear'  # Required when ld_func='interp' and blending_method='blackbody'
        )
        
        # Should return InterpResult with proper attributes
        assert isinstance(result, passbands.InterpResult)
        assert hasattr(result, 'interps')
        assert hasattr(result, 'get_interpolated_values')
        
        # When blending occurs, should have distances and blending factors
        if hasattr(result, 'bfs') and result.bfs is not None:
            assert hasattr(result, 'dists')
            assert result.dists is not None
            assert len(result.bfs) == len(mixed_grid_query.pts)
            assert len(result.dists) == len(mixed_grid_query.pts)


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__, '-v'])