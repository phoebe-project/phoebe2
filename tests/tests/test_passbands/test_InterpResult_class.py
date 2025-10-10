import pytest
import numpy as np

from phoebe.atmospheres.passbands import InterpResult


class TestInterpResult:
    """Test suite for the InterpResult class."""

    @pytest.fixture
    def sample_interps(self):
        """Sample interpolated values for testing."""
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    @pytest.fixture
    def sample_dists(self):
        """Sample distances for testing."""
        return np.array([0.1, 0.2, 0.3])

    @pytest.fixture
    def sample_bfs(self):
        """Sample blending factors for testing."""
        return np.array([0.5, 0.7, 0.9])

    @pytest.fixture
    def basic_result(self, sample_interps):
        """Basic InterpResult with only interpolated values."""
        return InterpResult(interps=sample_interps)

    @pytest.fixture
    def full_result(self, sample_interps, sample_dists, sample_bfs):
        """Full InterpResult with all optional attributes."""
        return InterpResult(interps=sample_interps, dists=sample_dists, bfs=sample_bfs, meta={'test': 'value'})

    def test_init_basic(self, sample_interps):
        """Test basic initialization with only interpolated values."""
        result = InterpResult(interps=sample_interps)

        assert isinstance(result, InterpResult)
        np.testing.assert_array_equal(result.interps, sample_interps)
        assert not hasattr(result, 'dists') or result.dists is None
        assert not hasattr(result, 'bfs') or result.bfs is None

    def test_init_with_kwargs(self, sample_interps, sample_dists, sample_bfs):
        """Test initialization with additional keyword arguments."""
        result = InterpResult(interps=sample_interps, dists=sample_dists, bfs=sample_bfs, extra_attr='test')

        np.testing.assert_array_equal(result.interps, sample_interps)
        np.testing.assert_array_equal(result.dists, sample_dists)
        np.testing.assert_array_equal(result.bfs, sample_bfs)
        assert result.extra_attr == 'test'

    def test_from_ndpolator_basic(self, sample_interps):
        """Test creation from ndpolator dictionary with basic content."""
        ndp_dict = {'interps': sample_interps}
        result = InterpResult.from_ndpolator(ndp_dict)

        assert isinstance(result, InterpResult)
        np.testing.assert_array_equal(result.interps, sample_interps)

    def test_from_ndpolator_full(self, sample_interps, sample_dists, sample_bfs):
        """Test creation from ndpolator dictionary with full content."""
        ndp_dict = {
            'interps': sample_interps,
            'dists': sample_dists,
            'bfs': sample_bfs,
            'extra_data': 'test'
        }
        result = InterpResult.from_ndpolator(ndp_dict)

        np.testing.assert_array_equal(result.interps, sample_interps)
        np.testing.assert_array_equal(result.dists, sample_dists)
        np.testing.assert_array_equal(result.bfs, sample_bfs)
        assert result.extra_data == 'test'

    def test_from_ndpolator_missing_interps(self):
        """Test error when 'interps' key is missing."""
        ndp_dict = {'dists': np.array([1, 2, 3])}

        with pytest.raises(ValueError, match="ndp_output must contain 'interps' key"):
            InterpResult.from_ndpolator(ndp_dict)

    def test_from_ndpolator_invalid_input(self):
        """Test error when input is not a dictionary."""
        with pytest.raises(TypeError, match="ndp_output must be a dictionary"):
            InterpResult.from_ndpolator([1, 2, 3])

    def test_shape_property(self, basic_result):
        """Test the shape property."""
        assert basic_result.shape == (3, 2)

    def test_size_property(self, basic_result):
        """Test the size property."""
        assert basic_result.size == 6

    def test_len(self, basic_result):
        """Test the __len__ method."""
        assert len(basic_result) == 3

    def test_get_interpolated_values(self, full_result, sample_interps):
        """Test the get_interpolated_values method."""
        values = full_result.get_interpolated_values()
        np.testing.assert_array_equal(values, sample_interps)

    def test_get_interpolated_values_none(self):
        """Test get_interpolated_values when interps attribute doesn't exist."""
        # Create a result without interps (shouldn't normally happen, but test edge case)
        result = InterpResult.__new__(InterpResult)
        assert result.get_interpolated_values() is None

    def test_get_distances(self, full_result, sample_dists):
        """Test the get_distances method."""
        dists = full_result.get_distances()
        np.testing.assert_array_equal(dists, sample_dists)

    def test_get_distances_none(self, basic_result):
        """Test get_distances when dists attribute doesn't exist."""
        assert basic_result.get_distances() is None

    def test_getitem_single_row(self, full_result, sample_interps, sample_dists, sample_bfs):
        """Test slicing a single row."""
        sliced = full_result[1]

        assert isinstance(sliced, InterpResult)
        np.testing.assert_array_equal(sliced.interps, sample_interps[1:2])  # Should be 2D
        np.testing.assert_array_equal(sliced.dists, sample_dists[1])
        np.testing.assert_array_equal(sliced.bfs, sample_bfs[1])

    def test_getitem_slice(self, full_result, sample_interps, sample_dists, sample_bfs):
        """Test slicing multiple rows."""
        sliced = full_result[0:2]

        assert isinstance(sliced, InterpResult)
        np.testing.assert_array_equal(sliced.interps, sample_interps[0:2])
        np.testing.assert_array_equal(sliced.dists, sample_dists[0:2])
        np.testing.assert_array_equal(sliced.bfs, sample_bfs[0:2])

    def test_getitem_2d_slice(self, full_result, sample_interps):
        """Test 2D slicing (like for LD coefficients filtering)."""
        # This tests the special case for LD coefficient filtering
        sliced = full_result[:, 1:]

        assert isinstance(sliced, InterpResult)
        np.testing.assert_array_equal(sliced.interps, sample_interps[:, 1:])
        # Other attributes should be preserved unchanged for 2D slicing
        np.testing.assert_array_equal(sliced.dists, full_result.dists)
        np.testing.assert_array_equal(sliced.bfs, full_result.bfs)

    def test_getitem_boolean_mask(self, full_result, sample_interps, sample_dists, sample_bfs):
        """Test boolean indexing."""
        mask = np.array([True, False, True])
        sliced = full_result[mask]

        assert isinstance(sliced, InterpResult)
        np.testing.assert_array_equal(sliced.interps, sample_interps[mask])
        np.testing.assert_array_equal(sliced.dists, sample_dists[mask])
        np.testing.assert_array_equal(sliced.bfs, sample_bfs[mask])

    def test_getitem_with_none_attributes(self, basic_result):
        """Test slicing when some attributes are None."""
        sliced = basic_result[0:2]

        assert isinstance(sliced, InterpResult)
        assert sliced.interps.shape == (2, 2)
        # None attributes should remain None or not exist
        assert not hasattr(sliced, 'dists') or sliced.dists is None
        assert not hasattr(sliced, 'bfs') or sliced.bfs is None

    def test_getitem_slicing_failure_fallback(self, full_result):
        """Test fallback when slicing fails on non-array attributes."""
        # Add a non-sliceable attribute
        full_result.scalar_value = 42

        # This should still work, preserving the scalar value
        sliced = full_result[0:2]
        assert sliced.scalar_value == 42

    def test_repr(self, basic_result):
        """Test the string representation."""
        repr_str = repr(basic_result)
        assert '<InterpResult:' in repr_str
        assert '(3, 2)>' in repr_str

    def test_empty_result(self):
        """Test with empty arrays."""
        empty_interps = np.array([]).reshape(0, 2)
        result = InterpResult(interps=empty_interps)

        assert result.shape == (0, 2)
        assert result.size == 0
        assert len(result) == 0

    def test_1d_interps(self):
        """Test with 1D interpolated values."""
        interps_1d = np.array([1.0, 2.0, 3.0])
        result = InterpResult(interps=interps_1d)

        assert result.shape == (3,)
        assert result.size == 3
        assert len(result) == 3

    def test_multiple_slicing_operations(self, full_result):
        """Test multiple consecutive slicing operations."""
        # First slice
        sliced1 = full_result[1:]
        assert len(sliced1) == 2

        # Second slice on the result
        sliced2 = sliced1[0:1]
        assert len(sliced2) == 1
        assert isinstance(sliced2, InterpResult)

    def test_attribute_preservation(self, sample_interps):
        """Test that all attributes are properly preserved during operations."""
        result = InterpResult(
            interps=sample_interps,
            custom_attr='preserved',
            numerical_attr=3.14,
            array_attr=np.array([1, 2, 3])
        )

        sliced = result[0:2]
        assert sliced.custom_attr == 'preserved'
        assert sliced.numerical_attr == 3.14
        np.testing.assert_array_equal(sliced.array_attr, np.array([1, 2]))

    def test_complex_slicing_scenario(self, sample_interps):
        """Test complex slicing scenario similar to LD coefficient filtering."""
        # Create result with 11 LD coefficients (similar to real use case)
        ld_coeffs = np.random.rand(5, 11)  # 5 query points, 11 LD coefficients
        result = InterpResult(interps=ld_coeffs, dists=np.random.rand(5))

        # Filter to 'power' LD function coefficients (columns 7:11)
        power_coeffs = result[:, 7:11]

        assert power_coeffs.interps.shape == (5, 4)
        assert len(power_coeffs.dists) == 5  # Distances should be preserved

        # Then slice to first 3 query points
        subset = power_coeffs[0:3]
        assert subset.interps.shape == (3, 4)
        assert len(subset.dists) == 3

    def test_ndpolator_dict_modification(self, sample_interps):
        """Test that from_ndpolator doesn't modify the original dictionary."""
        ndp_dict = {
            'interps': sample_interps.copy(),
            'dists': np.array([1, 2, 3]),
            'extra': 'value'
        }
        original_keys = set(ndp_dict.keys())

        InterpResult.from_ndpolator(ndp_dict)

        # Original dict should be unchanged
        assert set(ndp_dict.keys()) == original_keys
        assert 'interps' in ndp_dict  # Should still be there


if __name__ == '__main__':
    pytest.main([__file__])
