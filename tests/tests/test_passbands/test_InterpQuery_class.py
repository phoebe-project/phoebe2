"""
Unit tests for InterpQuery class from phoebe.atmospheres.passbands
"""

import pytest
import numpy as np
from phoebe.atmospheres.passbands import InterpQuery


class TestInterpQuery:
    """Test suite for InterpQuery class"""

    def test_init_valid_input(self):
        """Test initialization with valid input"""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5000, 4.0, 0.0], [6000, 4.5, -0.5]])
        query = InterpQuery(cols, pts)

        assert query.cols == cols
        np.testing.assert_array_equal(query.pts, pts)
        assert query.meta == {}
        assert len(query) == 2

    def test_init_with_meta(self):
        """Test initialization with metadata"""
        cols = ['teffs', 'loggs']
        pts = np.array([[5000, 4.0]])
        meta = {'source': 'test', 'version': 1.0}
        query = InterpQuery(cols, pts, meta=meta)

        assert query.meta == meta

    def test_init_shape_mismatch(self):
        """Test that shape mismatch raises ValueError"""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5000, 4.0]])  # Missing one column

        with pytest.raises(ValueError, match="Shape mismatch"):
            InterpQuery(cols, pts)

    def test_init_1d_array(self):
        """Test that 1D array is rejected"""
        cols = ['teffs']
        pts = np.array([5000, 6000])  # 1D array

        with pytest.raises(ValueError, match="Shape mismatch"):
            InterpQuery(cols, pts)

    def test_index(self):
        """Test column index retrieval"""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5000, 4.0, 0.0]])
        query = InterpQuery(cols, pts)

        assert query.index('teffs') == 0
        assert query.index('loggs') == 1
        assert query.index('abuns') == 2

    def test_index_not_found(self):
        """Test that missing column raises KeyError"""
        cols = ['teffs', 'loggs']
        pts = np.array([[5000, 4.0]])
        query = InterpQuery(cols, pts)

        with pytest.raises(KeyError, match="Column 'abuns' not found"):
            query.index('abuns')

    def test_get_column(self):
        """Test getting column data by name"""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5000, 4.0, 0.0], [6000, 4.5, -0.5]])
        query = InterpQuery(cols, pts)

        np.testing.assert_array_equal(query.get_column('teffs'), [5000, 6000])
        np.testing.assert_array_equal(query.get_column('loggs'), [4.0, 4.5])
        np.testing.assert_array_equal(query.get_column('abuns'), [0.0, -0.5])

    def test_get_column_not_found(self):
        """Test that getting missing column raises KeyError"""
        cols = ['teffs', 'loggs']
        pts = np.array([[5000, 4.0]])
        query = InterpQuery(cols, pts)

        with pytest.raises(KeyError, match="Column 'abuns' not found"):
            query.get_column('abuns')

    def test_subset(self):
        """Test creating subset with selected columns"""
        cols = ['teffs', 'loggs', 'abuns', 'mus']
        pts = np.array([[5000, 4.0, 0.0, 1.0], [6000, 4.5, -0.5, 0.8]])
        meta = {'test': True}
        query = InterpQuery(cols, pts, meta=meta)

        subset = query.subset(['teffs', 'abuns'])

        assert subset.cols == ['teffs', 'abuns']
        np.testing.assert_array_equal(subset.pts, [[5000, 0.0], [6000, -0.5]])
        assert subset.meta == meta  # Meta should be preserved

    def test_subset_missing_column(self):
        """Test that subset with missing column raises KeyError"""
        cols = ['teffs', 'loggs']
        pts = np.array([[5000, 4.0]])
        query = InterpQuery(cols, pts)

        with pytest.raises(KeyError, match="Column 'abuns' not found"):
            query.subset(['teffs', 'abuns'])

    def test_getitem_single_row(self):
        """Test row slicing with single index"""
        cols = ['teffs', 'loggs']
        pts = np.array([[5000, 4.0], [6000, 4.5], [7000, 5.0]])
        query = InterpQuery(cols, pts)

        subset = query[1]

        assert subset.cols == cols
        np.testing.assert_array_equal(subset.pts, [[6000, 4.5]])
        assert len(subset) == 1

    def test_getitem_slice(self):
        """Test row slicing with slice"""
        cols = ['teffs', 'loggs']
        pts = np.array([[5000, 4.0], [6000, 4.5], [7000, 5.0]])
        query = InterpQuery(cols, pts)

        subset = query[0:2]

        assert subset.cols == cols
        np.testing.assert_array_equal(subset.pts, [[5000, 4.0], [6000, 4.5]])
        assert len(subset) == 2

    def test_getitem_boolean_mask(self):
        """Test row slicing with boolean mask"""
        cols = ['teffs', 'loggs']
        pts = np.array([[5000, 4.0], [6000, 4.5], [7000, 5.0]])
        query = InterpQuery(cols, pts)

        mask = np.array([True, False, True])
        subset = query[mask]

        assert subset.cols == cols
        np.testing.assert_array_equal(subset.pts, [[5000, 4.0], [7000, 5.0]])
        assert len(subset) == 2

    def test_len(self):
        """Test length method"""
        cols = ['teffs']
        pts = np.array([[5000], [6000], [7000]])
        query = InterpQuery(cols, pts)

        assert len(query) == 3

    def test_contiguous_array_conversion(self):
        """Test that non-contiguous arrays are converted to contiguous"""
        cols = ['teffs', 'loggs']
        pts_orig = np.array([[5000, 4.0], [6000, 4.5], [7000, 5.0]])

        # Create non-contiguous array by slicing
        pts_non_contiguous = pts_orig[:, ::-1][:, ::-1]  # Create non-contiguous array

        # Verify it's non-contiguous (skip if numpy creates contiguous anyway)
        if pts_non_contiguous.flags.c_contiguous:
            # Try a different approach to create non-contiguous
            big_array = np.ones((6, 4))  # Make sure it's big enough
            pts_non_contiguous = big_array[::2, :2]  # This should be non-contiguous
            # Copy the original data into the non-contiguous array
            for i in range(3):
                pts_non_contiguous[i] = pts_orig[i]

        # Only run test if we successfully created non-contiguous array
        if not pts_non_contiguous.flags.c_contiguous:
            query = InterpQuery(cols, pts_non_contiguous)

            # Should be converted to contiguous
            assert query.pts.flags.c_contiguous
            np.testing.assert_array_equal(query.pts, pts_orig)
        else:
            # Skip test if we can't create non-contiguous array
            pytest.skip("Cannot create non-contiguous array in this numpy version")

    def test_empty_query(self):
        """Test handling of empty query points"""
        cols = ['teffs', 'loggs']
        pts = np.empty((0, 2))  # Empty array with correct shape
        query = InterpQuery(cols, pts)

        assert len(query) == 0
        assert query.pts.shape == (0, 2)

    def test_single_point_query(self):
        """Test single point query"""
        cols = ['teffs']
        pts = np.array([[5000]])
        query = InterpQuery(cols, pts)

        assert len(query) == 1
        assert query.get_column('teffs')[0] == 5000

    def test_meta_persistence_through_operations(self):
        """Test that metadata persists through subset and slicing operations"""
        cols = ['teffs', 'loggs', 'abuns']
        pts = np.array([[5000, 4.0, 0.0], [6000, 4.5, -0.5]])
        meta = {'test_id': 123, 'created_by': 'test'}
        query = InterpQuery(cols, pts, meta=meta)

        # Test subset preserves meta
        subset = query.subset(['teffs', 'loggs'])
        assert subset.meta == meta

        # Test slicing preserves meta
        sliced = query[0:1]
        assert sliced.meta == meta


if __name__ == '__main__':
    pytest.main([__file__])
