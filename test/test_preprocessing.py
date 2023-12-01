# ...
# Copyright 2021 Brooks M. Musangu and Jan Drugowitsch.
# license Modified BSD, see LICENSE.txt for details.
# ...
"""
Preprocessing Unittests
"""

import unittest
import numpy as np
from gpfa.preprocessing import EventTimesToCounts
try:
    import neo
    neo_imported = True
except ImportError:
    print('neo failed to import. Skipping neo-specific tests.')
    neo_imported = False


class TestProprocessing(unittest.TestCase):
    """
    Unit tests for preprocessing EventTimesToCounts class
    Any method that starts with ``test_`` will be considered
    as a test case.
    """
    def setUp(self):
        """
        Set up sample data and parameters
        """
        # ========================
        # Parameters
        # ========================
        self.bin_size = 0.1  # [s]
        self.t_stop1 = 0.4  # [s]
        self.t_stop2 = 0.48  # [s]

        # =====================================
        # Sample data where `t_stop1 = 0.4 [s]`
        # =====================================
        # For X1 - X4_at_Tstop1, the input data is the same but
        # the data type is different (i.e., list-of-list,
        # list-of-arrays, numpy.ndarray and neo.SpikeTrain,
        # respectively)
        self.X1_at_tstop1 = [[0, 0.1, 0.15, 0.4], [0.05, 0.3]]
        self.X2_at_tstop1 = [np.array(self.X1_at_tstop1[0]),
                             np.array(self.X1_at_tstop1[1])]
        self.X3_at_tstop1 = np.array(self.X2_at_tstop1, object)
        if neo_imported:
            self.X4_at_tstop1 = [neo.SpikeTrain(self.X1_at_tstop1[0],
                                 units='sec', t_stop=self.t_stop1),
                                 neo.SpikeTrain(self.X1_at_tstop1[1],
                                 units='sec', t_stop=self.t_stop1)]

        # =======================================
        #  Sample data where `t_stop2 = 0.48 [s]`
        # =======================================
        # For X1 - X4_at_Tstop2, the input data is the same but
        # the data type is different (i.e., list-of-list,
        # list-of-arrays, numpy.ndarray and neo.SpikeTrain,
        # respectively)
        self.X1_at_tstop2 = [[0, 0.1, 0.15, 0.4, 0.45], [0.05, 0.3]]
        self.X2_at_tstop2 = [np.array(self.X1_at_tstop2[0]),
                             np.array(self.X1_at_tstop2[1])]
        self.X3_at_tstop2 = np.array(self.X2_at_tstop2, object)

        if neo_imported:
            self.X4_at_tstop2 = [neo.SpikeTrain(self.X1_at_tstop2[0],
                                 units='sec', t_stop=self.t_stop2),
                                 neo.SpikeTrain(self.X1_at_tstop2[1],
                                 units='sec', t_stop=self.t_stop2)]

        # ===============================
        # # initiate `EventTimesToCounts`
        # ===============================
        # initiate `EventTimesToCounts` with `extrapolate_last_bin=False`
        # for `t_stop=0.4 [s]`
        self.T_at_tstop1 = EventTimesToCounts(bin_size=self.bin_size,
                                              t_stop=self.t_stop1,
                                              extrapolate_last_bin=False)
        # initiate `EventTimesToCounts` with `extrapolate_last_bin=True`
        # for `t_stop=0.4 [s]`
        self.T_with_extrapolatelastbin_at_tstop1 = EventTimesToCounts(
                                                    bin_size=self.bin_size,
                                                    t_stop=self.t_stop1,
                                                    extrapolate_last_bin=True
                                                    )

        # initiate `EventTimesToCounts` with `extrapolate_last_bin=False`
        # for `t_stop=0.48 [s]`
        self.T_at_tstop2 = EventTimesToCounts(bin_size=self.bin_size,
                                              t_stop=self.t_stop2,
                                              extrapolate_last_bin=False)

        # initiate `EventTimesToCounts` with `extrapolate_last_bin=True`
        # for `t_stop=0.48 [s]`
        self.T_with_extrapolatelastbin_at_tstop2 = EventTimesToCounts(
                                                    bin_size=self.bin_size,
                                                    t_stop=self.t_stop2,
                                                    extrapolate_last_bin=True)

        # initiate `EventTimesToCounts` with `extrapolate_last_bin=True`
        # for `t_stop=None`
        self.T_with_extrapolatelastbin_tstopNone = EventTimesToCounts(
                                                    bin_size=self.bin_size,
                                                    extrapolate_last_bin=True)

        # ======================
        # # The expected results
        # ======================
        # The expected results when `t_stop=0.4 [s]` whether extrapolating
        # the last bin or not.
        # Furthermore, we expect the results to be the same for `t_stop=0.48
        # [s]` when `extrapolate_last_bin = False`
        self.results_at_tstop1 = np.array(
                                          [[1, 2, 0, 1],
                                           [1, 0, 1, 0]]
                                         )

        # The expected results when `t_stop=0.48 [s]` only when
        # extrapolating the last bin or not.
        self.results_at_tstop2 = np.array(
                                          [[1., 2., 0., 0., 2.5],
                                           [1., 0., 1., 0., 0.]]
                                         )

    # ==========
    # Test cases
    # ==========
    def test_transform_at_tstop1(self):
        """
        Test `EventTimesToCounts.transform` for `t_stop = 0.4`,
        `bin_size = 0.1` and `extrapolate_last_bin=False`.
        """
        trans1 = self.T_at_tstop1.transform(self.X1_at_tstop1)
        trans2 = self.T_at_tstop1.transform(self.X2_at_tstop1)
        trans3 = self.T_at_tstop1.transform(self.X3_at_tstop1)

        self.assertEqual(self.results_at_tstop1.all(), trans1.all())
        self.assertEqual(self.results_at_tstop1.all(), trans2.all())
        self.assertEqual(self.results_at_tstop1.all(), trans3.all())

    def test_transform_with_extrapolatedlastbin_tstop1(self):
        """
        Test `EventTimesToCounts.transform` for `t_stop = 0.4`,
        `bin_size = 0.1` and `extrapolate_last_bin=True`.
        """
        trans1 = self.T_with_extrapolatelastbin_at_tstop1.transform(
            self.X1_at_tstop1)
        trans2 = self.T_with_extrapolatelastbin_at_tstop1.transform(
            self.X2_at_tstop1)
        trans3 = self.T_with_extrapolatelastbin_at_tstop1.transform(
            self.X3_at_tstop1)

        self.assertEqual(self.results_at_tstop1.all(), trans1.all())
        self.assertEqual(self.results_at_tstop1.all(), trans2.all())
        self.assertEqual(self.results_at_tstop1.all(), trans3.all())

    def test_transform_at_tstop2(self):
        """
        Test `EventTimesToCounts.transform` for `t_stop = 0.48`,
        `bin_size = 0.1` and `extrapolate_last_bin=False`.
        """
        trans1 = self.T_at_tstop2.transform(self.X1_at_tstop2)
        trans2 = self.T_at_tstop2.transform(self.X2_at_tstop2)
        trans3 = self.T_at_tstop2.transform(self.X3_at_tstop2)

        self.assertEqual(self.results_at_tstop1.all(), trans1.all())
        self.assertEqual(self.results_at_tstop1.all(), trans2.all())
        self.assertEqual(self.results_at_tstop1.all(), trans3.all())

    def test_transform_x_at_tstop2_with_extrapolatedlastbin_tstop1(self):
        """
        Test `EventTimesToCounts.transform` for `t_stop = 0.4`,
        `bin_size = 0.1` and `extrapolate_last_bin=True` but on
        input data with the last spike times at 0.48 [s]. Here it
        (last spike time) should be ignored.
        """
        trans1 = self.T_with_extrapolatelastbin_at_tstop1.transform(
            self.X1_at_tstop2)
        trans2 = self.T_with_extrapolatelastbin_at_tstop1.transform(
            self.X2_at_tstop2)
        trans3 = self.T_with_extrapolatelastbin_at_tstop1.transform(
            self.X3_at_tstop2)

        self.assertEqual(self.results_at_tstop1.all(), trans1.all())
        self.assertEqual(self.results_at_tstop1.all(), trans2.all())
        self.assertEqual(self.results_at_tstop1.all(), trans3.all())

    def test_transform_with_extrapolatedlastbin_tstop2(self):
        """
        Test `EventTimesToCounts.transform` for `t_stop = 0.48`,
        `bin_size = 0.1` and `extrapolate_last_bin=True`. The
        results should be different from the last three test cases.
        """
        trans1 = self.T_with_extrapolatelastbin_at_tstop2.transform(
            self.X1_at_tstop2)
        trans2 = self.T_with_extrapolatelastbin_at_tstop2.transform(
            self.X2_at_tstop2)
        trans3 = self.T_with_extrapolatelastbin_at_tstop2.transform(
            self.X3_at_tstop2)

        self.assertTrue(np.allclose(self.results_at_tstop2, trans1))
        self.assertTrue(np.allclose(self.results_at_tstop2, trans2))
        self.assertTrue(np.allclose(self.results_at_tstop2, trans3))

    @unittest.skipIf(not neo_imported, "neo not imported")
    def test_transform_on_neospiketrains(self):
        """
        Test `EventTimesToCounts.transform` for difference
        test conditions.
        """
        # ============================
        # `extrapolate_last_bin=False`
        # ============================
        # 1. When `t_stop=0.4` and the last spike time is at 0.4 [s]
        trans1 = self.T_at_tstop1.transform(self.X4_at_tstop1)
        self.assertEqual(self.results_at_tstop1.all(), trans1.all())

        # 2. When `t_stop=0.48` and the last spike time is at 0.48 [s]
        trans2 = self.T_at_tstop2.transform(self.X4_at_tstop2)
        self.assertEqual(self.results_at_tstop1.all(), trans2.all())

        # ===========================
        # `extrapolate_last_bin=True`
        # ===========================
        # 3. When `t_stop=0.4` and the last spike time is at 0.4 [s]
        trans3 = self.T_with_extrapolatelastbin_at_tstop1.transform(
            self.X4_at_tstop1)
        self.assertEqual(self.results_at_tstop1.all(), trans3.all())

        # 4. When `t_stop=0.48` and the last spike time is at 0.48 [s]
        trans4 = self.T_with_extrapolatelastbin_at_tstop2.transform(
            self.X4_at_tstop2)
        self.assertTrue(np.allclose(self.results_at_tstop2, trans4))

        # 5. When `t_stop=0.4` but the last spike time is at 0.48 [s],
        #    which means the spike time at 0.48 [s] should be ignored.
        #    However, `EventTimesToCounts.transform` should throw an
        #    exception since the `t_stop` initialized in
        #    `T_with_extrapolatelastbin_at_tstop1` is different from
        #    `t_stop` in `X4_at_tstop2.t_stop.magnitude`. Let's test
        #    if indeeded this the case.
        self.assertRaises(ValueError,
                          self.T_with_extrapolatelastbin_at_tstop1.transform,
                          self.X4_at_tstop2)

        # 6. Now try `t_stop=None` and the last spike time is at 0.4 [s]
        trans6 = self.T_with_extrapolatelastbin_tstopNone.transform(
            self.X4_at_tstop2)
        self.assertEqual(self.results_at_tstop1.all(), trans6.all())
