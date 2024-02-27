import unittest

import numpy as np

from bayrob.utils.utils import pnt2line
from ddt import ddt, data, unpack


@ddt
class UtilsTests(unittest.TestCase):

    # @classmethod
    # def setUpClass(cls) -> None:
    #
    #     pass

    @data(
        ([5., 5.], [6., 3.], [6., 7.], (1., [6., 5.])),  # line inside orthogonal vector to pnt
        ([5., 5.], [4., 1.], [4., 3.], (2.24, [4., 3.])),  # line outside orthogonal vector to pnt
        ([1., 0.], [1., 0.], [3., 4.], (0., [1., 0.])),  # start equals pnt ERROR
        ([1., 0.], [1., 0.], [-1., 3.], (0., [1., 0.])),  # end equals pnt
        ([1., 0.], [1., 1.], [1., -1.], (0., [1., 0.])),  # line goes through pnt
    )
    @unpack
    def test_pnt2line_2d(
            self,
            pnt,
            start,
            end,
            expres
    ) -> None:
        d1, pt1 = pnt2line(pnt, start, end)
        d2, pt2 = pnt2line(pnt, end, start)  # ERROR FOR RUN 3

        self.assertEqual((d1, pt1), (d2, pt2), msg=f'pnt2line not symmetric')
        self.assertAlmostEqual(expres[0], d1, 2, msg='pnt2line unexpected result')
        # Unit test cannot compare collections, so we use np.testing here
        np.testing.assert_almost_equal(expres[1], pt1, 2, err_msg='pnt2line unexpected result')

    @data(
        ([5., 5., 0.], [6., 3., 0.], [6., 7., 0.], (1., [6., 5., 0.])),
    )
    @unpack
    def test_pnt2line_3d(
            self,
            pnt,
            start,
            end,
            expres
    ) -> None:
        d1, pt1 = pnt2line(pnt, start, end)
        d2, pt2 = pnt2line(pnt, end, start)

        self.assertEqual((d1, pt1), (d2, pt2), msg=f'pnt2line not symmetric')
        self.assertAlmostEqual(expres[0], d1, 2, msg='pnt2line unexpected result')
        # Unit test cannot compare collections, so we use np.testing here
        np.testing.assert_almost_equal(expres[1], pt1, 2, err_msg='pnt2line unexpected result')

    @data(
        ([1., 2.], [-4., -1.], [7., -4.], [-1., 5.], [9., 3.], (2.55, [1.5, 4.5])),
    )
    @unpack
    def test_pnt2line_closest_pt_in_rectangle(
            self,
            pnt,
            ll,
            lr,
            ul,
            ur,
            expres
    ) -> None:
        d, pt = min([d for d in [
            pnt2line(pnt, ll, lr),
            pnt2line(pnt, ll, ul),
            pnt2line(pnt, lr, ur),
            pnt2line(pnt, ul, ur)
        ]], key=lambda x: x[0])

        self.assertAlmostEqual(expres[0], d, 2, msg='pnt2line unexpected result')
        # Unit test cannot compare collections, so we use np.testing here
        np.testing.assert_almost_equal(expres[1], pt, 2, err_msg='pnt2line unexpected result')
