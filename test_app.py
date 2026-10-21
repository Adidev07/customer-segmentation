import unittest
from app import *
class DomainTests(unittest.TestCase):
    def test_two_groups(self):
        x=np.array([[-5.,0],[-4.,0],[4.,0],[5.,0]]);p=assign(x,fit(x,2));self.assertEqual(p[0],p[1]);self.assertNotEqual(p[0],p[3])
    def test_constant_input(self):self.assertTrue(np.isfinite(fit(np.ones((10,2)),3)).all())
    def test_permutation_invariant_ari(self):self.assertAlmostEqual(ari(np.array([0,0,1,1]),np.array([1,1,0,0])),1.)
