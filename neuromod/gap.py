"""Gap junctions (electrical synapses).

Port of ``mechabrain/core/gap_junctions.h`` (fwmc): ``PropagateGapCurrents``
and ``BuildFromRegion``.

Separate table from chemical synapses: parallel arrays
``(a, b, conductance, rectification)``::

    I = g * (V_b - V_a);  a.i_ext += I;  b.i_ext -= I   (symmetric)

with innexin voltage-dependent rectification (Phelan 2005)::

    if rectification < 1 and dv < 0: g *= rectification

Port note: the EM connectome has no gap junctions, so placement must be
engineered (region-based density as below, or literature GF->motor pairs).
Used in fwmc for clock neurons (Inx6/Inx7), antennal lobe, and the giant
fiber escape path (~1 ms latency).
"""

import numpy as np


class GapJunctions:
    """Electrical synapse table.

    Parameters
    ----------
    a, b : array_like int
        Neuron index pairs (a[i], b[i]).
    conductance : array_like float
        Gap conductance g per junction.
    rectification : array_like float, optional
        In [0, 1]. 1.0 = symmetric (default); < 1 attenuates current when
        dv = Vb - Va < 0.
    """

    def __init__(self, a, b, conductance, rectification=None):
        self.a = np.asarray(a, dtype=np.int64)
        self.b = np.asarray(b, dtype=np.int64)
        self.conductance = np.asarray(conductance, dtype=np.float64)
        n = self.a.shape[0]
        if rectification is None:
            self.rectification = np.ones(n, dtype=np.float64)
        else:
            self.rectification = np.asarray(rectification, dtype=np.float64)
        assert self.b.shape[0] == n and self.conductance.shape[0] == n
        assert self.rectification.shape[0] == n
        assert np.all((self.rectification >= 0.0) & (self.rectification <= 1.0))

    @property
    def n_junctions(self):
        return self.a.shape[0]

    def currents(self, v, i_ext):
        """Add gap-junction currents into ``i_ext`` in place.

        Parameters
        ----------
        v : ndarray float, shape (n_neurons,)
            Membrane voltages.
        i_ext : ndarray float, shape (n_neurons,)
            External current array; modified in place.
        """
        if self.n_junctions == 0:
            return
        dv = v[self.b] - v[self.a]
        g = self.conductance.copy()
        rect_mask = (self.rectification < 1.0) & (dv < 0.0)
        g[rect_mask] *= self.rectification[rect_mask]
        I = g * dv
        np.add.at(i_ext, self.a, I)
        np.add.at(i_ext, self.b, -I)


def build_from_region(region_neurons, density, g_default, seed=42,
                      rectification=1.0):
    """Create gap junctions between every pair in a region with probability
    ``density`` (fwmc's ``BuildFromRegion``).

    Parameters
    ----------
    region_neurons : array_like int
        Neuron indices belonging to the region.
    density : float in [0, 1]
        Connection probability per pair.
    g_default : float
        Conductance assigned to each created junction.
    seed : int
        RNG seed (fwmc default 42).
    rectification : float in [0, 1]
        Rectification factor for all created junctions.
    """
    region = np.asarray(region_neurons, dtype=np.int64)
    rng = np.random.default_rng(seed)
    a_list, b_list = [], []
    m = region.shape[0]
    for ii in range(m):
        for jj in range(ii + 1, m):
            if rng.random() < density:
                a_list.append(region[ii])
                b_list.append(region[jj])
    a = np.array(a_list, dtype=np.int64)
    b = np.array(b_list, dtype=np.int64)
    g = np.full(a.shape[0], g_default, dtype=np.float64)
    r = np.full(a.shape[0], rectification, dtype=np.float64)
    return GapJunctions(a, b, g, r)


if __name__ == "__main__":
    # symmetric: current flows from high V to low V, equal and opposite
    gj = GapJunctions([0], [1], [0.5])
    v = np.array([10.0, 0.0])
    i_ext = np.zeros(2)
    gj.currents(v, i_ext)
    assert abs(i_ext[0] - 0.5 * (0.0 - 10.0)) < 1e-12, i_ext
    assert abs(i_ext[1] + i_ext[0]) < 1e-12  # bidirectional, symmetric

    # rectification: dv < 0 attenuates by the rectification factor
    gj2 = GapJunctions([0], [1], [1.0], [0.25])
    i2 = np.zeros(2)
    gj2.currents(v, i2)  # dv = -10 < 0 -> g *= 0.25
    assert abs(i2[0] - 0.25 * 1.0 * (0.0 - 10.0)) < 1e-12, i2
    # dv > 0 passes unattenuated
    v3 = np.array([0.0, 10.0])
    i3 = np.zeros(2)
    gj2.currents(v3, i3)
    assert abs(i3[0] - 1.0 * (10.0 - 0.0)) < 1e-12, i3

    # region builder: density 1.0 connects every pair, seed reproducibility
    region = np.array([3, 7, 9])
    g1 = build_from_region(region, 1.0, 0.1, seed=42)
    assert g1.n_junctions == 3, g1.n_junctions  # C(3,2)
    g2 = build_from_region(region, 1.0, 0.1, seed=42)
    assert np.array_equal(g1.a, g2.a) and np.array_equal(g1.b, g2.b)
    g3 = build_from_region(region, 0.0, 0.1, seed=42)
    assert g3.n_junctions == 0
    print("gap.py smoke test OK")
