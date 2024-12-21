# Authored by ultragreed
# Credit for algorithm theory to P.Mesenev - A.Kulikov - R.Williams (creator)
import numpy as np


def fill_h_part(h_part: np.ndarray, g11: np.ndarray, g12: np.ndarray) -> None:
    """
    Fill part of h matrix with sums of possible selections of two different sets of g matrix split.

    :param h_part: Part of h matrix which will be filled
    :param g11: Part of g matrix with inner edges of first set
    :param g12: Part of g matrix with edges connecting nodes of first set and nodes of second set
    """
    n = g11.shape[0]
    k = 2**n
    # This loop can easily be reimplemented in pure numpy, without python loops, but current
    #   implementation is much more readable.
    # This function is O(2^(2n/3)), so unless matrix multiplication exponent goes as low as 2
    #   one day, this time waste is insignificant in terms of full Williams algorithm.
    for x1 in range(k):
        selection1 = np.asarray([i == '1' for i in f'{x1:0>{n}b}'])
        x1_v1 = g11[np.ix_(selection1, ~selection1)].sum()

        for x2 in range(k):
            selection2 = np.asarray([i == '1' for i in f'{x2:0>{n}b}'])

            x1_v2 = g12[np.ix_(selection1, ~selection2)].sum()
            x2_v1 = g12[np.ix_(~selection1, selection2)].sum()

            h_part[x1, x2] = x1_v1 + x1_v2 + x2_v1


def maxcut_williams(graph: np.ndarray):
    """Calculate set of nodes with maxcut solution with first node for provided adjacency matrix."""
    # If graph number of nodes % 3 != 0, we have to add new isolated nodes
    n = graph.shape[0] + (3 - graph.shape[0]) % 3
    pad_n = n - graph.shape[0]
    g = np.pad(graph, ((0, pad_n), (0, pad_n)), constant_values=0)

    k = 2 ** (n // 3)

    h = np.zeros((3 * k, 3 * k), dtype=np.uint64)

    h12 = h[:k, k : 2 * k]
    h23 = h[k : 2 * k, 2 * k :]
    h31 = h[2 * k :, :k]

    # From set1 to set2
    g11 = g[: n // 3, : n // 3]
    g12 = g[: n // 3, n // 3 : 2 * n // 3]
    fill_h_part(h12, g11, g12)

    # From set2 to set3
    g22 = g[n // 3 : 2 * n // 3, n // 3 : 2 * n // 3]
    g23 = g[n // 3 : 2 * n // 3, 2 * n // 3 :]
    fill_h_part(h23, g22, g23)

    # From set3 to set1
    g33 = g[2 * n // 3 :, 2 * n // 3 :]
    g31 = g[2 * n // 3 :, : n // 3]
    fill_h_part(h31, g33, g31)

    edges = g.sum() // 2

    maxcut_sum = 0
    maxcut = [0]

    w12_variants = np.arange(edges + 1)[np.isin(np.arange(edges + 1), h12)]
    w23_variants = np.arange(edges + 1)[np.isin(np.arange(edges + 1), h23)]
    w31_variants = np.arange(edges + 1)[np.isin(np.arange(edges + 1), h31)]
    for w12 in w12_variants:
        h_filtered = np.zeros_like(h)
        h_filtered[:k, k : 2 * k] = np.where(h12 == w12, 1, 0)

        for w23 in w23_variants:
            h_filtered[k : 2 * k, 2 * k :] = np.where(h23 == w23, 1, 0)

            for w31 in w31_variants:
                if w12 + w23 + w31 <= maxcut_sum:
                    continue

                h_filtered[2 * k :, :k] = np.where(h31 == w31, 1, 0)

                tri_ind = np.argwhere(np.linalg.matrix_power(h_filtered, 3).diagonal() >= 1)
                x1_triag = tri_ind[tri_ind < k]
                x2_triag = tri_ind[(tri_ind >= k) & (tri_ind < 2 * k)] - k
                x3_triag = tri_ind[tri_ind >= 2 * k] - 2 * k

                if x1_triag.size == 0 or x2_triag.size == 0 or x3_triag.size == 0:
                    continue

                # Find all vertices from set1 and set2 which are connected with w12 weight edge
                x12_variants = np.argwhere(h12 == w12)
                # and are part of some triangle
                x12_variants = x12_variants[
                    np.isin(x12_variants[:, 0], x1_triag) & np.isin(x12_variants[:, 1], x2_triag)
                ]

                if x12_variants.size == 0:
                    continue

                # For selected set1 and set2 vertices find all set3 vertices which are connected
                #   with w31 and w23 weight edges
                x3_variants = np.argwhere(
                    (h31[:, x12_variants[:, 0]] == w31) & (h23[x12_variants[:, 1], :] == w23).T
                )
                # and are part of some triangle
                x3_variants = x3_variants[np.isin(x3_variants[:, 0], x3_triag)]

                if x3_variants.size == 0:
                    continue

                # All the remaining x3 variants will do, but not all x1 and x2 variants
                x3, x12_ind = x3_variants[0]
                x1, x2 = x12_variants[x12_ind]

                maxcut_sum = w12 + w23 + w31

                maxcut = np.arange(n)[
                    [i == '1' for i in f'{x1:0>{n//3}b}']
                    + [i == '1' for i in f'{x2:0>{n//3}b}']
                    + [i == '1' for i in f'{x3:0>{n//3}b}']
                ]

                # Always return set with first node
                if 0 not in maxcut:
                    maxcut = np.arange(n)[~np.isin(np.arange(n), maxcut)]

                # Do not return nodes which were not present in original graph
                maxcut = maxcut[maxcut < graph.shape[0]]
    return maxcut


