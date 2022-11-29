from typing import List
import numpy as np
import hypertiling.arraytransformation as array_trans
from hypertiling.check_numba import NumbaChecker
from hypertiling.generative.generative_reflection_util import PI2, any_close_matrix, f_dist_disc
import networkx as nx

"""
p: Number of edges/vertices of a polygon
q: Number of polygons that meet at a vertex
n: Number of layers (classical definition)
m: Number of polygons
"""

colors = ["#FF000080", "#00FF0080", "#0000FF80"]


def plot_graph(adjacent_matrix: List[List[int]], center_coords: np.array, p: int):
    """
    Plot a network of the connections
    :param adjacent_matrix: List[List[int]] = matrix storing the neighboring relations
    :param center_coords: np.array[n] = positions of the node coords as complex
    :param p: int = number of edges of a single polygon in the tiling == rotational symmetry
    :return: void
    """
    graph = nx.Graph()
    for y in range(len(adjacent_matrix)):
        if y >= center_coords.shape[0]:
            sector = (y - 1) // (center_coords.shape[0] - 1)
            index = (y - 1) % (center_coords.shape[0] - 1)
            index += 1
            rot = center_coords[index] * np.exp(1j * sector * np.pi * 2 / p)
            x_ = np.real(rot)
            y_ = np.imag(rot)
        else:
            x_ = np.real(center_coords[y])
            y_ = np.imag(center_coords[y])
            sector = 0

        graph.add_node(y, pos=(x_, y_), node_color=colors[sector % len(colors)])

    for y, row in enumerate(adjacent_matrix):
        for index in row:
            if index >= len(adjacent_matrix):
                print(f"Skip: {y} -> {index}")
                continue
            graph.add_edge(y, index)

    nx.draw_networkx(graph, pos=nx.get_node_attributes(graph, 'pos'),
                     node_color=list(nx.get_node_attributes(graph, 'node_color').values()))


@NumbaChecker("Tuple((uint32[:, :], complex128[:]))(int64, int64, int64, float64, uint32[::1], int64, float64)")
def generate_nbrs(p: int, q: int, n: int, r: float, sector_lengths: np.array, degtol: int,
                  mangle: float) -> np.array:
    """
    Generates the tiling with the given parameters p, q, n.
    Time-complexity: O(p^2 m(p, q, n) + n), with m(p, q, n) is the number of polygons
    :param p: int = number of edges
    :param q: int = number of polys per vertex
    :param n: int = number of layers (reflective)
    :param r: float = radius of the fundamental polygon
    :param sector_lengths: np.array[int] = length
    :param degtol: float = tolerance at the boundary
    :param mangle: float = rotation of the center polygon
    :return: np.array[np.uint8] = stores for every polygon which reflection level it has
    """
    dphi = PI2 / p
    phis = np.array([dphi * i + mangle for i in range(p)])  # p

    # coord arrays
    current_coords = np.empty((sector_lengths[-1], p + 1), dtype=np.complex128)
    next_coords = np.empty((sector_lengths[-1], p + 1), dtype=np.complex128)
    # edge arrays
    current_edges = np.empty(sector_lengths[-1], dtype=np.uint16)
    next_edges = np.empty(sector_lengths[-1], dtype=np.uint16)
    # neighbors array
    neighbors = np.empty((np.sum(sector_lengths), p + 1), dtype=np.uint32)
    neighbors.fill(-1)
    neighbors[:, 0] = 1  # +1 for counter in array (skip itself)
    # boundary stuff
    boundary_indices = np.empty((n + 1, 2), dtype=np.uint32)
    boundary_indices[0].fill(0)
    # for plotting
    center_coords = np.empty((np.sum(sector_lengths),), dtype=np.complex128)

    # most inner polygon
    current_coords[0, 0] = 0
    current_coords[0, 1:] = r * np.exp(1j * phis)  # p
    center_coords[0] = current_coords[0, 0]

    # prepare edge_array
    edges = int(2 ** p - 1)
    # eliminate parent edge
    edges ^= 1 << (p - 1)
    current_edges.fill(edges)  # m/p
    next_edges.fill(edges)  # m/p
    # for first poly create only one neighbor
    current_edges[0] = 1

    boundary = PI2 / p + (degtol / 360 * PI2)

    parent_absolut = 0
    child_absolut = 1
    current_counter = 0
    current_open = 1
    next_level_counter = 0
    current_level = 0
    while current_level < n:
        # check for filler polys of 1st order
        if current_counter != 0 and current_level != 0:
            connection = any_close_matrix(next_coords[next_level_counter - 1],
                                          current_coords[current_counter])  # (p+1)^2
            if connection.shape[0] == 2 and child_absolut > 3:
                # block edges in number-bit-array (see. GRK __init__ for explanation)
                next_edges[next_level_counter - 1] ^= 1 << (connection[1, 1] - 1)
                current_edges[current_counter] ^= 1 << (connection[0, 0] - 1)

                # add connection to neighbors
                neighbors[child_absolut - 1, neighbors[child_absolut - 1, 0]] = parent_absolut
                neighbors[child_absolut - 1, 0] += 1
                neighbors[parent_absolut, neighbors[parent_absolut, 0]] = child_absolut - 1
                neighbors[parent_absolut, 0] += 1

        # select polygon
        poly = current_coords[current_counter]
        # for all vertices of polygon:
        for i, vertex in enumerate(poly[1:]):
            """
            Algorithm:
             1. shift vertex into origin
             2. rotate poly such that two vertices are on the x-axis
             3. reflection on the x-axis (inversion of the imaginary part)
             4. rotate poly back to original orientation (it is now reflected)
             5. shift poly back to original position
            """
            # check if edge is blocked
            if not (current_edges[current_counter] & 1 << i):  # not important (time complexity)
                # print(f"continue: {parent_absolut} continues child {child_absolut}")
                continue

            z = poly.copy()  # p  + 1
            array_trans.morigin(p, vertex, z)  # p + 1
            phi = np.angle(z[1:][(i + 1) % p])  # 1
            array_trans.mrotate(p, phi, z)  # p + 1
            z = np.conjugate(z)  # p + 1
            array_trans.mrotate(p, - phi, z)  # p + 1
            array_trans.morigin(p, - vertex, z)  # p + 1

            angle = np.angle(z[0])  # 1
            if angle > boundary:
                break

            if angle >= 0:
                next_coords[next_level_counter, 0] = z[0]
                next_coords[next_level_counter, 1:] = np.roll(np.flip(z[1:]), i + 1)  # p

                # save neighboring relations
                neighbors[parent_absolut, neighbors[parent_absolut, 0]] = child_absolut
                neighbors[parent_absolut, 0] += 1
                neighbors[child_absolut, neighbors[child_absolut, 0]] = parent_absolut
                neighbors[child_absolut, 0] += 1

                # save center coords
                center_coords[child_absolut] = z[0]

                # check for filler of 2nd order and q == 3
                if i == 0:
                    # shares edge with former polygon -> filler of 2nd Order
                    connection = any_close_matrix(next_coords[next_level_counter],
                                                  next_coords[next_level_counter - 1])  # (p+1)^2
                    if connection.shape[0] == 2 and child_absolut > 2:
                        # block edges in number-bit-array (see. GRK __init__ for explanation)
                        next_edges[next_level_counter] ^= 1 << (connection[1, 1] - 1)
                        next_edges[next_level_counter - 1] ^= 1 << (connection[0, 0] - 1)

                        # add connection to neighbors
                        neighbors[child_absolut, neighbors[child_absolut, 0]] = child_absolut - 1
                        neighbors[child_absolut, 0] += 1
                        neighbors[child_absolut - 1, neighbors[child_absolut - 1, 0]] = child_absolut
                        neighbors[child_absolut - 1, 0] += 1

                elif q == 3 and next_level_counter != 0:
                    # close first edge because of sibling
                    next_edges[next_level_counter] ^= 1
                    # close last edge because of sibling
                    if not (next_edges[next_level_counter - 1] & 1 << (p - 2)):
                        # if filler polygon of first order
                        next_edges[next_level_counter - 1] ^= 1 << (p - 3)
                    else:
                        next_edges[next_level_counter - 1] ^= 1 << (p - 2)

                    neighbors[child_absolut, neighbors[child_absolut, 0]] = child_absolut - 1
                    neighbors[child_absolut, 0] += 1
                    neighbors[child_absolut - 1, neighbors[child_absolut - 1, 0]] = child_absolut
                    neighbors[child_absolut - 1, 0] += 1

                next_level_counter += 1
                child_absolut += 1

        current_counter += 1
        parent_absolut += 1
        if current_counter == current_open:
            # add last polygon to boundary list
            boundary_indices[current_level + 1, 0] = child_absolut - next_level_counter
            boundary_indices[current_level + 1, 1] = child_absolut - 1

            # update counters
            current_open = next_level_counter
            current_counter = 0
            next_level_counter = 0

            # update arrays
            current_coords = next_coords
            next_coords = np.empty_like(next_coords)
            current_edges = next_edges
            next_edges = np.empty_like(next_edges)
            next_edges.fill(edges)
            current_level += 1

    # boundary
    ndiff = int(round((q - 1) / 2, 0))
    jump = (p - 1) * (child_absolut - 1)
    dist_ref = f_dist_disc(center_coords[0], center_coords[1]) + 1e-12
    for n1 in range(1, boundary_indices.shape[0]):
        # right
        index_right = boundary_indices[n1, 0]
        for n2 in range(max(1, n1 - ndiff), min(n1 + ndiff + 1, n + 1)):
            # left side
            index_left = boundary_indices[n2, 1]
            dist = f_dist_disc(center_coords[index_left] * np.exp(- 1j * dphi), center_coords[index_right])
            if dist <= dist_ref:
                neighbors[index_right, neighbors[index_right, 0]] = index_left + jump
                neighbors[index_right, 0] += 1
                neighbors[index_left, neighbors[index_left, 0]] = index_right + child_absolut - 1
                neighbors[index_left, 0] += 1

    for i in range(p):
        neighbors[0, 2 + i] = neighbors[0, i + 1] + child_absolut - 1

    return neighbors[:child_absolut, 1:], center_coords[:child_absolut]
