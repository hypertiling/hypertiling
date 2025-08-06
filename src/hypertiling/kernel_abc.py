import abc
import numpy as np
from .ion import htprint
from .neighbors import find_radius_optimized, find_radius_optimized_single
from .util import lattice_spacing_weierstrass, fund_radius

# Magic number: transcendental number (Champernowne constant)
# used as an angular offset, rotates the entire construction slightly during construction
MAGICANGLE = np.radians(0.12345678910111213141516171819202122232425262728293031)


class Graph(abc.ABC):
    """
    The abstract base class of a hyperbolic graph.

    Parameters
    ----------
    p : int
        The first fundamental lattice parameter.
    q : int
        The second fundamental lattice parameter.
    n : int
        A parameter defining the size of the graph.

    Raises
    ------
    AttributeError
        If the combination of p and q is invalid: For hyperbolic lattices (p-2)*(q-2) > 4 must hold.

    Attributes
    ----------
    p : int
        The first fundamental lattice parameter.
    q : int
        The second fundamental lattice parameter.
    n : int
        A parameter defining the size of the graph.
    """

    def __init__(self, p: int, q: int, n: int):
        self.p = p
        self.q = q
        self.n = n

    def __repr__(self):
        """
        Return a string representation of the GraphExtended object.

        Returns
        -------
        str
            String representation of the GraphExtended object.
        """
        return f"Graph {self.p, self.q, self.n}"

    @abc.abstractmethod
    def __len__(self):
        """
        Abstract method to return the length of the GraphExtended object.
        """
        pass


class GraphExtended(Graph):
    """
    Extension to the hyperbolic Graph ABC yielding further information for lattice construction

    Parameters
    ----------
    p : int
        The first fundamental lattice parameter.
    q : int
        The second fundamental lattice parameter.
    n : int
        A parameter defining the size of the graph.
    mangle : float, optional
        Magic angle required for technical reasons, by default MAGICANGLE

    Raises
    ------
    AttributeError
        If the combination of p and q is invalid: For hyperbolic lattices (p-2)*(q-2) > 4 must hold.

    Attributes
    ----------
    p : int
        The first fundamental lattice parameter.
    q : int
        The second fundamental lattice parameter.
    n : int
        A parameter defining the size of the graph.
    phi : float
        Angle of rotation that leaves the lattice invariant when cell centered.
    qhi : float
        Angle of rotation that leaves the lattice invariant when vertex centered.
    r : float
        Radius of the fundamental polygon in the Poincare disk.
    h : float
        Hyperbolic/geodesic lattice spacing, i.e., the edge length of any cell.
    hr : float
        Geodesic radius, i.e., distance between the center and any vertex of cells in a regular p,q tiling.
    mangle : float
        Magic angle required for technical reasons.
    _nbrs : None
        A placeholder for storing adjacency relations.
    """

    def __init__(self, p: int, q: int, n: int, mangle: float = MAGICANGLE):
        if not ((p - 2) * (q - 2) > 4):
            raise AttributeError("Invalid combination of p and q: For hyperbolic lattices (p-2)*(q-2) > 4 must hold!")
        super().__init__(p, q, n)

        self.phi = 2 * np.pi / self.p
        self.qhi = 2 * np.pi / self.q

        self.r = fund_radius(self.p, self.q)
        self.h = lattice_spacing_weierstrass(self.p, self.q)
        self.hr = lattice_spacing_weierstrass(self.q, self.p)

        self.mangle = mangle
        self._nbrs = None

    def __repr__(self):
        """
        Return a string representation of the GraphExtended object.

        Returns
        -------
        str
            String representation of the GraphExtended object.
        """
        return f"GraphExtended {self.p, self.q, self.n}"


class Tiling(GraphExtended):
    """
    This is the abstract base class of a hyperbolic tiling.

    Attributes
    ----------
    p : int
        The first fundamental lattice parameter.
    q : int
        The second fundamental lattice parameter.
    n : int
        A parameter defining the size of the tiling. 
    phi : float
        Angle of rotation that leaves the lattice invariant when cell centered.
    qhi : float
        Angle of rotation that leaves the lattice invariant when vertex centered.
    r : float
        Radius of the fundamental polygon in the Poincare disk.
    h : float
        Hyperbolic/geodesic lattice spacing, i.e., the edge length of any cell.
    hr : float
        Geodesic radius, i.e., distance between the center and any vertex of cells in a regular p,q tiling.
    mangle : float
        Magic angle required for technical reasons.
    _nbrs : None
        A placeholder for storing adjacency relations.
    """


    def __repr__(self):
        return f"Tiling {self.p, self.q, self.n}"

    @abc.abstractmethod
    def get_layer(self, index: int) -> int:
        """
        Returns the layer to the center of the polygon at index.

        Parameters
        ----------
        index : int
            index of the polygon

        Returns
        -------
        int
            layer of the polygon
        """
        pass

    @abc.abstractmethod
    def get_sector(self, index: int) -> int:
        """
        Returns the sector, the polygon at index refers to.

        Parameters
        ----------
        index : int
            index of the polygon

        Returns
        -------
        int
            number of the sector
        """
        pass

    @abc.abstractmethod
    def get_center(self, index: int) -> np.complex128:
        """
        Returns the center of the polygon at index.

        Parameters
        ----------
        index : int
            index of the polygon

        Returns
        -------
        np.complex128
            center of the polygon
        """
        pass



    @abc.abstractmethod
    def get_vertices(self, index: int) -> np.array:
        """
        Returns the p vertices of the polygon at index.

        Parameters
        ----------
        index : int
            index of the polygon

        Returns
        -------
        np.array[np.complex128][p]
            vertices of the polygon
        """
        pass

    @abc.abstractmethod
    def get_angle(self, index: int) -> float:
        """
        Returns the angle to the center of the polygon at index.

        Parameters
        ----------
        index : int
            index of the polygon

        Returns
        -------
        float
            center of the polygon
        """
        pass


    def get_nbrs_list(self, **kwargs):
        """
        Calculates for each vertex the neighbours a returns a list.

        Returns
        -------
        List of list
            with neighbour indices
        """
        if self._nbrs is None:
            htprint("Status", "Mapping neighbours for entire lattice using 'optimized radius search' algorithm.")
            self._nbrs = find_radius_optimized(self, **kwargs)
        return self._nbrs


    def get_nbrs(self, i, **kwargs):
        """
        Returns the indices of the neighbours of vertex i.

        Parameters
        ----------
        i : int
            the cell index one is interested in.

        Returns
        -------
        list
            of all neighbour indice
        """
        if self._nbrs is None:
            htprint("Status", "Performing radius search for one vertex. If neighbours of many points are required, we recommend to use 'get_nbrs_list'.")
            return find_radius_optimized_single(self, i, **kwargs)
        else:
            return self._nbrs[i]
