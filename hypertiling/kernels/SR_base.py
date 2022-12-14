import numpy as np
import math
import copy
from ..kernel_abc import Tiling
from ..transformation import moeb_rotate_trafo
from ..representations import p2w
from ..arraytransformation import mfull, mrotate, morigin, multi_rotation_around_vertex
from ..util import fund_radius, lattice_spacing_weierstrass, euclidean_center
from ..geodesics import geodesic_midpoint
from ..ion import htprint
from ..distance import lorentzian_distance
from ..neighbors import find_radius_brute_force, find_radius_optimized

PI2 = 2 * np.pi

class KernelStaticBase(Tiling):
    """
    Base class of the static rotational kernel family
    provides interfaces and fundamental polygon

    Attributes
    ----------


    Methods
    -------
    __getitem__(idx)
        returns the idx-th HyperPolygon in the tiling

    __iter__()
        traverses throught all HyperPolygons in the tiling

    __next__()
        returns the next HyperPolygon in the tiling

    __len__()
        returns the size of the tiling, which is the number of cells

    """

    def __init__(self, p, q, nlayers, center="cell", autogenerate=True, radius=None):
        super().__init__(p, q, nlayers)

        # main attributes
        self.p = p  # number of edges (and thus number of vertices) per polygon
        self.q = q  # number of polygons that meet at each vertex
        self.nlayers = nlayers  # layers of the tessellation
        self.center = center  # tiling can be centered around a "cell" (default) or a "vertex"
        self.radius = radius # a cut-off radius (implement me!)
        self.autogenerate = autogenerate # determines whether the lattice is constructed upon class instantiation or only after call to self.generate

        # half fundamental radius
        self.fr2 = fund_radius(self.p, self.q) / 2

        # symmetry angles
        self.phi = 2 * math.pi / self.p  # angle of rotation that leaves the lattice invariant when cell centered
        self.qhi = 2 * math.pi / self.q  # angle of rotation that leaves the lattice invariant when vertex centered

        # prepare list to store polygons 
        self.polygons = []

        if center not in ['cell', 'vertex']:
            raise ValueError('[hypertiling] Error: Invalid value for argument "center"!')

    def __getitem__(self, idx):
        return self.polygons[idx]

    def __iter__(self):
        self.iterctr = 0
        self.itervar = self.polygons[self.iterctr]
        return self

    def __next__(self):
        if self.iterctr < len(self.polygons):
            retval = self.polygons[self.iterctr]
            self.iterctr += 1
            return retval
        else:
            raise StopIteration

    def __len__(self):
        return len(self.polygons)

    def get_vertices(self, index: int) -> np.array:
        """
        Returns the p vertices of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: np.array[np.complex128][p] = vertices of the polygon
        """
        return self.polygons[index].verticesP[:-1]

    def get_center(self, index: int) -> np.complex128:
        """
        Returns the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: np.complex128 = center of the polygon
        """
        return self.polygons[index].verticesP[-1]

    def get_sector(self, index: int) -> int:
        """
        Returns the sector, the polygon at index refers to.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: int = number of the sector
        """
        return self.polygons[index].sector

    def get_angle(self, index: int) -> float:
        """
        Returns the angle to the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: float = angle of the polygon
        """
        return self.polygons[index].angle

    def get_layer(self, index: int) -> int:
        """
        Returns the layer to the center of the polygon at index.
        Time-complexity: O(1)
        :param index: int = index of the polygon
        :return: int = layer of the polygon
        """
        return self.polygons[index].layer



    def create_fundamental_polygon(self, rotate_by=None):
        """
        Constructs the vertices of the fundamental hyperbolic {p,q} polygon

        Parameters
        ----------
        center : str
            decides whether the fundamental cell is construct centered at the origin ("cell", default) 
            or with the origin being one of its vertices ("vertex")
        rotate_by : float
            angle of rotation of the fundamental polygon, default is the magic angle mangle
        """
        if rotate_by is None:
            rotate_by = self.mangle

        r = fund_radius(self.p, self.q)
        polygon = HyperPolygon(self.p)

        for i in range(self.p):
            z = complex(math.cos(i * self.phi), math.sin(i * self.phi))  # = exp(i*phi)
            z = z / abs(z)
            z = r * z
            polygon.verticesP[i] = z

        # rotate by angle (to get away from the coordinate axis)
        mrotate(self.p, -rotate_by, polygon.verticesP)

        return polygon


    def _create_first_layer(self):
        """
        generate the first layer
        this is one polygon for cell-centered and q polygons for vertex-centered
        """

        # create fundamental polygon
        self.fund_poly = self.create_fundamental_polygon()

        # prepare polygon counter
        self.counter = 0

        # tiling centered around cell
        # add fundamental cell and set bounds of current layer
        if self.center == "cell":
            self.polygons.append(self.fund_poly)
            self.outmost_layer_lower = 0
            self.outmost_layer_upper = 1


        # tiling centered around vertex
        if self.center == 'vertex':

            # shift fundamental polygon such that one of its vertices is on the origin
            vertidx = 0           
            morigin(self.p, self.fund_poly.verticesP[vertidx], self.fund_poly.verticesP)
            
            # generate the q polygons of the first layer
            for rot_ind in range(self.q):
                polycopy = copy.deepcopy(self.fund_poly)
                adj_pgon = self.generate_adj_poly(polycopy, vertidx, rot_ind)
                self.polygons.append(adj_pgon)

            self.outmost_layer_lower = 0
            self.outmost_layer_upper = self.q




class KernelRotationalCommon(KernelStaticBase):
    """
    Extend base class of the static rotational kernel family towards sector contruction
    """

    def __init__(self, p, q, n, center, autogenerate, radius):
        super(KernelRotationalCommon, self).__init__(p, q, n, center, autogenerate, radius)

        # sector boundary tolerance / softness
        # do not change, unless you know what you are doing!)
        self.degtol = 1
        self.radtol = np.radians(self.degtol)

        # angle width of the fundamental sector
        if self.center == "cell":
            self.sect_angle     = self.phi
        if self.center == "vertex":
            self.sect_angle     = self.qhi

        # required for construction algorithm
        self.sect_lbound = 0
        self.sect_ubound = self.mangle + self.sect_angle + self.radtol

        self.upper_slice = self.sect_angle - self.radtol
        self.lower_slice = self.mangle + self.radtol

   
    def _replicate(self):
        """
        tessellate the entire disk by replicating the fundamental sector
        """
        if self.center == 'cell':
            self._angular_replicate(copy.deepcopy(self.polygons), self.p)
        elif self.center == 'vertex':
            self._angular_replicate(copy.deepcopy(self.polygons), self.q)


    def _generate_adj_poly(self, polygon, ind, k):
        """
        finds the next polygon by k-fold rotation of polygon around the vertex number ind
        """
        mfull(self.p, k * self.qhi, ind, polygon.verticesP)
        return polygon


    def _populate_sector(self, dupl_large, dupl_small):
        """
        compute all polygons within one sector of the tiling
        """

        startpgon = 0
        endpgon = 1

        # loop over layers to be constructed
        for l in range(1, self.nlayers):

            # computes all neighbor polygons of layer l
            for pgon in self.polygons[startpgon:endpgon]:

                # center of current polygon
                pgon_center = pgon.verticesP[self.p]
                
                # iterate over every vertex of pgon
                for vert_ind in range(self.p):

                    # rotate polygon around current vertex
                    # compute center coordinates of all polygons which share this vertex...
                    adj_centers = multi_rotation_around_vertex(self.q, self.qhi, pgon.verticesP[vert_ind], pgon_center)            
                    
                    # ... and iterate over them
                    for rot_ind in range(self.q):

                        center = adj_centers[rot_ind]

                        # check whether candidate polygon is in fundamental sector
                        if self._in_sector(center):   

                            # check whether candidate polygon already exists
                            if not dupl_large.is_duplicate(center):

                                # add to duplicate container
                                dupl_large.add(center)

                                # create copy
                                polycopy = copy.deepcopy(pgon)

                                # generate adjacent polygon and add to large list
                                adj_pgon = self._generate_adj_poly(polycopy, vert_ind, rot_ind)
                                adj_pgon.layer = l + 1
                                self.polygons.append(adj_pgon)

                                # if angle is in lower soft sector boundary, add to second duplicate container
                                if self._in_slice_lower(center): 
                                    if not dupl_small.is_duplicate(center):
                                        dupl_small.add(center)

            startpgon = endpgon
            endpgon = len(self.polygons)

        # free mem of centerset
        del dupl_large

        # --- filter out rotational duplicates
        deletelist = []
        
        # go through every polygon
        for kk, pgon in enumerate(self.polygons):
            center = pgon.verticesP[self.p]
            # if poly is inside soft boundary 
            # it has to be considered for rotational duplicate check
            if self._in_slice_upper(center): 
                # rotate center of poly back by sector angle
                center = moeb_rotate_trafo(-self.sect_angle, pgon.verticesP[self.p])
                # check whether we already have this rotated center
                # if so: rotational duplicate
                if dupl_small.is_duplicate(center):
                    # delete
                    deletelist.append(kk)

        # delete all rotational duplicates
        self.polygons = list(np.delete(self.polygons, deletelist))




    def _angular_replicate(self, polygons, k):
        """
        tessellates the disk by applying a rotation of 2pi/p to the pizza slice
        """

        # central polygon is not assigned to a sector and will hence not be replicated
        if self.center == 'cell':
            polygons.pop(0)  
            angle = self.phi
            k = self.p
        # no central polygon if tiling is centered around a vertex
        elif self.center == 'vertex':
            angle = self.qhi
            k = self.q

        # perform angular replication
        for p in range(1, k):
            for polygon in polygons:
                pgon = copy.deepcopy(polygon)
                mrotate(self.p, -p * angle, pgon.verticesP)
                self.polygons.append(pgon)

        # assign index and angles 
        for num, poly in enumerate(self.polygons):
            poly.idx = num
            poly.find_angle()
            poly.find_sector(k)


# ------------- Filters -------------

    def _in_sector(self, z0):
        """
        Check whether point z0 is located in fundamental sector of the tiling
        """
        cangle = math.atan2(z0.imag, z0.real)
        if (self.sect_lbound <= cangle < self.sect_ubound) and (abs(z0) > self.fr2):
            return True
        else:
            return False

    def _in_slice_lower(self, z0):
        """
        Check whether point z0 is located in lower soft boundary of fundamental sector
        This is required in order to check for rotational duplicates during the construction
        """
        cangle = math.atan2(z0.imag, z0.real)
        return cangle < self.lower_slice

    def _in_slice_upper(self, z0):
        """
        Check whether point z0 is located in upper soft boundary of fundamental sector
        This is required in order to check for rotational duplicates during the construction
        """
        cangle = math.atan2(z0.imag, z0.real)
        return cangle > self.upper_slice


# ------------- Other Stuff -------------

    def populate_edge_list(self, digits=12):
        """
        populate the "edges" list of all polygons in the tiling        
        note: some neighbour methods employ the fact that adjacent polygons share an edge
        hence these will later be identified via floating point comparison and we need to round
        """

        for poly in self.polygons:
            poly.edges = []
            verts = np.round(poly.verticesP[0:-1], digits)

            # append edges as tuples
            for i, vert in enumerate(verts[:-1]):
                poly.edges.append((verts[i], verts[i + 1]))
            poly.edges.append((verts[-1], verts[0]))


# ------------- Transformations -------------


    def rotate(self, angle: float, deg=False):
        """
        Rotates the whole tiling about the origin.
        
        Parameters
        ----------
        angle: float
            Angle in radians by which the tiling is rotated.
        
        deg: bool, default: False
            If True, then angle is considered in units of degrees.
        """

        if deg:
            angle = angle * math.pi / 180

        for poly in self.polygons:
            mrotate(self.p, angle, poly.verticesP)

    def translate(self, z):
        """ 
        Translates the whole tiling so that the point z lays in the origin.
        
        Parameters
        ----------
        z: complex
            The point which will be translated to the origin.
        """

        for poly in self.polygons:
            morigin(self.p, z, poly.verticesP)




# ------------- Refinements -------------


    def refine_lattice(self, iterations=1):
        """ 
        Refine a regular lattice, by subdividing each triangle into four new polygons
        If the tiling is not triangular, in the first step, all cells will be subdivided
        into p triangular cells
        Note that new cells are not isometric anymore!
        
        Parameters
        ----------
        
        iterations: int
            Determines how many times the lattice will be refined; for each iteration the
            total number of polygons will be multiplied by a factor of four
            
        """

        if iterations == 0:
            return

        # if tiling is not triangular, the first refinement steps subdivided all cells
        # into p triangles
        if self.p > 3:
            newpolygons = []
            for pgon in self.polygons:
                for vrtx in range(self.p):
                    child = HyperPolygon(3) 
                    child.verticesP[0] = pgon.verticesP[vrtx]
                    child.verticesP[1] = pgon.verticesP[(vrtx+1)%self.p]
                    child.verticesP[2] = pgon.verticesP[-1]
                    child.verticesP[3] = euclidean_center(child.verticesP[:-1])
                    child.layer = pgon.layer
                    newpolygons.append(child)
            self.polygons = newpolygons
            iterations -= 1 # we have already done one iteration

        
        for _ in range(iterations):
            p = 3 # we use this quite frequently, hence the short form
            newpolygons = []  # stores the new polygons
            for num, pgon in enumerate(self.polygons):  # find the new vertices of each polygon
                ref_vertices = []  # stores newly found vertices through refinement
                # loop through polygon edges
                for vrtx in range(p):
                    # find geodesic midpoint
                    zm = geodesic_midpoint( pgon.verticesP[vrtx], pgon.verticesP[(vrtx+1)%p] )
                    ref_vertices.append(zm)


                # one "mother" triangle bears 4 "children" triangles, one in its mid
                # and three that each share one vertex with their mother

                child = HyperPolygon(p)  # the center triangle whose vertices are the newly found refined ones
                child.layer = pgon.layer
                for i in range(p):
                    child.verticesP[i] = ref_vertices[i]
                child.verticesP[-1] = pgon.centerP()  # the center triangle shares its center with its mother
                child.idx = 4*num+1  # assigning a unique number
                newpolygons.append(child)

                for vrtx in range(p):  # for each vertex of the mother triangle that is being refined
                    child = HyperPolygon(p)  # these are the non-center children
                    child.layer = pgon.layer
                    vP = [pgon.verticesP[vrtx], ref_vertices[vrtx], ref_vertices[vrtx-1]]
                    for i in range(p):
                        child.verticesP[i] = vP[i]
                    child.verticesP[-1] = euclidean_center(child.verticesP[:-1])
                    child.idx = (4*num+1)+1+vrtx  # assign a unique number
                    newpolygons.append(child)

            self.polygons = newpolygons
        return



# ------------- Neighbours -------------

    def get_nbrs_list(self, method="ROS", **kwargs):

        methods = { "RBF":  self.get_nbrs_radius_brute_force,
                    "RO":   self.get_nbrs_radius_optimized,
                    "ROS":  self.get_nbrs_radius_optimized_slice,
                    "EMO":  self.get_nbrs_edge_map_optimized,
                    "EMBF": self.get_nbrs_edge_map_brute_force}

        return methods[method](**kwargs)


    def get_nbrs_radius_brute_force(self, **kwargs):
        return find_radius_brute_force(self, **kwargs)

    def get_nbrs_radius_optimized(self, **kwargs):
        return find_radius_optimized(self, **kwargs)


    # Radius Optimized Slice (ROS)
    def get_nbrs_radius_optimized_slice(self, radius=None, eps=1e-5):
        """
        Uses both the benefits of of numpy vectorization (used also in neighbours.find_radius_optimized)
        and furthermore applies the radius search only to a p-fold sector of the tiling

        currently only working for cell-centered tilings, although there have already been 
        attempts in this direction (TODO!)

        Attributes
        ----------
        radius : float
            the expected distance between neighbours
        eps : float
            increase radius a little in order to make it more stable

        Returns
        -------
        List of list of integers, where sublists i contains the indices of the neighbour of vertex i in the tiling 
        """

        if self.center == "vertex":
            raise NotImplementedError("[hypertiling] Error: Currently this method does not support vertex-centered tilings!")


        if radius is None:
            htprint("Status", "No search radius provided; Assuming lattice spacing of the (p,q) tessellation!")
            radius = lattice_spacing_weierstrass(self.p, self.q)
            htprint("Status", "Found (p,q) = (%i,%i) and auto-calculated a neighbour distance of %5.4f. Can be changed using the 'radius' argument." % (self.p, self.q, radius))

        totalnum = len(self)  # total number of polygons
        p = self.p  # number of edges of each polygon
        q = self.q
        if self.center == "cell":
            pps = int((totalnum - 1) / p)  # polygons per sector (excluding center)
            inc = 1
        elif self.center == "vertex":
            pps = int(totalnum / q)
            inc = 0

        # shifts local neighbour list to another sector
        def shift(lst, times):
            lst = sorted(lst)
            haszero = (0 in lst)

            # fundamental cell requires a little extra care
            if haszero:
                lsta = np.array(lst[1:]) + times * pps
                lsta[lsta > totalnum - 1] -= (totalnum - 1)
                lsta[lsta < 1] += (totalnum - 1)
                return sorted([0] + list(lsta))
            else:
                lsta = np.array(lst) + times * pps
                lsta[lsta > totalnum - 1] -= (totalnum - 1)
                lsta[lsta < 1] += (totalnum - 1)
                return sorted(list(lsta))


        # slice the first three sectors
        # we are gonna look for neighbours of polygons in the second sector 
        pgons = self.polygons[:3 * pps + inc]
        # this is a place where the algorithm can be further improved, performance-wise
        # we do not need the entire 1st and 3rd sectors, but only those cells close to 
        # the boundary of the 2nd sector

        # store center coordinates in array for faster access
        v = np.zeros((len(pgons), 3))
        for i, poly in enumerate(pgons):
            v[i] = poly.centerW()

        # the search distance (we are doing a radius search)
        searchdist = radius + eps
        searchdist = np.cosh(searchdist)

        # prepare list
        nbrlst = []
        # loop over polygons
        for i, poly in enumerate(pgons[pps + inc:2 * pps + inc]):
            w = poly.centerW()
            dists = lorentzian_distance(v, w)
            dists[(dists < 1)] = 1  # this costs some %, but reduces warnings
            indxs = np.where(dists < searchdist)[0]  # radius search
            selff = np.argwhere(indxs == poly.idx)  # find self
            indxs = np.delete(indxs, selff)  # delete self
            nums = [pgons[ind].idx for ind in indxs]  # replacing indices by actual polygon number
            nbrlst.append(nums)

        # prepare full output list
        retlist = []

        if self.center == "cell":
            k = p
        elif self.center == "vertex":
            k = q

        if self.center == "cell":
            # fundamental cell
            lstzero = []
            for ps in range(0, k):
                lstzero.append(ps * pps + 1)
            retlist.append((lstzero))

        # first sector
        for lst in nbrlst:
            retlist.append(shift(lst, -1))

        # second sector
        for lst in nbrlst:
            retlist.append(lst)

        # remaining sectors
        for ps in range(2, k):
            for lst in nbrlst:
                retlist.append(shift(lst, ps - 1))

        return retlist




    # Edge Map Optimized (EMO)
    def get_nbrs_edge_map_optimized(self):
        """
        Find neighbours by identifying corresponding edges among polygons
        This is a coordinate-free algorithm, it uses only the graph structure
        Can probably be further improved

        Returns
        -------
        List of list of integers, where sublists i contains the indices of the neighbour of vertex i in the tiling 
        """

        self.populate_edge_list()

        # we create a kind of "dictionary" where keys are the
        # edges and values are the corresponding polygon indices
        edges, vals = [], []
        for poly in self.polygons:
            for edge in poly.edges:
                edges.append(edge)
                vals.append(poly.idx)

        # reshape "edges" into its components in order to 
        # make use of numpy vectorization later
        edge0 = np.zeros(len(edges)).astype(complex)
        edge1 = np.zeros(len(edges)).astype(complex)
        for i, edge in enumerate(edges):
            edge0[i] = edge[0]
            edge1[i] = edge[1]

        # create empty neighbour array
        nbrs = []
        for i in range(len(self.polygons)):
            nbrs.append([])

        # an edge that is share by two polygons appears twice
        # in the "edges" list; we find the corresponding polygon
        # indices by looping over that list
        for i, edge in enumerate(edges):
            # compare against full edges arrays
            # this avoids a double loop which is slow ...
            # check edge
            bool_array1 = (edge[0] == edge0)
            bool_array2 = (edge[1] == edge1)
            # check also reverse orientation
            bool_array3 = (edge[0] == edge1)
            bool_array4 = (edge[1] == edge0)

            # put everything together; we require 
            # (True and True) or (True and True)
            b = bool_array1 * bool_array2 + bool_array3 * bool_array4

            # find indices where resulting boolean array is true
            w = np.where(b)

            # these indices are neighbours of each other
            for x in w[0]:
                if vals[i] is not vals[x]:
                    nbrs[vals[i]].append(vals[x])

        return nbrs




    # Edge Map Brute Force (EMBF)
    def get_nbrs_edge_map_brute_force(self):
        """
        Find neighbours by identifying corresponding edges among polygons
        This is a coordinate-free algorithm, it uses only the graph structure
    
        There is an equivalent method available that is much faster: get_nbrs_edge_map
        Use this method only for debugging purposes

        Returns
        -------
        List of list of integers, where sublists i contains the indices of the neighbour of vertex i in the tiling 
        """

        self.populate_edge_list()

        # we create a kind of "dictionary" where keys are the
        # edges and values are the corresponding polygon indices
        edges, vals = [], []
        for poly in self.polygons:
            for edge in poly.edges:
                edges.append(edge)
                vals.append(poly.idx)

        # create empty neighbour array       
        nbrs = []
        for i in range(len(self.polygons)):
            nbrs.append([])

        # an edge that is share by two polygons appears twice
        # in the "edges" list; we find the corresponding polygon
        # indices by looping over that list twice
        for i, k1 in enumerate(edges):
            for j, k2 in enumerate(edges):
                if k1[0] == k2[0] and k1[1] == k2[1]:
                    # check edge
                    if vals[i] is not vals[j]:
                        nbrs[vals[i]].append(vals[j])
                # check also reverse orientation    
                elif k1[1] == k2[0] and k1[0] == k2[1]:
                    if vals[i] is not vals[j]:
                        nbrs[vals[i]].append(vals[j])

        return nbrs



          
# defines a hyperbolic polygon

class HyperPolygon:
    """
    Hyperbolic polygon object

    Attributes
    ----------

    p : int
        number of outer vertices (edges)

    verticesP : ndarray
        1D array of np.complex128 type, containting positions of vertices and the polygon center
        in Poincare disk coordinates

    idx : int
        auxiliary scalar index; can be used, e.g, for easy identifaction inside a tiling

    layer : int
        encodes in which layer of a tessellation this polygons is located
        
    sector : int
        index of the sector this polygons is located

    angle : float
        angle between center and the positive x-axis

    val : float
        assign a value (useful in any application)

    orientation : float
        the angle between the line defined by the center and vertices 0, and the abscissa


    Methods
    -------

    centerP()
        returns the center of the polygon in Poincare coordinates

    centerW()
        returns the center of the polygon in Weierstrass coordinates
    
    __equal__()
        checks whether two polygons are equal by comparing centers and orientations

    transform(tmat)
        apply Moebius transformation matrix "tmat" to  all points (vertices + center) of the polygon

    tf_full(ind, phi)
        transforms the entire polygon: to the origin, rotate it and back again


    ... to be completed


    """

    def __init__(self, p):

        self.p = p
        self.idx = 1
        self.layer = 1
        self.sector = 0
        self.angle = 0
        self.val = 0
        self.orientation = 0

        # Poincare disk coordinates
        self.verticesP = np.zeros(shape=self.p + 1, dtype=np.complex128)  # vertices + center


    def centerP(self):
        return self.verticesP[self.p]

    def centerW(self):
        return p2w(self.verticesP[self.p])

    # checks whether two polygons are equal
    def __eq__(self, other):
        if isinstance(other, HyperPolygon):
            centers = cmath.isclose(self.centerP, other.centerP)
            if not centers:
                return False
            orientations = cmath.isclose(self.orientation, other.orientation)
            if not orientations:
                return False
            if self.p == other.p:
                return True
        return False

    # transforms the entire polygon: to the origin, rotate it and back again
    def tf_full(self, ind, phi):
        mfull(self.p, phi, ind, self.verticesP)

    # transforms the entire polygon such that z0 is mapped to origin
    def moeb_origin(self, z0):
        morigin(self.p, z0, self.verticesP)

    def moeb_rotate(self, phi):  # rotates each point of the polygon by phi
        mrotate(self.p, phi, self.verticesP)

    def moeb_translate(self, s):
        for i in range(self.p + 1):
            z = moeb_translate_trafo(self.verticesP[i], s)
            self.verticesP[i] = z

    def rotate(self, phi):
        rotation = np.exp(complex(0, phi))
        for i in range(self.p + 1):
            z = self.verticesP[i]
            z = z * rotation
            self.verticesP[i] = z

    # compute angle between center and the positive x-axis
    def find_angle(self):
        self.angle = math.atan2(self.centerP().imag, self.centerP().real)
        self.angle += PI2 if self.angle < 0 else 0

    def find_sector(self, k, offset=0):
        """ 
        compute in which sector out of k sectors the polygon resides

        Arguments
        ---------
        k : int
            number of equal-sized sectors
        offset : float, optional
            rotate sectors by an angle
        """

        self.sector = math.floor((self.angle - offset) / (PI2 / k))

    # mirror on the x-axis
    def mirror(self):
        for i in range(self.p + 1):
            self.verticesP[i] = complex(self.verticesP[i].real, -self.verticesP[i].imag)
        self.find_angle()

    # returns value between -pi and pi
    def find_orientation(self):
        self.orientation = np.angle(self.verticesP[0] - self.centerP())
