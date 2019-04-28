import os, time
import numpy as np
from scipy import optimize
from spatialnde.coordframes import coordframe
from spatialnde.ndeobj import ndepart
from spatialnde.cadpart.polygonalsurface_texcoordparameterization import polygonalsurface_texcoordparameterization

from autofiber import geodesic as GEO
from autofiber import analyze_uv as AUV
from autofiber import optimization as OP


def calcunitvector(vector):
    """ Returns the unit vector of the vector.  """
    if len(vector.shape) >= 2:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]
    else:
        return vector / np.linalg.norm(vector)


class AutoFiber:

    def __init__(self, cadfile, initpoint, initdirection, initnormal, error=0.01, **kwargs):
        """
        Calculate composite fiber orientations on a surface
        :param cadfile: Path to CAD file (currently supports x3d and stl)
        :param initpoint: 3D point closest to the corner of the surface we would like to work on
        :param initdirection: 3D vector representing the fiber direction at initpoint
        :param initnormal: 3D vector inicating the surface normal at initpoint
        (used to determine which surface to operate on)
        :param kwargs: options: E (Young's Modulus) = 228, nu (Poisson's ratio) = 0.2, fiberint = 0.1, accel = False
        """
        # Get CAD file of part that we would like to lay composite fiber over
        self.cadfile = cadfile

        # Gather options, set to default if option doesn't exist
        # accel: activate opencl optimization features
        self.accel = kwargs.get("accel", False)

        # Init spatialnde objects
        self.obj = None
        self.objframe = None

        # load model into spatialnde obj
        self.loadobj()

        # Init model variables
        self.vertices = None
        self.vertexids_indices = None
        self.vertexids = None
        self.facetnormals = None
        self.refpoints = None
        self.inplanemat = None
        self.edges = None
        self.adjacencyidx = None
        self.surfaces = None
        self.surface_vertexids = None
        self.boxes = None
        self.boxpolys = None
        self.boxcoords = None

        # load relevant model variables
        self.loadvars()

        # Calculate the centroid of each element
        self.centroids = self.vertices[self.vertexids].sum(axis=1)/3

        # Init fiber material properties
        self.fiberint = kwargs.get("fiberint", 0.1)
        self.E = kwargs.get("E", 228.0)
        self.nu = kwargs.get("nu", 0.2)

        # Init geodesic variables
        self.startpoints = np.empty((0, 3))
        self.startuv = np.empty((0, 2))
        self.startelements = np.empty(0, dtype=int)
        self.sfiberdirections = np.empty((0, 3))
        self.fiberdirections = np.empty((self.vertexids.shape[0], 3)) * np.nan
        self.fiberpoints_local = np.empty((self.vertices.shape[0], 2)) * np.nan

        # initnormal: a vector in 3 space which corresponds to the normal of the surface we are working on
        self.surfacenormal = initnormal
        # Init geodesic path record
        self.georecord = {}
        self.geoints = []

        # Init uv parameterization parameters
        self.geoparameterization = np.empty((self.vertices.shape[0], 2)) * np.nan
        # Init optimization parameterization
        self.optimizedparameterization = None
        # Calculate compliance tensor
        if isinstance(self.E, list):
            # Orthotropic
            G = kwargs.get("G", None)
            if G is None:
                raise ValueError("G property is not defined.")
            self.compliance_tensor = np.array([[1/self.E[0], -self.nu[0]/self.E[1], 0],
                                               [-self.nu[0]/self.E[0], 1/self.E[1], 0],
                                               [0, 0, 1/(2*G[0])]])
        else:
            # Isotropic
            G = self.E / (2 * (1 + self.nu))
            self.compliance_tensor = np.array([[1/self.E, -self.nu/self.E, 0],
                                               [-self.nu/self.E, 1/self.E, 0],
                                               [0, 0, 1/G]])
        # Calculate stiffness tensor
        self.stiffness_tensor = np.linalg.inv(self.compliance_tensor)
        # Calculate 2D normalized points for each element
        self.normalized_2d = OP.calc2d(self.obj, self.vertices[self.vertexids])

        initdirection = GEO.rot_vector_angle(initdirection, initnormal, error)

        t1, t2, projpnt = GEO.find_element_within(initpoint, initdirection, initnormal, self.vertices, self.vertexids, self.facetnormals, self.inplanemat)
        if t1 is None:
            closestvertex_ind = np.where(np.linalg.norm(self.vertices - initpoint, axis=1) == np.min(np.linalg.norm(
                self.vertices - initpoint, axis=1)))
            initpoint = self.vertices[closestvertex_ind[0], :][0]
        elif projpnt is not None:
            initpoint = projpnt
        self.initpoint = initpoint

        # Determine which surface we will be laying the geodesics upon
        self.determine_surface(initpoint, initdirection)

        # Find start points for all geodesics
        self.find_startpoints(initpoint, initdirection, initnormal, np.array([0.0, 0.0]))

        # Calculate each geodesic across the surface
        self.calc_geodesics(0)

        self.create_parameterization()

        # self.plot_geodesics()

        # Optimize the geodesic parametrization based on strain energy density
        # self.optimizedparameterization, self.loss = self.fiberoptimize(self.geoparameterization)

        # With results we will calculate the fiber directions based on the available parametrizations
        # texcoords2inplane = self.calctransform(self.optimizedparameterization)
        # self.orientations = calcorientations_abaqus(self.centroids, self.vertices, self.vertexids, self.inplanemat,
        #                                             texcoords2inplane, self.obj.implpart.surfaces[0].boxes,
        #                                             self.obj.implpart.surfaces[0].boxpolys,
        #                                             self.obj.implpart.surfaces[0].boxcoords)

    def loadobj(self):
        if isinstance(self.cadfile, str):
            if os.path.splitext(self.cadfile)[1] == ".x3d":
                self.objframe = coordframe()
                self.obj = ndepart.fromx3d(self.objframe, None, self.cadfile, tol=1e-6)
            elif os.path.splitext(self.cadfile)[1] in [".stl", ".STL"]:
                self.objframe = coordframe()
                self.obj = ndepart.fromstl(self.objframe, None, self.cadfile, tol=1e-6)
            else:
                raise Exception("Unsupported file type.")
        elif isinstance(self.cadfile, object):
            print("Loading %s data type." % self.cadfile.__class__.__name__)
            if self.cadfile.__class__.__name__ is "DMObject":
                self.objframe = coordframe()
                self.obj = ndepart.fromobject(self.objframe, None, self.cadfile, recalcnormals=False, tol=1e-6)
            else:
                raise Exception("Unsupported object type.")
        else:
            raise Exception("Unsupported data type.")

    def loadvars(self):
        """ Load spatialnde data """
        self.vertices = self.obj.implpart.surfaces[0].vertices
        self.vertexids_indices = self.obj.implpart.surfaces[0].vertexidx_indices
        self.vertexids = self.obj.implpart.surfaces[0].vertexidx.reshape(self.vertexids_indices.shape[0], 4)[:, 0:3]
        self.facetnormals = self.obj.implpart.surfaces[0].facetnormals
        self.refpoints = self.obj.implpart.surfaces[0].refpoints
        self.inplanemat = self.obj.implpart.surfaces[0].inplanemats
        self.edges = AUV.BuildEdgeDict(self.obj.implpart.surfaces[0])
        self.adjacencyidx = AUV.DetermineAdjacency(self.obj.implpart.surfaces[0], self.edges)
        self.surfaces = AUV.FindTexPatches(self.obj.implpart.surfaces[0], self.adjacencyidx)
        self.obj.implpart.surfaces[0].intrinsicparameterization = None
        self.boxes = self.obj.implpart.surfaces[0].boxes
        self.boxpolys = self.obj.implpart.surfaces[0].boxpolys
        self.boxcoords = self.obj.implpart.surfaces[0].boxcoords

    def find_close_geodesic(self, elements, point):
        geodets = None
        for element in elements:
            if element in self.georecord.keys():
                # [pointuv (bary), int_pnt (bary), point (3D), unitfiberdirection (3D), closest_point_idx (idx), uv_start, length, direction]
                geodesics = self.georecord[element][0]
                for g in range(0, len(geodesics)):
                    d2left = GEO.calcdistance(geodesics[g][7]*geodesics[g][3], geodesics[g][2], point)
                    if geodets is None or np.linalg.norm(d2left[0]) < np.linalg.norm(geodets[2][0]):
                        geodets = (geodesics[g], element, d2left)
        if geodets:
            return geodets
        else:
            raise IndexError("Cannot find a close geodesic")

    def determine_surface(self, initpoint, initdirection):
        # Determine if start point is on a vertex or not then perform the necessary calculation to find the next element
        if 0 in np.linalg.norm(self.vertices - initpoint, axis=1):
            # Current point is a vertex:
            element, newvector = GEO.find_element_vertex(initpoint, initdirection, self.surfacenormal,
                                                         self.vertices, self.vertexids, self.facetnormals)
        else:
            # Current point is not on a vertex but within a polygon:
            element, newvector, _ = GEO.find_element_within(initpoint, initdirection, self.surfacenormal,
                                                            self.vertices, self.vertexids, self.facetnormals,
                                                            self.inplanemat)

        surface = [(i, surface.index(element)) for i, surface in enumerate(self.surfaces) if element in surface][0][0]

        surface_polygons = self.surfaces[surface]
        self.surface_vertexids = self.vertexids[surface_polygons]

        # self.vertexids = self.surface_vertexids
        # self.vertices = self.vertices[np.unique(self.surface_vertexids)].reshape(-1, 3)

    def find_startpoints(self, initpoint, initdirection, normal, cfpoint, interpolate=False):
        """
        Finds the starting points for each geodesic
        Spawns geodesics in principle directions x, y, z, -x, -y, -z
        For any valid directions we drop start points in interval self.fiberint as the geodesic is traced
        """
        pointdirections = np.array([np.cross(initdirection, normal)])

        directions = [1, -1]
        for i in range(0, pointdirections.shape[0]):
            for direction in directions:
                pointdirection = direction * pointdirections[i]
                point = initpoint

                # Determine if start point is on a vertex or not then perform
                # the necessary calculation to find the next element for the seed geodesic
                if 0 in np.linalg.norm(self.vertices - point, axis=1):
                    # Current point is a vertex:
                    element, newvector = GEO.find_element_vertex(point, pointdirection, normal, self.vertices,
                                                                 self.vertexids, self.facetnormals)
                else:
                    # Current point is not on a vertex but within a polygon:
                    element, newvector, _ = GEO.find_element_within(point, pointdirection, normal, self.vertices,
                                                                    self.vertexids, self.facetnormals, self.inplanemat)

                if 0 in np.linalg.norm(self.vertices - point, axis=1):
                    # Current point is a vertex:
                    selement, _ = GEO.find_element_vertex(point, initdirection, normal, self.vertices, self.vertexids,
                                                          self.facetnormals)
                else:
                    # Current point is not on a vertex but within a polygon:
                    selement, _, _ = GEO.find_element_within(point, initdirection, normal, self.vertices, self.vertexids, self.facetnormals, self.inplanemat)

                if element is None or newvector is None:
                    # print("Point: %s , can't find element or newvector. Element: %s, Newvector: %s" % (point, element, newvector))
                    continue

                if not GEO.check_proj_inplane_pnt(point, self.vertices[self.vertexids[element]]):
                    # print("Point: %s , proj_inplane_pnt failed." % point)
                    continue

                vdirection = GEO.calcunitvector(np.cross(normal, initdirection))
                angle = GEO.angle_between_vectors(vdirection, pointdirection)

                if selement is not None:
                    # Rotate the given fiber vector to be in plane with the start element
                    try:
                        newfibervector = GEO.rot_vector(normal, self.facetnormals[selement], initdirection)
                    except GEO.EdgeError:
                        continue

                    self.startpoints = np.vstack((self.startpoints, point))
                    self.startuv = np.vstack((self.startuv, cfpoint))
                    self.startelements = np.append(self.startelements, selement)

                    if interpolate:
                        try:
                            closest_geodesic, _, _ = self.find_close_geodesic([element], point)
                            self.sfiberdirections = np.vstack((self.sfiberdirections, closest_geodesic[7] * closest_geodesic[3]))
                        except IndexError:
                            self.sfiberdirections = np.vstack((self.sfiberdirections, newfibervector))
                    else:
                        self.sfiberdirections = np.vstack((self.sfiberdirections, newfibervector))
                else:
                    continue

                pointdirection = GEO.rot_vector(normal, self.facetnormals[element], pointdirection)

                p = 1
                while True:
                    dleft = np.abs(self.fiberint / np.cos(angle))
                    try:
                        while True:
                            int_pnt_3d, nextunitvector, nextelement, _ = GEO.traverse_element(self, element, point, pointdirection, None, None, None, direction=direction, parameterization=False)

                            d2int = np.linalg.norm(int_pnt_3d - point)

                            dleft = dleft - d2int

                            if dleft <= 0:
                                point = int_pnt_3d + pointdirection * dleft

                                selementvec = GEO.rot_vector(self.facetnormals[self.startelements[-1]],
                                                             self.facetnormals[element], self.sfiberdirections[-1],
                                                             force=True)

                                self.startpoints = np.vstack((self.startpoints, point))
                                self.startelements = np.append(self.startelements, element)

                                if interpolate:
                                    try:
                                        closest_geodesic, _, _ = self.find_close_geodesic([element], point)
                                        self.sfiberdirections = np.vstack((self.sfiberdirections, closest_geodesic[7] * closest_geodesic[3]))
                                    except IndexError:
                                        self.sfiberdirections = np.vstack((self.sfiberdirections, selementvec))
                                else:
                                    self.sfiberdirections = np.vstack((self.sfiberdirections, selementvec))
                                break
                            else:
                                if nextelement is None:
                                    # print("Couldn't find next element.")
                                    raise GEO.EdgeError
                                point = int_pnt_3d
                                element = nextelement
                                pointdirection = nextunitvector
                    except GEO.EdgeError:
                        # print("End of geodesic detected. %s" % direction)
                        break
                    self.startuv = np.vstack((self.startuv, np.array([cfpoint[0], np.sign(self.fiberint / np.cos(angle)) * self.fiberint * p + cfpoint[1]])))
                    p += 1
        if self.startpoints.shape[0] < 1:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import axes3d

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            ax.scatter(initpoint[0], initpoint[1], initpoint[2])
            ax.quiver(initpoint[0], initpoint[1], initpoint[2], initdirection[0], initdirection[1], initdirection[2])
            plt.show()
            raise IndexError("No geodesic start points found.")

    def calc_geodesics(self, startidx):
        """
        Determines the approximate path of a geodesic along a fiber direction
        :param fiberdirection: Fiber orientation in degrees
        :param iters: Number of elements to iterate through
        :param obj: x3d model object
        :param nearby: Should we assign close points?
        :return: An array of fiber points relative to an approximate geodesic of the surface
        """

        print("Number of geodesics: %i" % (self.startpoints.shape[0] - startidx))
        start_time = time.time()
        for i in range(startidx, self.startpoints.shape[0]):
            print ".",
            self.calc_geodesic(self.startpoints[i], self.startelements[i], self.sfiberdirections[i],
                               self.startuv[i], self.fiberpoints_local, direction=1, parameterization=True)
            self.calc_geodesic(self.startpoints[i], self.startelements[i], self.sfiberdirections[i],
                               self.startuv[i], self.fiberpoints_local, direction=-1, parameterization=True)

        stop_time = time.time()
        elapsed = stop_time - start_time
        print("\r\nTime to calculate geodesics: %f seconds" % elapsed)

    def calc_geodesic(self, point, element, unitfiberdirection, uv_start, fiberpoints_local, direction=1, parameterization=False, save_ints=True):
        int_pnt_3d = point
        unitfiberdirection = direction * unitfiberdirection

        # Create an empty array of intersection points to visualize geodesics
        int_pnts = np.array([point])

        length = 0.0
        p = 0
        while True:
            try:
                int_pnt_3d, nextunitvector, nextelement, fiberpoints_local = GEO.traverse_element(self, element, point,
                                                                                                  unitfiberdirection,
                                                                                                  fiberpoints_local,
                                                                                                  length, uv_start,
                                                                                                  parameterization=parameterization,
                                                                                                  direction=direction)
            except GEO.EdgeError:
                break

            # Update and store the calculated fiber points and the intersection points
            int_pnts = np.vstack((int_pnts, int_pnt_3d))

            # Calculate the new length of the fiber
            length = direction * np.linalg.norm(int_pnt_3d - point) + length

            if nextelement is None:
                break

            # Update previous iteration values with the next iteration values
            point = int_pnt_3d
            unitfiberdirection = nextunitvector
            element = nextelement
            p += 1

        if save_ints:
            self.geoints.append(int_pnts)
        return length, int_pnt_3d, element

    def interpolate_geodesic(self, point, element, minassigned):
        fiberdirection, cfpoint, shared_cg1, shared_cg2, v = None, None, None, None, None

        neighbors = GEO.find_neighbors(element, self.vertexids_indices, self.adjacencyidx)
        neighbors = np.intersect1d(neighbors, self.georecord.keys())

        if neighbors.shape[0] > minassigned:
            for neighbor in neighbors:
                sharedvertex = self.vertices[np.intersect1d(self.vertexids[neighbor], self.vertexids[element])][1]
                shared_cg, _, distance = self.find_close_geodesic([neighbor], sharedvertex)
                check_dir = np.cross(shared_cg[7] * shared_cg[3], self.facetnormals[element])
                distance1, int_pnt_3d1, element1 = self.calc_geodesic(point, element, check_dir, None, None,
                                                                      parameterization=False)
                distance2, int_pnt_3d2, element2 = self.calc_geodesic(point, element, check_dir, None, None, direction=-1,
                                                                      parameterization=False)

                ulist = []
                flist = []
                dlist = []
                try:
                    shared_cg1, _, d1 = self.find_close_geodesic([element1], int_pnt_3d1)
                    u1 = d1[1] + shared_cg1[6] + shared_cg1[5][0]
                    f1 = shared_cg1[7] * shared_cg1[3]
                    ulist.append(u1)
                    flist.append(f1)
                    dlist.append(distance1)
                except IndexError:
                    pass

                try:
                    shared_cg2, _, d2 = self.find_close_geodesic([element2], int_pnt_3d2)
                    u2 = d2[1] + shared_cg2[6] + shared_cg2[5][0]
                    f2 = shared_cg2[7] * shared_cg2[3]
                    v2 = shared_cg2[5][1]
                    ulist.append(u2)
                    flist.append(f2)
                    dlist.append(distance2)
                except IndexError:
                    v2 = shared_cg1[5][1]
                    pass

                if shared_cg1 is not None and shared_cg2 is not None:
                    num_v = int(np.abs(shared_cg1[5][1] - shared_cg2[5][1]) / self.fiberint)
                    if num_v > 1:
                        v = dlist[-1]
                    else:
                        v = sum(dlist) / len(dlist)
                elif shared_cg1 is not None or shared_cg2 is not None:
                    v = dlist[-1]

                cfpoint = np.array([sum(ulist) / len(ulist), v + v2])

                fiberdirection = sum(flist) / len(flist)
                fiberdirection = GEO.proj_vector(fiberdirection, self.facetnormals[element])
        return fiberdirection, cfpoint

    def fill_missing_geodesics(self, minassigned):
        print("Filling missing elements")
        loc = self.startpoints.shape[0]
        # self.fiberint = self.fiberint

        for i in range(0, self.vertexids.shape[0]):
            centroid = self.vertices[self.vertexids[i]].sum(axis=0) / 3
            try:
                num_geos = len(self.georecord[i][0])
            except KeyError:
                # print("No geodesics found in element %s." % i)
                fiberdirection, cfpoint = self.interpolate_geodesic(centroid, i, minassigned)
                if fiberdirection is not None and cfpoint is not None:
                    self.find_startpoints(centroid, fiberdirection, self.facetnormals[i], cfpoint)
                    self.calc_geodesics(loc)
            loc = self.startpoints.shape[0]

    def fill_low_density_geodesics(self, minassigned):
        print("Filling low density elements")
        loc = self.startpoints.shape[0]
        # self.fiberint = self.fiberint

        for i in range(0, self.vertexids.shape[0]):
            centroid = self.vertices[self.vertexids[i]].sum(axis=0) / 3
            try:
                num_geos = len(self.georecord[i][0])
            except KeyError:
                # print("Still can't find a geodesic in element %s." % i)
                continue
            area = 0.5 * np.linalg.det(self.vertices[self.vertexids[i]])
            geos2area = num_geos / area
            if geos2area < 400.0:
                fiberdirection, cfpoint = self.interpolate_geodesic(centroid, i, minassigned)
                if fiberdirection is not None and cfpoint is not None:
                    self.find_startpoints(centroid, fiberdirection, self.facetnormals[i], cfpoint)
                    self.calc_geodesics(loc)
            loc = self.startpoints.shape[0]

    def assign_vertices(self, cleanup):
        for tests in [1, 0]:
            try:
                mask = np.ones((self.geoparameterization.shape[0]), dtype=bool)
                mask[np.unique(self.surface_vertexids)] = False
                for i in range(0, self.vertices.shape[0]):
                    vidx = np.where(np.linalg.norm(self.vertices - self.vertices[i], axis=1) == np.min(np.linalg.norm(self.vertices - self.vertices[i], axis=1)))
                    neighbors = np.unique(np.where((self.vertexids == vidx))[0])

                    for neighbor in neighbors:
                        closest_geodesic, element, distance = self.find_close_geodesic([neighbor], self.vertices[i])

                        testval = np.dot(calcunitvector(np.cross(closest_geodesic[7]*closest_geodesic[3], distance[0])), self.facetnormals[element])[0]
                        fpoint_t = np.array([closest_geodesic[6] + distance[1] + closest_geodesic[5][0], testval*np.linalg.norm(distance[0]) + closest_geodesic[5][1]])
                        if np.isnan(fpoint_t[1]):
                            fpoint_t[1] = closest_geodesic[5][1]

                        fiberrec = np.copy(self.geoparameterization)
                        fiberrec[i] = fpoint_t

                        rel_uvw = np.pad(fiberrec[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1)
                        with np.errstate(invalid='ignore'):
                            vdir = 0.5 * np.linalg.det(rel_uvw)
                        vdir[np.isnan(vdir)] = 0
                        if (vdir < 0).any():
                            pass
                        else:
                            self.geoparameterization[i] = fpoint_t
                rel_uvw = np.pad(self.geoparameterization[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant",
                                 constant_values=1)
                vdir = 0.5 * np.linalg.det(rel_uvw)
                assert (vdir > 0).all()
                break
            except (AssertionError, IndexError):
                cleanup(tests)

    def create_parameterization(self):
        mask = np.ones((self.geoparameterization.shape[0]), dtype=bool)
        mask[np.unique(self.surface_vertexids)] = False

        self.assign_vertices(self.fill_missing_geodesics)

        self.interpolate(np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0], mask)

        self.average_fpoint(np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0], mask)

        if np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0].size > 0:
            self.assign_vertices(self.fill_low_density_geodesics)

        print("Missed vertices: %s" % np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0].size)

        rel_uvw = np.pad(self.geoparameterization[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1)
        vdir = 0.5 * np.linalg.det(rel_uvw)
        assert (vdir > 0).all()
        assert np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0].size == 0

    def interpolate(self, leftover_idxs, mask):
        timeout = 0
        while leftover_idxs.shape[0] > 0 and timeout < 50:
            for i in range(0, leftover_idxs.shape[0]):
                unassigned_facets = np.unique(np.where((self.vertexids == leftover_idxs[i]))[0])
                done = False
                for j in unassigned_facets:
                    elementvertices = self.vertices[self.vertexids[j]]
                    assigned_vertsids = np.where((~np.isnan(self.geoparameterization[self.vertexids[j]]).all(axis=1)))[0]
                    if assigned_vertsids.shape[0] > 0:
                        if not np.isnan(self.fiberdirections[j]).any():
                            fiberdirection = self.fiberdirections[j]
                            for k in assigned_vertsids:
                                assigned_fpoint = self.geoparameterization[self.vertexids[j]][k]
                                fdistance, closest_point = GEO.calcclosestpoint(fiberdirection, elementvertices[k],
                                                                                np.array([self.vertices[leftover_idxs[i]]]),
                                                                                self.facetnormals[j])
                                unassigned_fpoint = assigned_fpoint + fdistance

                                fiberrec = np.copy(self.geoparameterization)
                                fiberrec[leftover_idxs[i]] = unassigned_fpoint
                                rel_uvw = np.pad(fiberrec[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1).transpose(0, 2, 1)
                                vdir = 0.5 * np.linalg.det(rel_uvw)
                                if (vdir < 0).any():
                                    pass
                                else:
                                    self.geoparameterization[leftover_idxs[i]] = unassigned_fpoint
                                    done = True
                                    break
                    if done:
                        break
            leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
            timeout += 1
        return leftover_idxs

    def average_fpoint(self, leftover_idxs, mask):
        for i in range(0, leftover_idxs.shape[0]):
            neighbors = np.unique(np.where((self.vertexids == leftover_idxs[i]))[0])
            neighbor_fpoint = self.geoparameterization[self.vertexids[neighbors]]
            neighbor_fpoint = neighbor_fpoint.reshape(neighbor_fpoint.shape[0]*neighbor_fpoint.shape[1], 2)

            count = ~np.isnan(neighbor_fpoint).all(axis=1)

            average_fpoint = np.sum(neighbor_fpoint[count], axis=0)/neighbor_fpoint[count].shape[0]

            fiberrec = np.copy(self.geoparameterization)
            fiberrec[leftover_idxs[i]] = average_fpoint
            rel_uvw = np.pad(fiberrec[self.vertexids], [(0, 0), (0, 0), (0, 1)], "constant", constant_values=1)
            vdir = 0.5 * np.linalg.det(rel_uvw)

            if (np.sign(vdir) < 0).any():
                pass
            else:
                self.geoparameterization[leftover_idxs[i]] = average_fpoint

            del fiberrec
        leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]
        return leftover_idxs

    def layup(self, angle, orienation_locations=None, precision=1e-4, maxsteps=10000, lr=1e-3, decay=0.7, eps=1e-8, mu=0.8, plotting=False, save=False):
        orientations = None
        rmatrix = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
                            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]])
        parameterization = np.matmul(self.geoparameterization, rmatrix)

        optimizedparamterization, loss = self.fiberoptimize(parameterization, precision=precision, maxsteps=maxsteps, lr=lr, decay=decay, eps=eps, mu=mu)
        texcoords2inplane = self.calctransform(optimizedparamterization)

        if orienation_locations is not None:
            orientations = self.calcorientations_abaqus(orienation_locations, self.vertices, self.vertexids, self.inplanemat,
                                                        texcoords2inplane, self.obj.implpart.surfaces[0].boxes,
                                                        self.obj.implpart.surfaces[0].boxpolys,
                                                        self.obj.implpart.surfaces[0].boxcoords)

        if plotting:
            if orientations is None:
                orientations = self.calcorientations_abaqus(self.centroids, self.vertices, self.vertexids, self.inplanemat,
                                                            texcoords2inplane, self.obj.implpart.surfaces[0].boxes,
                                                            self.obj.implpart.surfaces[0].boxpolys,
                                                            self.obj.implpart.surfaces[0].boxcoords)

            if save:
                np.save("orientation_%s.npy" % angle, orientations)

            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import axes3d

            self.plot_geodesics()

            fig = plt.figure()
            plt.plot(range(len(loss)), loss)

            fig = plt.figure()
            plt.scatter(parameterization[:, 0], parameterization[:, 1])
            plt.scatter(optimizedparamterization[:, 0], optimizedparamterization[:, 1])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2])
            ax.quiver(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], orientations[:, 0], orientations[:, 1], orientations[:, 2], length=0.1)
            # ax.quiver(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2], self.facetnormals[:, 0], self.facetnormals[:, 1], self.facetnormals[:, 2], length=0.1)
            plt.show()

        return texcoords2inplane

    def fiberoptimize(self, seed, precision=None, maxsteps=None, lr=None, decay=None, eps=None, mu=None):
        def f(x, *args):
            return OP.computeglobalstrain(self.normalized_2d, x, self.vertexids, self.stiffness_tensor)

        def gradf(x, *args):
            oc = np.argmin(np.linalg.norm(self.geoparameterization, axis=1))
            return OP.computeglobalstrain_grad(self.normalized_2d, x, self.vertexids, self.stiffness_tensor, oc)

        print("Optimizing...")
        initenergy = OP.computeglobalstrain(self.normalized_2d, seed.flatten(), self.vertexids, self.stiffness_tensor)
        print("Initial strain energy: %s J/m" % initenergy)

        start_time = time.time()

        print("Optimizing with rmsprop...")
        optimizedparameterization, loss = OP.rmsprop_momentum(f, gradf, seed, precision=precision, maxsteps=maxsteps, lr=lr, decay=decay, eps=eps, mu=mu)

        stop_time = time.time()
        elapsed = stop_time - start_time
        print("Time to optimize: %f seconds" % elapsed)

        return optimizedparameterization, loss

    def calctransform(self, parameterization):
        self.obj.implpart.surfaces[0].intrinsicparameterization = polygonalsurface_texcoordparameterization.new(self.obj.implpart.surfaces[0], parameterization, self.obj.implpart.surfaces[0].vertexidx, None)
        self.obj.implpart.surfaces[0].intrinsicparameterization.buildprojinfo(self.obj.implpart.surfaces[0])

        texcoords2inplane = self.obj.implpart.surfaces[0].intrinsicparameterization.texcoords2inplane

        return texcoords2inplane

    def plot_geodesics(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import axes3d

        mask = np.ones((self.geoparameterization.shape[0]), dtype=bool)
        mask[np.unique(self.surface_vertexids)] = False
        leftover_idxs = np.where((np.isnan(self.geoparameterization).all(axis=1) & np.array(~mask)))[0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.initpoint[0], self.initpoint[1], self.initpoint[2])
        ax.scatter(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], alpha=0.1)
        ax.scatter(self.startpoints[:, 0], self.startpoints[:, 1], self.startpoints[:, 2])
        # ax.scatter(self.vertices[leftover_idxs][:, 0], self.vertices[leftover_idxs][:, 1], self.vertices[leftover_idxs][:, 2])
        # ax.quiver(self.startpoints[:, 0], self.startpoints[:, 1], self.startpoints[:, 2], self.sfiberdirections[:, 0], self.sfiberdirections[:, 1], self.sfiberdirections[:, 2])
        for i in self.geoints:
            ax.plot(i[:, 0], i[:, 1], i[:, 2])

        fig = plt.figure()
        plt.scatter(self.geoparameterization[:, 0], self.geoparameterization[:, 1])
        plt.scatter(self.startuv[:, 0], self.startuv[:, 1])

    def calcorientations_abaqus(self, modellocs, vertices, vertexids, inplanemat, texcoords2inplane, boxes, boxpolys, boxcoords):
        orientations = np.zeros((modellocs.shape[0], 3))
        for i in range(0, modellocs.shape[0]):
            vert = modellocs[i]
            element = None

            rangex = np.array([boxcoords[:, 0], boxcoords[:, 3]]).T
            rangey = np.array([boxcoords[:, 1], boxcoords[:, 4]]).T
            rangez = np.array([boxcoords[:, 2], boxcoords[:, 5]]).T

            containers = np.where(np.logical_and(np.logical_and(
                np.logical_and(vert[0] > rangex[:, 0], vert[0] < rangex[:, 1]),
                np.logical_and(vert[1] > rangey[:, 0], vert[1] < rangey[:, 1])),
                np.logical_and(vert[2] > rangez[:, 0], vert[2] < rangez[:, 1])))[0]

            box_idx = boxes[containers][:, -1][boxes[containers][:, -1] != -1][0]

            polys = np.array([boxpolys[box_idx]])
            count = 1
            while True:
                if boxpolys[box_idx+count] == -1:
                    break
                polys = np.append(polys, boxpolys[box_idx+count])
                count += 1

            for j in polys:
                if self.point_in_polygon_3d(vertices[vertexids][j], vert, inplanemat[j]):
                    element = j
                    continue
            if element is not None:
                red_texcoords2inplane = texcoords2inplane[element][:2, :2]
                texutexbasis = np.array([1.0, 0.0])
                texu2dbasis = np.dot(red_texcoords2inplane, texutexbasis)
                u3D = np.dot(inplanemat[element].T, texu2dbasis)
                orientations[i] = calcunitvector(u3D)
            else:
                print("Failed to find point on surface: %s" % vert)
        return orientations

    def point_in_polygon_2d(self, vertices_rel_point):
        import sys
        # Apply winding number algorithm.
        # This algorithm is selected -- in its most simple form --
        # because it is so  simple and robust in the case of the
        # intersect point being on or near the edge. It may well
        # be much slower than optimal. It tries to return True
        # in the edge case.

        # Should probably implement a faster algorithm then drop
        # down to this for the special cases.

        # See Hormann and Agathos, The point in polygon problem
        # for arbitrary polygons, Computational Geometry 20(3) 131-144 (2001)
        # http://dx.doi.org/10.1016/S0925-7721(01)00012-8
        # https://pdfs.semanticscholar.org/e90b/d8865ddb7c7af2b159d413115050d8e5d297.pdf

        # Winding number is sum over segments of
        # acos((point_to_vertex1 dot point_to_vertex2)/(magn(point_to_vertex1)*magn(point_to_vertex_2))) * sign(det([ point_to_vertex1  point_to_vertex2 ]))
        # where sign(det) is really: What is the sign of the z
        # component of (point_to_vertex1 cross point_to_vertex2)

        # Special cases: magn(point_to_vertex1)==0 or
        #  magn_point_to_vertex2   -> point is on edge
        # det([ point_to_vertex1  point_to_vertex2 ]) = 0 -> point may be on edge

        windingnum = 0.0
        numvertices = vertices_rel_point.shape[0]

        for VertexCnt in range(numvertices):
            NextVertex = VertexCnt + 1
            if NextVertex == numvertices:
                # final vertex... loop back to the start
                NextVertex = 0
                pass

            # calculate (thisvertex - ourpoint) -> vec1
            vec1 = vertices_rel_point[VertexCnt, :]
            magn1 = np.linalg.norm(vec1)

            # calculate (nextvertex - ourpoint) -> vec2
            vec2 = vertices_rel_point[NextVertex, :]
            magn2 = np.linalg.norm(vec2)

            if magn1 == 0.0 or magn2 == 0.0:
                # Got it!!!
                return True

            vec1 = vec1 / magn1
            vec2 = vec2 / magn2

            det = vec1[0] * vec2[1] - vec2[0] * vec1[1]  # matrix determinant

            cosparam = (vec1[0] * vec2[0] + vec1[1] * vec2[1])  # /(magn1*magn2);

            if cosparam < -1.0:
                # Shouldn't be possible...just in case of weird roundoff
                cosparam = -1.0

            if cosparam > 1.0:
                # Shouldn't be possible...just in case of weird roundoff
                cosparam = 1.0

            if det > 0:
                windingnum += np.arccos(cosparam)
            elif det < 0:
                windingnum -= np.arccos(cosparam)
            else:
                # det==0.0
                # Vectors parallel or anti-parallel

                if cosparam > 0.9:
                    # Vectors parallel. We are OUTSIDE. Do Nothing
                    pass
                elif cosparam < -0.9:
                    # Vectors anti-parallel. We are ON EDGE */
                    return True
                else:
                    assert 0  # Should only be able to get cosparam = +/- 1.0 if abs(det) > 0.0 */
                    pass
                pass
            pass

        windingnum = abs(windingnum) * (
                    1.0 / (2.0 * np.pi))  # divide out radians to number of winds; don't care about clockwise vs. ccw
        if windingnum > .999 and windingnum < 1.001:
            # Almost exactly one loop... got it!
            return True
        elif windingnum >= .001:
            #
            sys.stderr.write(
                "spatialnde.geometry.point_in_polygon_2d() Got weird winding number of %e; assuming inaccurate calculation on polygon edge\n" % (
                    windingnum))
            # Could also be self intersecting polygon
            # got it !!!
            return True

        # If we got this far, the search failed
        return False

    def point_in_polygon_3d(self, vertices, point, inplanemat):
        """ assumes vertices are coplanar, with given orthonormal 2D basis inplanemat.  """
        vert3d_rel_point = vertices-point[np.newaxis, :]
        vert2d_rel_point = np.inner(vert3d_rel_point, inplanemat)

        return self.point_in_polygon_2d(vert2d_rel_point)
