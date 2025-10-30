import numpy as np
import pymeshlab
from blender_tissue_cartography import interface_pymeshlab as intmsl
from blender_tissue_cartography import mesh as tcmesh
from blender_tissue_cartography.mesh import ObjMesh


def mesh_area_density(mesh: ObjMesh, points: np.ndarray) -> np.ndarray:
    """
    Calculate area density of mesh vertices, given a mesh and 3d locations of the vertices.

    :param mesh: mesh containing n vertices
    :param points: n x ndim array of vertex locations
    :return: areas_normed: n array of area densities, normalized to sum to n
    """

    faces = np.array(mesh.faces)
    areas = np.zeros(len(points))

    for shift in range(3):
        order = np.roll(faces, shift, axis=-1)

        ptx = points[order, :]

        ab = ptx[:, 1] - ptx[:, 0]
        ac = ptx[:, 2] - ptx[:, 0]

        area = np.linalg.norm(np.cross(ab, ac), axis=-1) / 2

        areas[order[:, 0]] += area / 3

    areas_normed = len(areas) * areas / (np.sum(areas))

    return areas_normed


def mesh_from_points(points) -> ObjMesh:
    point_cloud = tcmesh.ObjMesh(vertices=points, faces=[])
    point_cloud_pymeshlab = intmsl.convert_to_pymeshlab(point_cloud)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(point_cloud_pymeshlab)

    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.apply_coord_hc_laplacian_smoothing()
    ms.meshing_close_holes()

    mesh_reconstructed = intmsl.convert_from_pymeshlab(ms.current_mesh())

    return mesh_reconstructed


def smoothed_mesh_from_points(points, targetlen=1) -> ObjMesh:
    point_cloud = tcmesh.ObjMesh(vertices=points, faces=[])
    point_cloud_pymeshlab = intmsl.convert_to_pymeshlab(point_cloud)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(point_cloud_pymeshlab)

    ms.compute_normal_for_point_clouds(k=20, smoothiter=2)
    ms.generate_surface_reconstruction_screened_poisson(depth=8, fulldepth=5, )

    ms.meshing_isotropic_explicit_remeshing(iterations=10, targetlen=pymeshlab.PercentageValue(targetlen))

    return intmsl.convert_from_pymeshlab(ms.current_mesh())


def calculate_surface_area_along_axis(mesh: ObjMesh, dividers, axis=1):
    """

    Parameters
    ----------
    mesh
        mesh containing vertices and faces
    dividers
        list of positions along the axis to divide the mesh
    axis
        axis along which to divide the mesh (in point coordinates)

    Returns
    -------
        binned surface areas between dividers

    """

    surface_areas = [0., ]

    for div in dividers:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces), "embryo")
        ms.generate_polyline_from_planar_section(planeaxis=axis, planeoffset=div, splitsurfacewithsection=True)
        surface_areas.append(ms.get_geometric_measures()["surface_area"])

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces), "embryo")
    surface_areas.append(ms.get_geometric_measures()["surface_area"])

    return np.diff(surface_areas)
