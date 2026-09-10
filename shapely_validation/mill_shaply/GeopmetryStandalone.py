"""This file contains all the geometric entities that will is needed throughout the iBrus
project."""
import numpy as np
from typing import *
from numpy import array_split
from shapely import measurement
import copy
from typing import Mapping, Callable

# from shapely.geometry import get_type_id
# import shapely
# from shapely.ops import unary_union, split
import shapely
import dill


# helpers copied from simulation toolbox helpers
np.random.seed(0)
r_nrs = np.random.rand(100)
default_precision = 1e-10 #1e-10
wp_slice_precision = 1e-3 #1e-3


def compare_np_arrays(array_1: np.ndarray, array_2: np.ndarray, precision: float):
    number_of_decimals_in_precision = -int(np.log10(precision))
    np.testing.assert_array_almost_equal(
        array_1, array_2, number_of_decimals_in_precision)


def flatten_list(nested_list: list):
    '''A very useful helper function to flatten out arbitrarily nested python list.
    To get the list from the result, do:  list(flatten_list(nested_list))
    '''
    for element in nested_list:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten_list(element)
        else:
            yield element


def remove_objects_from_list(list, object_removal_bool_list):
    list[:] = [element for idx, element in enumerate(
        list) if not object_removal_bool_list[idx]]


class OptionalAttributeNotCalculatedYet(Exception):
    pass


class InputAreNotTheSameDataType(Exception):
    pass


class ShapelyErrorWillOccur(Exception):
    pass








class Vector:
    """Vector is the most basic geometry class which has all the basic operations needed for
    vector calculation and manipulation that is used in iBrus.
    """
    __value: np.ndarray

    def __init__(self, x: float, y: float, z: float):
        self.value = np.array([x, y, z, 1])

    @classmethod
    def from_ndArray(cls, array: np.ndarray):
        return cls(array[0], array[1], array[2])

    @classmethod
    def from_xz_nparrays(cls, arrays: np.ndarray):
        return [cls(array[0], 0, array[1]) for array in arrays]

    @classmethod
    def e_x(cls):
        return cls(1, 0, 0)

    @classmethod
    def e_y(cls):
        return cls(0, 1, 0)

    @classmethod
    def e_z(cls):
        return cls(0, 0, 1)

    @classmethod
    def origin(cls):
        return cls(0, 0, 0)

    def x(self) -> float:
        return self.value[0]

    def y(self) -> float:
        return self.value[1]

    def z(self) -> float:
        return self.value[2]

    def xyz_in_array(self):
        return np.array([self.value[0], self.value[1], self.value[2]])

    def scale(self, factor: float):
        return Vector(self.value[0] * factor, self.value[1] * factor, self.value[2] * factor)

    def norm(self) -> float:
        return np.linalg.norm(self.value[0:3])

    def normalize(self):
        norm = self.norm()
        if norm <= 1e-20:
            return Vector.origin()
        else:
            s = 1 / norm
            return Vector(self.value[0] * s, self.value[1] * s, self.value[2] * s)

    def add(self, other):
        if isinstance(other, Vector):
            return Vector(self.value[0] + other.value[0], self.value[1] +
                          other.value[1], self.value[2] + other.value[2])
        else:
            raise InputAreNotTheSameDataType

    def subtract(self, other):
        if isinstance(other, Vector):
            return Vector(self.value[0] - other.value[0], self.value[1] -
                          other.value[1], self.value[2] - other.value[2])
        else:
            raise InputAreNotTheSameDataType

    def difference(self, other):
        if isinstance(other, Vector):
            return Vector(self.value[0] - other.value[0], self.value[1] -
                          other.value[1], self.value[2] - other.value[2])
        else:
            raise InputAreNotTheSameDataType

    def dot(self, other) -> float:
        if isinstance(other, Vector):
            return self.x() * other.x() + self.y() * other.y() + self.z() * other.z()
        else:
            raise InputAreNotTheSameDataType

    def cross(self, other):
        if isinstance(other, Vector):
            return Vector((self.y() * other.z() - self.z() * other.y()),
                          (self.z() * other.x() - self.x() * other.z()),
                          (self.x() * other.y() - self.y() * other.x()))

        else:
            raise InputAreNotTheSameDataType

    def distance(self, other):
        if isinstance(other, Vector):
            return self.difference(other).norm()
        else:
            raise InputAreNotTheSameDataType

    def transform(self, transform):
        if isinstance(transform, Transform):
            value = np.matmul(transform.value, self.value)
            return Vector(value[0], value[1], value[2])
        else:
            raise Exception('input is not of type Transform')

    def transform_list(list_of_vectors, transform):
        points = [point.transform(transform) for point in list_of_vectors]
        return points

    def change_frame_point(self, down_chain, up_chain):
        frame_change_transform = \
            Transform.get_frame_change_transform(
                down_chain, up_chain)
        return self.transform(frame_change_transform)

    def change_frame_vector(self, down_chain, up_chain):
        return self.change_frame_point(down_chain, up_chain).\
            subtract(Vector.origin().change_frame_point(down_chain, up_chain))

    def change_frame_vector_list(list_of_vectors, down_chain, up_chain):
        frame_change_transform = Transform.get_frame_change_transform(
            down_chain, up_chain)
        end_points = [point.transform(frame_change_transform)
                      for point in list_of_vectors]
        start_point = Vector.origin().transform(frame_change_transform)
        transformed_vectors = [end_point.subtract(
            start_point) for end_point in end_points]
        return transformed_vectors

    # TODO vector list operations add operation to list class

    def change_frame_point_list(list_of_vectors, down_chain, up_chain):
        # TODO instance check does not work on List?
        frame_change_transform = Transform.get_frame_change_transform(
            down_chain, up_chain)
        points = [point.transform(frame_change_transform)
                  for point in list_of_vectors]
        return points

    def check_coplanarity(list_of_vectors):

        if len(list_of_vectors) <= 3:
            return

        plane_vector_1 = list_of_vectors[1].subtract(list_of_vectors[0])
        plane_vector_2 = list_of_vectors[2].subtract(list_of_vectors[0])
        plane_normal = plane_vector_1.cross(plane_vector_2)

        for i in range(len(list_of_vectors) - 3):
            test_vector = list_of_vectors[i + 3].subtract(list_of_vectors[0])
            if np.abs(test_vector.dot(plane_normal)) > 1e-6:
                raise Exception('List of vectors is not coplanar')

    def compare(self, vector, precision: float):
        if isinstance(vector, Vector):
            compare_np_arrays(self.value, vector.value, precision)
        else:
            raise InputAreNotTheSameDataType


class Transform:
    """Transform is wrapper class for geometric transformation, including rotation, translation,
    and frame change. It's represented as homogenous transform in a 4x4 matrix."""
    value: np.ndarray

    def __init__(self, matrix: np.ndarray):
        self.value = matrix

    def __repr__(self):
        return "<Transform value:%s>" % (self.value)

    def __str__(self):
        return "%s" % (self.value)

    @ classmethod
    def identity(cls):
        matrix = np.identity(4)
        return cls(matrix)

    @ classmethod
    def from_translation(cls, translation: Vector):
        matrix = np.identity(4)
        matrix[:, 3] = translation.value
        return cls(matrix)

    @ classmethod
    def from_axis_angle(cls, axis: Vector, angle: float):

        axis_normalized = axis.normalize()
        u_1: float = axis_normalized.x()
        u_2: float = axis_normalized.y()
        u_3: float = axis_normalized.z()
        c_2 = np.cos(angle / 2.0)
        s = np.sin(angle)
        s_2 = np.sin(angle / 2.0)

        R_11 = c_2 ** 2 + s_2 ** 2 * (2 * u_1 ** 2 - 1)
        R_22 = c_2 ** 2 + s_2 ** 2 * (2 * u_2 ** 2 - 1)
        R_33 = c_2 ** 2 + s_2 ** 2 * (2 * u_3 ** 2 - 1)
        R_12 = 2 * u_1 * u_2 * s_2 ** 2 - u_3 * s
        R_13 = 2 * u_1 * u_3 * s_2 ** 2 + u_2 * s
        R_21 = 2 * u_2 * u_1 * s_2 ** 2 + u_3 * s
        R_23 = 2 * u_2 * u_3 * s_2 ** 2 - u_1 * s
        R_31 = 2 * u_3 * u_1 * s_2 ** 2 - u_2 * s
        R_32 = 2 * u_3 * u_2 * s_2 ** 2 + u_1 * s

        rotation_matrix = np.array([[R_11, R_12, R_13, 0],
                                    [R_21, R_22, R_23, 0],
                                    [R_31, R_32, R_33, 0],
                                    [0, 0, 0, 1]])

        return cls(rotation_matrix)

    @classmethod
    def from_projection_along_vector(cls, v: Vector):
        v_unit = v.normalize()
        identity_matrix = np.identity(4)
        projection_transform = np.array([[v_unit.x()*v_unit.x(), v_unit.y() *
                                          v_unit.x(), v_unit.z()*v_unit.x(), 0],
                                         [v_unit.x()*v_unit.y(), v_unit.y() *
                                          v_unit.y(), v_unit.z()*v_unit.y(), 0],
                                         [v_unit.x()*v_unit.z(), v_unit.y() *
                                          v_unit.z(), v_unit.z()*v_unit.z(), 0],
                                         [0, 0, 0, 1]])
        rejection_transform = identity_matrix - projection_transform
        return cls(rejection_transform)

    @classmethod
    def from_projection_along_vector_to_x_z_plane(cls, v: Vector):
        if np.abs((v.y() - 0)) >= 1e-12:
            matrix = np.identity(4)
            matrix[0, 1] = -v.x() / v.y()
            matrix[1, 1] = 0
            matrix[2, 1] = -v.z() / v.y()
            return cls(matrix)
        else:
            raise Exception(
                "vector  is almost parallel to x_z plane ")

    @ staticmethod
    def from_rotation_first_then_translation(axis: Vector, angle: float, translation: Vector):
        rotation_transform = Transform.from_axis_angle(axis, angle)
        translation_transform = Transform.from_translation(translation)
        return rotation_transform.concatenate(translation_transform)

    @ staticmethod
    def from_translation_first_then_rotation(translation: Vector, axis: Vector, angle: float):
        translation_transform = Transform.from_translation(translation)
        rotation_transform = Transform.from_axis_angle(axis, angle)
        return translation_transform.concatenate(rotation_transform)

    def scale(self, scale: float):
        scaled_matrix = self.value * scale
        return Transform(scaled_matrix)

    def get_rotation(self):
        transformation_matrix = self.value
        # delete translation
        transformation_matrix[:, 3] = np.array([0, 0, 0, 1])
        return Transform(transformation_matrix)

    def inverse_transpose(self):
        matrix = np.empty([4, 4])
        matrix[3, :] = [0, 0, 0, 1]
        matrix[0:3, 0:3] = np.transpose(self.value[0:3, 0:3])
        matrix[0:3, 3] = -np.matmul(matrix[0:3, 0:3], self.value[0:3, 3])

        return Transform(matrix)

    def get_translation(self):
        return Transform.from_translation(
            Vector.from_ndArray(self.value[:, 3]))

    def defines_an_orthonormal_right_handed_frame(self) -> bool:
        # check for righthanded-ness (determinant==1)
        rot_matrix = self.value[0:3, 0:3]
        AtA = np.matmul(np.transpose(rot_matrix), rot_matrix)
        if np.sum(AtA - np.identity(3)) > 1e-12:
            return False
        if np.abs(np.linalg.det(rot_matrix) - 1) > 1e-12:
            return False
        return True

    # @njit
    def inverse(self):
        # TODO make inversion with transpose of rotation part....
        # only do this for orthogonal matrices!!!
        return Transform(np.linalg.inv(self.value))

    def concatenate(self, transform2):
        # concatenates the transformation with transform
        # such that they are sucessively applied
        # TransformConcat*v -> Transform2*(Transform1*v)
        return Transform(np.matmul(transform2.value, self.value))

    def get_frame_change_transform(down_chain, up_chain):
        # istance check does not work yet
        # if isinstance(down_chain, List[Pose]) and isinstance(up_chain, List[Pose]):
        if len(down_chain) == 0 and len(up_chain) == 1:
            return up_chain[0].value.inverse_transpose()
        if len(down_chain) == 1 and len(up_chain) == 0:
            return down_chain[0].value

        if len(down_chain) == 1 and len(up_chain) == 1:
            return down_chain[0].value.concatenate(up_chain[0].value.inverse_transpose())

        frame_change_transform = Transform.identity()
        for k in range(len(down_chain)):
            frame_change_transform = frame_change_transform.concatenate(
                down_chain[k].value)
        for k in range(len(up_chain)):
            frame_change_transform = frame_change_transform.concatenate(
                up_chain[k].value.inverse_transpose())
        return frame_change_transform

    def compare(self, transform, precision: float):
        if isinstance(transform, Transform):
            # we compare the effect of the transformations and not the reperesentations of them
            ex_before_transformed_1 = Vector.e_x()
            ex_before_transformed_2 = Vector.e_x()
            ex_after_transformed_1 = ex_before_transformed_1.transform(
                transform)
            ex_after_transformed_2 = ex_before_transformed_2.transform(self)
            # Then it's comparing 2 vectors
            ex_after_transformed_1.compare(ex_after_transformed_2, precision)

            ey_before_transformed_1 = Vector.e_y()
            ey_before_transformed_2 = Vector.e_y()
            ey_after_transformed_1 = ey_before_transformed_1.transform(
                transform)
            ey_after_transformed_2 = ey_before_transformed_2.transform(self)
            ey_after_transformed_1.compare(ey_after_transformed_2, precision)

            ez_before_transformed_1 = Vector.e_z()
            ez_before_transformed_2 = Vector.e_z()
            ez_after_transformed_1 = ez_before_transformed_1.transform(
                transform)
            ez_after_transformed_2 = ez_before_transformed_2.transform(self)
            ez_after_transformed_1.compare(ez_after_transformed_2, precision)

        else:
            raise InputAreNotTheSameDataType


class Pose:
    """Pose is a basic geometry properties for all the geometric objects defined in iBrus.
    It is represented as the transform which transforms the basis vectors of
    world coordinate system to the basis vectors of the frame attached to the pose.
    Each pose is defined in a certain frame and has strict hierarchy.
    For example: A grain's pose is defined in tool frame, A tool's pose is defined in global frame.
    A FlatManifold's pose is defined in Workpiece frame,
    A workpiece's pose is defined in global frame.
    """
    value: Transform

    def __init__(self, transform: Transform):
        self.value = transform

    def __repr__(self):
        return "<Pose value:%s>" % (self.value.value)

    def __str__(self):
        return "%s" % (self.value.value)

    @ classmethod
    def from_rotation_first_then_translation(cls, axis: Vector, angle: float, translation: Vector):
        '''This method first apply rotation in the frame using rotational axis, where the Pose is defined. Then apply translation.
        This means the object will first be rotater. If the rotational center overlaps with the object center, then it's a self rotation.
        Then translation is applied. This means, the translation will not create a leverage for rotation.
        While in the from_translation_first_then_rotation method, the translation will act as a leverage.'''
        return cls(Transform.from_rotation_first_then_translation(
            axis, angle, translation))

    @ classmethod
    def from_translation_first_then_rotation(cls, translation: Vector, axis: Vector, angle: float, ):
        '''This method first apply translation in the frame where the Pose is defined. Then apply the rotation using rotational axis, which is also defined in the same frame.
        For example, to distribute several grains to different position on the premeter of tool circle,
        grain first needs to be translated along the radius,
        then rotated based on their destination.'''
        return cls(Transform.from_translation_first_then_rotation(translation, axis, angle))

    @ classmethod
    def identity(cls):
        return cls(Transform.identity())

    @ classmethod
    def from_rotation_axis_angle(cls, axis: Vector, angle: float):
        return cls(Transform.from_axis_angle(axis, angle))

    @ classmethod
    def from_translation(cls, translation: Vector):
        return cls(Transform.from_translation(translation))

    def transform(self, transform: Transform):
        # check if transform is a proper pose transform
        if transform.defines_an_orthonormal_right_handed_frame():
            return Pose(self.value.concatenate(transform))
        else:
            raise Exception('transform is no proper pose transform')

    def get_position(self) -> Vector:
        return Vector(self.value.value[0, 3], self.value.value[1, 3], self.value.value[2, 3])

    def compare(self, other, precision):
        if isinstance(other, Pose):
            return self.value.compare(other.value, precision)
        else:
            raise InputAreNotTheSameDataType


class PoseTrajectory:
    """PoseTrajectory describes a trajectory of poses in 3D space at discrete times. """
    times: List[float]
    poses: List[Pose]

    def __init__(self, times: List[float], poses: List[Pose]):
        self.poses = poses
        self.times = times

    def extend_trajectory(self, other) -> 'PoseTrajectory':
        '''Append the second PoseTrajectory to the first trajectory'''
        if isinstance(other, PoseTrajectory):
            # increment the index of the incoming PoseTrajectory.
            new_list_index = [element + self.times[-1]
                              for element in other.times]
            times = [*self.times, *new_list_index]
            poses = [*self.poses, *other.poses]
            return PoseTrajectory(times, poses)
        else:
            raise InputAreNotTheSameDataType

    @ classmethod
    def from_feedrate_and_rotation_speed(cls, start_point: Vector, axis: Vector,
                                         rotation_speed: float, feed: Vector,
                                         end_time: float, time_step_size):
        # rotation_speed in rads/s
        # feedrate feed in m/s
        # feed Direction = feed/¦feed¦
        # use pose.from_axis_angle_rotation in a loop and vary angle and translation accordingly
        feed_rate = feed.norm()
        feed_direction = feed.normalize()
        feed_incremet = feed_direction.scale(feed_rate * time_step_size)
        times = np.arange(0, end_time + time_step_size, time_step_size)

        rot_angle = 0
        current_origin = start_point
        current_pose = Pose(Transform.from_axis_angle(axis.cross(Vector.e_z()),
                                                      -np.arccos((axis.dot(Vector.e_z()) / axis.norm()))))
        current_pose = current_pose.transform(Transform.from_rotation_first_then_translation(
            axis, rot_angle, current_origin))
        poses = [current_pose]

        for t in times[1:]:
            rot_angle = t * rotation_speed
            current_origin = current_origin.add(feed_incremet)
            current_pose = Pose(Transform.from_axis_angle(axis.cross(Vector.e_z()),
                                                          -np.arccos((axis.dot(Vector.e_z()) / axis.norm()))))
            current_pose = current_pose.transform(Transform.from_rotation_first_then_translation(
                axis, rot_angle, current_origin))

            poses.append(current_pose)

        return cls(times, poses)

    def transform(self, transform: Transform):
        poses = [pose.transform(transform) for pose in self.poses]
        return PoseTrajectory(self.times, poses)

    def update_by_translation(self, first_updated_pose_index: int, last_updated_pose_index: int,  translation_vector: Vector):
        past_trajectory = self.slice(0, first_updated_pose_index)
        changed_future_trajectory = self.slice(
            first_updated_pose_index, last_updated_pose_index + 1)
        unchanged_future_trajectory = self.slice(last_updated_pose_index + 1)

        transform_for_translation = Transform.from_translation(
            translation_vector)
        translated_future_trajectory = changed_future_trajectory.transform(
            transform_for_translation)

        self.poses = past_trajectory.poses + \
            translated_future_trajectory.poses + unchanged_future_trajectory.poses

    def update_from_feed_and_rotation_speed(self, first_updated_pose_index: int, last_updated_pose_index: int, new_feed: Vector, new_rotation_speed: float, new_axis: Vector, time_step_size: float):
        past_trajectory = self.slice(0, first_updated_pose_index)
        unchanged_future_trajectory = self.slice(last_updated_pose_index)
        new_start_point = Vector.origin().change_frame_point(
            [past_trajectory.poses[-1]], [])
        new_end_time = (last_updated_pose_index -
                        first_updated_pose_index + 1)*time_step_size
        changed_future_trajectory = PoseTrajectory.from_feedrate_and_rotation_speed(new_start_point, new_axis, new_rotation_speed, new_feed,
                                                                                    new_end_time, time_step_size)

        transform_unchanged_trajectory_start = unchanged_future_trajectory.poses[0].value.inverse_transpose(
        ).concatenate(changed_future_trajectory.poses[-1].value)
        unchanged_future_trajectory = unchanged_future_trajectory.transform(
            transform_unchanged_trajectory_start)

        self.poses = past_trajectory.poses[:-1] + \
            changed_future_trajectory.poses[:] + \
            unchanged_future_trajectory.poses[1:]

    def slice(self, start: int = 0, end: int = None, step: int = 1):
        if end is None:
            end == len(self.poses)
        sliced_times = self.times[start:end:step]
        sliced_poses = self.poses[start:end:step]

        return PoseTrajectory(sliced_times, sliced_poses)

    def compare(self, other, precision: float):
        if isinstance(other, PoseTrajectory):
            # compare times

            if not len(self.times) == len(self.poses) == len(other.times) == len(other.poses):
                raise Exception('Compared objects do not have same size')

            index = 0
            for t in self.times:
                compare_np_arrays(t, other.times[index], precision)
                index = index + 1

            index = 0
            for pose in self.poses:
                pose.compare(other.poses[index], precision)
                index = index + 1

        else:
            raise InputAreNotTheSameDataType


class Line2D:
    """Line2D is a object discribing a 2d line that lives in x_z plane.
    It's a general line that can be curved or straight.
    """
    x_values: np.ndarray
    z_values: np.ndarray
    values: np.ndarray

    def __init__(self, x_values: np.ndarray, z_values: np.ndarray):
        self.x_values = x_values
        self.z_values = z_values
        self.values = np.vstack((x_values, z_values))

    def compare(self, other, precision):
        if isinstance(other, Line2D):
            compare_np_arrays(
                self.values, other.values, precision)
        else:
            raise InputAreNotTheSameDataType

    def to_area(self):
        line = self
        return Area2D(line)

    def get_points(self) -> List[Vector]:
        return [Vector(self.x_values[index], 0, self.z_values[index])
                for index in range(len(self.x_values))]


class Area2D:
    """Area2D defines an area in 2D using polygon from shapely.
    Polygon has non-zero area. Area2D like other 2D geometry, lives in x_z plane.
    """

    area: shapely.polygons

    def __init__(self, line: Line2D):
        # convert here the list of points to the format polygon needs
        line_c = line
        if len(line_c.x_values) == 0:
            area_polygon = shapely.polygons(None)
        else:
            area_polygon = shapely.polygons(shapely.linearrings(
                line_c.x_values.T, line_c.z_values.T))

        self.area = area_polygon
        self.check_and_make_valid()
        # self.check_and_get_largest_polygon()

    @ classmethod
    def from_np_arrays(cls, x_values: np.ndarray, y_values: np.ndarray):
        if len(x_values) == len(y_values):
            line = Line2D(x_values, y_values)
            return cls(line)
        else:
            raise Exception(
                "The input x_values and y_values should have the same length.")

    @ staticmethod
    def from_shapely_polygon(area: shapely.GeometryType.POLYGON):
        area_2d = Area2D.empty()
        area_2d.area = area
        area_2d.check_and_make_valid()
        # area_2d.check_and_get_largest_polygon()
        return area_2d

    @ classmethod
    def empty(cls):
        line = Line2D(np.array([]), np.array([]))
        return cls(line)

    def union(self, other):
        if isinstance(other, Area2D):
            union = shapely.set_operations.union(
                self.area, other.area)
            return Area2D.from_shapely_polygon(union)
        else:
            raise InputAreNotTheSameDataType

    def intersect_old(self, other):
        # returns intersected area
        if isinstance(other, Area2D):
            intersection = shapely.set_operations.intersection(
                self.area, other.area, grid_size=default_precision)
            return Area2D.from_shapely_polygon(intersection)
        else:
            raise InputAreNotTheSameDataType

    def intersect(self, other):
        # returns intersected area
        if isinstance(other, Area2D):
            intersection = shapely.set_operations.intersection(
                self.area, other.area, grid_size=default_precision)
            if intersection.geom_type == 'GeometryCollection':
               if intersection.geoms[0].geom_type == 'Polygon':
                   polygon = intersection.geoms[0]
               else:
                   polygon = shapely.polygons(None)
            else:
                polygon = intersection
            return Area2D.from_shapely_polygon(polygon)
        else:
            raise InputAreNotTheSameDataType

    def remove_area_old(self, other):
        if isinstance(other, Area2D):
            remain_areas = shapely.constructive.simplify(
                shapely.set_operations.difference(self.area, other.area, grid_size=default_precision), tolerance=wp_slice_precision)
            return Area2D.from_shapely_polygon(remain_areas)
        else:
            raise InputAreNotTheSameDataType
    def remove_area(self, other):
        if isinstance(other, Area2D):
            remain_areas = shapely.constructive.simplify(
                shapely.set_operations.difference(self.area, other.area, grid_size=default_precision), tolerance=wp_slice_precision)
            if remain_areas.geom_type == 'MultiPolygon':
                area_sort = sorted(remain_areas.geoms, key=lambda a: a.area, reverse=True)
            elif remain_areas.geom_type == 'GeometryCollection':
                if remain_areas.geoms[0].geom_type == 'Polygon':
                    area_sort = [remain_areas.geoms[0]]
                else:
                    area_sort = [shapely.polygons(None)]
            else:
                area_sort = [remain_areas]
            return Area2D.from_shapely_polygon(area_sort[0])
        else:
            raise InputAreNotTheSameDataType

    def size(self) -> float:
        return shapely.measurement.area(self.area)

    def check_and_make_valid(self):
        "Inspect if the input is a valid shapely geometry, if not make it valid."

        if not shapely.predicates.is_valid(self.area):
            new_area = shapely.make_valid(self.area)
            if shapely.predicates.is_valid(new_area):
                self.area = new_area
            else:
                self.save_to_disk(
                    r"./Resources/BugObjectCollections/area_2D_not_valid_after_make_valid.pkl")
                raise Exception(
                    "The polygon is not valid. Even after using make_valid method. The invalid polygon is saved in ./Resources/BugObjectCollections/area_2D_not_valid_after_make_valid.pkl")
        else:
            pass

    def check_and_get_largest_polygon(self):
        "Inspect the input geometry. If it is not polygon, extract the polygon with largest area if exists, otherwise return empty polygon."
        if shapely.get_type_id(self.area) == 3:
            pass
        elif shapely.get_type_id(self.area) == 6:
            self.area = sorted(shapely.get_parts(
                self.area), key=lambda a: shapely.area(a), reverse=True)[0]
        elif shapely.get_type_id(self.area) == 7:
            parts = shapely.get_parts(
                self.area)
            polygon_in_geom_collection = False
            for part in parts:
                if shapely.get_type_id(part) == 3:
                    polygon_in_geom_collection = True
                    break
            if polygon_in_geom_collection:
                self.area = sorted(
                    parts, key=lambda a: shapely.area(a), reverse=True)[0]
            else:
                self.area = shapely.polygons(None)
        else:
            self.area = shapely.polygons(None)
        if shapely.get_type_id(self.area) != 3:
            raise Exception(
                "Failure to extract polygon")

    def compare(self, other, precision):
        """areas are regarded as equal if the area of the intersection is.
        """
        # Possible Problem: shapely.equals and almost_equals is not stable: see discussion:
        # https://stackoverflow.com/questions/63402333/how-does-almost-equals-function-of-shapely-treat-the-starting-point-and-errors
        if isinstance(other, Area2D):
            # since shapely runs into a problem when intersection identical polygons
            # we first check with shapely built in functions which does not have this problem
            # the shapely compare function has a lot of other problems
            # (raises if representations are diffrent but defined area are equal)
            # therefore we check later with intersection approach
            number_of_decimals_in_precision = -int(np.log10(precision))

            # if self.area.almost_equals(other.area, number_of_decimals_in_precision):
            #     pass
            # elif np.abs(self.intersect(other).area.area-self.area.area)+np.abs(
            #         self.intersect(other).area.area-other.area.area) > precision:
            #

            if shapely.predicates.equals_exact(self.area, other.area, tolerance=precision):
                pass
            elif np.abs(shapely.measurement.area(self.area) - shapely.measurement.area(other.area)) > precision:
                raise Exception("The areas are not the same")
        else:
            raise InputAreNotTheSameDataType

    def to_line2d(self) -> Line2D:

        x_values_list = []
        y_values_list = []
        vertex_coordinates_list = []
        if isinstance(self.area, shapely.Geometry):
            vertex_coordinates_list += shapely.coordinates.get_coordinates(
                self.area).tolist()
        else:
            nr_of_polygons = len(self.area)
            for polygon_idx in range(nr_of_polygons):
                vertex_coordinates_list += shapely.coordinates.get_coordinates(
                    self.area[polygon_idx]).tolist()

        for coordinate in vertex_coordinates_list:
            x_values_list.append(coordinate[0])
            y_values_list.append(coordinate[1])

        return Line2D(np.asarray(x_values_list), np.asarray(y_values_list))

    def get_boundary_points(self) -> List[Vector]:
        return Vector.from_xz_nparrays(shapely.coordinates.get_coordinates(self.area)[:-1])

    def save_to_disk(self, file_name: str):
        with open(file_name, 'wb') as file:
            dill.dump(self, file)

    @ staticmethod
    def load_from_disk(file_name):
        with open(file_name, 'rb') as file:
            area2d = dill.load(file)
        if isinstance(area2d, Area2D):
            return area2d
        else:
            Exception(
                "The loaded Area2D object is not from the current version of this class.")


class Triangle:
    """Triangle defines a triangle in 3D space. It's the basic unit to compose a mesh.
    """
    vertex_1: Vector
    vertex_2: Vector
    vertex_3: Vector
    vertices: List[Vector]

    def __init__(self, vertex_1: Vector, vertex_2: Vector, vertex_3: Vector) -> None:
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.vertex_3 = vertex_3
        self.vertices = [vertex_1, vertex_2, vertex_3]

    def transform(self, transform: Transform):
        if isinstance(transform, Transform):
            return Triangle(self.vertex_1.transform(transform),
                            self.vertex_2.transform(transform),
                            self.vertex_3.transform(transform))
        else:
            raise Exception('input is not of type Transform')

    def compare(self, other, precision):
        if isinstance(other, Triangle):
            # TODO: simplify the compare algorithms
            number_of_same_elements = 0
            # check if it's a valid triangle first, otherwise it's
            # ambiguious how the count of same number
            # of elements represents equality of triangles.
            condition_1 = self.vertex_1.distance(self.vertex_2) > precision
            condition_2 = self.vertex_1.distance(self.vertex_3) > precision
            condition_3 = self.vertex_2.distance(self.vertex_3) > precision

            condition_4 = other.vertex_1.distance(other.vertex_2) > precision
            condition_5 = other.vertex_1.distance(other.vertex_3) > precision
            condition_6 = other.vertex_2.distance(other.vertex_3) > precision

            # TODO: we might need a unified function to compare 2 list of vectors,
            # if they are the same or not.
            # This could be done by creating a stand alone helper function,
            # or as a method in a potentail VectorList
            # class, where other useful methods on list of vectors are also included.
            if condition_1 & condition_2 & condition_3:
                if condition_4 & condition_5 & condition_6:
                    for index in range(0, len(self.vertices)):
                        for index_j in range(0, len(self.vertices)):
                            if self.vertices[index].distance(other.vertices[index_j]) < precision:
                                number_of_same_elements += 1
                else:
                    raise Exception(
                        "The second triangle is not a valid Triangle")
            else:
                raise Exception("The first triangle is not a valid Triangle")

            if number_of_same_elements >= 3:
                pass
            else:
                raise Exception("These 2 Triangles are not the same")

        else:
            raise InputAreNotTheSameDataType


def get_bounding_sphere_radius(vertices: List[Vector]) -> float:
    # mean of vertices
    mean_point = Vector.origin()
    for vertex in vertices:
        mean_point = mean_point.add(vertex)
    mean_point = mean_point.scale(1.0 / float(len(vertices)))

    # get maximal distance to the vertices from mean_point
    distances = [mean_point.subtract(vertex).norm() for vertex in vertices]
    bounding_sphere_radius = max(distances)

    return bounding_sphere_radius
