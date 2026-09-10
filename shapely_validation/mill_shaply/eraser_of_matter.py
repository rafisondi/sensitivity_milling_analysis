import time

try:                                    # imported as mill_shaply.eraser_of_matter
    from .GeopmetryStandalone import *
except ImportError:                      # run directly from inside mill_shaply/
    from GeopmetryStandalone import *
import math as m
import numpy as np
from scipy.interpolate import splprep, splev

import copy
# from multi_robot import transformation_adjoint, trans_disp, trans_rot_x, trans_rot_y, trans_rot_z, T_inv

from matplotlib import pyplot as plt
from matplotlib import animation as ani
import matplotlib.patches as patches

def transformation_adjoint(transformation_matrix):
    rotation = transformation_matrix[:3, :3]
    translation = transformation_matrix[:3, 3]
    adjoint = np.zeros((6, 6))
    adjoint[:3, :3] = rotation
    adjoint[3:, :3] = np.dot(skew_symmertic_matrix(translation), rotation)
    adjoint[3:, 3:] = rotation
    return adjoint

def skew_symmertic_matrix(vector):
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def trans_disp(x, y, z):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def trans_rot_x(alpha):
    T = np.eye(4)
    T[1, 1] = m.cos(alpha)
    T[1, 2] = -m.sin(alpha)
    T[2, 1] = m.sin(alpha)
    T[2, 2] = m.cos(alpha)
    return T


def trans_rot_y(alpha):
    T = np.eye(4)
    T[0, 0] = m.cos(alpha)
    T[0, 2] = m.sin(alpha)
    T[2, 0] = -m.sin(alpha)
    T[2, 2] = m.cos(alpha)
    return T


def trans_rot_z(alpha):
    T = np.eye(4)
    T[0, 0] = m.cos(alpha)
    T[0, 1] = -m.sin(alpha)
    T[1, 0] = m.sin(alpha)
    T[1, 1] = m.cos(alpha)
    return T


def T_inv(T):
    p = T[:3, 3]
    RT = T[:3, :3].T
    Tinv = np.eye(4)
    Tinv[:3, :3] = RT
    Tinv[:3, 3] = -np.dot(RT, p)
    return Tinv

class milling_workpiece:
    def __init__(self, xy_workpiece=np.array([[0, -76, -76, 0], [0, 0, -100, -100]]), axial_cutting_depth=1., T_base_workpiece=None,
                 number_of_teeth=1, diameter_end_mill=20.0, helix_angle_deg=45.0,
                 max_layer_rot_between_slices_deg=10.0,
                 Ktc=1930.4, Krc=1159.6, Kac=200.6):
        self.T_base_workpiece = T_base_workpiece if T_base_workpiece is not None else np.eye(4)
        self.number_of_teeth = number_of_teeth # 4
        self.helix_angle = helix_angle_deg / 180 * m.pi#48 / 180 * m.pi
        self.diameter_end_mill = diameter_end_mill #10.
        self.radius_tool = self.diameter_end_mill / 2
        self.axial_cutting_depth = axial_cutting_depth
        self.max_layer_rot_between_slices = max_layer_rot_between_slices_deg / 180 * m.pi
        self.number_of_slices = 1 + m.floor(self.axial_cutting_depth * m.tan(self.helix_angle) / (self.radius_tool * self.max_layer_rot_between_slices))
        self.slice_height = self.axial_cutting_depth / self.number_of_slices

        self.num_tolerance = 1 / 1000000000

        self.force_plot_scaling = 0.04 * self.slice_height # 1 / 500 * self.number_of_slices #

        # standard model steel
        self.cutting_force_coefficient_Ktc = Ktc
        self.cutting_force_coefficient_Krc = Krc
        self.cutting_force_coefficient_Kac = Kac

        self.total_milling_wrench_wpframe = np.zeros(6)
        self.total_milling_wrench_dynframe = np.zeros(6)
        self.total_milling_force = np.zeros(3)


        self.scaling_area = 1
        self.workpiece_slice = []
        self.helix_offset = np.zeros(self.number_of_slices)
        self.x_workpiece = []
        self.y_workpiece = []
        for k in range(self.number_of_slices):
            self.workpiece_slice.append(Area2D(Line2D(xy_workpiece[0, :] * self.scaling_area, xy_workpiece[1, :] * self.scaling_area))) # 2D area defined by boundary points xy_workpiece = (x_points, y_points)
            self.helix_offset[k] = k * self.slice_height / self.radius_tool * m.tan(self.helix_angle)
            boundary_workpiece_k = self.workpiece_slice[k].get_boundary_points()
            self.x_workpiece.append(np.zeros(len(boundary_workpiece_k)))
            self.y_workpiece.append(np.zeros(len(boundary_workpiece_k)))
            for p in range(len(boundary_workpiece_k)):
                self.x_workpiece[k][p] = boundary_workpiece_k[p].value[0] * 1 / self.scaling_area
                self.y_workpiece[k][p] = boundary_workpiece_k[p].value[2] * 1 / self.scaling_area

        if self.number_of_slices > 1:
            self.local_length = m.sqrt(self.radius_tool ** 2 * ((m.cos(self.helix_offset[1]) - 1) ** 2 + m.sin(self.helix_offset[0]) ** 2) + self.slice_height ** 2)
        else:
            self.local_length = self.slice_height

        self.xy_tool_center = None
        self.xy_tool_center_T1 = None
        self.xy_tool_center_T2 = None
        self.orientation = None
        self.orientation_T1 = None
        self.orientation_T2 = None


    def plot_slices(self, axs):
        tips = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
        tips_hist = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
        chip = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
        chip_edge = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
        tooth_force = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
        workpiece = [None] * self.number_of_slices
        tool = [None] * self.number_of_slices
        slice_force = [None] * self.number_of_slices

        for k in range(self.number_of_slices):

            # --- remaining workpiece slice ---
            workpiece[-k-1], = axs[-k-1, 0].fill(
                self.x_workpiece[k], self.y_workpiece[k],
                color='b', label=f"Workpiece slice {k}", alpha=0.5
            )

            # --- per-tooth plotting (paths, chips, forces) ---
            for j in range(self.number_of_teeth):

                # tooth tip path
                tips[j][-k-1], = axs[-k-1, 0].plot(
                    self.tip_paths[j][k][0],
                    self.tip_paths[j][k][1]
                )
                int_color = axs[-k-1, 0].lines[-1].get_color()

                # tooth tip history
                tips_hist[j][-k-1], = axs[-k-1, 0].plot(
                    [self.tooth_tips[j, k, 0],
                    self.tooth_tips_T1[j, k, 0],
                    self.tooth_tips_T2[j, k, 0]],
                    [self.tooth_tips[j, k, 1],
                    self.tooth_tips_T1[j, k, 1],
                    self.tooth_tips_T2[j, k, 1]],
                    'x', color=int_color
                )

                # chip polygon
                chip[j][-k-1], = axs[-k-1, 0].fill(
                    self.x_chip[j][k],
                    self.y_chip[j][k],
                    color=int_color,
                    alpha=0.5
                )

                # chip thickness line
                if self.B[j][k] is not None:
                    chip_edge[j][-k-1], = axs[-k-1, 0].plot(
                        [self.tooth_tips[j, k, 0], self.B[j][k][0]],
                        [self.tooth_tips[j, k, 1], self.B[j][k][1]],
                        'r'
                    )

                # tooth force arrow
                tooth_force[j][-k-1] = axs[-k-1, 0].arrow(
                    self.tooth_tips[j, k, 0],
                    self.tooth_tips[j, k, 1],
                    self.wrench_wpframe[j][k][3] * self.force_plot_scaling,
                    self.wrench_wpframe[j][k][4] * self.force_plot_scaling,
                    length_includes_head=True,
                    color=int_color,
                    head_width=0.2,
                    head_length=0.3
                )

            # center of tool
            center_xy = np.asarray(self.xy_tool_center, dtype=float).reshape(-1)
            cx, cy = float(center_xy[0]), float(center_xy[1])
            # inner circle radius: 0.75 * full radius
            r_inner = 0.7 * self.radius_tool

            # 1) draw inner circle
            circle = patches.Circle(
                (cx, cy),
                radius=r_inner,
                color='grey',
            )
            axs[-k-1, 0].add_patch(circle)

            # 2) for each tooth, draw a triangle tip → inner circle → center
            for j in range(self.number_of_teeth):
                # Curret tooth tip coordinates
                tx = float(self.tooth_tips[j, k, 0])
                ty = float(self.tooth_tips[j, k, 1])

                # vector from center to tooth tip
                vx = tx - cx
                vy = ty - cy
                norm = np.hypot(vx, vy)
                if norm < 1e-12:
                    continue  # safety

                # unit radial direction
                ux = vx / norm
                uy = vy / norm

                # unit orthogonal direction (perpendicular)
                ox = -uy      # = -vy/norm
                oy = ux       # =  vx/norm

                # choose how "long" the flange should be (e.g. along inner circle)
                flange_len = 0.4*self.radius_tool   # or smaller if you want a thinner flange

                # point on inner circle in orthogonal direction
                ix = cx + flange_len * ox
                iy = cy + flange_len * oy

                # triangle vertices: [tip, inner-orthogonal, center]
                tri = patches.Polygon(
                    [[tx, ty], [ix, iy], [cx, cy]],
                    closed=True,
                    facecolor='grey',
                )
                axs[-k-1, 0].add_patch(tri)

            # store any one object as 'tool' for this slice (e.g., the circle)
            tool[-k-1] = circle

            # --- slice resultant force at tool center ---
            slice_force[-k-1] = axs[-k-1, 0].arrow(
                cx, cy,
                self.slice_k_milling_wrench_wpframe[k][3] * self.force_plot_scaling,
                self.slice_k_milling_wrench_wpframe[k][4] * self.force_plot_scaling,
                label=f"Force from slice {k}",
                length_includes_head=True,
                color='grey',
                head_width=0.2,
                head_length=0.3,
                zorder=9999
                )

        return tips, tips_hist, chip, chip_edge, tooth_force, workpiece, tool, slice_force


    def show_plot(self, window_name="Milling process", clear=True):
        fig, axs = plt.subplots(
            self.number_of_slices,
            1,
            squeeze=False,
            figsize=(12, self.number_of_slices * 6),
            num=window_name,
            clear=clear,
        )
        self.plot_slices(axs)
        for k in range(self.number_of_slices):
            axs[-k-1, 0].axis('equal')
            axs[-k-1, 0].set_xlim(self.xy_tool_center[0] - 2.5 * self.radius_tool, self.xy_tool_center[0] + 2.5 * self.radius_tool)
            axs[-k-1, 0].set_ylim(self.xy_tool_center[1] - 1.25 * self.radius_tool, self.xy_tool_center[1] + 1.25 * self.radius_tool)
            axs[-k-1, 0].set_xlabel('Workpiece x-axis [mm]')
            axs[-k-1, 0].set_ylabel('Workpiece y-axis [mm]')
            axs[-k-1, 0].legend(loc='lower right')
        fig.suptitle(f"Time {self.time:.3f} s")
        # plt.show()
        return fig, axs

    def show_plot_oneslice(self, window_name="Milling process (one slice)", clear=True):
        self.number_of_slices_old = self.number_of_slices
        self.number_of_slices = 1

        fig, axs = plt.subplots(
            self.number_of_slices,
            1,
            squeeze=False,
            figsize=(6, self.number_of_slices * 3),
            num=window_name,
            clear=clear,
        )
        self.plot_slices(axs)
        for k in range(self.number_of_slices):
            axs[-k-1, 0].axis('equal')
            axs[-k-1, 0].set_xlim(self.xy_tool_center[0] - 4 * self.radius_tool, self.xy_tool_center[0] + 4 * self.radius_tool)
            axs[-k-1, 0].set_ylim(self.xy_tool_center[1] - 2 * self.radius_tool, self.xy_tool_center[1] + 2 * self.radius_tool)
            axs[-k-1, 0].set_xlabel('Workpiece x-axis [mm]')
            axs[-k-1, 0].set_ylabel('Workpiece y-axis [mm]')
            axs[-k-1, 0].legend(loc='lower right')
        fig.suptitle(f"Time {self.time:.6f} s")
        # plt.show()
        self.number_of_slices = self.number_of_slices_old
        return fig, axs

    def show_animation(self, fig, axs, artists):
        print(artists)
        for k in range(self.number_of_slices):
            axs[-k-1].axis('equal')
            axs[-k-1].set_xlim(-10, 60)
            axs[-k-1].set_ylim(-10,35)
            axs[-k-1].set_xlabel('Workpiece x-axis [mm]')
            axs[-k-1].set_ylabel('Workpiece y-axis [mm]')
        a = ani.ArtistAnimation(fig=fig, artists=artists, interval=100)
        plt.show()

    def erase_step(self, xy_tool_center, orientation, direction=-1, t=0):
        self.time = t
        self.direction = direction
        self.xy_tool_center_T2 = self.xy_tool_center_T1
        self.orientation_T2 = self.orientation_T1
        self.xy_tool_center_T1 = self.xy_tool_center
        self.orientation_T1 = self.orientation
        center_xy = np.asarray(xy_tool_center, dtype=float).reshape(-1)
        if center_xy.size < 2:
            raise ValueError(f"xy_tool_center must contain at least 2 elements, got shape {np.shape(xy_tool_center)}")
        self.xy_tool_center = center_xy[:2].copy()
        self.orientation = orientation
        self.T_WPDyn = trans_rot_z(self.orientation)
        T_tn = trans_rot_z(m.pi / 2) @ trans_rot_x(m.pi)

        if self.xy_tool_center_T2 is None:
            return
        else:
            self.tooth_tips = self.tool_to_teeth(self.xy_tool_center, self.orientation)
            self.tooth_tips_T1 = self.tool_to_teeth(self.xy_tool_center_T1, self.orientation_T1)
            self.tooth_tips_T2 = self.tool_to_teeth(self.xy_tool_center_T2, self.orientation_T2)

            self.tip_paths = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
            self.cutting_area = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]

            self.x_workpiece = []
            self.y_workpiece = []
            self.x_chip = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
            self.y_chip = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]

            self.B = [[None] * self.number_of_slices for j in range(self.number_of_teeth)]
            self.chip_thickness = np.zeros((self.number_of_teeth, self.number_of_slices)) 

            self.tooth_orientation = np.zeros((self.number_of_teeth, self.number_of_slices))


            self.wrench_local = np.zeros((self.number_of_teeth, self.number_of_slices, 6))
            self.force_wpframe = np.zeros((self.number_of_teeth, self.number_of_slices, 3))
            self.wrench_wpframe = np.zeros((self.number_of_teeth, self.number_of_slices, 6))

            self.slice_k_milling_wrench_wpframe = np.zeros((self.number_of_slices, 6))
            self.slice_k_milling_force = np.zeros((self.number_of_slices, 3))
            self.total_milling_wrench_wpframe = np.zeros(6)
            self.total_milling_wrench_dynframe = np.zeros(6)
            self.total_milling_force = np.zeros(3)


            for k in range(self.number_of_slices):
                for j in range(self.number_of_teeth):
                    x_tip_path_jk = np.array([self.tooth_tips[j, k, 0], self.tooth_tips_T1[j, k, 0], self.tooth_tips_T2[j, k, 0]])
                    y_tip_path_jk = np.array([self.tooth_tips[j, k, 1], self.tooth_tips_T1[j, k, 1], self.tooth_tips_T2[j, k, 1]])
                    tck, u = splprep([x_tip_path_jk, y_tip_path_jk], s=0, k=2)
                    u_int_1 = np.linspace(u[0], u[1], 5)
                    # u_int_2 = np.linspace(u[1], u[2], 5)
                    self.tip_paths[j][k] = splev(u_int_1, tck)
                    cutting_area_jk_a = Area2D(Line2D(np.append(self.tip_paths[j][k][0], self.xy_tool_center_T1[0]) * self.scaling_area, np.append(self.tip_paths[j][k][1], self.xy_tool_center_T1[1]) * self.scaling_area))
                    cutting_area_jk_b = Area2D(Line2D(np.append(self.tip_paths[j][k][0], self.xy_tool_center[0]) * self.scaling_area, np.append(self.tip_paths[j][k][1], self.xy_tool_center[1]) * self.scaling_area))
                    self.cutting_area[j][k] = cutting_area_jk_b.union(cutting_area_jk_a)

                    chip_jk = self.workpiece_slice[k].intersect(self.cutting_area[j][k])



                    boundary_chip_jk = chip_jk.get_boundary_points()
                    self.x_chip[j][k] = np.zeros(len(boundary_chip_jk))
                    self.y_chip[j][k] = np.zeros(len(boundary_chip_jk))
                    for p in range(len(boundary_chip_jk)):
                        self.x_chip[j][k][p] = boundary_chip_jk[p].value[0] * 1 / self.scaling_area
                        self.y_chip[j][k][p] = boundary_chip_jk[p].value[2] * 1 / self.scaling_area

                    self.workpiece_slice[k] = self.workpiece_slice[k].remove_area(self.cutting_area[j][k])


                    tip_index = np.argwhere(np.logical_and(abs(self.tooth_tips[j, k, 0] - self.x_chip[j][k]) < self.num_tolerance, abs(self.tooth_tips[j, k, 1] - self.y_chip[j][k]) < self.num_tolerance))
                    if tip_index.size > 0:
                        candidate_plus = (tip_index[0, 0] + 1) % len(boundary_chip_jk)
                        candidate_minus = (tip_index[0, 0] - 1) % len(boundary_chip_jk)
                        length_tip20 = np.sqrt((self.xy_tool_center[0] - self.tooth_tips[j, k, 0]) ** 2 + (self.xy_tool_center[1] - self.tooth_tips[j, k, 1]) ** 2)
                        proj_plus_on_tip20 = 1. / length_tip20 * ((self.xy_tool_center[0] - self.tooth_tips[j, k, 0]) * (self.x_chip[j][k][candidate_plus] - self.tooth_tips[j, k, 0]) + (self.xy_tool_center[1] - self.tooth_tips[j, k, 1]) * (self.y_chip[j][k][candidate_plus] - self.tooth_tips[j, k, 1]))
                        length_Aplus = np.sqrt((self.x_chip[j][k][candidate_plus] - self.tooth_tips[j, k, 0]) ** 2 + (self.y_chip[j][k][candidate_plus] - self.tooth_tips[j, k, 1]) ** 2)
                        proj_minus_on_tip20 = 1. / length_tip20 * ((self.xy_tool_center[0] - self.tooth_tips[j, k, 0]) * (self.x_chip[j][k][candidate_minus] - self.tooth_tips[j, k, 0]) + (self.xy_tool_center[1] - self.tooth_tips[j, k, 1]) * (self.y_chip[j][k][candidate_minus] - self.tooth_tips[j, k, 1]))
                        length_Aminus = np.sqrt((self.x_chip[j][k][candidate_minus] - self.tooth_tips[j, k, 0]) ** 2 + (self.y_chip[j][k][candidate_minus] - self.tooth_tips[j, k, 1]) ** 2)
                        if abs(length_Aplus - proj_plus_on_tip20) <= abs(length_Aminus - proj_minus_on_tip20):
                            self.B[j][k] = np.array([self.x_chip[j][k][candidate_plus], self.y_chip[j][k][candidate_plus]])
                        else:
                            self.B[j][k] = np.array([self.x_chip[j][k][candidate_minus], self.y_chip[j][k][candidate_minus]])

                        self.chip_thickness[j, k] = np.sqrt((self.B[j][k][0] - self.tooth_tips[j, k, 0]) ** 2 + (self.B[j][k][1] - self.tooth_tips[j, k, 1]) ** 2)

                        self.wrench_local[j, k, 3] = -direction * self.cutting_force_coefficient_Ktc    * self.chip_thickness[j, k] * self.slice_height 
                        self.wrench_local[j, k, 4] = -self.cutting_force_coefficient_Krc                * self.chip_thickness[j, k] * self.slice_height
                        self.wrench_local[j, k, 5] = -self.cutting_force_coefficient_Kac                * self.chip_thickness[j, k] * self.slice_height

                        self.tooth_orientation[j, k] = np.arctan2(self.tooth_tips[j, k, 1] - self.xy_tool_center[1], self.tooth_tips[j, k, 0] - self.xy_tool_center[0])
                        T_WPtooth = trans_rot_z(self.tooth_orientation[j, k]) @ T_tn @ trans_disp(0, self.radius_tool, -k * self.slice_height)
                        self.wrench_wpframe[j, k, :] = transformation_adjoint(T_inv(T_WPtooth)).T @ self.wrench_local[j, k, :]

                        self.slice_k_milling_wrench_wpframe[k, :] += self.wrench_wpframe[j, k, :]

                self.total_milling_wrench_wpframe += self.slice_k_milling_wrench_wpframe[k, :]

                boundary_workpiece_k = self.workpiece_slice[k].get_boundary_points()
                self.x_workpiece.append(np.zeros(len(boundary_workpiece_k)))
                self.y_workpiece.append(np.zeros(len(boundary_workpiece_k)))
                for p in range(len(boundary_workpiece_k)):
                    self.x_workpiece[k][p] = boundary_workpiece_k[p].value[0] * 1 / self.scaling_area
                    self.y_workpiece[k][p] = boundary_workpiece_k[p].value[2] * 1 / self.scaling_area

            self.total_milling_wrench_dynframe = transformation_adjoint(self.T_WPDyn).T @ self.total_milling_wrench_wpframe
            self.total_milling_force = self.total_milling_wrench_wpframe[3:6].copy()


    def tool_to_teeth(self, xy_tool_center, orientation_0):
        # all in workpiece frame
        # tooth_tips = [[np.zeros(3)] * self.number_of_slices for j in range(self.number_of_teeth)]
        # orientation_tooth = [[0] * self.number_of_slices for j in range(self.number_of_teeth)]
        center_xy = np.asarray(xy_tool_center, dtype=float).reshape(-1)
        if center_xy.size < 2:
            raise ValueError(f"xy_tool_center must contain at least 2 elements, got shape {np.shape(xy_tool_center)}")
        cx, cy = float(center_xy[0]), float(center_xy[1])
        tooth_tips = np.zeros((self.number_of_teeth, self.number_of_slices, 3))
        for j in range(self.number_of_teeth):
            for k in range(self.number_of_slices):
                # self.helix_offset = k * m.pi / 10 #need to compute properly based on number of slices and helix angle
                orientation_tooth = orientation_0 + self.helix_offset[k] + j * 2 * m.pi / self.number_of_teeth
                tooth_tips[j, k, 0] = cx + self.radius_tool * m.cos(orientation_tooth)
                tooth_tips[j, k, 1] = cy + self.radius_tool * m.sin(orientation_tooth)
                tooth_tips[j, k, 2] = k * self.slice_height
        return tooth_tips







