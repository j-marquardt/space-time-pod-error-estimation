import sys
sys.path.append("./../src/")

import numpy as np
import matplotlib.pyplot as plt

from helper import *
from settings_pod_simple import *

settings = settings_pod_simple()
grid = helper.grid_from_settings(settings)


####################
# GENERAL SETTINGS #
####################

show_fom = True
show_rom = True

show_projection_error = True
show_rom_error = True

create_eigevalues_comparison_plot = True

depict_spatial_pod_basis = True
depict_temporal_pod_basis = True

save_plots = False

# set the reduced sizes
# this is the number of reduced basis functions, not the number of nodes.
# Hence, FOM dimension is reached for x_red_size = x_num - 2 and
# t_red_size = t_num
x_red_size = 20
t_red_size = 20


######################
# BASIC COMPUTATIONS #
######################

# get x and t spaces
x_space = grid.get_x_space()
t_space = grid.get_t_space()

# number of nodes and basis functions
num_x_nodes = grid.get_num_x_nodes()
num_t_nodes = grid.get_num_t_nodes()

num_x_basis_funct = num_x_nodes - 2
num_t_basis_funct = num_t_nodes

# assuming an equidistant grid in space and time for now
dx = abs(x_space[1] - x_space[0])
dt = abs(t_space[1] - t_space[0])


#############
# SOLVE FOM #
#############

# solve the FOM using the product FEM approach

sys_mat, f_SY = helper.get_fom(grid, settings)

# solve the system
v = np.linalg.solve(sys_mat, f_SY)

# measurement matrix
X = v.reshape((num_t_basis_funct, num_x_basis_funct)).T

# full order model
Y = np.zeros((num_x_nodes, num_t_nodes))
Y[1:-1,:] = X


if show_fom:
    helper.plot_function(Y, grid, "simple_fom.pdf" if save_plots else "")


####################
# CHOLESKY FACTORS #
####################


L_s, L_y = helper.get_mass_matrix_factors(grid, settings)

L_s_inv = np.linalg.inv(L_s)
L_y_inv = np.linalg.inv(L_y)

#################################
# CREATE SINGULAR VALUE GRAPHIC #
#################################
    
if create_eigevalues_comparison_plot:
    helper.singular_value_comparison_plot(X, 15, 5, grid, settings, "simple_s_vals_comp.pdf" if save_plots else "")

##########################
# COMPUTE REDUCED SPACES #
##########################


X_space_red, V_q, _ = helper.space_reduction(X, x_red_size, grid, settings)

X_space_time_red, U_s, _ = helper.time_reduction(X_space_red, t_red_size, grid, settings)

projected_dynamics = np.zeros((num_x_nodes, num_t_nodes))
projected_dynamics[1:-1,:] = X_space_time_red

if show_projection_error:
    helper.plot_heatmap_error_log(np.abs(Y-projected_dynamics), grid, "simple_y_proj_diff.pdf" if save_plots else "")

#############
# SOLVE ROM #
#############

red_sys_mat, f_SY_red = helper.get_rom(V_q, U_s, grid, settings)

v = np.linalg.solve(red_sys_mat, f_SY_red)

red_sol= np.zeros((num_x_nodes, num_t_nodes))
red_sol[1:-1,:] = (np.kron(L_s_inv.T @ U_s, L_y_inv.T @ V_q)@ v).reshape((num_t_basis_funct, num_x_basis_funct)).T

if show_rom:
    helper.plot_function(red_sol, grid, "simple_rom.pdf" if save_plots else "")

if show_rom_error:
    helper.plot_heatmap_error_log(np.abs(Y-red_sol), grid, "simple_y_red_diff.pdf" if save_plots else "")



#########################
# INVESTIGATE POD BASES #
#########################

if depict_spatial_pod_basis:
    
    disp_x_basis_func = 23
    
    # full dimensional basis
    fem_basis_x = np.zeros((num_x_basis_funct, num_x_nodes))
    fem_basis_x[:, 1:-1] = np.eye(num_x_basis_funct)
    
    # compute the pod basis after just the spatial reduciton
    _, V_q, _ = helper.space_reduction(X, disp_x_basis_func, grid, settings)
    pod_basis_x = V_q.T @ L_y_inv @ fem_basis_x
    
    plt.figure("Eigenfunctions space POD")
    plt.xlim((x_space[0], x_space[-1]))
    plt.ylim((-3, 3))
    for i in range(disp_x_basis_func):
        plt.plot(x_space, pod_basis_x[i, :], label=f"$\\nu_{i+1}$", c="red", linewidth=0.2, alpha=0.2)
    
    plt.plot(x_space, pod_basis_x[disp_x_basis_func - 1, :], "--", c="blue", linewidth=1.1)
    plt.xlabel('$\\xi$', fontsize=15)

    if save_plots:
        plt.savefig("simple_pod_basis_space.pdf", bbox_inches='tight')
    plt.show()



if depict_temporal_pod_basis:
    
    disp_t_basis_func = 6
    
    # full dimensional basis
    fem_basis_t = np.eye(num_t_basis_funct)
    
    # in case of a temporal reduction, we have not computed U_s before
    # (but only for a temporal reduciton after a spatial reduction)
    _, U_s, _ = helper.time_reduction(X, disp_t_basis_func, grid, settings)
    
    pod_basis_t = U_s.T @ L_s_inv @ fem_basis_t
    
    plt.figure("Eigenfunctions time POD")
    plt.xlim((t_space[0], t_space[-1]))
    plt.ylim((-3, 3))
    plt.plot(t_space, pod_basis_t[0, :], "--", c="blue", linewidth=0.8)

    for i in [1,2,3,4]:
        plt.plot(t_space, pod_basis_t[i, :], label=f"$\\psi_{i+1}$", c="red", linewidth=0.25)

    plt.plot(t_space, pod_basis_t[5, :], "-.", c="green", linewidth=0.8)
    plt.xlabel('$\\tau$', fontsize=15)
    
    if save_plots:
        plt.savefig("simple_pod_basis_time.pdf", bbox_inches='tight')
    plt.show()


