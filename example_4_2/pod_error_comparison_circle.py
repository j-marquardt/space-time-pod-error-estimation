import sys
sys.path.append("./../src/")

import numpy as np
import matplotlib.pyplot as plt

from helper import *
from settings_pod_circle import *

from matplotlib.colors import LogNorm

settings = settings_pod_circle()
grid = helper.grid_from_settings(settings)


####################
# GENERAL SETTINGS #
####################

# dimensions of the ROM to start with
space_start = 1
time_start = 2

visualize_f = True
show_fom = True
create_eigevalues_comparison_plot = True

only_q_s_hat_equal = False

load_matrices_from_file = True
save_plots = False


show_error_plot_for = []
#show_error_plot_for = range(0, grid.get_num_x_nodes() - 2) #<- show error plot for all iteration


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

if visualize_f:
    helper.visualize_f(grid, settings, "circle_f.pdf" if save_plots else "")


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
    helper.plot_function(Y, grid, "circle_fom.pdf" if save_plots else "")


####################
# CHOLESKY FACTORS #
####################

L_s, L_y = helper.get_mass_matrix_factors(grid, settings)

L_s_inv, L_y_inv = helper.get_mass_matrix_inv_factors(grid, settings)

J_s, _ = helper.get_stiff_matrix_factors(grid, settings)


#################################
# CREATE SINGULAR VALUE GRAPHIC #
#################################
    
if create_eigevalues_comparison_plot:
    helper.singular_value_comparison_plot(X, 40, 20, grid, settings, "circle_s_vals_comp.pdf" if save_plots else "")


#############################
# INVESTIGATION OF CONSTANT #
#############################

# we investigate the constant from the proof of Proposition 3.3
C_Y_hat_to_S_hat = np.linalg.norm(L_s_inv @ J_s, ord="fro")
print("Constant: " + str(C_Y_hat_to_S_hat))

if load_matrices_from_file:
    
    data = np.load("circle_data.npz")
    sigma_Y_S_diag_vals = data["sigma_Y_S_diag_vals"]
    proj_error_Y_S_diag_vals = data["proj_error_Y_S_diag_vals"]
    true_error_Y_S_diag_vals = data["true_error_Y_S_diag_vals"]
    sigma_Y_S = data["sigma_Y_S"]
    proj_error_Y_S = data["proj_error_Y_S"]
    true_error_Y_S = data["true_error_Y_S"]
    
else:
    
    # x times t matrix. Error in each entry
    sigma_Y_S = np.zeros((num_x_basis_funct-space_start, num_t_basis_funct-time_start))
    sigma_Y_S_diag_vals = []
    
    proj_error_Y_S = np.zeros((num_x_basis_funct-space_start, num_t_basis_funct-time_start))
    proj_error_Y_S_diag_vals = []
    
    true_error_Y_S = np.zeros((num_x_basis_funct-space_start, num_t_basis_funct-time_start))
    true_error_Y_S_diag_vals = []
    
    # loop through all possible combinations of number of space / time basis functions
    for x_red_size in range(space_start, num_x_basis_funct):
        for t_red_size in range(time_start, num_t_basis_funct):
            
            # skip loops if we only consider diagonal elements
            if only_q_s_hat_equal:
                if x_red_size != t_red_size:
                    continue
                    
            print(f"Computing errors for x_red_size={x_red_size}, t_red_size={t_red_size}")
            
            
            #####################
            # compute Sigma_Y_S #
            #####################
            
            X_space_red, V_q, S = helper.space_reduction(X, x_red_size, grid, settings)
            X_space_time_red, U_s, S_space_red = helper.time_reduction(X_space_red, t_red_size, grid, settings)
            
            # compute sigma-value
            sigma_Y_S_val = np.sqrt(sum(np.square(S[x_red_size-1 + 1:]))) + np.sqrt(sum(np.square(S_space_red[t_red_size-1:])))
            
            # store values
            # start with the case where reduced dimensions in space and time are equal
            if x_red_size == t_red_size:
                sigma_Y_S_diag_vals.append(sigma_Y_S_val)
                
            sigma_Y_S[x_red_size - space_start, t_red_size-time_start] = sigma_Y_S_val
            
            
            ##############################################
            # compute the projection error for P = P_Y_S #
            ##############################################
            
            proj_error_Y_S_val = np.linalg.norm(L_y.T @ (X - X_space_time_red) @ L_s, ord="fro")
            
            if x_red_size == t_red_size:
                proj_error_Y_S_diag_vals.append(proj_error_Y_S_val)
            
            proj_error_Y_S[x_red_size - space_start, t_red_size-time_start] = proj_error_Y_S_val
            
            
            ########################################
            # compute the true error for P = P_Y_S #
            ########################################
            
            red_sys_mat, f_SY_red = helper.get_rom(V_q, U_s, grid, settings)

            v = np.linalg.solve(red_sys_mat, f_SY_red)
            
            red_sol_coeffs = (np.kron(L_s_inv.T @ U_s, L_y_inv.T @ V_q)@ v).reshape((num_t_basis_funct, num_x_basis_funct)).T

            red_sol= np.zeros((num_x_nodes, num_t_nodes))
            red_sol[1:-1,:] = red_sol_coeffs
            
            true_error_Y_S_val = np.linalg.norm(L_y.T @ (X - red_sol_coeffs) @ L_s, ord="fro")
            
            # store errors
            if x_red_size == t_red_size:
                true_error_Y_S_diag_vals.append(true_error_Y_S_val)
                
                # plot error if wanted
                if x_red_size in show_error_plot_for:
                    helper.plot_heatmap_error_log(np.abs(Y-red_sol), grid, f"circle_y_red_diff_{x_red_size}.pdf" if save_plots else "")
                
            true_error_Y_S[x_red_size - space_start, t_red_size-time_start] = true_error_Y_S_val
            
            print("FOM-ROM error: " + str(true_error_Y_S_val))
            
            
            
    
    # store matrices for future runs
    np.savez("circle_data.npz", 
        sigma_Y_S_diag_vals=sigma_Y_S_diag_vals, 
        proj_error_Y_S_diag_vals=proj_error_Y_S_diag_vals, 
        true_error_Y_S_diag_vals=true_error_Y_S_diag_vals, 
        sigma_Y_S=sigma_Y_S,
        proj_error_Y_S=proj_error_Y_S,
        true_error_Y_S=true_error_Y_S)
                
        
            
##########################
# PLOT ALL OF THE ERRORS #
##########################

indicator_lines = 50

# plot all errors if not only diagonal ones required. 
if not only_q_s_hat_equal:
    helper.plot_error_distribution(
        sigma_Y_S, 
        grid, 
        time_start, 
        space_start, 
        "circle_sigma_vals.pdf" if save_plots else "",
        s_line = indicator_lines,
        q_line = indicator_lines
    )
    
    helper.plot_error_distribution(
        proj_error_Y_S, 
        grid, 
        time_start, 
        space_start, 
        "circle_proj_error.pdf" if save_plots else "",
        s_line = indicator_lines,
        q_line = indicator_lines
    )
    
    helper.plot_error_distribution(
        true_error_Y_S, 
        grid, 
        time_start, 
        space_start, 
        "circle_rom_error.pdf" if save_plots else "",
        s_line = indicator_lines,
        q_line = indicator_lines
    )

# lets start with the diagonal entries
plot_start = max(space_start, time_start)
plot_end = min(num_x_basis_funct, num_t_basis_funct)

plt.semilogy(
    range(plot_end - plot_start), 
    sigma_Y_S_diag_vals,
    "-",
    label="$\\Sigma_{\hat{\\mathcal{Y}}\\to \hat{\\mathcal{S}}}$")

plt.semilogy(
    range(plot_end - plot_start), 
    proj_error_Y_S_diag_vals,
    "-.",
    label="$\\Vert x - \\mathcal{P}_{\hat{\\mathcal{Y}}\\to \hat{\\mathscr{S}}} x\\Vert$")
    
plt.semilogy(
    range(plot_end - plot_start),
    true_error_Y_S_diag_vals,
    "--",
    label="$\\Vert x - \\hat x \\Vert_{\\mathcal{S} \\cdot \\mathcal{Y}}$")

plt.xlim((0, plot_end - plot_start - 1))
plt.xticks([0, plot_end - plot_start - 1, indicator_lines - plot_start], [plot_start, plot_end, indicator_lines], fontsize=13)
plt.xlabel('$\\hat q = \\hat s$', fontsize=15)
plt.axvline(x = indicator_lines - plot_start, color = 'black', linewidth=0.5, linestyle="--")

plt.legend(loc='lower left')

if save_plots:
    plt.savefig("circle_comparison_errors.pdf", bbox_inches='tight')
plt.show()
