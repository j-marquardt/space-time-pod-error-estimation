import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm

from matplotlib.ticker import LinearLocator
import sys

from matplotlib.colors import LogNorm

import numpy as np

from grid import grid

class helper:
    """
    Class which contains only static functions and variables, 
    which can perform the tasks required for i.e. the fe discretization,
    error estimation or plotting
    """
    
    ###########################
    # preparation of Gramians #
    ###########################
    
    
    M_Y = None
    M_S = None
    
    y_stiff = None
    s_stiff = None
    
    s_semi_stiff = None
    
    L_y = None
    L_s = None
    
    L_y_inv = None
    L_s_inv = None
    
    J_y = None
    J_s = None
    
    
    
    ###################################################
    # everything needed for gauss quadrature with n=4 #
    ###################################################
    
    # gauss weights for numerical integration
    gauss_weights = [
        0.347854845137454, 0.652145154862546, 0.652145154862546, 0.347854845137454
    ]
    
    # corresponding nodes for numerical integration
    gauss_nodes = [
        -0.861136311594053, -0.339981043584856, 0.339981043584856, 0.861136311594053
    ]
    
    ################################################
    # methods which help to create and plot a grid #
    ################################################
    
    
    def grid_from_settings(settings):
        """
        Creates a grid which fits the params given with the settings
        """
        
        x_min, x_max = settings.get_bounds_x()
        t_min, t_max = settings.get_bounds_t()
        
        num_x_elem = settings.get_num_x_elem()
        num_t_elem = settings.get_num_t_elem()
        
        return grid(x_min, x_max, t_min, t_max, num_x_elem, num_t_elem)
        
    def grid_from_settings_const_t_steps(settings, steps = 0):
        """
        Creates a grid which fits the params given with the settings
        """
        
        if steps == 0:
            steps = settings.get_uni_t_steps()
        
        x_min, x_max = settings.get_bounds_x()
        t_min, t_max = settings.get_bounds_t()
        
        num_x_elem = settings.get_num_x_elem()
        
        return grid(x_min, x_max, t_min, t_max, num_x_elem, steps)
        
    def reduced_const_grid_from_settings(settings, steps = 0):
        """
        Creates a grid which fits the params given with the settings
        for the reduced order model
        """
        
        if steps == 0:
            steps = settings.get_uni_t_steps()
        
        x_min, x_max = settings.get_bounds_x()
        t_min, t_max = settings.get_bounds_t()
        
        num_x_elem = settings.get_reduced_basis_size() - 1
        
        return grid(x_min, x_max, t_min, t_max, num_x_elem, steps)
        
        
    def plot_grid_from_settings(settings):
        """
        Creates a grid which fits the params given with the settings
        with the differing number of plot elements from the settings.
        Requires the setting to define the number of plot elements
        """
        
        x_min, x_max = settings.get_bounds_x()
        t_min, t_max = settings.get_bounds_t()
        
        num_x_elem = settings.get_num_x_elem()
        num_t_elem = settings.get_num_t_plot_elem()
        
        return grid(x_min, x_max, t_min, t_max, num_x_elem, num_t_elem)
    
    
    def get_mass_y_matrix(grid, settings):
        """
        Initialises M_Y if not already done and returns M_Y (afterwards)
        """
        
        if helper.M_Y is None:
            
            node_num = grid.get_num_x_nodes()
            num_x_basis_funct = node_num - 2
            
            x_min, x_max = settings.get_bounds_x()
            
            dx = abs(x_max - x_min) / (node_num - 1)
         
            M = np.diag(np.ones(num_x_basis_funct-1), -1) + np.diag(4*np.ones(num_x_basis_funct), 0) + np.diag(np.ones(num_x_basis_funct-1), 1)
            helper.M_Y = M * dx/6.0
            
        return helper.M_Y
        
    def get_mass_s_matrix(grid, settings):
        """
        Initialises M_S if not already done and returns M_S (afterwards)
        """
        
        
        if helper.M_S is None:
            
            node_num = grid.get_num_t_nodes()
            num_t_basis_funct = node_num
            
            t_min, t_max = settings.get_bounds_t()
            
            dt = abs(t_max - t_min) / (node_num - 1)
         
            M = np.diag(np.ones(num_t_basis_funct-1), -1) + np.diag(4*np.ones(num_t_basis_funct), 0) + np.diag(np.ones(num_t_basis_funct-1), 1)
            M[0,0] = 2
            M[-1,-1] = 2
            helper.M_S = M * dt/6.0
            
        return helper.M_S
        
    def get_semi_s_stiff(grid, settings):
        """
        Initialises dM_S if not already done and returns dM_S (afterwards)
        """
        
        
        if helper.s_semi_stiff is None:
            
            node_num = grid.get_num_t_nodes()
            num_t_basis_funct = node_num
            
            t_min, t_max = settings.get_bounds_t()
            
            dt = abs(t_max - t_min) / (node_num - 1)
            
            S = np.diag(-0.5 * np.ones(num_t_basis_funct-1), -1) + np.diag(0*np.ones(num_t_basis_funct), 0) + np.diag(0.5*np.ones(num_t_basis_funct-1), 1)
            S[0,0] = -0.5
            S[-1,-1] = 0.5
            
            helper.s_semi_stiff = S
        
        return helper.s_semi_stiff
        
        
    def get_y_stiff(grid, settings):
        """
        Initialises K_Y if not already done and returns K_Y (afterwards)
        """
        
        
        if helper.y_stiff is None:
            
            node_num = grid.get_num_x_nodes()
            num_x_basis_funct = node_num - 2
            
            x_min, x_max = settings.get_bounds_x()
            
            dx = abs(x_max - x_min) / (node_num - 1)
        
            S = np.diag(-np.ones(num_x_basis_funct-1), -1) + np.diag(2*np.ones(num_x_basis_funct), 0) + np.diag(-np.ones(num_x_basis_funct-1), 1)
            helper.y_stiff = S * 1/dx
            
        return helper.y_stiff
        
    def get_s_stiff(grid, settings):
        """
        Initialises K_S if not already done and returns K_S (afterwards)
        """
        
        if helper.s_stiff is None:
            
            node_num = grid.get_num_t_nodes()
            num_t_basis_funct = node_num
            
            t_min, t_max = settings.get_bounds_t()
            
            dt = abs(t_max - t_min) / (node_num - 1)
        
            S = np.diag(-np.ones(num_t_basis_funct-1), -1) + np.diag(2*np.ones(num_t_basis_funct), 0) + np.diag(-np.ones(num_t_basis_funct-1), 1)
            helper.s_stiff = S * 1/dt
            
        return helper.s_stiff
        
    def get_mass_matrix_factors(grid, settings):
        """
        Computes Cholesky factors of the mass matrices, i.e. L_S, L_Y if not already done and returns them.
        The Cholesky triangular factors are upper triangular matrices
        """
        
        
        if helper.L_s is None:
            
            helper.L_s = np.linalg.cholesky(helper.get_mass_s_matrix(grid, settings), upper=True)
            
            
        if helper.L_y is None:
            
            helper.L_y = np.linalg.cholesky(helper.get_mass_y_matrix(grid, settings), upper=True)
            
            
        return helper.L_s, helper.L_y
        
    def get_mass_matrix_inv_factors(grid, settings):
        """
        Computes the inverses of the Cholesky factors L_S, L_Y if not already done and returns them.
        """
        
        if helper.L_s_inv is None or helper.L_y_inv is None:
            
            L_s, L_y = helper.get_mass_matrix_factors(grid, settings)
            
            helper.L_s_inv = np.linalg.inv(L_s)
            helper.L_y_inv = np.linalg.inv(L_y)
            
            
        return helper.L_s_inv, helper.L_y_inv
        
    def get_stiff_matrix_factors(grid, settings):
        """
        Computes Cholesky factors of the stiffness matrices, i.e. J_S, J_Y if not already done and returns them.
        The Cholesky triangular factors are upper triangular matrices
        """
        
        if helper.J_s is None:
            
            helper.J_s = np.linalg.cholesky(helper.get_s_stiff(grid, settings), upper=True)
            
            
        if helper.J_y is None:
            
            helper.J_y = np.linalg.cholesky(helper.get_y_stiff(grid, settings), upper=True)
            
            
        return helper.J_s, helper.J_y
        
        
    ###############################
    # methods fot solving the FOM #
    ###############################
    
    def get_fom(grid, settings):
        """
        Returns the system matrix and the right hand side of the full order model, 
        which are constructed with the Kronecker product space finite elements.
        """
        
        x_space = grid.get_x_space()
        
        # number of nodes and basis functions
        num_x_nodes = grid.get_num_x_nodes()
        num_t_nodes = grid.get_num_t_nodes()

        num_x_basis_funct = num_x_nodes - 2
        num_t_basis_funct = num_t_nodes
        
        M_Y = helper.get_mass_y_matrix(grid, settings)
        M_S = helper.get_mass_s_matrix(grid, settings)
        
        dM_S = helper.get_semi_s_stiff(grid, settings)
        K_Y = helper.get_y_stiff(grid, settings)
        
        # get the system matrix
        sys_mat = np.kron(dM_S, M_Y) + settings.get_nu() * np.kron(M_S, K_Y)
        
        f_SY = helper.get_fom_rhs(num_t_basis_funct, num_x_basis_funct, grid, settings)
        
        sys_mat[:num_x_basis_funct, :] = np.eye(num_x_basis_funct, num_x_basis_funct*num_t_basis_funct)
        
        initial = np.zeros((num_x_basis_funct, 1))
        for i in range(num_x_basis_funct):
            initial[i, 0] = settings.y_0(x_space[i + 1])
        
        f_SY[:num_x_basis_funct] = initial
        
        return sys_mat, f_SY
        
    def get_fom_rhs(num_t_basis_funct, num_x_basis_funct, grid, settings):
        """
        Returns the right hand side of the full order model system for the product of the basis functions
        with number (num_t_basis_funct,num_x_basis_funct) given number of basis functions. 
        In order to solve the integrals Gaussian quadrature is used.
        """
        
        f_SY = np.zeros((num_x_basis_funct * num_t_basis_funct, 1))
        
        x_space = grid.get_x_space()
        t_space = grid.get_t_space()
        
        dx = x_space[1] - x_space[0]
        dt = t_space[1] - t_space[0]

        for i in range(num_t_basis_funct):
            for j in range(num_x_basis_funct):
                
                x_prev, x_curr, x_next = x_space[j], x_space[j+1], x_space[j+2]
                
                if i > 0 and i < num_t_basis_funct - 1:
                    t_prev, t_curr, t_next =t_space[i-1], t_space[i], t_space[i+1]
                    
                    # split up the cases in the two basis functions
                    func_1_1 = lambda t, x: settings.f(t,x) * (t - t_prev) / dt * (x - x_prev) / dx
                    func_1_2 = lambda t, x: settings.f(t,x) * (t - t_prev) / dt * (x_next - x) / dx
                    func_2_1 = lambda t, x: settings.f(t,x) * (t_next - t) / dt * (x - x_prev) / dx
                    func_2_2 = lambda t, x: settings.f(t,x) * (t_next - t) / dt * (x_next - x) / dx
                        
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_prev, t_curr, x_prev, x_curr, func_1_1)
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_prev, t_curr, x_curr, x_next, func_1_2)
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_curr, t_next, x_prev, x_curr, func_2_1)
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_curr, t_next, x_curr, x_next, func_2_2)
                    
                elif i == 0:
                    t_curr, t_next = t_space[0], t_space[1]
                    
                    func_2_1 = lambda t, x: settings.f(t,x) * (t_next - t) / dt * (x - x_prev) / dx
                    func_2_2 = lambda t, x: settings.f(t,x) * (t_next - t) / dt * (x_next - x) / dx
                    
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_curr, t_next, x_prev, x_curr, func_2_1)
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_curr, t_next, x_curr, x_next, func_2_2)
                    
                elif i == num_t_basis_funct - 1:
                    t_prev, t_curr = t_space[num_t_basis_funct - 2], t_space[num_t_basis_funct - 1]
                    
                    func_1_1 = lambda t, x: settings.f(t,x) * (t - t_prev) / dt * (x - x_prev) / dx
                    func_1_2 = lambda t, x: settings.f(t,x) * (t - t_prev) / dt * (x_next - x) / dx
                    
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_prev, t_curr, x_prev, x_curr, func_1_1)
                    f_SY[j + num_x_basis_funct * i] += helper.space_time_gauss_quadrature(t_prev, t_curr, x_curr, x_next, func_1_2)
        
        return f_SY
        
    def space_time_gauss_quadrature(at, bt, ax, bx, func):
        """
        Uses Gaussian quadrature with n=4 in order to integrate 'func' over a
        domain in space time with the bounds 'at', 'bt', 'ax' and 'bx'.
        """
        
        # compute two factors which we need during Guass integration
        dx2 = (bx - ax) / 2
        dt2 = (bt - at) / 2
        
        mx2 = (ax + bx) / 2
        mt2 = (at + bt) / 2
        
        # get precomputed weights and node positions for gauss integration
        w = helper.gauss_weights
        nodes = helper.gauss_nodes
        
        # start with the integration
        int_val = 0
        
        for i in range(len(w)):
            for j in range(len(w)):
                val = w[i] * w[j]
                val *= func(dt2 * nodes[i] + mt2, dx2 * nodes[j] + mx2)
                
                int_val += val
                
        return dx2 * dt2 * int_val
        
        
    #####################################
    # methods for model order reduction #
    #####################################
    
    def space_reduction(X, q_hat, grid, settings):
        """
        Executes the spatial reduction for given measurements X and reduced dimension 'q_hat'.
        Returns the reduced measurement matrix, as well as V_q and the corersponding singular values
        from the SVD
        """
        
        L_s, L_y = helper.get_mass_matrix_factors(grid, settings)
        _, L_y_inv = helper.get_mass_matrix_inv_factors(grid, settings)
        
        # singular value decomposition for reduction 
        left, S, _ = np.linalg.svd(L_y.T @ X @ L_s)

        V_q = left[:, :q_hat]
        
        red_space_mat = L_y_inv.T @ V_q @ V_q.T @ L_y.T @ X
        
        return red_space_mat, V_q, S
       
        
    def time_reduction(X, s_hat, grid, settings):
        """
        Executes the temporal reduction for given measurements X and reduced dimension 's_hat'.
        Returns the reduced measurement matrix, as well as U_s and the corersponding singular values
        from the SVD
        """
        
        L_s, L_y = helper.get_mass_matrix_factors(grid, settings)
        
        L_s_inv, _ = helper.get_mass_matrix_inv_factors(grid, settings)
        
        X_0 = X.copy()
        X_0[:,0] = np.zeros(X_0.shape[0])

        _, S, right_T = np.linalg.svd(L_y.T @ X_0 @ L_s)


        U_s_0 = right_T.T[:, :s_hat - 1]
        
        U_s = np.block([[L_s[0,:].reshape((U_s_0.shape[0], 1)), U_s_0]])

        # coefficient matrix for the space-time-projected solution
        red_time_mat = X_0 @ L_s @ U_s_0 @ U_s_0.T @ L_s_inv + (X - X_0)
        
        return red_time_mat, U_s, S
        
        
    def get_rom(V_q, U_s, grid, settings):
        
        """
        Returns the system matrix and the right hand side of the reduced order model, 
        which are constructed with the Kronecker product space finite elements.
        Requires the matrices V_q and U_s from the reduction
        """
        
        L_s, L_y = helper.get_mass_matrix_factors(grid, settings)
        
        L_s_inv, L_y_inv = helper.get_mass_matrix_inv_factors(grid, settings)
        
        dM_S = helper.get_semi_s_stiff(grid, settings)
        K_Y = helper.get_y_stiff(grid, settings)
        
        # compute reduced gramians
        M_S_red =  U_s.T @ U_s 

        M_Y_red = V_q.T @ V_q

        d_M_S_red = U_s.T @ L_s_inv @ dM_S @ L_s_inv.T @ U_s
        K_Y_red = V_q.T @ L_y_inv @ K_Y @ L_y_inv.T @ V_q

        sys_mat = np.kron(d_M_S_red, M_Y_red) + settings.get_nu() * np.kron(M_S_red, K_Y_red)
        
        num_x_nodes = grid.get_num_x_nodes()
        num_t_nodes = grid.get_num_t_nodes()

        num_x_basis_funct = num_x_nodes - 2
        num_t_basis_funct = num_t_nodes
        
        f_SY = helper.get_fom_rhs(num_t_basis_funct, num_x_basis_funct, grid, settings)
        

        f_SY_red = np.kron(U_s.T @ L_s_inv, V_q.T @ L_y_inv) @ f_SY

        # in order to compute the initial condition, we have to project onto the subspace.
        # We first compute the coefficients for the orthogonal projection 
        # solve system M_y_red @ c = b. to do so
        
        x_space = grid.get_x_space()
        
        initial = np.zeros((num_x_basis_funct, 1))
        for i in range(num_x_basis_funct):
            initial[i, 0] = settings.y_0(x_space[i + 1])


        c = np.linalg.solve(M_Y_red, V_q.T @ L_y.T @ initial)

        # integrate it into the system
        for i in range(c.shape[0]):
            sys_mat[i, :] = np.zeros(sys_mat.shape[1])
            sys_mat[i,i] = 1

        f_SY_red[:c.shape[0], 0] = c.flatten()
        
        return sys_mat, f_SY_red
        
    
    #############################
    # methods used for plotting #
    #############################
    
    def plot_heatmap_error_log(mat, grid, save_name=""):
        """
        Plots the error for the function on the space time domain scaled logarithmicly.
        If 'save_name' is empty the plot is just shown but not saved. Otherwise, it is also saved
        """
        
        fig, ax = plt.subplots()
    
        c = plt.imshow(mat, 
            origin='lower',
            norm=LogNorm(),
            interpolation='nearest',
            cmap='autumn')
        plt.xlabel('$\\tau$', fontsize=15)
        plt.ylabel('$\\xi$', fontsize=15, rotation="horizontal")
    
        plt.xticks([0, int(grid.get_num_x_nodes() / 2), grid.get_num_x_nodes() - 1], [0, 0.5, 1.0], fontsize=13)
        plt.yticks([0, int(grid.get_num_t_nodes() / 2), grid.get_num_t_nodes() - 1], [0, 0.5, 1.0], fontsize=13)
        plt.locator_params(nbins=3)
    
        plt.colorbar(c)
        
        # save if name is not empty
        if save_name:
            plt.savefig(save_name, bbox_inches='tight')
        plt.show()
        
    def plot_error_distribution(mat, grid, s_start, q_start, save_name="", s_line=-1, q_line=-1):
        """
        Plots the error for all possible choices of s_hat and q_hat as described in Example 4.2 logarithmicly.
        If 'save_name' is empty the plot is just shown but not saved. Otherwise, it is also saved.
        The position of the indivator lines can be set with 's_line' and 'q_line'. If they are
        smaller than the starting values, no lines are drawn.
        """
        
        num_x = mat.shape[0]
        num_s = mat.shape[1]
        
        fig, ax = plt.subplots()
    
        c = plt.imshow(mat, 
            origin='lower',
            norm=LogNorm(),
            interpolation='nearest',
            cmap='autumn')
        plt.xlabel('$\hat s$', fontsize=15)
        plt.ylabel('$\hat q$', fontsize=15, rotation="horizontal")
        
        s_ticks_x = [0, num_s - 1]
        q_ticks_x = [0, num_x - 1]
        
        s_ticks_y = [s_start, s_start + num_s - 1]
        q_ticks_y = [q_start, q_start + num_x - 1]
        
        # include lines for comparison between plots
        if s_line >= s_start:
            plt.axvline(x = s_line - s_start, color = 'black', linewidth=0.5, linestyle="--")
            
            s_ticks_x.append(s_line - s_start)
            s_ticks_y.append(s_line)
            
        if q_line >= q_start:
            plt.axhline(y = q_line - q_start, color = 'black', linewidth=0.5, linestyle="--")
            
            q_ticks_x.append(q_line - q_start)
            q_ticks_y.append(q_line)
        
        plt.xticks(s_ticks_x, s_ticks_y, fontsize=13)
        plt.yticks(q_ticks_x, q_ticks_y, fontsize=13)
        plt.locator_params(nbins=3)
    
        plt.colorbar(c)
        
        # save if name is not empty
        if save_name:
            plt.savefig(save_name, bbox_inches='tight')
        plt.show()
        
    def plot_function(mat, grid, save_name=""):
        """
        Plots a function on the space time domain.
        If 'save_name' is empty the plot is just shown but not saved. Otherwise, it is also saved.
        """
        
        fig, ax = plt.subplots()
        c= plt.imshow(mat, interpolation='nearest', origin='lower')
        plt.xlabel('$\\tau$', fontsize=15)
        plt.ylabel('$\\xi$', fontsize=15, rotation="horizontal")
    
        plt.xticks([0, int(grid.get_num_x_nodes() / 2), grid.get_num_x_nodes() - 1], [0, 0.5, 1.0], fontsize=13)
        plt.yticks([0, int(grid.get_num_t_nodes() / 2), grid.get_num_t_nodes() - 1], [0, 0.5, 1.0], fontsize=13)
        plt.locator_params(nbins=3)
    
        plt.colorbar(c)
        
        # if savename is not empty
        if save_name:
            plt.savefig(save_name, bbox_inches='tight')
        plt.show()
        
    def singular_value_comparison_plot(measurement_matrix, s_hat, q_hat, grid, settings, save_name=""):
        """
        Creates a singular value comparison plot for a specific choice of the reduced dimension. 
        If 'save_name' is empty the plot is just shown but not saved. Otherwise, it is also saved.
        The position of the indivator lines is determined in dependence on the reduced dimensions.
        """
        
        L_s = np.linalg.cholesky(helper.get_mass_s_matrix(grid, settings), upper=True)
        L_y = np.linalg.cholesky(helper.get_mass_y_matrix(grid, settings), upper=True)
        
        L_s_inv = np.linalg.inv(L_s)
        L_y_inv = np.linalg.inv(L_y)
        
        X_0 = measurement_matrix.copy()
        X = measurement_matrix.copy()

        X_0[:,0] = np.zeros(X_0.shape[0])

        left, _, _ = np.linalg.svd(L_y.T @ X_0 @ L_s)
        V_q = left[:, :q_hat]

        _, S_0, right_T = np.linalg.svd(L_y.T @ X_0 @ L_s)
        _, S, right_T = np.linalg.svd(L_y.T @ X @ L_s)

        U_s_0 = right_T.T[:, :s_hat - 1]
        U_s = np.block([[L_s[0,:].reshape((U_s_0.shape[0], 1)), U_s_0]])

        red_mat = L_y_inv.T @ V_q @ V_q.T @ L_y.T @ X
        red_mat[:,0] = np.zeros(red_mat.shape[0])

        red_mat_0 = X_0 @ L_s @ U_s_0 @ U_s_0.T @ L_s_inv + (X-X_0)

        _, E, _ = np.linalg.svd(L_y.T @ red_mat @ L_s)
        _, E_0, _ = np.linalg.svd(L_y.T @ red_mat_0 @ L_s)

        plt.axvline(x = q_hat, color = 'black', linewidth=0.25)
        plt.text(q_hat + 1, 10e-3, "$\hat q =" + str(q_hat) + "$", fontsize=9)

        plt.axvline(x = s_hat, color = 'black', linewidth=0.25)
        plt.text(s_hat + 1, 10e-3, "$\hat s =" + str(s_hat) + "$", fontsize=9)

        plt.semilogy(range(len(S)), S, label="$L_y^T X L_s$")
        plt.semilogy(range(len(S)), S_0, "--", label="$L_y^T X^0 L_s$")
        plt.semilogy(range(len(E)), E, "-.", label="$L_y^T X^0_{S \cdot \hat{Y}} L_s$")
        plt.semilogy(range(len(E_0)), E_0, ":", label="$L_y^T X_{\hat{S} \cdot Y} L_s$")

        plt.xlim(0,len(S)-1)

        plt.legend()
        
        # if savename is not empty
        if save_name:
            plt.savefig(save_name, bbox_inches='tight')
        plt.show()
        
        
    def visualize_f(grid, settings, save_name=""):
        """
        Plots the right hand side f of the system.
        If 'save_name' is empty the plot is just shown but not saved. Otherwise, it is also saved.
        """
        
        num_x_nodes = grid.get_num_x_nodes()
        num_t_nodes = grid.get_num_t_nodes()
        
        x_space = grid.get_x_space()
        t_space = grid.get_t_space()
    
        f_mat = np.zeros((num_x_nodes, num_t_nodes))
    
        for i in range(num_x_nodes):
            for j in range(num_t_nodes):
                f_mat[i,j] = settings.f(t_space[j], x_space[i])
        
        c = plt.imshow(f_mat, 
            origin='lower',
            interpolation='nearest')
            
        plt.xlabel('$\\tau$', fontsize=15)
        plt.ylabel('$\\xi$', fontsize=15, rotation="horizontal")
        
        plt.xticks([0, int(grid.get_num_x_nodes() / 2), grid.get_num_x_nodes() - 1], [0, 0.5, 1.0], fontsize=13)
        plt.yticks([0, int(grid.get_num_t_nodes() / 2), grid.get_num_t_nodes() - 1], [0, 0.5, 1.0], fontsize=13)
        plt.locator_params(nbins=3)
    
        plt.colorbar(c)
    
        if save_name:
            plt.savefig(save_name, bbox_inches='tight')
            
        plt.title("f")
        plt.show()
        
    
	


                      
        

        
        
        