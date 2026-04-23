from settings import *

class settings_pod_simple(settings):
    
    # variables for discretization
    _num_x_elem = 100
    _num_t_elem = 100
    
    _t_start = 0
    _t_end = 1.0
    
    _x_start = 0
    _x_end = 1
    
    # pde parameters
    _nu = 0.4
    
    # functions
    
    def y_0(self, x):
        return self.y_real(0, x)
    
    def f(self, t, x):
        y_t = 10 * (x-1)**3 * x * (np.pi * np.sin(2*t - np.pi * x) * np.sin(np.pi * t - 3 * x) - 2 * np.cos(2*t - np.pi * x) * np.cos(np.pi * t - 3*x))
        y_xx = -10 * (x - 1) * (((24 * x**2 - 30 * x + 6) * np.sin(3 * x - np.pi * t) + ((np.pi**2 + 9) * x**3 + (-2 * np.pi**2 - 18) * x**2 + (np.pi**2 - 3) * x + 6) * np.cos(3 * x - np.pi * t)) * np.sin(np.pi * x - 2 * t) + ((6 * np.pi * x**3 - 12 * np.pi * x**2 + 6 * np.pi * x) * np.sin(3 * x - np.pi * t) + (-8 * np.pi * x**2 + 10 * np.pi * x - 2 * np.pi) * np.cos(3 * x - np.pi * t)) * np.cos(np.pi * x - 2 * t))
        
        return y_t - self._nu * y_xx
        
    def y_real(self, t, x):
        return 10 * np.sin(np.pi * x - 2 * t) * np.cos(np.pi*t-3*x)*x*(x-1)**3