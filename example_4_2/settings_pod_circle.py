from settings import *

class settings_pod_circle(settings):
    
    # variables for discretization
    _num_x_elem = 100
    _num_t_elem = 100
    
    _t_start = 0
    _t_end = 1.0
    
    _x_start = 0
    _x_end = 1
    
    # pde parameters
    _nu = 1.0
    
    # functions
    
    def f(self, t, x):
        
        if (x-0.5)**2 + (t-0.5)**2 <= 1/5 and (x-0.5)**2 + (t-0.5)**2 >= 1/8:
            return 5
        else:
            return 0
        
    def y_0(self, x):
        
        return x-x