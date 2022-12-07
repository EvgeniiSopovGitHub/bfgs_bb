import numpy as np


def problem_1(x, dim):
    """
        min: f(0,0,...,0) = 0
    """
    return np.sum(x**2)

def problem_2(x, dim):
    """
        min: f(1,1) = 0
    """
    return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2



def get_derivative_approximation(x, f_x, dim, func, delta=0.0001):
    drv = np.zeros(dim)    
    for i in range(dim):
        x_dx = x.copy()
        x_dx[i] = x_dx[i] + delta
        df = func(x_dx, dim) - f_x
        drv[i] = df / delta;
    return drv

def get_alpha(x, f, dim, drv, direction, objective, verbose=0):
    a = 1
    a_min = 0
    a_max = np.inf
    iter_count = 0
    c1 = 0.001
    c2 = 0.999
    FEs = 0
    while iter_count<10: 
        f_new = objective((x+a*direction), dim)
        FEs += 1
        f_cond_1 = f + c1*a*np.dot(drv, direction)
        if f_new > f_cond_1:
            a_max = a
            a = (a_min + a_max) / 2
        else:
            drv_new = get_derivative_approximation((x+a*direction),f_new, dim, objective)
            FEs += dim
            f_cond2_left = np.dot(drv_new, direction)
            f_cond2_right = c2*np.dot(drv, direction)
            if f_cond2_left < f_cond2_right:
                a_min = a
                if a_max == np.inf:
                    a = 2*a
                else:
                    a = (a_min + a_max) / 2
            else:
                if verbose != 0:
                    print("Wolfe conditions are satisfied.")
                return a, FEs
    if verbose != 0:
        print("line search. 10 iterations done.")
    return a, FEs

def bfgs_derivative_free(objective, dim, x_0, epsilon_stop=0.0001, max_iteration_stop=100, verbose=0):

    count_FEs = 0
    
    # initial point, value and Hessian
    x = x_0
    f = objective(x, dim)
    count_FEs += 1
    drv = get_derivative_approximation(x, f, dim, objective)
    count_FEs += dim
    H = np.eye(dim)
    I = np.eye(dim)

    if verbose != 0:
        print("## step = 0", "FEs =", count_FEs)
        print("x =", x, "f =", f)


    for i in range(max_iteration_stop):
        
        drv_norm = np.sqrt(np.sum(drv**2))
        if drv_norm <= epsilon_stop:
            if verbose != 0:
                print("Accuracy is reached!")
            break

        # get direction and step along it
        direction_p = -1*np.dot(H, drv)
        # use line search
        alpha, _fes = get_alpha(x, f, dim, drv, direction_p, objective, verbose=verbose)
        count_FEs += _fes

        # get new point,value and its derivative
        x_k = x + alpha*direction_p
        f_k = objective(x_k, dim)
        count_FEs += 1
        drv_k = get_derivative_approximation(x_k, f_k, dim, objective)
        count_FEs += dim
        
        # get changes and update Hessian
        s_k = x_k - x
        y_k = drv_k - drv
        k = 1 / np.dot(y_k, s_k)
        A1 = I - k*np.outer(s_k, y_k)
        A2 = I - k*np.outer(y_k, s_k)
        A3 = k*np.outer(s_k, s_k)
        H_k = np.dot(A1, np.dot(H, A2)) + A3

        # substitute
        x = x_k
        f = f_k
        drv = drv_k
        H = H_k
        
        # verbose
        if verbose != 0:
            print("## step =", (i+1), "FEs =", count_FEs)
            print("x =", x, "f =", f)
    
    return f, x, count_FEs


if __name__ == "__main__":
    
    n = 2
    my_objective = problem_2
    x_0 = np.array([-1.5, 2.5])
    # x_0 = np.random.uniform(5,10, n)

    f_min, x_min, FEs = bfgs_derivative_free(my_objective, n, x_0, verbose=1)
    print("\nThe results:")
    print("f_min =", f_min)
    print("x_min =", x_min)
    print("FEs =", FEs)
