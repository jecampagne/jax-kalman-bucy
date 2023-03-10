from jaxkalm import KalmanFilterStd, KalmanFilterNonEquid
import jax.random as jaxrnd
from timeit import timeit
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax


import jax.scipy as jsc
from jax.scipy import optimize

def mean_fn(t,p):
    return p[0] + p[1]*t

def lik(p,xi,yi, sigma_obs):
    resid = mean_fn(xi, p)-yi
    return 0.5*jnp.sum((resid/sigma_obs) ** 2) 

def plot(tMes,yMes,sigmaMes,result,fname="test.png",y_fit=None):

    res = result.filtered_means()             # get the vector state at each step

    y_arr = res[:,0]
    x_arr = tMes
    dx = jnp.ones_like(x_arr)
    dy = res[:,1]

    cov_res = result.filtered_covariances()    # get the vector state covariance matrix at each state
    cov_res.shape
    y_res_err = jnp.sqrt(cov_res[:,0,0])

    
    plt.figure(figsize=(10,8))
    plt.errorbar(tMes,yMes,yerr=sigmaMes,fmt='o', linewidth=2, capsize=0, c='k', label="data")
    plt.errorbar(x_arr,y_arr,yerr=y_res_err,fmt='o', linewidth=2, capsize=0, c='r', label="Kalman")

    plt.quiver(x_arr, y_arr, dx,dy,color="r",
               angles = 'xy',
               scale_units='xy',
               scale=2, width=0.002)
    if y_fit is not None:
        plt.plot(tMes, y_fit, label="fit MSE")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid();

    plt.savefig(fname)


def testStd():

    # test simple equidistant steps, fixed transport state matrix and transport covariance matrix

    #Random gene
    rng_key = jaxrnd.PRNGKey(42)
    rng_key, rng_key0 = jaxrnd.split(rng_key)

    dT = 0.7
    tMes = jnp.arange(0.,20.,dT)

    par_true = jnp.array([0.1,0.2])
    sigma_obs = 0.3

    yMes = mean_fn(tMes,par_true) + sigma_obs * jaxrnd.normal(rng_key0,shape=tMes.shape)


    #BFGS mini
    bfgs_fit= optimize.minimize(lik, jnp.array([0.,0.]),
                               args=(tMes,yMes,sigma_obs), method='BFGS', tol=1e-6, options=None)
    p_fit = bfgs_fit.x
    y_fit = mean_fn(tMes, p_fit)

    #
    # The vector state is  X=(y,dy/dx)^T
    #

    trans_mat = jnp.array([[1., dT],[0.,1.]])  # transport matrix k-1 -> k of state vector  (the lib assume regular steps: dT=cte)
    trans_cov = jnp.zeros_like(trans_mat)      # no perturbation during transport

    obs_mat = jnp.array([1.0,0.])              # projection matrix from mesurement to vector state (the measure is y_mes only)
    obs_cov = sigma_obs**2                     # error (here 1D) on mesurement

    kf = KalmanFilterStd(trans_mat, trans_cov, obs_mat, obs_cov) # prepare Kalman filter

    result = kf.filter(yMes)                  # iterate the kalman filtering automatic init.

    res = result.filtered_means()             # get the vector state at each step

    plot(tMes,yMes,sigma_obs,result,fname="test0.png", y_fit=y_fit)

def testNonEquid1():

    # test simple equidistant steps in the framework of non equi. steps

    #Random gene
    rng_key = jaxrnd.PRNGKey(42)
    rng_key, rng_key0 = jaxrnd.split(rng_key)

    dT = 0.7
    tMes = jnp.arange(0.,20.,dT)
    par_true = jnp.array([0.1,0.2])
    sigma_obs = 0.3
    yMes = mean_fn(tMes,par_true) + sigma_obs * jaxrnd.normal(rng_key0,shape=tMes.shape)
    
    #BFGS mini
    bfgs_fit= optimize.minimize(lik, jnp.array([0.,0.]),
                               args=(tMes,yMes,sigma_obs), method='BFGS', tol=1e-6, options=None)
    p_fit = bfgs_fit.x
    y_fit = mean_fn(tMes, p_fit)


    #
    # The vector state is  X=(y,dy/dx)^T
    #

    # transport matrix k-1 -> k of state vector
    def trans_mat_func(dt: float) -> jnp.array:
        return  jnp.array([[1., dt],[0.,1.]]) 

    def trans_cov_func(dt: float) -> jnp.array:
        return jnp.zeros(shape=(2,2))


    obs_mat = jnp.array([1.0,0.])              # projection matrix from mesurement to vector state (the measure is y_mes only)
    obs_cov = sigma_obs**2                     # error (here 1D) on mesurement

    kf = KalmanFilterNonEquid(trans_mat_func, trans_cov_func,
                              obs_mat, obs_cov) # prepare Kalman filter


    result = kf.filter(yMes,tMes)             # iterate the kalman filtering automatic init.

    res = result.filtered_means()             # get the vector state at each step


    plot(tMes,yMes,sigma_obs,result,fname="test1.png", y_fit=y_fit)


def testNonEquid2():

    # test simple non equi. steps

    #Random gene
    rng_key = jaxrnd.PRNGKey(10)
    rng_key, rng_key0 = jaxrnd.split(rng_key)

    tMes = jax.random.uniform(rng_key0,minval=0.,maxval=20.0,shape=(30,))
    tMes=jnp.sort(tMes)

    par_true = jnp.array([0.1,0.2])
    sigma_obs = 0.3
    yMes = mean_fn(tMes,par_true) + sigma_obs * jaxrnd.normal(rng_key0,shape=tMes.shape)

    #BFGS mini
    bfgs_fit= optimize.minimize(lik, jnp.array([0.,0.]),
                               args=(tMes,yMes,sigma_obs), method='BFGS', tol=1e-6, options=None)
    p_fit = bfgs_fit.x
    y_fit = mean_fn(tMes, p_fit)
    
    #
    # The vector state is  X=(y,dy/dx)^T
    #

    # transport matrix k-1 -> k of state vector
    def trans_mat_func(dt: float) -> jnp.array:
        return  jnp.array([[1., dt],[0.,1.]]) 

    def trans_cov_func(dt: float) -> jnp.array:
        return jnp.zeros(shape=(2,2))


    obs_mat = jnp.array([1.0,0.])              # projection matrix from mesurement to vector state (the measure is y_mes only)
    obs_cov = sigma_obs**2                     # error (here 1D) on mesurement

    kf = KalmanFilterNonEquid(trans_mat_func, trans_cov_func,
                              obs_mat, obs_cov) # prepare Kalman filter


    result = kf.filter(yMes,tMes)             # iterate the kalman filtering automatic init.

    plot(tMes,yMes,sigma_obs,result,fname="test2.png", y_fit=y_fit)



if __name__ == '__main__':

    # test simple equidistant steps
    testStd()

    # test simple equidistant steps in the framework of non equi. steps
    testNonEquid1()

    # test simple non equi. steps
    testNonEquid2()
