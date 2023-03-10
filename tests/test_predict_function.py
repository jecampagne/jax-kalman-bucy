from jaxkalm import KalmanFilterStd, KalmanFilterNonEquid
import jax.random as jaxrnd
from timeit import timeit
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

def mean_fn(t,p):
    return p[0] + p[1]*t

def testStd():

    # test simple equidistant steps

    #Random gene
    rng_key = jaxrnd.PRNGKey(42)
    rng_key, rng_key0 = jaxrnd.split(rng_key)

    dT = 0.7
    tMes = jnp.arange(0.,20.,dT)

    par_true = jnp.array([0.1,0.2])
    sigma_obs = 0.3

    yMes = mean_fn(tMes,par_true) + sigma_obs * jaxrnd.normal(rng_key0,shape=tMes.shape)


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

    y_arr = res[:,0]
    x_arr = tMes
    dx = jnp.ones_like(x_arr)
    dy = res[:,1]

    cov_res = result.filtered_covariances()    # get the vector state covariance matrix at each state
    cov_res.shape
    y_res_err = jnp.sqrt(cov_res[:,0,0])

    plt.figure(figsize=(8,8))
    plt.errorbar(tMes,yMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
    plt.errorbar(x_arr,y_arr,yerr=y_res_err,fmt='o', linewidth=2, capsize=0, c='r', label="fit")

    plt.quiver(x_arr, y_arr, dx,dy,color="r")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid();

    plt.savefig("test0.png")


def testNonEquid():

    # test simple equidistant steps

    #Random gene
    rng_key = jaxrnd.PRNGKey(42)
    rng_key, rng_key0 = jaxrnd.split(rng_key)

    dT = 0.7
    tMes = jnp.arange(0.,20.,dT)
    par_true = jnp.array([0.1,0.2])
    sigma_obs = 0.3
    yMes = mean_fn(tMes,par_true) + sigma_obs * jaxrnd.normal(rng_key0,shape=tMes.shape)
    
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

    y_arr = res[:,0]
    x_arr = tMes
    dx = jnp.ones_like(x_arr)
    dy = res[:,1]

    cov_res = result.filtered_covariances()    # get the vector state covariance matrix at each state
    cov_res.shape
    y_res_err = jnp.sqrt(cov_res[:,0,0])

    plt.figure(figsize=(8,8))
    plt.errorbar(tMes,yMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
    plt.errorbar(x_arr,y_arr,yerr=y_res_err,fmt='o', linewidth=2, capsize=0, c='r', label="fit")

    plt.quiver(x_arr, y_arr, dx,dy,color="r")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.grid();

    plt.savefig("test1.png")

if __name__ == '__main__':

    # simple test
    # testStd()

    testNonEquid()
