import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable
import jax.random as jrnd


from .typing import ArrayLike
from .utils import to_array, coerce_covariance, coerce_matrix, predict, correct
from .result import Correction, Prediction, Result


class KalmanFilterStd(object):
    r"""
    Implements a standard [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) which enables [Single Instruction Multiple Data](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) in JAX.
    Inspired by [simdkalman](https://github.com/oseiskar/simdkalman) and [pykalman](https://pykalman.github.io/).
    """

    def __init__(
        self,
        trans_mat: ArrayLike,
        trans_cov: ArrayLike,
        obs_mat: ArrayLike,
        obs_cov: ArrayLike,
        init_mean: ArrayLike = None,
        init_cov: ArrayLike = None,
    ):
        """
        Internal initializer for :class:`KalmanFilter`.

        Args:
            trans_mat (ArrayLike): transition matrix.
            trans_cov (ArrayLike): transition covariance.
            obs_mat (ArrayLike): observation matrix.
            obs_cov (ArrayLike): observation covariance.
            init_mean (ArrayLike, optional): initial mean. Defaults to None.
            init_cov (ArrayLike, optional): initial covariance. Defaults to None.
        """

        trans_mat, trans_cov, obs_mat, obs_cov = to_array(trans_mat, trans_cov, obs_mat, obs_cov)

        self.trans_cov = coerce_covariance(trans_cov)
        self.trans_ndim = self.trans_cov.shape[-1]
        self.trans_mat = coerce_matrix(trans_mat, self.trans_cov.shape[-2], self.trans_cov.shape[-1])

        self.obs_cov = coerce_covariance(obs_cov)
        self.obs_ndim = self.obs_cov.shape[-1]
        self.obs_mat = coerce_matrix(obs_mat, self.obs_cov.shape[-1], self.trans_cov.shape[-1])

        if init_mean is None:
            init_mean = jnp.zeros(self.trans_mat.shape[-1])

        if init_cov is None:
            init_cov = jnp.eye(self.trans_cov.shape[-1])

        self.init_cov = coerce_covariance(init_cov)
        self.init_mean = coerce_matrix(init_mean, 0, self.trans_mat.shape[-1])

    def initialize(self) -> Prediction:
        """
        Generates the initial state.

        Returns:
            Prediction: initial state.
        """

        return Prediction(self.init_mean, self.init_cov)

    def predict(self, correction: Correction) -> Prediction:
        """
        Predicts from the latest correction.

        Args:
            correction (Correction): latest correction.

        Returns:
            Prediction: new prediction.
        """

        mean, cov = predict(correction.mean, correction.covariance, self.trans_mat, self.trans_cov)

        return Prediction(mean, cov)

    def correct(self, y: jnp.ndarray, prediction: Prediction) -> Correction:
        """
        Corrects latest prediction.

        Args:
            y (jnp.ndarray): latest observation.
            prediction (Prediction): latest prediction.

        Returns:
            Correction: corrected Kalman state.
        """

        mean, cov, gain = correct(prediction.mean, self.obs_mat, prediction.covariance, self.obs_cov, y)

        # TODO: Fix log-likelihood
        return Correction(mean, cov, gain, 0.0)

    def filter(self, y: jnp.ndarray) -> Result:
        """
        Filters the data `y`.

        Args:
            y (jnp.ndarray): data to filter, must be of shape `{time, [batch], [dim]}`

        Returns:
            Result: result object.
        """
        
        result = Result()        

        p = self.initialize()
        for yt in y:
            c = self.correct(yt, p)
            result.append(p, c)

            p = self.predict(c)            

        return result

    def sample(
        self, timesteps: int, prng_key: jrnd.PRNGKey, batch_shape: tuple = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Samples the LDS `timesteps` into the future.

        Args:
            timesteps (int): number of future timesteps.
            prng_key (jax.random.PRNGKey): random number generator key.
            batch_shape (tuple): batch shape.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: returns the sampled states for the latent and observable processes.
        """

        x = jrnd.multivariate_normal(prng_key, self.init_mean, self.init_cov, batch_shape)

        x_res = tuple()
        y_res = tuple()

        for t in range(timesteps):
            _, prng_key = jrnd.split(prng_key)
            x = jrnd.multivariate_normal(
                prng_key, (self.trans_mat @ x[..., None]).squeeze(-1), self.trans_cov, method="svd"
            )

            _, prng_key = jrnd.split(prng_key)
            y = jrnd.multivariate_normal(
                prng_key, (self.obs_mat @ x[..., None]).squeeze(-1), self.obs_cov, method="svd"
            )

            x_res += (x,)
            y_res += (y,)

        return jnp.stack(x_res), jnp.stack(y_res)




class KalmanFilterNonEquid(object):
    r"""
    """

    def __init__(
        self,
        trans_mat_func : Callable[[float], ArrayLike],
        trans_cov_func : Callable[[float], ArrayLike],
        obs_mat: ArrayLike,
        obs_cov: ArrayLike,
        init_mean: ArrayLike = None,
        init_cov: ArrayLike = None,
    ):
        """
        Internal initializer for :class:`KalmanFilter`.

        Args:
            trans_mat_func (float->ArrayLike): transition matrix function  for prediction
            trans_cov_func (flaot->ArrayLike): transition covariance function for prediction
            
            obs_mat (ArrayLike): observation matrix.
            obs_cov (ArrayLike): observation covariance.
            init_mean (ArrayLike, optional): initial mean. Defaults to None.
            init_cov (ArrayLike, optional): initial covariance. Defaults to None.
        """
        self.trans_mat_func = trans_mat_func
        self.trans_cov_func = trans_cov_func
        

        obs_mat, obs_cov = to_array(obs_mat, obs_cov)
        trans_mat_dum =  to_array(self.trans_mat_func(0.))

        trans_cov_dum =  to_array(self.trans_cov_func(0.))
        trans_cov_dum = coerce_covariance(trans_cov_dum)

        self.trans_mdim = trans_cov_dum.shape[-2]
        self.trans_ndim = trans_cov_dum.shape[-1]

        trans_mat_dum = coerce_matrix(trans_mat_dum, self.trans_mdim, self.trans_ndim)
        self.obs_cov = coerce_covariance(obs_cov)

        self.obs_ndim = self.obs_cov.shape[-1]
        self.obs_mat = coerce_matrix(obs_mat, self.obs_ndim, self.trans_ndim)


        #just an illustration
        if init_mean is None:
            init_mean = jnp.zeros(trans_mat_dum.shape[-1])

        if init_cov is None:
            init_cov = jnp.eye(self.trans_ndim)

        self.init_cov = coerce_covariance(init_cov)
        self.init_mean = coerce_matrix(init_mean, 0, trans_mat_dum.shape[-1])


    def _initialize(self) -> Prediction:
        """
        Generates the initial state.

        Returns:
            Prediction: initial state.
        """

        return Prediction(self.init_mean, self.init_cov)

    def _predict(self, correction: Correction, dt: float ) -> Prediction:
        """
        Predicts from the latest correction.

        Args:
            correction (Correction): latest correction.

        Returns:
            Prediction: new prediction.
        """
        trans_mat = self.trans_mat_func(dt)
        trans_cov = self.trans_cov_func(dt)
        
        mean, cov = predict(correction.mean, correction.covariance, trans_mat, trans_cov)

        return Prediction(mean, cov)

    def _correct(self, y: jnp.ndarray, prediction: Prediction) -> Correction:
        """
        Corrects latest prediction.

        Args:
            y (jnp.ndarray): latest observation.
            prediction (Prediction): latest prediction.

        Returns:
            Correction: corrected Kalman state.
        """

        mean, cov, gain = correct(prediction.mean, self.obs_mat, prediction.covariance, self.obs_cov, y)

        # TODO: Fix log-likelihood
        return Correction(mean, cov, gain, 0.0)


    #main for the user
    def filter(self, y: jnp.ndarray, t: jnp.array) -> Result:
        """
        Filters the data `y`.

        Args:
            t (jnp.ndarray): `{time}`
            y (jnp.ndarray): data to filter, must be of shape `{time, [batch], [dim]}`

        Returns:
            Result: result object.
        """
        # array of differences t_{i+1}-t_i with the last to 0.
        # same size as y
        dT=jnp.ediff1d(jnp.insert(t, len(t),t[-1], axis=0))

        result = Result()        

        p = self._initialize()
        for yt,dt in zip(y,dT):

            c = self._correct(yt, p)
            result.append(p, c)

            p = self._predict(c,dt)            

        return result


###
KalmanFilter = KalmanFilterStd
