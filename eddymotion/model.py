"""A factory class that adapts DIPY's dMRI models."""
from os import cpu_count
import warnings
from concurrent.futures import ThreadPoolExecutor
import asyncio
import nest_asyncio

import numpy as np
from dipy.core.gradients import gradient_table

nest_asyncio.apply()


class ModelFactory:
    """A factory for instantiating diffusion models."""

    @staticmethod
    def init(gtab, model="DTI", **kwargs):
        """
        Instatiate a diffusion model.

        Parameters
        ----------
        gtab : :obj:`numpy.ndarray`
            An array representing the gradient table in RAS+B format.
        model : :obj:`str`
            Diffusion model.
            Options: ``"3DShore"``, ``"SFM"``, ``"DTI"``, ``"DKI"``, ``"S0"``

        Return
        ------
        model : :obj:`~dipy.reconst.ReconstModel`
            An model object compliant with DIPY's interface.

        """
        if model.lower() in ("s0", "b0"):
            return TrivialB0Model(gtab=gtab, S0=kwargs.pop("S0"))

        # Generate a GradientTable object for DIPY
        gtab = _rasb2dipy(gtab)
        param = {}

        if model.lower().startswith("3dshore"):
            from dipy.reconst.shore import ShoreModel as Model

            param = {
                "radial_order": 6,
                "zeta": 700,
                "lambdaN": 1e-8,
                "lambdaL": 1e-8,
            }

        elif model.lower().startswith("sfm"):
            from eddymotion.utils.model import (
                SFM4HMC as Model,
                ExponentialIsotropicModel,
            )

            param = {
                "isotropic": ExponentialIsotropicModel,
            }

        elif model.lower() in ("dti", "dki"):
            Model = DTIModel if model.lower() == "dti" else DKIModel

        else:
            raise NotImplementedError(f"Unsupported model <{model}>.")

        param.update(kwargs)
        return Model(gtab, **param)


class TrivialB0Model:
    """
    A trivial model that returns a *b=0* map always.

    Implements the interface of :obj:`dipy.reconst.base.ReconstModel`.
    Instead of inheriting from the abstract base, this implementation
    follows type adaptation principles, as it is easier to maintain
    and to read (see https://www.youtube.com/watch?v=3MNVP9-hglc).

    """

    __slots__ = ("_S0",)

    def __init__(self, gtab, S0=None, **kwargs):
        """Implement object initialization."""
        if S0 is None:
            raise ValueError("S0 must be provided")

        self._S0 = S0

    def fit(self, *args, **kwargs):
        """Do nothing."""

    def predict(self, gradient, **kwargs):
        """Return the *b=0* map."""
        return self._S0


class AverageDWModel:
    """A trivial model that returns an average map."""

    __slots__ = ("_data",)

    def __init__(self, gtab, **kwargs):
        """Implement object initialization."""
        return  # do nothing at initialization time

    def fit(self, data, **kwargs):
        """Calculate the average."""
        self._data = data.mean(-1)

    def predict(self, gradient, **kwargs):
        """Return the average map."""
        return self._data


class DTIModel:
    """A wrapper of :obj:`dipy.reconst.dti.TensorModel."""

    __slots__ = ("_model", "_S0", "_mask")

    def __init__(self, gtab, S0=None, mask=None, **kwargs):
        """Instantiate the wrapped tensor model."""
        from dipy.reconst.dti import TensorModel as DipyTensorModel

        n_threads = kwargs.pop("n_threads", 0) or 0
        n_threads = n_threads if n_threads > 0 else cpu_count()

        self._S0 = None
        if S0 is not None:
            self._S0 = np.clip(
                S0.astype("float32") / S0.max(),
                a_min=1e-5,
                a_max=1.0,
            )

        self._mask = mask > 0 if mask is not None else None
        if self._mask is None and self._S0 is not None:
            self._mask = self._S0 > np.percentile(self._S0, 35)

        if self._S0 is not None:
            self._S0 = self._S0[self._mask]

        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "min_signal",
                "return_S0_hat",
                "fit_method",
                "weighting",
                "sigma",
                "jac",
            )
        }
        self._model = [DipyTensorModel(gtab, **kwargs)] * n_threads

    def fit(self, data, **kwargs):
        """Fit the model chunk-by-chunk asynchronously."""
        _nthreads = len(self._model)

        # All-true mask if not available
        if self._mask is None:
            self._mask = np.ones(data.shape[:3], dtype=bool)

        # Apply mask (ensures data is now 2D)
        data = data[self._mask, ...]

        # Split data into chunks of group of slices
        data_chunks = np.array_split(data, _nthreads)

        # Run asyncio tasks in a limited thread pool.
        with ThreadPoolExecutor(max_workers=_nthreads) as executor:
            loop = asyncio.new_event_loop()

            fit_tasks = [
                loop.run_in_executor(
                    executor,
                    _model_fit,
                    model,
                    data,
                )
                for model, data in zip(self._model, data_chunks)
            ]

            try:
                self._model = loop.run_until_complete(asyncio.gather(*fit_tasks))
            finally:
                loop.close()

    @staticmethod
    def _predict_sub(submodel, gradient, S0_chunk, step):
        """Call predict for chunk and return the predicted diffusion signal."""
        return submodel.predict(gradient, S0=S0_chunk, step=step)

    def predict(self, gradient, step=None, **kwargs):
        """Predict asynchronously chunk-by-chunk the diffusion signal."""
        _nthreads = len(self._model)
        S0 = [None] * _nthreads
        if self._S0 is not None:
            S0 = np.array_split(self._S0, _nthreads)

        # Run asyncio tasks in a limited thread pool.
        with ThreadPoolExecutor(max_workers=_nthreads) as executor:
            loop = asyncio.new_event_loop()

            predict_tasks = [
                loop.run_in_executor(
                    executor,
                    self._predict_sub,
                    model,
                    _rasb2dipy(gradient),
                    S0_chunk,
                    step,
                )
                for model, S0_chunk in zip(self._model, S0)
            ]

            try:
                predicted = loop.run_until_complete(asyncio.gather(*predict_tasks))
            finally:
                loop.close()

        predicted = np.squeeze(np.concatenate(predicted, axis=0))
        retval = np.zeros_like(self._mask, dtype="float32")
        retval[self._mask] = predicted
        return retval


class DKIModel:
    """A wrapper of :obj:`dipy.reconst.dki.DiffusionKurtosisModel."""

    __slots__ = ("_model", "_S0", "_mask")

    def __init__(self, gtab, S0=None, mask=None, **kwargs):
        """Instantiate the wrapped tensor model."""
        from dipy.reconst.dki import DiffusionKurtosisModel

        self._S0 = None
        if S0 is not None:
            self._S0 = np.clip(
                S0.astype("float32") / S0.max(),
                a_min=1e-5,
                a_max=1.0,
            )
        self._mask = mask
        if mask is None and S0 is not None:
            self._mask = self._S0 > np.percentile(self._S0, 35)

        if self._mask is not None:
            self._S0 = self._S0[self._mask.astype(bool)]

        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in (
                "min_signal",
                "return_S0_hat",
                "fit_method",
                "weighting",
                "sigma",
                "jac",
            )
        }
        self._model = DiffusionKurtosisModel(gtab, **kwargs)

    def fit(self, data, **kwargs):
        """Clean-up permitted args and kwargs, and call model's fit."""
        self._model = self._model.fit(data[self._mask, ...])

    def predict(self, gradient, **kwargs):
        """Propagate model parameters and call predict."""
        predicted = np.squeeze(
            self._model.predict(
                _rasb2dipy(gradient),
                S0=self._S0,
            )
        )
        if predicted.ndim == 3:
            return predicted

        retval = np.zeros_like(self._mask, dtype="float32")
        retval[self._mask, ...] = predicted
        return retval


def _rasb2dipy(gradient):
    gradient = np.asanyarray(gradient)
    if gradient.ndim == 1:
        if gradient.size != 4:
            raise ValueError("Missing gradient information.")
        gradient = gradient[..., np.newaxis]

    if gradient.shape[0] != 4:
        gradient = gradient.T
    elif gradient.shape == (4, 4):
        print("Warning: make sure gradient information is not transposed!")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        retval = gradient_table(gradient[3, :], gradient[:3, :].T)
    return retval


def _model_fit(model, data):
    return model.fit(data)
