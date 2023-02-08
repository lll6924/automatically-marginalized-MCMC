import abc

class Model:

    """
        Base class of a model

        model: a function that declares a model with the same grammars as NumPyro, except that `numpyro.sample` statement is replaced with `primitives.my_sample`.
                Function name `register` is held for special uses in my_sample.

        args: a tuple of parameters that should always send to `model` (for example, covariates).

        kwargs: a dictionary of observations, without which `model` becomes the generative model.

        name: name of the model
    """

    @abc.abstractmethod
    def model(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def args(self):
        pass

    @abc.abstractmethod
    def kwargs(self):
        pass

    @abc.abstractmethod
    def name(self):
        pass
