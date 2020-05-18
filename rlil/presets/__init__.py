from rlil.presets.validate_agent import env_validation, trainer_validation
import inspect

__all__ = ["env_validation", "trainer_validation"]


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }