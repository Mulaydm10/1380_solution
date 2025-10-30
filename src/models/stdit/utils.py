import importlib


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def load_module(name):
    p, m = name.rsplit(".", 1)
    mod = importlib.import_module(p)
    model_cls = getattr(mod, m)
    return model_cls
