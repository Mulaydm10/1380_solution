from copy import deepcopy

import torch.nn as nn
print("[registry.py] Importing registry.py...")
from mmengine.registry import Registry

print("[registry.py] Defining MODELS registry...")
MODELS = Registry(
    name="models",
    build_func=None,
    parent=None,
    scope=None,
)
print("[registry.py] MODELS registry defined.")

SCHEDULERS = Registry(
    name="schedulers",
    build_func=None,
    parent=None,
    scope=None,
)


def build_module(cfg, registry):
    print(f"[registry.py] build_module called for {cfg.get('type', 'Unknown')}")
    builder = registry
    return builder.build(cfg)
