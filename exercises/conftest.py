"""Shared pytest setup for the drills.

Each test module sets MODULE = "<name>" and uses the `impl` fixture, which imports
exercises/drills/<name>.py (your code) by default, or exercises/solutions/<name>.py
when run with --solutions (or LLM_SOLUTIONS=1).
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def pytest_addoption(parser):
    parser.addoption("--solutions", action="store_true", default=False,
                     help="run tests against exercises/solutions instead of exercises/drills")


def _use_solutions(config) -> bool:
    return config.getoption("--solutions") or os.environ.get("LLM_SOLUTIONS") == "1"


@pytest.fixture(scope="module")
def impl(request):
    name = getattr(request.module, "MODULE", None)
    if name is None:
        raise RuntimeError("test module must define MODULE = '<name>'")
    pkg = "solutions" if _use_solutions(request.config) else "drills"
    return importlib.import_module(f"{pkg}.{name}")


@pytest.fixture(autouse=True)
def _seed():
    try:
        import torch
        torch.manual_seed(0)
    except ImportError:
        pass
    import random
    random.seed(0)
