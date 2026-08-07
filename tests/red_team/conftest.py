"""
Workaround for a stray import in deepteam 1.0.8.

`deepteam/test_case/test_case.py` opens with `from nntplib import
NNTPDataError`. `nntplib` was removed from the stdlib in Python 3.13
(PEP 594) and this project requires >=3.14, so `import deepteam` fails
outright with ModuleNotFoundError.

`NNTPDataError` is never referenced anywhere else in the package — it
reads like an editor auto-import accident upstream — so a stub is enough
to get past the import. 1.0.8 is the latest release on PyPI, so there is
no version to bump to yet.

Every deepteam import in this repo lives under tests/red_team/, so a
conftest here covers all of them: pytest imports this module before
collecting anything in this directory.

DELETE THIS FILE once upstream drops the import (track deepteam >1.0.8).
The guard below is deliberately narrow — if a real `nntplib` is importable,
or deepteam stops needing it, this does nothing.
"""
import importlib.util
import sys
import types

if importlib.util.find_spec("nntplib") is None:
    _stub = types.ModuleType("nntplib")

    class NNTPDataError(Exception):
        """Stand-in for the removed stdlib exception; never raised."""

    _stub.NNTPDataError = NNTPDataError  # type: ignore[attr-defined]
    sys.modules.setdefault("nntplib", _stub)
