"""DAPKEL: Data Analysis Package for KELpie.

On import, dapkel applies its matplotlib house style through komorebi_mpl, so
plots look consistent out of the box. The plotting functions never touch
rcParams themselves, so whatever style is active when they draw is what you
get - call ``komorebi_mpl.use(...)`` after the imports to override.

To change the default look, edit ``_DEFAULT_STYLE`` below or
``komorebi_mpl/styles/dapkel.mplstyle``. See ``docs/index.md``.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("dapkel")
except PackageNotFoundError:  # running from a source tree, not installed
    __version__ = "0.0.0.dev0"

# Name of the registered komorebi_mpl style applied on import. Rewrite freely.
_DEFAULT_STYLE = "dapkel"

try:
    import komorebi_mpl as _komorebi_mpl

    _komorebi_mpl.apply_default(_DEFAULT_STYLE)
except Exception:
    # Styling is optional and must never block importing the analysis code
    # (e.g. komorebi_mpl not installed, or the style name not yet registered).
    pass
