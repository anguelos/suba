# Configuration file for the Sphinx documentation builder.
import os, sys
sys.path.insert(0, os.path.abspath("../src"))

project = "suba"
author = "Anguelos Nicolaou"
from suba.version import suba_version
release = suba_version

extensions = [
    "myst_parser",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autoapi_dirs = ["../src"]
autoapi_type = "python"

myst_enable_extensions = ["colon_fence", "deflist"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
html_theme = "alabaster"

# Stop autoapi from injecting 'autoapi/index' into the master toctree at
# build time — that file only exists in the build tree, not the source tree,
# so Sphinx warns about a missing document.  We include the API docs
# explicitly via the toctree in index.md instead.
autoapi_add_toctree_entry = False

# Keep generated autoapi .md files in docs/autoapi/ between builds so that
# Sphinx can resolve toctree references to them as normal source documents.
autoapi_keep_files = True
