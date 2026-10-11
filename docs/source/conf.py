"""Sphinx configuration for the installed package or local source tree."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from oncapintada._version import __version__

project = "Onça-pintada"
author = "Leandro Seixas Rocha"
copyright = "2026, Leandro Seixas Rocha"
release = __version__
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "myst_parser"]
exclude_patterns = []
html_title = project
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {"light_logo": "icon.png", "dark_logo": "icon.png"}
html_favicon = "_static/favicon.png"
html_css_files = ["custom.css"]
html_show_sphinx = False
html_show_copyright = True
html_show_sourcelink = False
