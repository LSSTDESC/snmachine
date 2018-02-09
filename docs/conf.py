#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Specify extensions / features
extensions = ['sphinx.ext.autodoc',      # auto-documentation
              'sphinx.ext.viewcode',     # adds links to source code
              'sphinx.ext.githubpages',  # creates .nojekyll file
              'sphinx.ext.napoleon']     # adds support for google doc style

# The suffixes of source files.
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'snmachine'
copyright = '2018, Michelle Lochner, Robert Schuhmann, Jason McEwen, Hiranya Peiris, Rahul Biswas, Ofer Lahav, Johnny Holland, Max Winter'
author = 'Michelle Lochner, Robert Schuhmann, Jason McEwen, Hiranya Peiris, Rahul Biswas, Ofer Lahav, Johnny Holland, Max Winter'

# The version info for the project
version = '1.0.0'
release = '1.0.0'

# Patterns relative to source directory to ignore
exclude_patterns = ['.build', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# The theme to use for HTML and HTML Help pages.
html_theme = 'sphinx_rtd_theme'

# Paths that contain custom static files
html_static_path = []

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'snmachinedoc'
