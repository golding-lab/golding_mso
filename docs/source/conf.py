# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../golding_mso/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GoldingMSO'
copyright = '2025, Jared Casarez'
author = 'Jared Casarez'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'myst_parser']
autodoc_member_order = 'groupwise'

templates_path = ['_templates']
exclude_patterns = []
add_module_names = False
toc_object_entries_show_parents = 'hide'
python_display_short_literal_types = True
autodoc_typehints = 'description'
autodoc_default_options = {'show-inheritance':False}
autodoc_typehints_description_target='documented'
autoclass_content='both'
html_extra_path=['readme_files']
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
