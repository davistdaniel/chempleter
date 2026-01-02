# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Chempleter'
copyright = '2026, Davis Thomas Daniel'
author = 'Davis Thomas Daniel'
release = '0.1.0b5'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx_design","sphinx_copybutton"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['images']
html_favicon = 'images/chempleter.ico'

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
    ]
}

html_theme_options = {
    'logo': 'chempleter_logo.png',
    'logo_name': False,
    'show_powered_by': False,
    'page_width':'1200px',
    'extra_nav_links' : {"Github": "https://github.com/davistdaniel/chempleter","PyPi":"https://pypi.org/project/chempleter/"},
}
