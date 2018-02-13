Documentation for snmachine is generated automatically using
[sphinx](http://www.sphinx-doc.org/en/master/) in conjunction with the
[read_the_docs](https://github.com/rtfd/sphinx_rtd_theme) theme. To update or
modify package documentation, you will need to install both of these components.
This can be done using the pip package manager:

    pip install sphinx
    pip install sphinx_rtd_theme
    
To generate new html documentation, from the `rst_source` directory run

    make gh-pages

In addition to generating a new set of html files, the `gh-pages` command will
also organize the documentation in a way that is compatible with the GitHub web
hosting service. Source code for the documentation can be found in the 
`rst_source` directory. Static images used in the documentation are expected to
be found in `rst_source/images`.

**Note:** There is currently no make file for windows machines - only for
UNIX based operating systems.

**Note:** The Sphinx configuration file `conf.py` is not explicit set up for
other types of documentation (such as pdf). 
