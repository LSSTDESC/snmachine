# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = snmachine
SOURCEDIR     = .
BUILDDIR      = .

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

gh-pages:
	make html

	rm -fr ../_modules
	mv -f html/_modules ../_modules

	rm -fr ../_sources
	mv -f html/_sources ../_sources

	rm -fr ../_static
	mv -f html/_static ../_static
	cp -r images/* ../_static

	rm -rf ../_images
	mv -f html/_images ../_images

	mv -f html/* ../.
	rm -fr html/
	rm -fr doctrees/
