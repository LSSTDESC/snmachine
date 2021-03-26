# Contributing to `snmachine`

The goal of this guide is to help users contribute to `snmachine` as quickly and as easily possible.

# Table of contents
1. [Creating an Issue](#issues)
2. [Submitting a Pull Request](#prs)
3. [Code Style](#style)
4. [Running Tests Locally](#testing)
5. [Package Versioning](#version)

<a name="issues"></a>
## Creating an Issue

To report bugs, request features, or ask questions about the structure of the
code, please [open an issue](https://github.com/LSSTDESC/snmachine/issues)

### Bugs

For program crash, or to report a specific error, please file an issue by
opening a [Bug Report](https://github.com/LSSTDESC/snmachine/issues/new?assignees=&labels=bug&template=BUG.md&title=%5BBUG%5D)

Follow instructions contained in the Github issue template in order to provide
enough detail that the `snmachine` developers can tackle the problem.

### Feature Requests

For feature requests or enhancement suggestions, please file a [Feature Request issue](https://github.com/LSSTDESC/snmachine/issues/new?assignees=&labels=feature&template=feature_request.md&title=%5BFEATURE%5D)

A detailed explanation of what idea is being proposed should be put in the body
of the issue. The title can be appended with `[RFC]` to signify a "Request for
Comments" is desired on how best to incorporate said feature or to discuss at
length `[RFD]` may be used instead as a marker for "Request for Discussion".

### General

If the issue is general, perhaps just a question, or minor fix required, a plain
issue would be appropriate. These can be opened with a [normal issue](https://github.com/LSSTDESC/snmachine/issues/new).

<a name="prs"></a>
## Submitting a Pull Request

We welcome [pull requests](https://help.github.com/articles/about-pull-requests/) from anyone
interested in contributing to `snmachine`. This section describes best practices
for submitting PRs to this project.

If you are interested in making changes that impact the way `snmachine` works,
please [open an issue](#issues) proposing what you would like to work on before
you spend time creating a PR.

To submit a PR, please follow these steps:

0. [Create an issue](#issues) and make note of the issue number
1. Fork `snmachine` to your GitHub account
2. Create a branch from `dev` on your fork with the convention of: `issue/<issue-number>/one-or-two-word-token-description-of-issue`
	- If you are working on a `[FEATURE]`, please branch from `dev` with the
	    convention of: `feauture/<issue-number>/one-or-two-word-token-description-of-issue`, this is to allow for issues to be raised on specific features also, which would then have the convention of:
	    `feature/<original-issue-number>/issue/<new-issue-number>/short-description`(where issues on a particular feature would be branched from the feature-branch in question)

3. Make changes, add code etc and open a PR

Please ensure your branch is up-to-date with `dev` branch by rebasing your changes. If unsure how to approach this, make a comment in the PR that has been opened.

<a name="style"></a>
## Code Style

Much of our code style follows the convention defined in [PEP8](https://pep8.org/), with the exception, in some cases, of [`E501`](https://lintlyci.github.io/Flake8Rules/rules/E501.html).

In addition, codebase guidelines outlined in the [LSST Developer Guide](https://developer.lsst.io/python/style.html)
and in the [LSST DESC Coding Guidelines](https://confluence.slac.stanford.edu/display/LSSTDESC/Interim+LSST+DESC+Paper+Tracking?preview=/217813295/244908471/LSST%20DESC%20Coding%20Guidelines%20v1.1.pdf)
document.

#### Import Conventions

Imports should be grouped in the following order:

1. standard library imports (https://docs.python.org/3/library/)
2. related third party imports (like `numpy`, `pandas` or `snmachine`)
3. local application/library specific imports

There should be a blank line between each group of imports.

### Naming Conventions

* class names -> PascalCase
* function and variable names -> snake_case
* function names -> start with descriptive verb (eg. get, compute, fit, load)
* hidden function names -> same as function names but starting with `_`
* descriptive names that minimize the number of necessary comments
* how to name:
	- directories: `directories`

### Documentation Conventions

#### Functions

They should follow NumPy Style (https://numpydoc.readthedocs.io/en/latest/format.html).

See a complete example in https://docs.scipy.org/doc/numpy-1.15.0/docs/howto_document.html#docstring-standard .
Note that not all sections are mandatory. See a short example edited from the above page:

```
def foo(var1, var2, long_var_name='hi'):
    """A one-line summary that does not use variable names or the
    function name.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    var1 : array_like
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    var2 : int
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    long_var_name : {'hi', 'ho'}, optional
        Choices in brackets, default first when optional.

    Returns
    -------
    type
        Explanation of anonymous return value of type ``type``.
    describe : type
        Explanation of return value named `describe`.

    Raises
    ------
    BadException
        Because you shouldn't have done that.

    Notes
    -----
    Notes about the implementation algorithm (if needed).

    You may include some math:

    .. math:: X(e^{j\omega } ) = x(n)e^{ - j\omega n}

    And even use a greek symbol like :math:`omega` inline.

    References
    ----------
    Cite the relevant literature, e.g. [1]_.  You may also cite these
    references in the notes section above.

    .. [1] O. McNoleg, "The integration of GIS, remote sensing,
       expert systems and adaptive co-kriging for environmental habitat
       modelling of the Highland Haggis using object-oriented, fuzzy-logic
       and neural-network techniques," Computers & Geosciences, vol. 22,
       pp. 585-588, 1996.

    Examples
    --------
    These are written in doctest format, and should illustrate how to
    use the function.

    >>> a = [1, 2, 3]
    >>> print [x + 3 for x in a]
    [4, 5, 6]
    >>> print "a\n\nb"
    a
    b
    """

    pass
```

<a name="testing"></a>
## Running Tests Locally

We use Azure Pipelines to automatically run tests and other
automated checks on every PR commit and merge to `master` and `dev` branches.

However, if you would like to run the test suite locally, one can simply run:
```{bash}
cd /path/to/snmachine/ && pytest -vs tests/
```
<a name="version"></a>
## Package Versioning

### Version Format

We follow semantic versioning for `snmachine` releases, `MAJOR`.`MINOR`.`PATCH`:

* the `MAJOR` version will be updated when incompatible API changes are made,
* the `MINOR` version will be updated when functionality is added in a
backwards-compatible manner, and
* the `PATCH` version will be updated when backwards-compatible bug
fixes are made.

For more details, see https://semver.org/

Furthermore, `snmachine` adopts the release formatting of
[PEP440](https://www.python.org/dev/peps/pep-0440/)

### Release Planning

The authors of this package have adopted [milestones on Github](https://help.github.com/en/articles/about-milestones) as a vehile to
scope and schedule upcoming releases.  The main goal for a release is written in
the milestone description.  Then, any ideas, specific functionality, bugs, etcs
submitted as [issues](https://help.github.com/en/articles/about-issues)
pertinent to that goal are tagged for that milestone.  Goals for milestone are
discussed openly via a Github issue.

Past and upcoming releases can be seen on the  [snmachine milestones
page](https://github.com/LSSTDESC/snmachine/milestones).

For details on how the API has altered across versions, please refer to the
[`CHANGELOG.md`](https://www.github.com/LSSTDESC/snmachine/CHANGELOG.md) file,
which documents changes made and where breaking changes would have been introduced.
