# Conventions for pushing code to `snmachine`

## Import Conventions

Conventions from Pep8 (https://pep8.org/).

Imports should be grouped in the following order:

1. standard library imports (https://docs.python.org/3/library/)
2. related third party imports (like `numpy`, `pandas` or `snmachine`)
3. local application/library specific imports

There should be a blank line between each group of imports.

## Naming Conventions

* class names -> PascalCase
* function and variable names -> snake_case
* function names -> start with descriptive verb (eg. get, compute, fit, load)
* hidden function names -> same as function names but starting with `_`
* descriptive names that minimize the number of necessary comments

## Documentation Conventions

### Functions

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
