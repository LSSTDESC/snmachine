## Unit Tests

To execute the full test suite, go into the `test` folder and run:

```
py.test [flags] [arguments]
```

You can use the following flags (all optional):

  `-h`:
print a full list of flags

  `-v`:
verbose mode

  `-s`:
enable stdout capture

  `-mpl`:
plots will explicitly be compared to the baseline plots in `test/baseline` with the pytest-mpl package (see https://github.com/astrofrog/pytest-mpl). Without this flag, the default behaviour for the plotting tests is: create the plot, save it, check that the output file is nonempty.

  `-m "..."`:
set pytest marker that restrict to a subset of tests. The following markers might be interesting:

* `"not slow"`: exclude the the slow tests.

* `"slow"`: run only the slow tests.

* `"plots"`: run only the plotting tests.

If you do not have george or pymultinest installed, then the corresponding tests are automatically skipped.
Another way to restrict the number of tests is by passing a filename as argument. `py.test sndata_test.py` will only run the tests in this file. It is possible to select single tests via, e.g., `py.test snfeatures_test.py::test_gp_extraction`

The full suite should need 20-30min to run, the subset excluding the slow ones in about 5min.
