# Plans for Refactoring

This is a directory for laying out plans for refactoring. There are different aspects to this, and they should be prioritized differently.

The main reasons to refactor the code are probably:
- Extend snmachine using bits of code already available to build a more easily usable 
framework for checking intermediate steps in `snmachine`. This is going to be in a new module (`analysis.py`) within the snmachine directory.
- Simplify the use of `snmachine`, as it is difficult to figure out what is running. This means that using snmachine (both individual steps and the end to end run) should be more transparent than it already is. 
- It should be flexible enough that several things can be done at the same time, without having to write alternative code. The key to this is not having functionality/structures that require too many things that might not be there in different datasets.
 
## Plans

It would be good to have a clear breakdown of the key functions of `snmachine` and what are the minimal requirements of each of these functions, and what might be convenience functions. This will likely be true of each module and so it might be good to start this module by module. We have some of this here already, but more needs to be done. 

- [ ] There is a need to be able to parse different datasets. Parsing different datasets is not a `snmachine` specific activity and will be required by several analysis codes in SN. Therefore, this is better written outside `snmachine`, with the requirement that it can work with `snmachine`. There is a set of requirements on such an interface (been provided for discussion in DESC) with data [here](./transient_data.md). Even if this works, we need to define the `snmachine` requirements we have for this interface code. I am trying to put these [here](./snmachine_data.md). This should be looked at and improved. Note, not all of the requested features will be there at the beginning, so it is important to define which minimal features would work.
    - [ ] `pandas` support aside from `astropy`
    - [ ] removing attributes that might only work for simulations, or require redshifts to get started.
- [ ] We need to set requirements for the features class (currently `snfeatures/SNFeatures`) in `snmachine`.
- [ ] It might be good to move the plotting into a visualization module.
