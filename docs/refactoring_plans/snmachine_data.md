# Requirements for data class for snmachine. 

## Background 
At various times, everyone working on snmachine development has talked about refactoring snmachine. 
This came up in a short chat at the DESC collaboration meeting (Hiranya, Catarina, Christian, Rahul). One of the things we agreed upon was that it makes sense to have a common class for DESC SN (a bit of a rehash of an old idea from Pittsburgh, a few years back, where Robert and Kara played with snmachine). This should have the capability of accessing different datasets (starting with a few) that people need, and provide some common functionality. Since each analysis code needs to be doing the same thing, it makes sense not to re-invent that wheel from scratch.

An important functionality of such a code should be providing things in a usable format for DESC products like snmachine. But setting that requirement involves clearly stating what `snmachine` requires. So, we need to discuss a few things:
    - What data sets are essential for snmachine : Ans (SNANA, plasticc, SPCC are the ones we currently have)
    - What functionality is required for snmachine (This should either provide all of the functionality in snmachine `sndata` or a subset of that: This is a large part of the discussion).
    - How should the changes be implemented ? Assuming there is no major pushback on the basic idea, I think we will want a version of `snmachine` to be up and running all the time. As we start incorporating changes, how do we want to work to ensure this? We could either start with a minimal set that will be important for `plasticc` and `SPCC`, and write the interface to `snfeatures` etc.. An alternative is to write a module that will first convert this data class into the current sndata class and then write a separate module to build the interface.
 
## Define the uses of the data class for SNMachine
The idea is to first define:
    - what operations should be possible on the data (rather than how it interacts with a particular dataset)
    - Minimal set of data (SDSS, SPCC, plasticc)


I think the way to move forward for snmachine is to be used. Not just for classification which is a specific calculation, but some of the steps themselves in other calculations. 
- Therefore, it should be possible to use `snmachine` or at least the data io class when not everything that might have been there for plasticc/SPCC is there. This means there should be a minimal schema of inputs, and extended schemas which can be used for different calculations. Basically, I think we should try to be able to read in any kind of single object light curve easily.
- It is time to think about splitting. While we could move to higher memory computers as we deal with larger numbers of objects, this is bad. One way is using `dask`, which can perform out of core operations, and therefore solve the problem, in principle. The problem is that if we build in `dask` as an essential step in the engine of `snmachine`, changing this in future is likely to get complicated. Anecdotally, when people have used `dask` at large scales, the performance has not been very satisfactory and I think it might not be great to bake this into `snmachine` (but it is totally good to explore in a pipeline script that calls the `snmachine` library). Consequently this means tuning the number of objects that should be read into memory at a time. The `tuning` can be done at a later time, but we need to be able to split this as sets. This means we need to ask whether we want operations like `get_redshift` to give us values of such split sets, or the total population.

### Current requirements
- should be able to access data from plasticc, SNANA simulations.
- should have a structure where we can do joins of object properties provided with simulation with features extracted by snmachine.
- should be amenable to perfoming fits through sncosmo 

### Global attributes of data needed in snmachine 
- maximal set of filters
- maximal length of light curves (How is this defined?)

### Current Functionality and changes proposed

- [ ] The `__init__` contains subsetting by object names. (need to take this out of `__init__`) as this might not work on every data set.
- [ ] `get_object_names` (Should this apply to all data)? Is this really important? Does sorting make sense
- [ ] get global attributes
- [ ] `get_lightcurve` : essential functionality to get sncosmo fittable light curves.
- [ ] `set_model` (Why do we need this in data .... move elsewhere ?) 
- [ ] `plot_lc` : move to visualization ?
- [ ] `__on_press`: move to visualization ? 
- [ ] `plot_all`: move to visualization ?
- [ ] `get_types`: This will probably be used for the training set (rather than for simulation truths which may sometimes be available). 
- [ ] `get_redshift`:  
- [ ] `sim_stats` : What do we really need here ?
