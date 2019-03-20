# Requirements

- A data structure or in memory representation of the information associated with a single supernova. This should access the `metadata` or properties of the SN, as well as the Time Series (light curve) information (potentially from multiple sources) each of which should have their own schema. 
    - The code or schema variables should be stable define the standard variable names to be used by all our software.
    - Each schema should have flags for stages. The minimal schema should be designed so that the usage is very broad. In practice, one should be using flags to turn on non-minimal schema used in almost any applications.
    - It should be able to gracefully handle extra columns, for example if someone were to import an LSST dia object catalog with columns like aperture mag that are not part of the schema, it should be able to put the additional columns as additional columns inheriting the names. It should be possible to switch this of by providing a list of columns to import.
    - This obviously means that this should have a default dictionary to associate commonly used words with the variables in the minimal schema.
    - It should be possible for a user to pass a dictionary and override the default interpretation
    
    For a single SN, we have the following parts:
    - The metadata / properties.  This should  be a single row per SN. Schema should be staged.
        - Minimal set of features : Ids, Time of max brightness, observed in any band. If not observed NaN
        - Light Curve Feature Summary: with defined bands. Time of Max observed, NObs, max SNR,  for default bands (LSST bands) but could be swithed to a different set of bands by user, Length of light curve with SNR > 3 in each of the default bands, 3 highest max SNR in bands (eg. if max SNR in band is u 1, g 2, r 5.3, i 7, z 4, y 1) this would be 7., 5.3 4.
        - Hosts and redshifts. Spectroscopic redshift if available, Hosts and host properties if available. Photoz of hosts if available.
        - If simulations: truth properties (class, model, model parameters (These should be a sequence in a single colum) 
    - The Time Series part should be extensible, but have a minimal schema, extensible in stages:
        - The minimal schema for Time Series should be flux, time, bandpass. The next extensible stage should additionally have flux error, (and should immediately have zero pts, even if not provided)
        - It would be great if Indexing of each epoch in a SN whould allow reconstruction of the objectId and the visitID , CCDId. and ObjectID

- Input: 
    - It should have a method of obtaining data from files output by a few well known codes:
        -  SNCosmo
        - SNANA
        - cesium
    - DataSets
        - WiseRep
        - Open SN Catalog
        - BerkeleyDB
        - plasticc
        - LSST DRP
- Output:
    - It should be able to provide an in memory representation of single SN light curves for analysis. ie consumed by codes performing this. So, it should have the ability to create light curves of single SN for the following codes:
        - SNCosmo
        - should be in a format usable by 8Nmachine/ Rapid/ Pelican etc.
        - SNANA ? (probably not, it is not very useful to do single SN on SNANA)] 
    - Serialization: It should support a serialization that can be used by several codes.
        - SNANA
        - SNCosmo
        - Classification Codes (SNMachine/Rapid/Pelican)

- It should support on the fly operations
    - [ ] obtain co-added light curves (fluxes, and errors) over requested time intervals. This should immediately compute SNR of coadded variables.
    - [ ] threshold on SNR (of type requested)
    
- Collection of SN : (For distribution and restricting RAM usage), it is good to have a logical division unit of light curves. 
    - [ ] Allow a tesselation based division of SN into zones with a user defined resolution. 
        - [ ] Heapix tesselation
        - [ ] HTM
        - [ ] In each of the tiles, we should have two tables : one for the metadata part of all the SN in the tile, and one for the Time Series part.
