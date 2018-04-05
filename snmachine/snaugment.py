import george

"""
Module handling the data augmentation of supernova data sets
"""

class SNAugment:
    """
    Skeletal base class outlining the structure for the augmentation of a 
    sndata instance. Classes that encapsulate a specific data augmentation 
    procedure should be derived from this class.
    """

    def __init__(self, d):
        """
	class constructor.

        Parameters: (why would you call this constructor in the first place?)
        ----------
        d: sndata object
            the supernova data set we want to augment

        """
        self.dataset=d
        self.meta={}#This can contain any metadata that the augmentation 
                    #process produces, and we want to keep track of.
        self.algorithm=None
        self.original=d.get_object_names()#this is a list of object names that
                                          #were in the data set prior to 
                                          #augmenting. 
                                          #DO NOT TOUCH FOR SELFISH PURPOSES

    def augment(self):
        pass

    def remove(self, obj=None):
        """
        reverts the augmentation step by fully or partially removing those 
        light curves that have been added in the augmentation procedure from 
        the data set.

        Parameters:
        ----------
        obj: list of strings
            These are the objects we will remove. If None is given, then we
            remove all augmented objects that have not been in the data set
            when we created the SNAugment object.
            NB: If obj contains object names that are in the original data set
            then we do not throw an error, but follow through on what you tell
            us to do. 
            
        """
        if obj is None:
            obj=list(set(self.dataset.object_names())-set(self.original))

        for o in obj:
            assert(o in self.dataset.object_names)
            self.dataset.data.pop(o)
            self.dataset.object_names=[x for x in self.dataset.object_names if x!=o]


class GPAugment(SNAugment):
    """
    Derived class that encapsulates data augmentation via Gaussian Processes
    """

    def __init__(self, d, templates=None):
        """
        class constructor. 

        Parameters:
        ----------
        d: sndata object
            the supernova data set we want to augment
        templates: list of strings
            If the templates argument is given (as a list of object names
             that are in the data set), then the augmentation step will take 
            these light curves to train the GPs on. If not, then every object 
            in the data set is considered fair game.
        """

        self.dataset=d
        self.meta={}
        self.algorithm='GP augmentation'
        if templates is None:
            templates=d.get_object_names()


    def augment(self, fractions):
        """
        HIC SUNT DRACONES
        """
        pass
