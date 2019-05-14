"""
Simple post-processing code for recombining feature sets after running in parallel.
"""

from astropy.table import Table, vstack
import numpy as np
from collections import Counter
import sys

dataset='lsst_main'

root='/home/mlochner/sn_output/output_%s_no_z/features/' %dataset
features=sys.argv[1]
nruns=18 #How many files to combine
subset_name=dataset+'_subset_%d_%s.dat'
subset='all'
output_name=root+'%s_%s_%s.dat' %(dataset,subset, features)

if 'restart' in sys.argv:
    out=Table.read(output_name,format='ascii')
else:
    out=[]

for i in range(nruns):
    try:
        tab=Table.read(root+subset_name %(i, features), format='ascii')
        if len(out)==0:
            out=tab
        else:
            out=vstack((out, tab))
    except IOError:
        print '%s doesn\'t exist' %(subset_name %(i, features))

out.sort('Object')
#Find non-unique items (might have been repeated)
n=[k for (k,v) in Counter(out['Object']).iteritems() if v > 1]

if len(n)>0:
    new_out=out[0]
    objs=out['Object']
    for i in range(1, len(out)):
        if not objs[i]==objs[i-1]:
            new_out=vstack((new_out, out[i]))
    new_out.write(output_name, format='ascii')
else:
    out.write(output_name, format='ascii')

#Just check we ran all objects
if 'lsst' in dataset:
    if dataset=='lsst_main':
        rt_name='ENIGMA_1189_10YR_MAIN'
    else:
        rt_name='ENIGMA_1189_10YR_DDF'
    all_objs=np.loadtxt('/home/mlochner/sn/%s/high_SNR_snids.txt' %rt_name,dtype='str')
else:
    if subset=='spectro':
        all_objs=np.loadtxt('/home/mlochner/snmachine/DES_spectro.list', dtype='str')
    else:
        all_objs=np.loadtxt('/home/mlochner/sn/SIMGEN_PUBLIC_DES/SIMGEN_PUBLIC_DES.LIST', dtype='str')

mask=np.in1d(all_objs, out['Object'])
not_done=all_objs[np.where(~mask)[0]]

if len(not_done)!=0:
    flname='missing_objects.txt'
    print 'Some objects missing, saved to ', flname
    np.savetxt(flname, not_done, fmt='%s')
else:
    print 'All objects accounted for'
