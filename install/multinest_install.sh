#!/usr/bin/env bash
# usage: $0 [-a absolute path to install]
# Install MultiNest and pyMultiNest to an a
# Install script from https://github.com/JohannesBuchner/pymultinest-tutorial with a few changes.

set -e
set -u
set -o pipefail


### Default variables for install locations of software
snmachine_dir="$( cd "$(dirname "$0")" ; pwd -P )"
software_repo=${HOME}/soft
mnestdir=$software_repo/MultiNest
pymnestdir=$software_repo/PyMultiNest
# remote repository for MultiNest/pyMultiNest
multinest_repo=https://github.com/JohannesBuchner/MultiNest

echo "The snmachine install directory is $snmachine_dir"
## Get optional arguments for location of install
while getopts 'a:' OPTION; do
  case "$OPTION" in
    a)
      avalue="$OPTARG"
      echo "will create sofware in $OPTARG making the directory if necessary"
      ;;
    ?)
      echo "script usage: $(basename $0) [-a absolute_path_to_install_location], defaults to install under ${software_repo}" >&2
      exit 1
      ;;
  esac
done
shift "$(($OPTIND -1))"


## Set the install locations 
if [ -z ${avalue+x} ]; then
	echo "default locations $mnestdir and $pymnestdir used for location of MultiNest and PyMultiNest"
else
        software_repo=$avalue	
        mnestdir=$software_repo/MultiNest
        pymnestdir=$software_repo/PyMultiNest
	echo "Based on arguments, MultiNest will be installed in $mnestdir and PyMultiNest in $pymnestdir"
fi


### Check if install location already exists
if [ ! -d ${software_repo} ]; then
	echo "creating ${software_repo} "
	mkdir -p ${software_repo}
fi

### Check if directory for multinest is already present
if [ -d $mnestdir ]; then
      echo ">> Multinest repo already exists, please delete to run script"
      exit 1
fi

### Clone the MultiNest repository
echo ">> clone Multinest from $multinest_repo to $software_repo/MultiNest"
pushd $software_repo
git clone ${multinest_repo} 

### Build multinest
echo ">> Building MultiNest"
pushd $mnestdir 
echo $PWD
echo 'I am in ' $PWD 
mkdir -p build; cd build
echo 'I am in ' $PWD 
cmake ../ > /dev/null && make > /dev/null
popd

### Build pymultinest
echo ">> clone PyMultiNest to $pymnestdir"
if [ ! -d $pymnestdir ]; then
      echo ">> PyMultinest repo does not exist, creating directoryt"
fi
git clone https://github.com/JohannesBuchner/PyMultiNest  $pymnestdir
echo ">> Building PyMultiNest"
pushd $pymnestdir
python setup.py install
popd

echo "------------------ libaray paths"

echo "mnestdir=$mnestdir" > $snmachine_dir/setup.sh
if [ $OSTYPE=='linux'* ] ; then
	echo "found linux"
	echo 'export LD_LIBRARY_PATH=$mnestdir/lib/:$LD_LIBRARY_PATH' >> $snmachine_dir/setup.sh
elif [ $OSTYPE=='darwin'* ]; then
       echo 'found mac'
       echo 'export DYLD_LIBRARY_PATH=$mnestdir/lib/:$DYLD_LIBRARY_PATH' >> $snmachine_dir/setup.sh
		
else
	echo "other operating system, please set the library paths appropriately manually"
fi
	

## TODO 
### Check whether OS is Linux or Darwin
## export DYLD_LIBRARY_PATH=/Users/rbiswas/soft/MultiNest/lib in  bash_profile for Darwin
## For linux: export LD_LIBRARY_PATH=/my/directory/MultiNest/lib/:$LD_LIBRARY_PATH 

