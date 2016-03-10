#!/bin/sh
# script for execution of deployed applications
#
# Sets up the MATLAB Runtime environment for the current $ARCH and executes 
# the specified command.
#

echo $SHELL
hostname
export MCR_CACHE_ROOT="/scratch/${LOGNAME}/mcr_$1"
export MCR_CACHE_VERBOSE="1"
mkdir -p ${MCR_CACHE_ROOT}

exe_name=$0
exe_dir=`dirname "$0"`
echo "------------------------------------------"
if [ "x$1" = "x" ]; then
  echo Usage:
  echo    $0 \<deployedMCRroot\> args
else
  echo Setting up environment variables
  MCRROOT="$1"
  echo ---
  export matlabroot="/usr/local/matlab-2015b"
  export PATH="$PATH:$matlabroot/bin"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:\
$matlabroot/runtime/glnxa64:\
$matlabroot/bin/glnxa64:\
$matlabroot/sys/os/glnxa64:\
$matlabroot/sys/opengl/lib/glnxa64"
  export MCR_INHIBIT_CTF_LOCK=1

  shift 1
  args=
  while [ $# -gt 0 ]; do
      token=$1
      args="${args} \"${token}\"" 
      shift
  done
  eval "\"${exe_dir}/fml_dist_single\"" $args
fi
exit

