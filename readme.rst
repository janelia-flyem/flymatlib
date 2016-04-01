flymatlib : FlyEM Matlab library
--------------------------------

overview of features:

* 3d deep learning / convolutional neural networks for electron
  microscopy (EM) analysis, using `MatConvNet
  <http://www.vlfeat.org/matconvnet/>`_ and `MexConv3d
  <https://github.com/pengsun/MexConv3D>`_

* distributed inference over Grid Engine cluster

* synapse / object detection, precision-recall metrics

set-up
______

0. quick start

   At bare minimum to get started after downloading flymatlib, run
   ``fml_setup.m`` at each Matlab session to set the path (point 1
   below), and if this is the first time using flymatlib, run
   ``fml_make_mex`` to compile necessary mex files (point 2 below).

1. setting Matlab path

   The path for flymatlib can be set by running the ``fml_setup.m``
   function.  For instance, assuming the library was extracted to
   ``~huangg/flymatlib``, the following call in Matlab will set the
   path for the library: ``run('~huangg/flymatlib/fml_setup.m')``

2. compiling third-party mex functions

   flymatlib makes use of third-party libraries that contain mex code
   that needs to first be compiled.  ``fml_make_mex.m`` will attempt
   to do all the necessary compilation.  If errors occur, then it may
   be necessary to consult the compilation code for each library:
   ``third_party/matconvnet/matlab/vl_compilenn.m`` and
   ``third_party/mexconv3d/make_all.m``

3. (optional) compiling for distributed inference over cluster

   flymatlib was written to allow for distributed computing over a
   Grid Engine cluster.  The following steps may be necessary to get
   this functionality to work:

   a. Univa Grid Engine

      flymatlib assumes jobs are submitted using the Univa Grid
      Engine.  Check ``utils/fml_qsub.m`` and in particular the value
      of ``qsub_loc`` to see if anything needs to be modified for your
      environment.

   b. Matlab Compiler Runtime

      Distributed cluster jobs require compiling flymatlib with the
      Matlab compiler.  Running the resulting executable requires
      specifying the location of the Matlab libraries as well as the
      location to unpack the executable.  These are indicated in the
      shell scripts ``fml_dist/my_run_fml_dist.sh`` and
      ``fml_dist/my_run_fml_dist_single.sh``, as ``matlabroot`` and
      ``MCR_CACHE_ROOT`` respectively.  Again, these may need to be
      modified for your environment.

      Please note that ``MCR_CACHE_ROOT`` should be a location local
      to the machine that is running the distributed job, and not a
      location on a network drive, as this will likely significantly
      degrade performance.  As written, the shell scripts assume that
      there is a local ``/scratch`` drive on every cluster machine,
      and that the user will have a folder ``/scratch/${LOGNAME}``
      with write permissions.

   c. compiling flymatlib

      flymatlib can be compiled at the Matlab prompt with the function
      ``fml_make.m``

   d. directory for temporary files

      distributed computation will produce temporary files, as a means
      of passing information to the distributed workers.  flymatlib
      assumes that there is a global variable ``DFEVAL_DIR`` that
      specifies the directory to place such files in.  ``fml_setup.m``
      will give an error if this global variable does not exist or
      does not specify a valid directory

example usage for synapse detection
___________________________________

see ``examples/fml_tbar_classifier_example.m`` for an example script
that demonstrates CNN synapse (T-bar) detector training, inference,
and precision/recall computation.
