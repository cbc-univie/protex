Getting Started
===============

Protex documentation
--------------------


**Protex** is an lightweight object-oriented python
extension to easily enable bond breaking and formation in molecular
dynamics simulations. 

Getting involved
----------------

Please report **bugs** or **enhancement requests** through the `Issue
Tracker`_.

.. _Issue Tracker: https://github.com/florianjoerg/protex/issues


User Guide
----------

The Protex :ref:`Quick-Start-Guide` provides information on how to
get startet with setting up a system for Protex.

.. _installation-instructions:

Installing Protex and Source Code
---------------------------------

**Source code** is available from
https://github.com/florianjoerg/protex/ under the MIT License.
Obtain the sources with `git`_.

.. code-block:: bash

   git clone https://github.com/florianjoerg/protex.git
   cd protex
   pip install .

.. _git: https://git-scm.com/

**Using conda environment**

Information how to obtain conda can be found `here <https://docs.conda.io/projects/conda/en/latest/>`_.

.. Important:: 
    It is recommended to use a (conda) environment for the installation.

First create a conda environment and install the dependencies. You can clone the github project and use the yml file. 

.. Note::
   If you plan to use protex with polarizable MD simulations it is import to get the current master branch of ParmEd.
   The last release (v3.4.3) is not enough since the very recently added Drude functionality is necessary.

.. code-block:: bash

   git clone https://github.com/florianjoerg/protex.git
   cd protex/devtools/conda_envs
   conda create --name protex --file protex.yml
   conda activate protex
   cd ../../
   pip install .

.. Tip::

   It is recommended to use the VVIntegrator Plugin. |:arrow_down:|

**Usage with the VVIntegrator Plugin for OpenMM**

To use the better Drude Integrator install the plugin from `velocity-verlet <https://github.com/z-gong/openmm-velocityVerlet>`_.
Here a quick install guide is given. Use the documentation provided there for further details.

.. code-block:: bash

   conda activate protex # if not already done
   conda install swig
   git clone https://github.com/z-gong/openmm-velocityVerlet.git
   cd openmm-velocityVerlet
   # set OPENMM_DIR
   sed -i "s~SET(OPENMM_DIR \"/usr/local/openmm~SET(OPENMM_DIR \"${CONDA_PREFIX}../../pkgs/openmm-7.6(!.bz2)" CMakeLists.txt
   # set CMAKE_INSTALL_PREFIX
   sed -i "/^ENDIF (\${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})/a SET(CMAKE_INSTALL_PREFIX \"${CONDA_PREFIX}\" CACHE PATH \"Where to install the plugin\" FORCE)" CMakeLists.txt
   mkdir build
   cd build
   cmake ../.
   make install
   cd python
   make PythonInstall

.. admonition:: |:confetti_ball:| Success! |:confetti_ball:|
   :class: successstyle

   Now you are ready to go!

Citation
--------

When using Protex in published work, please cite
tba.

Thank you!

