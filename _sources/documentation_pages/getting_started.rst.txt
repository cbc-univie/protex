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
   cd protex
   conda create --name protex -f devtools/conda_envs/protex.yml
   conda activate protex
   pip install .

.. admonition:: Success!
   :class: successstyle

   Now you are ready to go!

Citation
--------

When using Protex in published work, please cite
tba.

Thank you!

