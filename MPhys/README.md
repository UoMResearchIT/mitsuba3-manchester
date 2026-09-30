# Running Mitsuba3 in Windows Subsystem for Linux (WSL2)

If you need to compile in a different OS than Linux then these instructions may be somewhat helpful but you'll have to work out what to do yourself in Windows, for example.

## Compiling source code

This in effect worked "out of the box" by following the [Mitsuba3 compilation from source instructions](https://mitsuba.readthedocs.io/en/stable/src/developer_guide/compiling.html), but to make it clear, the steps are as follows:

* Prerequisites: git, Python, pip, virtualenv
* Create a virtual environment for this: `virtualenv <env_name>` then `cd <env_name>` then `source bin/activate`
* Clone the fork of mitsuba3 recursively in order to capture the submodules: `git clone --recursive https://github.com/UoMResearchIT/mitsuba3-manchester`
* Install the relevant build tools using apt:
  * `sudo apt install clang-15 libc++-15-dev libc++abi-15-dev cmake ninja-build` (some of these may already be installed; you could also choose a different `clang` version should you wish to)
  * `sudo apt install libpng-dev libjpeg-dev` (for image I/O)
  * `sudo apt install libpython3-dev python3-distutils` (may be unnnecessary if you already have python installed)
* Export relevant environment variables: `export CC=clang-15`, `export CXX=clang++-15` (ideally, add these to your `~/.bashrc` file)
* Now build (from inside the mitsuba3-manchester directory that was created when cloning):
  * `mkdir build`
  * `cd build`
  * `cmake -GNinja ..`
  * (note: at this point it will tell you to edit the `mitsuba.conf` file to add any extra mitsuba variants that you may wish to use e.g. cuda/llvm; you can edit the conf file and then rerun the cmake command)
  * `ninja`
  * (note: it is probably possible to build using a system tool other than `ninja` but the parallel building helps a lot)
  * `source setpath.sh` (note: repeat this step every time you go into the virtual environment, or work out some way of adding it to the `activate` script)

This will give you the option of using `mitsuba` from the command line as well as making it accessible from Python scripts.

If you edit any C code, all that's required to recompile is to (re)run the `ninja` command from the `build` directory.

Finally, you may find that you'll need to install requirements for mitsuba3 as appropriate; just use `pip` to do this.

## Input files

As stated below, most of the examples we have tested and used are in the `Single_Emitter` directory, in particular for the experiments in the `Single_Emitter/performance_scaling_experiments` directory. The original CSV file used to generate the photons in these scripts is zipped at `Single_Emitter/csv/photons_1000000_filtered.zip`, which when unzipped (simply use the `unzip` command in Linux) will get you the required file.

The remaining csv input files in the `Single_Emitter/csv` directory (such as e.g. `test_new_photons_detected_spectral.csv`) were generated from this file, but you can also run the examples using these files by commenting/uncommenting code in the script files as required.

## Running examples

Most of the examples that have been run are the python scripts in the `Single_Emitter` directory, so please look at those to get ideas for your examples. It is likely that you will also have to build a scene for your simulations, to find out more about this look at the [mitsuba documentation](https://mitsuba.readthedocs.io/en/stable/index.html). For creating object files for such scenes, look at the Geant4 example code and geometry conversion code in our examples repository at [Geant4-Mitsuba3-Coupling](https://github.com/UoMResearchIT/Geant4-Mitsuba3-Coupling).

Note: the repository also contains Jupyter notebooks; it's possible to set up your environment to be able to then run notebooks in a Windows browser.  Follow the instructions at [https://code.adonline.id.au/jupyter-notebook-in-windows-subsystem-for-linux-wsl/](https://code.adonline.id.au/jupyter-notebook-in-windows-subsystem-for-linux-wsl/).

## Contact

If you have any issues please contact [andrew.gait@manchester.ac.uk](mailto:andrew.gait@manchester.ac.uk).

## Contributing

If after reading and understanding our implementation you wish to contribute to it then please feel free to make pull requests with any changes and they will be reviewed in the usual manner. You are also free to fork this repository to do any further work.