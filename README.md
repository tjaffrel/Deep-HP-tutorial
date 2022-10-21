# Deep-HP: Multi-GPUs platform for hybrid Machine Learning Polarizable Potential

Deep-HP is a multi-GPU Machine Learning Potential platform which is part of the Tinker-HP package and aims to couple Machine Learning with force fields for biological simulations. 

## What is Deep-HP?

Deep-High Performance aims to democratize the use of Machine Learning in biological simulations. Especially, Deep-HP is here to scale up Machine Learning Potential code from laptop to hexascale and from quantum chemistry to biophysics.  

What can I do? Here's a few examples:

* Combine trained machine learning potential with force fields (long-range interactions and many-body polarization effects).
* Predict solvation free energies of drug-like molecules.
* Predict binding free energies.
* Conformational sampling with state-of-the-art enhanced sampling techniques (Colvars, Plumed).
* ...
* For more check-out [TinkerTools](https://tinkertools.org/), [Tinker-HP](https://tinker-hp.org/)

Currently, the platform can't be use to train a model and is compatible with TorchANI-type and DeePMD models but we will broaden it capabilities in a close future. <br />

For more exotic models have a look into the source code (python libraries) or [contact us](https://piquemalresearch.com/)!

## Installation 

### Python Environment 

We provide a python environment through the `tinkerml.yaml` file in the main folder `/home/user/.../tinker-hp/GPU`. Inside you can find all the required libraries. If you don't have Anaconda or Miniconda you should download it. This environment must be activate before running or compiling Tinker-HP's Deep-HP branch. <br />
If you want to install Anaconda or Miniconda, have a look here [Anaconda](https://www.anaconda.com/products/distribution)<br />
To create the environment with Anaconda or Miniconda, run in your terminal: `conda env create -f tinkerml.yaml`<br />
To activate or deactivate your environment: `conda activate tinkerml` or `conda deactivate`  <br />

Composition of the environment:
* [Pytorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/) are the building block of most of machine learning potential libraries.
* Deepmd-kit and libdeepmd are used for [DeePMD](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html) models.
* Our [TorchANI](https://aiqm.github.io/torchani/)-based library composed of lot of new features, more coming soon!

### Prerequisites

The prerequisites for building Tinker-HP can be found [here](https://github.com/TinkerTools/tinker-hp/blob/Deep-HP/GPU/Prerequisites.md)

### Bash Environment 

Example of two bash environments with and without module, that should be load before running or compiling Tinker-HP's Deep-HP branch, the only modifications to make are `/home/user/.../` and `path_to_gnu`:

```bash 
#/bin/bash
module purge
module load Core/Portland/nvhpc/21.5 Core/Gnu/9.2.0
unset CC CXX FC OMPI_MCA_btl
export GNUROOT=/path_to_gnu
export LD_LIBRARY_PATH=/home/user/.../tinker-hp/GPU/lib:$LD_LIBRARY_PATH
conda deactivate
conda activate tinkerml
export PYTHONPATH=/home/user/.../anaconda3/envs/tinkerml/lib/python3.9/site-packages/torchani:$PYTHONPATH
```

```bash 
#!/bin/bash
module purge
NVARCH=`uname -s`_`uname -m`; export NVARCH
NVCOMPILERS=/opt/nvidia/hpc_sdk; export NVCOMPILERS
unset CC CXX FC OMPI_MCA_btl
export GNUROOT=/path_to_gnu
conda deactivate
conda activate tinkerml
export PYTHONPATH=/home/user/.../anaconda3/envs/tinkerml/lib/python3.9/site-packages/torchani:$PYTHONPATH
export PATH=$NVCOMPILERS/$NVARCH/21.5/comm_libs/mpi/bin:$NVCOMPILERS/$NVARCH/21.5/compilers/bin:$NVCOMPILERS/$NVARCH/21.5/cuda/bin:$PATH
export LD_LIBRARY_PATH=$NVCOMPILERS/$NVARCH/21.5/comm_libs/mpi/lib:$NVCOMPILERS/$NVARCH/21.5/comm_libs/nccl/lib:$NVCOMPILERS/$NVARCH/21.5/comm_libs/nvshmen/lib:$NVCOMPILERS/$NVARCH/21.5/math_libs/lib64:$NVCOMPILERS/$NVARCH/21.5/compilers/lib:$NVCOMPILERS/$NVARCH/21.5/cuda/lib64:/home/user/.../tinker-hp/GPU/lib:$LD_LIBRARY_PATH
```

### Build 

After clone Tinker-HP's Deep-HP branch github depository `git clone -b Deep-HP https://github.com/TinkerTools/tinker-hp.git`, set your environment as explain before and proceed to installation as explain [Build Tinker-HP (GPU)](https://github.com/TinkerTools/tinker-hp/blob/Deep-HP/GPU/build.md). <br />
Additional building configuration options of Deep-HP: 

#### Easy Build with install.sh
* `build_ml` enable Deep-HP if set to 1. (Default value to 1 for Deep-HP)

#### Using Makefile
* `NN_SUPPORT` enable Deep-HP if set to 1. (Default value to 1 for Deep-HP)

Before building Tinker-HP check if your CUDA version is matching `cuda_ver` (install.sh) or `cuda_version` (Makefile) and same for GPU [compute capability](https://en.wikipedia.org/wiki/CUDA) `c_c` (install.sh) or `compute_capability` (Makefile). <br />
Deep-HP use the cudatoolkit version 11.3 so your version should a, e.g, `cuda_ver >= 11.3`.

If you chose to use the easy build way, run:
```bash
$> pwd
#> /home/user/.../tinker-hp/GPU
$> ci/install.sh
```

:warning: Deep-HP is, for now, only available with Double and Mixed precision codes, so no fixed precision, `FPA_SUPPORT` is set by default to 0 on the Deep-HP branch.

## Run Deep-HP

Deep-HP has only two main **KEYWORDS**: `MLPOT` and `ML-MODEL`. 

`MLPOT` set the type of simulation:
* `MLPOT NONE` deactivate the machine learning potential evaluation
* `MLPOT ONLY` activate only the machine learning potential (Example 1, 3, 6)
* `MLPOT` activate the machine learning potential but also the force field. As both are evaluate at each time step don't forget to disable the terms of the forcefield you don't want to use (Example 2)
* `MLPOT EMBEDDING` active the machine learning potential on a group of atoms, like a QM/MM embedding. (Example 4, 5)

`ML-MODEL` set the machine learning model. For Tinker-HP native machine learning model, [ANI1X](https://pubs.rsc.org/en/content/articlehtml/2017/sc/c6sc05720a), [ANI1CCX](https://www.nature.com/articles/s41467-019-10827-4), [ANI2X](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00121) and [ML_MBD](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.2c00936), **KEYWORDS** are explicit:
* `ML-MODEL ANI2X` use ANI2X as machine learning potential. Same for ANI1X, ANI1CXX and ML_MBD. (Example 1, 2, 4, 5, 6)

For non native machine learning model, this is no more difficult. You should put your machine learning potential model inside your simulation directory (with your .xyz and .key) and write explicity the name, including extension, of your model. <br />
The begining of the **KEYWORDS** depend of whether the model was build with TorchANI or [DeePMD](https://docs.deepmodeling.com/projects/deepmd/en/master/index.html):
* `ML-MODEL ANI_GENERIC my_mlp_model.pt # my_mlp_model.json my_mlp_model.pkl my_mlp_model.yaml`
* `ML-MODEL DEEPMD my_mlp_model.pb` (Example 3)

### TorchANI format

If your `model` was trained with TorchANI, you don't have to do anything in particular but make sure that :warning: **your predicted energies is in Hartree and atomic coordinates is in Angstrom** :warning: (default units in TorchANI). Tinker-HP will directly convert to kcal/mol. <br /> 
For more information have a look in [TorchANI](https://aiqm.github.io/torchani/examples/energy_force.html) or directly in our source code extension located in your tinkerml environment `/home/user/.../anaconda3/envs/tinkerml/lib/python3.9/site-packages/torchani`. <br />
Our TorchANI extension has also functions that convert your `model` in `pkl`, `yaml` and `json` formats. These formats are more compact and human readable friendly than TorchANI's `jit` format. But more importantly, when you are saving a `model` in `jit` format you are saving the whole code which may cause issues and you will not be able to use Tinker-HP's full capabilities (e.g neighbor list, multi-GPU, Particle Mesh Ewald). <br />
We recommend to save your model in `pkl`, `yaml` or `json` formats. Here is a python code that explain how to do it from this [Example](https://aiqm.github.io/torchani/examples/nnp_training.html) of TorchANI:

```python
# once you have your trained model:
model = torchani.nn.Sequential(aev_computer, nn).to(device)
# ...
_, predicted_energies = model((species, coordinates))
# you can save it with:
model.to_json("my_mlp_model.json")
model.to_pickle("my_mlp_model.pkl")
model.to_yaml("my_mlp_model.yaml")
```

### DeePMD

For DeePMD it is similar but we don't provide other formats than the original `pb`. :warning: **In DeePMD your predicted energies is in eV and atomic coordinates is in Angstrom** (default units in DeePMD).

# Example

We provide 6 examples that encompass the basics of Deep-HP inputs witch which you can do almost everything you want. They are located in `/home/user/.../tinker-hp/GPU/examples/`. Some toy machine learning potential models are located in `/home/user/.../tinker-hp/GPU/ml_models/`.

* **Example 1:** <br />
*Objective:* Perform machine learning potential simulation - on full system. <br />
Simulation parameter: NPT with montecarlo barostat and bussi thermostat, velocity-verlet integrator and ANI2X potential. <br />
Command GPU: `mpirun -np 1 ../bin/dynamic_ml.mixed Deep-HP_example1 1000 0.2 100 4 300 1` <br />

* **Example 2:** <br />
*Objective:* Perform hybrid machine learning potential/MM simulation - on full system.<br />
Simulation parameter: NPT with montecarlo barostat and bussi thermostat, velocity-verlet integrator and hybrid ANI2X/AMOEBA VdW energy.<br />
Command GPU: `mpirun -np 1 ../bin/dynamic_ml.mixed Deep-HP_example2 1000 0.2 100 4 300 1`<br />

* **Example 3:**<br />
*Objective:* Perform DeePMD machine learning potential simulation - on full system.<br />
Simulation parameter: NPT with montecarlo barostat and bussi thermostat, velocity-verlet integrator and toy model DeePMD potential. <br />
Command GPU: `mpirun -np 1 ../bin/dynamic_ml.mixed Deep-HP_example3 1000 0.2 100 4 300 1`<br />

* **Example 4:**<br />
*Objective:* Perform hybrid machine learning potential/MM simulation - on a ligand of the SAMPL4 challenge.<br />
Simulation parameter: NPT with montecarlo barostat and bussi thermostat, RESPA integrator with 0.2fs inner time-step/ 2fs outer time-step and ANI2X potential applied only to ligand-ligand interactions (atoms 1 to 24), ligand-water and water-water interactions use AMOEBA.<br />
Command GPU: `mpirun -np 1 ../bin/dynamic_ml.mixed Deep-HP_example4 1000 2.0 100 4 300 1`<br />

* **Example 5:**<br />
*Objective:* Perform hybrid machine learning potential/MM simulation - on a host-guest complex of the SAMPL4 challenge.<br />
Simulation parameter: NPT with montecarlo barostat and bussi thermostat, RESPA integrator with 0.2fs inner time-step/ 2fs outer time-step and ANI2X potential applied only to ligand-ligand interactions (atoms 1 to 24), the rest of the interactions use AMOEBA.<br />
Command GPU: `mpirun -np 1 ../bin/dynamic_ml.mixed Deep-HP_example5 1000 2.0 100 4 300 1`<br />

* **Example 6:**<br />
*Objective:* Perform machine learning potential simulation - on a full large system (100 000 atoms) with multi-GPUs.<br />
Simulation parameter: NPT with montecarlo barostat and bussi thermostat, velocity-verlet integrator and ANI2X potential.<br />
Command GPU: `mpirun -np 2 ../bin/dynamic_ml.mixed Deep-HP_example5 1000 0.2 100 4 300 1`<br />

# Contact
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.<br />

If you want to add your favorite machine learning potential code inside Deep-HP, [Contact us](https://piquemalresearch.com/)!

# Please Cite

```bash
@misc{https://doi.org/10.48550/arxiv.2207.14276,
  doi = {10.48550/ARXIV.2207.14276},
  url = {https://arxiv.org/abs/2207.14276},
  author = {Jaffrelot Inizan, Théo and Plé, Thomas and Adjoua, Olivier and Ren, Pengyu and Gökcan, Hattice and Isayev, Olexandr and Lagardère, Louis and Piquemal, Jean-Philip},
  keywords = {Chemical Physics (physics.chem-ph), FOS: Physical sciences, FOS: Physical sciences},
  title = {Scalable Hybrid Deep Neural Networks/Polarizable Potentials Biomolecular Simulations including long-range effects},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
