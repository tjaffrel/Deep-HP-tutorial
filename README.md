# Deep-HP: Multi-GPUs platform for hybrid Machine Learning Polarizable Potential

Deep-HP is a multi-GPU Machine Learning Potential platform which is part of the Tinker-HP package and aims to couple Machine Learning with force fields for biological simulations. 

## What is Deep-HP?

Deep-High Performance aims to democratize the use of Machine Learning in biological simulations. Especially, Deep-HP is here to scale up Machine Learning Potential code from laptop to hexascale and from quantum chemistry to biophysics.  

What can I do? Here's a few examples:

* Combine trained machine learning potential with force fields components, such as long-range interactions and many-body polarization effects.
* Predict solvation free energies of drug-like molecules
* Predict ligand-binding free energies of host-guest systems
* Conformational sampling with state-of-the-art enhanced sampling techniques (Colvars, Plumed)
* ...
* For more check-out [TinkerTools](https://tinkertools.org/), [Tinker-HP](https://tinker-hp.org/)

Currently, the platform can only be use on trained machine learning potential and will broaden it capabilities in a close future. It is currently only compatible with Pytorch and TensorFlow. TorchANI and DeePMD like models can fully access the capabilities of the Platform. For more exotic models have a look into the source code (python libraries) or contact us!

## Installation 

### Python Environment 

We provide a python environment through the `tinkerml.yaml` file in the main folder `/my_directory/tinker-hp/GPU`. Inside you can find all the required libraries. If you don't have Anaconda or Miniconda you should download and load them before running or compiling Tinker-HP's Deep-HP branch.
If you want to install Anaconda or Miniconda, have a look here [Anaconda](https://www.anaconda.com/products/distribution)
To create the environment with Anaconda or Miniconda, in your terminal: `conda env create -f tinkerml.yaml`
To activate or deactivate your environment: `conda activate tinkerml` or `conda deactivate`  

Composition of the environment:
* Pytorch, TensorFlow, Keras are the most used python machine libraries and are the building block of numerous machine learning potential libraries.
* Deepmd-kit and libdeepmd are used for DeePMD models.
* Our TorchANI-based library composed of lot of new features, more coming soon!

### Prerequisites

The prerequisites for building Tinker-HP can be found [here](https://github.com/TinkerTools/tinker-hp/blob/Deep-HP/GPU/Prerequisites.md)

### Bash Environment 

Example of two bash environments with and without module, only modification to make are `/home/user/.../` and `path_to_gnu`:

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

After setting your environment according to the Prerequisites, clone Tinker-HP's Deep-HP branch github depository `git clone -b Deep-HP https://github.com/TinkerTools/tinker-hp.git` and proceed to installation as explain [Build Tinker-HP (GPU)](https://github.com/TinkerTools/tinker-hp/blob/Deep-HP/GPU/build.md)
Additional configuration options, enable by default on Deep-HP branch: 

#### Easy Build with install.sh
* `build_ml` enable Deep-HP if set to 1.

#### Using Makefile
* `NN_SUPPORT` enable Deep-HP if set to 1.

Before building Tinker-HP check if your CUDA version is matching `cuda_ver` (install.sh) or `cuda_version` (Makefile) and same for GPU [compute capability](https://en.wikipedia.org/wiki/CUDA) `c_c` (install.sh) or `compute_capability` (Makefile).

If you chose to use the easy build way, run:
```bash
$> pwd
#> /home/user/.../tinker-hp/GPU
$> ci/install.sh
```

:warning: Deep-HP is, for now, only available with Double and Mixed precision codes, so no fixed precision, `FPA_SUPPORT` should be set to 0.

## Run Deep-HP

Tinker need 
Existing Machine Learning Potential: ANI-2x, ANI-1x, ANI-1ccx
Compatible integrators: Verlet, RESPA/BAOAB RESPA, RESPA1/BAOAB RESPA1
To enable their use, in the key file:

Deep-HP has only two main **KEYWORDS**: `MLPOT` and `ML-MODEL`. 

`MLPOT` set the type of simulation:
* `MLPOT NONE` deactivate the MLP evaluation
* `MLPOT ONLY` activate only the MLP
* `MLPOT` activate the MLP but also the FF 
For the last keyword as the MLP and the FF is evaluate at each time step don't forget to disable the terms of the FF you don't want to use.

Example of a key file for which a trained MLP `my_mlp_model.pt` is couple with AMOEBA Van Der Waals energy:

```bash
parameters    amoebabio09
verbose
integrator                    respa
neighbor-list
a-axis                        62.23
vdw-cutoff                     12.0
ewald
ewald-cutoff                    7.0
pme-grid                   64 64 64
pme-order                         5
randomseed                    12345
polar-eps                   0.00001
polar-alg                         1
polar-prt                         2

MLPOT 
ML-MODEL my_mlp_model.pt
bondterm                       none
angleterm                      none
strbndterm                     none
ureyterm                       none
opbendterm                     none
torsionterm                    none
pitorsterm                     none
tortorterm                     none
mpoleterm                      none
polarizeterm                   none
# The only term of the amoebabio09 potential that is not disable is vdwterm
```

`ML-MODEL` set the machine learning model. For Tinker-HP native machine learning model, [ANI1X](https://pubs.rsc.org/en/content/articlehtml/2017/sc/c6sc05720a), [ANI1CCX](https://www.nature.com/articles/s41467-019-10827-4), [ANI2X](https://pubs.acs.org/doi/full/10.1021/acs.jctc.0c00121) and [ML_MBD](https://pubs.acs.org/doi/full/10.1021/acs.jpclett.2c00936), **KEYWORDS** are explicit:
* `ML-MODEL ANI2X` use ANI2X as MLP

For non native machine learning model, this is no more difficult, put your model  
* `MLPOT ONLY` activate only the MLP
* `MLPOT` activate the MLP but also the FF 


```python
e, f, v, ae, av = model.eval(coordinates, cell, atomic_species, True)
ev2kcalmol

predictor = self.model((atomic_species, coordinates), cell=cell, pbc=self.pbc)
predictor = self.model((atomic_species, coordinates), cell=cell, pbc=self.pbc, nblist=nblist, shift_energies=True)
hartree2kcalmol
```


```bash
MLPOT ONLY # NONE EMBEDDING
ANI2X ONLY # ANI2X ANI1X AN1CCX
ANI2X NONE # ANI2X ANI1X AN1CCX
ANI2X # ANI2X ANI1X AN1CCX
```

```bash
ML-MODEL ANI2X # ANI1X AN1CCX ANI2X ML_MBD 
ML-MODEL ANI_GENERIC my_ml_model.json # my_ml_model.pt my_ml_model.pkl
ML-MODEL DEEPMD my_ml_model.pb
```
QM/MM ligand/protein in water systems. water FF, ligand/protein MLP
```bash
MLPOT EMBEDDING
ML-MODEL ANI2X # ANI1X AN1CCX ANI2X ML_MBD my_ml_model.json my_ml_model.pt my_ml_model.pkl my_ml_model.pb

group 1 -1 100 # group 1 is the ML-MODEL
ligand -1 100
```

QM/MM Host-guest systems. Host and water FF, guest MLP
```bash
MLPOT EMBEDDING
ML-MODEL ANI2X # my_ml_model.json my_ml_model.pt my_ml_model.pkl my_ml_model.pb

group 1 -1 100 # group 1 ML-MODEL
ligand -1 100

group 2 -101 1000 # group 2 FF (e.g AMOEBA, AMBER, CHARMM, ...)
```

## Contact
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Please Cite

```bash
@misc{https://doi.org/10.48550/arxiv.2207.14276,
  doi = {10.48550/ARXIV.2207.14276},
  
  url = {https://arxiv.org/abs/2207.14276},
  
  author = {Inizan, Théo Jaffrelot and Plé, Thomas and Adjoua, Olivier and Ren, Pengyu and Gökcan, Hattice and Isayev, Olexandr and Lagardère, Louis and Piquemal, Jean-Philip},
  
  keywords = {Chemical Physics (physics.chem-ph), FOS: Physical sciences, FOS: Physical sciences},
  
  title = {Scalable Hybrid Deep Neural Networks/Polarizable Potentials Biomolecular Simulations including long-range effects},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
