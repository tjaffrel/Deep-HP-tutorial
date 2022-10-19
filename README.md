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

We provide a python environment through the `tinkerml.yaml` file in the main folder `tinker-hp/GPU`. Inside you can find all the required libraries. If you don't have Anaconda or Miniconda you should download and load them before running or compiling Tinker-HP's Deep-HP branch.
If you want to install Anaconda or Miniconda, have a look here [Anaconda](https://www.anaconda.com/products/distribution)
To create the environment with Anaconda or Miniconda, in your terminal: ```bash conda env create -f tinkerml.yaml```
To activate or deactivate your environment: ```bash conda activate tinkerml``` or ```bash conda deactivate```  

Composition of the environment:
* Pytorch, TensorFlow, Keras are the most used python machine libraries and are the building block of numerous machine learning potential libraries.
* Deepmd-kit and libdeepmd are used for DeePMD models.
* Our TorchANI-based library composed of lot of new features, more coming soon!

### Prerequisites

The prerequisites for building Tinker-HP can be found [here](https://github.com/TinkerTools/tinker-hp/blob/Deep-HP/GPU/Prerequisites.md)

### Bash Environment 

Example of bash environment with and without module:

```bash 
#/bin/bash
module purge
module load Core/Portland/nvhpc/21.5 Core/Gnu/9.2.0
unset CC CXX FC OMPI_MCA_btl
export GNUROOT=/usr/local/gcc-9.2.0
export LD_LIBRARY_PATH=/home/jaffrelot/PME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jaffrelot/PME_hyb/lib:$LD_LIBRARY_PATH
conda activate tinkerml
export PYTHONPATH=/home/jaffrelot/Programs/torchani:$PYTHONPATH
```



```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/) 
