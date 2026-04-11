# PENSim: A Toolkit for PEN-DNA Systems

PENSim is a Python-based toolkit for simulating bio-molecular systems based on the PEN-DNA toolbox. This repository contains all the necessary code and resources to run simulations, validate models, and explore advanced features.

---

## Repository Structure

The simulator is organized into three main Python files:

- **`strands.py`**: Defines three core classes—`Signal`, `Drain`, `Template`, and `Reporter`—used to model components of the PEN-DNA system.
- **`rates.py`**: Contains all kinetic constants (hybridization and enzymatic rates) used in the simulations.
- **`PEN-simulation.py`**: Defines the `PEN_System` class, which includes functions for managing and simulating PEN-DNA systems.

### Additional Folders

- **`tutorials/`**: Contains three tutorials covering basic usage and advanced features for signals and templates.
- **`validation/`**: Includes validation files for:
  - Bistable systems (Montagne et al., 2016)
  - Sensing circuits (Rondelez et al., 2023)
  - Neural networks (Okumura et al., 2022)
- **`resources/`**: Provides a comparison of hybridization rate calculations and a description of the full set of ordinary differential equations supporting PENSim.

---

## Dependencies

PENSim requires the following non-standard Python packages:
- **matplotlib**
- **scipy**
- **networkx**: For graph-like representation
- **[NuPACK](https://nupack.org/download/software)**: A free license is required to download and use this package.

Additionally, some Jupyter notebooks require the **Arial font** for proper rendering. If Arial is not installed, you can still run the notebooks by commenting out the relevant line. Moreover, you can set the `saving` flag to `False` to avoid saving figures.

---

## Installation

1. Clone this repository:
  ```bash
   git clone [repository-url]
  ```
2. Ensure matplotlib, scipy, and networkx are installed (could be done via pip otherwise).
3. Ensure NuPACK is installed and properly licensed. Follow the instructions on the [NuPACK website](https://nupack.org/download/software).



---

## Usage

To run a simulation, import the `PEN_System` class from `PEN-simulation.py` and follow the tutorials in the `tutorials/` folder for step-by-step guidance.

Example for importing code:

```python
# Get the absolute path to the root directory
root_dir = "path/to/this/repository/" 
# or os.path.abspath(os.path.join(os.getcwd(), "..")) if used in a folder
sys.path.append(root_dir)

# Import PEN_System and other classes for defining the PEN_System
from PEN_simulation import PEN_System
from strands import Signal, Drain, Template, Reporter, reverse_complement
```

---

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes. For major changes, open an issue first to discuss your ideas.

---

## Contact

For questions or contributions, please contact:  
**Gwendal Ducloz**  
Email: gwendal.ducloz[at]ens-lyon.fr
