# Setup
## Install Requirements
Some of the baselines require specific R packages to be installed. After [making sure that R is installed on your system](https://cran.r-project.org/), you can install the required packages by running the following commands in the R console:
```R
install.package("devtools");
library(devtools);
install_github("https://github.com/cran/CAM");
```
Moreover, the baselines require the [`cdt`](https://github.com/FenTechSolutions/CausalDiscoveryToolbox) package, which can be installed using the following command (assuming the conda environment is activated):
```bash
python -m pip install cdt
```
You may need to install the R requirements for the `cdt` package (See [here](https://github.com/FenTechSolutions/CausalDiscoveryToolbox/blob/master/r_requirements.txt)).

### DAGuerreo
For `DAGuerreo` ([DAG Learning on the Permutahedron](https://arxiv.org/abs/2301.11898)), you need to run:
```bash
git clone git@github.com:vzantedeschi/DAGuerreotype.git
cd DAGuerreotype
chmod +x linux-install.sh
```
Then, you may remove the last lines in the `linux-install.sh` file (from the line `# install R and libraries to compute SID (optional)` to the end) if you do not have `sudo` access. Finally, you run the following (assuming the conda environment is activated):
```bash
./linux-install.sh
python setup.py install
```
### VI-DP-DAG

For `VI-DP-DAG` ([Differentiable Dag Sampling](https://arxiv.org/abs/2203.08509)), run the following commands to install the `src` package used in the [repo](https://github.com/sharpenb/Differentiable-DAG-Sampling) (assuming the conda environment is activated):

```bash
git clone git@github.com:sharpenb/Differentiable-DAG-Sampling.git
cd Differentiable-DAG-Sampling

```
