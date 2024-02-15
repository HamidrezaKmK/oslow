# Setup
## Install dependencies
Some of the baselines require specific R packages to be installed. After making sure that R is installed on your system, you can install the required packages by running the following commands in the R console:
```R
install.package("devtools");
library(devtools);
install_github("https://github.com/cran/CAM");
```


For `DAG LEARNING ON THE PERMUTAHEDRON`, you need to install the following:
```
git clone git@github.com:vzantedeschi/DAGuerreotype.git
cd DAGuerreotype
chmod +x linux-install.sh
./linux-install.sh
python setup.py install
```


For `Differentiable Dag Sampling`, check the original repository [here](https://github.com/sharpenb/Differentiable-DAG-Sampling), and
install the `src` package following the instructions in the repository.
