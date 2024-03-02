# Setup
## DAGuerreo
For `DAGuerreo` ([DAG Learning on the Permutahedron](https://arxiv.org/abs/2301.11898)), you first need to install the following packages (assuming the conda environment is activated):
```bash
python -m pip install causaldag entmax
```
You then run:
```bash
git clone git@github.com:vzantedeschi/DAGuerreotype.git
cd DAGuerreotype
chmod +x linux-install.sh
```
Then, you may remove the last lines in the `linux-install.sh` file (from the line `# install R and libraries to compute SID (optional)` to the end) if you do not have `sudo` access. Finally, you run the following (assuming the conda environment is activated):
```bash
./linux-install.sh
python setup.py develop
```
## VI-DP-DAG

For `VI-DP-DAG` ([Differentiable Dag Sampling](https://arxiv.org/abs/2203.08509)), run the following commands to install the `src` package used in the [repo](https://github.com/sharpenb/Differentiable-DAG-Sampling) (assuming the conda environment is activated):

```bash
git clone git@github.com:sharpenb/Differentiable-DAG-Sampling.git
cd Differentiable-DAG-Sampling
python setup.py install
```

# Run Baselines

The set of baselines includes `CAM, bi_LSNM, DAGuerreo, SCORE, VarSort, VI_DP_DAG`. You can run a baseline using the following command :
```bash
python run_baseline.py baseline=CAM data=sachs
```
Similar to the original OSLow model, you can use the datasets provided in [here](https://github.com/HamidrezaKmK/oslow/tree/main/config/data) and custom datasets for baselines. 