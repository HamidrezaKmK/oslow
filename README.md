# Ordered Causal Discovery with Autoregressive Flows

![main_fig](https://github.com/vahidzee/ocdaf/assets/33608325/2352686b-965b-44d9-bd88-ee8b20ce7588)

<p align="center" markdown="1">
    <img src="https://img.shields.io/badge/Python-3.10-green.svg" alt="Python Version" height="18">
    <a href="https://arxiv.org/abs/2308.07480"><img src="https://img.shields.io/badge/arXiv-TODO-blue.svg" alt="arXiv" height="18"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#license">License</a>
</p>

# Documentation Coming Soon ...

## Environment

```bash
conda env create -f env.yml
conda activate oslow
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Running


### Causal Discovery

```bash
python train.py
```

Run on linear Laplace with a model that has an appropriate latent noise:

```bash
python train.py data/dataset=linear_laplace model=specified 
```

Run on larger covariate size (default is 3 but you can change it to 4 with the following command):

```bash
python train.py data.dataset.graph.num_nodes=4
```

### Ensemble
Run ensemble code that plots a scatterplot:

```bash
python ensemble.py
```

Run it on Laplace Linear data:

```bash
python main.py --data/dataset=laplace_linear
```
