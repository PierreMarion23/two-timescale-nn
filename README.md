# Two-timescale regime for global convergence of neural networks

## Environment

### With conda

```
conda env create -f environment.yml
```

### With pip

Install Python 3.9.9 and pip 21.3.1, then

```
pip3 install -r requirements.txt
```

## Reproducing the experiments of the paper

For the experiments in 1D with piecewise constant functions, uncomment the relevant lines in main.py (see comments) and then run

```
python main.py
```

For the other experiments, see the relevant notebooks. The code is lightweight since we do not need to compare precisely to the two-timescale limit, which is why we created a self-contained notebook for each experiment.

The notebook manual_plots_paper.ipynb creates Figures 1, 2 and 7. 