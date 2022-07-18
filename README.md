# Self-recognition as Optimisation

The repository provides source code for reproducing the experiments described in:

- Timothy Atkinson and Nihat Engin Toklu, "Self-recognition as Optimisation", Proceedings of ALIFE 2022: The 2022 Conference on Artificial Life. MIT Press, 2022.

## Usage

To use this code, the easiest way is to create a conda environment using `env.yml`:

```bash
git clone https://github.com/nnaisense/self-recog-as-optim.git
cd self-recog-as-optim
conda env create -f env.yml
```

You can then activate the environment:

```bash
conda activate selfrec
```

And run experiments. To run an evolutionary experiment do:

```bash
python run/evolution.py [NUM REPEATS] [SEPARATE POLICIES]
```

where `[NUM REPEATS]` is the number of repeated evolutionary runs (in the paper: 10) and `[SEPARATE POLICIES]` is whether to use separate policies (`True`) or a single policy which must learn to self-recognise (`False`). Our recommendation is that you do:

```bash
python run/evolution.py 10 False
python run/evolution.py 10 True
```

This will create a folder `experiment_logs/` containing two sub-folders: `64_15000_sep` and `64_15000_com`, each containing 10 evolution runs and corresponding to the separate-policies scenario ('`sep`') or a single combined policy scenario ('`com`') respectively.

You can run:

```bash
python run/analysis.py experiment_logs/64_15000_com experiment_logs/64_15000_sep 10
```

to obtain statistical analysis of the experiments and

```bash
python run/evolution_playback.py experiment_logs/64_15000_com/0/generation_4000.pkl experiment_logs/64_15000_sep/0/generation_4000.pkl
```

replacing `0/generation_4000.pkl` with any particular evolutionary run you prefer to obtain a replay of learned behaviours and plots such as those shown in Fig. 2.

## Maintainers

Developed and maintained by [Timothy Atkinson](https://github.com/NaturalGradient).
