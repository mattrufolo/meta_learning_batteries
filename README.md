# Structured Meta-Learning for Battery State-of-Charge Estimation

This repository contains the Python code to reproduce the results of the paper *Structured Meta-Learning for Battery State-of-Charge Estimation: Data Efficiency and Sim-to-Real Transfer* by M. Rufolo, D. Piga, et al.

The paper studies how to estimate the state of charge (SOC) of a lithium-ion cell from current and voltage measurements, when only a few labelled data of the target cell are available. The estimator is split into a part that is **shared** across all batteries and operating conditions and a small **task-specific** part that is adapted to the new cell from a short calibration segment. Training uses **simulated cells only**; testing is done on unseen simulated batteries and on two **real** cells (A123 LFP, out-of-distribution chemistry, and Samsung INR21700-50E NCA), without any retraining. The compared methods are:

* **CAMEL**: shared nonlinear features with a linear task-specific head, adapted in closed form (ridge regression);
* **CoDA**: low-dimensional task-specific parameters that modulate the shared network, adapted by gradient-based optimization;
* **MAML-Windowed**: gradient-based meta-learning of an initialization that is fine-tuned on the calibration set;
* **Baseline MLP**: supervised multilayer perceptron (in the style of He et al., 2014), trained in the standard (non-meta) framework.

All the methods use the same sliding-window input (current, voltage, sampling time and their increments) and the same Kalman-filter post-processing.

## Data and checkpoints

The simulated and real battery datasets and the trained checkpoints are **not stored in the git tree**. In the **release** of this repository we will insert:

* `dataset_battery/`: the simulated cells used for meta-training, validation and testing, and the two real cells (A123, INR21700-50E);
* `checkpoints/`: the trained models, 5 training seeds for each method.

Download them from the *Releases* page and unzip them in the root of the repository, so that the layout is

```
.
├── dataset_battery/
│   └── <battery name>/*.csv          # one file per operating condition (i, v, SOC, ...)
├── checkpoints/
│   ├── seed_0/
│   │   ├── camel_best.pt
│   │   ├── coda_best.pt
│   │   └── maml_windowed_best.pt
│   ├── ...
│   └── seed_4/
└── plots/                            # created by the scripts
```

## Main files

### Training
* [soc_metalearning_training.ipynb](soc_metalearning_training.ipynb): meta-training of CAMEL, CoDA, MAML-Windowed and of the supervised Baseline on the simulated cells, with the battery-level train/validation/test split. It saves the best checkpoint of each method in `checkpoints/seed_<k>/`.

### Evaluation and plots
* [plot_sim_to_real_2.py](plot_sim_to_real_2.py): evaluation-only script, it loads the checkpoints and reproduces the data-efficiency results (simulated test batteries and sim-to-real on A123 and INR21700-50E), averaged over the 5 seeds. It produces:
  * `data_efficiency_final.png/.pdf`: RMSE versus calibration fraction on the simulated test batteries;
  * `data_efficiency_sim2real_true_<cell>.pdf`: same for the real cells;
  * `<cell>_trajectory_true_capacity_by_calibfrac.pdf`: SOC trajectories on the real cells for different calibration fractions;
  * `simulated_test_summary_true.csv`, `sim2real_summary_true_<cell>.csv`: tables with the numbers.
* [soc0_robustness_ablation.py](soc0_robustness_ablation.py): robustness of the Kalman filter to a wrong initial SOC (true / 0.5 / 0.0) and to the initial covariance, with the recovery trajectories.

### Calibration protocol
In all the experiments the calibration set is the **first** `N_c = max(16, floor(c N))` windows of the test run (the same for all methods and seeds), and the error is computed only on the windows after the calibration segment. The seeds change only the checkpoint (or the initialization of the Baseline).

### Additional python files
* [check_coherence.py](check_coherence.py): checks that the shared definitions (constants, windowing, battery split) are identical in two scripts.

### Figures
* [meta_learning_workflow.tex](meta_learning_workflow.tex): TikZ source of the workflow figure (meta-training and deployment on an unseen battery).

## How to run

The scripts are tested with Python 3.10+ and PyTorch. Install the dependencies with

```
$ pip install -r requirements.txt
```

Once `dataset_battery/` and `checkpoints/` are in place, the results of the paper can be reproduced from the command line

```
$ python plot_sim_to_real_2.py
$ python soc0_robustness_ablation.py
```

It is preferable to run the training notebook with Jupyter (or to export it to a script and run it through ipython: `$ ipython` and then `run <file_name>`). The training is repeated for 5 seeds.

# Hardware requirements
While all the scripts can run on CPU, the training may be slow. For faster training, a GPU is recommended. To run the paper's examples, we used a server equipped with an NVIDIA RTX 3090 GPU. The evaluation scripts run comfortably on CPU once the checkpoints are available.

# Citing
If you find this project useful, we encourage you to:

* Star this repository :star:
* Cite the [paper](https://github.com/mattrufolo) (link to be updated)

```
@article{rufolo2026structured,
  title   = {Structured Meta-Learning for Battery State-of-Charge Estimation: Data Efficiency and Sim-to-Real Transfer},
  author  = {Rufolo, Matteo and Piga, Dario and others},
  journal = {Applied Energy},
  year    = {2026},
  note    = {under review}
}
```

# License
This project is released under the MIT license, see [LICENSE](LICENSE).