# An Empirical Study of Representation Learning for Reinforcement Learning in Healthcare
Learning representations for RL in Healthcare under a POMDP assumption, honoring the sequential nature that data wherewith data is generate. This paper, accepted to the proceedings of the [2020 Machine Learning for Healthcare Workshop at NeurIPS](https://ml4health.github.io/2020/), empirically evaluates several recurrent autoencoding architectures to assess the quality of the internal representations each learn. 

The motivation for this systematic analysis is that most prior work developing RL solutions for healthcare neglect to rigorously define state representations that respect the partial or sequential nature of the data generating process. We evaluate several recurrent autoencoding architectures, trained by predicting the subsequent physiological observation, by investigating the quality of the internal learned representations of patient state as well as develop treatment policies from them.

## Paper
If you use this code in your research, please cite the following publication [link to PMLR version](http://proceedings.mlr.press/v136/killian20a):
```
@inproceedings{killian2020empirical,
  title={An Empirical Study of Representation Learning for Reinforcement Learning in Healthcare},
  author={Killian, Taylor W and Zhang, Haoran and Subramanian, Jayakumar and Fatemi, Mehdi and Ghassemi, Marzyeh},
  booktitle={Machine Learning for Health},
  pages={139--160},
  year={2020},
  organization={PMLR}
}
```

This paper can also be found on arxiv: [https://arxiv.org/abs/2011.11235](https://arxiv.org/abs/2011.11235)

## To replicate the experiments in the paper:
### Step 0: Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment
```
git clone https://github.com/MLforHealth/rl_representations.git
cd rl_representations/
conda env create -f environment.yml
conda activate rl4h_rep
```

### Step 1: Data Processing
The data used to develop, run and evaluate our experiments is extracted from the [MIMIC-III database](https://mimic.physionet.org/), based on the Sepsis cohort used by [Komorowski, et al (2018)](https://www.nature.com/articles/s41591-018-0213-5). We replicate and refine the code to extract this cohort at the following repository [https://github.com/microsoft/mimic_sepsis](https://github.com/microsoft/mimic_sepsis). For replication purposes, save both the z-normalized and "RAW" features.

This extracted cohort is for general purpose use, for the usage in this paper it required additional preprocessing which we outline here.
- Run `scripts/compute_acuity_scores.py` to compute additional acuity scores with the raw patient features.
- Run `scripts/split_sepsis_cohort.py` to compose a training, validation and testing split as well as remove binary or demographic information from the temporal patient features. This script also organizes the patient data into convenient trajectory formats for easier use with sequential or recurrent models.
- Use Behavior Cloning on the data provided by the previous step to develop a baseline "expert" policy for use in training and evaluating RL policies from the learned patient representations.
  * This is done by running `scripts/train_behavCloning.py`. An example of how this can be done is provided in `slurm_scripts/slurm_build_BC.py` --> `slurm_scripts/slurm_bc_exp`.
  * This will generate either `behav_policy_file` or `behav_policy_file_wDemo` which is used internal to Steps 2 and 3 below.

### Step 2: Learning Representations
- In `configs/common.yaml`, update the paths at the bottom of the file.
- Select a `model` - one of [`AE`, `AIS`, `CDE`, `DDM`, `DST`, `ODERNN`, `RNN`]
- Run `python slurm_build_{model}.py` to output a text file containing an array of launch arguments to `scripts/train_model.py` that were used to generate the results from the paper.
- If you are using a Slurm cluster, you can run `sbatch slurm_{model}_exp` after altering the Slurm header to run a Slurm task array where each job corresponds to training a single model with one line of arguments from the generated file.
- Otherwise, models can also be trained on an ad-hoc basis by manually calling `python scripts/train_model.py` with the arguments shown in the generated text file.
- The configuration files in `config_sepsis_{model}.yaml` contain the hyperparameters used in the paper. These can be changed if desired.

### Step 3: RL Evaluation

RL policies are learned using the discretized form of [Batch Constrained Q-learning](https://github.com/sfujim/BCQ) from [Fujimoto, et al (2019)](https://arxiv.org/abs/1910.01708). These policies are learned as the final part of the previous Step 2 inside of `scripts/train_model.py`. The policies are evaluated intermittently using a form of Weighted Importance Sampling (procedure found in `scripts/dBCQ_utils.py`, line 284).

We compare various evaluations of the learned policies using the iPython notebook: `notebooks/Evaluate\ dBCQ\ policies.ipynb`.

### Step 4: Result Aggregation and Visualizations
We aggregate the results of all model training and the analysis of the learned representations in the notebook `notebooks/Aggregate\ Next\ Step\ Results.ipynb`.

### Prerequisites
See `requirements.txt`

## Authors
* Taylor W. Killian [@twkillian](https://github.com/twkillian)
* Haoran Zhang [@hzhang0](https://github.com/hzhang0)
* Jaykumar Subramanian
* Mehdi Fatemi
* Marzyeh Ghassemi

## License

The source code and documentation are licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT)

## Questions or Feedback?
Please don't hesitate to [log an issue](https://github.com/MLforHealth/rl_representations/issues) and we'll be as prompt as possible when responding.

