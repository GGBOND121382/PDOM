# Run adv_setting/run_experiments.py on Google Colab

This guide shows how to run `adv_setting/run_experiments.py` in Colab after the
code is pushed to GitHub.

## 1) Open a new Colab notebook
Go to https://colab.research.google.com and create a new notebook.

## 2) Clone your repository
Replace `YOUR_GITHUB_REPO_URL` with your repo URL.

```bash
!git clone https://github.com/GGBOND121382/PDOM.git
```

If the repo is private, use a personal access token or GitHub CLI:

```bash
!git clone https://<TOKEN>@github.com/<USER>/<REPO>.git
```

## 3) Install Python dependencies
If your repo has a `requirements.txt`, use it:

```bash
!pip -q install -r requirements.txt
```

If not, install the minimal set used by this project:

```bash
!pip -q install numpy scikit-learn matplotlib
```

## 4) Run the experiments
Change directory into the repo, then run:

```bash
%cd <REPO_DIR_NAME>
!python adv_setting/run_experiments.py
```

## 5) Check outputs
Results are written to:

- `adv_setting/experiment_results/*.csv`
- `adv_setting/experiment_results/*.png`

You can view the output files from Colab’s file browser or download them:

```bash
from google.colab import files
files.download('adv_setting/experiment_results/impact_learning_time_summary.csv')
files.download('adv_setting/experiment_results/impact_learning_time.png')
```

## 6) Save outputs directly to Google Drive (persistent storage)
Colab storage resets after the session ends. You can write results straight to
Drive by setting `PDOM_OUTPUT_DIR` before running the script.

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%env PDOM_OUTPUT_DIR=/content/drive/MyDrive/PDOM_results
!python adv_setting/run_experiments.py
```

You can change the destination path to any folder in your Drive.

## Notes
- The script uses the `data/adv_setting/bodyfat` dataset from the repo.
  Make sure that file exists in your GitHub project.
- Running all experiments can take a long time on Colab. You can edit
  `adv_setting/run_experiments.py` to reduce `runs` or shrink the T grid
  if you want a quick sanity check.
