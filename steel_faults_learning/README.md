# Steel Plates Fault Classification - Phased Learning Reference

**Status: worked learning reference, not a completed assignment submission.** The assignment permits AI for learning support and prohibits direct copy-paste submissions. Study the explanations, make and justify your own implementation decisions, execute your work in BITS Virtual Lab, and write your own discussion. Merely changing names, styling, or random seeds does not establish independent work.

The results below were actually computed on **8 November 2026 in the ChatGPT execution environment, not BITS Virtual Lab**. The recorded environment was Python 3.12.14 and scikit-learn 1.8.0. The project has **not** been uploaded to your GitHub account or deployed to Streamlit Community Cloud. Repository links, deployment links, and genuine BITS execution evidence remain your tasks.

## a. Problem statement

Build and compare five classifiers that assign a steel plate fault to one of seven categories using 27 numerical measurements. Compare their performance on the same held-out examples, explain their different error patterns, and demonstrate saved models through an interactive Streamlit test-data evaluation app.

This is multiclass classification of already described faults. The dataset does not contain a separate "no defect" class, and the app accepts numerical CSV measurements rather than raw inspection images. The exercise therefore evaluates fault-type classification, not an end-to-end image defect detector.

## b. Dataset description

The selected dataset is [Steel Plates Faults from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/198/steel+plates+faults). It provides a useful industrial inspection example while meeting the assignment's minimum sizes.

| Property | Value in this reference |
|---|---|
| Task | Single-label, seven-class classification |
| Instances | 1,941 |
| Input features | 27, excluding every target indicator |
| Minimum assignment requirements | At least 500 instances and 12 input features; both met |
| Feature types | Integer and real-valued measurements, including encoded steel-type indicators |
| Example inputs | `Pixels_Areas`, `X_Perimeter`, `Y_Perimeter`, `Steel_Plate_Thickness`, `Luminosity_Index` |
| Original targets | Seven mutually exclusive indicator columns |
| Processed target | One string column named `target` |
| Missing input values | 0 in the inspected source |
| Duplicate complete rows / duplicate feature rows | 0 / 0 in the inspected source |
| Train/test split | Stratified 80% / 20%; 1,552 training and 389 test rows |
| Reproducibility | Split seed 42, recorded row membership, source and test-file hashes |

The source label `K_Scatch` is retained exactly as spelled by UCI. Changing its spelling in only an uploaded CSV would create an unknown target label.

| Fault class | Full dataset | Training | Test |
|---|---:|---:|---:|
| Bumps | 402 | 321 | 81 |
| Dirtiness | 55 | 44 | 11 |
| K_Scatch | 391 | 313 | 78 |
| Other_Faults | 673 | 538 | 135 |
| Pastry | 158 | 126 | 32 |
| Stains | 72 | 58 | 14 |
| Z_Scratch | 190 | 152 | 38 |
| **Total** | **1,941** | **1,552** | **389** |

These counts are calculated from the downloaded data and split; see `data/dataset_manifest.json` and `results/class_distribution.csv`. The classes are imbalanced, so this reference uses macro F1 for baseline selection and examines per-class recall alongside accuracy.

### Target preparation and leakage prevention

`prepare_data.py` verifies that every source row has exactly one active class indicator. It creates `target` from that indicator and removes **all seven original class columns** from the predictors. Keeping any of those indicators among the inputs would expose the answer to the model.

The source contains complete numerical data, so this reference does not impute missing values or apply categorical one-hot encoding. Uploaded test files must contain all 27 named features and the true `target`, in raw feature units. The app reorders features to the training order and rejects missing values, nonnumeric features, unexpected columns, and unknown labels.

### Dataset attribution and permitted reuse

Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates Faults* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5J88N](https://doi.org/10.24432/C5J88N).

UCI lists the dataset under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Keep this attribution when redistributing the source or derived CSVs. Changes made here are conversion of the seven target indicators into one categorical label, CSV formatting, and a reproducible train/test split. The original source archive is included under `data/` so dataset preparation can run without a new download; installing Python dependencies still requires them to be available.

## c. GitHub Repository Link

**GitHub Repository Link: PENDING - replace with the actual accessible repository URL for your own work.**

**Live Streamlit App Link: PENDING - replace with the actual working Streamlit Community Cloud URL after deployment.**

Neither placeholder is a submission link. Verify both real links in a fresh browser session before making the final PDF.

## d. Models used

### Experiment design

All five models use the same 27 inputs and the same fixed train/test split. A five-fold shuffled `StratifiedKFold`, using seed 42, evaluates each fixed baseline **within the 1,552 training rows only**. The baseline with the highest mean training-only cross-validation macro F1 is selected. No hyperparameter search or choice of random seed was made using test results.

Each model is then fitted on the full training split and evaluated on the same 389 held-out rows. Scaling is part of the saved `Pipeline`, so each cross-validation fold learns its own scaler from that fold's training portion. The saved pipelines apply the learned transformation during inference. See the [scikit-learn guidance on preprocessing and data leakage](https://scikit-learn.org/stable/common_pitfalls.html).

| ML Model Name | Fixed reference configuration | Preprocessing |
|---|---|---|
| Logistic Regression | `C=1.0`, `max_iter=5000`, default `lbfgs` solver | `StandardScaler` |
| Decision Tree | `max_depth=10`, `min_samples_leaf=3`, `random_state=42` | Original numerical units |
| KNN | `n_neighbors=7`, `weights="distance"`, Euclidean distance (`p=2`) | `StandardScaler` |
| Naive Bayes | `GaussianNB(var_smoothing=1e-9)` | `StandardScaler` |
| Support Vector Machine | `SVC(C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=42)` | `StandardScaler` |

Gaussian Naive Bayes is used because these are numerical measurements, rather than word/event count vectors for Multinomial Naive Bayes. SVM probability estimation is enabled to supply the class probabilities used for multiclass AUC. These are documented baselines, not claims of optimal parameter settings.

### Evaluation metric conventions

- **Accuracy:** correctly predicted rows divided by all evaluated rows.
- **Precision, recall, and F1:** unweighted macro averages across the same seven classes. Macro F1 averages the seven individual F1 values; it is not the harmonic mean of the displayed macro precision and recall.
- **AUC:** macro one-versus-rest ROC-AUC, computed from `predict_proba` with columns aligned to `pipeline.classes_`. Predicted class labels are not passed as AUC scores. See the [official ROC-AUC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- **MCC:** multiclass Matthews correlation coefficient calculated from the predictions and true labels.

The app sets undefined per-class precision/recall/F1 values to zero and continues averaging over all seven labels. If an uploaded CSV contains no true examples of one or more classes, the app displays **AUC unavailable** with an explanation. It does not silently switch to a different AUC definition. Use the complete `test_data.csv` to reproduce the reference comparison.

### Comparison on the held-out test set

**Actual reference results, not BITS Lab results.** Values are rounded to four decimals; full precision is in `results/comparison.csv`. Every row below uses the same 389 examples.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7275 | 0.9379 | 0.7613 | 0.7305 | 0.7418 | 0.6487 |
| Decision Tree | 0.7481 | 0.9056 | 0.7784 | 0.7364 | 0.7551 | 0.6728 |
| KNN | 0.7558 | 0.9251 | 0.7453 | 0.7687 | 0.7546 | 0.6884 |
| Naive Bayes | 0.6093 | 0.9225 | 0.6333 | 0.7169 | 0.6339 | 0.5484 |
| Support Vector Machine | 0.7635 | 0.9566 | 0.8018 | 0.7674 | 0.7810 | 0.6949 |

### Training-only model selection

| ML Model Name | Mean CV macro F1 | Fold sample standard deviation | Training accuracy |
|---|---:|---:|---:|
| Logistic Regression | 0.7220 | 0.0263 | 0.7339 |
| Decision Tree | 0.7379 | 0.0132 | 0.8505 |
| KNN | 0.7676 | 0.0222 | 1.0000 |
| Naive Bayes | 0.6153 | 0.0264 | 0.5999 |
| Support Vector Machine | 0.7760 | 0.0187 | 0.8860 |

Source: `results/cv_results.csv`. Fold standard deviations describe variability in these folds, not confidence intervals or a statistical significance test. Distance-weighted KNN can classify its training rows perfectly because each queried training point is its own zero-distance neighbor. Its training accuracy alone is not evidence of superior generalization.

### Observations about model performance

These are worked examples of evidence-based interpretation. Inspect your own outputs and write your own discussion after your lab run.

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Test macro F1 is 0.7418 and AUC is 0.9379. It recalls 18/32 Pastry examples and 46/81 Bumps examples, with 30 Bumps rows predicted as Other_Faults. A linear model offers a useful baseline; whether nonlinear class boundaries explain these errors is a hypothesis to investigate, not established by the scores alone. |
| Decision Tree | Test macro F1 is 0.7551. K_Scatch recall is 74/78, while Pastry recall is only 12/32. Training accuracy is 0.8505 versus test accuracy 0.7481, showing a generalization gap. Testing depth/leaf-size choices using training-only validation would investigate whether regularization improves that gap. |
| KNN | Test accuracy is 0.7558 and macro recall 0.7687, the highest macro recall in this comparison, narrowly above SVM. Macro F1 is 0.7546. K_Scatch recall is 75/78 but Pastry recall is 13/32, so good aggregate performance does not imply equal performance on all fault types. |
| Naive Bayes | Test accuracy is 0.6093 and macro F1 0.6339, the lowest here. It recalls only 34/135 Other_Faults rows, while its AUC remains 0.9225. AUC evaluates score rankings, whereas accuracy/F1 evaluate chosen labels, so these values need not track each other. Dependence between measurements or non-Gaussian class distributions are possible explanations to check; this run does not prove either explanation. |
| Support Vector Machine | Test macro F1 is 0.7810 and AUC 0.9566, both highest in this run. It also has the highest accuracy, macro precision, and MCC. Remaining weaknesses include Pastry recall of 18/32 and confusion between Bumps and Other_Faults. It does not have the highest macro recall. |
| **Overall, winner for this dataset?** | **SVM is the preferred reference baseline under the declared training-only CV rule:** mean macro F1 0.7760 versus KNN 0.7676. Its held-out macro F1 is also highest. The small CV difference and one held-out split do not establish universal or statistically significant superiority. |

For the selected SVM, 297 of 389 test labels are correct. Of its 92 errors, 20 are Bumps predicted as Other_Faults and 23 are the reverse. This pair accounts for 43 errors and is a concrete starting point for error analysis. Dirtiness has only 11 test examples, so a single changed prediction changes its recall by about 9.09 percentage points. These calculations come from `results/svm_confusion_matrix.csv` and `results/svm_classification_report.csv`.

## Running the learning reference

Extract the complete ZIP first. Open `ML_Assignment_2_Phased_Guide.ipynb` **inside the extracted project folder**; it imports the accompanying Python files and uses their data/models. Opening only a separately downloaded notebook is insufficient.

Use a Python 3.12 environment. In a Linux/Bash terminal, from the folder containing the extracted project:

```bash
cd steel_faults_learning
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
```

In Windows PowerShell, the environment creation/activation equivalents are:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

Run dataset preparation and inspect its output:

```bash
python prepare_data.py
```

Study and adapt the model code, then run training. The context argument must describe where you actually execute it; the text does not prove where execution occurred.

```bash
python train.py --execution-context "My actual execution environment - replace this description"
```

Only if this command is actually being run in BITS Virtual Lab, use:

```bash
python train.py --execution-context "BITS Virtual Lab"
```

Training regenerates `test_data.csv`, all five saved pipelines, metadata, and result files. If you change the experiment, update the README numbers and interpretations from your new outputs rather than retaining the reference table. The app evaluates the saved files; it does not train them:

```bash
python -m streamlit run app.py
```

For notebook use with the lab's existing Jupyter or VS Code interface, install the optional notebook dependencies and select this Python environment as the kernel:

```bash
python -m pip install -r requirements-notebook.txt
```

The pinned requirements match the reference ML artifacts. Keep the same scikit-learn version when loading them, or regenerate all models and update the dependency pins together in a compatible environment. The app reports a version mismatch instead of silently accepting mismatched artifacts.

## Project file roles

| File or directory | Role |
|---|---|
| `START_HERE.txt` | Short practical checklist in the assignment's phase order |
| `ML_Assignment_2_Phased_Guide.ipynb` | Guided explanations, executable exploration, and learning checkpoints |
| `prepare_data.py` | Source loading, target conversion, schema checks, and provenance |
| `ml_core.py` | Feature/label contract, model definitions, metric implementation |
| `train.py` | Training-only CV, final fitting, held-out evaluation, saved outputs |
| `app.py` | Streamlit CSV upload, model selection, metrics, comparison, error inspection |
| `requirements.txt` | Pinned app/training dependencies |
| `requirements-notebook.txt` | Additional notebook execution dependencies |
| `test_data.csv` | Raw-unit held-out features plus true target; use this in the app |
| `model/` | All five fitted pipelines and experiment metadata |
| `data/` | Attributed UCI source, processed data, and data manifest |
| `results/` | Test/CV tables, per-class reports, predictions, matrices, figures, split membership |
| `README.md` | Required problem/data/model discussion plus learning/run guidance |
| `submission_outline.txt` | Pending evidence and exact final PDF order; not a submission |

## Work to complete yourself, following the assignment phases

### Phase 1 - Choose and understand the dataset

Verify the minimum sizes, explain why this dataset fits your problem, and trace how target indicators become one label. Explain why those indicators cannot remain inputs. Keep the public dataset attribution.

### Phase 2 - Implement, evaluate, and explain

Reimplement or adapt the approach with decisions you can justify. Before additional experimentation, decide which metric you will optimize and restrict all tuning to the training folds. Do not repeatedly try random splits and keep the one with the best test score. Calculate all six required metrics for all five models on one common test set.

Useful learning exercises are to hand-calculate one class's precision/recall from its confusion matrix, explain macro versus weighted averages, trace a pipeline through one CV fold, and examine the Bumps/Other_Faults errors. If you experiment with hyperparameters, record the training-only CV evidence and the reason for your final choice. Treat the already-visible reference test results as benchmark illustrations, not a fresh source of tuning feedback.

### Phase 3 - Create your GitHub repository

Maintain genuine commits as you develop your own work. Include `app.py`, training source, `requirements.txt`, your updated `README.md`, `test_data.csv`, and all five pipeline files in `model/`. Include the modules that `app.py` imports, particularly `ml_core.py`. Keep the lab-generated preprocessing/model artifacts together. Do not upload a virtual environment or account credentials.

### Phase 4 - Verify dependencies

Dependency installation is an execution prerequisite even though the assignment lists this as Step 4. Verify the requirements in a clean compatible environment and retain the versions used for your saved models. Include `joblib` because it is used for model serialization.

### Phase 5 - Write your README

Keep sections **a, b, c, d** in the required order. Replace pending links with real links, state your actual execution context, and update all tables from your own run. Write your model observations and explain the selection criterion in your own words. Revisit the app footer and documentation so any retained reference material and your own completed work are accurately identified.

### Phase 6 - Build, test, and deploy the Streamlit app

Understand and customize the interaction: upload a labelled test CSV, choose a saved classifier, show all six metrics, and provide a confusion matrix or classification report. The app must evaluate uploaded rows without retraining. Test the complete test CSV, switch all five models, and inspect the same-row comparison table. Also check the visible response to a missing target column or a subset containing fewer than seven classes.

Create a Streamlit Community Cloud app from your GitHub repository, select the correct branch and `app.py`, and choose Python 3.12 in deployment settings. The [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) describes the current flow. Open the deployed URL and repeat the upload/model checks. A screenshot or a localhost URL does not replace the required live app link.

### Genuine lab evidence and final submission

Perform the assignment on BITS Virtual Lab and capture **one genuine screenshot** showing that environment and your execution/results clearly. An `execution_context` string is documentation, not evidence; this reference's output is not a substitute for the lab run.

Submit **one PDF**, in this exact order:

1. Your real, accessible GitHub repository link.
2. Your real, working Streamlit Community Cloud app link.
3. One screenshot of your assignment execution on BITS Virtual Lab.
4. The full revised GitHub README content, including all metrics and model observations.

Check links are clickable in the PDF and open for the evaluator. The assignment specifies **8 November 2026 at 23:59** and says to submit the final work rather than leave a draft. Its text does not state a timezone; follow the course portal's displayed deadline. This learning package does not itself complete or submit those requirements.
