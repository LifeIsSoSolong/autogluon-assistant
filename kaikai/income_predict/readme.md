## Task Overview

This toy project exercises the full AutoGluon Assistant pipeline on a census-style dataset. Each record describes an individual using demographic and employment attributes. The goal is to predict whether that person’s annual income exceeds $50K.

- **Input directory**: `kaikai/income_predict`
- **Training data**: `kaikai/income_predict/data/train.csv`
- **Test data**: `kaikai/income_predict/data/test.csv`
- **Target column**: the final column in each CSV (named `income` in the training set)
- **Prediction requirement**: produce the missing `income` values for the test set

## Data Details

The CSV files share the same schema. Representative feature columns include:

- `age`
- `workclass`
- `fnlwgt`
- `education` / `education-num`
- `marital-status`
- `occupation`
- `relationship`
- `race`
- `sex`
- `capital-gain`
- `capital-loss`
- `hours-per-week`
- `native-country`

Additional columns may appear; treat all non-target fields as usable predictors. The target column (`income`) is binary with labels such as "`<=50K`" and "`>50K`". In the test set this column is blank and must be filled with model predictions.

Assume the dataset may contain missing values, mixed categorical levels across splits, and class imbalance.

## Modeling Requirements

1. Treat the problem as **binary classification**.
2. Prioritize the following evaluation metrics when comparing candidate solutions:
   - Accuracy
   - Recall
   - Precision
3. Generate at least one runnable training script or notebook that can be reused.
4. Output the final predictions in a file compatible with the original test schema (e.g., a copy of `data/test.csv` with the `income` column populated, or an equivalent submission CSV).

## Recommended Workflow for the Assistant

- Perform exploratory analysis on the training data to understand feature distributions and class balance.
- Handle categorical encoding and missing values robustly; ensure the same preprocessing is applied to train and test.
- Consider different models.
- Run basic hyperparameter tuning or ensembling to improve metrics.
- You should split the training data into train and validation sets for model selection and hyperparameter tuning.
- Record evaluation metrics (Accuracy, Recall, Precision) on validation data and save them to the output directory.
- Save all generated code, configuration files, and prediction artifacts inside the designated output path for review.

This README is prepared for AutoGluon Assistant. Feel free to extend it with additional instructions as the project evolves.
