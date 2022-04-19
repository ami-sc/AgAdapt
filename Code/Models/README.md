`/Models`: This directory contains 1 file:

- `XGBoost_Model.py`:
    - This script trains an XGBoost Model using a given set of predictor features and a target phenotype trait.
    - Model training follows a _Leave-One-Field-Out_ (LOFO) approach, where the data for a target field will be excluded from
      the training dataset and used exclusively for testing.
    - Additionally, a summary containing parameter tuning data is also generated.
