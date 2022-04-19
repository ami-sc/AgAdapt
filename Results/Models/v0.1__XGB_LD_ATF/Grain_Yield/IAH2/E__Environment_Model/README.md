# IAH2 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 24.6722
- Baseline Model [RMSE] = 30.8744
- Trained Model [MAE] = 26.5822
- Trained Model [RMSE] = 33.9951
- Prediction [MAE] = 28.5577
- Prediction [RMSE] = 35.6338
***

### Dataset Statistics

- LOFO Field [Mean] = 157.0696 [bu/A]
- LOFO Field [Standard Deviation] = 30.8065 [bu/A]
- Model Dataset [Mean] = 154.7004 [bu/A]
- Model Dataset [Standard Deviation] = 46.5889 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 | 25.4922 |
|          2 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 | 25.4902 |
|          3 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|          4 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|          5 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|          6 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|          7 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|          8 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|          9 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         10 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         11 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         12 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         13 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         14 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         15 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         16 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         17 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         18 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         19 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |
|         20 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 | 25.4912 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.433  | 25.4473 | 25.4693 | 25.496  | 25.5307 | 25.5487 | 25.5083 | 25.4998 | 25.5066 | 25.4755 |
|          0.2 | 25.4661 | 25.4797 | 25.4666 | 25.4798 | 25.4622 | 25.4649 | 25.4477 | 25.4645 | 25.4631 | 25.4478 |
|          0.3 | 25.484  | 25.4783 | 25.4654 | 25.4493 | 25.4649 | 25.4799 | 25.4791 | 25.4744 | 25.4862 | 25.4742 |
|          0.4 | 25.4664 | 25.4599 | 25.4536 | 25.4653 | 25.4611 | 25.4612 | 25.4572 | 25.4582 | 25.447  | 25.463  |
|          0.5 | 25.4867 | 25.493  | 25.4894 | 25.4838 | 25.4851 | 25.4835 | 25.4982 | 25.5042 | 25.4859 | 25.4744 |
|          0.6 | 25.4918 | 25.486  | 25.495  | 25.482  | 25.4799 | 25.4843 | 25.4845 | 25.4911 | 25.49   | 25.4838 |
|          0.7 | 25.4802 | 25.4838 | 25.4886 | 25.4916 | 25.4911 | 25.4871 | 25.4867 | 25.4819 | 25.4866 | 25.4832 |
|          0.8 | 25.4924 | 25.4898 | 25.4889 | 25.4804 | 25.4871 | 25.487  | 25.4847 | 25.4873 | 25.4855 | 25.4846 |
|          0.9 | 25.4835 | 25.4828 | 25.4842 | 25.4897 | 25.4883 | 25.4878 | 25.4885 | 25.4923 | 25.4894 | 25.4862 |
|          1   | 25.491  | 25.4902 | 25.4912 | 25.491  | 25.4914 | 25.4915 | 25.4904 | 25.4912 | 25.4912 | 25.4902 |

***

### eta Grid Search

| eta   |    0.5 |     0.4 |    0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|--------|---------|--------|---------|---------|---------|---------|
| MAE   | 25.487 | 25.4886 | 25.433 | 25.5614 | 25.4728 | 25.6365 | 60.6005 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 12
- subsample = 0.1
- colsample_bytree = 0.1
- eta = 0.3
- num_boost_round = 23