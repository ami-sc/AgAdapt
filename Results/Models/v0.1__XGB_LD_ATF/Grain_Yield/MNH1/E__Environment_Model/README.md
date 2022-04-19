# MNH1 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 27.8416
- Baseline Model [RMSE] = 38.0609
- Trained Model [MAE] = 25.8538
- Trained Model [RMSE] = 33.1473
- Prediction [MAE] = 31.9202
- Prediction [RMSE] = 42.5656
***

### Dataset Statistics

- LOFO Field [Mean] = 146.2302 [bu/A]
- LOFO Field [Standard Deviation] = 37.0035 [bu/A]
- Model Dataset [Mean] = 155.3298 [bu/A]
- Model Dataset [Standard Deviation] = 45.8728 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4872 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4873 | 25.4872 |
|          2 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 |
|          3 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 | 25.4853 |
|          4 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|          5 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|          6 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|          7 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|          8 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|          9 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         10 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         11 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         12 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         13 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         14 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         15 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         16 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         17 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         18 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         19 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |
|         20 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 | 25.4854 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.4567 | 25.4868 | 25.4786 | 25.4612 | 25.5123 | 25.4952 | 25.474  | 25.4976 | 25.5116 | 25.4771 |
|          0.2 | 25.4569 | 25.448  | 25.4228 | 25.4486 | 25.4616 | 25.4577 | 25.4485 | 25.4405 | 25.4404 | 25.448  |
|          0.3 | 25.4978 | 25.4752 | 25.4734 | 25.4759 | 25.4876 | 25.4807 | 25.5019 | 25.4736 | 25.4957 | 25.4722 |
|          0.4 | 25.4734 | 25.4798 | 25.4696 | 25.4768 | 25.4821 | 25.4779 | 25.4708 | 25.4693 | 25.4663 | 25.4727 |
|          0.5 | 25.4775 | 25.4735 | 25.4763 | 25.4746 | 25.462  | 25.4668 | 25.4811 | 25.4756 | 25.4796 | 25.4734 |
|          0.6 | 25.4591 | 25.4909 | 25.4791 | 25.4693 | 25.4669 | 25.461  | 25.4674 | 25.4748 | 25.4792 | 25.4748 |
|          0.7 | 25.4744 | 25.4739 | 25.4822 | 25.4764 | 25.4831 | 25.4788 | 25.4795 | 25.4798 | 25.4843 | 25.476  |
|          0.8 | 25.484  | 25.4763 | 25.4841 | 25.4749 | 25.474  | 25.4781 | 25.4726 | 25.477  | 25.4754 | 25.4752 |
|          0.9 | 25.4835 | 25.4838 | 25.4832 | 25.4806 | 25.479  | 25.4809 | 25.4773 | 25.475  | 25.4783 | 25.4807 |
|          1   | 25.4838 | 25.4819 | 25.4845 | 25.4846 | 25.48   | 25.4818 | 25.4829 | 25.4846 | 25.4854 | 25.4853 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |    0.1 |   0.01 |   0.001 |
|-------|---------|---------|---------|---------|--------|--------|---------|
| MAE   | 25.4678 | 25.4238 | 25.4228 | 25.4745 | 25.483 | 25.534 | 59.9334 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 11
- subsample = 0.2
- colsample_bytree = 0.3
- eta = 0.3
- num_boost_round = 36