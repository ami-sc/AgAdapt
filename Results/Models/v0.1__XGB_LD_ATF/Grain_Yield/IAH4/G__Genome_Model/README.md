# IAH4 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 33.1402
- Baseline Model [RMSE] = 40.9132
- Trained Model [MAE] = 35.7902
- Trained Model [RMSE] = 44.7129
- Prediction [MAE] = 30.2474
- Prediction [RMSE] = 37.1218
***

### Dataset Statistics

- LOFO Field [Mean] = 179.2561 [bu/A]
- LOFO Field [Standard Deviation] = 31.1966 [bu/A]
- Model Dataset [Mean] = 152.7585 [bu/A]
- Model Dataset [Standard Deviation] = 45.9602 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 35.9031 | 35.9031 | 35.9031 | 35.9277 | 35.9277 | 35.9277 | 35.9277 | 35.9277 | 35.9277 | 35.9289 | 35.9869 | 35.9869 | 35.9568 | 35.9568 | 35.9568 | 35.9568 | 35.9833 | 35.9833 | 35.9833 | 35.9833 | 35.9833 |
|          2 | 35.4587 | 35.4587 | 35.4587 | 35.436  | 35.4473 | 35.3932 | 35.3795 | 35.3487 | 35.3937 | 35.4289 | 35.4522 | 35.3831 | 35.4949 | 35.4798 | 35.4924 | 35.419  | 35.3563 | 35.4539 | 35.4254 | 35.5381 | 35.4699 |
|          3 | 35.2673 | 35.2673 | 35.3298 | 35.3531 | 35.3435 | 35.3477 | 35.326  | 35.312  | 35.2971 | 35.3375 | 35.3403 | 35.338  | 35.37   | 35.3734 | 35.4064 | 35.4459 | 35.4349 | 35.4651 | 35.379  | 35.4175 | 35.5006 |
|          4 | 35.2864 | 35.2864 | 35.3285 | 35.2845 | 35.2811 | 35.2891 | 35.2889 | 35.3325 | 35.3487 | 35.3835 | 35.3837 | 35.4092 | 35.3999 | 35.377  | 35.3667 | 35.3891 | 35.3834 | 35.382  | 35.409  | 35.4195 | 35.4176 |
|          5 | 35.3616 | 35.3616 | 35.3443 | 35.3963 | 35.3806 | 35.3934 | 35.3485 | 35.336  | 35.32   | 35.3494 | 35.3741 | 35.3484 | 35.3998 | 35.4082 | 35.3353 | 35.411  | 35.3997 | 35.3932 | 35.4209 | 35.4103 | 35.3873 |
|          6 | 35.3726 | 35.3726 | 35.3775 | 35.3297 | 35.3374 | 35.3399 | 35.3626 | 35.2823 | 35.3164 | 35.3206 | 35.3649 | 35.4006 | 35.3723 | 35.3416 | 35.3757 | 35.3864 | 35.3695 | 35.3128 | 35.3897 | 35.3903 | 35.3817 |
|          7 | 35.3953 | 35.3953 | 35.3697 | 35.3382 | 35.3088 | 35.3177 | 35.3037 | 35.3434 | 35.3362 | 35.3695 | 35.3651 | 35.3924 | 35.3948 | 35.3758 | 35.3765 | 35.3515 | 35.3618 | 35.4186 | 35.3652 | 35.3737 | 35.3976 |
|          8 | 35.4393 | 35.4393 | 35.4403 | 35.4359 | 35.4232 | 35.4541 | 35.4248 | 35.4048 | 35.3736 | 35.4093 | 35.3851 | 35.3836 | 35.3804 | 35.3818 | 35.3844 | 35.3762 | 35.3666 | 35.418  | 35.3819 | 35.3705 | 35.3636 |
|          9 | 35.4789 | 35.4789 | 35.4678 | 35.5146 | 35.4435 | 35.4328 | 35.4564 | 35.4369 | 35.4185 | 35.4148 | 35.427  | 35.3998 | 35.4043 | 35.393  | 35.3791 | 35.385  | 35.409  | 35.372  | 35.3909 | 35.3979 | 35.4527 |
|         10 | 35.4913 | 35.4913 | 35.4975 | 35.4926 | 35.4791 | 35.4642 | 35.4808 | 35.4127 | 35.4596 | 35.3966 | 35.451  | 35.4361 | 35.4303 | 35.4402 | 35.4432 | 35.4316 | 35.4085 | 35.4109 | 35.4207 | 35.4012 | 35.3723 |
|         11 | 35.604  | 35.604  | 35.6016 | 35.5689 | 35.5321 | 35.5225 | 35.5285 | 35.4833 | 35.4576 | 35.461  | 35.4996 | 35.4653 | 35.4651 | 35.4348 | 35.4645 | 35.4423 | 35.4127 | 35.428  | 35.4529 | 35.4059 | 35.3656 |
|         12 | 35.6253 | 35.6253 | 35.6048 | 35.6274 | 35.5944 | 35.6146 | 35.5692 | 35.5458 | 35.5302 | 35.4969 | 35.5207 | 35.5154 | 35.4644 | 35.4546 | 35.4493 | 35.4867 | 35.4563 | 35.4534 | 35.4591 | 35.4392 | 35.4365 |
|         13 | 35.6621 | 35.6621 | 35.662  | 35.6499 | 35.648  | 35.6005 | 35.6061 | 35.5829 | 35.5684 | 35.5164 | 35.517  | 35.542  | 35.5219 | 35.4832 | 35.4463 | 35.4817 | 35.5034 | 35.4566 | 35.4645 | 35.406  | 35.4442 |
|         14 | 35.7395 | 35.7395 | 35.7311 | 35.698  | 35.684  | 35.6357 | 35.6138 | 35.5818 | 35.5669 | 35.5942 | 35.5569 | 35.5572 | 35.5502 | 35.5134 | 35.5007 | 35.5274 | 35.5252 | 35.4908 | 35.4776 | 35.4717 | 35.456  |
|         15 | 35.7538 | 35.7538 | 35.7396 | 35.7301 | 35.6895 | 35.6735 | 35.6853 | 35.6515 | 35.6187 | 35.5983 | 35.6244 | 35.5677 | 35.5575 | 35.5748 | 35.5268 | 35.5519 | 35.517  | 35.5346 | 35.5229 | 35.4759 | 35.4579 |
|         16 | 35.8268 | 35.8268 | 35.7962 | 35.739  | 35.7534 | 35.7215 | 35.7056 | 35.6384 | 35.6412 | 35.6061 | 35.6044 | 35.6215 | 35.5857 | 35.555  | 35.5309 | 35.5569 | 35.543  | 35.5279 | 35.5203 | 35.4653 | 35.4863 |
|         17 | 35.8211 | 35.8211 | 35.8093 | 35.8095 | 35.7539 | 35.7593 | 35.7364 | 35.6769 | 35.6396 | 35.643  | 35.6241 | 35.6233 | 35.6296 | 35.5883 | 35.5683 | 35.5598 | 35.5638 | 35.5707 | 35.5121 | 35.4839 | 35.4908 |
|         18 | 35.8488 | 35.8488 | 35.8327 | 35.8142 | 35.7948 | 35.7451 | 35.7614 | 35.712  | 35.6797 | 35.6556 | 35.6579 | 35.6542 | 35.614  | 35.571  | 35.5965 | 35.5801 | 35.5645 | 35.55   | 35.5525 | 35.5198 | 35.5043 |
|         19 | 35.8781 | 35.8781 | 35.8411 | 35.8096 | 35.8003 | 35.7594 | 35.7755 | 35.7167 | 35.6875 | 35.6622 | 35.6622 | 35.6477 | 35.6299 | 35.6329 | 35.591  | 35.5877 | 35.5967 | 35.5747 | 35.5469 | 35.5104 | 35.5222 |
|         20 | 35.8848 | 35.8848 | 35.8744 | 35.8547 | 35.8371 | 35.8019 | 35.7693 | 35.7457 | 35.7049 | 35.7018 | 35.6702 | 35.6638 | 35.645  | 35.6179 | 35.6145 | 35.5943 | 35.6009 | 35.5747 | 35.5664 | 35.5344 | 35.5164 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 36.4169 | 36.4169 | 36.4169 | 36.4169 | 36.4169 | 36.4169 | 36.4169 | 36.4169 | 36.4169 | 36.0415 |
|          0.2 | 35.9699 | 35.9699 | 35.9699 | 35.9699 | 35.9699 | 35.9699 | 35.9699 | 35.9699 | 35.9699 | 35.9957 |
|          0.3 | 35.8426 | 35.8426 | 35.8426 | 35.8426 | 35.8426 | 35.8426 | 35.8426 | 35.8426 | 35.8426 | 35.7406 |
|          0.4 | 35.694  | 35.694  | 35.694  | 35.694  | 35.694  | 35.694  | 35.694  | 35.694  | 35.694  | 35.6286 |
|          0.5 | 35.6263 | 35.6263 | 35.6263 | 35.6263 | 35.6263 | 35.6263 | 35.6263 | 35.6263 | 35.6263 | 35.6634 |
|          0.6 | 35.6372 | 35.6372 | 35.6372 | 35.6372 | 35.6372 | 35.6372 | 35.6372 | 35.6372 | 35.6372 | 35.4963 |
|          0.7 | 35.4767 | 35.4767 | 35.4767 | 35.4767 | 35.4767 | 35.4767 | 35.4767 | 35.4767 | 35.4767 | 35.4802 |
|          0.8 | 35.4079 | 35.4079 | 35.4079 | 35.4079 | 35.4079 | 35.4079 | 35.4079 | 35.4079 | 35.4079 | 35.402  |
|          0.9 | 35.3715 | 35.3715 | 35.3715 | 35.3715 | 35.3715 | 35.3715 | 35.3715 | 35.3715 | 35.3715 | 35.3947 |
|          1   | 35.4857 | 35.4857 | 35.4857 | 35.4857 | 35.4857 | 35.4857 | 35.4857 | 35.4857 | 35.4857 | 35.2673 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 35.3879 | 35.2672 | 35.2673 | 35.3272 | 35.2978 | 35.4839 | 62.1992 |

***

### Tuned Parameters

- max_depth = 3
- min_child_weight = 0
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.4
- num_boost_round = 65