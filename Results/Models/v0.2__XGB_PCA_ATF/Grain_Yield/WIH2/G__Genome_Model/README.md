# WIH2 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 37.2996
- Baseline Model [RMSE] = 43.9803
- Trained Model [MAE] = 36.2511
- Trained Model [RMSE] = 45.5187
- Prediction [MAE] = 34.0781
- Prediction [RMSE] = 40.1529
***

### Dataset Statistics

- LOFO Field [Mean] = 183.3416 [bu/A]
- LOFO Field [Standard Deviation] = 31.5944 [bu/A]
- Model Dataset [Mean] = 152.7188 [bu/A]
- Model Dataset [Standard Deviation] = 45.6962 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 35.6992 | 35.6992 | 35.6992 | 35.6992 | 35.6992 | 35.6992 | 35.6992 | 35.6992 | 35.6644 | 35.7518 | 35.7518 | 35.6833 | 35.6833 | 35.6833 | 35.6833 | 35.6833 | 35.6905 | 35.6905 | 35.7164 | 35.7164 | 35.7509 |
|          2 | 35.1133 | 35.1133 | 35.1133 | 35.0996 | 35.102  | 35.1191 | 35.1703 | 35.1638 | 35.2029 | 35.1453 | 35.1854 | 35.1515 | 35.1545 | 35.1408 | 35.1925 | 35.1192 | 35.3421 | 35.1558 | 35.1518 | 35.2074 | 35.2779 |
|          3 | 35.037  | 35.037  | 35.0417 | 35.0542 | 35.0288 | 35.0066 | 35.0331 | 35.088  | 35.078  | 35.0364 | 35.0856 | 35.0347 | 35.0683 | 35.0523 | 35.1024 | 35.0444 | 35.0701 | 35.0877 | 35.1077 | 35.1081 | 35.0964 |
|          4 | 35.0086 | 35.0086 | 35.0246 | 35.0393 | 35.0248 | 35.0252 | 35.041  | 34.986  | 35.0053 | 35.0002 | 35.0162 | 35.0054 | 35.0054 | 35.0473 | 34.9977 | 35.0273 | 35.0333 | 35.0231 | 35.0157 | 35.0315 | 35.0106 |
|          5 | 35.0009 | 35.0009 | 34.982  | 35.0139 | 35.002  | 35.0206 | 35.0256 | 35.0505 | 35.0475 | 35.0354 | 35.0541 | 35.0473 | 34.9625 | 35.0133 | 35.0102 | 35.0611 | 35.0229 | 35.0619 | 35.0066 | 35.0063 | 35.0374 |
|          6 | 35.0455 | 35.0455 | 35.0235 | 35.0138 | 34.9887 | 35.0076 | 35.0384 | 35.0336 | 34.9842 | 35.0385 | 35.0358 | 35.0315 | 35.0551 | 35.0283 | 35.0527 | 35.0527 | 35.1031 | 35.0632 | 35.0465 | 35.0161 | 35.0473 |
|          7 | 35.0095 | 35.0095 | 35.0125 | 35.0471 | 35.0461 | 35.0371 | 35.0145 | 35.0261 | 35.0361 | 35.0539 | 35.0174 | 35.0815 | 35.0079 | 35.0385 | 35.0346 | 35.0758 | 35.0556 | 35.0824 | 35.0198 | 34.9967 | 35.0406 |
|          8 | 35.0967 | 35.0967 | 35.0983 | 35.0823 | 35.0735 | 35.0657 | 35.0822 | 35.0887 | 35.0347 | 35.065  | 35.0708 | 35.0878 | 35.1316 | 35.094  | 35.0512 | 35.04   | 35.0575 | 35.0674 | 35.0895 | 35.0766 | 35.0288 |
|          9 | 35.099  | 35.099  | 35.1769 | 35.1164 | 35.1275 | 35.1306 | 35.1125 | 35.132  | 35.1246 | 35.132  | 35.1152 | 35.1293 | 35.1394 | 35.115  | 35.0538 | 35.0809 | 35.1136 | 35.082  | 35.0173 | 35.0577 | 35.0327 |
|         10 | 35.2186 | 35.2186 | 35.1966 | 35.1914 | 35.1717 | 35.1784 | 35.1719 | 35.1941 | 35.2185 | 35.1447 | 35.1766 | 35.1421 | 35.1311 | 35.1544 | 35.1409 | 35.0945 | 35.1587 | 35.1149 | 35.092  | 35.0643 | 35.0781 |
|         11 | 35.2818 | 35.2818 | 35.2777 | 35.2438 | 35.2331 | 35.2596 | 35.2123 | 35.2648 | 35.2339 | 35.1997 | 35.2262 | 35.2027 | 35.1964 | 35.199  | 35.2209 | 35.192  | 35.1575 | 35.1172 | 35.1153 | 35.1466 | 35.1086 |
|         12 | 35.3193 | 35.3193 | 35.3257 | 35.3274 | 35.3416 | 35.3344 | 35.2603 | 35.3135 | 35.2947 | 35.2692 | 35.2299 | 35.2657 | 35.2413 | 35.2781 | 35.2104 | 35.1661 | 35.1928 | 35.1665 | 35.177  | 35.1739 | 35.1308 |
|         13 | 35.3817 | 35.3817 | 35.3474 | 35.3628 | 35.3798 | 35.3548 | 35.3561 | 35.3507 | 35.3154 | 35.2958 | 35.3018 | 35.2982 | 35.264  | 35.2671 | 35.2054 | 35.2154 | 35.1983 | 35.1653 | 35.1702 | 35.1953 | 35.1768 |
|         14 | 35.4045 | 35.4045 | 35.4531 | 35.4033 | 35.4068 | 35.3991 | 35.3561 | 35.3887 | 35.3514 | 35.354  | 35.3101 | 35.3056 | 35.291  | 35.3372 | 35.2424 | 35.251  | 35.2426 | 35.2371 | 35.2107 | 35.2153 | 35.1844 |
|         15 | 35.4183 | 35.4183 | 35.4751 | 35.5023 | 35.4585 | 35.412  | 35.4219 | 35.4121 | 35.3931 | 35.3497 | 35.348  | 35.3314 | 35.3545 | 35.325  | 35.2919 | 35.2944 | 35.2657 | 35.2563 | 35.2017 | 35.2517 | 35.2042 |
|         16 | 35.4874 | 35.4874 | 35.489  | 35.5052 | 35.4834 | 35.4835 | 35.438  | 35.4343 | 35.4042 | 35.3669 | 35.3539 | 35.3768 | 35.3636 | 35.3495 | 35.2992 | 35.2713 | 35.2846 | 35.294  | 35.2575 | 35.2528 | 35.2418 |
|         17 | 35.5293 | 35.5293 | 35.546  | 35.5135 | 35.5007 | 35.4777 | 35.4565 | 35.4674 | 35.4032 | 35.4181 | 35.3892 | 35.3967 | 35.3795 | 35.3714 | 35.3567 | 35.3113 | 35.3283 | 35.3164 | 35.255  | 35.2648 | 35.2162 |
|         18 | 35.5582 | 35.5582 | 35.5565 | 35.5637 | 35.5349 | 35.5085 | 35.4673 | 35.4787 | 35.4443 | 35.4043 | 35.4236 | 35.4038 | 35.3838 | 35.3844 | 35.3562 | 35.3367 | 35.3403 | 35.3283 | 35.2753 | 35.278  | 35.2748 |
|         19 | 35.564  | 35.564  | 35.57   | 35.571  | 35.5512 | 35.5273 | 35.4844 | 35.4916 | 35.4518 | 35.4458 | 35.4236 | 35.4166 | 35.3907 | 35.4026 | 35.3751 | 35.3417 | 35.3558 | 35.317  | 35.3248 | 35.2898 | 35.2769 |
|         20 | 35.5757 | 35.5757 | 35.5796 | 35.582  | 35.5537 | 35.5288 | 35.5027 | 35.5058 | 35.4666 | 35.4424 | 35.4344 | 35.4198 | 35.3959 | 35.4199 | 35.3729 | 35.3708 | 35.3499 | 35.3392 | 35.3134 | 35.2978 | 35.2523 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 36.0276 | 36.0276 | 36.0276 | 36.0276 | 36.0276 | 36.0276 | 36.0276 | 36.0276 | 36.0276 | 35.7163 |
|          0.2 | 35.5862 | 35.5862 | 35.5862 | 35.5862 | 35.5862 | 35.5862 | 35.5862 | 35.5862 | 35.5862 | 35.5863 |
|          0.3 | 35.5024 | 35.5024 | 35.5024 | 35.5024 | 35.5024 | 35.5024 | 35.5024 | 35.5024 | 35.5024 | 35.3857 |
|          0.4 | 35.3742 | 35.3742 | 35.3742 | 35.3742 | 35.3742 | 35.3742 | 35.3742 | 35.3742 | 35.3742 | 35.2907 |
|          0.5 | 35.2659 | 35.2659 | 35.2659 | 35.2659 | 35.2659 | 35.2659 | 35.2659 | 35.2659 | 35.2659 | 35.2826 |
|          0.6 | 35.2743 | 35.2743 | 35.2743 | 35.2743 | 35.2743 | 35.2743 | 35.2743 | 35.2743 | 35.2743 | 35.1987 |
|          0.7 | 35.2473 | 35.2473 | 35.2473 | 35.2473 | 35.2473 | 35.2473 | 35.2473 | 35.2473 | 35.2473 | 35.1975 |
|          0.8 | 35.186  | 35.186  | 35.186  | 35.186  | 35.186  | 35.186  | 35.186  | 35.186  | 35.186  | 35.0135 |
|          0.9 | 35.1853 | 35.1853 | 35.1853 | 35.1853 | 35.1853 | 35.1853 | 35.1853 | 35.1853 | 35.1853 | 35.1137 |
|          1   | 35.1118 | 35.1118 | 35.1118 | 35.1118 | 35.1118 | 35.1118 | 35.1118 | 35.1118 | 35.1118 | 34.9625 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 34.9709 | 35.0344 | 34.9625 | 35.0258 | 35.0177 | 35.0322 |  62.033 |

***

### Tuned Parameters

- max_depth = 5
- min_child_weight = 12
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 50