# MNH1 - Plant_Height Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 20.3548
- Baseline Model [RMSE] = 26.0467
- Trained Model [MAE] = 10.5192
- Trained Model [RMSE] = 14.0767
- Prediction [MAE] = 18.0079
- Prediction [RMSE] = 27.0130
***

### Dataset Statistics

- LOFO Field [Mean] = 220.8246 [cm]
- LOFO Field [Standard Deviation] = 25.8469 [cm]
- Model Dataset [Mean] = 217.3549 [cm]
- Model Dataset [Standard Deviation] = 40.3801 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 12.5825 | 12.5825 | 12.5825 | 12.5825 | 12.5825 | 12.5825 | 12.5825 | 12.5819 | 12.5819 | 12.5819 | 12.5834 | 12.5837 | 12.5837 | 12.5858 | 12.5859 | 12.5859 | 12.5876 | 12.5899 | 12.5899 | 12.5899 | 12.5899 |
|          2 | 10.5598 | 10.5598 | 10.5466 | 10.5533 | 10.5363 | 10.6439 | 10.7161 | 10.642  | 10.6326 | 10.7012 | 10.6475 | 10.614  | 10.6299 | 10.6779 | 10.8713 | 10.8124 | 10.7222 | 10.7652 | 10.6939 | 10.7785 | 10.742  |
|          3 | 10.6637 | 10.6637 | 10.6345 | 10.6182 | 10.6832 | 10.6498 | 10.6582 | 10.6357 | 10.6569 | 10.6702 | 10.6349 | 10.6964 | 10.7049 | 10.7052 | 10.6455 | 10.7033 | 10.6298 | 10.6693 | 10.6967 | 10.8352 | 10.6721 |
|          4 | 10.8066 | 10.8066 | 10.78   | 10.807  | 10.7321 | 10.7591 | 10.7036 | 10.6795 | 10.7142 | 10.7298 | 10.7383 | 10.7473 | 10.8383 | 10.7026 | 10.718  | 10.715  | 10.6882 | 10.6731 | 10.7283 | 10.7955 | 10.7366 |
|          5 | 10.8368 | 10.8368 | 10.7956 | 10.8209 | 10.7862 | 10.7417 | 10.7801 | 10.7546 | 10.7235 | 10.7535 | 10.7514 | 10.808  | 10.7116 | 10.7874 | 10.7317 | 10.7414 | 10.8304 | 10.7242 | 10.7204 | 10.7301 | 10.7624 |
|          6 | 10.9284 | 10.9284 | 10.9525 | 10.8451 | 10.8998 | 10.8715 | 10.837  | 10.856  | 10.8018 | 10.8799 | 10.8228 | 10.7956 | 10.7984 | 10.845  | 10.8436 | 10.8223 | 10.7942 | 10.7509 | 10.7709 | 10.8365 | 10.8515 |
|          7 | 10.9407 | 10.9407 | 10.9599 | 10.93   | 10.902  | 10.8481 | 10.843  | 10.876  | 10.8526 | 10.8882 | 10.9049 | 10.8888 | 10.8559 | 10.8276 | 10.8788 | 10.925  | 10.8339 | 10.8684 | 10.8845 | 10.8054 | 10.7643 |
|          8 | 11.04   | 11.04   | 11.0518 | 11.011  | 10.9941 | 10.9839 | 10.9724 | 10.9216 | 10.9189 | 10.8639 | 10.913  | 10.8511 | 10.7761 | 10.8896 | 10.8658 | 10.8886 | 10.863  | 10.8353 | 10.8438 | 10.8886 | 10.8526 |
|          9 | 11.1218 | 11.1218 | 11.0234 | 11.0215 | 10.9921 | 11.0225 | 10.9657 | 10.9323 | 10.9287 | 10.9289 | 10.9219 | 10.867  | 10.8805 | 10.8689 | 10.8803 | 10.9458 | 10.8842 | 10.8947 | 10.9065 | 10.86   | 10.9093 |
|         10 | 11.1415 | 11.1415 | 11.0902 | 11.0249 | 11.0786 | 11.0364 | 11.0149 | 10.9544 | 10.9607 | 10.9639 | 11.0305 | 10.9051 | 10.9331 | 10.8825 | 11.0005 | 10.9506 | 10.9354 | 10.897  | 10.8792 | 10.8833 | 10.919  |
|         11 | 11.2442 | 11.2442 | 11.1398 | 11.1108 | 11.0607 | 11.1009 | 11.0615 | 11.0638 | 10.9811 | 10.9904 | 10.994  | 10.9903 | 10.9867 | 10.9915 | 10.9268 | 10.9111 | 10.9719 | 10.9399 | 10.9255 | 10.9299 | 10.9526 |
|         12 | 11.3796 | 11.3796 | 11.2798 | 11.1822 | 11.1427 | 11.0717 | 11.0933 | 11.0437 | 11.0856 | 11.0667 | 11.0178 | 10.9535 | 10.9501 | 10.9568 | 10.9413 | 11.0155 | 10.941  | 11.0019 | 10.9275 | 10.973  | 10.9469 |
|         13 | 11.3575 | 11.3575 | 11.2953 | 11.1657 | 11.217  | 11.1316 | 11.2179 | 11.094  | 11.0773 | 11.0498 | 11.0583 | 11.0754 | 11.0025 | 11.0154 | 10.9907 | 11.0575 | 10.9699 | 10.9789 | 11.0043 | 10.954  | 10.928  |
|         14 | 11.4837 | 11.4837 | 11.3393 | 11.2831 | 11.2223 | 11.1311 | 11.1366 | 11.1585 | 11.0793 | 11.0602 | 11.0351 | 11.0567 | 11.0872 | 11.0422 | 10.9752 | 11.0249 | 11.0083 | 11.0186 | 11.0053 | 10.9433 | 10.9712 |
|         15 | 11.5146 | 11.5146 | 11.4711 | 11.3038 | 11.2126 | 11.192  | 11.2    | 11.1083 | 11.0988 | 11.0728 | 11.0625 | 11.113  | 11.0175 | 11.0963 | 11.0622 | 11.0566 | 10.9782 | 10.9773 | 11.0075 | 10.9935 | 10.9324 |
|         16 | 11.5306 | 11.5306 | 11.4639 | 11.3378 | 11.2606 | 11.2722 | 11.1885 | 11.1589 | 11.1419 | 11.0846 | 11.0986 | 11.1238 | 11.0841 | 11.0523 | 11.0643 | 11.0163 | 11.0355 | 11.0156 | 10.981  | 10.9405 | 10.9997 |
|         17 | 11.6009 | 11.6009 | 11.4928 | 11.358  | 11.2861 | 11.2615 | 11.2084 | 11.203  | 11.1748 | 11.1231 | 11.0899 | 11.0867 | 11.0521 | 11.0245 | 11.0935 | 11.0449 | 11.003  | 11.0251 | 10.9461 | 10.9924 | 10.9486 |
|         18 | 11.6068 | 11.6068 | 11.4562 | 11.4455 | 11.3576 | 11.2751 | 11.2253 | 11.2162 | 11.1248 | 11.126  | 11.1106 | 11.0992 | 11.0717 | 11.0844 | 11.0379 | 11.0929 | 11.0646 | 11.0108 | 11.0028 | 10.9786 | 10.9772 |
|         19 | 11.7155 | 11.7155 | 11.5693 | 11.4169 | 11.3652 | 11.3162 | 11.2474 | 11.2389 | 11.1844 | 11.1546 | 11.1222 | 11.1002 | 11.0137 | 11.0858 | 11.0533 | 11.0967 | 11.0058 | 11.0457 | 10.9934 | 10.9734 | 10.9731 |
|         20 | 11.7021 | 11.7021 | 11.5184 | 11.4953 | 11.3698 | 11.3163 | 11.2155 | 11.2265 | 11.1754 | 11.1531 | 11.1229 | 11.1576 | 11.0644 | 11.0931 | 11.0476 | 11.0915 | 11.067  | 11.0702 | 11.0093 | 10.9994 | 11.0429 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 13.3996 | 13.396  | 13.2574 | 12.9565 | 13.3348 | 13.4351 | 12.9273 | 12.7984 | 13.2768 | 13.3049 |
|          0.2 | 13.0345 | 12.7472 | 12.6141 | 12.523  | 12.5027 | 12.9116 | 12.3812 | 12.3256 | 12.3476 | 12.2568 |
|          0.3 | 12.9049 | 12.2856 | 12.2357 | 12.2767 | 12.1666 | 12.0372 | 11.8315 | 12.0164 | 12.167  | 11.7454 |
|          0.4 | 12.5559 | 11.922  | 12.2908 | 12.0637 | 11.824  | 11.8259 | 11.8114 | 11.6579 | 11.6317 | 11.6525 |
|          0.5 | 12.4568 | 11.906  | 12.0388 | 11.6735 | 11.7179 | 11.9253 | 11.4213 | 11.5043 | 11.5366 | 11.5382 |
|          0.6 | 12.0273 | 11.8819 | 11.8337 | 11.8248 | 11.4757 | 11.7617 | 11.3777 | 11.5024 | 11.308  | 11.1576 |
|          0.7 | 12.0009 | 11.5357 | 11.2798 | 11.2151 | 11.3244 | 11.3014 | 11.2802 | 11.3061 | 11.1866 | 11.0439 |
|          0.8 | 11.9702 | 11.6207 | 11.5876 | 11.2323 | 11.0653 | 11.2367 | 10.8721 | 11.0356 | 10.9039 | 10.8901 |
|          0.9 | 11.4283 | 11.2601 | 10.9167 | 10.9827 | 10.8579 | 11.0331 | 10.7007 | 10.8133 | 10.9639 | 10.7532 |
|          1   | 11.2621 | 10.9491 | 10.7778 | 10.8736 | 10.6609 | 10.595  | 10.5691 | 10.5227 | 10.6325 | 10.5363 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 10.5407 | 10.6935 | 10.5227 | 10.7511 | 10.9629 | 13.2506 | 79.9526 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 4
- subsample = 1.0
- colsample_bytree = 0.8
- eta = 0.3
- num_boost_round = 585
