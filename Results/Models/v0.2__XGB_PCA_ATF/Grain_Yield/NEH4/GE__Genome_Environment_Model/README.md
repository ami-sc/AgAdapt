# NEH4 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 109.1117
- Baseline Model [RMSE] = 109.6402
- Trained Model [MAE] = 22.7072
- Trained Model [RMSE] = 29.6975
- Prediction [MAE] = 98.6892
- Prediction [RMSE] = 99.6520
***

### Dataset Statistics

- LOFO Field [Mean] = 47.3089 [bu/A]
- LOFO Field [Standard Deviation] = 10.7983 [bu/A]
- Model Dataset [Mean] = 156.4206 [bu/A]
- Model Dataset [Standard Deviation] = 43.9706 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 24.6578 | 24.6578 | 24.6578 | 24.6579 | 24.6579 | 24.6578 | 24.6578 | 24.6578 | 24.5969 | 24.5937 | 24.5636 | 24.564  | 24.564  | 24.75   | 24.75   | 24.75   | 24.75   | 24.7675 | 24.7675 | 24.6429 | 24.6512 |
|          2 | 22.8082 | 22.8082 | 22.762  | 22.7692 | 22.7975 | 22.8315 | 22.8334 | 22.6851 | 22.7014 | 23.1161 | 23.1316 | 22.8328 | 22.8049 | 22.9457 | 22.7972 | 22.9632 | 22.8163 | 22.8679 | 22.8308 | 22.8924 | 22.7866 |
|          3 | 22.6787 | 22.6787 | 22.7285 | 22.617  | 22.698  | 22.726  | 22.6015 | 22.6104 | 22.63   | 22.6507 | 22.7494 | 22.7896 | 22.7379 | 22.6164 | 22.7625 | 22.7859 | 22.8829 | 22.8027 | 22.6724 | 22.8344 | 22.711  |
|          4 | 22.7919 | 22.7919 | 22.6137 | 22.9001 | 22.8112 | 22.6229 | 22.7837 | 22.7242 | 22.6655 | 22.8193 | 22.6753 | 22.7074 | 22.6609 | 22.788  | 22.6163 | 22.6676 | 22.8115 | 22.6883 | 22.6961 | 22.7052 | 22.8242 |
|          5 | 22.7264 | 22.7264 | 22.6941 | 22.7228 | 22.6956 | 22.7666 | 22.7884 | 22.7176 | 22.6928 | 22.5567 | 22.7218 | 22.7223 | 22.7439 | 22.6305 | 22.6868 | 22.675  | 22.6305 | 22.6338 | 22.699  | 22.6137 | 22.6734 |
|          6 | 22.824  | 22.824  | 22.7299 | 22.7734 | 22.7203 | 22.667  | 22.6629 | 22.777  | 22.7463 | 22.6872 | 22.6832 | 22.6958 | 22.697  | 22.6067 | 22.8053 | 22.8346 | 22.7068 | 22.7228 | 22.7367 | 22.748  | 22.6376 |
|          7 | 22.9568 | 22.9568 | 22.8569 | 22.9093 | 22.6856 | 22.7519 | 22.811  | 22.7814 | 22.7018 | 22.7394 | 22.8163 | 22.7923 | 22.6485 | 22.7847 | 22.7507 | 22.7401 | 22.7636 | 22.8111 | 22.7518 | 22.7899 | 22.774  |
|          8 | 23.0385 | 23.0385 | 23.0804 | 22.9189 | 22.9213 | 22.8264 | 22.8731 | 22.8348 | 22.8689 | 22.7584 | 22.8973 | 22.6584 | 22.7859 | 22.6955 | 22.7723 | 22.8543 | 22.7148 | 22.8663 | 22.7516 | 22.7804 | 22.7666 |
|          9 | 23.1735 | 23.1735 | 23.091  | 23.096  | 23.0118 | 22.9885 | 22.8612 | 22.989  | 22.9466 | 22.9499 | 22.851  | 22.7186 | 22.9258 | 22.8074 | 22.9026 | 22.7942 | 22.9147 | 22.767  | 22.8335 | 22.8982 | 22.9263 |
|         10 | 23.3121 | 23.3121 | 23.3179 | 23.1379 | 23.2255 | 23.1274 | 23.0151 | 22.9761 | 22.9387 | 23.0527 | 22.8965 | 22.9435 | 22.9396 | 22.8675 | 23.0255 | 22.9949 | 22.958  | 22.9033 | 22.945  | 22.7963 | 22.8961 |
|         11 | 23.5355 | 23.5355 | 23.384  | 23.2784 | 23.2903 | 23.1976 | 23.0534 | 23.0577 | 23.1494 | 23.1362 | 22.9746 | 22.9562 | 22.9477 | 22.9587 | 23.0395 | 23.0709 | 22.931  | 22.8748 | 22.9307 | 22.9722 | 22.8588 |
|         12 | 23.6581 | 23.6581 | 23.5659 | 23.4385 | 23.3245 | 23.3157 | 23.2816 | 23.268  | 23.1948 | 23.1447 | 23.0901 | 23.0114 | 23.0777 | 23.062  | 22.9845 | 22.9024 | 22.9627 | 22.9224 | 23.013  | 22.8478 | 22.9459 |
|         13 | 23.7621 | 23.7621 | 23.584  | 23.5775 | 23.4271 | 23.394  | 23.2047 | 23.178  | 23.1649 | 23.1768 | 23.0245 | 23.089  | 23.0824 | 23.0388 | 22.9998 | 22.9418 | 22.9747 | 22.9359 | 22.9998 | 22.9612 | 22.9972 |
|         14 | 23.9763 | 23.9763 | 23.856  | 23.5845 | 23.4317 | 23.3812 | 23.3729 | 23.164  | 23.1341 | 23.2336 | 23.1574 | 22.9358 | 23.1689 | 23.0122 | 22.9384 | 22.9677 | 23.0798 | 23.0189 | 23.093  | 22.9698 | 23.0747 |
|         15 | 24.0655 | 24.0655 | 23.9123 | 23.6917 | 23.6515 | 23.5036 | 23.4011 | 23.4111 | 23.3709 | 23.3278 | 23.1626 | 23.1064 | 23.1377 | 23.011  | 23.0362 | 23.0515 | 22.91   | 23.0339 | 23.0612 | 22.9    | 23.034  |
|         16 | 24.2965 | 24.2965 | 24.0553 | 23.7622 | 23.7102 | 23.5577 | 23.4527 | 23.411  | 23.3993 | 23.262  | 23.1065 | 23.0211 | 23.1583 | 23.0165 | 23.1828 | 23.0556 | 23.0261 | 22.9692 | 23.0806 | 22.9334 | 23.0317 |
|         17 | 24.4294 | 24.4294 | 24.0591 | 23.8668 | 23.7655 | 23.627  | 23.4531 | 23.5224 | 23.3227 | 23.3473 | 23.2616 | 23.0377 | 23.2148 | 23.1223 | 23.1123 | 22.9911 | 23.0318 | 23.0351 | 23.0641 | 23.0456 | 23.0289 |
|         18 | 24.4226 | 24.4226 | 24.1885 | 23.8071 | 23.8592 | 23.6659 | 23.4885 | 23.5354 | 23.3835 | 23.3696 | 23.2418 | 23.1437 | 23.2146 | 23.2117 | 23.1279 | 22.9955 | 23.0065 | 23.1171 | 23.0499 | 22.9369 | 22.981  |
|         19 | 24.4673 | 24.4673 | 24.1728 | 23.9839 | 23.9325 | 23.6135 | 23.4546 | 23.6239 | 23.3579 | 23.2556 | 23.1957 | 23.0926 | 23.1183 | 23.206  | 23.0629 | 23.0159 | 23.0922 | 23.0538 | 23.0402 | 22.9424 | 23.1032 |
|         20 | 24.6175 | 24.6175 | 24.2973 | 23.9883 | 23.8883 | 23.752  | 23.5455 | 23.6107 | 23.36   | 23.4548 | 23.2399 | 23.2415 | 23.1571 | 23.1217 | 23.1584 | 23.09   | 23.1065 | 23.0581 | 23.0451 | 22.9966 | 23.0775 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.266  | 25.1648 | 25.1526 | 25.1746 | 25.188  | 25.3686 | 25.2183 | 25.1941 | 25.1871 | 25.0319 |
|          0.2 | 24.6064 | 24.7119 | 24.6207 | 24.6244 | 24.6188 | 24.7076 | 24.5179 | 24.6004 | 24.4659 | 24.5735 |
|          0.3 | 24.221  | 24.2075 | 24.0327 | 23.8852 | 24.0195 | 24.119  | 23.9844 | 24.1672 | 24.2063 | 24.2544 |
|          0.4 | 23.8537 | 23.8108 | 23.8989 | 23.5548 | 23.7595 | 23.8245 | 23.6105 | 23.7287 | 23.581  | 23.8093 |
|          0.5 | 23.6311 | 23.4514 | 23.4317 | 23.4197 | 23.4757 | 23.3871 | 23.3703 | 23.3734 | 23.3239 | 23.4431 |
|          0.6 | 23.3404 | 23.3501 | 23.3102 | 23.2902 | 23.3095 | 23.2392 | 23.108  | 23.1652 | 23.3086 | 23.1223 |
|          0.7 | 23.1601 | 23.1271 | 23.1467 | 23.1244 | 23.0818 | 23.2538 | 22.9382 | 23.1394 | 23.0577 | 23.0073 |
|          0.8 | 23.0826 | 23.1231 | 22.9371 | 23.1034 | 23.0136 | 22.8767 | 22.8301 | 22.9089 | 22.9513 | 22.9956 |
|          0.9 | 22.9582 | 22.7236 | 22.8601 | 22.8316 | 22.8942 | 22.8251 | 22.8006 | 22.813  | 22.9372 | 22.837  |
|          1   | 23.3199 | 22.7545 | 22.8196 | 22.7182 | 22.6852 | 22.6757 | 22.6857 | 22.7362 | 22.6995 | 22.5567 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 22.9404 | 22.8473 | 22.5567 | 22.6393 | 22.5787 | 23.0859 | 59.4224 |

***

### Tuned Parameters

- max_depth = 5
- min_child_weight = 9
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 99