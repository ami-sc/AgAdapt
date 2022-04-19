# WIH2 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 37.2996
- Baseline Model [RMSE] = 43.9803
- Trained Model [MAE] = 22.9875
- Trained Model [RMSE] = 29.8435
- Prediction [MAE] = 20.0903
- Prediction [RMSE] = 26.1329
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
|          1 | 24.5568 | 24.5568 | 24.5568 | 24.5568 | 24.5568 | 24.5568 | 24.5568 | 24.5568 | 24.5483 | 24.487  | 24.7029 | 24.7091 | 24.7091 | 24.7091 | 24.7091 | 24.7091 | 24.7252 | 24.6727 | 24.6818 | 24.5914 | 24.7019 |
|          2 | 23.0288 | 23.0288 | 23.2572 | 23.0726 | 22.9831 | 23.2137 | 23.16   | 23.1329 | 22.9712 | 23.0492 | 23.1113 | 23.0901 | 23.0952 | 23.0496 | 23.1521 | 23.2243 | 23.273  | 23.1081 | 23.3128 | 23.1732 | 23.1555 |
|          3 | 22.9612 | 22.9612 | 22.9739 | 22.9537 | 22.7621 | 22.9375 | 22.8641 | 22.954  | 23.0282 | 22.819  | 22.9774 | 22.9794 | 23.0003 | 22.923  | 22.9087 | 22.9154 | 22.899  | 22.9389 | 22.9041 | 22.8953 | 23.003  |
|          4 | 22.8974 | 22.8974 | 22.8554 | 22.8343 | 22.8997 | 22.8316 | 22.8909 | 22.8577 | 22.8357 | 22.9253 | 22.9304 | 22.9002 | 22.9228 | 22.8678 | 22.8744 | 22.7947 | 22.9145 | 22.8554 | 22.8577 | 22.9618 | 22.9012 |
|          5 | 22.8198 | 22.8198 | 22.8988 | 22.8171 | 22.857  | 22.8616 | 22.8736 | 22.9148 | 22.8942 | 22.9824 | 22.9476 | 22.8411 | 22.9515 | 22.8984 | 22.8208 | 22.9016 | 22.9193 | 22.8726 | 22.9031 | 22.9752 | 23.0205 |
|          6 | 22.7932 | 22.7932 | 22.9636 | 22.7322 | 22.8791 | 22.8808 | 22.872  | 22.8709 | 22.8823 | 22.9087 | 23.0201 | 22.8912 | 22.9057 | 22.8923 | 22.8897 | 22.902  | 22.9905 | 22.8746 | 22.9289 | 22.8839 | 22.8863 |
|          7 | 22.9959 | 22.9959 | 22.9491 | 22.826  | 22.9299 | 22.8827 | 22.857  | 22.8623 | 22.9597 | 22.838  | 22.9381 | 22.9828 | 22.9652 | 22.957  | 22.9846 | 22.9859 | 22.9779 | 22.8734 | 22.8242 | 22.8494 | 22.9459 |
|          8 | 23.1292 | 23.1292 | 22.9649 | 23.1011 | 22.9425 | 22.8488 | 22.9274 | 23.072  | 23.0355 | 23.0652 | 23.0185 | 23.0563 | 22.9351 | 22.9971 | 22.9669 | 22.9464 | 22.9203 | 22.9597 | 23.0392 | 22.9478 | 22.9561 |
|          9 | 23.1688 | 23.1688 | 23.0965 | 22.9817 | 23.1797 | 23.0196 | 23.0061 | 22.9831 | 23.111  | 23.2    | 23.1277 | 23.0336 | 22.9517 | 22.9682 | 23.0051 | 22.9877 | 23.0571 | 23.0032 | 23.0282 | 22.8467 | 22.9072 |
|         10 | 23.3543 | 23.3543 | 23.2461 | 23.145  | 23.1731 | 23.0955 | 23.1385 | 23.165  | 23.1499 | 23.1802 | 23.1817 | 23.0813 | 23.0811 | 23.0179 | 23.0894 | 23.0934 | 23.0913 | 22.9608 | 22.9842 | 22.8625 | 23.0323 |
|         11 | 23.4925 | 23.4925 | 23.4003 | 23.3172 | 23.2283 | 23.2371 | 23.267  | 23.184  | 23.244  | 23.1713 | 23.1715 | 23.1111 | 23.0971 | 23.0674 | 23.0324 | 23.0799 | 23.0556 | 23.0189 | 23.1438 | 23.0804 | 23.0051 |
|         12 | 23.6699 | 23.6699 | 23.5    | 23.4333 | 23.4209 | 23.2351 | 23.2934 | 23.2877 | 23.3346 | 23.1875 | 23.0919 | 23.2268 | 23.2679 | 23.198  | 23.124  | 23.1793 | 23.0338 | 23.1959 | 23.0427 | 23.0756 | 23.0071 |
|         13 | 23.8711 | 23.8711 | 23.6442 | 23.5366 | 23.4073 | 23.3568 | 23.4573 | 23.3107 | 23.3514 | 23.2694 | 23.2709 | 23.1591 | 23.2802 | 23.1163 | 23.1597 | 23.1116 | 23.1743 | 23.112  | 23.2274 | 23.0312 | 23.081  |
|         14 | 24.0007 | 24.0007 | 23.7807 | 23.6394 | 23.457  | 23.4282 | 23.3819 | 23.4057 | 23.4363 | 23.3154 | 23.3674 | 23.2899 | 23.2983 | 23.2256 | 23.1803 | 23.1615 | 23.0023 | 23.1112 | 23.1344 | 23.0539 | 22.9575 |
|         15 | 24.1279 | 24.1279 | 23.934  | 23.7036 | 23.5971 | 23.4992 | 23.4711 | 23.3876 | 23.3497 | 23.3893 | 23.2804 | 23.2225 | 23.306  | 23.162  | 23.1675 | 23.1998 | 23.0885 | 23.1901 | 23.1477 | 23.0185 | 23.0461 |
|         16 | 24.2295 | 24.2295 | 24.0514 | 23.9017 | 23.7441 | 23.5654 | 23.5429 | 23.3886 | 23.4221 | 23.3219 | 23.2238 | 23.2883 | 23.2701 | 23.2225 | 23.352  | 23.1612 | 23.0531 | 23.1713 | 23.2266 | 23.0482 | 23.0436 |
|         17 | 24.298  | 24.298  | 24.1847 | 23.7847 | 23.6448 | 23.5964 | 23.5647 | 23.451  | 23.4552 | 23.4294 | 23.3074 | 23.2042 | 23.2899 | 23.3338 | 23.1484 | 23.2885 | 23.1495 | 23.1266 | 23.2411 | 22.9632 | 23.2184 |
|         18 | 24.3809 | 24.3809 | 24.2269 | 23.9436 | 23.7801 | 23.6863 | 23.5789 | 23.5074 | 23.5121 | 23.3935 | 23.3409 | 23.296  | 23.2593 | 23.3049 | 23.1598 | 23.167  | 23.1643 | 23.2245 | 23.1513 | 23.0931 | 23.1339 |
|         19 | 24.363  | 24.363  | 24.2437 | 23.8835 | 23.8121 | 23.7249 | 23.6339 | 23.494  | 23.6386 | 23.494  | 23.3609 | 23.3191 | 23.318  | 23.2934 | 23.2424 | 23.2615 | 23.1166 | 23.1902 | 23.1396 | 23.1046 | 23.1209 |
|         20 | 24.5467 | 24.5467 | 24.2914 | 24.0208 | 23.844  | 23.7515 | 23.6721 | 23.4219 | 23.5801 | 23.4779 | 23.3675 | 23.3145 | 23.2676 | 23.2917 | 23.2589 | 23.2151 | 23.1485 | 23.1775 | 23.1692 | 23.2182 | 23.149  |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.1779 | 25.2417 | 25.1189 | 25.3963 | 25.1839 | 25.2172 | 25.3339 | 25.3177 | 25.1985 | 25.2719 |
|          0.2 | 24.5175 | 24.5263 | 24.5943 | 24.4543 | 24.4858 | 24.5545 | 24.6002 | 24.6346 | 24.5501 | 24.4359 |
|          0.3 | 24.0864 | 24.3738 | 24.2055 | 24.0365 | 24.0803 | 23.9552 | 24.1714 | 23.8544 | 24.1698 | 24.1409 |
|          0.4 | 23.8757 | 23.9201 | 24.0551 | 23.823  | 23.7949 | 23.6747 | 23.7141 | 23.6997 | 23.7513 | 23.7965 |
|          0.5 | 23.5888 | 23.7048 | 23.6155 | 23.7346 | 23.5542 | 23.5978 | 23.7558 | 23.7148 | 23.5956 | 23.5692 |
|          0.6 | 23.5251 | 23.4145 | 23.2871 | 23.2429 | 23.2425 | 23.4883 | 23.5299 | 23.1432 | 23.334  | 23.3315 |
|          0.7 | 23.405  | 23.2506 | 23.1659 | 23.2103 | 23.2746 | 23.1711 | 23.3172 | 23.2264 | 23.1919 | 23.0686 |
|          0.8 | 23.2222 | 23.0964 | 23.0999 | 23.1898 | 23.1001 | 23.0674 | 23.0834 | 23.04   | 23.0589 | 23.0984 |
|          0.9 | 22.9205 | 23.0448 | 23.0787 | 22.9536 | 23.0311 | 23.0089 | 23.0459 | 23.0594 | 22.9097 | 22.9784 |
|          1   | 23.116  | 22.903  | 22.7523 | 22.9006 | 22.808  | 22.9334 | 22.858  | 22.8389 | 22.989  | 22.7322 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 23.0109 | 23.0316 | 22.7322 | 22.7695 | 22.7313 | 22.8169 | 58.2612 |

***

### Tuned Parameters

- max_depth = 6
- min_child_weight = 3
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.1
- num_boost_round = 196