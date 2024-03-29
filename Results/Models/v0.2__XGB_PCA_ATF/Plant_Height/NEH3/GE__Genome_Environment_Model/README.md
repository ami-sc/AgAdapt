# NEH3 - Plant_Height Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 37.8364
- Baseline Model [RMSE] = 44.1814
- Trained Model [MAE] = 10.7757
- Trained Model [RMSE] = 14.5099
- Prediction [MAE] = 31.2973
- Prediction [RMSE] = 34.2783
***

### Dataset Statistics

- LOFO Field [Mean] = 180.8831 [cm]
- LOFO Field [Standard Deviation] = 23.7029 [cm]
- Model Dataset [Mean] = 218.2170 [cm]
- Model Dataset [Standard Deviation] = 39.7258 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 12.8264 | 12.8264 | 12.8264 | 12.8264 | 12.8264 | 12.8264 | 12.8259 | 12.8259 | 12.8259 | 12.8248 | 12.8266 | 12.8261 | 12.8202 | 12.8211 | 12.8215 | 12.8189 | 12.8189 | 12.8174 | 12.8163 | 12.815  | 12.815  |
|          2 | 10.9356 | 10.9356 | 10.9257 | 10.9356 | 10.9561 | 10.8721 | 10.9263 | 10.9345 | 10.9475 | 10.893  | 10.9541 | 10.9341 | 10.8791 | 11.0382 | 11.2206 | 11.0374 | 10.9953 | 11.1049 | 11.0331 | 11.021  | 10.9992 |
|          3 | 10.9304 | 10.9304 | 10.9524 | 10.958  | 10.9395 | 10.9182 | 10.9749 | 10.8903 | 10.9137 | 10.9324 | 10.9045 | 10.9665 | 10.9721 | 11.0002 | 10.9458 | 11.0104 | 11.1616 | 10.9304 | 10.9597 | 10.9739 | 11.0284 |
|          4 | 11.0469 | 11.0469 | 11.0236 | 10.9901 | 11.0736 | 10.9457 | 10.9944 | 10.9834 | 10.9852 | 10.9938 | 10.983  | 10.9998 | 10.9798 | 11.0411 | 10.9608 | 11.1081 | 11.0875 | 10.9805 | 11.0153 | 11.0683 | 11.0185 |
|          5 | 11.0631 | 11.0631 | 11.0729 | 11.1251 | 11.1055 | 11.0591 | 11.0938 | 11.0513 | 11.0318 | 11.1225 | 11.0212 | 11.0237 | 11.0743 | 11.0716 | 10.9943 | 11.0478 | 11.0518 | 11.0813 | 11.1101 | 11.0308 | 11.0321 |
|          6 | 11.2001 | 11.2001 | 11.2011 | 11.1835 | 11.1651 | 11.1414 | 11.1505 | 11.1371 | 11.113  | 11.1179 | 11.1835 | 11.1336 | 11.1362 | 11.1317 | 11.1428 | 11.1162 | 11.1601 | 11.1011 | 11.0928 | 11.129  | 11.1374 |
|          7 | 11.2221 | 11.2221 | 11.1811 | 11.1612 | 11.2532 | 11.2519 | 11.2047 | 11.1856 | 11.1986 | 11.1936 | 11.1437 | 11.2032 | 11.1524 | 11.1716 | 11.1705 | 11.1427 | 11.2181 | 11.1805 | 11.1747 | 11.1661 | 11.1549 |
|          8 | 11.3153 | 11.3153 | 11.3117 | 11.2804 | 11.2081 | 11.2259 | 11.2764 | 11.222  | 11.1392 | 11.2138 | 11.1585 | 11.2048 | 11.212  | 11.1924 | 11.2078 | 11.1707 | 11.2352 | 11.2036 | 11.1733 | 11.1301 | 11.1584 |
|          9 | 11.3793 | 11.3793 | 11.3302 | 11.3797 | 11.2805 | 11.2752 | 11.2894 | 11.3002 | 11.3002 | 11.177  | 11.2571 | 11.2691 | 11.3285 | 11.2793 | 11.2567 | 11.2513 | 11.2505 | 11.206  | 11.1842 | 11.2211 | 11.1502 |
|         10 | 11.5354 | 11.5354 | 11.3993 | 11.4186 | 11.411  | 11.3742 | 11.3352 | 11.3014 | 11.3688 | 11.3043 | 11.3185 | 11.3173 | 11.3712 | 11.2472 | 11.2947 | 11.2432 | 11.2751 | 11.2663 | 11.1882 | 11.2779 | 11.2216 |
|         11 | 11.5717 | 11.5717 | 11.4888 | 11.4356 | 11.3753 | 11.3435 | 11.3766 | 11.3561 | 11.3983 | 11.309  | 11.3335 | 11.3218 | 11.2807 | 11.262  | 11.2824 | 11.2693 | 11.2575 | 11.2587 | 11.2667 | 11.2616 | 11.2103 |
|         12 | 11.6208 | 11.6208 | 11.5084 | 11.5173 | 11.4368 | 11.4178 | 11.406  | 11.3664 | 11.4122 | 11.369  | 11.3187 | 11.3987 | 11.3393 | 11.3147 | 11.2925 | 11.3024 | 11.3006 | 11.2285 | 11.3084 | 11.25   | 11.2906 |
|         13 | 11.7423 | 11.7423 | 11.594  | 11.5438 | 11.5045 | 11.4241 | 11.3848 | 11.3907 | 11.3736 | 11.3416 | 11.397  | 11.4056 | 11.3394 | 11.3783 | 11.3459 | 11.3629 | 11.3825 | 11.284  | 11.308  | 11.31   | 11.2639 |
|         14 | 11.8042 | 11.8042 | 11.6683 | 11.692  | 11.577  | 11.4915 | 11.4778 | 11.461  | 11.4177 | 11.3901 | 11.4459 | 11.3604 | 11.3318 | 11.364  | 11.3466 | 11.3055 | 11.3222 | 11.2913 | 11.3347 | 11.3402 | 11.3496 |
|         15 | 11.8595 | 11.8595 | 11.735  | 11.6925 | 11.633  | 11.5464 | 11.5605 | 11.4951 | 11.4578 | 11.4692 | 11.4138 | 11.4108 | 11.3718 | 11.3808 | 11.3453 | 11.4296 | 11.3192 | 11.3362 | 11.4079 | 11.3094 | 11.3568 |
|         16 | 11.9234 | 11.9234 | 11.7825 | 11.7412 | 11.6535 | 11.5926 | 11.5441 | 11.4657 | 11.4564 | 11.3889 | 11.4299 | 11.4461 | 11.3947 | 11.442  | 11.3235 | 11.3453 | 11.334  | 11.3326 | 11.379  | 11.3724 | 11.3307 |
|         17 | 11.9457 | 11.9457 | 11.8749 | 11.7511 | 11.6823 | 11.5887 | 11.6117 | 11.5223 | 11.4971 | 11.4138 | 11.523  | 11.4292 | 11.4479 | 11.3503 | 11.4218 | 11.4299 | 11.397  | 11.3662 | 11.3948 | 11.3693 | 11.3428 |
|         18 | 12.0392 | 12.0392 | 11.8414 | 11.761  | 11.6974 | 11.5717 | 11.5958 | 11.5065 | 11.5082 | 11.431  | 11.4753 | 11.4862 | 11.4029 | 11.41   | 11.3944 | 11.4169 | 11.3618 | 11.3568 | 11.3907 | 11.34   | 11.3181 |
|         19 | 12.039  | 12.039  | 11.9109 | 11.7766 | 11.7462 | 11.5552 | 11.5611 | 11.5456 | 11.5127 | 11.4342 | 11.5215 | 11.4538 | 11.3946 | 11.4613 | 11.3742 | 11.387  | 11.3484 | 11.3661 | 11.3695 | 11.3455 | 11.3341 |
|         20 | 12.0441 | 12.0441 | 11.8987 | 11.8016 | 11.7736 | 11.6372 | 11.6374 | 11.5725 | 11.4973 | 11.4064 | 11.4729 | 11.4591 | 11.392  | 11.4252 | 11.3643 | 11.4296 | 11.3563 | 11.3401 | 11.367  | 11.3053 | 11.3272 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 13.6827 | 13.9726 | 13.7387 | 13.4902 | 13.3812 | 13.4122 | 13.3866 | 13.5859 | 13.4177 | 13.3023 |
|          0.2 | 13.3807 | 13.0344 | 13.1417 | 12.95   | 12.9017 | 12.6756 | 12.9283 | 12.6449 | 12.7209 | 12.8213 |
|          0.3 | 13.1003 | 12.6548 | 12.4426 | 12.6504 | 12.5382 | 12.2288 | 12.4482 | 12.4793 | 12.1409 | 12.2838 |
|          0.4 | 12.5485 | 12.7161 | 12.2785 | 12.4027 | 12.4902 | 12.0536 | 12.3199 | 12.2256 | 12.0454 | 11.951  |
|          0.5 | 12.8966 | 12.2061 | 12.0124 | 11.9436 | 11.9658 | 11.8882 | 11.8662 | 11.8765 | 11.7402 | 11.7888 |
|          0.6 | 12.6148 | 12.0862 | 12.197  | 11.9516 | 11.7341 | 11.686  | 11.9712 | 11.5858 | 11.7373 | 11.5667 |
|          0.7 | 12.3461 | 12.3277 | 12.0817 | 11.5293 | 11.7975 | 11.6322 | 11.6441 | 11.5346 | 11.3729 | 11.4502 |
|          0.8 | 12.2364 | 11.6261 | 11.7979 | 11.5717 | 11.6833 | 11.2774 | 11.3655 | 11.4627 | 11.3291 | 11.1796 |
|          0.9 | 11.6837 | 11.4344 | 11.5345 | 11.2744 | 11.0908 | 11.2302 | 11.1193 | 11.0214 | 11.2153 | 11.2407 |
|          1   | 11.7233 | 11.2655 | 10.9472 | 10.9219 | 10.975  | 10.9286 | 11.0534 | 10.9991 | 10.9202 | 10.8721 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 10.9185 | 10.8805 | 10.8721 | 10.9749 | 11.3032 | 13.5802 | 80.2551 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 5
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 579
