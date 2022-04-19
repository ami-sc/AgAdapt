# IAH4 - Plant_Height Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 20.6225
- Baseline Model [RMSE] = 25.6383
- Trained Model [MAE] = 10.9303
- Trained Model [RMSE] = 14.4998
- Prediction [MAE] = 48.6287
- Prediction [RMSE] = 50.3359
***

### Dataset Statistics

- LOFO Field [Mean] = 230.6847 [cm]
- LOFO Field [Standard Deviation] = 21.2860 [cm]
- Model Dataset [Mean] = 216.3701 [cm]
- Model Dataset [Standard Deviation] = 40.8317 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 13.079  | 13.079  | 13.079  | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0795 | 13.0823 | 13.0823 | 13.0852 |
|          2 | 11.0111 | 11.0111 | 11.0502 | 11.1598 | 11.0761 | 11.1247 | 11.0358 | 11.0478 | 11.1075 | 11.1977 | 11.1984 | 11.1199 | 11.1337 | 11.1038 | 11.1438 | 11.2056 | 11.1699 | 11.2407 | 11.1962 | 11.1144 | 11.2089 |
|          3 | 11.1358 | 11.1358 | 11.1798 | 11.1096 | 11.1419 | 11.1371 | 11.1243 | 11.1291 | 11.1589 | 11.1451 | 11.1087 | 11.1174 | 11.1264 | 11.0875 | 11.1184 | 11.1038 | 11.1427 | 11.2164 | 11.2073 | 11.1625 | 11.2297 |
|          4 | 11.2881 | 11.2881 | 11.2496 | 11.3192 | 11.1888 | 11.1923 | 11.219  | 11.1787 | 11.1691 | 11.1719 | 11.1448 | 11.1309 | 11.1359 | 11.087  | 11.1637 | 11.1146 | 11.2339 | 11.1785 | 11.1634 | 11.2062 | 11.1958 |
|          5 | 11.2897 | 11.2897 | 11.3201 | 11.2877 | 11.2555 | 11.2801 | 11.3005 | 11.2616 | 11.2448 | 11.2039 | 11.3045 | 11.2175 | 11.2228 | 11.2393 | 11.2039 | 11.2027 | 11.1676 | 11.2167 | 11.2608 | 11.2078 | 11.2273 |
|          6 | 11.3655 | 11.3655 | 11.3506 | 11.375  | 11.3027 | 11.3376 | 11.3017 | 11.2865 | 11.3222 | 11.2646 | 11.2146 | 11.3191 | 11.2543 | 11.2929 | 11.2875 | 11.2942 | 11.3167 | 11.279  | 11.272  | 11.3228 | 11.2565 |
|          7 | 11.5373 | 11.5373 | 11.4811 | 11.4681 | 11.348  | 11.3521 | 11.3513 | 11.3587 | 11.3739 | 11.3519 | 11.3315 | 11.3373 | 11.2914 | 11.3389 | 11.2871 | 11.3436 | 11.3332 | 11.2604 | 11.3079 | 11.3612 | 11.3034 |
|          8 | 11.5614 | 11.5614 | 11.5666 | 11.5038 | 11.4602 | 11.4601 | 11.4598 | 11.4581 | 11.347  | 11.3973 | 11.3671 | 11.4095 | 11.373  | 11.3449 | 11.3641 | 11.3633 | 11.4151 | 11.379  | 11.3605 | 11.3266 | 11.3338 |
|          9 | 11.5948 | 11.5948 | 11.6193 | 11.4605 | 11.5338 | 11.4382 | 11.4812 | 11.456  | 11.4467 | 11.4137 | 11.4005 | 11.3813 | 11.4125 | 11.4262 | 11.448  | 11.4132 | 11.4011 | 11.4039 | 11.3916 | 11.3422 | 11.345  |
|         10 | 11.6782 | 11.6782 | 11.6664 | 11.5818 | 11.5414 | 11.5694 | 11.4984 | 11.5    | 11.544  | 11.478  | 11.466  | 11.4796 | 11.4093 | 11.4726 | 11.4415 | 11.4644 | 11.3959 | 11.4091 | 11.3537 | 11.4208 | 11.3844 |
|         11 | 11.7589 | 11.7589 | 11.75   | 11.7111 | 11.6329 | 11.6246 | 11.6148 | 11.5422 | 11.4874 | 11.5493 | 11.5616 | 11.5251 | 11.5515 | 11.5396 | 11.4213 | 11.4284 | 11.3661 | 11.5119 | 11.4364 | 11.4141 | 11.3656 |
|         12 | 11.9345 | 11.9345 | 11.7565 | 11.7524 | 11.6597 | 11.5837 | 11.5862 | 11.5324 | 11.5134 | 11.5332 | 11.5249 | 11.5792 | 11.4923 | 11.4847 | 11.4809 | 11.4681 | 11.4952 | 11.3956 | 11.4103 | 11.4785 | 11.4391 |
|         13 | 11.9575 | 11.9575 | 11.9054 | 11.8239 | 11.7322 | 11.6488 | 11.5885 | 11.5272 | 11.5572 | 11.5885 | 11.608  | 11.5779 | 11.5316 | 11.4925 | 11.5198 | 11.4919 | 11.4846 | 11.5236 | 11.5424 | 11.5022 | 11.4883 |
|         14 | 12.1042 | 12.1042 | 12.0335 | 11.8326 | 11.8038 | 11.6322 | 11.7014 | 11.6666 | 11.6332 | 11.5564 | 11.5908 | 11.6389 | 11.5431 | 11.576  | 11.5202 | 11.4914 | 11.4964 | 11.5626 | 11.5482 | 11.4609 | 11.4225 |
|         15 | 12.1829 | 12.1829 | 12.0719 | 11.9154 | 11.8695 | 11.7654 | 11.7066 | 11.6796 | 11.6686 | 11.5991 | 11.6655 | 11.5986 | 11.641  | 11.5686 | 11.5028 | 11.5236 | 11.4848 | 11.5597 | 11.5315 | 11.5168 | 11.4401 |
|         16 | 12.2741 | 12.2741 | 12.1321 | 11.9601 | 11.8905 | 11.804  | 11.7813 | 11.6711 | 11.6442 | 11.6608 | 11.6728 | 11.6299 | 11.6364 | 11.5945 | 11.5828 | 11.5724 | 11.5397 | 11.5022 | 11.5692 | 11.4912 | 11.4833 |
|         17 | 12.392  | 12.392  | 12.2054 | 12.044  | 11.958  | 11.7856 | 11.7449 | 11.7424 | 11.7321 | 11.6701 | 11.6615 | 11.6896 | 11.6036 | 11.5987 | 11.5579 | 11.5646 | 11.5399 | 11.4883 | 11.497  | 11.5294 | 11.496  |
|         18 | 12.4125 | 12.4125 | 12.2542 | 12.1071 | 11.8839 | 11.8678 | 11.7862 | 11.7458 | 11.7687 | 11.7806 | 11.6619 | 11.7015 | 11.6648 | 11.613  | 11.5723 | 11.5984 | 11.5463 | 11.5986 | 11.5781 | 11.5533 | 11.5128 |
|         19 | 12.4382 | 12.4382 | 12.2571 | 12.1394 | 12.0278 | 11.899  | 11.8479 | 11.7996 | 11.7598 | 11.7002 | 11.6707 | 11.6517 | 11.6392 | 11.6612 | 11.5805 | 11.6303 | 11.5489 | 11.5905 | 11.5753 | 11.4786 | 11.5081 |
|         20 | 12.4759 | 12.4759 | 12.3055 | 12.179  | 12.0142 | 11.8621 | 11.8401 | 11.7964 | 11.6747 | 11.7549 | 11.7381 | 11.7113 | 11.6439 | 11.677  | 11.5862 | 11.5781 | 11.5588 | 11.5564 | 11.5696 | 11.5398 | 11.4847 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 13.9964 | 13.798  | 13.7285 | 13.8679 | 13.7431 | 13.5488 | 13.4987 | 13.4405 | 13.5431 | 13.2684 |
|          0.2 | 13.8627 | 13.2571 | 13.1566 | 13.0431 | 13.2061 | 13.107  | 13.0407 | 12.9194 | 12.9194 | 12.6707 |
|          0.3 | 13.1395 | 12.749  | 12.7767 | 12.9369 | 12.9096 | 12.4013 | 12.4068 | 12.289  | 12.2253 | 12.2454 |
|          0.4 | 13.0843 | 13.0357 | 13.051  | 12.2255 | 12.2952 | 12.4468 | 12.202  | 12.116  | 12.6321 | 12.1557 |
|          0.5 | 12.7002 | 12.3913 | 12.0612 | 12.3685 | 12.2389 | 12.2573 | 12.0795 | 11.8823 | 11.7767 | 11.9343 |
|          0.6 | 13.0568 | 12.2095 | 11.9775 | 11.9736 | 11.9815 | 11.6351 | 11.7425 | 11.9361 | 11.8287 | 11.6273 |
|          0.7 | 12.7233 | 12.0466 | 12.122  | 12.0472 | 11.9815 | 11.7247 | 11.675  | 11.4952 | 11.7814 | 11.5722 |
|          0.8 | 12.7018 | 11.918  | 11.5839 | 11.5582 | 11.5922 | 11.452  | 11.4148 | 11.5578 | 11.5083 | 11.3646 |
|          0.9 | 12.1003 | 11.5682 | 11.3833 | 11.3267 | 11.3846 | 11.324  | 11.4247 | 11.26   | 11.2627 | 11.2581 |
|          1   | 11.85   | 11.3495 | 11.3809 | 11.1699 | 11.0714 | 11.2183 | 11.1229 | 11.0694 | 11.0981 | 11.0111 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 11.0832 | 11.0309 | 11.0111 | 11.0493 | 11.4398 | 13.7115 | 79.5384 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 1
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 406