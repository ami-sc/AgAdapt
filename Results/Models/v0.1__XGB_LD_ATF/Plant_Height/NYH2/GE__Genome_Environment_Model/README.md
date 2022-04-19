# NYH2 - Plant_Height Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 42.8093
- Baseline Model [RMSE] = 46.5130
- Trained Model [MAE] = 10.4479
- Trained Model [RMSE] = 13.9176
- Prediction [MAE] = 38.0744
- Prediction [RMSE] = 40.7851
***

### Dataset Statistics

- LOFO Field [Mean] = 257.5984 [cm]
- LOFO Field [Standard Deviation] = 20.1031 [cm]
- Model Dataset [Mean] = 215.6411 [cm]
- Model Dataset [Standard Deviation] = 39.5113 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 13.0866 | 13.0866 | 13.0866 | 13.0874 | 13.0899 | 13.0899 | 13.0899 | 13.0899 | 13.0899 | 13.0891 | 13.0891 | 13.0875 | 13.0875 | 13.0875 | 13.0875 | 13.0875 | 13.0875 | 13.0875 | 13.0875 | 13.0875 | 13.088  |
|          2 | 11.0202 | 11.0202 | 10.987  | 11.0223 | 11.0204 | 11.0106 | 10.9618 | 11.0429 | 11.1189 | 11.0334 | 11.0253 | 11.0056 | 11.0087 | 11.0124 | 11.1075 | 11.1781 | 11.0709 | 11.045  | 11.0188 | 11.1272 | 11.0974 |
|          3 | 11.0283 | 11.0283 | 11.09   | 11.0142 | 11.0145 | 11.004  | 11.0222 | 11.1099 | 10.9854 | 11.0303 | 11.0952 | 11.0014 | 11.0185 | 11.035  | 11.0039 | 11.0303 | 10.9964 | 11.0151 | 11.0367 | 10.9734 | 11.0439 |
|          4 | 11.1803 | 11.1803 | 11.1741 | 11.0854 | 11.138  | 11.1004 | 11.085  | 11.0698 | 11.0284 | 11.0694 | 11.0551 | 11.0766 | 11.0604 | 11.0615 | 11.0786 | 10.9566 | 11.0494 | 11.0549 | 11.0839 | 11.1418 | 11.082  |
|          5 | 11.1782 | 11.1782 | 11.1837 | 11.1666 | 11.1522 | 11.1358 | 11.1132 | 11.1219 | 11.212  | 11.1282 | 11.1505 | 11.0892 | 11.1174 | 11.0679 | 11.1369 | 11.1514 | 11.0946 | 11.1164 | 11.0803 | 11.1636 | 11.0924 |
|          6 | 11.2801 | 11.2801 | 11.2597 | 11.2038 | 11.2441 | 11.2719 | 11.2045 | 11.2068 | 11.1709 | 11.1178 | 11.1334 | 11.2007 | 11.1416 | 11.1425 | 11.1461 | 11.1166 | 11.1965 | 11.2044 | 11.1252 | 11.1045 | 11.1175 |
|          7 | 11.3203 | 11.3203 | 11.3093 | 11.3006 | 11.2348 | 11.2492 | 11.2537 | 11.294  | 11.2891 | 11.1541 | 11.195  | 11.1674 | 11.2027 | 11.2491 | 11.1512 | 11.2132 | 11.2505 | 11.1677 | 11.1707 | 11.1836 | 11.1181 |
|          8 | 11.4622 | 11.4622 | 11.3672 | 11.3746 | 11.3505 | 11.3688 | 11.3496 | 11.2762 | 11.2115 | 11.2514 | 11.2567 | 11.1985 | 11.2385 | 11.2101 | 11.2067 | 11.1708 | 11.2075 | 11.227  | 11.2318 | 11.214  | 11.2128 |
|          9 | 11.5134 | 11.5134 | 11.4512 | 11.4168 | 11.3285 | 11.3284 | 11.37   | 11.3854 | 11.3543 | 11.2575 | 11.3195 | 11.2762 | 11.2318 | 11.2504 | 11.2941 | 11.2358 | 11.2186 | 11.2845 | 11.2427 | 11.2263 | 11.273  |
|         10 | 11.5758 | 11.5758 | 11.5254 | 11.4012 | 11.438  | 11.4025 | 11.3841 | 11.3863 | 11.2997 | 11.3313 | 11.323  | 11.2545 | 11.223  | 11.3358 | 11.2652 | 11.2669 | 11.2338 | 11.2251 | 11.2013 | 11.2422 | 11.2537 |
|         11 | 11.6495 | 11.6495 | 11.6107 | 11.5125 | 11.4372 | 11.4263 | 11.4046 | 11.3481 | 11.4069 | 11.315  | 11.3899 | 11.356  | 11.279  | 11.3091 | 11.3045 | 11.2478 | 11.2677 | 11.2741 | 11.2377 | 11.2211 | 11.2398 |
|         12 | 11.6928 | 11.6928 | 11.5991 | 11.5893 | 11.4637 | 11.4894 | 11.4879 | 11.4407 | 11.3495 | 11.3305 | 11.3565 | 11.3312 | 11.2343 | 11.2483 | 11.3186 | 11.2573 | 11.2616 | 11.2393 | 11.2503 | 11.2007 | 11.2755 |
|         13 | 11.8145 | 11.8145 | 11.7127 | 11.6306 | 11.5995 | 11.5355 | 11.4564 | 11.4393 | 11.4232 | 11.3921 | 11.3542 | 11.3367 | 11.3139 | 11.2779 | 11.3164 | 11.29   | 11.3018 | 11.2781 | 11.2439 | 11.2954 | 11.2478 |
|         14 | 11.8791 | 11.8791 | 11.7428 | 11.7297 | 11.5902 | 11.5319 | 11.5002 | 11.4391 | 11.4683 | 11.4206 | 11.3843 | 11.4331 | 11.3691 | 11.2823 | 11.3346 | 11.2981 | 11.3201 | 11.288  | 11.2084 | 11.262  | 11.3005 |
|         15 | 12.0258 | 12.0258 | 11.764  | 11.7563 | 11.6388 | 11.6137 | 11.5298 | 11.4656 | 11.413  | 11.4248 | 11.4274 | 11.3984 | 11.4234 | 11.3148 | 11.3142 | 11.3472 | 11.2698 | 11.2055 | 11.2844 | 11.2981 | 11.2742 |
|         16 | 12.011  | 12.011  | 11.8473 | 11.7928 | 11.7651 | 11.6111 | 11.5952 | 11.5209 | 11.4838 | 11.3948 | 11.4127 | 11.3632 | 11.3613 | 11.3465 | 11.3421 | 11.3049 | 11.3163 | 11.2576 | 11.2506 | 11.3109 | 11.2538 |
|         17 | 12.0731 | 12.0731 | 11.8866 | 11.8357 | 11.6765 | 11.6271 | 11.6342 | 11.5768 | 11.5645 | 11.4845 | 11.4506 | 11.4135 | 11.3756 | 11.3589 | 11.3527 | 11.3489 | 11.2644 | 11.3285 | 11.3397 | 11.3266 | 11.2797 |
|         18 | 12.1328 | 12.1328 | 11.9527 | 11.8324 | 11.7371 | 11.6633 | 11.613  | 11.5765 | 11.5007 | 11.516  | 11.4086 | 11.4028 | 11.4034 | 11.3571 | 11.327  | 11.3343 | 11.2537 | 11.26   | 11.2956 | 11.3061 | 11.2762 |
|         19 | 12.2234 | 12.2234 | 11.9933 | 11.9255 | 11.7274 | 11.664  | 11.6661 | 11.6078 | 11.5781 | 11.5769 | 11.4154 | 11.4335 | 11.4025 | 11.3937 | 11.361  | 11.3689 | 11.3247 | 11.328  | 11.2928 | 11.2826 | 11.3238 |
|         20 | 12.2237 | 12.2237 | 11.998  | 11.8702 | 11.7656 | 11.7156 | 11.6663 | 11.5226 | 11.52   | 11.5402 | 11.4902 | 11.4182 | 11.3894 | 11.4006 | 11.3616 | 11.3979 | 11.2892 | 11.3198 | 11.2919 | 11.2769 | 11.3035 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 13.7191 | 13.5371 | 13.6806 | 13.4783 | 12.9359 | 13.2988 | 13.3161 | 13.2349 | 12.9971 | 12.9896 |
|          0.2 | 12.9125 | 13.1745 | 12.5762 | 12.4777 | 12.3997 | 12.1098 | 12.3139 | 12.4184 | 12.1579 | 12.3392 |
|          0.3 | 12.6496 | 12.33   | 12.1757 | 12.0754 | 12.2603 | 11.9143 | 11.9877 | 12.0988 | 11.8521 | 11.9929 |
|          0.4 | 12.2728 | 12.138  | 11.9542 | 11.8954 | 11.7799 | 11.6236 | 11.7366 | 11.6089 | 11.6769 | 11.6242 |
|          0.5 | 12.177  | 11.8105 | 11.8234 | 11.7162 | 11.7331 | 11.6139 | 11.6982 | 11.7165 | 11.5908 | 11.4871 |
|          0.6 | 12.0119 | 11.6478 | 11.5627 | 11.6675 | 11.552  | 11.4574 | 11.4266 | 11.3969 | 11.4459 | 11.4236 |
|          0.7 | 11.6172 | 11.5094 | 11.3386 | 11.49   | 11.4621 | 11.4009 | 11.4204 | 11.2971 | 11.293  | 11.3186 |
|          0.8 | 11.5837 | 11.272  | 11.3565 | 11.2606 | 11.317  | 11.2912 | 11.1671 | 11.1784 | 11.4415 | 11.2137 |
|          0.9 | 11.5554 | 11.2991 | 11.181  | 11.2141 | 11.1603 | 11.0682 | 11.1844 | 11.1068 | 11.1509 | 11.1912 |
|          1   | 11.2302 | 11.0876 | 11.1333 | 11.1356 | 11.1066 | 11.0479 | 11.0421 | 11.0436 | 10.9822 | 10.9566 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 11.2482 | 11.1279 | 10.9566 | 11.0341 | 10.9686 | 12.1252 | 79.3611 |

***

### Tuned Parameters

- max_depth = 4
- min_child_weight = 15
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 108