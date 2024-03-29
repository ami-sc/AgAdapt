# GAH2 - Plant_Height Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 17.6119
- Baseline Model [RMSE] = 21.3293
- Trained Model [MAE] = 10.6624
- Trained Model [RMSE] = 14.3794
- Prediction [MAE] = 14.3806
- Prediction [RMSE] = 17.9877
***

### Dataset Statistics

- LOFO Field [Mean] = 223.2357 [cm]
- LOFO Field [Standard Deviation] = 20.5845 [cm]
- Model Dataset [Mean] = 217.4117 [cm]
- Model Dataset [Standard Deviation] = 40.0808 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0256 | 13.0255 | 13.0255 | 13.0257 | 13.0257 | 13.0257 | 13.0257 | 13.0257 | 13.0257 | 13.0251 | 13.0292 |
|          2 | 10.8937 | 10.8937 | 11.1274 | 10.9245 | 10.9245 | 10.9423 | 10.9763 | 10.8921 | 10.9927 | 10.8852 | 10.9028 | 10.9091 | 10.9672 | 10.969  | 10.9092 | 10.8928 | 11.151  | 11.0381 | 10.9046 | 11.0055 | 11.0093 |
|          3 | 10.9826 | 10.9826 | 11.0443 | 11.0024 | 11.0286 | 11.0218 | 10.9972 | 11.0032 | 10.9862 | 10.923  | 10.9535 | 10.9948 | 10.9259 | 10.9548 | 10.9593 | 10.8985 | 10.9476 | 10.934  | 11.0211 | 10.9256 | 10.9821 |
|          4 | 11.0617 | 11.0617 | 11.1324 | 11.0962 | 11.0629 | 11.0721 | 10.9871 | 11.0481 | 11.0044 | 11.0368 | 11.0366 | 11.0001 | 11.0058 | 10.9879 | 11.0002 | 11.0282 | 10.9859 | 11.0273 | 11.0113 | 10.9651 | 11.0143 |
|          5 | 11.1763 | 11.1763 | 11.1933 | 11.1403 | 11.1653 | 11.0928 | 11.1505 | 11.0927 | 11.1535 | 11.1092 | 11.1038 | 11.0946 | 11.0855 | 11.0474 | 11.0601 | 11.0468 | 11.1331 | 11.1208 | 11.0905 | 11.1281 | 11.0444 |
|          6 | 11.2131 | 11.2131 | 11.2277 | 11.2441 | 11.2488 | 11.184  | 11.2151 | 11.1928 | 11.1902 | 11.1943 | 11.0871 | 11.1622 | 11.1538 | 11.1198 | 11.1366 | 11.1158 | 11.1094 | 11.083  | 11.1174 | 11.1505 | 11.08   |
|          7 | 11.2952 | 11.2952 | 11.3305 | 11.3156 | 11.3354 | 11.2753 | 11.245  | 11.2739 | 11.2749 | 11.2505 | 11.1742 | 11.1469 | 11.1788 | 11.1627 | 11.163  | 11.1757 | 11.1996 | 11.1541 | 11.1726 | 11.157  | 11.1932 |
|          8 | 11.3915 | 11.3915 | 11.4148 | 11.4053 | 11.3398 | 11.3073 | 11.2229 | 11.2986 | 11.2283 | 11.2698 | 11.2678 | 11.2636 | 11.2262 | 11.2125 | 11.1902 | 11.1922 | 11.182  | 11.1541 | 11.1938 | 11.1835 | 11.1939 |
|          9 | 11.4689 | 11.4689 | 11.4046 | 11.4691 | 11.3984 | 11.2969 | 11.2927 | 11.3569 | 11.3178 | 11.3193 | 11.2932 | 11.2849 | 11.3096 | 11.309  | 11.2238 | 11.2216 | 11.2124 | 11.2333 | 11.2284 | 11.2304 | 11.2288 |
|         10 | 11.5602 | 11.5602 | 11.4414 | 11.4612 | 11.4647 | 11.4022 | 11.3367 | 11.3552 | 11.4105 | 11.3967 | 11.3207 | 11.2549 | 11.2968 | 11.2524 | 11.2327 | 11.2453 | 11.2138 | 11.2471 | 11.2463 | 11.2519 | 11.301  |
|         11 | 11.6638 | 11.6638 | 11.5977 | 11.5321 | 11.5298 | 11.3963 | 11.4016 | 11.3844 | 11.3304 | 11.3449 | 11.291  | 11.2898 | 11.346  | 11.2539 | 11.2797 | 11.2163 | 11.3055 | 11.2746 | 11.2457 | 11.2579 | 11.2773 |
|         12 | 11.7234 | 11.7234 | 11.6703 | 11.5733 | 11.5307 | 11.5077 | 11.4522 | 11.4107 | 11.4203 | 11.4054 | 11.3896 | 11.4076 | 11.398  | 11.3666 | 11.3433 | 11.2832 | 11.2811 | 11.2274 | 11.2819 | 11.2903 | 11.2509 |
|         13 | 11.8006 | 11.8006 | 11.7155 | 11.6929 | 11.606  | 11.5249 | 11.5061 | 11.477  | 11.4208 | 11.3823 | 11.3929 | 11.3528 | 11.3415 | 11.358  | 11.3162 | 11.3475 | 11.2684 | 11.2635 | 11.2476 | 11.314  | 11.3068 |
|         14 | 11.8485 | 11.8485 | 11.8019 | 11.6935 | 11.6234 | 11.5157 | 11.5025 | 11.4373 | 11.4964 | 11.4176 | 11.4221 | 11.3791 | 11.3887 | 11.3631 | 11.2907 | 11.3536 | 11.3005 | 11.3093 | 11.3257 | 11.2877 | 11.311  |
|         15 | 11.9642 | 11.9642 | 11.8644 | 11.776  | 11.6645 | 11.5439 | 11.4741 | 11.5262 | 11.4927 | 11.5107 | 11.3804 | 11.3982 | 11.4152 | 11.4164 | 11.3724 | 11.341  | 11.3258 | 11.2993 | 11.3046 | 11.356  | 11.304  |
|         16 | 12.053  | 12.053  | 11.8977 | 11.8036 | 11.7368 | 11.5961 | 11.5522 | 11.4961 | 11.5239 | 11.4786 | 11.4802 | 11.4288 | 11.4468 | 11.4153 | 11.406  | 11.3053 | 11.3141 | 11.2758 | 11.3312 | 11.3566 | 11.36   |
|         17 | 12.0718 | 12.0718 | 11.9419 | 11.8018 | 11.7482 | 11.6518 | 11.5684 | 11.5767 | 11.5671 | 11.4778 | 11.5013 | 11.431  | 11.423  | 11.4126 | 11.4146 | 11.3642 | 11.3148 | 11.3169 | 11.3419 | 11.3228 | 11.3531 |
|         18 | 12.1039 | 12.1039 | 11.9822 | 11.8534 | 11.7412 | 11.6558 | 11.602  | 11.608  | 11.5597 | 11.4406 | 11.5237 | 11.4451 | 11.4079 | 11.4463 | 11.4031 | 11.3706 | 11.3134 | 11.3419 | 11.3307 | 11.3704 | 11.3873 |
|         19 | 12.1241 | 12.1241 | 12.0103 | 11.8742 | 11.7763 | 11.7226 | 11.6184 | 11.5906 | 11.5378 | 11.5202 | 11.5406 | 11.4172 | 11.4158 | 11.4256 | 11.3949 | 11.3574 | 11.3314 | 11.318  | 11.364  | 11.4026 | 11.3956 |
|         20 | 12.1449 | 12.1449 | 12.0524 | 11.9063 | 11.789  | 11.7225 | 11.5831 | 11.627  | 11.5675 | 11.5509 | 11.5515 | 11.4413 | 11.464  | 11.4433 | 11.3718 | 11.3591 | 11.2785 | 11.3002 | 11.3172 | 11.3464 | 11.3825 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 13.7939 | 13.9866 | 13.4143 | 13.7245 | 13.2858 | 13.3992 | 13.4248 | 13.0609 | 13.2702 | 13.4597 |
|          0.2 | 13.9233 | 13.5701 | 13.4096 | 13.0082 | 12.8322 | 12.7877 | 12.742  | 12.8077 | 12.7911 | 12.656  |
|          0.3 | 13.3269 | 13.0325 | 12.8013 | 12.6076 | 12.5337 | 12.5048 | 12.4129 | 12.6563 | 12.5723 | 12.2583 |
|          0.4 | 13.4131 | 12.9587 | 12.7078 | 12.6261 | 12.254  | 11.9107 | 12.0886 | 11.927  | 11.9518 | 11.9149 |
|          0.5 | 13.8871 | 12.4396 | 12.0746 | 12.4824 | 11.679  | 12.0228 | 12.2266 | 11.9354 | 11.854  | 11.9119 |
|          0.6 | 12.8035 | 12.0886 | 11.9431 | 12.0337 | 12.1359 | 11.7527 | 11.8099 | 11.5103 | 11.68   | 11.6772 |
|          0.7 | 12.4677 | 11.9878 | 11.7726 | 11.6675 | 11.4741 | 11.7501 | 11.4685 | 11.5146 | 11.3898 | 11.6897 |
|          0.8 | 12.2811 | 11.7874 | 11.6535 | 11.407  | 11.6122 | 11.4418 | 11.4018 | 11.2518 | 11.3338 | 11.3904 |
|          0.9 | 12.0794 | 11.5067 | 11.3639 | 11.3999 | 11.2043 | 11.1462 | 11.2578 | 11.2021 | 11.0273 | 11.0901 |
|          1   | 11.6653 | 11.2864 | 11.2991 | 11.3421 | 10.9494 | 10.976  | 10.9213 | 10.9405 | 10.9351 | 10.8852 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 10.9646 | 10.9244 | 10.8852 | 10.9595 | 11.3103 | 13.6758 | 80.0482 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 9
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 444
