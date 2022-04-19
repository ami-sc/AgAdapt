# GAH1 - Plant_Height Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 36.3029
- Baseline Model [RMSE] = 40.8312
- Trained Model [MAE] = 10.6746
- Trained Model [RMSE] = 14.2971
- Prediction [MAE] = 28.1482
- Prediction [RMSE] = 31.8136
***

### Dataset Statistics

- LOFO Field [Mean] = 182.9388 [cm]
- LOFO Field [Standard Deviation] = 19.9676 [cm]
- Model Dataset [Mean] = 218.5773 [cm]
- Model Dataset [Standard Deviation] = 39.7899 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 12.9213 | 12.9213 | 12.9213 | 12.9213 | 12.9213 | 12.9213 | 12.9197 | 12.9189 | 12.9209 | 12.9209 | 12.9209 | 12.9213 | 12.9213 | 12.9213 | 12.9235 | 12.9235 | 12.9246 | 12.9239 | 12.9239 | 12.9233 | 12.9232 |
|          2 | 11.0354 | 11.0354 | 11.0114 | 11.0497 | 10.9767 | 11.1734 | 11.1361 | 11.1211 | 11.1801 | 11.107  | 11.1586 | 11.1692 | 11.1156 | 11.1438 | 11.1607 | 11.0867 | 11.1327 | 11.1263 | 11.1786 | 11.1904 | 11.2833 |
|          3 | 11.1508 | 11.1508 | 11.1524 | 11.1173 | 11.1319 | 11.2211 | 11.1641 | 11.2388 | 11.1703 | 11.1525 | 11.108  | 11.1457 | 11.1202 | 11.1081 | 11.2028 | 11.223  | 11.1556 | 11.2071 | 11.1924 | 11.2056 | 11.1797 |
|          4 | 11.3027 | 11.3027 | 11.2645 | 11.2218 | 11.2224 | 11.2375 | 11.2159 | 11.2186 | 11.2298 | 11.1617 | 11.1537 | 11.1845 | 11.175  | 11.2025 | 11.2266 | 11.1896 | 11.2109 | 11.2402 | 11.188  | 11.2149 | 11.2436 |
|          5 | 11.3407 | 11.3407 | 11.2926 | 11.3396 | 11.3054 | 11.3112 | 11.2995 | 11.2582 | 11.2547 | 11.2273 | 11.2423 | 11.2647 | 11.3266 | 11.2648 | 11.2798 | 11.284  | 11.2525 | 11.247  | 11.2494 | 11.2512 | 11.2652 |
|          6 | 11.4226 | 11.4226 | 11.3708 | 11.3175 | 11.3547 | 11.3784 | 11.3427 | 11.3339 | 11.3387 | 11.3875 | 11.2427 | 11.3465 | 11.3092 | 11.2376 | 11.3036 | 11.2425 | 11.2789 | 11.278  | 11.2864 | 11.3284 | 11.3268 |
|          7 | 11.5089 | 11.5089 | 11.4477 | 11.4494 | 11.4179 | 11.4006 | 11.4129 | 11.4212 | 11.3736 | 11.3389 | 11.3892 | 11.3798 | 11.3555 | 11.3127 | 11.3509 | 11.3053 | 11.3119 | 11.3199 | 11.2888 | 11.3822 | 11.3596 |
|          8 | 11.5077 | 11.5077 | 11.4355 | 11.4588 | 11.4332 | 11.4612 | 11.4472 | 11.4534 | 11.4576 | 11.4182 | 11.4005 | 11.4324 | 11.4227 | 11.3782 | 11.3745 | 11.3327 | 11.3751 | 11.4079 | 11.4253 | 11.3902 | 11.365  |
|          9 | 11.597  | 11.597  | 11.6217 | 11.6085 | 11.5055 | 11.5061 | 11.469  | 11.4759 | 11.4982 | 11.4975 | 11.4006 | 11.4356 | 11.4671 | 11.4249 | 11.3762 | 11.4398 | 11.3874 | 11.4124 | 11.3711 | 11.401  | 11.4068 |
|         10 | 11.7097 | 11.7097 | 11.7138 | 11.5936 | 11.5434 | 11.5316 | 11.5703 | 11.563  | 11.4854 | 11.5326 | 11.4693 | 11.4688 | 11.4403 | 11.4395 | 11.4064 | 11.4298 | 11.4519 | 11.4285 | 11.3945 | 11.3924 | 11.4339 |
|         11 | 11.8154 | 11.8154 | 11.7088 | 11.6665 | 11.613  | 11.618  | 11.568  | 11.5625 | 11.5766 | 11.5423 | 11.4879 | 11.5183 | 11.4583 | 11.5044 | 11.5472 | 11.5212 | 11.4352 | 11.3808 | 11.4512 | 11.3572 | 11.4659 |
|         12 | 11.8577 | 11.8577 | 11.791  | 11.7462 | 11.7274 | 11.6234 | 11.6345 | 11.5669 | 11.5459 | 11.5869 | 11.5558 | 11.5003 | 11.4902 | 11.5032 | 11.4988 | 11.4623 | 11.4775 | 11.5019 | 11.4131 | 11.4802 | 11.429  |
|         13 | 11.9169 | 11.9169 | 11.9007 | 11.7086 | 11.7227 | 11.7404 | 11.699  | 11.6798 | 11.6518 | 11.5609 | 11.6131 | 11.5308 | 11.5753 | 11.47   | 11.5419 | 11.5169 | 11.5201 | 11.4347 | 11.4843 | 11.4799 | 11.4668 |
|         14 | 12.0673 | 12.0673 | 11.9186 | 11.8235 | 11.8006 | 11.7399 | 11.7214 | 11.6883 | 11.6752 | 11.6094 | 11.5908 | 11.5965 | 11.5581 | 11.5984 | 11.5549 | 11.5385 | 11.5426 | 11.5727 | 11.4879 | 11.4059 | 11.4289 |
|         15 | 12.1006 | 12.1006 | 11.9833 | 11.8783 | 11.7662 | 11.7681 | 11.6969 | 11.693  | 11.6442 | 11.6626 | 11.6369 | 11.6339 | 11.5607 | 11.5444 | 11.4995 | 11.5182 | 11.4799 | 11.5536 | 11.4386 | 11.4646 | 11.4406 |
|         16 | 12.0737 | 12.0737 | 12.0349 | 11.8637 | 11.8287 | 11.7711 | 11.7383 | 11.6357 | 11.6591 | 11.7001 | 11.6255 | 11.5645 | 11.6027 | 11.5882 | 11.5816 | 11.5087 | 11.5137 | 11.5289 | 11.4692 | 11.4106 | 11.5149 |
|         17 | 12.1804 | 12.1804 | 12.0424 | 11.9739 | 11.8292 | 11.8311 | 11.7944 | 11.7332 | 11.6974 | 11.6301 | 11.6563 | 11.6279 | 11.5579 | 11.5805 | 11.5275 | 11.5411 | 11.5651 | 11.5164 | 11.4576 | 11.4669 | 11.4802 |
|         18 | 12.2286 | 12.2286 | 12.0881 | 11.9472 | 11.891  | 11.8424 | 11.7768 | 11.7014 | 11.6686 | 11.6731 | 11.6759 | 11.6114 | 11.5747 | 11.5853 | 11.5852 | 11.5462 | 11.5623 | 11.5248 | 11.5143 | 11.5147 | 11.5064 |
|         19 | 12.2522 | 12.2522 | 12.1435 | 11.9851 | 11.9    | 11.8964 | 11.7977 | 11.7377 | 11.7459 | 11.6713 | 11.6858 | 11.646  | 11.639  | 11.5945 | 11.6177 | 11.5466 | 11.5709 | 11.5234 | 11.4969 | 11.4768 | 11.4242 |
|         20 | 12.2985 | 12.2985 | 12.2278 | 11.9743 | 11.9466 | 11.8692 | 11.8316 | 11.7461 | 11.6941 | 11.7753 | 11.6774 | 11.6681 | 11.6318 | 11.5685 | 11.6058 | 11.5592 | 11.5905 | 11.5764 | 11.481  | 11.5209 | 11.4434 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 13.8047 | 13.7167 | 13.759  | 13.5603 | 13.6636 | 13.7638 | 13.5402 | 13.2725 | 13.6553 | 13.6481 |
|          0.2 | 13.3441 | 13.2144 | 13.0425 | 13.0168 | 13.004  | 12.9852 | 13.1344 | 12.7772 | 13.1218 | 12.9969 |
|          0.3 | 13.1832 | 13.0327 | 12.8547 | 12.8768 | 12.8634 | 12.7103 | 12.6733 | 12.7496 | 12.733  | 12.4756 |
|          0.4 | 13.1256 | 12.8404 | 12.5829 | 12.4247 | 12.1764 | 12.4533 | 12.242  | 12.441  | 12.3811 | 12.1396 |
|          0.5 | 12.785  | 12.6909 | 12.5669 | 12.2201 | 12.3526 | 11.9918 | 12.2592 | 12.1713 | 11.9388 | 12.2575 |
|          0.6 | 12.707  | 12.4032 | 12.3624 | 12.3221 | 12.0685 | 12.1553 | 11.9321 | 11.9087 | 12.0413 | 11.9486 |
|          0.7 | 12.1339 | 11.998  | 12.1099 | 11.9368 | 11.6747 | 11.6787 | 11.7009 | 11.7291 | 11.6192 | 11.6752 |
|          0.8 | 12.0817 | 12.3804 | 11.695  | 11.8813 | 11.6108 | 11.7191 | 11.5867 | 11.7058 | 11.5856 | 11.5586 |
|          0.9 | 12.0384 | 11.693  | 11.4845 | 11.5149 | 11.5726 | 11.3673 | 11.3501 | 11.291  | 11.2744 | 11.2681 |
|          1   | 11.6453 | 11.3957 | 11.2917 | 11.3536 | 11.1093 | 11.0475 | 11.057  | 11.0993 | 11.0794 | 10.9767 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 11.0071 | 11.0443 | 10.9767 | 11.2031 | 11.4311 | 13.6548 | 80.4003 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 4
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 427