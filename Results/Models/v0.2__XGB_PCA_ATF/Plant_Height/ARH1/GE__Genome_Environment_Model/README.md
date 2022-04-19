# ARH1 - Plant_Height Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 20.4624
- Baseline Model [RMSE] = 25.8319
- Trained Model [MAE] = 10.7161
- Trained Model [RMSE] = 14.3552
- Prediction [MAE] = 22.4840
- Prediction [RMSE] = 27.0168
***

### Dataset Statistics

- LOFO Field [Mean] = 199.7125 [cm]
- LOFO Field [Standard Deviation] = 17.9702 [cm]
- Model Dataset [Mean] = 218.2947 [cm]
- Model Dataset [Standard Deviation] = 40.3116 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 12.978  | 12.978  | 12.978  | 12.978  | 12.978  | 12.9762 | 12.9737 | 12.9732 | 12.9759 | 12.9759 | 12.9759 | 12.9759 | 12.9759 | 12.9768 | 12.9768 | 12.9755 | 12.9753 | 12.9758 | 12.9747 | 12.9727 | 12.9727 |
|          2 | 11.17   | 11.17   | 11.0185 | 11.0297 | 11.0408 | 11.041  | 11.0627 | 11.1034 | 11.0254 | 11.0177 | 11.1003 | 11.0829 | 11.0751 | 11.1333 | 11.2627 | 11.0333 | 11.1051 | 11.1753 | 11.062  | 11.1736 | 11.1407 |
|          3 | 11.109  | 11.109  | 11.1161 | 11.0355 | 11.1525 | 11.0377 | 11.0895 | 11.0261 | 11.0842 | 11.0283 | 11.1087 | 11.0885 | 11.1185 | 11.086  | 11.1215 | 11.1172 | 11.0921 | 11.1491 | 11.2192 | 11.126  | 11.1217 |
|          4 | 11.1441 | 11.1441 | 11.1517 | 11.1467 | 11.1037 | 11.1286 | 11.1369 | 11.0587 | 11.1107 | 11.1297 | 11.1348 | 11.1473 | 11.1508 | 11.1183 | 11.1188 | 11.1294 | 11.1411 | 11.1397 | 11.1845 | 11.2166 | 11.1738 |
|          5 | 11.3004 | 11.3004 | 11.2191 | 11.2343 | 11.2189 | 11.1742 | 11.158  | 11.1737 | 11.2084 | 11.1874 | 11.1174 | 11.1241 | 11.1763 | 11.2014 | 11.168  | 11.2021 | 11.1924 | 11.2068 | 11.1439 | 11.1998 | 11.1755 |
|          6 | 11.2943 | 11.2943 | 11.3125 | 11.2938 | 11.2779 | 11.2338 | 11.159  | 11.2731 | 11.2023 | 11.2667 | 11.2153 | 11.1774 | 11.1854 | 11.2694 | 11.2837 | 11.2361 | 11.2394 | 11.2887 | 11.3179 | 11.2965 | 11.2801 |
|          7 | 11.4533 | 11.4533 | 11.4002 | 11.3396 | 11.321  | 11.3485 | 11.3272 | 11.3728 | 11.3127 | 11.3008 | 11.2955 | 11.3574 | 11.3244 | 11.2579 | 11.3221 | 11.3106 | 11.3368 | 11.2989 | 11.2977 | 11.2888 | 11.306  |
|          8 | 11.5009 | 11.5009 | 11.4531 | 11.4799 | 11.4063 | 11.4265 | 11.4181 | 11.2936 | 11.3347 | 11.3171 | 11.3217 | 11.3351 | 11.3062 | 11.3165 | 11.3212 | 11.3306 | 11.3227 | 11.3389 | 11.3027 | 11.3416 | 11.3885 |
|          9 | 11.5736 | 11.5736 | 11.5525 | 11.4433 | 11.5296 | 11.5146 | 11.4756 | 11.4143 | 11.3901 | 11.4368 | 11.4075 | 11.3587 | 11.3872 | 11.3862 | 11.385  | 11.3222 | 11.3496 | 11.3342 | 11.3558 | 11.3497 | 11.369  |
|         10 | 11.644  | 11.644  | 11.5655 | 11.5393 | 11.5689 | 11.4808 | 11.4139 | 11.4228 | 11.4546 | 11.4466 | 11.464  | 11.4146 | 11.426  | 11.4422 | 11.4224 | 11.4302 | 11.4097 | 11.4054 | 11.4227 | 11.3613 | 11.3454 |
|         11 | 11.7151 | 11.7151 | 11.6243 | 11.5947 | 11.5735 | 11.56   | 11.5315 | 11.5717 | 11.4838 | 11.5006 | 11.4797 | 11.4448 | 11.445  | 11.484  | 11.4658 | 11.4372 | 11.4457 | 11.4208 | 11.4076 | 11.4831 | 11.4194 |
|         12 | 11.8009 | 11.8009 | 11.7301 | 11.7089 | 11.6581 | 11.6065 | 11.5566 | 11.4894 | 11.5409 | 11.5585 | 11.5262 | 11.4521 | 11.483  | 11.4613 | 11.4788 | 11.5232 | 11.4188 | 11.418  | 11.5158 | 11.4418 | 11.4504 |
|         13 | 11.8881 | 11.8881 | 11.7978 | 11.6792 | 11.6695 | 11.6047 | 11.6856 | 11.6063 | 11.5474 | 11.6039 | 11.5415 | 11.5503 | 11.504  | 11.5169 | 11.4781 | 11.489  | 11.5033 | 11.4769 | 11.4501 | 11.4635 | 11.4009 |
|         14 | 12.0179 | 12.0179 | 11.8906 | 11.8289 | 11.7785 | 11.6858 | 11.6773 | 11.6194 | 11.6534 | 11.5753 | 11.5451 | 11.4805 | 11.5467 | 11.511  | 11.5203 | 11.4854 | 11.4477 | 11.5212 | 11.4628 | 11.4139 | 11.438  |
|         15 | 12.0017 | 12.0017 | 11.9324 | 11.8816 | 11.7819 | 11.7612 | 11.6993 | 11.6501 | 11.6606 | 11.5639 | 11.5627 | 11.6133 | 11.5091 | 11.4949 | 11.562  | 11.5152 | 11.4854 | 11.4842 | 11.4358 | 11.4439 | 11.4235 |
|         16 | 12.0723 | 12.0723 | 11.9796 | 11.9044 | 11.8317 | 11.7853 | 11.7282 | 11.6941 | 11.658  | 11.5505 | 11.5906 | 11.6396 | 11.5016 | 11.5342 | 11.4666 | 11.5242 | 11.5196 | 11.5067 | 11.5179 | 11.4832 | 11.516  |
|         17 | 12.1685 | 12.1685 | 11.9884 | 11.9409 | 11.846  | 11.7202 | 11.7293 | 11.7243 | 11.704  | 11.6188 | 11.6273 | 11.5722 | 11.5394 | 11.5705 | 11.5026 | 11.5673 | 11.518  | 11.4997 | 11.5031 | 11.5036 | 11.4943 |
|         18 | 12.2066 | 12.2066 | 12.1064 | 11.9164 | 11.9433 | 11.7862 | 11.7597 | 11.7797 | 11.661  | 11.6529 | 11.582  | 11.6067 | 11.5455 | 11.6065 | 11.539  | 11.5124 | 11.5181 | 11.5197 | 11.4884 | 11.4658 | 11.4835 |
|         19 | 12.1926 | 12.1926 | 12.1951 | 12.0528 | 11.9352 | 11.8003 | 11.7947 | 11.7673 | 11.7161 | 11.6147 | 11.6261 | 11.5852 | 11.5244 | 11.5537 | 11.5651 | 11.5633 | 11.4872 | 11.5217 | 11.4948 | 11.4872 | 11.4669 |
|         20 | 12.2335 | 12.2335 | 12.1228 | 12.0224 | 11.9439 | 11.7988 | 11.8077 | 11.7312 | 11.7532 | 11.6861 | 11.6438 | 11.6069 | 11.5066 | 11.5945 | 11.4737 | 11.553  | 11.5134 | 11.5489 | 11.5041 | 11.4985 | 11.4997 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 13.7023 | 13.6141 | 13.6592 | 13.5571 | 13.6751 | 13.447  | 13.339  | 13.7561 | 13.5381 | 13.5652 |
|          0.2 | 13.923  | 13.0175 | 13.0515 | 12.9944 | 12.974  | 13.0105 | 13.0375 | 12.7396 | 12.8179 | 12.7701 |
|          0.3 | 13.0386 | 12.8805 | 12.6842 | 12.57   | 12.9601 | 12.8815 | 12.6765 | 12.5741 | 12.4542 | 12.7153 |
|          0.4 | 13.0483 | 12.7547 | 12.9306 | 12.3026 | 12.5978 | 12.2574 | 12.4705 | 12.3293 | 12.6294 | 12.3514 |
|          0.5 | 12.9623 | 12.6792 | 12.3825 | 12.5458 | 12.2813 | 12.3887 | 12.2757 | 12.263  | 12.3466 | 12.2308 |
|          0.6 | 12.5706 | 12.5876 | 12.1849 | 11.9552 | 12.2986 | 11.9069 | 11.7802 | 11.7492 | 11.7967 | 11.9579 |
|          0.7 | 12.5024 | 12.2367 | 12.0444 | 12.0988 | 11.9238 | 12.0376 | 11.7421 | 11.7001 | 11.6454 | 11.7954 |
|          0.8 | 12.9919 | 12.0707 | 11.6259 | 11.7045 | 11.713  | 11.493  | 11.6124 | 11.6364 | 11.4852 | 11.3753 |
|          0.9 | 12.0559 | 11.6592 | 11.6012 | 11.5049 | 11.6997 | 11.2188 | 11.2868 | 11.3397 | 11.3674 | 11.3291 |
|          1   | 11.982  | 11.7715 | 11.1797 | 11.2331 | 11.398  | 11.0751 | 11.0233 | 11.139  | 11.0685 | 11.0177 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |   0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|--------|---------|
| MAE   | 11.0171 | 11.0419 | 11.0177 | 11.0351 | 11.4122 | 13.703 |  80.361 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 9
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.5
- num_boost_round = 239