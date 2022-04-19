# IAH1 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 40.9612
- Baseline Model [RMSE] = 48.2679
- Trained Model [MAE] = 23.0500
- Trained Model [RMSE] = 30.3145
- Prediction [MAE] = 19.3434
- Prediction [RMSE] = 24.5159
***

### Dataset Statistics

- LOFO Field [Mean] = 190.9080 [bu/A]
- LOFO Field [Standard Deviation] = 29.2959 [bu/A]
- Model Dataset [Mean] = 152.5255 [bu/A]
- Model Dataset [Standard Deviation] = 45.4045 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 24.4216 | 24.4216 | 24.4216 | 24.4216 | 24.4216 | 24.4216 | 24.4303 | 24.3521 | 24.3548 | 24.4792 | 24.3515 | 24.3515 | 24.3946 | 24.3946 | 24.3593 | 24.3754 | 24.3754 | 24.3742 | 24.382  | 24.4108 | 24.4069 |
|          2 | 22.9545 | 22.9545 | 22.9582 | 23.0367 | 23.0628 | 23.0607 | 22.9883 | 22.9117 | 23.2653 | 22.9651 | 23.0223 | 23.0649 | 22.9379 | 22.9734 | 23.1081 | 23.0091 | 23.1075 | 23.0722 | 23.0263 | 23.1073 | 23.027  |
|          3 | 22.8707 | 22.8707 | 23.0267 | 22.9117 | 22.8404 | 23.0156 | 22.8181 | 22.8607 | 22.8909 | 23.0058 | 22.8828 | 22.83   | 22.9181 | 23.0735 | 22.9764 | 23.0135 | 22.9074 | 23.0146 | 23.1396 | 22.9684 | 22.9317 |
|          4 | 22.8665 | 22.8665 | 22.8567 | 22.9695 | 22.8978 | 22.9016 | 22.8506 | 22.8312 | 22.8538 | 22.885  | 22.8971 | 22.916  | 22.9086 | 22.804  | 22.8374 | 22.9239 | 22.8648 | 22.8757 | 22.9351 | 22.9489 | 22.8412 |
|          5 | 22.9022 | 22.9022 | 22.8996 | 22.8199 | 22.8787 | 22.8743 | 22.9163 | 22.7842 | 22.9181 | 22.773  | 22.9132 | 22.8549 | 22.8214 | 22.89   | 22.7524 | 22.8673 | 22.8944 | 22.8983 | 22.8825 | 22.8573 | 22.9577 |
|          6 | 22.7817 | 22.7817 | 22.9581 | 22.9443 | 22.9488 | 22.7683 | 22.8973 | 22.9593 | 22.8428 | 22.8021 | 22.8581 | 22.9493 | 22.8887 | 22.8112 | 22.876  | 22.9363 | 22.8659 | 22.8616 | 22.974  | 22.907  | 22.9448 |
|          7 | 22.8686 | 22.8686 | 22.9608 | 22.924  | 22.8726 | 22.8442 | 22.8371 | 23.0075 | 22.8469 | 22.8385 | 22.9163 | 22.8116 | 22.8532 | 22.9072 | 22.8345 | 23.0012 | 23.0169 | 23.0208 | 22.9793 | 22.8973 | 22.8889 |
|          8 | 23.0263 | 23.0263 | 23.0247 | 23.0249 | 22.9207 | 23.0044 | 22.9951 | 22.9757 | 22.9461 | 23.0095 | 23.0348 | 23.0436 | 22.9631 | 22.839  | 23.0495 | 22.946  | 22.9848 | 22.8865 | 23.016  | 23.0101 | 22.8839 |
|          9 | 23.0803 | 23.0803 | 23.0896 | 23.0262 | 23.0485 | 22.9826 | 23.149  | 22.9076 | 23.0905 | 23.0172 | 23.0775 | 22.9208 | 22.928  | 22.9798 | 23.0005 | 23.0386 | 22.9444 | 22.9444 | 22.9646 | 23.0854 | 23.019  |
|         10 | 23.3032 | 23.3032 | 23.2451 | 23.1311 | 23.1912 | 23.0737 | 23.0947 | 23.0065 | 23.1383 | 23.1254 | 23.135  | 23.0005 | 22.9592 | 23.0379 | 23.0476 | 23.0433 | 23.0433 | 22.9932 | 22.9375 | 22.897  | 22.9642 |
|         11 | 23.3905 | 23.3905 | 23.233  | 23.2079 | 23.2659 | 23.2818 | 23.184  | 23.1377 | 23.0368 | 23.2261 | 23.1878 | 23.0245 | 23.0637 | 23.0411 | 23.1283 | 23.0677 | 23.0026 | 22.9738 | 23.0694 | 23.0456 | 23.0467 |
|         12 | 23.5478 | 23.5478 | 23.3966 | 23.3255 | 23.2607 | 23.4001 | 23.3328 | 23.1931 | 23.1136 | 23.2119 | 23.2558 | 23.1073 | 22.9991 | 23.0675 | 23.1818 | 23.0473 | 23.0028 | 23.1302 | 23.0581 | 23.0455 | 23.1109 |
|         13 | 23.7053 | 23.7053 | 23.5649 | 23.3773 | 23.2944 | 23.2349 | 23.3526 | 23.1949 | 23.3313 | 23.1716 | 23.2223 | 23.123  | 23.0624 | 23.139  | 23.2356 | 23.0567 | 23.167  | 23.0627 | 22.9569 | 23.0663 | 22.9838 |
|         14 | 23.8246 | 23.8246 | 23.7131 | 23.431  | 23.4975 | 23.34   | 23.3592 | 23.2351 | 23.2644 | 23.1771 | 23.1541 | 23.0997 | 23.0881 | 23.1315 | 23.1421 | 23.0645 | 23.0814 | 23.0135 | 23.0785 | 23.0428 | 23.0223 |
|         15 | 24.0641 | 24.0641 | 23.8342 | 23.5879 | 23.5466 | 23.453  | 23.3486 | 23.4362 | 23.3569 | 23.1994 | 23.3221 | 23.2393 | 23.2227 | 23.1129 | 23.1334 | 23.1601 | 23.1551 | 23.0219 | 23.0153 | 23.0125 | 23.0101 |
|         16 | 24.1527 | 24.1527 | 23.933  | 23.736  | 23.6243 | 23.4797 | 23.4201 | 23.3776 | 23.3842 | 23.2791 | 23.3222 | 23.2585 | 23.1639 | 23.2546 | 23.1381 | 23.1349 | 23.0795 | 23.0158 | 23.0078 | 23.1136 | 23.025  |
|         17 | 24.4154 | 24.4154 | 23.9509 | 23.8702 | 23.7171 | 23.6042 | 23.4117 | 23.4128 | 23.4266 | 23.2825 | 23.3436 | 23.3087 | 23.218  | 23.2807 | 23.1445 | 23.1961 | 23.0992 | 23.023  | 23.0444 | 23.0385 | 23.0873 |
|         18 | 24.4765 | 24.4765 | 24.0261 | 23.8092 | 23.6169 | 23.6226 | 23.5295 | 23.5152 | 23.3616 | 23.2896 | 23.3753 | 23.2203 | 23.244  | 23.1394 | 23.1892 | 23.1031 | 23.0813 | 23.1492 | 23.1366 | 22.9824 | 23.0092 |
|         19 | 24.5193 | 24.5193 | 24.0534 | 23.953  | 23.6635 | 23.5589 | 23.4866 | 23.4476 | 23.3493 | 23.3534 | 23.3657 | 23.2115 | 23.2365 | 23.1903 | 23.2405 | 23.0955 | 23.0952 | 23.0579 | 23.0443 | 23.0999 | 23.114  |
|         20 | 24.5603 | 24.5603 | 24.1476 | 23.9388 | 23.8304 | 23.7004 | 23.5359 | 23.4935 | 23.3346 | 23.2762 | 23.2431 | 23.3093 | 23.1928 | 23.1931 | 23.1472 | 23.119  | 23.0452 | 23.1462 | 23.1258 | 23.0441 | 23.0863 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.0214 | 24.8929 | 24.8127 | 25.0035 | 25.0217 | 25.0179 | 24.9751 | 25.0375 | 25.0295 | 24.9596 |
|          0.2 | 24.287  | 24.4964 | 24.4515 | 24.458  | 24.4641 | 24.5166 | 24.5241 | 24.2864 | 24.3314 | 24.4424 |
|          0.3 | 24.1563 | 24.046  | 23.862  | 24.0699 | 24.0603 | 24.1868 | 24.1823 | 23.9401 | 24.1845 | 24.048  |
|          0.4 | 23.8056 | 23.8144 | 23.7643 | 23.7551 | 23.7334 | 23.8367 | 23.7284 | 23.8252 | 23.8599 | 23.6579 |
|          0.5 | 23.7602 | 23.7754 | 23.6444 | 23.6218 | 23.4603 | 23.5141 | 23.7696 | 23.5732 | 23.5314 | 23.5462 |
|          0.6 | 23.5821 | 23.6395 | 23.5677 | 23.4275 | 23.5398 | 23.3434 | 23.5262 | 23.4004 | 23.3606 | 23.36   |
|          0.7 | 23.3772 | 23.2896 | 23.5312 | 23.3501 | 23.2297 | 23.2564 | 23.2935 | 23.2016 | 23.1508 | 23.1734 |
|          0.8 | 23.0948 | 23.3154 | 23.221  | 23.3642 | 23.1813 | 23.2905 | 23.0362 | 23.0934 | 23.2963 | 23.1001 |
|          0.9 | 23.2508 | 23.1422 | 22.8564 | 23.0103 | 23.0158 | 23.0528 | 23.1343 | 23.0432 | 22.9093 | 22.9941 |
|          1   | 23.3464 | 22.9542 | 22.9568 | 22.9076 | 22.867  | 22.8302 | 22.8494 | 22.9165 | 22.9414 | 22.7524 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 23.0984 | 22.9753 | 22.7524 | 22.7652 | 22.6746 | 23.1138 | 58.0485 |

***

### Tuned Parameters

- max_depth = 5
- min_child_weight = 14
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.1
- num_boost_round = 243