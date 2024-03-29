# GAH2 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 32.1528
- Baseline Model [RMSE] = 42.4481
- Trained Model [MAE] = 35.1474
- Trained Model [RMSE] = 44.0131
- Prediction [MAE] = 29.0159
- Prediction [RMSE] = 38.3047
***

### Dataset Statistics

- LOFO Field [Mean] = 128.0962 [bu/A]
- LOFO Field [Standard Deviation] = 32.5976 [bu/A]
- Model Dataset [Mean] = 155.4093 [bu/A]
- Model Dataset [Standard Deviation] = 45.5852 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 35.5737 | 35.5737 | 35.5737 | 35.5147 | 35.5147 | 35.5139 | 35.5139 | 35.5576 | 35.548  | 35.5439 | 35.5428 | 35.5147 | 35.5274 | 35.5099 | 35.5099 | 35.5099 | 35.5099 | 35.5099 | 35.5099 | 35.5832 | 35.5832 |
|          2 | 35.0948 | 35.0948 | 35.0948 | 35.0225 | 35.0341 | 35.0204 | 34.9303 | 35.066  | 35.0296 | 34.9871 | 35.0899 | 34.937  | 34.9846 | 35.018  | 35.0126 | 35.1791 | 35.1057 | 34.9893 | 35.0761 | 35.2958 | 35.1441 |
|          3 | 34.8652 | 34.8652 | 34.868  | 34.9034 | 34.9557 | 34.9065 | 34.9095 | 34.9023 | 34.8435 | 34.9092 | 34.8886 | 34.917  | 34.8718 | 34.9044 | 34.8667 | 34.9212 | 34.8725 | 34.9125 | 34.9377 | 34.9976 | 34.9547 |
|          4 | 34.8726 | 34.8726 | 34.8653 | 34.8883 | 34.8739 | 34.8864 | 34.8423 | 34.8576 | 34.8778 | 34.8593 | 34.8876 | 34.8606 | 34.8308 | 34.849  | 34.8276 | 34.8804 | 34.8636 | 34.8939 | 34.9001 | 34.9264 | 34.9428 |
|          5 | 34.8206 | 34.8206 | 34.8103 | 34.8383 | 34.8252 | 34.8596 | 34.833  | 34.8179 | 34.7977 | 34.8604 | 34.8448 | 34.8428 | 34.8307 | 34.8332 | 34.8737 | 34.8635 | 34.8525 | 34.8603 | 34.8123 | 34.8722 | 34.8883 |
|          6 | 34.8789 | 34.8789 | 34.8755 | 34.8898 | 34.8679 | 34.8555 | 34.836  | 34.8021 | 34.8084 | 34.7826 | 34.8009 | 34.8064 | 34.8507 | 34.8218 | 34.8189 | 34.8626 | 34.8453 | 34.8623 | 34.8307 | 34.8749 | 34.8565 |
|          7 | 34.8526 | 34.8525 | 34.8571 | 34.8354 | 34.8476 | 34.8501 | 34.8613 | 34.7792 | 34.8184 | 34.8512 | 34.8677 | 34.8187 | 34.7931 | 34.8338 | 34.8386 | 34.8559 | 34.8126 | 34.8139 | 34.8364 | 34.8412 | 34.8198 |
|          8 | 34.8902 | 34.8902 | 34.8646 | 34.8669 | 34.8681 | 34.8497 | 34.8484 | 34.8525 | 34.831  | 34.8294 | 34.8235 | 34.8163 | 34.8709 | 34.8165 | 34.8536 | 34.8312 | 34.8668 | 34.8503 | 34.8598 | 34.8434 | 34.8501 |
|          9 | 34.9264 | 34.9264 | 34.896  | 34.9126 | 34.8988 | 34.9393 | 34.8964 | 34.8586 | 34.8776 | 34.8819 | 34.8483 | 34.8487 | 34.8815 | 34.8601 | 34.847  | 34.8446 | 34.8522 | 34.8736 | 34.8603 | 34.8727 | 34.8662 |
|         10 | 34.95   | 34.95   | 34.949  | 34.9514 | 34.934  | 34.9174 | 34.9256 | 34.8911 | 34.8809 | 34.9054 | 34.8881 | 34.8938 | 34.882  | 34.8483 | 34.882  | 34.8243 | 34.8793 | 34.8636 | 34.8543 | 34.822  | 34.8868 |
|         11 | 35.0326 | 35.0326 | 35.0294 | 34.9961 | 35.0068 | 34.9652 | 34.9577 | 34.9207 | 34.8904 | 34.9289 | 34.9028 | 34.896  | 34.9145 | 34.9004 | 34.8735 | 34.867  | 34.8724 | 34.8394 | 34.8692 | 34.8524 | 34.9421 |
|         12 | 35.0741 | 35.0741 | 35.0805 | 35.0274 | 35.0102 | 34.9988 | 34.9806 | 34.956  | 34.9411 | 34.947  | 34.9562 | 34.9394 | 34.9762 | 34.9136 | 34.8678 | 34.8536 | 34.9098 | 34.9352 | 34.8777 | 34.9014 | 34.9231 |
|         13 | 35.1056 | 35.1056 | 35.0917 | 35.0799 | 35.0359 | 35.0266 | 35.0244 | 34.9814 | 34.978  | 34.9525 | 35.0092 | 34.9733 | 34.9977 | 34.9528 | 34.931  | 34.8875 | 34.8975 | 34.911  | 34.9222 | 34.8995 | 34.9391 |
|         14 | 35.1583 | 35.1583 | 35.1403 | 35.0977 | 35.08   | 35.084  | 35.0369 | 35.0058 | 35.0069 | 34.9924 | 34.9831 | 34.9865 | 34.9752 | 34.9618 | 34.9547 | 34.9198 | 34.9138 | 34.9315 | 34.9071 | 34.9169 | 34.9313 |
|         15 | 35.158  | 35.158  | 35.1552 | 35.1362 | 35.101  | 35.1057 | 35.0595 | 35.0124 | 34.9995 | 35.0054 | 35.0182 | 35.0001 | 34.9995 | 34.9621 | 34.9715 | 34.9247 | 34.9185 | 34.9238 | 34.931  | 34.9082 | 34.9826 |
|         16 | 35.1917 | 35.1917 | 35.1618 | 35.1679 | 35.1213 | 35.0963 | 35.0864 | 35.052  | 35.0458 | 35.0374 | 35.0363 | 35.0144 | 35.0068 | 34.9883 | 34.9777 | 34.9254 | 34.9523 | 34.9343 | 34.9415 | 34.9434 | 34.9722 |
|         17 | 35.2207 | 35.2208 | 35.2142 | 35.1855 | 35.1436 | 35.1125 | 35.0806 | 35.0516 | 35.049  | 35.0579 | 35.0297 | 35.0541 | 35.0257 | 34.9984 | 34.9641 | 34.9472 | 34.948  | 34.9678 | 34.9339 | 34.9483 | 34.966  |
|         18 | 35.2537 | 35.2537 | 35.2177 | 35.1976 | 35.1648 | 35.1516 | 35.1003 | 35.0754 | 35.0709 | 35.0658 | 35.0514 | 35.0378 | 35.0425 | 35.0213 | 34.9934 | 34.9732 | 34.9577 | 34.9627 | 34.9453 | 34.954  | 34.9821 |
|         19 | 35.2595 | 35.2595 | 35.2312 | 35.2117 | 35.1737 | 35.1642 | 35.1256 | 35.0817 | 35.0878 | 35.0731 | 35.0491 | 35.0472 | 35.0336 | 35.0176 | 34.9931 | 34.9766 | 34.9601 | 34.9804 | 34.9608 | 34.9657 | 34.9703 |
|         20 | 35.256  | 35.256  | 35.251  | 35.2045 | 35.1926 | 35.1747 | 35.1146 | 35.0883 | 35.0999 | 35.0845 | 35.0703 | 35.072  | 35.0297 | 35.0282 | 35.0272 | 34.9982 | 35.0035 | 35.0027 | 34.9627 | 34.9446 | 34.9789 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 35.9795 | 35.9795 | 35.9795 | 35.9795 | 35.9795 | 35.9795 | 35.9795 | 35.9795 | 35.9795 | 35.779  |
|          0.2 | 35.7444 | 35.7444 | 35.7444 | 35.7444 | 35.7444 | 35.7444 | 35.7444 | 35.7444 | 35.7444 | 35.3774 |
|          0.3 | 35.3702 | 35.3702 | 35.3702 | 35.3702 | 35.3702 | 35.3702 | 35.3702 | 35.3702 | 35.3702 | 35.2171 |
|          0.4 | 35.3424 | 35.3424 | 35.3424 | 35.3424 | 35.3424 | 35.3424 | 35.3424 | 35.3424 | 35.3424 | 35.0366 |
|          0.5 | 35.1642 | 35.1642 | 35.1642 | 35.1642 | 35.1642 | 35.1642 | 35.1642 | 35.1642 | 35.1642 | 35.0451 |
|          0.6 | 35.0613 | 35.0613 | 35.0613 | 35.0613 | 35.0613 | 35.0613 | 35.0613 | 35.0613 | 35.0613 | 34.9076 |
|          0.7 | 34.9658 | 34.9658 | 34.9658 | 34.9658 | 34.9658 | 34.9658 | 34.9658 | 34.9658 | 34.9658 | 34.9962 |
|          0.8 | 34.9803 | 34.9803 | 34.9803 | 34.9803 | 34.9803 | 34.9803 | 34.9803 | 34.9803 | 34.9803 | 34.9364 |
|          0.9 | 34.8956 | 34.8956 | 34.8956 | 34.8956 | 34.8956 | 34.8956 | 34.8956 | 34.8956 | 34.8956 | 34.8232 |
|          1   | 34.9012 | 34.9012 | 34.9012 | 34.9012 | 34.9012 | 34.9012 | 34.9012 | 34.9012 | 34.9012 | 34.7792 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 34.8469 | 34.8579 | 34.7792 | 34.8006 | 34.8274 | 34.8044 |  63.151 |

***

### Tuned Parameters

- max_depth = 7
- min_child_weight = 7
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 29
