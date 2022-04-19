# COH1 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 53.9303
- Baseline Model [RMSE] = 59.3609
- Trained Model [MAE] = 32.7770
- Trained Model [RMSE] = 43.0443
- Prediction [MAE] = 59.5373
- Prediction [RMSE] = 64.8155
***

### Dataset Statistics

- LOFO Field [Mean] = 106.8251 [bu/A]
- LOFO Field [Standard Deviation] = 26.1265 [bu/A]
- Model Dataset [Mean] = 160.1351 [bu/A]
- Model Dataset [Standard Deviation] = 44.0983 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 33.1138 | 33.1138 | 33.1138 | 33.1138 | 33.1138 | 33.1138 | 33.1138 | 33.1138 | 32.9761 | 32.9725 | 32.9841 | 33.0381 | 33.0381 | 33.0381 | 33.0393 | 33.0393 | 33.0393 | 33.0393 | 33.1066 | 33.1428 | 33.0112 |
|          2 | 31.9762 | 31.9762 | 32.0203 | 32.0032 | 31.9301 | 31.9071 | 32.1648 | 32.1648 | 31.8592 | 31.906  | 32.0256 | 32.0867 | 32.1095 | 32.0576 | 32.0711 | 32.1574 | 31.97   | 32.0285 | 32.0268 | 32.1002 | 32.0043 |
|          3 | 31.8241 | 31.8241 | 31.843  | 31.7828 | 31.835  | 31.8074 | 31.8109 | 31.799  | 31.7878 | 31.8484 | 31.9068 | 31.8023 | 31.8002 | 31.8195 | 31.9081 | 31.8494 | 31.8498 | 31.873  | 31.8357 | 31.8203 | 31.8184 |
|          4 | 31.8293 | 31.8293 | 31.904  | 31.828  | 31.7726 | 31.7562 | 31.7752 | 31.7886 | 31.8152 | 31.7905 | 31.7763 | 31.7837 | 31.8173 | 31.847  | 31.7486 | 31.7287 | 31.8027 | 31.7701 | 31.7656 | 31.7658 | 31.8106 |
|          5 | 31.7817 | 31.7817 | 31.7922 | 31.7547 | 31.8137 | 31.7674 | 31.7654 | 31.7951 | 31.7391 | 31.7364 | 31.7886 | 31.7807 | 31.7248 | 31.759  | 31.7462 | 31.7308 | 31.7515 | 31.7431 | 31.7468 | 31.7719 | 31.7311 |
|          6 | 31.773  | 31.773  | 31.7479 | 31.7334 | 31.7696 | 31.7246 | 31.7376 | 31.7812 | 31.7209 | 31.7759 | 31.7565 | 31.7638 | 31.7317 | 31.7468 | 31.72   | 31.7549 | 31.7176 | 31.7345 | 31.7134 | 31.742  | 31.7246 |
|          7 | 31.7838 | 31.7838 | 31.7827 | 31.7974 | 31.7639 | 31.7243 | 31.706  | 31.6924 | 31.7413 | 31.7222 | 31.7658 | 31.729  | 31.6933 | 31.7043 | 31.7052 | 31.7191 | 31.6803 | 31.7032 | 31.7311 | 31.7262 | 31.7341 |
|          8 | 31.8003 | 31.8003 | 31.7823 | 31.7747 | 31.7437 | 31.7527 | 31.7283 | 31.7297 | 31.7162 | 31.7096 | 31.7678 | 31.7212 | 31.7417 | 31.7088 | 31.7359 | 31.6989 | 31.6813 | 31.7097 | 31.7258 | 31.6921 | 31.6928 |
|          9 | 31.8035 | 31.8035 | 31.7799 | 31.7728 | 31.7583 | 31.7611 | 31.767  | 31.7506 | 31.7443 | 31.7424 | 31.7707 | 31.7722 | 31.732  | 31.6849 | 31.7383 | 31.7015 | 31.7145 | 31.6896 | 31.6769 | 31.6596 | 31.6843 |
|         10 | 31.8405 | 31.8405 | 31.8346 | 31.803  | 31.7658 | 31.7636 | 31.7587 | 31.7594 | 31.758  | 31.7583 | 31.7405 | 31.764  | 31.7027 | 31.7254 | 31.7272 | 31.7057 | 31.7024 | 31.6703 | 31.7198 | 31.6892 | 31.7365 |
|         11 | 31.8402 | 31.8402 | 31.8462 | 31.8134 | 31.8079 | 31.7977 | 31.7582 | 31.7603 | 31.7645 | 31.7249 | 31.7598 | 31.7649 | 31.7739 | 31.7342 | 31.7101 | 31.6884 | 31.7061 | 31.7098 | 31.6817 | 31.7083 | 31.6764 |
|         12 | 31.8826 | 31.8826 | 31.8216 | 31.8343 | 31.8088 | 31.8187 | 31.8005 | 31.7708 | 31.7525 | 31.7792 | 31.7764 | 31.7719 | 31.7453 | 31.7337 | 31.7138 | 31.7334 | 31.6769 | 31.6928 | 31.7054 | 31.7078 | 31.7152 |
|         13 | 31.9187 | 31.9187 | 31.9099 | 31.8736 | 31.8208 | 31.8142 | 31.7888 | 31.7472 | 31.8095 | 31.7751 | 31.8018 | 31.792  | 31.7578 | 31.7459 | 31.742  | 31.7218 | 31.7403 | 31.7239 | 31.7244 | 31.7106 | 31.6977 |
|         14 | 31.9296 | 31.9296 | 31.8789 | 31.8712 | 31.8287 | 31.8215 | 31.8296 | 31.7846 | 31.7472 | 31.7923 | 31.8064 | 31.8043 | 31.7759 | 31.7659 | 31.7429 | 31.7383 | 31.7112 | 31.7073 | 31.7396 | 31.7278 | 31.7203 |
|         15 | 31.9599 | 31.9599 | 31.9156 | 31.8949 | 31.8602 | 31.8806 | 31.8466 | 31.8501 | 31.8289 | 31.8251 | 31.8093 | 31.8187 | 31.7793 | 31.7939 | 31.7713 | 31.7672 | 31.7342 | 31.7447 | 31.7197 | 31.7193 | 31.7221 |
|         16 | 31.9855 | 31.9855 | 31.9477 | 31.9043 | 31.8839 | 31.8836 | 31.8996 | 31.8315 | 31.8489 | 31.8391 | 31.8327 | 31.8486 | 31.8043 | 31.8014 | 31.7648 | 31.7653 | 31.7535 | 31.7385 | 31.7146 | 31.7474 | 31.7218 |
|         17 | 32.0088 | 32.0088 | 31.9929 | 31.9387 | 31.9055 | 31.9099 | 31.8928 | 31.8678 | 31.867  | 31.8542 | 31.8336 | 31.8677 | 31.8073 | 31.7877 | 31.7906 | 31.7678 | 31.7683 | 31.7362 | 31.7517 | 31.7279 | 31.7329 |
|         18 | 32.0256 | 32.0256 | 32.0051 | 31.9732 | 31.9458 | 31.9269 | 31.8888 | 31.8767 | 31.8656 | 31.8686 | 31.8541 | 31.8742 | 31.8181 | 31.7951 | 31.808  | 31.7866 | 31.7628 | 31.767  | 31.7588 | 31.7538 | 31.7412 |
|         19 | 32.0549 | 32.0549 | 32.0229 | 31.9919 | 31.9513 | 31.9349 | 31.923  | 31.899  | 31.8776 | 31.8693 | 31.8524 | 31.8642 | 31.8171 | 31.8198 | 31.8163 | 31.7867 | 31.7539 | 31.7579 | 31.7589 | 31.7589 | 31.7464 |
|         20 | 32.0848 | 32.0848 | 32.0446 | 31.9991 | 31.9563 | 31.9474 | 31.9368 | 31.8985 | 31.8941 | 31.8849 | 31.8684 | 31.8791 | 31.8258 | 31.809  | 31.8127 | 31.8239 | 31.7922 | 31.7615 | 31.7701 | 31.7542 | 31.7132 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 32.8957 | 32.8957 | 32.8957 | 32.8957 | 32.8957 | 32.8957 | 32.8957 | 32.8957 | 32.8957 | 32.6289 |
|          0.2 | 32.2809 | 32.2809 | 32.2809 | 32.2809 | 32.2809 | 32.2809 | 32.2809 | 32.2809 | 32.2809 | 32.1038 |
|          0.3 | 32.0385 | 32.0385 | 32.0385 | 32.0385 | 32.0385 | 32.0385 | 32.0385 | 32.0385 | 32.0385 | 31.8526 |
|          0.4 | 31.8689 | 31.8689 | 31.8689 | 31.8689 | 31.8689 | 31.8689 | 31.8689 | 31.8689 | 31.8689 | 31.8983 |
|          0.5 | 31.8288 | 31.8288 | 31.8288 | 31.8288 | 31.8288 | 31.8288 | 31.8288 | 31.8288 | 31.8288 | 31.8259 |
|          0.6 | 31.854  | 31.854  | 31.854  | 31.854  | 31.854  | 31.854  | 31.854  | 31.854  | 31.854  | 31.8419 |
|          0.7 | 31.886  | 31.886  | 31.886  | 31.886  | 31.886  | 31.886  | 31.886  | 31.886  | 31.886  | 31.7903 |
|          0.8 | 31.8513 | 31.8513 | 31.8513 | 31.8513 | 31.8513 | 31.8513 | 31.8513 | 31.8513 | 31.8513 | 31.7666 |
|          0.9 | 31.8234 | 31.8234 | 31.8234 | 31.8234 | 31.8234 | 31.8234 | 31.8234 | 31.8234 | 31.8234 | 31.6632 |
|          1   | 31.8152 | 31.8152 | 31.8152 | 31.8152 | 31.8152 | 31.8152 | 31.8152 | 31.8152 | 31.8152 | 31.6596 |

***

### eta Grid Search

| eta   |    0.5 |    0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|--------|--------|---------|---------|---------|---------|---------|
| MAE   | 31.766 | 31.751 | 31.6596 | 31.7115 | 31.6611 | 31.6939 | 64.4604 |

***

### Tuned Parameters

- max_depth = 9
- min_child_weight = 19
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 32