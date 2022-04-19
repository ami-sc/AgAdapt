# INH1 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 43.5604
- Baseline Model [RMSE] = 49.6735
- Trained Model [MAE] = 35.5080
- Trained Model [RMSE] = 44.8560
- Prediction [MAE] = 38.7604
- Prediction [RMSE] = 44.8689
***

### Dataset Statistics

- LOFO Field [Mean] = 192.7764 [bu/A]
- LOFO Field [Standard Deviation] = 30.2195 [bu/A]
- Model Dataset [Mean] = 153.3175 [bu/A]
- Model Dataset [Standard Deviation] = 45.3726 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 35.427  | 35.427  | 35.427  | 35.427  | 35.4195 | 35.4195 | 35.4195 | 35.4195 | 35.4195 | 35.4195 | 35.4219 | 35.4219 | 35.4204 | 35.4204 | 35.3228 | 35.3228 | 35.3251 | 35.4185 | 35.3956 | 35.3508 | 35.3508 |
|          2 | 34.8615 | 34.8615 | 34.8615 | 34.8215 | 34.7829 | 34.8149 | 34.9637 | 34.7597 | 34.8876 | 34.7075 | 34.78   | 34.7989 | 34.8787 | 34.9135 | 34.8456 | 34.8969 | 34.9016 | 34.8856 | 34.983  | 34.9967 | 34.9782 |
|          3 | 34.6973 | 34.6973 | 34.6827 | 34.6447 | 34.6638 | 34.7093 | 34.6899 | 34.7151 | 34.656  | 34.6614 | 34.6969 | 34.7467 | 34.7006 | 34.7393 | 34.7666 | 34.7303 | 34.7357 | 34.7747 | 34.848  | 34.7939 | 34.8128 |
|          4 | 34.6865 | 34.6865 | 34.6459 | 34.6756 | 34.6639 | 34.6556 | 34.6554 | 34.6659 | 34.642  | 34.6638 | 34.7034 | 34.72   | 34.7057 | 34.7299 | 34.7029 | 34.7879 | 34.672  | 34.6692 | 34.7744 | 34.757  | 34.7486 |
|          5 | 34.5573 | 34.5573 | 34.5553 | 34.5587 | 34.5721 | 34.5727 | 34.5595 | 34.6005 | 34.6019 | 34.6247 | 34.631  | 34.6462 | 34.6389 | 34.6509 | 34.6803 | 34.7272 | 34.7061 | 34.725  | 34.7397 | 34.7127 | 34.7416 |
|          6 | 34.64   | 34.64   | 34.6282 | 34.6189 | 34.574  | 34.6161 | 34.615  | 34.5546 | 34.5667 | 34.6214 | 34.6174 | 34.6406 | 34.6454 | 34.6333 | 34.6522 | 34.6653 | 34.6782 | 34.6648 | 34.6878 | 34.7094 | 34.7032 |
|          7 | 34.6493 | 34.6493 | 34.6692 | 34.6689 | 34.6563 | 34.6459 | 34.6946 | 34.6667 | 34.6324 | 34.6306 | 34.6913 | 34.6626 | 34.7086 | 34.68   | 34.6866 | 34.6812 | 34.659  | 34.6942 | 34.7082 | 34.6786 | 34.7265 |
|          8 | 34.6991 | 34.6991 | 34.6845 | 34.6984 | 34.6591 | 34.6523 | 34.6796 | 34.6783 | 34.6853 | 34.6428 | 34.615  | 34.6894 | 34.6917 | 34.6893 | 34.6919 | 34.6706 | 34.6633 | 34.7245 | 34.7316 | 34.6966 | 34.7518 |
|          9 | 34.7326 | 34.7326 | 34.7069 | 34.7099 | 34.6759 | 34.6972 | 34.6869 | 34.727  | 34.6803 | 34.6852 | 34.697  | 34.7237 | 34.7437 | 34.734  | 34.7064 | 34.715  | 34.6965 | 34.7805 | 34.7122 | 34.7642 | 34.7325 |
|         10 | 34.7611 | 34.7611 | 34.723  | 34.7292 | 34.7474 | 34.7502 | 34.7715 | 34.7613 | 34.7242 | 34.7451 | 34.7271 | 34.7591 | 34.7676 | 34.7122 | 34.7365 | 34.7474 | 34.7468 | 34.749  | 34.7279 | 34.7357 | 34.7612 |
|         11 | 34.8597 | 34.8597 | 34.8059 | 34.7994 | 34.7765 | 34.8024 | 34.7947 | 34.7843 | 34.7538 | 34.7563 | 34.7631 | 34.7772 | 34.7865 | 34.7811 | 34.7731 | 34.7404 | 34.7551 | 34.7311 | 34.7295 | 34.7456 | 34.7599 |
|         12 | 34.8724 | 34.8723 | 34.8303 | 34.8034 | 34.8095 | 34.8056 | 34.8037 | 34.789  | 34.7866 | 34.7945 | 34.7886 | 34.7663 | 34.787  | 34.7705 | 34.7876 | 34.7478 | 34.7432 | 34.7764 | 34.7541 | 34.7699 | 34.7808 |
|         13 | 34.9026 | 34.9026 | 34.8655 | 34.8738 | 34.8448 | 34.8646 | 34.841  | 34.8131 | 34.79   | 34.7912 | 34.7859 | 34.8126 | 34.8151 | 34.787  | 34.831  | 34.7744 | 34.7695 | 34.8068 | 34.7729 | 34.7991 | 34.8117 |
|         14 | 34.9212 | 34.9212 | 34.8937 | 34.8901 | 34.8659 | 34.8661 | 34.8519 | 34.8101 | 34.8164 | 34.8192 | 34.827  | 34.8392 | 34.8327 | 34.8168 | 34.8068 | 34.8373 | 34.7738 | 34.7758 | 34.8089 | 34.7943 | 34.7711 |
|         15 | 34.9716 | 34.9716 | 34.9271 | 34.9515 | 34.9165 | 34.893  | 34.8944 | 34.8514 | 34.8673 | 34.8383 | 34.8301 | 34.8651 | 34.8536 | 34.8178 | 34.8343 | 34.8273 | 34.7924 | 34.8051 | 34.8019 | 34.7902 | 34.8092 |
|         16 | 34.9859 | 34.9859 | 34.9518 | 34.9495 | 34.9375 | 34.9351 | 34.9021 | 34.8984 | 34.8704 | 34.8404 | 34.8688 | 34.8698 | 34.8784 | 34.8393 | 34.8582 | 34.8446 | 34.8166 | 34.8079 | 34.8224 | 34.816  | 34.8065 |
|         17 | 35.0121 | 35.0121 | 34.9905 | 34.9695 | 34.9624 | 34.9314 | 34.9154 | 34.8921 | 34.8939 | 34.8748 | 34.8821 | 34.8983 | 34.8977 | 34.867  | 34.8665 | 34.8322 | 34.8433 | 34.828  | 34.8543 | 34.8101 | 34.8089 |
|         18 | 35.0392 | 35.0392 | 34.9845 | 34.9956 | 34.9721 | 34.9343 | 34.9245 | 34.9284 | 34.8964 | 34.8997 | 34.8914 | 34.8993 | 34.8841 | 34.8591 | 34.8714 | 34.8638 | 34.8257 | 34.8237 | 34.8213 | 34.8279 | 34.8244 |
|         19 | 35.0579 | 35.0579 | 35.0182 | 34.9831 | 34.9613 | 34.9375 | 34.9576 | 34.9389 | 34.9132 | 34.9211 | 34.9076 | 34.9152 | 34.8968 | 34.8743 | 34.8858 | 34.856  | 34.8573 | 34.8479 | 34.8569 | 34.8456 | 34.8266 |
|         20 | 35.0726 | 35.0726 | 35.0292 | 35.0068 | 34.965  | 34.9458 | 34.9548 | 34.9584 | 34.9254 | 34.8933 | 34.9299 | 34.9319 | 34.9231 | 34.8824 | 34.8977 | 34.8771 | 34.8315 | 34.8481 | 34.85   | 34.829  | 34.8258 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 35.7715 | 35.7715 | 35.7715 | 35.7715 | 35.7715 | 35.7715 | 35.7715 | 35.7715 | 35.7715 | 35.7165 |
|          0.2 | 35.1865 | 35.1865 | 35.1865 | 35.1865 | 35.1865 | 35.1865 | 35.1865 | 35.1865 | 35.1865 | 35.1754 |
|          0.3 | 35.1189 | 35.1189 | 35.1189 | 35.1189 | 35.1189 | 35.1189 | 35.1189 | 35.1189 | 35.1189 | 35.0858 |
|          0.4 | 34.868  | 34.868  | 34.868  | 34.868  | 34.868  | 34.868  | 34.868  | 34.868  | 34.868  | 34.8266 |
|          0.5 | 34.9411 | 34.9411 | 34.9411 | 34.9411 | 34.9411 | 34.9411 | 34.9411 | 34.9411 | 34.9411 | 34.9036 |
|          0.6 | 34.8109 | 34.8109 | 34.8109 | 34.8109 | 34.8109 | 34.8109 | 34.8109 | 34.8109 | 34.8109 | 34.7497 |
|          0.7 | 34.8098 | 34.8098 | 34.8098 | 34.8098 | 34.8098 | 34.8098 | 34.8098 | 34.8098 | 34.8098 | 34.6923 |
|          0.8 | 34.7584 | 34.7585 | 34.7584 | 34.7584 | 34.7584 | 34.7585 | 34.7585 | 34.7584 | 34.7584 | 34.6204 |
|          0.9 | 34.7779 | 34.7779 | 34.7779 | 34.7779 | 34.7779 | 34.7779 | 34.7779 | 34.7779 | 34.7779 | 34.6321 |
|          1   | 34.6985 | 34.6985 | 34.6984 | 34.6985 | 34.6985 | 34.6984 | 34.6984 | 34.6984 | 34.6985 | 34.5546 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |    0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|--------|---------|---------|---------|
| MAE   | 34.5561 | 34.5906 | 34.5546 | 34.611 | 34.6266 | 34.6498 | 62.1142 |

***

### Tuned Parameters

- max_depth = 6
- min_child_weight = 7
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 32