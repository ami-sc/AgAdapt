# MIH1 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 29.8197
- Baseline Model [RMSE] = 36.9323
- Trained Model [MAE] = 21.7770
- Trained Model [RMSE] = 28.6043
- Prediction [MAE] = 28.4862
- Prediction [RMSE] = 36.7053
***

### Dataset Statistics

- LOFO Field [Mean] = 159.4301 [bu/A]
- LOFO Field [Standard Deviation] = 36.6790 [bu/A]
- Model Dataset [Mean] = 154.6894 [bu/A]
- Model Dataset [Standard Deviation] = 45.8697 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 24.5828 | 24.5828 | 24.5828 | 24.5828 | 24.5828 | 24.5828 | 24.5828 | 24.5828 | 24.5638 | 24.4361 | 24.4943 | 24.5223 | 24.5083 | 24.5061 | 24.5061 | 24.5065 | 24.4876 | 24.4876 | 24.4876 | 24.4971 | 24.5135 |
|          2 | 22.9218 | 22.9218 | 22.8684 | 23.0686 | 23.0275 | 23.0496 | 23.0752 | 22.9666 | 23.1314 | 23.2002 | 23.0407 | 22.9412 | 23.0724 | 23.0149 | 22.8993 | 23.0857 | 22.8578 | 23.0055 | 22.9962 | 23.0005 | 23.185  |
|          3 | 22.9266 | 22.9266 | 22.7944 | 22.7426 | 22.8888 | 22.9202 | 22.8718 | 22.7568 | 22.7645 | 22.8038 | 22.8256 | 22.7594 | 22.7665 | 23.0543 | 22.8335 | 22.9493 | 22.9258 | 22.8297 | 22.89   | 22.9233 | 22.7698 |
|          4 | 22.7144 | 22.7144 | 22.8543 | 22.8263 | 22.7507 | 22.7586 | 22.8079 | 22.8022 | 22.7715 | 22.7345 | 22.7216 | 22.6837 | 22.7014 | 22.7486 | 22.7413 | 22.771  | 22.6921 | 22.6956 | 22.7265 | 22.6504 | 22.7316 |
|          5 | 22.8594 | 22.8594 | 22.8247 | 22.8087 | 22.6763 | 22.6654 | 22.857  | 22.7084 | 22.8425 | 22.7893 | 22.7516 | 22.7103 | 22.8045 | 22.7455 | 22.8263 | 22.7273 | 22.8106 | 22.7572 | 22.6609 | 22.6274 | 22.7613 |
|          6 | 22.892  | 22.892  | 22.9135 | 22.7959 | 22.7034 | 22.7267 | 22.8371 | 22.6636 | 22.7692 | 22.7433 | 22.7442 | 22.7267 | 22.7242 | 22.7351 | 22.7092 | 22.7259 | 22.7053 | 22.6111 | 22.7477 | 22.6713 | 22.7368 |
|          7 | 22.8219 | 22.8219 | 22.8255 | 22.8624 | 22.711  | 22.7112 | 22.8562 | 22.777  | 22.8119 | 22.9015 | 22.8836 | 22.7668 | 22.793  | 22.8272 | 22.7676 | 22.7687 | 22.7782 | 22.7266 | 22.7717 | 22.8055 | 22.8747 |
|          8 | 22.9423 | 22.9423 | 22.9057 | 22.8862 | 22.8983 | 22.821  | 22.7826 | 22.8707 | 22.8394 | 22.7476 | 22.9234 | 22.7811 | 22.8684 | 22.7544 | 22.8222 | 22.8032 | 22.7895 | 22.7952 | 22.7161 | 22.8662 | 22.7698 |
|          9 | 23.0369 | 23.0369 | 23.0298 | 22.9846 | 23.007  | 22.8536 | 22.8987 | 23.0107 | 22.9993 | 22.9027 | 22.8678 | 22.8334 | 22.9247 | 22.8216 | 22.7629 | 22.8868 | 22.8659 | 22.7699 | 22.8697 | 22.7996 | 22.8453 |
|         10 | 23.3255 | 23.3255 | 23.1447 | 23.0827 | 23.0405 | 23.0118 | 23.0052 | 23.0537 | 22.9781 | 22.8936 | 22.9174 | 22.8756 | 22.8175 | 22.8109 | 22.8682 | 22.9396 | 22.8105 | 22.7812 | 22.8307 | 22.9295 | 22.8274 |
|         11 | 23.344  | 23.344  | 23.2311 | 23.1673 | 23.211  | 23.0309 | 23.0323 | 22.9265 | 22.9804 | 23.0334 | 22.8396 | 22.8325 | 22.834  | 22.858  | 22.8961 | 22.8101 | 22.8777 | 22.8694 | 22.9027 | 22.8041 | 22.8536 |
|         12 | 23.4809 | 23.4809 | 23.3277 | 23.1722 | 23.1644 | 23.1689 | 23.1022 | 23.1816 | 22.9766 | 22.9517 | 22.9099 | 23.0574 | 22.8892 | 22.9384 | 22.9783 | 22.8447 | 22.8931 | 22.8753 | 22.8234 | 22.9061 | 22.8509 |
|         13 | 23.6508 | 23.6508 | 23.4109 | 23.2208 | 23.1722 | 23.18   | 23.1232 | 23.1457 | 23.0543 | 23.0413 | 22.9274 | 23.0656 | 23.0149 | 22.9579 | 23.0425 | 22.9495 | 22.9514 | 22.9271 | 22.8809 | 22.8161 | 22.8432 |
|         14 | 23.7269 | 23.7269 | 23.5419 | 23.44   | 23.3194 | 23.2521 | 23.1719 | 23.1746 | 23.0584 | 23.0945 | 22.9961 | 22.9991 | 22.8759 | 22.9168 | 22.9432 | 22.8643 | 23.02   | 22.8955 | 22.8796 | 22.9571 | 22.9675 |
|         15 | 23.7978 | 23.7978 | 23.5232 | 23.5368 | 23.2509 | 23.3038 | 23.2275 | 23.3266 | 23.181  | 23.1237 | 23.0095 | 22.9944 | 22.9391 | 22.9683 | 22.9029 | 22.8797 | 22.9416 | 22.9055 | 22.8579 | 22.9385 | 22.9377 |
|         16 | 23.9833 | 23.9833 | 23.6608 | 23.6602 | 23.3823 | 23.3216 | 23.1183 | 23.1984 | 23.0459 | 23.1691 | 22.9784 | 22.9717 | 23.038  | 22.9531 | 23.0171 | 22.8235 | 22.9609 | 22.9703 | 22.9358 | 22.9315 | 22.9283 |
|         17 | 24.1478 | 24.1478 | 23.7889 | 23.6417 | 23.38   | 23.3519 | 23.2828 | 23.3393 | 23.1404 | 23.2197 | 23.0677 | 23.0532 | 22.991  | 22.9051 | 23.0627 | 22.9801 | 22.8227 | 22.9415 | 22.9748 | 22.9538 | 22.7952 |
|         18 | 24.1895 | 24.1895 | 23.8946 | 23.6825 | 23.4433 | 23.4247 | 23.3033 | 23.2773 | 23.1732 | 23.1029 | 23.1089 | 22.9884 | 23.0102 | 22.92   | 23.0252 | 22.8895 | 22.9531 | 22.9232 | 22.9156 | 22.9198 | 22.9083 |
|         19 | 24.3073 | 24.3073 | 23.8963 | 23.6441 | 23.4434 | 23.378  | 23.2912 | 23.2584 | 23.221  | 23.205  | 22.961  | 22.9902 | 23.0101 | 22.8907 | 22.9405 | 22.9528 | 22.9355 | 22.9169 | 22.943  | 22.8504 | 22.9319 |
|         20 | 24.329  | 24.329  | 23.9798 | 23.7435 | 23.4673 | 23.4884 | 23.36   | 23.3334 | 23.2391 | 23.1441 | 22.977  | 22.9655 | 23.0298 | 22.9573 | 22.9441 | 22.9801 | 22.9187 | 22.9112 | 23.0061 | 22.9624 | 22.9496 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 24.991  | 24.9224 | 24.9991 | 24.9964 | 24.8376 | 24.7071 | 24.8817 | 24.87   | 24.8955 | 24.8766 |
|          0.2 | 24.4565 | 24.4847 | 24.3126 | 24.3241 | 24.2223 | 24.1705 | 24.1459 | 24.2429 | 24.0177 | 24.1077 |
|          0.3 | 24.2529 | 24.1905 | 24.1051 | 23.774  | 24.1773 | 23.9492 | 24.0059 | 23.8802 | 24.0409 | 23.9162 |
|          0.4 | 24.1302 | 23.8214 | 23.8349 | 23.7818 | 23.6948 | 23.5696 | 23.6594 | 23.6535 | 23.6053 | 23.5322 |
|          0.5 | 23.6786 | 23.6341 | 23.6326 | 23.4695 | 23.5113 | 23.4648 | 23.4949 | 23.3701 | 23.3542 | 23.5333 |
|          0.6 | 23.5626 | 23.6243 | 23.4409 | 23.373  | 23.3701 | 23.3722 | 23.3027 | 23.426  | 23.2099 | 23.2165 |
|          0.7 | 23.4101 | 23.1799 | 23.4011 | 23.195  | 23.1706 | 23.1331 | 23.1598 | 23.1926 | 23.1392 | 23.2096 |
|          0.8 | 23.2687 | 23.0212 | 23.0321 | 23.0339 | 23.18   | 23.0197 | 23.0007 | 22.9012 | 22.7964 | 23.0123 |
|          0.9 | 23.0644 | 22.9553 | 22.8459 | 22.8651 | 22.9106 | 22.8765 | 22.8043 | 22.7542 | 22.9376 | 22.8829 |
|          1   | 22.8833 | 22.7244 | 22.7842 | 22.9113 | 22.7973 | 22.7621 | 22.6467 | 22.7216 | 22.6891 | 22.6111 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 22.7993 | 22.8067 | 22.6111 | 22.6613 | 22.5881 | 22.8202 | 58.8915 |

***

### Tuned Parameters

- max_depth = 6
- min_child_weight = 17
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.1
- num_boost_round = 248
