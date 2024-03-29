# WIH1 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 46.9501
- Baseline Model [RMSE] = 54.7695
- Trained Model [MAE] = 21.5973
- Trained Model [RMSE] = 28.1729
- Prediction [MAE] = 28.1003
- Prediction [RMSE] = 35.0374
***

### Dataset Statistics

- LOFO Field [Mean] = 190.7239 [bu/A]
- LOFO Field [Standard Deviation] = 39.1190 [bu/A]
- Model Dataset [Mean] = 152.3548 [bu/A]
- Model Dataset [Standard Deviation] = 44.8749 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 23.9222 | 23.9222 | 23.9222 | 23.9269 | 24.0778 | 24.0778 | 24.0778 | 24.0778 | 24.0778 | 24.0787 | 23.9991 | 23.9282 | 23.9051 | 23.9051 | 23.9051 | 23.9051 | 23.9051 | 23.9051 | 23.9051 | 23.9039 | 23.9039 |
|          2 | 23.1445 | 23.1445 | 22.8775 | 22.9024 | 22.9565 | 23.038  | 23.0771 | 23.2228 | 22.7773 | 22.8523 | 22.9045 | 22.9794 | 23.0803 | 23.0577 | 22.9394 | 23.0588 | 23.0786 | 23.1708 | 23.0908 | 23.1137 | 23.1163 |
|          3 | 22.9706 | 22.9706 | 22.6686 | 22.5971 | 22.6733 | 22.6808 | 22.7667 | 22.7901 | 22.6211 | 22.7468 | 22.7681 | 22.6475 | 22.7057 | 22.8829 | 22.7836 | 22.7672 | 22.7502 | 22.6924 | 22.6391 | 22.7763 | 22.969  |
|          4 | 22.6013 | 22.6013 | 22.6301 | 22.5993 | 22.6117 | 22.6141 | 22.6124 | 22.5244 | 22.5925 | 22.5161 | 22.5373 | 22.559  | 22.5113 | 22.6273 | 22.5942 | 22.4839 | 22.5608 | 22.6191 | 22.4894 | 22.5687 | 22.5659 |
|          5 | 22.6887 | 22.6887 | 22.6921 | 22.4624 | 22.4359 | 22.612  | 22.5266 | 22.5714 | 22.5835 | 22.5868 | 22.5857 | 22.6134 | 22.5474 | 22.6094 | 22.4997 | 22.6064 | 22.5647 | 22.6175 | 22.5628 | 22.6723 | 22.7058 |
|          6 | 22.5572 | 22.5572 | 22.555  | 22.6074 | 22.6439 | 22.5068 | 22.5188 | 22.5022 | 22.5601 | 22.5454 | 22.5997 | 22.5323 | 22.6011 | 22.5528 | 22.5708 | 22.5056 | 22.5677 | 22.526  | 22.5031 | 22.5908 | 22.5238 |
|          7 | 22.6605 | 22.6605 | 22.6803 | 22.6349 | 22.5747 | 22.652  | 22.5984 | 22.6165 | 22.63   | 22.5885 | 22.6134 | 22.5993 | 22.5726 | 22.6541 | 22.742  | 22.6943 | 22.6319 | 22.6296 | 22.7192 | 22.5915 | 22.6049 |
|          8 | 22.6665 | 22.6665 | 22.7348 | 22.6713 | 22.6122 | 22.6585 | 22.6602 | 22.7107 | 22.7266 | 22.727  | 22.6338 | 22.5831 | 22.8108 | 22.5766 | 22.6515 | 22.6311 | 22.6416 | 22.6693 | 22.6348 | 22.6084 | 22.6023 |
|          9 | 22.8402 | 22.8402 | 22.7956 | 22.7898 | 22.636  | 22.7024 | 22.6761 | 22.6601 | 22.623  | 22.6181 | 22.7071 | 22.6286 | 22.658  | 22.6707 | 22.7064 | 22.6647 | 22.6893 | 22.6963 | 22.5085 | 22.6871 | 22.6715 |
|         10 | 22.9782 | 22.9782 | 22.8412 | 22.6828 | 22.7057 | 22.7036 | 22.7913 | 22.8068 | 22.5758 | 22.7966 | 22.7081 | 22.8061 | 22.6581 | 22.7301 | 22.6703 | 22.6599 | 22.6965 | 22.607  | 22.7639 | 22.6665 | 22.6548 |
|         11 | 23.0699 | 23.0699 | 22.9658 | 22.848  | 22.8596 | 22.9255 | 22.8637 | 22.786  | 22.8465 | 22.8596 | 22.7642 | 22.7758 | 22.7395 | 22.7737 | 22.6427 | 22.7964 | 22.6775 | 22.7996 | 22.7687 | 22.6726 | 22.7125 |
|         12 | 23.294  | 23.294  | 23.1202 | 22.9688 | 22.897  | 22.9199 | 22.8636 | 22.8343 | 22.8225 | 22.7891 | 22.809  | 22.816  | 22.8133 | 22.6809 | 22.774  | 22.712  | 22.7732 | 22.6874 | 22.6677 | 22.7043 | 22.7234 |
|         13 | 23.4065 | 23.4065 | 23.2594 | 23.0342 | 22.9032 | 23.0368 | 22.9237 | 22.8888 | 22.8002 | 22.7882 | 22.8808 | 22.8271 | 22.8398 | 22.8366 | 22.725  | 22.774  | 22.7303 | 22.7745 | 22.7183 | 22.7697 | 22.7291 |
|         14 | 23.6091 | 23.6091 | 23.297  | 23.2077 | 23.0818 | 23.0201 | 22.9456 | 22.9181 | 22.8828 | 22.8574 | 22.7871 | 22.7474 | 22.7941 | 22.8561 | 22.7257 | 22.7259 | 22.7728 | 22.7596 | 22.7629 | 22.7313 | 22.792  |
|         15 | 23.6333 | 23.6333 | 23.6217 | 23.2646 | 23.1553 | 23.1547 | 23.0909 | 22.9934 | 22.9538 | 22.9311 | 22.8989 | 22.8434 | 22.8951 | 22.789  | 22.7339 | 22.7868 | 22.7606 | 22.7537 | 22.7817 | 22.7149 | 22.7899 |
|         16 | 23.762  | 23.762  | 23.5165 | 23.4191 | 23.2181 | 23.2242 | 23.0662 | 22.9114 | 22.9552 | 22.987  | 22.9013 | 22.772  | 22.8652 | 22.9045 | 22.7467 | 22.7777 | 22.841  | 22.7168 | 22.7259 | 22.7312 | 22.7947 |
|         17 | 23.9593 | 23.9593 | 23.7021 | 23.3537 | 23.3038 | 23.1802 | 23.1181 | 23.1613 | 22.98   | 22.9736 | 22.8793 | 22.8585 | 22.8714 | 22.8052 | 22.8119 | 22.7236 | 22.7731 | 22.8004 | 22.8275 | 22.704  | 22.7688 |
|         18 | 24.0515 | 24.0515 | 23.7897 | 23.4822 | 23.3436 | 23.141  | 23.1643 | 22.9934 | 22.9364 | 22.935  | 22.8437 | 22.9155 | 22.8331 | 22.8509 | 22.7285 | 22.7772 | 22.8272 | 22.7228 | 22.7985 | 22.769  | 22.7917 |
|         19 | 24.0756 | 24.0756 | 23.8186 | 23.471  | 23.4033 | 23.2609 | 23.1887 | 23.0059 | 23.0368 | 23.0181 | 22.8284 | 22.8362 | 22.8379 | 22.7572 | 22.7265 | 22.8039 | 22.7912 | 22.7971 | 22.7912 | 22.6639 | 22.9011 |
|         20 | 24.2073 | 24.2073 | 23.893  | 23.5617 | 23.4182 | 23.3135 | 23.2051 | 23.0868 | 23.0587 | 22.9398 | 22.8812 | 22.9104 | 22.8519 | 22.851  | 22.7494 | 22.7503 | 22.7997 | 22.8319 | 22.8095 | 22.7126 | 22.8371 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.0866 | 24.967  | 24.6842 | 24.6221 | 24.8608 | 24.7489 | 24.801  | 24.6723 | 24.6296 | 24.8155 |
|          0.2 | 24.4717 | 24.4211 | 24.1305 | 24.2625 | 24.0981 | 23.9876 | 23.9849 | 24.2013 | 24.0759 | 24.1004 |
|          0.3 | 23.993  | 23.8012 | 23.8602 | 23.6783 | 23.6483 | 23.8338 | 23.6651 | 23.59   | 23.7846 | 23.4787 |
|          0.4 | 23.637  | 23.591  | 23.4715 | 23.4894 | 23.4562 | 23.3953 | 23.5031 | 23.4764 | 23.3214 | 23.2383 |
|          0.5 | 23.7593 | 23.465  | 23.2492 | 23.1206 | 23.2983 | 23.1447 | 23.1353 | 23.1843 | 23.1122 | 23.0402 |
|          0.6 | 23.6432 | 23.2858 | 23.0166 | 23.1476 | 23.0229 | 22.9558 | 22.97   | 22.9973 | 22.8418 | 22.8303 |
|          0.7 | 23.2573 | 23.0098 | 22.966  | 22.9037 | 23.0351 | 22.9295 | 22.773  | 22.8434 | 22.7349 | 22.7707 |
|          0.8 | 22.8885 | 23.1988 | 22.7682 | 22.9577 | 22.7088 | 22.7146 | 22.7012 | 22.6995 | 22.6998 | 22.717  |
|          0.9 | 22.7112 | 22.79   | 22.6642 | 22.7056 | 22.6066 | 22.651  | 22.7427 | 22.6145 | 22.6649 | 22.5362 |
|          1   | 22.8498 | 22.8085 | 22.5687 | 22.6559 | 22.592  | 22.664  | 22.6551 | 22.6456 | 22.6734 | 22.4359 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 22.7549 | 22.7314 | 22.4359 | 22.4501 | 22.4399 | 22.7303 | 58.0125 |

***

### Tuned Parameters

- max_depth = 5
- min_child_weight = 4
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 76
