# ILH1 - Plant_Height Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 19.6731
- Baseline Model [RMSE] = 23.0735
- Trained Model [MAE] = 26.9915
- Trained Model [RMSE] = 37.0083
- Prediction [MAE] = 16.4903
- Prediction [RMSE] = 19.3292
***

### Dataset Statistics

- LOFO Field [Mean] = 200.5931 [cm]
- LOFO Field [Standard Deviation] = 14.8537 [cm]
- Model Dataset [Mean] = 218.2676 [cm]
- Model Dataset [Standard Deviation] = 40.3922 [cm]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 28.204  | 28.204  | 28.204  | 28.204  | 28.204  | 28.204  | 28.204  | 28.204  | 28.204  | 28.178  | 28.1799 | 28.1841 | 28.2053 | 28.2147 | 28.2109 | 28.1925 | 28.2043 | 28.1864 | 28.2193 | 28.2193 | 28.1994 |
|          2 | 27.8807 | 27.8807 | 27.8807 | 27.808  | 27.8171 | 27.8725 | 27.8124 | 27.8923 | 27.8873 | 27.8656 | 27.9189 | 27.9492 | 27.946  | 27.9445 | 27.9014 | 27.935  | 27.9241 | 27.9235 | 27.8701 | 27.9136 | 27.9098 |
|          3 | 27.792  | 27.792  | 27.792  | 27.8354 | 27.8365 | 27.8498 | 27.8303 | 27.7949 | 27.8027 | 27.8331 | 27.8889 | 27.8486 | 27.9007 | 27.8372 | 27.8675 | 27.8609 | 27.8437 | 27.8326 | 27.8821 | 27.8265 | 27.8511 |
|          4 | 27.8129 | 27.8129 | 27.7725 | 27.7999 | 27.7883 | 27.7897 | 27.8013 | 27.7944 | 27.8288 | 27.8393 | 27.815  | 27.853  | 27.8036 | 27.8059 | 27.8193 | 27.8596 | 27.8399 | 27.7898 | 27.8448 | 27.8279 | 27.7999 |
|          5 | 27.7628 | 27.7628 | 27.7721 | 27.7663 | 27.7553 | 27.7455 | 27.7958 | 27.7881 | 27.8257 | 27.8426 | 27.8024 | 27.8055 | 27.7767 | 27.7879 | 27.8491 | 27.8361 | 27.8147 | 27.8278 | 27.7707 | 27.8425 | 27.8399 |
|          6 | 27.7784 | 27.7784 | 27.7818 | 27.7649 | 27.7606 | 27.7718 | 27.7417 | 27.7218 | 27.8099 | 27.8169 | 27.8025 | 27.7856 | 27.7861 | 27.7727 | 27.7541 | 27.7775 | 27.7666 | 27.7675 | 27.8039 | 27.8071 | 27.8804 |
|          7 | 27.7489 | 27.7489 | 27.7477 | 27.751  | 27.759  | 27.8056 | 27.7837 | 27.7741 | 27.8086 | 27.8016 | 27.8054 | 27.8057 | 27.8073 | 27.7979 | 27.7195 | 27.7806 | 27.792  | 27.7974 | 27.8276 | 27.8332 | 27.8264 |
|          8 | 27.7844 | 27.7844 | 27.8093 | 27.8044 | 27.8076 | 27.8307 | 27.8171 | 27.7948 | 27.8101 | 27.8249 | 27.8023 | 27.8305 | 27.8345 | 27.7991 | 27.8132 | 27.8087 | 27.827  | 27.8218 | 27.8213 | 27.8408 | 27.8369 |
|          9 | 27.898  | 27.898  | 27.9038 | 27.8891 | 27.8825 | 27.8719 | 27.8807 | 27.8384 | 27.8837 | 27.8483 | 27.8557 | 27.8914 | 27.8552 | 27.856  | 27.8465 | 27.8724 | 27.8637 | 27.824  | 27.8873 | 27.834  | 27.8475 |
|         10 | 27.9051 | 27.9051 | 27.9091 | 27.9145 | 27.885  | 27.8724 | 27.9064 | 27.8816 | 27.9188 | 27.9242 | 27.8788 | 27.9148 | 27.8949 | 27.8738 | 27.8735 | 27.8538 | 27.8803 | 27.9119 | 27.8666 | 27.8949 | 27.8783 |
|         11 | 27.9604 | 27.9604 | 27.9756 | 27.9697 | 27.9492 | 27.9522 | 27.9514 | 27.9449 | 27.9384 | 27.941  | 27.9144 | 27.9007 | 27.9388 | 27.9158 | 27.9011 | 27.9143 | 27.8624 | 27.8976 | 27.8924 | 27.9023 | 27.8687 |
|         12 | 28.0083 | 28.0083 | 28.0041 | 27.9892 | 28.0127 | 27.9692 | 27.9737 | 27.9676 | 27.9749 | 27.9727 | 27.9571 | 27.9337 | 27.9373 | 27.9256 | 27.9133 | 27.9278 | 27.9146 | 27.8978 | 27.9085 | 27.9067 | 27.8345 |
|         13 | 28.04   | 28.04   | 28.0429 | 28.0138 | 28.0223 | 28.0313 | 28.0211 | 28.003  | 28.0086 | 28.0158 | 27.9766 | 27.9786 | 27.9685 | 27.9459 | 27.9366 | 27.9441 | 27.9371 | 27.9094 | 27.9177 | 27.9007 | 27.8863 |
|         14 | 28.0701 | 28.0701 | 28.0779 | 28.0623 | 28.0824 | 28.0586 | 28.0289 | 28.0258 | 28.0274 | 28.0209 | 28.0049 | 27.9763 | 27.9816 | 27.9653 | 27.9404 | 27.9515 | 27.9392 | 27.9274 | 27.9013 | 27.9089 | 27.9092 |
|         15 | 28.0834 | 28.0834 | 28.1007 | 28.0867 | 28.0931 | 28.0634 | 28.0478 | 28.0307 | 28.0476 | 28.0425 | 28.0299 | 27.9893 | 27.9888 | 27.9724 | 27.9464 | 27.9661 | 27.9474 | 27.9521 | 27.932  | 27.9203 | 27.9006 |
|         16 | 28.1099 | 28.1099 | 28.1082 | 28.1052 | 28.1174 | 28.0909 | 28.0598 | 28.0766 | 28.0548 | 28.0428 | 28.0433 | 28.0191 | 28.0031 | 27.9672 | 27.9828 | 27.9619 | 27.9458 | 27.9478 | 27.9365 | 27.9471 | 27.9181 |
|         17 | 28.1287 | 28.1287 | 28.1461 | 28.1279 | 28.1117 | 28.0881 | 28.0934 | 28.0683 | 28.0609 | 28.0583 | 28.0463 | 28.0111 | 28.0038 | 27.9777 | 27.9835 | 27.9783 | 27.9789 | 27.9557 | 27.9544 | 27.9403 | 27.9234 |
|         18 | 28.1366 | 28.1366 | 28.1464 | 28.1343 | 28.1225 | 28.1096 | 28.1007 | 28.0941 | 28.0781 | 28.0618 | 28.0571 | 28.0319 | 28.0196 | 27.9968 | 28.004  | 27.9823 | 27.9783 | 27.9623 | 27.9313 | 27.9573 | 27.9258 |
|         19 | 28.1391 | 28.1391 | 28.1484 | 28.1302 | 28.1221 | 28.1057 | 28.1093 | 28.0964 | 28.0895 | 28.072  | 28.055  | 28.0543 | 28.0235 | 27.9963 | 28.0053 | 27.9966 | 27.9791 | 27.9629 | 27.9504 | 27.967  | 27.9497 |
|         20 | 28.1478 | 28.1478 | 28.1495 | 28.1343 | 28.1352 | 28.1186 | 28.1138 | 28.104  | 28.0821 | 28.0755 | 28.0599 | 28.0422 | 28.0302 | 28.0178 | 28.0039 | 27.9926 | 27.9886 | 27.9655 | 27.9538 | 27.9692 | 27.9464 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 28.6415 | 28.6415 | 28.6415 | 28.6415 | 28.6415 | 28.6415 | 28.6415 | 28.6415 | 28.6415 | 28.6417 |
|          0.2 | 28.2126 | 28.2125 | 28.2126 | 28.2126 | 28.2126 | 28.2125 | 28.2125 | 28.2125 | 28.2125 | 28.2144 |
|          0.3 | 28.0769 | 28.0769 | 28.0769 | 28.0769 | 28.0769 | 28.0769 | 28.0769 | 28.0769 | 28.0769 | 28.1497 |
|          0.4 | 28.0342 | 28.0342 | 28.0342 | 28.0342 | 28.0342 | 28.0342 | 28.0342 | 28.0342 | 28.0342 | 27.96   |
|          0.5 | 27.9934 | 27.9934 | 27.9934 | 27.9934 | 27.9934 | 27.9934 | 27.9934 | 27.9934 | 27.9934 | 28.0194 |
|          0.6 | 27.9685 | 27.9685 | 27.9685 | 27.9685 | 27.9685 | 27.9685 | 27.9685 | 27.9685 | 27.9686 | 27.8933 |
|          0.7 | 27.8798 | 27.8798 | 27.8798 | 27.8798 | 27.8798 | 27.8798 | 27.8798 | 27.8798 | 27.8798 | 27.8818 |
|          0.8 | 27.8154 | 27.8154 | 27.8154 | 27.8154 | 27.8154 | 27.8154 | 27.8154 | 27.8154 | 27.8154 | 27.826  |
|          0.9 | 27.8074 | 27.8074 | 27.8074 | 27.8074 | 27.8074 | 27.8074 | 27.8074 | 27.8074 | 27.8074 | 27.8434 |
|          1   | 27.8288 | 27.8288 | 27.8288 | 27.8288 | 27.8288 | 27.8288 | 27.8288 | 27.8288 | 27.8288 | 27.7195 |

***

### eta Grid Search

| eta   |    0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|--------|---------|---------|---------|---------|---------|---------|
| MAE   | 27.859 | 27.7936 | 27.7195 | 27.7602 | 27.7877 | 27.7878 | 83.5145 |

***

### Tuned Parameters

- max_depth = 7
- min_child_weight = 14
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 38