# Error Analysis: resnet18_unfrozen_from_frozen_lr1e-4_128_e8

- Predictions file: `/Users/littleotter/Desktop/I242BProject/result/evaluation/resnet18_unfrozen_from_frozen_lr1e-4_128_e8/test_20260508_104533/predictions.csv`
- Total test samples: `522`
- Misclassified samples: `20`
- Accuracy: `0.9617`
- Mean confidence among errors: `0.7843`
- High-confidence errors (confidence >= 0.80): `12`

## Error Types

| true_label | pred_label | error_type | count |
| --- | --- | --- | --- |
| healthy | unhealthy | healthy -> unhealthy | 10 |
| unhealthy | healthy | unhealthy -> healthy | 10 |

## Class Error Rates

| true_label | samples | errors | error_rate |
| --- | --- | --- | --- |
| healthy | 261 | 10 | 0.0383 |
| unhealthy | 261 | 10 | 0.0383 |

## Most Confident Errors

| path | true_label | pred_label | confidence | true_probability |
| --- | --- | --- | --- | --- |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/unhealthy/img_36d2d90bc1.jpg | unhealthy | healthy | 0.9988 | 0.0012 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/unhealthy/7834c4546809995bb99fb6ec302d3618.jpg | unhealthy | healthy | 0.9810 | 0.0190 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/unhealthy/db9a3ea70f74f82e9c8ef9d712ee8cf3.jpg | unhealthy | healthy | 0.9798 | 0.0202 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/unhealthy/img_915bac2c9f.jpg | unhealthy | healthy | 0.9602 | 0.0398 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/img_51f7253d21.jpg | healthy | unhealthy | 0.9400 | 0.0600 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/671daeb804d4a68ab92c687f2c75268f.jpg | healthy | unhealthy | 0.9393 | 0.0607 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/img_2f73000127.jpg | healthy | unhealthy | 0.9174 | 0.0826 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/bfb2c1a954d2903a51c83dc4b5248d33.jpg | healthy | unhealthy | 0.8739 | 0.1261 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/img_b5bbb5f2e6.jpg | healthy | unhealthy | 0.8513 | 0.1487 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/img_04b3aba842.jpg | healthy | unhealthy | 0.8424 | 0.1576 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/unhealthy/img_c070c1d8d7.jpg | unhealthy | healthy | 0.8102 | 0.1898 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/unhealthy/648ed1cb08ae00ef77e9fd85594dcddf.jpg | unhealthy | healthy | 0.8090 | 0.1910 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/img_879bba14d8.jpg | healthy | unhealthy | 0.7810 | 0.2190 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/img_405e73ffa2.jpg | healthy | unhealthy | 0.6589 | 0.3411 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/unhealthy/524364920cbf99f946977272e96ae72d.jpg | unhealthy | healthy | 0.6403 | 0.3597 |
| /Users/littleotter/Desktop/I242BProject/data/dataset of classification images (3)/dataset of classification images/test/healthy/img_9fb4d407da.jpg | healthy | unhealthy | 0.5954 | 0.4046 |
