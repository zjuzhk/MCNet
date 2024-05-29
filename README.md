# MCNet
Official implementation of MCNet: Rethinking the Core Ingredients for Accurate and Efficient Homography Estimation.


## Environment

torch1.10.0

torchvision 0.11.1

cupy 10.6

numpy 1.23.0

opencv-python 4.8.0.76

## Evaluate

Please modify the dataset path in datasets.py and run the following code with situable variables:
```bash
python test.py --gpuid ${GPU_ID} --dataset ${DATASET} --checkpoints ${WEIGHT_PATH}
```

## License

This project is released under the Apache 2.0 license.

## Contact
- hkzhu.zju@gmail.com

- cao_siyuan@zju.edu.cn