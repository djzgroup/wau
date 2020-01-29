
## CIFAR-10

#### AlexNet
* WAU
```
python cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet --optimizer SGD --wd 1e-3
```
  
#### ResNet-50
* WAU
```
python cifar.py -a resnet --depth 50 --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/resnet-50 --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a resnet --depth 50 --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/resnet-50 --optimizer SGD --wd 1e-3
```

#### VGG19 (BN)
* WAU
```
python cifar.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg19_bn --optimizer SGD --wd 1e-3
```

#### PreResNet-110
* WAU
```
python cifar.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/preresnet-110 --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/preresnet-110 --optimizer SGD --wd 1e-3
```

#### ResNeXt-29, 8x64d
* WAU
```
python cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d --optimizer SGD --wd 1e-3
```

#### WRN-28-10-drop
* WAU
```
python cifar.py -a wrn --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-28-10-drop --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a wrn --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-28-10-drop --optimizer SGD --wd 1e-3
```

#### DenseNet-BC (L=40, k=12) 
* WAU
```
python cifar.py -a densenet --depth 40 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-L40-k12 --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a densenet --depth 40 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-L40-k12 --optimizer SGD --wd 1e-3
```

#### DenseNet-BC (L=100, k=12)
* WAU
```
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12 --optimizer WSGD --wd 1e-3
```
* Baseline
```
python cifar.py -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12 --optimizer SGD --wd 1e-3
```
