# Manifold_Segmentation
Manifold segmentation is a method to capture the context information in the image through manifold regularization. The method with Resnet and MolibeNetv2 backbones for Pytorch.

## Manifold_Segmentation's applications
As a regularization item, the model can be widely used in image segmentation and has been proven effective.

# code
## Install dependencies
```
python -m pip install -r requirements.txt
```
This code was testd with python 3.7

## Train
### Visualize training (Optional)
Start visdom sever for visualization. Please remove '--enable_vis' if visualization is not needed. 
```bash
# Run visdom server on port 28333
visdom -port 28333
```
run the train code
```bash
python main.py --model 'model' --dataset 'dataset_name' --data_root 'path' --enable_vis --vis_port 28333 --gpu_id 0 --lr 0.01 --crop_size 'size' --batch_size 16 --download
```
Specify the model architecture with '--model ARCH_NAME'

| DeepLabV3    |  DeepLabV3+        | Head| Doubleattention|
| :---: | :---:     | :---:  |:---:  |
|deeplabv3_resnet50|deeplabv3plus_resnet50|head_resnet50 |doubleattention_resnet50|
|deeplabv3_resnet101|deeplabv3plus_resnet101|head_resnet101 |doubleattention_resnet101|
|deeplabv3_mobilenet|deeplabv3plus_mobilenet |:---:  |:---:  |

### dataset download
#### cityscapes
Select parameters
```
--dataset cityscapes  -- download
```
#### voc
Select parameters
```
--dataset voc --year 2012_aug  -- download
```

### Continue Traing
Run main.py with '--continue_training' to restore the state_dict of optimizer and scheduler from YOUR_CKPT.
```
--ckpt YOUR_CKPT --continue_training
```

## Test
Results will be saved at ./results.
```
python main.py --model 'model' --enable_vis --vis_port 28333 --gpu_idch 0 --lr 0.01 --crop_size 'size' --batch_size 16  --ckpt 'model path,such as ./checkpoints/***.pth ' --test_only --save_val_results
```
