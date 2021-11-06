# Pretrained-Pix2Seq
We provide the pre-trained model of Pix2Seq. This version contains new data augmentation. The model is trained for 300 epochs and can acheive 37 mAP without beam search or neucles search. 


## Installation

Install PyTorch 1.5+ and torchvision 0.6+ (recommend torch1.8.1 torchvision 0.8.0)

Install pycocotools (for evaluation on COCO):

```
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

That's it, should be good to train and evaluate detection models.

## Data preparation

Download and extract COCO 2017 train and val images with annotations from
[http://cocodataset.org](http://cocodataset.org/#download).
We expect the directory structure to be the following:
```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## Training

First link coco dataset to the project folder
```
ln -s /path/to/coco ./coco 
```

Training
```
sh train.sh --model pix2seq --output_dir /path/to/save
```

Evaluation
```
sh train.sh --model pix2seq --output_dir /path/to/save --resume /path/to/checkpoints --eval
```

### COCO 

| Method  | backbone | Epoch | Batch Size | AP   | AP50  | AP75  | Weights |
| :-----: | :------: | :----:| :---------:| :---:| :---: | :---: | :-----: |
| Pix2Seq | R50      | 300   | 32         | 37.0 | 53.4 | 39.4 | [weight](https://drive.google.com/file/d/1b7KzqnEBIQCTKmk9SqsXNqX2nlTZSFV_/view?usp=sharing) | 

# Contributor
Qiu Han, Peng Gao, Jingqiu Zhou(Beam Search)

# Acknowledegement
Pix2Seq, DETR 
