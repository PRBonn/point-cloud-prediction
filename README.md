# Self-supervised Point Cloud Prediction Using 3D Spatio-temporal Convolutional Networks
This is a Pytorch-Lightning implementation of the paper "Self-supervised Point Cloud Prediction Using 3D Spatio-temporal Convolutional Networks".

![](docs/introduction.png)
*Given a sequence of P past point clouds (left in red) at time T, the goal is to predict the F future scans (right in blue).*

## Table of Contents
1. [Publication](#Publication)
2. [Data](#Data)
3. [Installation](#Installation)
4. [Download](#Dwnload)
5. [License](#License)

![](docs/architecture.png)
*Overview of our architecture*

## Publication
If you use our code in your academic work, please cite the corresponding [paper](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/mersch2021corl.pdf):
    
```latex
@inproceedings{mersch2021corl,
  author = {B. Mersch and X. Chen and J. Behley and C. Stachniss},
  title = {{Self-supervised Point Cloud Prediction Using 3D Spatio-temporal Convolutional Networks}},
  booktitle = {Proc.~of the Conf.~on Robot Learning (CoRL)},
  year = {2021},
}
```

## Data
Download the Kitti Odometry data from the official [website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php).

## Installation

### Source Code
Clone this repository and run 
```bash
cd point-cloud-prediction
git submodule update --init
```
to install the Chamfer distance submodule. The Chamfer distance submodule is originally taken from [here](https://github.com/chrdiller/pyTorchChamferDistance) with some modifications to use it as a submodule. All parameters are stored in ```config/parameters.yaml```.

### Dependencies
In this project, we use CUDA 10.2. All other dependencies are managed with Python Poetry and can be found in the ```poetry.lock``` file. If you want to use Python Poetry (recommended), install it with:
```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
```

Install Python dependencies with Python Poetry
```bash
poetry install
```

and activate the virtual environment in the shell with
```bash
poetry shell
```

### Export Environment Variables to dataset
We process the data in advance to speed up training. The preprocessing is automatically done if ```GENERATE_FILES``` is set to true in ```config/parameters.yaml```. The environment variable ```PCF_DATA_RAW``` points to the directory containing the train/val/test sequences specified in the config file. It can be set with

```bash
export PCF_DATA_RAW=/path/to/kitti-odometry/dataset/sequences
```

and the destination of the processed files ```PCF_DATA_PROCESSED``` is set with

```bash
export PCF_DATA_PROCESSED=/desired/path/to/processed/data/
```

## Training
*Note* If you have not pre-processed the data yet, you need to set ```GENERATE_FILES: True``` in ```config/parameters.yaml```. After that, you can set ```GENERATE_FILES: False``` to skip this step.

The training script can be run by
```bash
python pcf/train.py
```
using the parameters defined in ```config/parameters.yaml```. Pass the flag ```--help``` if you want to see more options like resuming from a checkpoint or initializing the weights from a pre-trained model. A directory will be created in ```pcf/runs``` which makes it easier to discriminate between different runs and to avoid overwriting existing logs. The script saves everything like the used config, logs and checkpoints into a path ```pcf/runs/COMMIT/EXPERIMENT_DATE_TIME``` consisting of the current git commit ID (this allows you to checkout at the last git commit used for training), the specified experiment ID (```pcf``` by default) and the date and time.

*Example:*
```pcf/runs/7f1f6d4/pcf_20211106_140014```

```7f1f6d4```: Git commit ID

```pcf_20211106_140014```: Experiment ID, date and time

## Testing
Test your model by running
```bash
python pcf/test.py -m COMMIT/EXPERIMENT_DATE_TIME
```
where ```COMMIT/EXPERIMENT_DATE_TIME``` is the relative path to your model in ```pcf/runs```. *Note*: Use the flag ```-s``` if you want to save the predicted point clouds for visualiztion and ```-l``` if you want to test the model on a smaller amount of data.

*Example*
```bash
python pcf/test.py -m 7f1f6d4/pcf_20211106_140014
```
or 
```bash
python pcf/test.py -m 7f1f6d4/pcf_20211106_140014 -l 5 -s
```
if you want to test the model on 5 batches and save the resulting point clouds.

## Visualization
After passing the ```-s``` flag to the testing script, the predicted range images will be saved as .svg files in ```/pcf/runs/COMMIT/EXPERIMENT_DATE_TIME/range_view_predictions```. The predicted point clouds are saved to ```/pcf/runs/COMMIT/EXPERIMENT_DATE_TIME/test/point_clouds```. You can visualize them by running
```bash
python pcf/visualize.py -p /pcf/runs/COMMIT/EXPERIMENT_DATE_TIME/test/point_clouds
```

![](docs/predictions.gif)
*Five past and five future ground truth and our five predicted future range images.*

![](docs/qualitative.png)
*Last received point cloud at time T and the predicted next 5 future point clouds. Ground truth points
are shown in red and predicted points in blue.*

## Download
You can download our best performing model from the paper [here](https://www.ipb.uni-bonn.de/html/projects/point-cloud-prediction/pretrained.zip). Just extract the zip file into ```pcf/runs```.

## License
This project is free software made available under the MIT License. For details see the LICENSE file.
