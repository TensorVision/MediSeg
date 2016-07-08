# MediSeg
Models for Medical Segmentation


## Setup

1. [Optional] configure DATA folder in "config.py"
2. Download "Segmentation_Rigid_Training.zip" at [http://grand-challenge.org/site/EndoVisSub-Instrument/](http://grand-challenge.org/site/EndoVisSub-Instrument/)
3. Copy "Segmentation_Rigid_Training.zip" into the DATA folder and run `python DATA/create_filelist.py -d DATA` and `python DATA/create_filelist.py -d DATA --txt`.



## Usage

Install [TensorVision](https://github.com/TensorVision/TensorVision)

```
$ tv-train --hypes hypes/medseg.json
```
