# Automatic assessment of LNEN tumor proliferative active with Pathonet:
Supervised deep learning network dedicated to the detection of Ki-67 or PHH3 positive cells on immunostained whole slide images (WSI). This network is an adaptation of the [Pathonet model](https://www.nature.com/articles/s41598-021-86912-w) for pulmonary neuroendocrine neoplasms (LNEN); it classifies cells according to two classes, either negative or positive to an immunomarker. This directory also allows the creation of spatial statistics based on graph construction, as proposed by Bullloni and colleagues [See : Automated analysis of proliferating cells spatial organization predicts prognosis in lung neuroendocrine neoplasms, Cancers 2021](https://www.mdpi.com/2072-6694/13/19/4875)

- Original article for the deep learning framework: F. Negahbani [Pathonet](https://www.nature.com/articles/s41598-021-86912-w), SCI REPORT 2021.
- Original code: [https://github.com/SHIDCenter/PathoNet](https://github.com/SHIDCenter/PathoNet)
- Method used for the automated measurements of Ki-67 and PHH3 indices in "Assessment of the current and emerging criteria for the histopathological classification of lung neuroendocrine tumours in the lungNENomics project." ESMO Open 2023 (under review)

## Installation
- Clone this repository: tested on Python 3.6.13
- Install [tensorflow-gpu](https://www.tensorflow.org/?hl=fr): tested on v1.13.1
- Install [keras](https://keras.io/) tested on v2.2.4
- Install [cudatoolkit](https://developer.nvidia.com/cuda-toolkit) tested on v10.0
- Install [cudnn](https://developer.nvidia.com/cudnn) tested on v7.6.5
- Install [networkx](https://networkx.org/) tested on v2.5.1
- Install [opencv-python](https://docs.opencv.org/3.4/d6/d00/tutorial_py_root.html) tested on v4.1.1.26
- Install [scikit-image](https://scikit-image.org/) tested on 0.16.2
- Install [scikit-learn](https://scikit-learn.org/stable/) tested on 0.24.2
- Install [pillow](https://pillow.readthedocs.io/en/stable/)  tested on 8.4.0
- Install any version of pandas, numpy, matplotlib, scipy
- For simplicity [FrEIA Flows](https://github.com/VLL-HD/FrEIA): tested on [the recent branch](https://github.com/VLL-HD/FrEIA/tree/4e0c6ab42b26ec6e41b1ee2abb1a8b6562752b00) has already be cloned in this repository
- Other dependencies in environment.yml

Install all packages with this command:
```
$ conda env create -f environment.yml
```

## Datasets
This method has been tested for 2 types of immunostained WSI:
+ Ki-67:
    + Number of LNEN Ki-67 annotated tiles = 848 (5 patients)
    + To train Pathonet these tiles have been combined to the breast tumor annoted tiles, from the [SHIDC-B-Ki-67 data set](https://shiraz-hidc.com/service/ki-67-dataset/) used in the original paper.
+ PHH3:
    + Number of LNEN PHH3 annotated tiles = 2375 (21 patients)
+ LNEN tiles have autotated semi-automatically using the [QuPath](https://qupath.github.io/) software.

**These two dataset are available on request from mathiane[at]iarc[dot]who[dot]int and will soon be available online.**

## Step 1: Tiles preprocessing 
- Convert annotation files listing all cells on each tile, with their coordinates and class (positive or negative for an immunolabel) in `.json` format to matrices saved in `.npy` format.
- Command line:
```
python preprocessin.py --inputdir PPH3Dataset/train256 --outputdir PPH3DatasetPrepocessed/train256
```

## Step 2: Training
- An example of the configurations used to measure automatically the Ki-67 expression is given in `configs/train_Ki67_LNEN.json
- The commands below are used to train the model:
```
python train.py --configPath configs/train_Ki67_LNEN.json
```
- **Warnings: Network weights will be saved for all epochs in `config.weights-dir/config.class-name/meta-epoch/ModelName_ClassName_MetaEpoch_SubEpoch.pt`. Each checkpoint creates is associated 903MB file.**

## Testing Pretrained Models
- Download pretrained weights are available on request and will be soon available online 
- An example of the configurations used to infer the test set is gien in `Run/Test/TumorNormal/TestToyDataset.sh`
```
bash Run/Test/TumorNormal/TestToyDataset.sh
```
- Main configurations:
    + checkpoint: Path to model weights to be loaded to infer the test tiles.
    + viz-dir: Directory where the result table will be saved.
    + viz-anom-map: If specified, all anomaly maps will be written to the `viz-dir` directory in `.npy` format.

## Results exploration
For each tile, `results_table.csv` summarises:
- Its path, which may include the patient ID
- Binary tile labels, useful for sorted datasets: Tumour = 2 and Non-tumour = 1 
- Max anomaly scores: value of the highest anomaly score of the tile
- Mean anomaly scores: average anomaly score of the tile

**The distributions of these score are used to segment the WSI.**

An example of result exploration for the segmentation of HE/HES WSI is given in `ExploreResultsHETumorSeg.html`.

## Get tumor segmentation map 

The `TumorSegmentationMaps.py` script is used to create the tumour segmentation map for a WSI. An example configuration is given in `ExRunTumorSegmentationMap.sh`. The results of this script are stored in the `Example_SegmentationMap_PHH3` folder, which also gives an example of the model's performance in segmenting a PHH3-immunostained WSI.

## TO DO LIST

+ :construction: Check parallel training 
+ :construction: Check parallel test
+ :construction: Model checkpoints Ki-67 and HES/HE