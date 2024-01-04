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
This method has been tested for 3 types of histological images:
+ Haematoxylin and Eosin (HE) | Haematoxylin, Eosin Saffron (HES) stained WSI:
    + Number of tumor tiles (for train and test) = 12,991 (69 patients)
    + Number of non-tumor tiles (for test) = 4,815 (33 patients)
+ Ki-67 immunohistochemical stained WSI:
    + Number of tumor tiles (for train and test) = 19,053 (77 patients)
    + Number of non-tumor tiles (for test) = 10,259 (40 patients)
+ Phosphohistone histone H3 (PHH3)-stained WSIs can be segmented using Ki-67 tumor tiles as a training set.

**These two dataset are available on request from mathiane[at]iarc[dot]who[dot]int and will soon be available online.**

## Code Organization
- ./custom_datasets - contains dataloaders for TumorNormalDataset :
    - The dataloader is based on a file listing the path to the tiles.
    -  Examples: `./Datasets/ToyTrainingSetKi67Tumor.txt` and `./Datasets/ToyTestSetKi67Tumor.txt`

- ./custom_models 
    - contains pretrained `resnet` feature extractors:
        - For the tumor segmentations tasks we used a wide-Resnet 50 (see: `resnet.py` line 352)
        -  *Note: additional features extrators can be found in the original [CFlow AD repository](https://github.com/gudovskiy/cflow-ad)*
    - the `utils` contains functions to save and load the checkpoint


- ./FrEIA - clone from [https://github.com/VLL-HD/FrEIA](https://github.com/VLL-HD/FrEIA) repository.

- models - Build encoder and decoder
    - The encoder is based on a pretrained resnet (see: `custom_models/resnet.py`)
    - The decoder is based on FrEIA modules

- main: Main script to train and test the model.

## Training Models
- An example of the configurations used to segment HE/HES, Ki-67 and PHH3 WSI is available in `Run/Train/TumorNormal/TrainToyDataKi67.sh`
- *Configs can be viwed in `config.py`*
- The commands below are used to train the model based on the toy data set:
```
bash Run/Train/TumorNormal/TrainToyDataKi67.sh
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