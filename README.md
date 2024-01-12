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
- An example of the configurations used to train the model to automatically measure Ki-67 expression is given in `configs/train_Ki67_LNEN.json`
- The command below is used to train the model:
```
python train.py --configPath configs/train_Ki67_LNEN.json
```
- The trained model weights for Ki-67-stained LNEN WSIs are stored in the file `CheckpointKi67/Pathonet_Ki67_for_LNEN.hdf5`

## Step 3: Test
- An example of the configurations used to test the model is given in `configs/eval_Ki67_LNEN.json`
- The command below is used to evaluate the model:
```
python evaluation.py --inputPath test256_LNENonly --configPath configs/eval_Ki67_LNEN.json 
```

## Step 4: Optimize the cell detection threshold
- The post-processing pipeline applied after UNET uses dea seuila to establish which are the "true cells". We propose to optimize these thresholds for marker-positive and marker-negative cells using the `eval_opt_thresholds.py` script.
- The command below is used to run the script for channel 0 associated with cells detected as marker positive:
```
python eval_opt_thresholds.py --inputPath test256 --configPath configs/eval_Ki67_LNEN.json --dataname OptTh0_pos_cells.csv --minth 75 --minth 95 --channel 0
```
- Network performance is tested for all thresholds between `minth` and `minth` in steps of 5 units.
- Results are saved in the table specified by `dataname`.
- To optimize the threshold associated with cells detected as negative at the marker, specify `--channel 1`.



## TO DO LIST

+ :construction: Add PHH3 configs and weights
+ :construction: Add `demo.py` > Inference script
+ :construction: Add NetworkX construction
+ :construction: Add Spatial statistics
+ :construction: Add presentation WSI