# Violence Detection using Dense Multi Head Self-Attention with Bidirectional Convolutional LSTM


This is the code for the paper:

[ViolenceNet: Dense Multi Head Self-Attention with Bidirectional Convolutional LSTM for Detecting Violence<br/>
Fenando José Rendón Segador, Juan Antonio Álvarez-García, Fernando Enriquez, Oscar Deniz](https://www.mdpi.com/2079-9292/10/13/1601/htm)

The paper addresses the problem of detecting violent actions in videos, analyzing the state of the art of various deep learning models and developing a new one. Our aim is to develop a new neural network model capable of reaching or exceeding the state of the art.

## Abstract

Introducing efficient automatic violence detection in video surveillance or audiovisual content monitoring systems would greatly facilitate the work of closed-circuit television (CCTV) operators, rating agencies or those in charge of monitoring social network content. In this paper we present a new deep learning architecture, using an adapted version of DenseNet for three dimensions, a multi-head self-attention layer and a bidirectional convolutional long short-term memory (LSTM) module, that allows encoding relevant spatio-temporal features, to determine whether a video is violent or not. Furthermore, an ablation study of the input frames, comparing dense optical flow and adjacent frames subtraction and the influence of the attention layer is carried out, showing that the combination of optical flow and the attention mechanism improves results up to 4.4%. The conducted experiments using four of the most widely used datasets for this problem, matching or exceeding in some cases the results of the state of the art, reducing the number of network parameters needed (4.5 millions), and increasing its efficiency in test accuracy (from 95.6% on the most complex dataset to 100% on the simplest one) and inference time (less than 0.3 s for the longest clips). Finally, to check if the generated model is able to generalize violence, a cross-dataset analysis is performed, which shows the complexity of this approach: using three datasets to train and testing on the remaining one the accuracy drops in the worst case to 70.08% and in the best case to 81.51%, which points to future work oriented towards anomaly detection in new datasets.

## Requirements

This project is implemented in Python using the [Tensorflow](https://www.tensorflow.org/) and [Keras](https://keras.io/) libraries to develop the model.

- OpenCV v4.4.0.46
- Ipython v7.16.1
- Keras v2.4.3
- Scikit-Image v0.17.2
- Scikit-Learn v0.24.2
- Tensorflow v2.5.1
- Livelossplot v0.5.3
- Matplotlib v3.3.2
- Numpy v1.19.2


### Install the repository:

```
git clone https://github.com/FernandoJRS/violence-detection-deeplearning
```

### Install the requirements:

```
pip install -r requirements.txt
```

## Model Architecture

The following graph shows the architecture of our model.

![Model Architecture](figures/ModelArchitecture.png?raw=True "Model Architecture")

In section A the architecture of the ViolenceNet model, that takes the optical flow as input, is shown. It is composed of four parts: a DenseNet-121 network spatio-temporal encoder, a multi-head self-attention layer, a bidirectional convolution 2D LSTM (BiConvLSTM2D) layer and a classifier. Below each Dense Block its number of components is indicated. The variable h corresponds to the number of heads used in parallel by the multi-head self-attention layer and the variables Q,K,V their inputs. Section B shows the internal architecture of a 5-component DenseBlock (x5).

## Results

This section shows the results obtained from the different experiments carried out with the datasets (HF - Hockey Fights), (MF - Movies Figths), (VF - Violent Flows) and (RLVS - Real Life Violence Situations).


### Ablation Study With Attention Mechanism, 5-Fold Cross-Validation

| Dataset |         Input        | Test Accuracy (with Attention) | Test Accuracy (without Att.) | Test Inference Time (with Attention) | Test Inference Time (without Att.) |
|---------|----------------------|--------------------------------|------------------------------|--------------------------------------|------------------------------------|
|   HF	  | Optical Flow         |         99.20 ± 0.6%	          |         99.00 ± 1.0%         |   0.1397 ± 0.0024 s                  |    0.1626 ± 0.0034 s               |
|   HF	  | Pseudo-Optical Flow  |         97.50 ± 1.0%	          |         97.20 ± 1.0%         |   0.1397 ± 0.0024 s                  |    0.1626 ± 0.0034 s               |
|   MF	  | Optical Flow         |         100.00 ± 0.0%	      |         100.00 ± 0.0%        |   0.1916 ± 0.0093 s                  |    0.2019 ± 0.0045 s               |
|   MF	  | Pseudo-Optical Flow  |         100.00 ± 0.0%	      |         100.00 ± 0.0%        |   0.1916 ± 0.0093 s                  |    0.2019 ± 0.0045 s               |
|   VF	  | Optical Flow         |         96.90 ± 0.5%	          |         94.00 ± 1.0%         |   0.2991 ± 0.0030 s                  |    0.3114 ± 0.0073 s               |
|   VF	  | Pseudo-Optical Flow  |         94.80 ± 0.5%	          |         92.50 ± 0.5%         |   0.2991 ± 0.0030 s                  |    0.3114 ± 0.0073 s               |
|  RLVS	  | Optical Flow         |         95.60 ± 0.6%	          |         93.40 ± 1.0%         |   0.2767 ± 0.020 s                   |    0.3019 ± 0.0059 s               |
|  RLVS	  | Pseudo-Optical Flow  |         94.10 ± 0.8%	          |         92.20 ± 0.8%         |   0.2767 ± 0.020 s                   |    0.3019 ± 0.0059 s               |

### Performance Comparison For One Iteration

| Dataset |         Input        | Training Accuracy | Training Loss | Test Accuracy Violence | Test Accuracy Non-Violence | Test Accuracy |
|---------|----------------------|-------------------|---------------|------------------------|----------------------------|---------------|
|  HF	  | Optical Flow         |  100%             |  1.20×10−5    |  99.00%                |  100.00%                   |  99.50%       |
|  HF	  | Pseudo-Optical Flow  |  99%              |  1.35×10−5    |  97.00%                |  98.00%                    |  97.50%       |
|  MF	  | Optical Flow         |  100%             |  1.18×10−5    |  100%                  |  100%                      |  100%         |
|  MF	  | Pseudo-Optical Flow  |  100%             |  1.19×10−5    |  100%                  |  100%                      |  100%         |
|  VF	  | Optical Flow         |  98%              |  1.50×10−4    |  97.00%                |  96.00%                    |  96.50%       |
|  VF	  | Pseudo-Optical Flow  |  97%              |  2.94×10−4    |  95.00%                |  94.00%                    |  94.50%       |
|  RLVS	  | Optical Flow         |  97%              |  3.10×10−4    |  96.00%                |  95.00%                    |  95.50%       |
|  RLVS	  | Pseudo-Optical Flow  |  95%              |  7.31×10−4    |  94.00%                |  93.00%                    |  93.50%       |

### Cross-Dataset Experiment, 5-Fold Cross-Validation

| Dataset Training | Dataset Testing | Test Accuracy Optical Flow | Test Accuracy Pseudo-Optical Flow |
|------------------|-----------------|----------------------------|-----------------------------------|
|  HF	           | MF              |  65.18 ± 0.34%             |  64.86 ± 0.41%                    |
|  HF	           | VF              |  62.56 ± 0.33%             |  61.22 ± 0.22%                    |
|  HF	           | RLVS            |  58.22 ± 0.24%             |  57.36 ± 0.22%                    |
|  MF	           | HF              |  54.92 ± 0.33%             |  53.50 ± 0.12%                    |
|  MF	           | VF              |  52.32 ± 0.34%             |  51.77 ± 0.30%                    |
|  MF	           | RLVS            |  56.72 ± 0.19%             |  55.80 ± 0.20%                    |
|  VF	           | HF              |  65.16 ± 0.59%             |  64.76 ± 0.49%                    |
|  VF	           | MF              |  60.02 ± 0.24%             |  59.48 ± 0.16%                    |
|  VF	           | RLVS            |  58.76 ± 0.49%             |  58.32 ± 0.27%                    |
|  RLVS	           | HF              |  69.24 ± 0.27%             |  68.86 ± 0.14%                    |
|  RLVS	           | MF              |  75.82 ± 0.17%             |  74.64 ± 0.22%                    |
|  RLVS	           | VF              |  67.84 ± 0.32%             |  66.68 ± 0.22%                    |
|  HF + MF + VF	   | RLVS            |  70.08 ± 0.19%             |  69.84 ± 0.14%                    |
|  HF + MF + RLVS  | VF              |  76.00 ± 0.20%             |  75.68 ± 0.14%                    |
|  HF + RLVS + VF  | MF              |  81.51 ± 0.09%             |  80.49 ± 0.05%                    |
|  RLVS + MF + VF  | HF              |  79.87 ± 0.33%             |  78.63 ± 0.01%                    |


## Evaluation

For each dataset the model has been trained on, we provide three Jupyter notebooks in this repository. Each of these notebooks has a different role with the datasets. The first of the notebooks allows you to execute the training of the model and perform an evaluation of it. The second of the notebooks allows to run the pre-trained model with a percentage of testing the data set. Finally, the third notebook allows you to run the pre-trained model with the entire data set as a test, in addition to being able to cross-validate by changing the dataset for another with which the model has not been trained.

To run all the notebooks it is necessary to previously download the corresponding datasets. For the first notebook of each of the datasets, it is also necessary to download the kaggle.json file and load it by executing the first cell of the notebook. For the second and third notebooks of each of the datasets, it is also necessary to download the pre-trained models for each of the corresponding datasets.

### Pre-trained models

In the following [link](https://drive.google.com/drive/folders/1IN6KE9WOH2ix0kH_bVug0QuO1i8la0TE?usp=sharing) you can find the pre-trained models with the Movies Fights, Hockey Fights and Violent Flows datasets.

### Datasets

In the following [link](https://www.kaggle.com/frendon/datasets) you can find all the datasets that have been used in this project.

### Loss, Accuracy Hockey Fights Graphics

### Loss, Accuracy Movies Fights Graphics

### Loss, Accuracy Violent Flows Graphics

## Citation

This section gives the information to cite the paper: ViolenceNet: Dense Multi-Head Self-Attention with Bidirectional Convolutional LSTM for Detecting Violence (2021), doi: https://doi.org/10.3390/electronics10131601.

```bash
@article{rendon2021violencenet,
  title={ViolenceNet: Dense Multi-Head Self-Attention with Bidirectional Convolutional LSTM for Detecting Violence},
  author={Rend{\'o}n-Segador, Fernando J and {\'A}lvarez-Garc{\'\i}a, Juan A and Enr{\'\i}quez, Fernando and Deniz, Oscar},
  journal={Electronics},
  volume={10},
  number={13},
  pages={1601},
  year={2021},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

## Acknowledgements
This research is partially supported by The Spanish Ministry of Economy and Competitiveness through the project VICTORY (grant no.: TIN2017-82113-C2-1-R).
