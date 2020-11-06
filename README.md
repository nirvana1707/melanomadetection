# SIIM-ISIC Melanoma Classification
## Introduction
This project was created as part of a Kaggle research competition: https://www.kaggle.com/c/siim-isic-melanoma-classification
The objective is to identify melanoma in images of skin lesions. 

## Background
As one the most common types of skin cancer, melanoman, is highly deadly and contributes towards 75% of deaths due to skin cancer. Early and accurate detection is critical and can go a long way towards its treatment with minor surgeries.

Currently, dermatologists need to manually evaluate every one of a patient's moles to identify outlier lesions that are most likely to be melanoma. Such a effort heavy process can be exhaustive and often lead to manual errors. Thus, automating the detection of this melanoma can significantly contribute to the dermatologists' diagnostic accuracy, enhance the speed of diagnosis, and moreover help them focus towards remedial measures.

## Dataset
The Society for Imaging Informatics in Medicine (SIIM) joined by the International Skin Imaging Collaboration (ISIC), in its effort to improve melanoma diagnosis, hosts the largest publicly available collection of quality-controlled dermoscopic images of skin lesions.

## Methodology

I used different object detection frameworks on PyTorch to detect melanoma. Experimentations were done with hyperparameters of 3 different architectures:
1. EfficientNet
2. InceptionV4
3. SE-ResNeXt

Implementations of the above architectures (code base) can be found in the repo.

## Results
EfficientNet with an accuracy range of 89%-91% performed better compared to the other architectures.

Sample cases with melanoma

![alt text](https://github.com/nirvana1707/melanomadetection/blob/main/images/melanoma_positive.PNG)

Sample cases without melanoma

![alt text](https://github.com/nirvana1707/melanomadetection/blob/main/images/melanoma_negative.PNG)
