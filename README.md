# 'Pay "Attention" to Adverse Weather: Weather-aware Attention-based Object Detection'

This is the repository for our ICPR2022 paper ['Pay "Attention" to Adverse Weather: Weather-aware Attention-based Object Detection'](https://ieeexplore.ieee.org/abstract/document/9956149)

Despite the recent advances of deep neural networks, object detection for adverse weather remains challenging due to the poor perception of some sensors in adverse weather. Instead of relying on one single sensor, multimodal fusion has been one promising approach to provide redundant detection information based on multiple sensors. However, most existing multimodal fusion approaches are ineffective in adjusting the focus of different sensors under varying detection environments in dynamic adverse weather conditions. Moreover, it is critical to simultaneously observe local and global information under complex weather conditions, which has been neglected in most early or late-stage multimodal fusion works. In view of these, this paper proposes a Global-Local Attention (GLA) framework to adaptively fuse the multi-modality sensing streams, i.e., camera, gated, and lidar data, at two fusion stages. Specifically, GLA integrates an early-stage fusion via a local attention network and a late-stage fusion via a global attention network to deal with both local and global information, which automatically allocates higher weights to the modality with better detection features at the late-stage fusion to cope with the specific weather condition adaptively. Experimental results demonstrate the superior performance of the proposed GLA compared with state-of-the-art fusion approaches under various adverse weather conditions, such as light fog, dense fog, and snow.

<img width="1359" alt="Screenshot 2025-02-03 at 5 06 17 PM" src="https://github.com/user-attachments/assets/18be2a94-eb6f-4366-aba2-a2cc5ddbfb0a" />





## Contents

The contents of the repository:
1. 'Object_Detection_Codes' contains codes for Camera Only, Gated Only, Camera-Lidar Early fusion, Camera-Camera Fusion, Gated-Gated fusion, and Camera-Lidar fusion codes.
2. 'Faster_RCNN_Results' contains sample outputs of each object detection models.
3. ''SeeingThroughFogData_Samples' contains sample gated data images and sample lidar projections on camera images. 
4. 'tools' contains TFRecord creation codes for each object detection models. 
5. 'splits' contains training, validation dataset splits provided in SeeingThroughFog paper.


## Contributions

This repository has been developed for multimodal fusion from the open source repository [Tensoflow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection/README.md)


## Citation
If you find our code useful, please cite our paper. :)

`@INPROCEEDINGS{9956149,
  author={Chaturvedi, Saket S. and Zhang, Lan and Yuan, Xiaoyong},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Pay "Attention" to Adverse Weather: Weather-aware Attention-based Object Detection}, 
  year={2022},
  volume={},
  number={},
  pages={4573-4579},
  keywords={Deep learning;Laser radar;Neural networks;Object detection;Sensor fusion;Logic gates;Feature extraction;Object Detection;Adverse Weather;Multimodal Fusion;Attention Neural Network},
  doi={10.1109/ICPR56361.2022.9956149}}`
