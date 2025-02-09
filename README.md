# **Pay "Attention" to Adverse Weather: Weather-Aware Attention-Based Object Detection**  

This repository contains the implementation of our **ICPR 2022** paper:  
**[Pay "Attention" to Adverse Weather: Weather-Aware Attention-Based Object Detection](https://ieeexplore.ieee.org/abstract/document/9956149)**  

## **Overview**  
Despite advancements in deep neural networks, object detection in adverse weather remains a significant challenge due to sensor degradation under extreme conditions. Instead of relying on a single sensor, **multimodal fusion** enhances detection by leveraging complementary sensor data. However, existing multimodal fusion methods struggle to dynamically adjust sensor importance under varying weather conditions. Furthermore, most methods neglect the simultaneous need for **local and global information** during perception.  

To address these challenges, our paper introduces a **Global-Local Attention (GLA) framework** that adaptively fuses multiple sensor streams—**camera, gated, and LiDAR data**—at two fusion stages:  
- **Early-stage fusion** via a **local attention network**, capturing localized feature variations.  
- **Late-stage fusion** via a **global attention network**, assigning higher importance to the most reliable modality under the current weather condition.  

### **Key Contributions:**  
✅ A novel **Global-Local Attention (GLA) framework** for multimodal fusion.  
✅ Adaptive sensor weighting to improve detection under **dynamic weather conditions**.  
✅ **Experimental validation** demonstrating superior performance over existing fusion methods in **light fog, dense fog, and snow** scenarios.  

<p align="center">
  <img width="800" alt="GLA Framework" src="https://github.com/user-attachments/assets/18be2a94-eb6f-4366-aba2-a2cc5ddbfb0a">
</p>  

---

## **Contents**  

📁 **`Object_Detection_Codes`** – Implementations of different object detection models:  
  - Camera-Only  
  - Gated-Only  
  - Camera-LiDAR Early Fusion  
  - Camera-Camera Fusion  
  - Gated-Gated Fusion  
  - Camera-LiDAR Fusion  

📁 **`Faster_RCNN_Results`** – Sample object detection outputs from each model.  

📁 **`SeeingThroughFogData_Samples`** – Sample gated images and LiDAR projections on camera images.  

📁 **`tools`** – Scripts for **TFRecord** creation for different object detection models.  

📁 **`splits`** – Dataset splits for **training** and **validation**, as provided in the **SeeingThroughFog** paper.  

---

## **Installation & Usage**  

### **Dependencies:**  
- Python 3.x  
- TensorFlow (TF Object Detection API)  
- OpenCV  
- NumPy  
- Matplotlib  

### **Contributions**

This repository builds on the [TensorFlow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection), which has been adapted to support multi-sensor fusion settings for enhanced object detection in autonomous driving applications. Contributions and improvements are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

This repository builds on the TensorFlow Object Detection API. Contributions and improvements are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

### **Citation**

If you find our work useful, please consider citing our paper:
```bash
@INPROCEEDINGS{9956149,
  author={Chaturvedi, Saket S. and Zhang, Lan and Yuan, Xiaoyong},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)}, 
  title={Pay "Attention" to Adverse Weather: Weather-aware Attention-based Object Detection}, 
  year={2022},
  pages={4573-4579},
  keywords={Deep learning; Laser radar; Neural networks; Object detection; Sensor fusion; Attention Neural Network},
  doi={10.1109/ICPR56361.2022.9956149}
}
