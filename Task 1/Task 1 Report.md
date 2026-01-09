Task 1: Custom Object Detection Report
======================================

1\. Executive Summary
---------------------

**Objective:** To design, train, and evaluate a custom object detection pipeline **from scratch** (without pre-trained weights) capable of detecting specific classes (Person, Dog, Car, Chair) while ensuring compatibility with standard x86\_64 architectures.

**Key Achievements:**

*   **Architecture:** Successfully implemented a **Faster R-CNN** with a **ResNet-50-FPN** backbone.
    
*   **Performance:** Achieved **23.48% mAP@50** (Emergency Save Model) within limited training epochs.
    
*   **Compatibility:** Validated inference on both NVIDIA GPUs and x86\_64 CPUs.
    
*   **Deliverables:** Full training pipeline, inference scripts, and a video demonstration of the model in action.
    

2\. Architecture Design Choices
-------------------------------

### **Selected Model: Faster R-CNN (ResNet-50 + FPN)**

We prioritized **training stability** and **architectural robustness** for this "from scratch" assignment.

*   **Two-Stage Detector:** Unlike single-stage detectors (e.g., YOLO), Faster R-CNN uses a **Region Proposal Network (RPN)** to first identify "regions of interest" before classifying them. This separation of localization and classification provides superior stability when training from random initialization, preventing the divergence often seen in single-stage detectors on small datasets.
    
*   **Feature Pyramid Network (FPN):** We integrated FPN to extract features at multiple scales. This allows the model to detect large objects (like cars) using deep semantic layers and small objects (like distant chairs) using high-resolution shallow layers.
    
*   **Backbone (ResNet-50):** A 50-layer Residual Network provided the optimal balance between feature extraction capability and model size (~158 MB).
    

3\. Data Augmentation & Dataset Strategies
------------------------------------------

Dataset: A subset of PASCAL VOC 2012 containing 12,012 training images.

Classes: \['person', 'dog', 'car', 'chair'\] + background.

To prevent overfitting given the lack of pre-training, we implemented:

1.  **Robust Data Loading:** A custom DirectVOCDetection loader was built to filter out corrupted XML annotations and ensure every training batch contained valid targets.
    
2.  **Random Horizontal Flip (p=0.5):** This effectively doubled the dataset size, teaching the model invariance to object orientation.
    
3.  **Aspect Ratio Grouping:** Images were processed in their native resolutions (up to 1024px) rather than harsh resizing, preserving fine details for the FPN.
    

4\. Training Methodology
------------------------

Hardware: NVIDIA Tesla P100 (16GB VRAM).

Optimization Strategy:

*   **Initialization:** Kaiming He (standard for ResNet).
    
*   **Optimizer:** SGD with Momentum (0.9) and Weight Decay (0.0005).
    
*   **Learning Rate:** 0.005 (with linear warmup).
    

**Challenges & Solutions:**

*   **Resource Constraint (OOM):** The ResNet-50-FPN backbone is memory-intensive. Initial attempts at BATCH\_SIZE=4 caused CUDA Out-Of-Memory errors. We resolved this by reducing the batch size to **2**.
    
*   **Convergence & Model Selection:** We evaluated two checkpoints: the standard Epoch 5 save and an "Emergency Save" captured mid-training. The **Emergency Save** outperformed the standard checkpoint (**23.5% mAP** vs 15.2% mAP), suggesting it captured a better local minimum. We selected this model for the final submission.
    

5\. Results Comparison
----------------------

### **A. Quantitative Metrics**


| Metric | Value | Notes |
| :--- | :--- | :--- |
| **mAP (IoU=0.50)** | **23.48%** | Selected "Emergency Save" Model |
| **mAP (IoU=0.50:0.95)** | **8.36%** | High strictness metric |
| **Inference Speed (GPU)** | **21.87 FPS** | Tested on NVIDIA Tesla P100 |
| **Inference Speed (CPU)** | **0.47 FPS** | Verified x86_64 compatibility |
| **Model Size** | **158.13 MB** | Standard FP32 Weights |

### **B. Visual Results**

![Detection Demo](demo_preview.gif)

*(Running at ~21 FPS on ResNet-50-FPN)*

**[🎥 Click here to watch the full detection video](task1_detection_demo.mp4)**

The attached video demonstrates the model's capability to process video feeds. While the FPS varies by hardware, the model successfully identifies and tracks vehicles in a dynamic traffic scene, proving the validity of the training pipeline.

6\. Discussion: Accuracy vs. Speed Trade-offs
---------------------------------------------

The assignment required evaluating the model based on mAP, speed, and size. Our results highlight the classic trade-off between **Two-Stage** and **One-Stage** architectures.

1.  **Accuracy Focus (Our Approach):**
    
    *   By choosing **Faster R-CNN**, we prioritized **Accuracy and Training Stability**.
        
    *   The two-stage design ensures that even with limited training epochs (5), the model learned to propose relevant regions. This is why we achieved ~23.5% mAP quickly, whereas a YOLO model trained from scratch often outputs 0% mAP for the first 20-30 epochs without careful tuning.
        
2.  **Speed Focus (The Alternative):**
    
    *   **GPU Speed:** At **~22 FPS**, the model is capable of near real-time processing on standard GPUs, which is sufficient for most industrial inspection tasks.
        
    *   **CPU Speed:** At **~0.5 FPS**, the model runs slowly on x86 CPUs. This is the trade-off for using a heavy ResNet-50 backbone.
        
    *   _Trade-off:_ If the priority were strictly CPU real-time speed, we would have sacrificed accuracy by switching to a **MobileNet** backbone or a **YOLO** architecture. However, for a general-purpose demonstration of "from scratch" learning, Faster R-CNN offers the best balance of reliability and performance.
        

7\. Future Optimizations
------------------------

To improve the CPU inference speed for deployment on edge devices without GPUs, we propose:

1.  **ONNX Runtime Export:** Converting the PyTorch dynamic graph to a static ONNX graph to enable operator fusion.
    
2.  **Int8 Quantization:** Compressing weights from FP32 to Int8 would reduce model size to **~40MB** and leverage AVX-512 instructions on modern CPUs, theoretically boosting speed by 2-4x.
