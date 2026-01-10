# Sapiens-Robotics-Tasks


🟢 Task 1: Custom Object Detection from Scratch
-----------------------------------------------

**Objective:** Design and train a pipeline without pre-trained weights to detect four classes (Person, Dog, Car, Chair) while maintaining x86\_64 compatibility.

Report :- [Complete Task 1 Report](https://github.com/mathblender17/Sapiens-Robotics-Tasks/blob/main/Task%201/Task%201%20Report.md)

### Key Achievements

*   **Architecture:** Implemented **Faster R-CNN** with a **ResNet-50-FPN** backbone for superior training stability from random initialization.
    
*   **Performance:** Achieved **23.48% mAP@50** using an "Emergency Save" checkpoint that captured an optimal local minimum.
    
*   **Efficiency:** Validated GPU inference at **~21.87 FPS** on an NVIDIA Tesla P100.
    

🔵 Task 2: Automated Industrial Quality Inspection
--------------------------------------------------

**Objective:** Develop a high-precision prototype for detecting microscopic defects in Printed Circuit Boards (PCBs) using real-world industrial data.

Report :- [Complete Task 2 Report](https://github.com/mathblender17/Sapiens-Robotics-Tasks/blob/main/Task%202/Task%202%20Report.md)

### System Features

*   **Two-Phase Data Strategy:** Utilized a procedural generation engine for initial "cold start" training, followed by adaptation to a real-world dataset with 6 defect categories.
    
*   **Automated Triage:** Automatically calculates (x, y) center coordinates for robotic integration and categorizes defects as **CRITICAL** or **MEDIUM**.
    
*   **Performance:** Achieved a **global mAP50 of 0.716** using a fine-tuned **YOLOv8n** model.
    

### Real-World Performance Metrics
| Defect Type | Precision | Recall | mAP50 | Severity |
| :--- | :---: | :---: | :---: | :--- |
| **Missing Hole** | 0.974 | 0.980 | 0.994 | 🔴 CRITICAL |
| **Short** | 0.813 | 0.692 | 0.795 | 🔴 CRITICAL |
| **Open Circuit** | 0.666 | 0.625 | 0.651 | 🔴 CRITICAL |
| **Mouse Bite** | 0.615 | 0.502 | 0.574 | 🟡 MEDIUM |

🟣 Task 3: Custom VLM Design for Industrial Assistant
-----------------------------------------------------

**Objective:** Design a custom Vision-Language Model (VLM) for offline PCB inspection, enabling inspectors to ask natural language questions about defects with <2s inference.

Report :- [Complete Task 3 Report](https://github.com/mathblender17/Sapiens-Robotics-Tasks/blob/main/Task%203/Task%203%20Report.md)

### Architectural Blueprint

*   **Model Selection:** **Qwen2.5-VL-3B** was selected for its native grounding capabilities and "Naive Dynamic Resolution," which prevents the loss of microscopic pixel data.
    
*   **Localization Precision:** Proposed **Coordinate Token Expansion** (adding 1,000 specialized spatial tokens) and **2D Absolute Positional Encodings** to ensure mathematical grounding in visual features.
    
*   **Optimization:** Employs **4-bit AWQ Quantization** and **Knowledge Distillation** (Teacher-Student) to fit high-level reasoning into a local workstation's VRAM.
    

### Training & Validation Targets

*   **QA Synthesis:** Programmatically converts 50k bounding boxes into 500k instructional QA pairs to teach the model how to "speak" in coordinates.
    
*   **Success Thresholds:** Target **MAE < 0.1** for counting accuracy and **<1% Hallucination Rate** (FPR) using Polling-based Object Probing Evaluation (POPE).
