Task 2: Automated Quality Inspection System
===========================================

1\. Project Overview
--------------------

This project implements an automated visual inspection prototype designed for high-precision defect detection in manufactured electronic components. For this prototype, **Printed Circuit Boards (PCBs)** were selected as the primary manufactured item to demonstrate the system's ability to handle complex visible features.

2\. Dataset & Synthetic Generation
----------------------------------

To overcome the "cold start" challenge of limited defective samples, a **procedural data generation engine** was developed. This engine synthesizes unique PCB backgrounds with randomized circuitry and injects three distinct defect classes:

*   **Missing Components (Class 1):** Simulates absent Surface Mount Devices (SMDs).
    
*   **Scratches (Class 0):** High-contrast linear artifacts representing surface abrasions.
    
*   **Discoloration (Class 2):** Localized tinting simulating chemical stains or oxidation.
    

### Training Data Preview

Below is a consolidated grid showing 10 random samples from the procedurally generated training set, demonstrating the diversity of the circuitry and accurate ground-truth labeling.

3\. Computer Vision Solution
----------------------------

The inspection system utilizes a fine-tuned **YOLOv8n (Nano)** model. This model was chosen for its high efficiency and accuracy in real-time manufacturing environments.

### Key Script Features (run\_inference.py):

1.  **Detection & Localization:** Predicts bounding boxes around defect regions with associated confidence scores.
    
2.  **Spatial Mapping:** Automatically calculates the precise (x, y) pixel coordinates for the center of every detected defect.
    
3.  **Severity Assessment:** Categorizes defects as **CRITICAL** (e.g., Missing Components) or **MEDIUM** (e.g., Scratches) based on type and detection confidence.
    

4\. Final Performance Results
-----------------------------

The final model achieved a **Mean Average Precision (mAP50) of 0.948**, with near-perfect reliability in detecting missing components and discoloration.

### Sample Inspection Output

The following image demonstrates the system successfully localizing and classifying multiple concurrent defects on a single test board.
![dataset_samples_grid](https://github.com/user-attachments/assets/f6fd9f27-c3ab-464a-97ef-aa8396706d14)
<img width="682" height="198" alt="sample_output" src="https://github.com/user-attachments/assets/c08d8491-7512-442d-bfc1-7d9c7b6fa4fb" />

<img width="963" height="669" alt="image" src="https://github.com/user-attachments/assets/d6285a83-3a6b-416a-8b63-2e855bf4132b" />
<img width="691" height="674" alt="image" src="https://github.com/user-attachments/assets/7e1244e8-1224-4b3b-bd1a-7eebd60b770e" />
<img width="539" height="722" alt="image" src="https://github.com/user-attachments/assets/d56528f9-8024-4c27-856e-f7f83c08812a" />


5\. Repository Structure
------------------------

*   yolo\_pcb\_model.pt: The fine-tuned model weights.
    
*   run\_inference.py: Python script for running the end-to-end inspection pipeline.
    
*   data\_generator.py: Synthetic data generation engine used to create the training set.
    
*   dataset\_samples\_grid.jpg: Consolidated view of procedural training samples.
    
*   sample\_input.jpg & sample\_label.txt: Raw defective sample image and its ground-truth annotation.
