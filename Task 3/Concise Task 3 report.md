# Task 3: Custom VLM Design for Industrial Quality Inspection

## (A) Model Selection: Qwen2.5-VL (3B/7B)

For an offline, high-precision semiconductor inspection assistant, **Qwen2.5-VL** is the optimal backbone. It bridges the gap between deep linguistic reasoning and microscopic visual accuracy.

### Selection Justification
* **Naive Dynamic Resolution:** Unlike LLaVA or BLIP-2 which resize images to fixed squares, Qwen2.5-VL processes images at their **native aspect ratio and resolution**. This prevents the distortion of microscopic PCB traces and small defects.
* **Native Grounding:** The model is pre-trained to treat bounding boxes as a primary language feature, making it natively capable of "speaking" in coordinates.
* **Offline Feasibility:** The **3B parameter version** is compact enough to achieve **<2s inference** on a local industrial GPU (8GB-12GB VRAM) while maintaining 100% data privacy.

### Key Factors
| Factor | Choice / Detail | Impact |
| :--- | :--- | :--- |
| **Model Size** | 3B Parameters | Enables sub-2s latency on factory-floor hardware. |
| **Licensing** | Apache 2.0 | Allows for full commercial ownership and offline deployment. |
| **Flexibility** | LoRA/QLoRA Support | Facilitates low-cost fine-tuning on the 50,000 PCB samples. |

### Architectural Modifications for Localization
1. **Coordinate Token Expansion:** We expand the language vocabulary with 1,000+ spatial tokens (e.g., `<bin_0>...<bin_999>`) to represent normalized pixel coordinates as learned "words."
2. **Feature Pyramid Neck:** We implement a multi-scale neck between the vision encoder and LLM to preserve low-level geometric edges (traces) alongside high-level semantic labels (defect types).

---

## (B) Design Strategy: Architecture for Industrial Precision

To handle the high-density circuitry of a PCB, the architecture is modified to prioritize **spatial fidelity** over conversational "fluff."



### Component Modification Table

| Component | Modification Strategy | Engineering Impact |
| :--- | :--- | :--- |
| **Vision Encoder** | **Multi-Scale Swin-Transformer** | Captures hierarchical features, ensuring tiny "mouse-bites" aren't lost during feature extraction. |
| **Language Decoder** | **Spatial Token Integration** | Forces the model to treat locations as mathematical facts rather than descriptive phrases. |
| **Fusion Mechanism** | **Query-Based Cross-Attention** | Uses the inspector's text prompt to act as a "searchlight" over specific high-res visual patches. |
| **Output Layer** | **Schema-Enforced Decoding** | A logit processor restricts output to structured JSON: `{"defect": str, "bbox": [x,y,w,h]}`. |

### Key Design Pillars

#### 1. Vision Encoder: The "Macro-to-Micro" Eye
Standard encoders lose detail through aggressive pooling. 
* **Tiled Processing:** We implement a tiling strategy where the PCB is processed as a series of high-resolution patches ($1024 \times 1024$).
* **Edge Preservation:** The **Swin-Transformer** backbone uses window-based attention, which is superior at maintaining the sharp geometric edges of semiconductor traces.



#### 2. Fusion Mechanism: "Searchlight" Attention
Instead of simple token stacking, we use **Asymmetric Cross-Attention**.
* **Dynamic Grounding:** When an inspector asks about a specific area, the language tokens "pull" high-resolution features from that spatial coordinate. 
* **Focusing:** This allows the model to ignore 90% of the "clean" board features, reducing computation and preventing hallucinations.

#### 3. Absolute Spatial Awareness
To eliminate "guessing," we modify the embedding layer:
* **2D Positional Encodings:** We inject absolute 2D positional embeddings into the vision-to-language projection. 
* **GPS-Tagging:** Every visual token passed to the brain carries a permanent "GPS tag" of its location, ensuring the model never loses the defect's exact position during the reasoning process.
---
## (C) Optimization: Sub-2s Offline Deployment

To achieve industrial-grade latency on local hardware, we implement a multi-tier optimization strategy focusing on model compression and hardware-aware inference.

| Optimization Technique | Implementation | Latency Impact |
| :--- | :--- | :--- |
| **Quantization** | **4-bit AWQ (Activation-aware)** | Reduces VRAM footprint by 75%; ~1.2s reduction in TTFT. |
| **Model Distillation** | **Teacher-Student SFT** | Transfers reasoning from 72B to 3B model; maintains accuracy. |
| **Hardware-Aware** | **Flash-Attention 2** | Optimizes IO memory access; 2x throughput on high-res tiles. |
| **Caching** | **Static KV Caching** | Minimizes redundant compute for repeated board queries. |

### 1. 4-bit AWQ Quantization
We utilize **Activation-aware Weight Quantization (AWQ)** to compress the model from FP16 to 4-bit precision. Unlike standard rounding, AWQ identifies and protects the "salient" weights most critical for PCB trace recognition, ensuring high detection accuracy while allowing the model to fit into 4GB-8GB of VRAM.

### 2. Knowledge Distillation
To maximize reasoning power within a small 3B model, we employ a **Teacher-Student** framework. A large-scale VLM (e.g., Qwen-VL-72B) acts as a Teacher to generate high-fidelity reasoning paths for our 50,000 PCB images. The smaller 3B Student model is then trained to mimic these outputs, gaining "large model logic" without the associated latency.

---

## (D) Hallucination Mitigation: Ensuring Industrial Truth

Hallucinations in a semiconductor environment are costly. We implement a multi-layered approach to ensure the VLM remains strictly grounded in visual evidence.



### 1. Visual Contrastive Decoding (VCD)
To reduce over-reliance on statistical language priors (e.g., the model "expecting" a short near a missing hole), we utilize **Visual Contrastive Decoding**. 
* **Mechanism:** The model compares logit distributions from the original image against a "distorted" version. By subtracting the "hallucinated" features of the distorted image, we amplify tokens grounded in actual sharp visual features.

### 2. Negative Constraint Training (Hard-Negative Mining)
Standard VLMs are often biased toward "finding something" to please the user.
* **Instruction Tuning:** 30% of our training data consists of clean, defect-free boards. We prompt the model with adversarial questions (e.g., "Locate the crack" on a clean board) and reward it for responding with a specialized `<NULL_DETECTION>` token.

### 3. Visual Grounding (VG) Loss
During fine-tuning, we incorporate a **Grounded-Contrastive Loss**. This mathematical constraint penalizes the model if its internal attention map does not align with the ground-truth bounding box of the defect it is currently describing.

| Mitigation Layer | Technique | Industrial Benefit |
| :--- | :--- | :--- |
| **Inference** | VCD (Contrastive) | Prevents "guessing" based on common text patterns. |
| **Fine-Tuning** | Hard-Negative Mining | Eliminates false positives on high-quality clean boards. |
| **Validation** | Uncertainty Check | Triggers a sub-patch "re-scan" if token confidence is low. |
---
## (E) Training Plan: 3-Stage Instructional Curriculum

To transform 50,000 detection-only bounding boxes into a conversational industrial assistant, we implement a multi-stage training pipeline focusing on visual alignment and synthetic instruction generation.

### 1. Stage 1: Automated QA Synthesis
Since the industrial dataset lacks natural language pairs, we utilize **Template-Based Generation** to convert the existing 50,000 bounding boxes into a 500k-sample instruction set. 
* **Templates:** Questions cover counting ("How many shorts?"), localization ("Where is the crack?"), and verification ("Is the board clean?").
* **Spatial Normalization:** All coordinates are converted into normalized tokens `<bin_x>` to `<bin_y>` to align with the model's spatial vocabulary.

### 2. Stage 2: Feature Alignment (Projection Tuning)
We first train the **multimodal projector** using a frozen vision encoder and LLM. 
* **Objective:** This stage aligns the specific visual features of semiconductor materials (solder, silicon, copper) with the model’s existing linguistic knowledge of "defects" and "locations."

### 3. Stage 3: Instruction Fine-Tuning (LoRA)
We employ **LoRA (Low-Rank Adaptation)** for efficient end-to-end fine-tuning of the vision and language weights simultaneously.
* **Balanced Sampling:** Training batches consist of 70% positive samples (defective boards) and 30% negative samples (clean boards).
* **Adversarial Training:** Prompts include "trick" questions to strengthen the model's refusal capability when no defects are present.

| Training Stage | Component Focus | Data Volume | Target Metric |
| :--- | :--- | :--- | :--- |
| **Alignment** | Projection Head | 50,000 Images | Feature Overlap |
| **SFT (LoRA)** | LLM + Encoder | 500,000 QA Pairs | Instruction Following |
| **Constraint**| Logit Processor | All Samples | Structured JSON Output |

---

## (F) Validation: Proving Industrial Reliability

Validating a VLM for semiconductor deployment requires a three-dimensional evaluation framework focusing on spatial precision, counting reliability, and hallucination control.



### 1. Localization Precision (mAP)
We evaluate the model’s **Visual Grounding** capability using Mean Average Precision (mAP) across varying IoU thresholds (.50:.95). 
* **Precision Goal:** High mAP@.75 ensures that the model’s predicted coordinate tokens are precise enough for automated marking or robotic repair systems.

### 2. Counting Accuracy (MAE)
For batch-level quality control, we measure **Mean Absolute Error (MAE)** in defect enumeration. 
* **Reliability Check:** The model must maintain a low MAE even in high-density images where multiple defects (e.g., several spurious copper flecks) are clustered together.

### 3. Hallucination Rate (POPE Evaluation)
Using the **Polling-based Object Probing Evaluation (POPE)**, we measure how often the model generates non-existent defects when prompted with adversarial questions on clean boards.

| Validation Dimension | Metric | Success Threshold |
| :--- | :--- | :--- |
| **Localization** | mAP@.50 | > 0.85 |
| **Counting** | Mean Absolute Error (MAE) | < 0.1 |
| **Truthfulness** | Hallucination Rate (FPR) | < 1.0% |
| **Speed** | End-to-End Latency | < 2.0 Seconds |
