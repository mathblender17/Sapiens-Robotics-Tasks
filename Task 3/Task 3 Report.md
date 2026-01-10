### **(A) Model Selection: Qwen2.5-VL-3B (or 7B)**

For this design, **Qwen2.5-VL** is the superior choice. While LLaVA is a great generalist and BLIP-2 pioneered efficient alignment, Qwen-VL was built with a "spatial-first" mindset that is essential for circuitry.

#### **1\. Why Qwen2.5-VL?**

*   **Naive Dynamic Resolution:** Standard models (like LLaVA-1.5) resize images to a fixed square (e.g., 336x336), which would turn a tiny hairline crack on a PCB into a blurry smudge. Qwen-VL handles images at their **native resolution**, allowing it to "see" microscopic defects without losing pixel data.
    
*   **Native Grounding:** It is one of the few open-weight models that treats **bounding boxes as a primary language**. It doesn't just describe a defect; it can "speak" in coordinates natively.
    

#### **2\. Factors Influencing Choice**

*   **Model Size (3B vs 7B):** A **3B model** is preferred here. It is small enough to fit into the VRAM of a standard industrial workstation (8GB–12GB) while leaving enough "compute headroom" to meet the **<2s inference** requirement.
    
*   **Inference Speed:** Qwen2.5-VL supports **Flash Attention** and optimized quantization, which are critical for processing high-resolution industrial images locally.
    
*   **Fine-tuning Flexibility:** Because it uses a standard transformer architecture, we can apply **LoRA (Low-Rank Adaptation)** to specialize it on the 50,000 PCB images without needing a supercomputer cluster.
    
*   **Licensing:** Released under the **Apache 2.0 license**, it allows the manufacturer to keep the model, the weights, and the data 100% offline and proprietary.
    

### **3\. Architectural Modifications for Precise Localization**

A generic "out-of-the-box" VLM is still too "chatty" for semiconductor precision. To hit the required localization accuracy, we must implement these three modifications:

#### **I. Coordinate Token Expansion**

We modify the model's vocabulary to include **1,000 specialized tokens** representing normalized coordinates (e.g., to ). This stops the model from "guessing" numbers and instead treats the location as a specific, learned "word."

#### **II. Two-Stage "Zoom-In" Grounding**

Because PCB defects are often less than 1% of the total image area, we implement a **Recursive Vision-Language approach**:

1.  **Global Scan:** The model identifies a "Region of Interest" (ROI) on the full board.
    
2.  **Local Zoom:** The architecture automatically crops that specific coordinate and re-feeds the high-resolution patch back into the vision encoder for a "second look".
    

#### **III. 2D Absolute Positional Encodings**

Standard models only know the _order_ of image patches. We replace this with **2D Absolute Encodings** injected at the projection layer. This ensures every visual token carries a permanent "GPS coordinate," preventing the model from losing the defect's exact location when it starts "thinking" about the text response.

**Summary of Selection (A)**:By choosing **Qwen2.5-VL-3B** with **Coordinate Token Expansion** and a **Zoom-In Mechanism**, we create a model that is fast enough for the factory floor but precise enough to pinpoint a 10-micron short.

---

**(B) Design Strategy: Architecture for Industrial Precision**
--------------------------------------------------------------

### **1\. Vision Encoder: Multi-Scale "Expert" Perception**

Standard Vision Transformers (ViT) often "squash" images into low-resolution patches (e.g., $14 \\times 14$), which can delete microscopic defects like hairline cracks or 50μm "mouse bites."

*   **Dual-Path Visual Backbone:** We implement a **Global-Local Encoder**. A low-resolution path processes the entire PCB to understand global context (layout), while a **High-Resolution "Expert" path** (based on a Swin Transformer or SigLIP) processes overlapping tiles at $1024 \\times 1024$.
    
*   **Feature Pyramid Integration:** We modify the encoder to output a **Feature Pyramid**. Instead of a single visual embedding, the model extracts low-level geometric features (edges/traces) and high-level semantic features (defect types) simultaneously.
    

### **2\. Fusion Mechanism: "Searchlight" Cross-Attention**

Most VLMs use "Early Fusion," simply stacking image tokens with text tokens. In a dense PCB, this causes the model to lose track of where a defect is.

*   **Cross-Modal Attention Gating:** We replace the standard MLP projector with a **Gated Cross-Attention module**. When a user asks, "Find the short on the top-right corner," the text tokens act as a spatial query. This "searchlight" mechanism dynamically weighs visual tokens based on the linguistic prompt, focusing the model's computational energy on the relevant sub-region of the board.
    
*   **Coordinate-Injection Projection:** We modify the projection layer to inject **2D Absolute Positional Encodings**. Every visual token passed to the language decoder is permanently tagged with its $\[x, y\]$ normalized coordinate. This ensures the "Fusion" step doesn't just combine _what_ is there, but _where_ it is.
    

### **3\. Language Decoder: Constrained Industrial Output**

To prevent hallucinations and conversational "fluff," the language decoder is modified to act as a structured data generator.

*   **Coordinate-Aware Tokenization:** We expand the decoder’s vocabulary by adding **1,000 spatial tokens** ($, , ... $). This allows the model to "speak" in coordinates natively, treating a location as a learned concept rather than a string of random digits.
    
*   **Schema-Constrained Decoding:** The decoder is architecturally limited via a **Logit Processor** to only output valid JSON structures during inference. This ensures every response contains a location, class, and confidence score in a machine-readable format for the factory's logging software.

| Component | Modification Strategy | Engineering Impact |
| :--- | :--- | :--- |
| **Vision Encoder** | **Multi-Scale Feature Pyramid (Swin-T)** | Replaces fixed-size backbones to capture both large-scale board layout and micron-scale trace defects. |
| **Language Decoder** | **Coordinate-Aware Vocabulary Expansion** | Adds 1,000+ spatial tokens (e.g., `<x0>`...`<y999>`) to the internal dictionary, forcing the model to "speak" in pixels. |
| **Fusion Mechanism** | **Asymmetric Cross-Attention Gating** | Allows the textual query to act as a "spatial filter" over high-resolution image patches. |
| **Output Head** | **Schema-Enforced Decoding (JSON)** | Implements a "Logit Processor" that restricts model output to structured formats: `{class, bbox, conf}`. |

---

**(C) Optimization: Sub-2s Offline Deployment**
-----------------------------------------------

### **1\. 4-bit AWQ Quantization (The Speed Foundation)**

A raw VLM in FP16 precision is too heavy for rapid inference. We use **Activation-aware Weight Quantization (AWQ)**.

*   **Strategy:** We compress the model's weights from 16-bit to **4-bit**. Unlike standard quantization, AWQ identifies the "salient" weights—those most critical for accuracy—and protects them while aggressively compressing the rest.
    
*   **Impact:** Reduces VRAM usage by **70–75%** and provides a **2-3x speedup** in token generation, ensuring the language response is near-instant.
    

### **2\. Knowledge Distillation (Student-Teacher Learning)**

If the 7B model is too slow but the 2B model is too "uninformed," we use distillation to bridge the gap.

*   **Strategy:** We use a high-capacity "Teacher" model (e.g., Qwen-VL-72B) to process the PCB dataset and generate rich, reasoning-based labels. Our smaller "Student" model (the 3B version) is then trained to **mimic the Teacher's output distribution**.
    
*   **Impact:** The 3B model gains the analytical "logic" of a much larger model while maintaining the physical speed of a small architecture.
    

### **3\. Flash-Attention & KV Caching**

Handling high-resolution images generates a massive amount of "tokens" that can clog the GPU.

*   **Flash-Attention 2:** We implement an IO-aware attention mechanism that reduces the number of memory reads/writes, cutting down vision-processing time significantly.
    
*   **KV Caching:** For repetitive inspection tasks, we cache the "Key-Value" states of the board's static components. The model only has to "re-calculate" the parts of the board it is currently discussing in the query.
    

### **4\. LoRA Adapters for Modular Intelligence**

Instead of loading a whole new model for different PCB types, we use **Low-Rank Adaptation (LoRA)**.

*   **Strategy:** We keep the base VLM frozen and only load tiny (10MB-100MB) adapter files specialized for specific PCB families (e.g., "Motherboards" vs "Sensor Arrays").
    
*   **Impact:** These adapters can be swapped in milliseconds, allowing the same offline system to pivot between different product lines without reloading the entire 6GB+ model.

| Optimization Technique | Implementation | Latency Impact |
| :--- | :--- | :--- |
| **Quantization** | 4-bit AWQ | ~1.2s reduction |
| **Distillation** | Teacher-Student SFT | Maintains accuracy at speed |
| **Hardware-Aware** | Flash-Attention 2 | 2x vision throughput |
---

**(D) Hallucination Mitigation: Ensuring Industrial Truth**
-----------------------------------------------------------

### **1\. Visual Contrastive Decoding (VCD)**

Standard VLMs often rely on "language priors" (e.g., if it sees a "missing hole," it assumes there must also be a "short" because they often appear together in training data).

*   **The Strategy:** During inference, we run a dual-stream process. We contrast the output of the original image with a **distorted/blurred version** of the same image.
    
*   **The Impact:** If the model's prediction remains the same even when the image is blurred, it is "hallucinating" based on text patterns. By subtracting the blurred-image probability from the sharp-image probability, we force the model to rely only on what it **actually sees** in the pixels.
    

### **2\. Negative Sampling & "Hard-Negative" Mining**

Generic models are trained to always provide an answer. In a factory, the answer is often "Everything is normal."

*   **The Strategy:** During fine-tuning, 30% of the dataset consists of **Negative Samples** (clean PCBs). We include "trick" questions like _"Where is the scratch?"_ for a clean board.
    
*   **The Impact:** The model is explicitly trained to output a specific token with high confidence, rather than trying to satisfy the user by finding a non-existent issue.
    

### **3\. Grounded-Contrastive Loss (Training-Time)**

We modify the loss function used during training to penalize the model for "unfounded" claims.

*   **The Strategy:** We implement a **Visual Grounding (VG) Loss**. If the model mentions a "Short" but its internal attention weights aren't focused on the bounding box coordinates associated with that short, the loss increases.
    
*   **The Impact:** This creates a mathematical "tether" between the text the model speaks and the pixels it looks at, making it physically difficult for the model to hallucinate a label without corresponding visual evidence.
    

### **4\. Self-Correction via Uncertainty-Guided Re-Attention**

Before showing the result to the inspector, the model performs a "double-check."

*   **The Strategy:** If the model's "Token Entropy" (uncertainty) is high when naming a defect, it triggers a **Visual Re-attention** cycle. It "zooms in" on the area it just described and re-processes that patch at 2x resolution.
    
*   **The Impact:** This acts as an automated "second opinion," where the model verifies its own claim before committing it to the inspection log.

| Mitigation Layer | Technique | Industrial Benefit |
| :--- | :--- | :--- |
| **Inference** | VCD (Contrastive) | Prevents "guessing" based on common defect patterns. |
| **Fine-Tuning** | Hard-Negative Mining | Eliminates false positives on high-quality clean boards. |
| **Validation** | Uncertainty Check | Triggers a sub-patch "re-scan" if confidence is low. |
---

**(E) Training Plan: From Bounding Boxes to Dialogue**
------------------------------------------------------

### **Stage 1: Automated QA Pair Generation (Data Synthesis)**

Since we have no manual annotations, we use the 50k bounding boxes to programmatically build a massive "Conversation" dataset.

*   **The Strategy:** We use "Instruction Templates" to convert CSV/XML data into natural language.
    
    *   **Locating:** _"Where is the \[defect\_type\]?"_ → _"The \[defect\_type\] is at \[x,y,w,h\]."_
        
    *   **Existence:** _"Is there any \[defect\_type\] on this board?"_ → _"Yes, I found one at \[x,y\]."_
        
    *   **Counting:** _"How many defects are present?"_ → _"There are \[n\] defects total."_
        
*   **The Goal:** Create a 500,000-sample instructional dataset (10 questions per image) to teach the model the relationship between words and coordinates.
    

### **Stage 2: Vision-Language Alignment (Pre-training)**

In this stage, we align the "Eyes" (Vision Encoder) with the "Brain" (LLM) using the PCB data.

*   **The Strategy:** We freeze the Vision Encoder and the LLM, and only train the **Projection Layer** (the bridge).
    
*   **The Goal:** Teach the model how to interpret the specific textures of green solder masks, copper traces, and SMD components without destroying the model's general intelligence.
    

### **Stage 3: Supervised Fine-Tuning (SFT) with LoRA**

This is the "specialization" phase where the model learns industrial reasoning.

*   **The Strategy:** We use **LoRA (Low-Rank Adaptation)** to unfreeze small parts of the LLM and the Vision Encoder.
    
*   **The Mix:** We train on a balanced diet of:
    
    1.  **Positive Samples:** PCBs with defects.
        
    2.  **Negative Samples:** Clean boards to prevent hallucinations.
        
    3.  **Adversarial Samples:** Images with "simulated" noise or different lighting to ensure the model isn't just memorizing one factory setup.

| Training Stage | Component Focus | Data Volume | Target Metric |
| :--- | :--- | :--- | :--- |
| **Alignment** | Projection Head | 50,000 Images | Feature Overlap |
| **SFT (LoRA)** | LLM + Encoder | 500,000 QA Pairs | Instruction Following |
| **Constraint**| Logit Processor | All Samples | Structured JSON Output |
---

**(F) Validation: Industrial Reliability Metrics**
--------------------------------------------------

### **1\. Localization Precision (Spatial Accuracy)**

To ensure the VLM isn't just "roughly" pointing at defects, we use **Intersection over Union (IoU)** and **mAP**.

*   **Metric:** **mAP@.50:.95** (Mean Average Precision across IoU thresholds from 0.5 to 0.95).
    
*   **Target:** Higher mAP at strict thresholds (0.75+) indicates the model’s coordinate tokens are mathematically aligned with the real-world pixels of the semiconductor traces.
    
*   **VLM Specifics:** We evaluate "Visual Grounding" accuracy—checking if the tokens generated by the text decoder match the ground-truth box area.
    

### **2\. Counting Accuracy (Enumeration)**

Inspectors often ask, _"How many \[defect\_type\] are there?"_ Errors here lead to incorrect batch rejection.

*   **Metric:** **Mean Absolute Error (MAE)** and **Accuracy**.
    
    *   $MAE = \\frac{1}{n} \\sum |Predicted\\\_Count - Actual\\\_Count|$
        
*   **Target:** MAE < 0.1 on critical defects (like missing holes). This validates that the VLM is successfully "scanning" the board and not missing items in high-density areas.
    

### **3\. Hallucination Rate (Truthfulness)**

This is the "Trust Metric" for the factory floor. We use **POPE (Polling-based Object Probing Evaluation)**.

*   **Metric:** **False Positive Rate (FPR)** on Negative Samples.
    
*   **Validation Method:** We prompt the model about defects that are **not** present on a clean board.
    
*   **Target:** < 1% Hallucination Rate. If the model claims a defect exists on a clean board, it is a "critical failure" in the validation log.

| Validation Dimension | Metric | Success Threshold |
| :--- | :--- | :--- |
| **Localization** | mAP@.50 | > 0.85 |
| **Counting** | MAE | < 0.1 |
| **Truthfulness** | Hallucination Rate (FPR) | < 1.0% |
| **Speed** | End-to-End Latency | < 2.0 Seconds |
---

