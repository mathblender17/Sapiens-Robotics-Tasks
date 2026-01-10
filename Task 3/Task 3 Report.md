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
