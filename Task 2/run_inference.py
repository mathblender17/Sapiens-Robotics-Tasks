import cv2
import argparse
import os
from ultralytics import YOLO

def analyze_pcb_quality(image_path, model_path="yolo_pcb_model.pt"):
    # 1. Load the fine-tuned model
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights not found at {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 2. Analyze the input image
    # We use a standard confidence threshold of 0.25
    results = model.predict(image_path, conf=0.25, verbose=False)[0]
    
    img = cv2.imread(image_path)
    h_img, w_img, _ = img.shape
    
    print(f"\n🚀 PCB Inspection Report: {os.path.basename(image_path)}")
    print("-" * 75)
    print(f"{'Defect Type':<20} | {'Conf':<6} | {'Center (X,Y)':<15} | {'Severity'}")
    print("-" * 75)
    
    # 3. Detect, Localize, and Classify
    if len(results.boxes) == 0:
        print("✅ No defects detected. Product passed inspection.")
    else:
        for box in results.boxes:
            # Get coordinates (top-left, bottom-right)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 4. Output (x, y) pixel coordinates of defect centers
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # Get class name and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = results.names[cls_id]
            
            # 4. Severity Assessment Logic
            # 'missing_component' is critical for board function
            if label == 'missing_component':
                severity = "CRITICAL"
                color = (0, 0, 255) # Red
            # 'scratch' or 'discoloration' may be cosmetic or medium risk
            elif label == 'scratch' and conf > 0.7:
                severity = "HIGH"
                color = (0, 165, 255) # Orange
            else:
                severity = "MEDIUM"
                color = (0, 255, 255) # Yellow

            # Print results to console as required
            print(f"{label:<20} | {conf:.2f}  | ({center_x:>4}, {center_y:>4})   | {severity}")

            # Draw bounding boxes for GitHub submission requirement
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label} {severity}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the localized/annotated image
    output_path = "inspection_result.jpg"
    cv2.imwrite(output_path, img)
    print("-" * 75)
    print(f"💾 Visual result with bounding boxes saved to: {output_path}\n")

if __name__ == "__main__":
    # For command line usage: python run_inference.py --image test.jpg
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the PCB image to inspect")
    parser.add_argument("--model", default="yolo_pcb_model.pt", help="Path to trained .pt weights")
    args = parser.parse_args()
    
    analyze_pcb_quality(args.image, args.model)