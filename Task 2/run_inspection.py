
import cv2
from ultralytics import YOLO
import argparse

def inspect(img_path, model_path):
    # Load your custom trained model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(img_path, conf=0.25)[0]
    
    # Print findings
    print(f"Found {len(results.boxes)} defects:")
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = results.names[cls]
        print(f" - {name} (Conf: {conf:.2f})")
        
    # Save visual
    results.save("inspection_output.jpg")
    print("Saved to inspection_output.jpg")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image")
    parser.add_argument("--model", default="yolo_pcb_model.pt", help="Path to .pt file")
    args = parser.parse_args()
    inspect(args.img, args.model)
