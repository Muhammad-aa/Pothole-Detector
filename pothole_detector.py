import argparse
from ultralytics import YOLO
import cv2

def detect_pothole(image_path, model_path='best.pt', save_output=False, show_result=False):
    """
    Detect potholes in an image using YOLO model
    
    Args:
        image_path (str): Path to the input image
        model_path (str): Path to the YOLO model file (.pt)
        save_output (bool): Whether to save the output image
        show_result (bool): Whether to display the result
    """
    try:
        # Load the model
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)
        
        # Perform detection
        print(f"Processing image: {image_path}")
        results = model(image_path)
        
        # Get the first result
        result = results[0]
        
        # Show the result if requested
        if show_result:
            result.show()
            print("Displaying result window. Close it to continue...")
        
        # Save the result if requested
        if save_output:
            # Generate output filename
            import os
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = f"{name}_detected{ext}"
            
            # Save the annotated image
            result.save(filename=output_path)
            print(f"Result saved to: {output_path}")
        
        # Print detection summary
        if hasattr(result, 'boxes') and result.boxes is not None:
            num_detections = len(result.boxes)
            print(f"\nDetection Summary:")
            print(f"- Found {num_detections} pothole(s)")
            
            # Print confidence scores for each detection
            if num_detections > 0:
                print("- Confidence scores:")
                for i, box in enumerate(result.boxes):
                    conf = box.conf[0].item()
                    print(f"  Pothole {i+1}: {conf:.2%}")
        else:
            print("No potholes detected.")
            
        return result
        
    except Exception as e:
        print(f"Error during detection: {e}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect potholes in images using YOLO')
    parser.add_argument('image', type=str, help='Path to the input image')
    parser.add_argument('--model', '-m', type=str, default='best.pt', 
                       help='Path to the model file (default: best.pt)')
    parser.add_argument('--save', '-s', action='store_true',
                       help='Save the output image with detections')
    parser.add_argument('--show', action='store_true',
                       help='Display the result image')
    parser.add_argument('--output', '-o', type=str,
                       help='Custom output filename (default: [input_name]_detected.[ext])')
    
    args = parser.parse_args()
    
    # Call the detection function
    detect_pothole(
        image_path=args.image,
        model_path=args.model,
        save_output=args.save or args.output is not None,
        show_result=args.show
    )

if __name__ == "__main__":
    main()