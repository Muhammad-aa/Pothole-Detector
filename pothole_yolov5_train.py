"""
Pothole Detection with YOLOv5
Dataset: 665 images (70/20/10 train/valid/test split)
"""

# ==================== INSTALLATION INSTRUCTIONS ====================
# Run these commands in your terminal FIRST (one time setup):
#
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install ultralytics
# pip install opencv-python pillow matplotlib pyyaml
#
# Note: If you have CUDA 12.x, use cu121 instead of cu118
# ===================================================================

import os
from pathlib import Path
import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# ==================== CONFIGURATION ====================

# Update this to YOUR dataset path
DATASET_PATH = r"C:\Users\.."

# Training parameters
IMG_SIZE = 640          # Image size for training
BATCH_SIZE = 16         # Reduce to 8 if you get memory errors
EPOCHS = 100            # Training epochs (can stop early if needed)
DEVICE = 0              # 0 = first GPU, 'cpu' for CPU training

# Model selection (choose one):
MODEL_TYPE = 'yolov5s'  # Small - fast training, good for testing
# MODEL_TYPE = 'yolov5m'  # Medium - better accuracy, slower
# MODEL_TYPE = 'yolov5l'  # Large - best accuracy, slowest


# ==================== STEP 1: CREATE DATASET CONFIG ====================

def create_dataset_yaml():
    """Create dataset.yaml file for YOLOv5"""
    
    dataset_config = f"""
# Pothole Detection Dataset Configuration

# Paths
path: {DATASET_PATH}
train: images/train
val: images/valid
test: images/test

# Classes
nc: 1  # number of classes
names: ['pothole']  # class names
"""
    
    yaml_path = os.path.join(DATASET_PATH, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(dataset_config.strip())
    
    print(f"✓ Dataset config created: {yaml_path}")
    return yaml_path


# ==================== STEP 2: VERIFY DATASET ====================

def verify_dataset(dataset_path):
    """Check if dataset structure is correct"""
    
    print("\n=== Verifying Dataset ===")
    
    splits = ['train', 'valid', 'test']
    stats = {}
    
    for split in splits:
        img_dir = os.path.join(dataset_path, 'images', split)
        label_dir = os.path.join(dataset_path, 'labels', split)
        
        # Check if directories exist
        if not os.path.exists(img_dir):
            print(f" Warning: {img_dir} not found!")
            continue
            
        # Count files
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Check for labels directory
        if os.path.exists(label_dir):
            labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            stats[split] = {'images': len(images), 'labels': len(labels)}
        else:
            print(f" Warning: {label_dir} not found!")
            print(f"   Looking for labels in same folder as images...")
            # Check if labels are in same folder as images
            labels = [f for f in os.listdir(img_dir) if f.endswith('.txt')]
            stats[split] = {'images': len(images), 'labels': len(labels)}
    
    # Print statistics
    print("\nDataset Statistics:")
    for split, counts in stats.items():
        print(f"  {split:6s}: {counts['images']:3d} images, {counts['labels']:3d} labels")
    
    total_images = sum(s['images'] for s in stats.values())
    print(f"\n  Total: {total_images} images")
    
    return stats


# ==================== STEP 3: CHECK GPU ====================

def check_gpu():
    """Verify GPU is available"""
    
    print("\n=== GPU Check ===")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f" GPU Available: {gpu_name}")
        print(f" GPU Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print(" No GPU detected! Training will use CPU (very slow)")
        print("  Make sure you have:")
        print("  1. NVIDIA GPU drivers installed")
        print("  2. PyTorch with CUDA support installed")
        return False


# ==================== STEP 4: TRAIN MODEL ====================

def train_model(yaml_path, model_type=MODEL_TYPE):
    """Train YOLOv5 model on pothole dataset"""
    
    print(f"\n=== Training {model_type.upper()} ===")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMG_SIZE}")
    
    # Load pre-trained model
    model = YOLO(f'{model_type}.pt')
    
    # Train
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='pothole_detection',
        name='train',
        patience=20,          # Early stopping patience
        save=True,            # Save checkpoints
        plots=True,           # Generate plots
        verbose=True
    )
    
    print("\n Training Complete!")
    print(f" Best model saved to: /run/pothole_detection/train/weights/best.pt")
    
    return model, results


# ==================== STEP 5: EVALUATE MODEL ====================

def evaluate_model(model, yaml_path):
    """Evaluate model on validation set"""
    
    print("\n=== Evaluating Model ===")
    
    # Run validation
    metrics = model.val(data=yaml_path)
    
    print(f"\nValidation Results:")
    print(f"  mAP@0.5: {metrics.box.map50:.3f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"  Precision: {metrics.box.mp:.3f}")
    print(f"  Recall: {metrics.box.mr:.3f}")
    
    return metrics


# ==================== STEP 6: TEST ON IMAGES ====================

def test_on_images(model, test_dir, num_samples=5):
    """Run inference on test images and display results"""
    
    print("\n=== Testing on Sample Images ===")
    
    # Create output directory
    output_dir = 'pothole_detection/test_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png'))][:num_samples]
    
    if not test_images:
        print("No test images found!")
        return
    
    # Run inference
    results = model(test_images)
    
    # Display results
    for i, result in enumerate(results):
        # Plot
        plt.figure(figsize=(10, 8))
        img = result.plot()  # Get annotated image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Test Image {i+1} - Pothole Detection")
        plt.tight_layout()
        plt.savefig(f'{output_dir}/test_result_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        # Print detections
        boxes = result.boxes
        print(f"\nImage {i+1}: {len(boxes)} pothole(s) detected")
        for j, box in enumerate(boxes):
            conf = box.conf[0].item()
            print(f"  Pothole {j+1}: Confidence {conf:.2%}")
    
    print(f"\n✓ Test results saved to: {output_dir}/")


# ==================== STEP 7: SAVE FOR DEPLOYMENT ====================

def export_model(model):
    """Export model to different formats"""
    
    print("\n=== Exporting Model ===")
    
    # Export to ONNX (for deployment)
    model.export(format='onnx')
    print("✓ Model exported to ONNX format")
    
    # You can also export to other formats:
    # model.export(format='torchscript')  # TorchScript
    # model.export(format='tflite')       # TensorFlow Lite
    # model.export(format='engine')       # TensorRT (fastest inference)


# ==================== MAIN EXECUTION ====================

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("POTHOLE DETECTION - YOLOV5 TRAINING")
    print("=" * 60)
    
    # Step 1: Check GPU
    check_gpu()
    
    # Step 2: Verify dataset
    stats = verify_dataset(DATASET_PATH)
    
    if not stats:
        print("\n Dataset not found or incomplete!")
        print("Please check your dataset path and structure.")
        return
    
    # Step 3: Create dataset config
    yaml_path = create_dataset_yaml()
    
    # Step 4: Train model
    model, results = train_model(yaml_path)
    
    # Step 5: Evaluate model
    metrics = evaluate_model(model, yaml_path)
    
    # Step 6: Test on sample images
    test_dir = os.path.join(DATASET_PATH, 'images', 'test')
    if os.path.exists(test_dir):
        test_on_images(model, test_dir, num_samples=5)
    
    # Step 7: Export model
    export_model(model)
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print("\nYour trained model is ready!")
    print(f"Location: pothole_detection/train/weights/best.pt")
    print("\nTo use it:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('pothole_detection/train/weights/best.pt')")
    print("  results = model('path/to/your/image.jpg')")


# ==================== UTILITY FUNCTIONS ====================

def resume_training():
    """Resume training from last checkpoint"""
    model = YOLO('pothole_detection/train/weights/last.pt')
    model.train(resume=True)


def predict_single_image(image_path):
    """Quick prediction on a single image"""
    model = YOLO('pothole_detection/train/weights/best.pt')
    results = model(image_path)
    results[0].show()  # Display result
    return results


def predict_video(video_path, output_path='output_video.mp4'):
    """Run detection on video"""
    model = YOLO('pothole_detection/train/weights/best.pt')
    results = model(video_path, stream=True)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for result in results:
        frame = result.plot()
        out.write(frame)
    
    cap.release()
    out.release()
    print(f" Video saved to: {output_path}")


# ==================== RUN ====================

if __name__ == "__main__":
    main()
    
    # Uncomment to test individual functions:
    # predict_single_image('path/to/test/image.jpg')
    # predict_video('path/to/video.mp4')