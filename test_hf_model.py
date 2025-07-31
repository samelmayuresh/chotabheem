from transformers import pipeline
from PIL import Image
import numpy as np

# Test the Hugging Face emotion detection model
def test_hf_emotion_model():
    try:
        print("🤖 Loading ViT Face Expression model...")
        
        # Load the model
        classifier = pipeline(
            "image-classification", 
            model="trpakov/vit-face-expression",
            device=-1  # Use CPU
        )
        
        print("✅ ViT Face Expression model loaded successfully!")
        
        # Create a test image (random noise for testing)
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Get predictions
        predictions = classifier(test_image)
        
        print("🎯 Test predictions:")
        for pred in predictions:
            print(f"  • {pred['label']}: {pred['score']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_hf_emotion_model()