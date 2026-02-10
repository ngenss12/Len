import torch
import torch.nn as nn
import torch.optim as optim
from model import ResNet50WithRT
from dataload import OpenSARShipDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def inference_on_test_data(model_path, test_data_path, device='cuda'):
    """
    Perform inference on test data collected from real SAR-AIS integration
    """
    
    # Load model
    checkpoint = torch.load(model_path)
    model = ResNet50WithRT(rt_dim=8, num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    test_dataset = OpenSARShipDataset(root_dir=test_data_path)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Class names
    class_names = ['Bulk Carrier', 'Container Ship', 'Fishing', 'Tanker']
    
    # Inference
    results = []
    
    with torch.no_grad():
        for batch in test_loader:
            image = batch['image'].to(device)
            RT = batch['rt'].to(device)
            img_id = batch['img_id'][0]
            
            # Predict
            output = model(image, RT)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(1).item()
            confidence = probs[0, pred_class].item()
            
            results.append({
                'img_id': img_id,
                'predicted_class': class_names[pred_class],
                'confidence': confidence,
                'probabilities': probs[0].cpu().numpy()
            })
            
            print(f"Image: {img_id} | "
                  f"Predicted: {class_names[pred_class]} | "
                  f"Confidence: {confidence:.4f}")
    
    return results

# Usage
# results = inference_on_test_data('resnet50_ship_classifier.pth', '/path/to/test/data')