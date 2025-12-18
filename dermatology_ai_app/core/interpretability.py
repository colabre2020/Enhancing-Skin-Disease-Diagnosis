"""
Interpretability Layer
=====================

Implements explainable AI mechanisms for dermatological diagnosis.
Provides visual explanations, attention maps, and concept attribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional, Any
import io
import base64

# Optional imports
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import lime
    from lime import lime_image
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping implementation
    """
    
    def __init__(self, model: nn.Module, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate Class Activation Map"""
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam.numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()


class AttentionVisualizer:
    """
    Visualize attention maps from the multi-modal AI engine
    """
    
    @staticmethod
    def visualize_attention(attention_weights: torch.Tensor, 
                          image: Image.Image,
                          save_path: Optional[str] = None) -> Image.Image:
        """
        Visualize attention weights on the input image
        """
        # Convert attention weights to numpy
        if isinstance(attention_weights, torch.Tensor):
            attention = attention_weights.detach().cpu().numpy()
        else:
            attention = attention_weights
        
        # Ensure attention is 2D
        if attention.ndim > 2:
            attention = attention.squeeze()
            if attention.ndim > 2:
                attention = attention[0]  # Take first head/layer
        
        # Resize attention to match image size
        image_array = np.array(image)
        h, w = image_array.shape[:2]
        
        if HAS_CV2:
            attention_resized = cv2.resize(attention, (w, h))
        else:
            # Simple resizing without cv2
            attention_resized = np.array(Image.fromarray(attention).resize((w, h)))
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Attention overlay
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(attention_resized, cmap='jet', alpha=0.5)
        plt.title("Attention Map Overlay")
        plt.axis('off')
        
        # Save or return
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close()
        
        return result_image


class ConceptAttributor:
    """
    Attribute predictions to interpretable concepts
    """
    
    def __init__(self):
        # Define dermatological concepts
        self.visual_concepts = [
            "asymmetry", "border_irregularity", "color_variation", 
            "diameter", "evolution", "texture", "surface_features",
            "pigmentation", "inflammation", "scaling"
        ]
        
        self.clinical_concepts = [
            "patient_age", "lesion_location", "skin_type", 
            "family_history", "symptoms", "duration"
        ]
    
    def analyze_visual_concepts(self, image: Image.Image, 
                              attention_map: np.ndarray) -> Dict[str, float]:
        """
        Analyze visual concepts present in the image
        This is a simplified implementation - in practice, you'd use trained concept detectors
        """
        concepts = {}
        
        # Convert image to array
        img_array = np.array(image)
        
        # Simple heuristic-based concept detection
        # In practice, these would be learned from data
        
        # Asymmetry - measure symmetry
        height, width = img_array.shape[:2]
        left_half = img_array[:, :width//2]
        right_half = np.fliplr(img_array[:, width//2:])
        
        if left_half.shape == right_half.shape:
            asymmetry_score = np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0
            concepts["asymmetry"] = min(asymmetry_score, 1.0)
        else:
            concepts["asymmetry"] = 0.5
        
        # Color variation - measure color diversity
        rgb_std = np.std(img_array, axis=(0, 1))
        color_variation = np.mean(rgb_std) / 255.0
        concepts["color_variation"] = min(color_variation, 1.0)
        
        # Border irregularity - use attention map as proxy
        border_score = np.std(attention_map) if attention_map is not None else 0.5
        concepts["border_irregularity"] = min(border_score, 1.0)
        
        # Add other concepts with default values
        for concept in self.visual_concepts:
            if concept not in concepts:
                concepts[concept] = np.random.uniform(0.2, 0.8)  # Placeholder
        
        return concepts
    
    def analyze_clinical_concepts(self, clinical_history: str, 
                                patient_metadata: Dict) -> Dict[str, float]:
        """
        Analyze clinical concepts from text and metadata
        """
        concepts = {}
        
        # Patient age impact
        age = patient_metadata.get("age", 50)
        age_risk = min(max(age - 40, 0) / 40.0, 1.0)  # Higher risk with age
        concepts["patient_age"] = age_risk
        
        # Location-based risk
        high_risk_locations = ["face", "neck", "scalp", "ears", "lips"]
        location = patient_metadata.get("lesion_location", "").lower()
        location_risk = 0.8 if any(loc in location for loc in high_risk_locations) else 0.3
        concepts["lesion_location"] = location_risk
        
        # Skin type risk
        skin_type = patient_metadata.get("skin_type", "type_III")
        if "type_I" in skin_type or "type_II" in skin_type:
            skin_risk = 0.8  # Higher risk for fair skin
        else:
            skin_risk = 0.4
        concepts["skin_type"] = skin_risk
        
        # Symptom analysis from clinical history
        symptom_keywords = ["pain", "bleeding", "itching", "growth", "change"]
        symptom_score = sum(1 for keyword in symptom_keywords if keyword in clinical_history.lower())
        concepts["symptoms"] = min(symptom_score / len(symptom_keywords), 1.0)
        
        # Add other concepts
        for concept in self.clinical_concepts:
            if concept not in concepts:
                concepts[concept] = np.random.uniform(0.2, 0.8)  # Placeholder
        
        return concepts


class ExplanationGenerator:
    """
    Generate natural language explanations for diagnostic decisions
    """
    
    def __init__(self):
        self.concept_attributor = ConceptAttributor()
        self.disease_descriptions = {
            "melanoma": "A serious form of skin cancer that develops in melanocytes",
            "melanocytic_nevus": "A benign mole or nevus composed of melanocytes", 
            "basal_cell_carcinoma": "The most common type of skin cancer",
            "actinic_keratosis": "A precancerous skin condition caused by sun damage",
            "benign_keratosis": "A non-cancerous skin growth",
            "dermatofibroma": "A benign skin tumor",
            "vascular_lesion": "A lesion involving blood vessels in the skin"
        }
    
    def generate_explanation(self, predictions: Dict[str, float],
                           confidence: Dict[str, float],
                           image: Image.Image,
                           clinical_history: str,
                           patient_metadata: Dict,
                           attention_map: Optional[np.ndarray] = None) -> str:
        """
        Generate comprehensive explanation for the diagnosis
        """
        # Get top prediction
        top_disease = max(predictions.keys(), key=lambda k: predictions[k])
        top_confidence = predictions[top_disease]
        
        # Analyze concepts
        visual_concepts = self.concept_attributor.analyze_visual_concepts(image, attention_map)
        clinical_concepts = self.concept_attributor.analyze_clinical_concepts(
            clinical_history, patient_metadata
        )
        
        # Start building explanation
        explanation = f"**Diagnostic Assessment:**\n\n"
        explanation += f"Primary diagnosis: **{top_disease.replace('_', ' ').title()}** "
        explanation += f"(confidence: {top_confidence:.1%})\n\n"
        
        explanation += f"{self.disease_descriptions.get(top_disease, 'Unknown condition')}\n\n"
        
        # Visual features analysis
        explanation += "**Visual Features Analysis:**\n"
        significant_visual = {k: v for k, v in visual_concepts.items() if v > 0.6}
        if significant_visual:
            for concept, score in significant_visual.items():
                explanation += f"- {concept.replace('_', ' ').title()}: {score:.1%} presence\n"
        else:
            explanation += "- No significant concerning visual features detected\n"
        
        explanation += "\n**Clinical Factors:**\n"
        significant_clinical = {k: v for k, v in clinical_concepts.items() if v > 0.6}
        if significant_clinical:
            for concept, score in significant_clinical.items():
                explanation += f"- {concept.replace('_', ' ').title()}: {score:.1%} risk factor\n"
        else:
            explanation += "- No significant clinical risk factors identified\n"
        
        # Confidence assessment
        overall_conf = confidence.get("overall_confidence", 0.5)
        explanation += f"\n**Confidence Assessment:**\n"
        if overall_conf > 0.8:
            explanation += "High confidence diagnosis. The visual and clinical features strongly support this assessment.\n"
        elif overall_conf > 0.6:
            explanation += "Moderate confidence diagnosis. Some features support this assessment, but additional evaluation may be beneficial.\n"
        else:
            explanation += "Low confidence diagnosis. The features are ambiguous and specialist consultation is recommended.\n"
        
        # Recommendations
        explanation += "\n**Recommendations:**\n"
        if top_disease == "melanoma" or overall_conf > 0.7:
            explanation += "- Immediate dermatologist referral recommended\n"
            explanation += "- Consider biopsy for definitive diagnosis\n"
        elif "carcinoma" in top_disease:
            explanation += "- Dermatologist consultation within 2-4 weeks\n"
            explanation += "- Monitor for changes in size, color, or symptoms\n"
        else:
            explanation += "- Routine monitoring recommended\n"
            explanation += "- Return if lesion changes or symptoms develop\n"
        
        explanation += "- Continue regular skin examinations\n"
        explanation += "- Sun protection measures advised\n"
        
        return explanation


class InterpretabilityEngine:
    """
    Main interpretability engine coordinating all explanation components
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_visualizer = AttentionVisualizer()
        self.explanation_generator = ExplanationGenerator()
        
        # Initialize GradCAM for the vision encoder
        # Note: This assumes the vision encoder backbone is accessible
        self.gradcam = None
        try:
            # Try to set up GradCAM for the last convolutional layer
            if hasattr(model, 'vision_encoder') and hasattr(model.vision_encoder, 'backbone'):
                self.gradcam = GradCAM(model.vision_encoder, 'backbone.features')
        except Exception as e:
            print(f"Warning: Could not initialize GradCAM: {e}")
    
    def generate_comprehensive_explanation(self, 
                                        diagnostic_output,
                                        image: Image.Image,
                                        clinical_history: str,
                                        patient_metadata: Dict) -> Dict[str, Any]:
        """
        Generate comprehensive explanation including all interpretability components
        """
        explanations = {}
        
        # Generate attention visualization
        if 'visual_attention' in diagnostic_output.explanations:
            attention_viz = self.attention_visualizer.visualize_attention(
                diagnostic_output.explanations['visual_attention'],
                image
            )
            explanations['attention_visualization'] = attention_viz
        
        # Generate GradCAM if available
        if self.gradcam is not None:
            try:
                top_class_idx = max(range(len(diagnostic_output.predictions)), 
                                  key=lambda i: list(diagnostic_output.predictions.values())[i])
                
                # Prepare input tensor
                transform = self.model.image_transform
                input_tensor = transform(image).unsqueeze(0)
                
                cam = self.gradcam.generate_cam(input_tensor, top_class_idx)
                explanations['gradcam'] = cam
                
                # Create GradCAM visualization
                gradcam_viz = self.attention_visualizer.visualize_attention(
                    cam, image
                )
                explanations['gradcam_visualization'] = gradcam_viz
                
            except Exception as e:
                print(f"Warning: GradCAM generation failed: {e}")
        
        # Generate natural language explanation
        attention_map = diagnostic_output.explanations.get('visual_attention')
        if isinstance(attention_map, torch.Tensor):
            attention_map = attention_map.detach().cpu().numpy()
        
        text_explanation = self.explanation_generator.generate_explanation(
            diagnostic_output.predictions,
            diagnostic_output.confidence_scores,
            image,
            clinical_history,
            patient_metadata,
            attention_map
        )
        explanations['text_explanation'] = text_explanation
        
        # Add concept analysis
        concept_attributor = ConceptAttributor()
        visual_concepts = concept_attributor.analyze_visual_concepts(image, attention_map)
        clinical_concepts = concept_attributor.analyze_clinical_concepts(
            clinical_history, patient_metadata
        )
        
        explanations['visual_concepts'] = visual_concepts
        explanations['clinical_concepts'] = clinical_concepts
        
        return explanations
    
    def cleanup(self):
        """Clean up resources"""
        if self.gradcam:
            self.gradcam.cleanup()


# Utility functions
def save_explanation_report(explanations: Dict[str, Any], 
                          diagnostic_output,
                          output_path: str):
    """
    Save a comprehensive explanation report as HTML
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dermatological Diagnosis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f8ff; padding: 15px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .prediction {{ font-weight: bold; color: #2c5aa0; }}
            .high-confidence {{ color: #228b22; }}
            .medium-confidence {{ color: #ffa500; }}
            .low-confidence {{ color: #dc143c; }}
            .concept {{ margin: 5px 0; }}
            .concept-bar {{ display: inline-block; height: 10px; background-color: #4CAF50; margin-left: 10px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AI-Powered Dermatological Diagnosis Report</h1>
            <p>Generated on {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S') if hasattr(torch, 'datetime') else 'N/A'}</p>
        </div>
        
        <div class="section">
            <h2>Diagnostic Results</h2>
    """
    
    # Add predictions
    for disease, confidence in diagnostic_output.predictions.items():
        conf_class = "high-confidence" if confidence > 0.7 else "medium-confidence" if confidence > 0.4 else "low-confidence"
        html_content += f'<div class="prediction {conf_class}">{disease.replace("_", " ").title()}: {confidence:.1%}</div>'
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Clinical Explanation</h2>
            <div style="white-space: pre-line;">
    """
    
    html_content += explanations.get('text_explanation', 'No explanation available.')
    
    html_content += """
            </div>
        </div>
        
        <div class="section">
            <h2>Visual Concept Analysis</h2>
    """
    
    # Add visual concepts
    for concept, score in explanations.get('visual_concepts', {}).items():
        bar_width = int(score * 200)
        html_content += f'''
            <div class="concept">
                {concept.replace("_", " ").title()}: {score:.1%}
                <div class="concept-bar" style="width: {bar_width}px;"></div>
            </div>
        '''
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>Clinical Risk Factors</h2>
    """
    
    # Add clinical concepts
    for concept, score in explanations.get('clinical_concepts', {}).items():
        bar_width = int(score * 200)
        html_content += f'''
            <div class="concept">
                {concept.replace("_", " ").title()}: {score:.1%}
                <div class="concept-bar" style="width: {bar_width}px;"></div>
            </div>
        '''
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Explanation report saved to: {output_path}")


if __name__ == "__main__":
    # Demo usage
    print("Interpretability Engine Demo")
    print("This module provides explainable AI capabilities for dermatological diagnosis.")
    print("Key features:")
    print("- GradCAM visualization")
    print("- Attention map visualization")  
    print("- Concept attribution")
    print("- Natural language explanations")
    print("- Comprehensive reporting")
