# Configuration file for AI Dermatological Diagnosis System

# Model Configuration
MODEL_CONFIG = {
    "num_classes": 7,
    "disease_classes": [
        "melanoma",
        "melanocytic_nevus", 
        "basal_cell_carcinoma",
        "actinic_keratosis",
        "benign_keratosis",
        "dermatofibroma",
        "vascular_lesion"
    ],
    "image_size": (224, 224),
    "backbone": "efficientnet_b7",  # or "resnet50" as fallback
    "text_model": "bert-base-uncased"
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "num_epochs": 50,
    "patience": 5,
    "device": "auto"  # "cuda", "cpu", or "auto"
}

# Web Application Configuration
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
}

# Interpretability Configuration
INTERPRETABILITY_CONFIG = {
    "enable_gradcam": True,
    "enable_attention_maps": True,
    "enable_concept_attribution": True,
    "enable_natural_language": True,
    "confidence_threshold_high": 0.8,
    "confidence_threshold_low": 0.6
}

# Clinical Decision Support
CLINICAL_CONFIG = {
    "high_risk_diseases": ["melanoma", "basal_cell_carcinoma"],
    "urgent_referral_threshold": 0.7,
    "routine_monitoring_threshold": 0.3,
    "enable_recommendations": True,
    "include_patient_education": True
}

# Evaluation Metrics
EVALUATION_CONFIG = {
    "target_accuracy": 0.95,
    "target_inference_time": 2.0,  # seconds
    "target_interpretability_score": 0.8,
    "enable_per_class_metrics": True,
    "enable_confidence_analysis": True
}

# Data Processing
DATA_CONFIG = {
    "demo_dataset_size": 1000,
    "train_test_split": 0.8,
    "validation_split": 0.1,
    "random_seed": 42,
    "enable_data_augmentation": True
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "enable_file_logging": True,
    "log_file": "dermatology_ai.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}
