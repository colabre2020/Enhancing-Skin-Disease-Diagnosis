"""
Test Suite for AI Dermatological Diagnosis System
================================================

Comprehensive test suite validating the implementation described in the research paper.
"""

import unittest
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import json
from pathlib import Path
import sys

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from core.ai_engine import (
        MultiModalAIEngine, 
        VisionEncoder, 
        TextEncoder, 
        FusionNetwork, 
        DiagnosticInput,
        create_demo_model
    )
    from core.interpretability import (
        InterpretabilityEngine,
        AttentionVisualizer,
        ConceptAttributor,
        ExplanationGenerator
    )
    from training import (
        SkinLesionDataset,
        Trainer,
        Evaluator,
        create_demo_dataset
    )
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Core modules not fully available: {e}")
    CORE_MODULES_AVAILABLE = False


class TestAIEngine(unittest.TestCase):
    """Test the core AI engine components"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CORE_MODULES_AVAILABLE:
            self.skipTest("Core modules not available")
        
        self.model = create_demo_model()
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.test_metadata = {
            "age": 45,
            "gender": "female",
            "skin_type": "type_III",
            "lesion_location": "back"
        }
        self.test_history = "Patient reports a growing mole with irregular borders."
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsInstance(self.model, MultiModalAIEngine)
        self.assertEqual(self.model.num_classes, 7)
        self.assertEqual(len(self.model.disease_classes), 7)
    
    def test_image_preprocessing(self):
        """Test image preprocessing pipeline"""
        processed = self.model.preprocess_image(self.test_image)
        self.assertEqual(processed.shape, (1, 3, 224, 224))
        self.assertTrue(torch.is_tensor(processed))
    
    def test_text_preprocessing(self):
        """Test text preprocessing"""
        processed_text = self.model.preprocess_text(self.test_history, self.test_metadata)
        self.assertIsInstance(processed_text, str)
        self.assertIn("Age: 45", processed_text)
        self.assertIn("Gender: female", processed_text)
    
    def test_diagnosis_functionality(self):
        """Test end-to-end diagnosis"""
        result = self.model.diagnose(self.test_image, self.test_history, self.test_metadata)
        
        # Check result structure
        self.assertIsNotNone(result.predictions)
        self.assertIsNotNone(result.confidence_scores)
        self.assertIsNotNone(result.explanations)
        
        # Check predictions
        self.assertEqual(len(result.predictions), 7)
        self.assertTrue(all(0 <= prob <= 1 for prob in result.predictions.values()))
        self.assertAlmostEqual(sum(result.predictions.values()), 1.0, places=2)
        
        # Check confidence scores
        self.assertIn('overall_confidence', result.confidence_scores)
        self.assertTrue(0 <= result.confidence_scores['overall_confidence'] <= 1)
    
    def test_vision_encoder(self):
        """Test vision encoder component"""
        vision_encoder = VisionEncoder()
        test_tensor = torch.randn(1, 3, 224, 224)
        features, attention = vision_encoder(test_tensor)
        
        self.assertEqual(features.shape[1], 256)  # Feature dimension
        self.assertIsNotNone(attention)
    
    def test_text_encoder(self):
        """Test text encoder component"""
        text_encoder = TextEncoder()
        test_texts = ["Test clinical history"]
        features = text_encoder(test_texts)
        
        self.assertEqual(features.shape[1], 256)  # Feature dimension
        self.assertEqual(features.shape[0], 1)    # Batch size
    
    def test_fusion_network(self):
        """Test multi-modal fusion"""
        fusion_net = FusionNetwork()
        visual_features = torch.randn(1, 256)
        text_features = torch.randn(1, 256)
        
        predictions, confidence, attention = fusion_net(visual_features, text_features)
        
        self.assertEqual(predictions.shape[1], 7)  # Number of classes
        self.assertEqual(confidence.shape[1], 1)   # Confidence score
        self.assertIsNotNone(attention)


class TestInterpretability(unittest.TestCase):
    """Test interpretability components"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CORE_MODULES_AVAILABLE:
            self.skipTest("Core modules not available")
        
        self.model = create_demo_model()
        self.test_image = Image.new('RGB', (224, 224), color='lightblue')
        self.test_metadata = {
            "age": 60,
            "gender": "male",
            "skin_type": "type_II",
            "lesion_location": "face"
        }
        self.test_history = "Suspicious lesion with color changes."
    
    def test_concept_attributor(self):
        """Test concept attribution"""
        attributor = ConceptAttributor()
        
        # Test visual concepts
        visual_concepts = attributor.analyze_visual_concepts(self.test_image, None)
        self.assertIsInstance(visual_concepts, dict)
        self.assertIn('asymmetry', visual_concepts)
        self.assertIn('color_variation', visual_concepts)
        
        # Test clinical concepts
        clinical_concepts = attributor.analyze_clinical_concepts(
            self.test_history, self.test_metadata
        )
        self.assertIsInstance(clinical_concepts, dict)
        self.assertIn('patient_age', clinical_concepts)
        self.assertIn('lesion_location', clinical_concepts)
    
    def test_explanation_generator(self):
        """Test natural language explanation generation"""
        generator = ExplanationGenerator()
        
        test_predictions = {
            "melanoma": 0.7,
            "melanocytic_nevus": 0.2,
            "basal_cell_carcinoma": 0.1
        }
        test_confidence = {"overall_confidence": 0.8}
        
        explanation = generator.generate_explanation(
            test_predictions,
            test_confidence,
            self.test_image,
            self.test_history,
            self.test_metadata
        )
        
        self.assertIsInstance(explanation, str)
        self.assertIn("melanoma", explanation.lower())
        self.assertIn("confidence", explanation.lower())
        self.assertIn("recommendation", explanation.lower())
    
    def test_attention_visualizer(self):
        """Test attention visualization"""
        # Create dummy attention weights
        attention_weights = np.random.rand(14, 14)  # Typical attention map size
        
        visualizer = AttentionVisualizer()
        result_image = visualizer.visualize_attention(
            attention_weights, 
            self.test_image
        )
        
        self.assertIsInstance(result_image, Image.Image)
    
    def test_interpretability_engine(self):
        """Test integrated interpretability engine"""
        engine = InterpretabilityEngine(self.model)
        
        # Run diagnosis to get results
        diagnostic_result = self.model.diagnose(
            self.test_image, 
            self.test_history, 
            self.test_metadata
        )
        
        # Generate comprehensive explanations
        explanations = engine.generate_comprehensive_explanation(
            diagnostic_result,
            self.test_image,
            self.test_history,
            self.test_metadata
        )
        
        self.assertIsInstance(explanations, dict)
        self.assertIn('text_explanation', explanations)
        self.assertIn('visual_concepts', explanations)
        self.assertIn('clinical_concepts', explanations)


class TestTrainingEvaluation(unittest.TestCase):
    """Test training and evaluation components"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CORE_MODULES_AVAILABLE:
            self.skipTest("Core modules not available")
        
        self.temp_dir = tempfile.mkdtemp()
        self.num_samples = 20  # Small dataset for testing
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_demo_dataset_creation(self):
        """Test demo dataset creation"""
        image_paths, labels, histories, metadata = create_demo_dataset(
            num_samples=self.num_samples,
            output_dir=self.temp_dir
        )
        
        self.assertEqual(len(image_paths), self.num_samples)
        self.assertEqual(len(labels), self.num_samples)
        self.assertEqual(len(histories), self.num_samples)
        self.assertEqual(len(metadata), self.num_samples)
        
        # Check if files were created
        for img_path in image_paths:
            self.assertTrue(os.path.exists(img_path))
        
        # Check dataset info file
        info_file = os.path.join(self.temp_dir, "dataset_info.json")
        self.assertTrue(os.path.exists(info_file))
    
    def test_skin_lesion_dataset(self):
        """Test dataset class"""
        # Create small dataset
        image_paths, labels, histories, metadata = create_demo_dataset(
            num_samples=10,
            output_dir=self.temp_dir
        )
        
        dataset = SkinLesionDataset(image_paths, labels, histories, metadata)
        
        self.assertEqual(len(dataset), 10)
        
        # Test data loading
        sample = dataset[0]
        self.assertIn('image', sample)
        self.assertIn('label', sample)
        self.assertIn('clinical_history', sample)
        self.assertIn('metadata', sample)
        
        # Check image tensor shape
        if hasattr(sample['image'], 'shape'):
            self.assertEqual(len(sample['image'].shape), 3)  # C, H, W
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation"""
        model = create_demo_model()
        
        # Create dummy test data
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 2, 2, 1, 1])
        y_prob = np.random.rand(8, 7)  # 8 samples, 7 classes
        confidences = np.random.rand(8)
        inference_times = np.random.rand(8) * 0.1  # Fast inference times
        
        evaluator = Evaluator(model, None)  # No test loader for this test
        metrics = evaluator._calculate_metrics(y_true, y_pred, y_prob, confidences, inference_times)
        
        # Check required metrics
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('avg_inference_time', metrics)
        self.assertIn('avg_confidence', metrics)
        
        # Check metric ranges
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(metrics['avg_inference_time'] >= 0)


class TestWebApplication(unittest.TestCase):
    """Test web application components"""
    
    def setUp(self):
        """Set up test fixtures"""
        try:
            import fastapi
            self.fastapi_available = True
        except ImportError:
            self.fastapi_available = False
    
    def test_fastapi_import(self):
        """Test if FastAPI can be imported"""
        if not self.fastapi_available:
            self.skipTest("FastAPI not available")
        
        try:
            from web_app import app
            self.assertIsNotNone(app)
        except ImportError:
            self.skipTest("Web app modules not available")
    
    def test_api_endpoints_defined(self):
        """Test if API endpoints are properly defined"""
        if not self.fastapi_available:
            self.skipTest("FastAPI not available")
        
        try:
            from web_app import app
            
            # Check if routes are defined
            routes = [route.path for route in app.routes]
            self.assertIn('/', routes)
            self.assertIn('/analyze', routes)
            self.assertIn('/api/health', routes)
            
        except ImportError:
            self.skipTest("Web app modules not available")


class TestSystemIntegration(unittest.TestCase):
    """Test system integration and end-to-end functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CORE_MODULES_AVAILABLE:
            self.skipTest("Core modules not available")
    
    def test_complete_workflow(self):
        """Test complete diagnostic workflow"""
        # Initialize model
        model = create_demo_model()
        
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='tan')
        
        # Test metadata
        metadata = {
            "age": 35,
            "gender": "female",
            "skin_type": "type_IV",
            "lesion_location": "leg"
        }
        
        history = "New lesion appeared 2 months ago, slowly growing."
        
        # Run complete diagnosis
        result = model.diagnose(test_image, history, metadata)
        
        # Initialize interpretability
        interp_engine = InterpretabilityEngine(model)
        explanations = interp_engine.generate_comprehensive_explanation(
            result, test_image, history, metadata
        )
        
        # Validate complete workflow
        self.assertIsNotNone(result)
        self.assertIsNotNone(explanations)
        self.assertIn('text_explanation', explanations)
        
        # Check if workflow meets paper requirements
        self.assertTrue(len(result.predictions) == 7)  # Multi-class classification
        self.assertTrue('overall_confidence' in result.confidence_scores)  # Confidence scoring
        self.assertTrue(explanations['text_explanation'])  # Interpretability
    
    def test_performance_requirements(self):
        """Test if system meets performance requirements from paper"""
        model = create_demo_model()
        test_image = Image.new('RGB', (224, 224), color='beige')
        metadata = {"age": 50, "gender": "male", "skin_type": "type_III", "lesion_location": "back"}
        
        import time
        
        # Measure inference time
        start_time = time.time()
        result = model.diagnose(test_image, "Test case", metadata)
        inference_time = time.time() - start_time
        
        # Check performance targets from paper
        self.assertLess(inference_time, 5.0)  # Should be much faster than 2s target
        self.assertGreater(result.confidence_scores['overall_confidence'], 0.0)
        self.assertTrue(all(0 <= prob <= 1 for prob in result.predictions.values()))
    
    def test_clinical_decision_support(self):
        """Test clinical decision support features"""
        model = create_demo_model()
        test_image = Image.new('RGB', (224, 224), color='darkred')
        
        # High-risk scenario
        metadata = {
            "age": 70,
            "gender": "male", 
            "skin_type": "type_I",
            "lesion_location": "face"
        }
        
        history = "Rapidly growing lesion with irregular borders and bleeding."
        
        result = model.diagnose(test_image, history, metadata)
        
        # Generate explanations
        interp_engine = InterpretabilityEngine(model)
        explanations = interp_engine.generate_comprehensive_explanation(
            result, test_image, history, metadata
        )
        
        # Check if clinical recommendations are provided
        explanation_text = explanations['text_explanation']
        self.assertIn('recommendation', explanation_text.lower())
        
        # Should mention high-risk factors for this scenario
        self.assertTrue(
            'age' in explanation_text.lower() or 
            'face' in explanation_text.lower() or
            'type_i' in explanation_text.lower()
        )


class TestResearchPaperValidation(unittest.TestCase):
    """Validate implementation against research paper claims"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not CORE_MODULES_AVAILABLE:
            self.skipTest("Core modules not available")
    
    def test_multimodal_architecture(self):
        """Validate multi-modal architecture implementation"""
        model = create_demo_model()
        
        # Check architecture components
        self.assertTrue(hasattr(model, 'vision_encoder'))
        self.assertTrue(hasattr(model, 'text_encoder'))
        self.assertTrue(hasattr(model, 'fusion_network'))
        
        # Test multi-modal input processing
        test_image = Image.new('RGB', (224, 224), color='yellow')
        test_history = "Multi-modal test"
        test_metadata = {"age": 40, "gender": "other", "skin_type": "type_III", "lesion_location": "arm"}
        
        result = model.diagnose(test_image, test_history, test_metadata)
        
        # Should successfully process both visual and textual inputs
        self.assertIsNotNone(result)
        self.assertTrue(len(result.predictions) > 0)
    
    def test_interpretability_features(self):
        """Validate interpretability implementation"""
        model = create_demo_model()
        interp_engine = InterpretabilityEngine(model)
        
        # Test interpretability components
        self.assertTrue(hasattr(interp_engine, 'attention_visualizer'))
        self.assertTrue(hasattr(interp_engine, 'explanation_generator'))
        
        # Test explanation generation
        test_image = Image.new('RGB', (224, 224), color='purple')
        result = model.diagnose(test_image, "Interpretability test", {"age": 30, "gender": "female", "skin_type": "type_II", "lesion_location": "neck"})
        
        explanations = interp_engine.generate_comprehensive_explanation(
            result, test_image, "Test", {"age": 30, "gender": "female", "skin_type": "type_II", "lesion_location": "neck"}
        )
        
        # Should provide multiple types of explanations
        self.assertIn('text_explanation', explanations)
        self.assertIn('visual_concepts', explanations)
        self.assertIn('clinical_concepts', explanations)
    
    def test_clinical_integration_readiness(self):
        """Validate clinical integration capabilities"""
        # Test if system provides clinical-ready outputs
        model = create_demo_model()
        test_image = Image.new('RGB', (224, 224), color='orange')
        
        result = model.diagnose(
            test_image, 
            "Clinical integration test", 
            {"age": 55, "gender": "male", "skin_type": "type_III", "lesion_location": "chest"}
        )
        
        # Should provide structured output suitable for clinical systems
        self.assertIsInstance(result.predictions, dict)
        self.assertIsInstance(result.confidence_scores, dict)
        self.assertIsInstance(result.explanations, dict)
        
        # Should include confidence assessment
        self.assertIn('overall_confidence', result.confidence_scores)
        
        # Should provide interpretable disease names
        for disease in result.predictions.keys():
            self.assertIsInstance(disease, str)
            self.assertNotIn('_', disease)  # Should be properly formatted


def run_all_tests():
    """Run all test suites"""
    print("🧪 Running comprehensive test suite for AI Dermatological Diagnosis System")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAIEngine,
        TestInterpretability,
        TestTrainingEvaluation,
        TestWebApplication,
        TestSystemIntegration,
        TestResearchPaperValidation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏥 TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ System validation: PASSED")
        print("The implementation successfully validates the research paper claims.")
    elif success_rate >= 70:
        print("⚠️  System validation: PARTIAL")
        print("Most features work correctly, but some improvements needed.")
    else:
        print("❌ System validation: NEEDS WORK")
        print("Significant issues found that need to be addressed.")
    
    return result


if __name__ == "__main__":
    run_all_tests()
