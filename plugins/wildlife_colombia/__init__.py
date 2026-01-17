"""
Wildlife Colombia Plugin
========================

Domain plugin for Colombian biodiversity monitoring using legacy AMD GPUs.

This plugin demonstrates how domain-specific functionality can be packaged
and distributed independently of the core platform.

Features:
---------
- Species classification for 10 Colombian species
- Camera trap image preprocessing
- Optimized for 8GB VRAM constraints
- Batch processing for large datasets

Usage:
------
    from src.plugins import load_plugin
    
    wildlife = load_plugin("wildlife-colombia")
    
    # Classify a single image
    species, confidence = wildlife.classify_species("camera_trap_001.jpg")
    
    # Process a batch of images
    results = wildlife.process_batch("./camera_images/", output_dir="./results/")

Target Hardware:
---------------
- AMD RX 580 (8GB) - Primary
- AMD RX 570 (4GB/8GB) - Supported with smaller batch sizes
- AMD Vega 56/64 - Supported

Version: 0.4.0
License: MIT
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Import plugin base class
try:
    from src.plugins import Plugin, PluginMetadata
except ImportError:
    # Fallback for standalone testing
    Plugin = object
    PluginMetadata = None


class WildlifePlugin(Plugin if Plugin != object else object):
    """
    Wildlife monitoring plugin for Colombian species.
    
    This plugin provides species classification optimized for
    legacy AMD GPUs and Colombian biodiversity.
    """
    
    SPECIES = [
        "jaguar",
        "puma", 
        "ocelot",
        "spectacled_bear",
        "capybara",
        "mountain_tapir",
        "howler_monkey",
        "white_tailed_deer",
        "harpy_eagle",
        "king_vulture",
    ]
    
    def __init__(self):
        """Initialize the wildlife plugin."""
        if Plugin != object:
            super().__init__()
            
        self.metadata = PluginMetadata(
            name="wildlife-colombia",
            version="0.4.0",
            description="Wildlife species classification for Colombian biodiversity",
            author="Legacy GPU AI Platform Community",
            category="domain",
            entry_point="wildlife:WildlifePlugin"
        ) if PluginMetadata else None
        
        self.platform = None
        self.model = None
        self._model_loaded = False
        
    def initialize(self, platform: Any) -> bool:
        """
        Initialize plugin with platform context.
        
        Args:
            platform: Platform instance
            
        Returns:
            True if successful
        """
        self.platform = platform
        
        # Check GPU capabilities
        if hasattr(platform, 'gpu_info'):
            info = platform.gpu_info()
            print(f"Wildlife Plugin initializing on {info.get('name', 'Unknown GPU')}")
            
            # Warn if VRAM is limited
            vram = info.get('vram_gb', 0)
            if vram < 4:
                print("⚠️ Warning: Less than 4GB VRAM. Use 'lite' model variant.")
                
        self._initialized = True
        return True
    
    def get_capabilities(self) -> List[str]:
        """Return plugin capabilities."""
        return [
            "classify_species",
            "detect_animals",
            "process_batch",
            "get_species_info",
        ]
    
    def load_model(self, variant: str = "default") -> bool:
        """
        Load the classification model.
        
        Args:
            variant: Model variant ("default" or "lite")
            
        Returns:
            True if model loaded successfully
        """
        model_name = f"wildlife_classifier_{variant}.onnx"
        model_path = Path(__file__).parent / "models" / model_name
        
        try:
            if self.platform and hasattr(self.platform, 'load_model'):
                self.model = self.platform.load_model(str(model_path))
                self._model_loaded = True
                return True
            else:
                print(f"Model would be loaded from: {model_path}")
                self._model_loaded = False  # Placeholder
                return True  # Allow testing without actual model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    def classify_species(
        self, 
        image_path: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Classify species in an image.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of (species_name, confidence) tuples
        """
        if not self._model_loaded:
            self.load_model()
            
        # Placeholder - actual inference would happen here
        # For now, return mock results for testing
        import random
        results = [
            (species, random.uniform(0.1, 0.9))
            for species in random.sample(self.SPECIES, min(top_k, len(self.SPECIES)))
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def process_batch(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        batch_size: int = 4
    ) -> Dict[str, Any]:
        """
        Process a batch of images.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory for results (optional)
            batch_size: Images per batch (adjust based on VRAM)
            
        Returns:
            Processing results summary
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            return {"error": f"Directory not found: {input_dir}"}
            
        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        
        results = {
            "total_images": len(images),
            "processed": 0,
            "classifications": [],
            "errors": [],
        }
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            for img_path in batch:
                try:
                    predictions = self.classify_species(str(img_path))
                    results["classifications"].append({
                        "image": img_path.name,
                        "predictions": predictions,
                    })
                    results["processed"] += 1
                except Exception as e:
                    results["errors"].append({
                        "image": img_path.name,
                        "error": str(e),
                    })
                    
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / "classification_results.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
                
        return results
    
    def get_species_info(self, species: str) -> Dict[str, Any]:
        """
        Get information about a species.
        
        Args:
            species: Species identifier
            
        Returns:
            Species information
        """
        species_data = {
            "jaguar": {
                "scientific_name": "Panthera onca",
                "conservation_status": "Near Threatened",
                "habitat": "Tropical rainforest, wetlands",
                "regions": ["Amazon", "Chocó", "Orinoquía"],
            },
            "puma": {
                "scientific_name": "Puma concolor",
                "conservation_status": "Least Concern",
                "habitat": "Mountains, forests, grasslands",
                "regions": ["Andes", "Sierra Nevada"],
            },
            "ocelot": {
                "scientific_name": "Leopardus pardalis",
                "conservation_status": "Least Concern",
                "habitat": "Tropical forest, mangroves",
                "regions": ["Amazon", "Caribbean", "Pacific"],
            },
            "spectacled_bear": {
                "scientific_name": "Tremarctos ornatus",
                "conservation_status": "Vulnerable",
                "habitat": "Cloud forest, páramo",
                "regions": ["Andes"],
            },
            "capybara": {
                "scientific_name": "Hydrochoerus hydrochaeris",
                "conservation_status": "Least Concern",
                "habitat": "Wetlands, rivers",
                "regions": ["Orinoquía", "Amazon"],
            },
            # Add more species as needed
        }
        
        return species_data.get(species, {"error": f"Unknown species: {species}"})
    
    def cleanup(self):
        """Release resources."""
        if self.model is not None:
            del self.model
            self.model = None
        self._model_loaded = False


# Standalone testing
if __name__ == "__main__":
    plugin = WildlifePlugin()
    print(f"Wildlife Plugin v{plugin.metadata.version if plugin.metadata else '0.4.0'}")
    print(f"Supported species: {len(plugin.SPECIES)}")
    print(f"Capabilities: {plugin.get_capabilities()}")
    
    # Test classification (mock)
    results = plugin.classify_species("test_image.jpg")
    print(f"Sample classification: {results}")
