"""
Legacy GPU AI Platform - Plugin System
======================================

This module provides a plugin architecture that allows the community
to extend the platform with new capabilities without modifying core code.

Plugin Categories:
-----------------
1. Domain Plugins (e.g., wildlife, agriculture, medical)
   - Specialized models and preprocessing
   - Domain-specific optimizations
   
2. Hardware Plugins (e.g., specific GPU optimizations)
   - Custom memory strategies
   - GPU-specific kernels
   
3. Integration Plugins (e.g., web frameworks, databases)
   - API endpoints
   - Data connectors

Plugin Structure:
----------------
plugins/
├── wildlife/           # Domain plugin example
│   ├── __init__.py
│   ├── plugin.json     # Plugin metadata
│   ├── models/         # Pre-trained models
│   └── preprocessing/  # Domain-specific preprocessing
└── my_plugin/
    ├── __init__.py
    └── plugin.json

Example plugin.json:
-------------------
{
    "name": "wildlife-monitoring",
    "version": "1.0.0",
    "description": "Wildlife species classification for Colombia",
    "author": "Community",
    "category": "domain",
    "dependencies": ["pillow>=9.0"],
    "entry_point": "wildlife:WildlifePlugin"
}

Version: 0.5.0-dev
License: MIT
"""

__version__ = "0.5.0-dev"
__all__ = [
    "Plugin",
    "PluginManager",
    "PluginMetadata",
    "discover_plugins",
    "load_plugin",
]

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import importlib.util


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str = "Community"
    category: str = "general"
    dependencies: List[str] = field(default_factory=list)
    entry_point: str = ""
    enabled: bool = True
    
    @classmethod
    def from_json(cls, json_path: Path) -> "PluginMetadata":
        """Load metadata from plugin.json file."""
        with open(json_path, "r") as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert metadata to JSON string."""
        return json.dumps({
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "category": self.category,
            "dependencies": self.dependencies,
            "entry_point": self.entry_point,
        }, indent=2)


class Plugin(ABC):
    """
    Base class for all plugins.
    
    Plugins must extend this class and implement the required methods.
    
    Example:
        class WildlifePlugin(Plugin):
            def __init__(self):
                super().__init__()
                self.metadata = PluginMetadata(
                    name="wildlife-monitoring",
                    version="1.0.0",
                    description="Wildlife species classification",
                    category="domain"
                )
            
            def initialize(self, platform):
                # Setup plugin with platform context
                self.platform = platform
                self._load_models()
            
            def get_capabilities(self):
                return ["classify_species", "detect_animals"]
    """
    
    def __init__(self):
        """Initialize the plugin."""
        self.metadata: Optional[PluginMetadata] = None
        self._initialized = False
        
    @abstractmethod
    def initialize(self, platform: Any) -> bool:
        """
        Initialize the plugin with platform context.
        
        Args:
            platform: Platform instance for hardware access
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Return list of capabilities this plugin provides.
        
        Returns:
            List of capability identifiers
        """
        pass
    
    def cleanup(self):
        """Clean up plugin resources. Override if needed."""
        pass
    
    def get_info(self) -> dict:
        """Get plugin information."""
        return {
            "name": self.metadata.name if self.metadata else "Unknown",
            "version": self.metadata.version if self.metadata else "0.0.0",
            "initialized": self._initialized,
            "capabilities": self.get_capabilities() if self._initialized else [],
        }


class PluginManager:
    """
    Manages plugin discovery, loading, and lifecycle.
    
    Example:
        from src.plugins import PluginManager
        from src.sdk import Platform
        
        platform = Platform.initialize()
        manager = PluginManager(plugins_dir="./plugins")
        
        # Discover and load all plugins
        manager.discover()
        manager.load_all(platform)
        
        # Use a specific plugin
        wildlife = manager.get_plugin("wildlife-monitoring")
        result = wildlife.classify_species(image)
    """
    
    def __init__(self, plugins_dir: str = "./plugins"):
        """
        Initialize plugin manager.
        
        Args:
            plugins_dir: Directory containing plugins
        """
        self.plugins_dir = Path(plugins_dir)
        self.discovered: Dict[str, PluginMetadata] = {}
        self.loaded: Dict[str, Plugin] = {}
        
    def discover(self) -> List[PluginMetadata]:
        """
        Discover available plugins in the plugins directory.
        
        Returns:
            List of discovered plugin metadata
        """
        self.discovered.clear()
        
        if not self.plugins_dir.exists():
            return []
            
        for plugin_dir in self.plugins_dir.iterdir():
            if not plugin_dir.is_dir():
                continue
                
            metadata_file = plugin_dir / "plugin.json"
            if metadata_file.exists():
                try:
                    metadata = PluginMetadata.from_json(metadata_file)
                    self.discovered[metadata.name] = metadata
                except Exception as e:
                    print(f"Warning: Failed to load plugin metadata from {plugin_dir}: {e}")
                    
        return list(self.discovered.values())
    
    def load_plugin(self, name: str, platform: Any) -> Optional[Plugin]:
        """
        Load a specific plugin by name.
        
        Args:
            name: Plugin name
            platform: Platform instance
            
        Returns:
            Loaded Plugin instance or None
        """
        if name not in self.discovered:
            print(f"Plugin '{name}' not found. Run discover() first.")
            return None
            
        metadata = self.discovered[name]
        
        if not metadata.entry_point:
            print(f"Plugin '{name}' has no entry point defined.")
            return None
            
        try:
            # Parse entry point (format: "module:ClassName")
            module_name, class_name = metadata.entry_point.split(":")
            
            # Find the plugin module
            plugin_path = self.plugins_dir / name.replace("-", "_") / f"{module_name}.py"
            
            if not plugin_path.exists():
                # Try alternative path
                plugin_path = self.plugins_dir / name.replace("-", "_") / "__init__.py"
                
            if not plugin_path.exists():
                print(f"Plugin module not found: {plugin_path}")
                return None
                
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the plugin class and instantiate
            plugin_class = getattr(module, class_name)
            plugin = plugin_class()
            
            # Initialize with platform
            if plugin.initialize(platform):
                plugin._initialized = True
                self.loaded[name] = plugin
                return plugin
                
        except Exception as e:
            print(f"Failed to load plugin '{name}': {e}")
            
        return None
    
    def load_all(self, platform: Any) -> int:
        """
        Load all discovered plugins.
        
        Args:
            platform: Platform instance
            
        Returns:
            Number of successfully loaded plugins
        """
        loaded_count = 0
        
        for name, metadata in self.discovered.items():
            if metadata.enabled:
                if self.load_plugin(name, platform):
                    loaded_count += 1
                    
        return loaded_count
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self.loaded.get(name)
    
    def unload_plugin(self, name: str):
        """
        Unload a plugin.
        
        Args:
            name: Plugin name
        """
        if name in self.loaded:
            self.loaded[name].cleanup()
            del self.loaded[name]
    
    def unload_all(self):
        """Unload all plugins."""
        for plugin in self.loaded.values():
            plugin.cleanup()
        self.loaded.clear()
    
    def get_status(self) -> dict:
        """Get plugin manager status."""
        return {
            "plugins_dir": str(self.plugins_dir),
            "discovered": len(self.discovered),
            "loaded": len(self.loaded),
            "plugins": {
                name: plugin.get_info() 
                for name, plugin in self.loaded.items()
            }
        }
    
    def status_report(self) -> str:
        """Generate formatted status report."""
        status = self.get_status()
        
        lines = [
            "=" * 60,
            "Plugin Manager Status",
            "=" * 60,
            f"Plugins Directory: {status['plugins_dir']}",
            f"Discovered: {status['discovered']}",
            f"Loaded: {status['loaded']}",
            "-" * 60,
        ]
        
        if status['plugins']:
            lines.append("Loaded Plugins:")
            for name, info in status['plugins'].items():
                lines.append(f"  ✅ {name} v{info['version']}")
                lines.append(f"     Capabilities: {', '.join(info['capabilities'])}")
        else:
            lines.append("No plugins loaded.")
            
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Convenience functions

def discover_plugins(plugins_dir: str = "./plugins") -> List[PluginMetadata]:
    """
    Discover plugins in directory.
    
    Args:
        plugins_dir: Directory to search
        
    Returns:
        List of plugin metadata
    """
    manager = PluginManager(plugins_dir)
    return manager.discover()


def load_plugin(name: str, plugins_dir: str = "./plugins", platform: Any = None) -> Optional[Plugin]:
    """
    Load a single plugin.
    
    Args:
        name: Plugin name
        plugins_dir: Plugins directory
        platform: Platform instance (will auto-initialize if None)
        
    Returns:
        Loaded plugin or None
    """
    if platform is None:
        from src.sdk import Platform
        platform = Platform.initialize()
        
    manager = PluginManager(plugins_dir)
    manager.discover()
    return manager.load_plugin(name, platform)
