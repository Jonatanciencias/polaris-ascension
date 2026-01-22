"""
Plugin System for SDK Extension
================================

This module provides a flexible plugin architecture that allows
developers to extend the platform with custom functionality.

Design Principles:
-----------------
1. Simple plugin interface - inherit from Plugin base class
2. Automatic discovery - plugins in designated folders
3. Lifecycle management - initialize, execute, cleanup
4. Dependency resolution - plugins can depend on others
5. Version compatibility - check SDK version requirements

Use Cases:
---------
- Custom preprocessing pipelines
- Domain-specific model loaders
- Novel optimization techniques
- Hardware backend integrations
- Monitoring and logging extensions

Example Plugin:
--------------
    from src.sdk.plugins import Plugin, PluginMetadata
    
    class MyOptimizer(Plugin):
        metadata = PluginMetadata(
            name="my_optimizer",
            version="1.0.0",
            author="Your Name",
            description="Custom optimization technique"
        )
        
        def initialize(self):
            print("Initializing optimizer...")
        
        def execute(self, model):
            # Your optimization logic
            return optimized_model
        
        def cleanup(self):
            print("Cleaning up...")

Plugin Manager Usage:
--------------------
    from src.sdk.plugins import PluginManager
    
    # Initialize manager
    manager = PluginManager()
    manager.discover_plugins()  # Auto-find plugins
    
    # Load specific plugin
    plugin = manager.load_plugin("my_optimizer")
    
    # Execute plugin
    result = plugin.execute(model)
    
    # List all plugins
    for name in manager.list_plugins():
        print(name)

Version: 0.6.0-dev
Author: Legacy GPU AI Platform Team
License: MIT
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings
import json


class PluginStatus(Enum):
    """Status of a plugin."""
    UNLOADED = "unloaded"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


class PluginType(Enum):
    """Type of plugin functionality."""
    OPTIMIZER = "optimizer"          # Model optimization
    PREPROCESSOR = "preprocessor"    # Data preprocessing
    POSTPROCESSOR = "postprocessor"  # Output postprocessing
    MODEL_LOADER = "model_loader"    # Custom model formats
    BACKEND = "backend"              # Hardware backends
    MONITOR = "monitor"              # Monitoring/logging
    CUSTOM = "custom"                # Generic extension


@dataclass
class PluginMetadata:
    """
    Metadata for a plugin.
    
    Attributes:
        name: Unique plugin identifier
        version: Plugin version (semantic versioning)
        author: Plugin author
        description: Brief description
        plugin_type: Type of plugin
        dependencies: List of required plugin names
        sdk_min_version: Minimum SDK version required
        sdk_max_version: Maximum SDK version supported
        enabled: Whether plugin is enabled by default
    """
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType = PluginType.CUSTOM
    dependencies: List[str] = field(default_factory=list)
    sdk_min_version: Optional[str] = None
    sdk_max_version: Optional[str] = None
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'plugin_type': self.plugin_type.value,
            'dependencies': self.dependencies,
            'sdk_min_version': self.sdk_min_version,
            'sdk_max_version': self.sdk_max_version,
            'enabled': self.enabled,
        }


class Plugin(ABC):
    """
    Abstract base class for all plugins.
    
    All plugins must inherit from this class and implement:
    - initialize(): Setup plugin
    - execute(): Main plugin logic
    - cleanup(): Teardown plugin
    
    Plugins should also define a 'metadata' class attribute
    with PluginMetadata.
    """
    
    # Subclasses must define this
    metadata: PluginMetadata
    
    def __init__(self):
        """Initialize plugin base."""
        self.status = PluginStatus.UNLOADED
        self._context: Dict[str, Any] = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin.
        
        Called once when plugin is loaded. Use this to:
        - Load configuration
        - Initialize resources
        - Validate dependencies
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the plugin's main functionality.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Plugin execution result
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """
        Clean up plugin resources.
        
        Called when plugin is unloaded or system shutdown.
        Use this to:
        - Release resources
        - Save state
        - Close connections
        
        Returns:
            True if cleanup successful
        """
        pass
    
    def set_context(self, context: Dict[str, Any]):
        """
        Set execution context for plugin.
        
        Args:
            context: Dictionary with context information
        """
        self._context.update(context)
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get value from execution context.
        
        Args:
            key: Context key
            default: Default value if key not found
        
        Returns:
            Context value
        """
        return self._context.get(key, default)


class PluginManager:
    """
    Manages plugin discovery, loading, and execution.
    
    The PluginManager handles the complete lifecycle of plugins:
    1. Discovery - Find plugins in designated directories
    2. Loading - Import and instantiate plugins
    3. Initialization - Call plugin initialize() methods
    4. Execution - Route execution requests to plugins
    5. Cleanup - Properly shutdown plugins
    
    Example:
        manager = PluginManager()
        manager.discover_plugins()
        
        # Load and use a plugin
        optimizer = manager.load_plugin("custom_optimizer")
        result = optimizer.execute(model)
    """
    
    def __init__(
        self,
        plugin_dirs: Optional[List[Path]] = None,
        auto_discover: bool = True
    ):
        """
        Initialize PluginManager.
        
        Args:
            plugin_dirs: List of directories to search for plugins
            auto_discover: Automatically discover plugins on init
        """
        # Default plugin directories
        if plugin_dirs is None:
            plugin_dirs = [
                Path(__file__).parent.parent.parent / "plugins",
                Path.home() / ".legacy_gpu_ai" / "plugins",
                Path("/usr/local/share/legacy_gpu_ai/plugins"),
            ]
        
        self.plugin_dirs = [Path(d) for d in plugin_dirs if Path(d).exists()]
        
        # Plugin storage
        self._plugins: Dict[str, Plugin] = {}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}
        self._metadata: Dict[str, PluginMetadata] = {}
        
        # Execution hooks
        self._hooks: Dict[str, List[Callable]] = {}
        
        if auto_discover:
            self.discover_plugins()
    
    def discover_plugins(self) -> int:
        """
        Discover plugins in plugin directories.
        
        Searches for Python files in plugin directories and
        attempts to import them as plugins.
        
        Returns:
            Number of plugins discovered
        """
        discovered = 0
        
        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue
            
            # Find all Python files
            for py_file in plugin_dir.glob("**/*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                try:
                    self._load_plugin_from_file(py_file)
                    discovered += 1
                except Exception as e:
                    warnings.warn(f"Failed to load plugin {py_file}: {e}")
        
        return discovered
    
    def _load_plugin_from_file(self, file_path: Path):
        """
        Load a plugin from a Python file.
        
        Args:
            file_path: Path to plugin file
        """
        # Create module name
        module_name = f"plugin_{file_path.stem}"
        
        # Load module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Find Plugin subclasses in module
        for item_name in dir(module):
            item = getattr(module, item_name)
            
            if (isinstance(item, type) and 
                issubclass(item, Plugin) and 
                item != Plugin):
                
                # Check if plugin has metadata
                if not hasattr(item, 'metadata'):
                    warnings.warn(f"Plugin {item_name} missing metadata")
                    continue
                
                # Register plugin class
                metadata = item.metadata
                self._plugin_classes[metadata.name] = item
                self._metadata[metadata.name] = metadata
    
    def load_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Load and initialize a plugin.
        
        Args:
            plugin_name: Name of plugin to load
        
        Returns:
            Initialized plugin instance or None
        """
        # Check if already loaded
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]
        
        # Check if plugin class exists
        if plugin_name not in self._plugin_classes:
            warnings.warn(f"Plugin '{plugin_name}' not found")
            return None
        
        # Instantiate plugin
        plugin_class = self._plugin_classes[plugin_name]
        plugin = plugin_class()
        
        # Initialize plugin
        try:
            if plugin.initialize():
                plugin.status = PluginStatus.INITIALIZED
                self._plugins[plugin_name] = plugin
                return plugin
            else:
                plugin.status = PluginStatus.ERROR
                return None
        except Exception as e:
            warnings.warn(f"Failed to initialize plugin '{plugin_name}': {e}")
            plugin.status = PluginStatus.ERROR
            return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload and cleanup a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
        
        Returns:
            True if successful
        """
        if plugin_name not in self._plugins:
            return False
        
        plugin = self._plugins[plugin_name]
        
        try:
            success = plugin.cleanup()
            if success:
                plugin.status = PluginStatus.UNLOADED
                del self._plugins[plugin_name]
            return success
        except Exception as e:
            warnings.warn(f"Error cleaning up plugin '{plugin_name}': {e}")
            return False
    
    def list_plugins(self, loaded_only: bool = False) -> List[str]:
        """
        List available plugins.
        
        Args:
            loaded_only: Only list loaded plugins
        
        Returns:
            List of plugin names
        """
        if loaded_only:
            return list(self._plugins.keys())
        else:
            return list(self._plugin_classes.keys())
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin instance.
        
        Args:
            plugin_name: Name of plugin
        
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(plugin_name)
    
    def get_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        Get metadata for a plugin.
        
        Args:
            plugin_name: Name of plugin
        
        Returns:
            PluginMetadata or None
        """
        return self._metadata.get(plugin_name)
    
    def register_hook(self, event: str, callback: Callable):
        """
        Register a callback for an event.
        
        Args:
            event: Event name
            callback: Function to call on event
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)
    
    def trigger_hook(self, event: str, *args, **kwargs):
        """
        Trigger all callbacks for an event.
        
        Args:
            event: Event name
            *args: Positional arguments for callbacks
            **kwargs: Keyword arguments for callbacks
        """
        if event not in self._hooks:
            return
        
        for callback in self._hooks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                warnings.warn(f"Hook callback error for '{event}': {e}")
    
    def cleanup_all(self) -> bool:
        """
        Cleanup all loaded plugins.
        
        Returns:
            True if all cleanups successful
        """
        success = True
        for plugin_name in list(self._plugins.keys()):
            if not self.unload_plugin(plugin_name):
                success = False
        return success
    
    def export_plugin_list(self, output_path: Path):
        """
        Export list of available plugins to JSON.
        
        Args:
            output_path: Path to output file
        """
        plugin_data = {
            name: metadata.to_dict()
            for name, metadata in self._metadata.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(plugin_data, f, indent=2)


# Example plugins for demonstration
class ExampleOptimizerPlugin(Plugin):
    """Example optimizer plugin."""
    
    metadata = PluginMetadata(
        name="example_optimizer",
        version="1.0.0",
        author="Platform Team",
        description="Example optimization plugin",
        plugin_type=PluginType.OPTIMIZER
    )
    
    def initialize(self) -> bool:
        print(f"Initializing {self.metadata.name}...")
        return True
    
    def execute(self, model) -> Any:
        print(f"Executing {self.metadata.name} on model...")
        # Placeholder optimization
        return model
    
    def cleanup(self) -> bool:
        print(f"Cleaning up {self.metadata.name}...")
        return True


# Demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Legacy GPU AI Platform - Plugin System Demo")
    print("=" * 70)
    
    # Initialize manager
    manager = PluginManager()
    
    print(f"\n✅ Discovered {len(manager.list_plugins())} plugins")
    
    # List all plugins
    print("\nAvailable Plugins:")
    print("-" * 70)
    for name in manager.list_plugins():
        metadata = manager.get_metadata(name)
        print(f"  • {name} v{metadata.version}")
        print(f"    {metadata.description}")
        print(f"    Type: {metadata.plugin_type.value}")
    
    # Load and execute a plugin
    print("\nLoading and Executing Plugin:")
    print("-" * 70)
    plugin = manager.load_plugin("example_optimizer")
    if plugin:
        result = plugin.execute("dummy_model")
        print(f"✅ Plugin executed successfully")
    
    # Cleanup
    print("\nCleanup:")
    print("-" * 70)
    manager.cleanup_all()
    print("✅ All plugins cleaned up")
    
    print("\n" + "=" * 70)
