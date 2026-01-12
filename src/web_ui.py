"""
Radeon RX 580 AI - Web Interface

Simple, accessible web UI for running AI inference without writing code.
Designed for non-technical users (doctors, researchers, conservationists).

Features:
- Upload images for classification
- Select model and optimization mode
- View results with confidence scores
- Download results as JSON/CSV

Usage:
    python src/web_ui.py
    # Open browser to http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import tempfile
import json
from datetime import datetime
import logging

# Import our inference engine
from src.inference import ONNXInferenceEngine, InferenceConfig
from src.core.gpu import GPUManager
from src.core.memory import MemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='web_ui/templates',
            static_folder='web_ui/static')

# Configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'radeon_rx580_ai_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER = Path(__file__).parent.parent / 'examples' / 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize managers
gpu_manager = GPUManager()
memory_manager = MemoryManager()

# Global inference engine (lazy loaded)
current_engine = None
current_config = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_available_models():
    """Get list of available ONNX models."""
    models = []
    
    if not MODELS_FOLDER.exists():
        return models
    
    for model_file in MODELS_FOLDER.glob('*.onnx'):
        model_info = {
            'name': model_file.stem,
            'path': str(model_file),
            'size_mb': model_file.stat().st_size / (1024 * 1024),
            'type': 'classification'  # Default
        }
        
        # Determine model type and description
        if 'yolo' in model_file.stem.lower():
            model_info['type'] = 'detection'
            model_info['description'] = 'Object Detection (80 classes)'
            model_info['use_case'] = 'Wildlife monitoring, security, traffic'
        elif 'mobilenet' in model_file.stem.lower():
            model_info['description'] = 'Lightweight Classification'
            model_info['use_case'] = 'Real-time, mobile, edge devices'
        elif 'resnet' in model_file.stem.lower():
            model_info['description'] = 'Robust Classification'
            model_info['use_case'] = 'Medical imaging, scientific'
        elif 'efficientnet' in model_file.stem.lower():
            model_info['description'] = 'Efficient Classification'
            model_info['use_case'] = 'Balance of speed and accuracy'
        else:
            model_info['description'] = 'Classification Model'
            model_info['use_case'] = 'General purpose'
        
        models.append(model_info)
    
    return sorted(models, key=lambda x: x['name'])


def get_inference_engine(model_name, precision='fp32', batch_size=1):
    """Get or create inference engine with specified configuration."""
    global current_engine, current_config
    
    # Create new config
    new_config = InferenceConfig(
        device='auto',
        precision=precision,
        batch_size=batch_size,
        optimization_level=2,
        enable_profiling=True
    )
    
    # Check if we need to create new engine
    need_new_engine = (
        current_engine is None or
        current_config is None or
        current_config.precision != precision or
        current_engine.model_info is None or
        current_engine.model_info.name != model_name
    )
    
    if need_new_engine:
        logger.info(f"Creating new engine: {model_name} with {precision}")
        current_engine = ONNXInferenceEngine(new_config, gpu_manager, memory_manager)
        
        # Load model
        model_path = MODELS_FOLDER / f"{model_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        current_engine.load_model(model_path)
        current_config = new_config
    
    return current_engine


@app.route('/')
def index():
    """Main page."""
    # Get system info
    gpu_info = gpu_manager.get_info()
    memory_stats = memory_manager.get_stats()
    available_models = get_available_models()
    
    system_info = {
        'gpu': gpu_info.name if gpu_info else 'CPU Only',
        'vram_gb': gpu_info.vram_gb if gpu_info else 0,
        'ram_gb': memory_stats.total_ram_gb,
        'opencl': gpu_info.opencl_available if gpu_info else False,
        'models_available': len(available_models)
    }
    
    return render_template('index.html', 
                         system_info=system_info,
                         models=available_models)


@app.route('/api/system_info')
def api_system_info():
    """Get system information."""
    gpu_info = gpu_manager.get_info()
    memory_stats = memory_manager.get_stats()
    
    return jsonify({
        'gpu': {
            'name': gpu_info.name if gpu_info else 'Not detected',
            'vram_gb': gpu_info.vram_gb if gpu_info else 0,
            'driver': gpu_info.driver if gpu_info else 'Unknown',
            'opencl': gpu_info.opencl_available if gpu_info else False
        },
        'memory': {
            'ram_total_gb': memory_stats.total_ram_gb,
            'ram_available_gb': memory_stats.available_ram_gb,
            'vram_total_gb': memory_stats.gpu_vram_gb
        },
        'models': get_available_models()
    })


@app.route('/api/classify', methods=['POST'])
def api_classify():
    """Classify uploaded image."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, GIF, or BMP'}), 400
        
        # Get parameters
        model_name = request.form.get('model', 'mobilenetv2')
        precision = request.form.get('precision', 'fp32')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        file.save(filepath)
        
        logger.info(f"Processing: {filepath} with {model_name} ({precision})")
        
        # Get inference engine
        engine = get_inference_engine(model_name, precision)
        
        # Run inference
        result = engine.infer(filepath)
        
        # Get performance stats
        stats = engine.profiler.get_statistics() if engine.profiler else {}
        
        # Get optimization info
        opt_info = engine.get_optimization_info()
        
        # Clean up uploaded file
        try:
            filepath.unlink()
        except:
            pass
        
        # Format response
        response = {
            'success': True,
            'model': model_name,
            'precision': precision,
            'optimization': opt_info,
            'predictions': result.get('predictions', [])[:10],  # Top 10
            'top1_class': result.get('top1_class'),
            'top1_confidence': result.get('top1_confidence'),
            'inference_time_ms': stats.get('mean', 0) if stats else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/models')
def api_models():
    """Get list of available models."""
    return jsonify({
        'models': get_available_models(),
        'precision_modes': [
            {
                'value': 'fp32',
                'name': 'Standard (FP32)',
                'speedup': '1.0x',
                'description': 'Maximum accuracy, baseline speed'
            },
            {
                'value': 'fp16',
                'name': 'Fast (FP16)',
                'speedup': '~1.5x',
                'description': 'Good accuracy, faster (73.6 dB SNR)'
            },
            {
                'value': 'int8',
                'name': 'Ultra-Fast (INT8)',
                'speedup': '~2.5x',
                'description': 'High accuracy, maximum speed (99.99% correlation)'
            }
        ]
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'Radeon RX 580 AI Web UI'})


def create_templates():
    """Create HTML templates if they don't exist."""
    templates_dir = Path(__file__).parent / 'web_ui' / 'templates'
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Create index.html
    index_html = templates_dir / 'index.html'
    if not index_html.exists():
        with open(index_html, 'w') as f:
            f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Radeon RX 580 AI - Web Interface</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1em;
        }
        
        .system-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .info-item {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .info-label {
            font-weight: bold;
            color: #555;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }
        
        .card h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .upload-area:hover {
            background: #f8f9ff;
            border-color: #764ba2;
        }
        
        .upload-area.dragover {
            background: #e8ebff;
            border-color: #764ba2;
        }
        
        #imagePreview {
            max-width: 100%;
            max-height: 300px;
            margin: 20px 0;
            border-radius: 10px;
            display: none;
        }
        
        .form-group {
            margin: 20px 0;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            color: #333;
        }
        
        select, input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        select:focus, input[type="file"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .results {
            margin-top: 20px;
        }
        
        .result-item {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .confidence-bar {
            background: #e9ecef;
            height: 30px;
            border-radius: 15px;
            overflow: hidden;
            flex: 1;
            margin-left: 15px;
        }
        
        .confidence-fill {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            height: 100%;
            transition: width 0.5s;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .performance-info {
            background: #e8f5e9;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Radeon RX 580 AI</h1>
            <p>Affordable AI for Everyone - Simple Web Interface</p>
            
            <div class="system-info">
                <div class="info-item">
                    <span class="info-label">GPU:</span>
                    <span>{{ system_info.gpu }}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">VRAM:</span>
                    <span>{{ system_info.vram_gb }} GB</span>
                </div>
                <div class="info-item">
                    <span class="info-label">RAM:</span>
                    <span>{{ system_info.ram_gb|round(1) }} GB</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Models:</span>
                    <span>{{ system_info.models_available }} available</span>
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="card">
                <h2>📸 Upload Image</h2>
                
                <div class="upload-area" id="uploadArea">
                    <p style="font-size: 3em; margin-bottom: 10px;">📁</p>
                    <p>Drag & drop an image here or click to browse</p>
                    <p style="color: #999; margin-top: 10px;">Supported: PNG, JPG, JPEG, GIF, BMP</p>
                </div>
                
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
                <img id="imagePreview" alt="Preview">
                
                <div class="form-group">
                    <label for="modelSelect">Select Model:</label>
                    <select id="modelSelect">
                        {% for model in models %}
                        <option value="{{ model.name }}">
                            {{ model.name|upper }} - {{ model.description }} ({{ model.size_mb|round(1) }} MB)
                        </option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="precisionSelect">Optimization Mode:</label>
                    <select id="precisionSelect">
                        <option value="fp32">Standard (FP32) - Maximum accuracy</option>
                        <option value="fp16" selected>Fast (FP16) - ~1.5x speedup</option>
                        <option value="int8">Ultra-Fast (INT8) - ~2.5x speedup</option>
                    </select>
                </div>
                
                <button class="btn" id="classifyBtn" disabled>Classify Image</button>
                
                <div class="error" id="errorMessage"></div>
            </div>
            
            <div class="card">
                <h2>📊 Results</h2>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Processing image...</p>
                </div>
                
                <div id="resultsContainer">
                    <p style="color: #999; text-align: center; padding: 40px;">
                        Upload an image to see classification results
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const imagePreview = document.getElementById('imagePreview');
        const classifyBtn = document.getElementById('classifyBtn');
        const loading = document.getElementById('loading');
        const resultsContainer = document.getElementById('resultsContainer');
        const errorMessage = document.getElementById('errorMessage');
        
        let selectedFile = null;
        
        // Upload area click
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        // File input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            selectedFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Enable classify button
            classifyBtn.disabled = false;
            
            // Hide error
            errorMessage.style.display = 'none';
        }
        
        // Classify button
        classifyBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            
            // Show loading
            loading.style.display = 'block';
            resultsContainer.innerHTML = '';
            errorMessage.style.display = 'none';
            classifyBtn.disabled = true;
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model', document.getElementById('modelSelect').value);
            formData.append('precision', document.getElementById('precisionSelect').value);
            
            try {
                const response = await fetch('/api/classify', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Display results
                displayResults(data);
                
            } catch (error) {
                errorMessage.textContent = 'Error: ' + error.message;
                errorMessage.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                classifyBtn.disabled = false;
            }
        });
        
        function displayResults(data) {
            let html = '';
            
            // Performance info
            html += `
                <div class="performance-info">
                    <strong>⚡ Performance:</strong> ${data.inference_time_ms.toFixed(1)}ms<br>
                    <strong>🔧 Mode:</strong> ${data.precision.toUpperCase()}<br>
                    <strong>📦 Model:</strong> ${data.model}
                </div>
            `;
            
            // Top predictions
            html += '<div class="results"><h3 style="margin: 20px 0 10px 0;">Top Predictions:</h3>';
            
            data.predictions.forEach((pred, idx) => {
                const percentage = (pred.confidence * 100).toFixed(1);
                html += `
                    <div class="result-item">
                        <span style="font-weight: bold;">#${idx + 1}</span>
                        <span>Class ${pred.class_id}</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${percentage}%">
                                ${percentage}%
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            
            resultsContainer.innerHTML = html;
        }
    </script>
</body>
</html>""")
    
    logger.info(f"Created templates in {templates_dir}")


def main():
    """Run the web server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Radeon RX 580 AI Web Interface')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create templates
    create_templates()
    
    print("\n" + "="*70)
    print("🚀 RADEON RX 580 AI - WEB INTERFACE")
    print("="*70)
    print(f"\n📡 Server starting on http://{args.host}:{args.port}")
    print("\n💡 Usage:")
    print(f"   1. Open browser to http://localhost:{args.port}")
    print(f"   2. Upload an image")
    print(f"   3. Select model and optimization mode")
    print(f"   4. Click 'Classify Image'")
    print("\n⚠️  Make sure models are downloaded:")
    print("   python scripts/download_models.py --all")
    print("\n" + "="*70 + "\n")
    
    # Run server
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
