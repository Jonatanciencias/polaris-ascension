# Test Videos Directory

## 📁 Dónde Guardar Tus Videos

Guarda tus videos aquí para procesarlos con el optimizer.

### Formatos Soportados
- `.mp4` (recomendado)
- `.avi`
- `.mov`
- `.mkv`

### Ejemplos de Uso

```bash
# Procesar video
python ../optimizer.py --input test_videos/mi_video.mp4 --output optimized_output.mp4

# O simplemente:
python ../optimizer.py --input test_videos/mi_video.mp4 --output test_videos/mi_video_optimized.mp4
```

### Videos de Prueba Sugeridos

1. **Videos cortos** (10-30 seg) para pruebas rápidas
2. **Diferentes resoluciones**:
   - 720p (1280×720) - rápido
   - 1080p (1920×1080) - estándar
   - 4K (3840×2160) - challenge

3. **Dónde conseguir videos de prueba gratis**:
   - [Pexels Videos](https://www.pexels.com/videos/) - gratis, sin copyright
   - [Pixabay Videos](https://pixabay.com/videos/) - gratis, sin copyright
   - [Vimeo Creative Commons](https://vimeo.com/creativecommons) - varios
   - Sintel (open movie): https://durian.blender.org/download/

### Tu Video
```bash
# Ejemplo: Descargaste un video
wget https://example.com/video.mp4 -O test_videos/mi_prueba.mp4

# Procesar
cd ..
python optimizer.py --input test_videos/mi_prueba.mp4 --output test_videos/optimized.mp4
```
