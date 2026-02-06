#!/usr/bin/env python3
"""
Script para actualizar el progreso del roadmap de optimización.

Uso:
    python scripts/update_progress.py --task 1.1.1 --status completed --gflops 165.5
    python scripts/update_progress.py --task 1.2.1 --status in-progress
    python scripts/update_progress.py --summary
"""

import argparse
import datetime
import re
from pathlib import Path

# Paths
DOCS_DIR = Path(__file__).parent.parent / "docs"
TRACKING_FILE = DOCS_DIR / "PROGRESS_TRACKING.md"
ROADMAP_FILE = DOCS_DIR / "ROADMAP_OPTIMIZATION.md"


def update_task_status(task_id: str, status: str, notes: str = None):
    """Actualiza el status de una tarea en el roadmap."""
    
    status_emoji = {
        "pending": "⏳ PENDIENTE",
        "in-progress": "🔄 EN PROGRESO", 
        "completed": "✅ COMPLETADO",
        "blocked": "🚫 BLOQUEADO"
    }
    
    if status not in status_emoji:
        print(f"❌ Status inválido: {status}")
        print(f"   Opciones: {', '.join(status_emoji.keys())}")
        return False
    
    # Leer roadmap
    with open(ROADMAP_FILE, 'r') as f:
        content = f.read()
    
    # Buscar la tarea
    pattern = rf'(\[ \] \*\*Task {task_id}:\*\*.*?)(?=\n  -|\n\n|\*\*Entregables)'
    
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print(f"❌ Tarea {task_id} no encontrada en el roadmap")
        return False
    
    # Actualizar checkbox
    if status == "completed":
        content = content.replace(
            f"[ ] **Task {task_id}:",
            f"[x] **Task {task_id}:"
        )
    
    # Agregar/actualizar status
    task_block = match.group(0)
    
    # Verificar si ya tiene status
    if "Status:" in task_block:
        # Reemplazar status existente
        new_block = re.sub(
            r'Status: .*',
            f'Status: {status_emoji[status]}',
            task_block
        )
    else:
        # Agregar status después del título
        lines = task_block.split('\n')
        lines.insert(1, f'  - Status: {status_emoji[status]}')
        new_block = '\n'.join(lines)
    
    # Agregar fecha si está completado
    if status == "completed" and "Fecha fin:" not in new_block:
        date = datetime.date.today().strftime("%d/%m/%Y")
        lines = new_block.split('\n')
        # Encontrar línea de status
        for i, line in enumerate(lines):
            if "Status:" in line:
                lines.insert(i+1, f'  - Fecha fin: {date}')
                break
        new_block = '\n'.join(lines)
    
    # Agregar notas si se proporcionan
    if notes and "Notas:" not in new_block:
        lines = new_block.split('\n')
        lines.append(f'  - Notas: {notes}')
        new_block = '\n'.join(lines)
    
    # Reemplazar en contenido
    content = content.replace(task_block, new_block)
    
    # Guardar
    with open(ROADMAP_FILE, 'w') as f:
        f.write(content)
    
    print(f"✅ Task {task_id} actualizada a: {status_emoji[status]}")
    
    # Actualizar tracking
    update_tracking_file(task_id, status)
    
    return True


def update_tracking_file(task_id: str, status: str):
    """Actualiza el archivo de tracking."""
    
    with open(TRACKING_FILE, 'r') as f:
        content = f.read()
    
    # Actualizar timestamp
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    content = re.sub(
        r'\*\*Última actualización:\*\* .*',
        f'**Última actualización:** {now}',
        content
    )
    
    # Agregar al log
    today = datetime.date.today().strftime("%Y-%m-%d")
    log_entry = f"\n### {today}\n- Task {task_id}: {status}\n"
    
    # Encontrar sección de log
    log_section = content.find("## 📝 Log de Actividades")
    if log_section != -1:
        # Insertar después del título
        next_section = content.find("##", log_section + 10)
        if next_section == -1:
            next_section = len(content)
        
        # Verificar si ya existe entrada para hoy
        if today not in content[log_section:next_section]:
            content = content[:next_section] + log_entry + content[next_section:]
        else:
            # Agregar a entrada existente
            date_pos = content.find(f"### {today}", log_section)
            next_date = content.find("###", date_pos + 1)
            if next_date == -1:
                next_date = next_section
            content = content[:next_date] + f"- Task {task_id}: {status}\n" + content[next_date:]
    
    with open(TRACKING_FILE, 'w') as f:
        f.write(content)


def add_performance_metric(gflops: float, notes: str = None):
    """Agrega una nueva métrica de performance."""
    
    with open(TRACKING_FILE, 'r') as f:
        content = f.read()
    
    # Encontrar tabla de métricas
    metrics_section = content.find("## 📈 Métricas Actuales")
    if metrics_section == -1:
        print("❌ Sección de métricas no encontrada")
        return False
    
    # Buscar primera línea vacía después de la tabla
    table_end = content.find("| -- | -- |", metrics_section)
    if table_end == -1:
        print("❌ Formato de tabla incorrecto")
        return False
    
    # Calcular speedup vs baseline (150.96)
    baseline = 150.96
    speedup = gflops / baseline
    
    # Crear nueva fila
    date = datetime.date.today().strftime("%d/%m/%Y")
    new_row = f"| {date} | {gflops:.2f} | {speedup:.2f}x | -- | -- | {notes or 'Benchmark actualizado'} |\n"
    
    # Insertar antes de la línea --
    content = content[:table_end] + new_row + content[table_end:]
    
    with open(TRACKING_FILE, 'w') as f:
        f.write(content)
    
    print(f"✅ Métrica agregada: {gflops:.2f} GFLOPS ({speedup:.2f}x speedup)")
    return True


def show_summary():
    """Muestra resumen del progreso."""
    
    with open(TRACKING_FILE, 'r') as f:
        content = f.read()
    
    # Extraer sección de progreso
    start = content.find("## 🎯 Progreso Global")
    end = content.find("## 📈", start)
    
    if start != -1 and end != -1:
        print(content[start:end])
    else:
        print("❌ No se pudo leer el progreso")


def main():
    parser = argparse.ArgumentParser(
        description="Actualizar progreso del roadmap de optimización"
    )
    
    parser.add_argument(
        "--task",
        help="ID de la tarea (ej: 1.1.1)"
    )
    
    parser.add_argument(
        "--status",
        choices=["pending", "in-progress", "completed", "blocked"],
        help="Nuevo status de la tarea"
    )
    
    parser.add_argument(
        "--notes",
        help="Notas adicionales sobre la tarea"
    )
    
    parser.add_argument(
        "--gflops",
        type=float,
        help="Performance alcanzada en GFLOPS"
    )
    
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Mostrar resumen del progreso"
    )
    
    args = parser.parse_args()
    
    if args.summary:
        show_summary()
        return
    
    if not args.task:
        parser.print_help()
        return
    
    if not args.status:
        print("❌ Se requiere --status")
        return
    
    # Actualizar tarea
    success = update_task_status(args.task, args.status, args.notes)
    
    # Agregar métrica si se proporcionó
    if success and args.gflops:
        add_performance_metric(args.gflops, args.notes)
    
    print("\n" + "="*60)
    print("📊 Resumen actualizado:")
    show_summary()


if __name__ == "__main__":
    main()
