#!/bin/bash
# Monitor PyTorch Compilation Progress

LOGFILE="/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/compile_output.log"
BUILD_LOG="/home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/pytorch_build/build_complete.log"

echo "=== PyTorch Compilation Monitor ==="
echo ""

# Check if compilation process is running
PID=$(ps aux | grep "[c]ompile_pytorch.sh" | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "✓ Compilation RUNNING (PID: $PID)"
    echo "  Started: $(ps -p $PID -o lstart=)"
    echo "  Elapsed: $(ps -p $PID -o etime=)"
    echo "  CPU: $(ps -p $PID -o %cpu=)%"
    echo "  Memory: $(ps -p $PID -o %mem=)%"
else
    echo "✗ Compilation process not found"
fi

echo ""
echo "=== Recent Activity (last 20 lines) ==="
if [ -f "$BUILD_LOG" ]; then
    tail -20 "$BUILD_LOG"
elif [ -f "$LOGFILE" ]; then
    tail -20 "$LOGFILE"
else
    echo "No log files found yet"
fi

echo ""
echo "=== Compilation Statistics ==="
if [ -f "$BUILD_LOG" ]; then
    TOTAL_LINES=$(wc -l < "$BUILD_LOG")
    BUILD_TARGETS=$(grep -c "\[.*%\]" "$BUILD_LOG" 2>/dev/null || echo "0")
    ERRORS=$(grep -ci "error" "$BUILD_LOG" 2>/dev/null || echo "0")
    WARNINGS=$(grep -ci "warning" "$BUILD_LOG" 2>/dev/null || echo "0")
    
    echo "  Log lines: $TOTAL_LINES"
    echo "  Build targets: $BUILD_TARGETS"
    echo "  Errors: $ERRORS"
    echo "  Warnings: $WARNINGS"
fi

echo ""
echo "=== Disk Usage ==="
du -sh /home/jonatanciencias/Proyectos/Programacion/Radeon_RX_580/pytorch_build/

echo ""
echo "Tip: Run this script periodically to monitor progress"
echo "Estimated compilation time: 4-8 hours"
