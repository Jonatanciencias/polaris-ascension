#!/bin/bash
################################################################################
# AMD GPU Power Sensor Setup
# Enables direct power sensor readings for AMD Radeon RX 580
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}  AMD GPU Power Sensor Setup - Radeon RX 580${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root (sudo)${NC}"
   echo "   Usage: sudo bash scripts/setup_power_sensor.sh"
   exit 1
fi

echo -e "${GREEN}✅ Running as root${NC}"
echo ""

################################################################################
# 1. System Information
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}1. System Information${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "🖥️  Kernel: $(uname -r)"
echo "📦 Distribution: $(lsb_release -d | cut -f2)"
echo ""

# Detect GPU
echo "🎮 GPU Detection:"
lspci -nn | grep -i 'VGA\|Display\|3D' || echo "   ⚠️  No GPU detected"
echo ""

################################################################################
# 2. Check Current Driver Status
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}2. Current Driver Status${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check loaded modules
echo "📋 Loaded AMD GPU Modules:"
lsmod | grep amdgpu || echo "   ⚠️  amdgpu module not loaded"
echo ""

# Check current hwmon status
echo "📊 Current Power Sensors:"
if ls /sys/class/hwmon/hwmon*/power1_average 2>/dev/null; then
    echo -e "${GREEN}   ✅ Power sensors found!${NC}"
    for sensor in /sys/class/hwmon/hwmon*/power1_average; do
        hwmon_dir=$(dirname "$sensor")
        name=$(cat "$hwmon_dir/name" 2>/dev/null || echo "unknown")
        power=$(cat "$sensor" 2>/dev/null || echo "0")
        power_w=$(echo "scale=2; $power / 1000000" | bc)
        echo "      $name: ${power_w}W"
    done
else
    echo -e "${YELLOW}   ⚠️  No power sensors found${NC}"
fi
echo ""

# Check if amdgpu hwmon exists
echo "🔍 AMD GPU hwmon devices:"
for hwmon in /sys/class/hwmon/hwmon*; do
    name=$(cat "$hwmon/name" 2>/dev/null || echo "unknown")
    if [[ "$name" == *"amdgpu"* ]] || [[ "$name" == *"radeon"* ]]; then
        echo "   ✅ $hwmon: $name"
        echo "      Files available:"
        ls -1 "$hwmon"/{temp,power,in,fan}* 2>/dev/null | sed 's/^/         /' || echo "         (none)"
    fi
done
echo ""

################################################################################
# 3. Check GRUB Configuration
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}3. GRUB Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

GRUB_FILE="/etc/default/grub"

if [ -f "$GRUB_FILE" ]; then
    echo "📄 Current GRUB_CMDLINE_LINUX:"
    grep "^GRUB_CMDLINE_LINUX=" "$GRUB_FILE" || echo "   (not set)"
    echo ""
    
    # Check if ppfeaturemask is already set
    if grep -q "amdgpu.ppfeaturemask" "$GRUB_FILE"; then
        echo -e "${GREEN}✅ amdgpu.ppfeaturemask already configured${NC}"
        NEEDS_GRUB_UPDATE=false
    else
        echo -e "${YELLOW}⚠️  amdgpu.ppfeaturemask not configured${NC}"
        NEEDS_GRUB_UPDATE=true
    fi
else
    echo -e "${RED}❌ GRUB config file not found${NC}"
    NEEDS_GRUB_UPDATE=false
fi
echo ""

################################################################################
# 4. Backup Current Configuration
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}4. Backup Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

BACKUP_DIR="/root/amdgpu_power_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "💾 Creating backup in: $BACKUP_DIR"

# Backup GRUB
if [ -f "$GRUB_FILE" ]; then
    cp "$GRUB_FILE" "$BACKUP_DIR/grub.backup"
    echo "   ✅ GRUB configuration backed up"
fi

# Backup module parameters
if [ -f "/etc/modprobe.d/amdgpu.conf" ]; then
    cp "/etc/modprobe.d/amdgpu.conf" "$BACKUP_DIR/amdgpu_modprobe.backup"
    echo "   ✅ Module configuration backed up"
fi

# Save current module info
lsmod | grep amdgpu > "$BACKUP_DIR/lsmod.txt" 2>&1 || true
dmesg | grep -i amdgpu > "$BACKUP_DIR/dmesg.txt" 2>&1 || true

echo "   ✅ System information saved"
echo ""

################################################################################
# 5. Apply Configuration Changes
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}5. Configuration Changes${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ "$NEEDS_GRUB_UPDATE" = true ]; then
    echo "🔧 Updating GRUB configuration..."
    
    # Add ppfeaturemask to GRUB_CMDLINE_LINUX
    sed -i.bak 's/GRUB_CMDLINE_LINUX="\(.*\)"/GRUB_CMDLINE_LINUX="\1 amdgpu.ppfeaturemask=0xffffffff"/' "$GRUB_FILE"
    
    # Clean up double spaces
    sed -i 's/  / /g' "$GRUB_FILE"
    sed -i 's/" /"/' "$GRUB_FILE"
    
    echo "   ✅ Added: amdgpu.ppfeaturemask=0xffffffff"
    echo ""
    
    echo "📄 New GRUB_CMDLINE_LINUX:"
    grep "^GRUB_CMDLINE_LINUX=" "$GRUB_FILE"
    echo ""
    
    echo "🔄 Updating GRUB..."
    if command -v update-grub &> /dev/null; then
        update-grub
        echo -e "${GREEN}   ✅ GRUB updated successfully${NC}"
    elif command -v grub-mkconfig &> /dev/null; then
        grub-mkconfig -o /boot/grub/grub.cfg
        echo -e "${GREEN}   ✅ GRUB updated successfully${NC}"
    else
        echo -e "${RED}   ❌ Could not find GRUB update command${NC}"
        echo "   Please run manually: sudo update-grub"
    fi
else
    echo "ℹ️  GRUB already configured, skipping update"
fi

echo ""

# Update modprobe configuration
echo "🔧 Updating module configuration..."
MODPROBE_CONF="/etc/modprobe.d/amdgpu.conf"

cat > "$MODPROBE_CONF" << 'EOF'
# AMD GPU Power Management Configuration
# Enables full power monitoring features

options amdgpu ppfeaturemask=0xffffffff
options amdgpu dpm=1
EOF

echo -e "${GREEN}   ✅ Created $MODPROBE_CONF${NC}"
echo ""

################################################################################
# 6. Update initramfs
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}6. Update initramfs${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "🔄 Updating initramfs..."
if command -v update-initramfs &> /dev/null; then
    update-initramfs -u
    echo -e "${GREEN}   ✅ initramfs updated${NC}"
elif command -v dracut &> /dev/null; then
    dracut --force
    echo -e "${GREEN}   ✅ initramfs updated${NC}"
else
    echo -e "${YELLOW}   ⚠️  Could not update initramfs automatically${NC}"
fi
echo ""

################################################################################
# 7. Summary and Next Steps
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}7. Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -e "${GREEN}✅ Configuration Complete!${NC}"
echo ""
echo "📋 Changes made:"
echo "   • Added amdgpu.ppfeaturemask=0xffffffff to GRUB"
echo "   • Created /etc/modprobe.d/amdgpu.conf"
echo "   • Updated initramfs"
echo "   • Backup saved in: $BACKUP_DIR"
echo ""

echo -e "${YELLOW}⚠️  REBOOT REQUIRED${NC}"
echo ""
echo "Next steps:"
echo "   1. Reboot your system:"
echo "      ${BLUE}sudo reboot${NC}"
echo ""
echo "   2. After reboot, verify power sensor:"
echo "      ${BLUE}python3 scripts/diagnose_power_monitoring.py${NC}"
echo ""
echo "   3. Test power monitoring:"
echo "      ${BLUE}python3 scripts/power_monitor.py --duration 5 --verbose${NC}"
echo ""

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create verification script for after reboot
cat > /usr/local/bin/verify_amdgpu_power.sh << 'VERIFY_EOF'
#!/bin/bash
echo "AMD GPU Power Sensor Verification"
echo "=================================="
echo ""
echo "Kernel parameters:"
cat /proc/cmdline | grep -o "amdgpu[^ ]*" || echo "  No amdgpu parameters found"
echo ""
echo "Power sensors:"
for sensor in /sys/class/hwmon/hwmon*/power1_average; do
    if [ -f "$sensor" ]; then
        hwmon=$(dirname "$sensor")
        name=$(cat "$hwmon/name" 2>/dev/null || echo "unknown")
        power=$(cat "$sensor" 2>/dev/null || echo "0")
        power_w=$(echo "scale=2; $power / 1000000" | bc)
        echo "  ✅ $name: ${power_w}W"
    fi
done
[ ! -f /sys/class/hwmon/hwmon*/power1_average ] && echo "  ❌ No power sensors found"
VERIFY_EOF

chmod +x /usr/local/bin/verify_amdgpu_power.sh

echo "📝 Quick verification command created:"
echo "   ${BLUE}sudo verify_amdgpu_power.sh${NC}"
echo ""

################################################################################
# 8. Rollback Information
################################################################################

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Rollback Information${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo "If you need to rollback these changes:"
echo ""
echo "1. Restore GRUB:"
echo "   ${BLUE}sudo cp $BACKUP_DIR/grub.backup /etc/default/grub${NC}"
echo "   ${BLUE}sudo update-grub${NC}"
echo ""
echo "2. Remove module config:"
echo "   ${BLUE}sudo rm /etc/modprobe.d/amdgpu.conf${NC}"
echo ""
echo "3. Update initramfs and reboot:"
echo "   ${BLUE}sudo update-initramfs -u${NC}"
echo "   ${BLUE}sudo reboot${NC}"
echo ""

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}Setup complete! Please reboot your system.${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
