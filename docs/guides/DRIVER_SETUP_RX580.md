# Driver Setup Guide - AMD Radeon RX 580 (Polaris)

**Complete guide for installing and configuring drivers for optimal AI inference performance**

---

## 📋 Table of Contents

- [TL;DR - Quick Setup](#tldr---quick-setup)
- [Understanding Driver Options](#understanding-driver-options)
- [Recommended Stack (Mesa)](#recommended-stack-mesa)
- [Step-by-Step Installation](#step-by-step-installation)
- [Verification & Testing](#verification--testing)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Driver Comparison](#driver-comparison)
- [FAQ](#faq)

---

## TL;DR - Quick Setup

**For Ubuntu 20.04+ / Debian-based systems** (recommended setup):

```bash
# Install complete Mesa + OpenCL stack
sudo apt update
sudo apt install -y \
    mesa-opencl-icd \
    mesa-vulkan-drivers \
    clinfo \
    vulkan-tools \
    mesa-utils \
    ocl-icd-opencl-dev \
    opencl-headers \
    libdrm-amdgpu1

# Add user to video/render groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Logout and login, then verify
clinfo --list
vulkaninfo --summary
python scripts/verify_drivers.py
```

**Expected output:**
```
✅ DRIVERS ARE OPTIMAL FOR INFERENCE
   Your RX 580 is ready for AI workloads!
```

---

## Understanding Driver Options

The AMD Radeon RX 580 (Polaris/GCN 4.0) has **three main driver options**:

### 1. Mesa (Open Source) ✅ **RECOMMENDED**

**What it is:**
- Open-source graphics stack for Linux
- Kernel driver: **AMDGPU** (in-tree, always updated)
- OpenCL: **Clover** (OpenCL 1.2) or **RustiCL** (OpenCL 3.0)
- Vulkan: **RADV** driver

**Status for Polaris:** ✅ Actively maintained, full support

**Pros:**
- ✅ Included in mainline Linux kernel (no installation needed)
- ✅ Active development and bug fixes
- ✅ Excellent stability and compatibility
- ✅ Best integration with modern Linux systems
- ✅ Free and open source

**Cons:**
- ⚠️ ~5-10% slower than AMD PRO in some workloads (negligible for inference)

**Recommendation:** ✅ **Use Mesa for all RX 580 AI workloads**

---

### 2. AMD AMDGPU-PRO (Proprietary) ⚠️ **LEGACY/DEPRECATED**

**What it is:**
- AMD's proprietary driver package
- Includes optimized OpenCL runtime
- Last version: AMDGPU-PRO 23.20

**Status for Polaris:** ⚠️ Deprecated since 2023, no new updates

**Pros:**
- ✅ Slightly better performance in some compute workloads (~5-10%)

**Cons:**
- ❌ No longer maintained for Polaris
- ❌ Compatibility issues with modern kernels
- ❌ No security updates
- ❌ Complex installation/removal
- ❌ Conflicts with Mesa

**Recommendation:** ❌ **Avoid - use Mesa instead**

---

### 3. ROCm Platform ⚠️ **LIMITED POLARIS SUPPORT**

**What it is:**
- AMD's open compute platform (GPU compute ecosystem)
- Includes HIP, MIOpen, rocBLAS
- Focused on datacenter GPUs (Instinct, RDNA)

**Status for Polaris:**
- ROCm 5.7: Last version with Polaris support (deprecated)
- ROCm 6.x: **No official Polaris support**

**Pros:**
- ✅ Advanced ML libraries (MIOpen)
- ✅ Good for RDNA/CDNA GPUs

**Cons:**
- ❌ ROCm 6.x doesn't support Polaris
- ❌ ROCm 5.x is deprecated
- ❌ Complex installation
- ❌ Not needed for inference (OpenCL sufficient)

**Recommendation:** ⚠️ **Optional - OpenCL/Vulkan preferred for RX 580**

---

## Recommended Stack (Mesa)

Our framework uses this **production-tested stack**:

```
┌─────────────────────────────────────┐
│    AI Inference Framework (This)    │
├─────────────────────────────────────┤
│  ONNX Runtime │ PyTorch │ OpenCV    │ ← ML Libraries
├─────────────────────────────────────┤
│  OpenCL 1.2+  │ Vulkan 1.3          │ ← Compute APIs
├─────────────────────────────────────┤
│  Mesa Clover  │ Mesa RADV           │ ← Mesa Implementations
├─────────────────────────────────────┤
│        AMDGPU Kernel Driver         │ ← Kernel Module
├─────────────────────────────────────┤
│       Hardware (RX 580/Polaris)     │
└─────────────────────────────────────┘
```

**Why this stack:**
- ✅ Fully open source and maintained
- ✅ Best compatibility with modern Linux
- ✅ Sufficient performance for inference
- ✅ Stable and well-tested
- ✅ Simple to install and maintain

---

## Step-by-Step Installation

### Prerequisites

- **OS:** Ubuntu 20.04+ or Debian-based Linux
- **Kernel:** Linux 5.4+ (check with `uname -r`)
- **GPU:** AMD Radeon RX 580 properly installed
- **Permissions:** Sudo access

### Step 1: Update System

```bash
# Always start with updated packages
sudo apt update
sudo apt upgrade -y

# Check kernel version (should be 5.4+)
uname -r
```

### Step 2: Verify Kernel Driver

```bash
# Check if AMDGPU is loaded
lsmod | grep amdgpu

# Should output: amdgpu ... (with size and usage count)
```

If **not loaded**, load it manually:

```bash
# Load driver
sudo modprobe amdgpu

# Make it permanent
echo "amdgpu" | sudo tee -a /etc/modules

# Reboot to ensure it loads on boot
sudo reboot
```

### Step 3: Install Mesa Stack

```bash
# Install core Mesa packages
sudo apt install -y \
    mesa-opencl-icd \
    mesa-vulkan-drivers \
    libgl1-mesa-dri \
    libglx-mesa0 \
    mesa-common-dev \
    mesa-utils \
    libdrm-amdgpu1

# Verify Mesa installation
glxinfo | grep "OpenGL version"
# Expected: OpenGL version string: 4.x Mesa xx.x.x
```

### Step 4: Install OpenCL Support

```bash
# Install OpenCL runtime and tools
sudo apt install -y \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    ocl-icd-libopencl1

# Verify OpenCL installation
clinfo --list

# Expected output:
# Platform #0: Clover
#  `-- Device #0: AMD Radeon RX 580
```

**If OpenCL not detected:**

```bash
# Check ICD loader configuration
ls /etc/OpenCL/vendors/

# Should contain: mesa.icd or rusticl.icd

# If missing, reinstall:
sudo apt install --reinstall mesa-opencl-icd
```

### Step 5: Install Vulkan Support

```bash
# Install Vulkan runtime and tools
sudo apt install -y \
    mesa-vulkan-drivers \
    vulkan-tools \
    vulkan-validationlayers \
    libvulkan1

# Verify Vulkan installation
vulkaninfo --summary

# Expected output should include:
# GPU0: AMD Radeon RX 580
# apiVersion = 1.3.xxx
# driverName = radv
```

**If Vulkan not detected:**

```bash
# Check ICD configuration
ls /usr/share/vulkan/icd.d/

# Should contain: radeon_icd.x86_64.json

# If missing, reinstall:
sudo apt install --reinstall mesa-vulkan-drivers
```

### Step 6: Set Permissions

```bash
# Add current user to video and render groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Check group membership
groups $USER

# Should include: ... video render ...

# IMPORTANT: Logout and login for changes to take effect
# Or run: newgrp video && newgrp render
```

### Step 7: Verify Installation

```bash
# Clone and setup the framework (if not done already)
cd ~/Proyectos/Programacion/Radeon_RX_580
source venv/bin/activate

# Run driver verification
python scripts/verify_drivers.py

# Expected output:
# ✅ DRIVERS ARE OPTIMAL FOR INFERENCE
#    Your RX 580 is ready for AI workloads!
```

**Also verify hardware detection:**

```bash
python scripts/verify_hardware.py

# Expected:
# ✅ GPU Detected: AMD Radeon RX 580
# ✅ OpenCL: Available
# ⚠️  ROCm: Not available (optional)
```

---

## Verification & Testing

### Quick Verification Commands

```bash
# 1. Kernel driver
lsmod | grep amdgpu
# Expected: amdgpu module loaded

# 2. Mesa version
glxinfo | grep "OpenGL version"
# Expected: OpenGL version string: 4.x Mesa 22.x+

# 3. OpenCL platforms
clinfo --list
# Expected: Platform #0: Clover with AMD device

# 4. Vulkan devices
vulkaninfo --summary
# Expected: AMD Radeon RX 580 with RADV driver

# 5. GPU PCI info
lspci -v -s $(lspci | grep VGA | cut -d' ' -f1)
# Expected: Kernel driver in use: amdgpu
```

### Complete System Check

```bash
# Run comprehensive diagnostics
python scripts/diagnostics.py

# Run driver health check
python scripts/verify_drivers.py --verbose

# Run hardware verification
python scripts/verify_hardware.py
```

### Test Inference

```bash
# Test with a simple model
python examples/demo_real_simple.py

# Should output:
# Prediction: tiger cat (score: 0.87)
# ✅ Inference successful!
```

---

## Performance Tuning

### GPU Power Management

```bash
# Check current power state
cat /sys/class/drm/card0/device/power_dpm_state

# Enable dynamic power management (recommended)
echo "auto" | sudo tee /sys/class/drm/card0/device/power_dpm_state

# Set performance level (for inference workloads)
echo "high" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level

# Make permanent by adding to /etc/rc.local or systemd service
```

### GPU Clock Settings

```bash
# View available power profiles
cat /sys/class/drm/card0/device/pp_power_profile_mode

# Set to compute profile (profile 5)
echo "5" | sudo tee /sys/class/drm/card0/device/pp_power_profile_mode
```

### Memory Clock

```bash
# View memory clock ranges
cat /sys/class/drm/card0/device/pp_dpm_mclk

# Set to maximum (usually level 2 for RX 580)
echo "2" | sudo tee /sys/class/drm/card0/device/pp_dpm_mclk
```

### System Tuning

```bash
# Disable CPU frequency scaling for consistent inference
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase system file limits for large models
ulimit -n 65536
```

**Make permanent** by creating `/etc/systemd/system/amd-gpu-tuning.service`:

```ini
[Unit]
Description=AMD GPU Performance Tuning
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo auto > /sys/class/drm/card0/device/power_dpm_state'
ExecStart=/bin/bash -c 'echo high > /sys/class/drm/card0/device/power_dpm_force_performance_level'
ExecStart=/bin/bash -c 'echo 5 > /sys/class/drm/card0/device/pp_power_profile_mode'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable amd-gpu-tuning.service
sudo systemctl start amd-gpu-tuning.service
```

---

## Troubleshooting

### Issue 1: GPU Not Detected

**Symptoms:**
```bash
lspci | grep VGA
# Shows GPU but no driver
lsmod | grep amdgpu
# No output
```

**Fix:**
```bash
# Load AMDGPU module
sudo modprobe amdgpu

# Check for errors
dmesg | grep -i amdgpu | tail -20

# If firmware missing, install:
sudo apt install linux-firmware

# Reboot
sudo reboot
```

---

### Issue 2: OpenCL Not Available

**Symptoms:**
```bash
clinfo --list
# No platforms detected
```

**Fix:**
```bash
# Reinstall OpenCL stack
sudo apt install --reinstall mesa-opencl-icd ocl-icd-libopencl1

# Check ICD files
ls /etc/OpenCL/vendors/
# Should contain mesa.icd or rusticl.icd

# If missing, create manually:
echo "/usr/lib/x86_64-linux-gnu/libMesaOpenCL.so.1" | \
    sudo tee /etc/OpenCL/vendors/mesa.icd

# Verify
clinfo --list
```

---

### Issue 3: Permission Denied Errors

**Symptoms:**
```bash
python scripts/verify_hardware.py
# Error: Permission denied accessing /dev/dri/renderD128
```

**Fix:**
```bash
# Add user to groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Check device permissions
ls -la /dev/dri/
# Should show: crw-rw----+ video render

# If wrong permissions, fix:
sudo chmod 660 /dev/dri/*

# Logout and login
logout
```

---

### Issue 4: Vulkan Not Working

**Symptoms:**
```bash
vulkaninfo --summary
# Cannot create Vulkan instance
```

**Fix:**
```bash
# Reinstall Vulkan
sudo apt install --reinstall mesa-vulkan-drivers libvulkan1

# Check ICD
ls /usr/share/vulkan/icd.d/
# Should have: radeon_icd.x86_64.json

# Set environment variable (if needed)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json

# Verify
vulkaninfo --summary
```

---

### Issue 5: Poor Performance

**Check:**
```bash
# 1. GPU power state
cat /sys/class/drm/card0/device/power_dpm_state
# Should be: auto

# 2. Performance level
cat /sys/class/drm/card0/device/power_dpm_force_performance_level
# Should be: high

# 3. GPU clocks
cat /sys/class/drm/card0/device/pp_dpm_sclk
cat /sys/class/drm/card0/device/pp_dpm_mclk
# Highest level should have '*' marker

# 4. Check throttling
sudo dmesg | grep -i thermal
# Should not show thermal throttling
```

**Fix:**
- Apply performance tuning from section above
- Check cooling (GPU temp should be < 80°C)
- Use `radeontop` to monitor GPU usage

---

### Issue 6: Mesa Version Too Old

**Symptoms:**
```bash
glxinfo | grep "OpenGL version"
# OpenGL version string: 4.x Mesa 20.x.x  (< 22.0)
```

**Fix:**
```bash
# Add Mesa PPA (for Ubuntu)
sudo add-apt-repository ppa:kisak/kisak-mesa
sudo apt update
sudo apt upgrade

# Verify
glxinfo | grep "OpenGL version"
# Should show Mesa 22.x or newer

# Alternative: Use official Ubuntu backports
sudo add-apt-repository -y ppa:ubuntu-x-swat/updates
sudo apt update
sudo apt upgrade mesa-*
```

---

## Driver Comparison

Comprehensive comparison of driver options:

| Feature | Mesa (Clover/RADV) | AMD PRO | ROCm |
|---------|-------------------|---------|------|
| **Polaris Support** | ✅ Full | ⚠️ Legacy (EOL) | ⚠️ Limited (v5.x only) |
| **Active Maintenance** | ✅ Yes | ❌ No | ⚠️ No for Polaris |
| **Kernel Version** | ✅ Always latest | ⚠️ Conflicts | ⚠️ Specific versions |
| **OpenCL Version** | 1.2/3.0 (RustiCL) | 2.0 | 2.0 |
| **Vulkan Support** | ✅ 1.3 | ✅ 1.3 | ❌ N/A |
| **Ease of Install** | ✅ Simple (`apt`) | ⚠️ Complex | ❌ Very complex |
| **Stability** | ✅ Excellent | ⚠️ Good | ⚠️ Fair |
| **Inference Performance** | ✅ Excellent | ✅ ~5% better | ⚠️ Not tested |
| **Framework Compat** | ✅ Full | ⚠️ Good | ⚠️ Partial |
| **License** | ✅ Open (MIT) | ⚠️ Proprietary | ✅ Open (Custom) |
| **Community Support** | ✅ Excellent | ❌ Limited | ⚠️ For newer GPUs |
| **Documentation** | ✅ Extensive | ⚠️ Outdated | ✅ Good |
| **Cost** | ✅ Free | ✅ Free | ✅ Free |
| **Removal** | ✅ Easy | ⚠️ Can break system | ⚠️ Complex |
| **Update Frequency** | ✅ Regular | ❌ None | ⚠️ For RDNA only |
| **Our Recommendation** | ✅ **USE THIS** | ❌ **AVOID** | ⚠️ **OPTIONAL** |

---

## FAQ

### Q: Do I need ROCm for the RX 580?

**A:** No. ROCm 6.x doesn't officially support Polaris, and ROCm 5.x is deprecated. OpenCL (Mesa) provides all the compute capabilities needed for inference. Use OpenCL or Vulkan instead.

---

### Q: Will AMDGPU-PRO give better performance?

**A:** Marginally (~5-10%) in some compute workloads, but it's **not maintained anymore** for Polaris. The stability and compatibility issues aren't worth the small performance gain. Mesa is recommended.

---

### Q: Which is better: OpenCL or Vulkan?

**A:** Both work well:
- **OpenCL**: More mature support, widely compatible with ML frameworks
- **Vulkan**: Slightly better performance in some cases, more modern API

Our framework supports both. Try OpenCL first (better compatibility), then Vulkan if you want to optimize further.

---

### Q: Can I use both Mesa and AMD PRO drivers?

**A:** **No.** They conflict with each other. Installing AMD PRO will replace Mesa's OpenCL, which can cause system instability. Stick with Mesa.

---

### Q: My Mesa version is old (< 22.0). Should I update?

**A:** Recommended but not critical. Mesa 22.x+ has better Polaris support and bug fixes. Update using the PPA method shown in troubleshooting.

---

### Q: How do I check which driver I'm currently using?

**A:** Run:
```bash
# Check OpenCL implementation
clinfo | grep "Platform Name"
# Clover/RustiCL = Mesa, AMD APP = AMD PRO

# Check Vulkan driver
vulkaninfo --summary | grep "driverName"
# radv = Mesa, amdvlk = AMD official Vulkan

# Check kernel module
lsmod | grep -E "amdgpu|radeon"
# amdgpu = modern driver (correct)
```

---

### Q: Performance is worse than expected. What should I check?

**A:** Common issues:
1. **Thermal throttling** - Check GPU temperature (`sensors` command)
2. **Power management** - Set to "high" performance mode (see tuning section)
3. **CPU bottleneck** - Monitor CPU usage during inference
4. **Model not optimized** - Use INT8 quantization for better throughput
5. **Batch size** - Increase batch size for better GPU utilization

Run: `python scripts/verify_drivers.py --verbose` for diagnostics.

---

### Q: Do I need to install anything for multi-GPU setups?

**A:** Mesa handles multi-GPU automatically. For inference load balancing across multiple RX 580s, see [docs/CLUSTER_DEPLOYMENT_GUIDE.md](../CLUSTER_DEPLOYMENT_GUIDE.md).

---

### Q: Can I use proprietary NVIDIA drivers alongside Mesa for AMD?

**A:** Yes. NVIDIA and AMD drivers don't conflict. You can have both installed on the same system.

---

### Q: What about Windows support?

**A:** This framework targets Linux (best AMD driver support). For Windows:
- Use AMD Adrenalin drivers (latest available)
- OpenCL support is included
- Performance may vary compared to Linux

---

### Q: How do I completely remove AMD PRO drivers?

**A:** If you have AMD PRO installed:
```bash
# Remove AMD PRO packages
sudo apt remove --purge amdgpu-pro-*
sudo apt remove --purge rocm-*

# Reinstall Mesa
sudo apt install mesa-opencl-icd mesa-vulkan-drivers

# Clean up
sudo apt autoremove
sudo apt autoclean

# Reboot
sudo reboot
```

---

## Additional Resources

### Official Documentation
- [Mesa3D Project](https://www.mesa3d.org/)
- [AMDGPU Driver Documentation](https://dri.freedesktop.org/wiki/AMDGPU/)
- [Khronos OpenCL Specification](https://www.khronos.org/opencl/)
- [Vulkan Specification](https://www.vulkan.org/)

### Community Resources
- [AMD GPU Linux Discussion](https://www.phoronix.com/forums/forum/linux-graphics-x-org-drivers/open-source-amd-linux)
- [Mesa Development](https://gitlab.freedesktop.org/mesa/mesa)
- [r/AMD Subreddit](https://www.reddit.com/r/Amd/)

### Framework Documentation
- [Main README](../../README.md)
- [Quick Start Guide](QUICKSTART.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Cluster Deployment](../CLUSTER_DEPLOYMENT_GUIDE.md)

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Run diagnostics:**
   ```bash
   python scripts/verify_drivers.py --verbose
   python scripts/diagnostics.py > diagnostics.log
   ```

2. **Check logs:**
   ```bash
   dmesg | grep -i amdgpu > dmesg.log
   ```

3. **Open an issue** on GitHub with:
   - Output from `verify_drivers.py`
   - System info (`uname -a`, Mesa version)
   - Error messages/logs

4. **Community support:**
   - GitHub Discussions
   - Project Discord (if available)

---

**Last Updated:** January 2026  
**Framework Version:** 0.7.0  
**Tested on:** Ubuntu 20.04, 22.04, Debian 11/12  
**Hardware:** AMD Radeon RX 580 (Polaris 20)
