# 📚 Documentation Index
## Radeon RX 580 Energy-Efficient Computing Framework v1.0.0

**Last Updated**: February 2, 2026  
**Project Status**: ✅ Production Ready

---

## 📖 Quick Navigation

### Essential Documents (Root Directory)

| Document | Description | Audience |
|----------|-------------|----------|
| [README.md](../README.md) | Main project overview and quick start | Everyone |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guidelines | Developers |
| [LICENSE](../LICENSE) | MIT License | Everyone |

---

## 📚 Documentation Structure

```
docs/
├── 📁 guides/           # User and developer guides
│   ├── QUICKSTART.md    # 5-minute setup guide
│   ├── USER_GUIDE.md    # Complete usage guide
│   ├── DEVELOPER_GUIDE.md # SDK and API reference
│   ├── DRIVER_SETUP_RX580.md # Driver installation
│   └── CLUSTER_DEPLOYMENT_GUIDE.md # Distributed deployment
├── 📁 paper/            # Academic paper (LaTeX)
│   └── paper-energy-efficient-polaris/
├── 📁 archive/          # Historical documentation (100+ files)
│   ├── sessions/        # Development session logs
│   ├── phases/          # Phase completion reports
│   └── tasks/           # Task tracking documents
├── 📁 api_reference/    # API documentation
├── architecture.md      # System architecture
├── CHANGELOG.md         # Version history
├── contributing.md      # Contributing guide
├── MODEL_GUIDE.md       # Model loading and usage
├── optimization.md      # Performance tuning guide
├── OPTIMIZATION_ROADMAP.md # Future optimization plans
└── use_cases.md         # Real-world applications
```

---

## 🎓 User Documentation

### Getting Started

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](guides/QUICKSTART.md) | 5-minute setup guide |
| [DRIVER_SETUP_RX580.md](guides/DRIVER_SETUP_RX580.md) | Complete driver installation |
| [USER_GUIDE.md](guides/USER_GUIDE.md) | Complete usage guide |

### Technical Guides

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System design and components |
| [optimization.md](optimization.md) | Performance tuning guide |
| [OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md) | Future optimization plans |
| [MODEL_GUIDE.md](MODEL_GUIDE.md) | Model loading and usage |
| [NAS_IMPLEMENTATION.md](NAS_IMPLEMENTATION.md) | Neural Architecture Search guide |
| [use_cases.md](use_cases.md) | Real-world applications |
| [CLUSTER_DEPLOYMENT_GUIDE.md](guides/CLUSTER_DEPLOYMENT_GUIDE.md) | Distributed deployment |

### Hardware & Optimization

| Document | Description |
|----------|-------------|
| [ROADMAP_README.md](ROADMAP_README.md) | 📋 Roadmap system guide |
| [ROADMAP_OPTIMIZATION.md](ROADMAP_OPTIMIZATION.md) | 🎯 5-phase optimization plan |
| [PROGRESS_TRACKING.md](PROGRESS_TRACKING.md) | 📊 Daily progress tracking |
| [VALIDATION_REPORT_SESSION29.md](VALIDATION_REPORT_SESSION29.md) | ✅ Session 29 validation |
| [../results/hardware_benchmark_rx590_gme.md](../results/hardware_benchmark_rx590_gme.md) | 🔥 RX 590 GME benchmark |

### Developer Documentation

| Document | Description |
|----------|-------------|
| [DEVELOPER_GUIDE.md](guides/DEVELOPER_GUIDE.md) | SDK and API reference |
| [contributing.md](contributing.md) | How to contribute |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 📄 Academic Paper

The complete academic paper documenting this research:

**Location**: [paper/paper-energy-efficient-polaris/](paper/paper-energy-efficient-polaris/)

| Section | File | Description |
|---------|------|-------------|
| Main | `main.tex` | Complete paper (44 pages) |
| Abstract | `sections/abstract.tex` | Paper summary |
| Introduction | `sections/introduction.tex` | Motivation and objectives |
| Methodology | `sections/methodology.tex` | Experimental setup |
| Results | `sections/experimental_results.tex` | Benchmarks and validation |
| Conclusions | `sections/conclusions.tex` | Key findings |

**Compile**: `cd docs/paper/paper-energy-efficient-polaris && make all`

---

## 🗃️ Archive

Historical documentation is preserved in `docs/archive/` for reference:

| Subdirectory | Contents |
|--------------|----------|
| `sessions/` | Development session logs (Sessions 1-35) |
| `phases/` | Phase completion reports |
| `tasks/` | Task tracking documents |
| Root | Technical reports, validation results, research notes |

**Note**: Archive contains ~100+ historical files for project history and auditing.

---

## 🔧 Phase Documentation

Each optimization phase has its own README:

| Phase | Location | Description |
|-------|----------|-------------|
| Fase 6 | `fase_6_winograd/README.md` | Winograd transforms |
| Fase 7 | `fase_7_ai_kernel_predictor/README.md` | AI kernel predictor |
| Fase 8 | `fase_8_bayesian_optimization/README.md` | Bayesian optimization |
| Fase 9 | `fase_9_breakthrough_integration/README.md` | Breakthrough integration |
| Fase 10 | `fase_10_tensor_core_simulation/README.md` | Tensor core simulation |
| Fase 10b | `fase_10_multi_gpu/README.md` | Multi-GPU support |
| Fase 11 | `fase_11_winograd_transform/README.md` | Winograd integration |
| Fase 12 | `fase_12_mixed_precision/README.md` | Mixed precision |
| Fase 13 | `fase_13_gcn_architecture/README.md` | GCN architecture tuning |
| Fase 14 | `fase_14_ai_kernel_predictor/README.md` | Enhanced AI predictor |
| Fase 15 | `fase_15_bayesian_optimization/README.md` | Advanced Bayesian |
| Fase 16 | `fase_16_quantum_inspired_methods/README.md` | Quantum-inspired methods |
| Fase 17 | `fase_17_neuromorphic_computing/README.md` | Neuromorphic computing |
| Fase 18 | `fase_18_hybrid_quantum_classical/README.md` | Hybrid quantum-classical |

---

## 📊 Key Metrics

| Metric | Value |
|--------|-------|
| **Peak Performance** | 95.6 GFLOPS |
| **ML Selector Accuracy** | 94.2% |
| **Energy Efficiency** | 4.05 GFLOPS/W |
| **Total Improvement** | 14.8× over baseline |

---

*Documentation consolidated: February 2, 2026*
