# Claude Code Configuration for Obvivlorum

## Project Overview
Obvivlorum is an advanced AI symbiosis platform with consciousness research capabilities, implementing both traditional IIT/GWT approaches and Francisco Molina's Topo-Spectral consciousness framework.

**ESTADO ACTUAL**: Pipeline Científico de Producción - FASE 2 EN PROGRESO
- ✅ FASE 1 COMPLETADA: Auditoría crítica de rendimiento y análisis
- ✅ FASE 2 PARCIAL: Formalizaciones matemáticas críticas (Quantum formalism)
- 🔄 FASE 2 CONTINUACIÓN: Sistema holográfico de memoria
- 📋 FASE 3 PLANIFICADA: Optimización de cuellos de botella (53ms → <5ms)
- 📋 FASE 4-6: Publicación científica IEEE Neural Networks

## Execution Modes

### Mode Configuration
The system supports multiple execution modes through the `--mode` parameter:

```bash
# Standard consciousness metrics (IIT/GWT only)
python ai_symbiote.py --mode=standard

# Topo-Spectral consciousness framework  
python ai_symbiote.py --mode=topoespectro

# Full research mode (both frameworks)
python ai_symbiote.py --mode=research

# GUI mode with mode selection
python ai_symbiote_gui.py
```

### Topo-Spectral Framework Toggle
- **Standard Mode**: Uses basic IIT/GWT consciousness metrics
- **Topo-Spectral Mode**: Activates Francisco Molina's research framework
- **Research Mode**: Enables both for comparison studies

## Development Commands

### Testing
```bash
# Run standard tests
python -m pytest tests/

# Run research validation (includes Topo-Spectral)
python -m pytest research_tests/ -m "not slow"

# Full benchmark suite
python -m pytest benchmarks/ --benchmark-only
```

### Build and Package
```bash
# Create executable
python -m PyInstaller --onefile ai_symbiote.py

# Create portable version
./CREATE_PORTABLE_VERSION.bat
```

### Research Components

#### Topo-Spectral Framework
- **Spectral Information Integration**: O(n³) polynomial-time approximation
- **Topological Resilience**: Persistent homology via Rips filtration  
- **Temporal Synchronization**: Variance-based stability analysis
- **Consciousness Index**: Ψ(St) = ³√(Φ̂spec(St) · T̂(St) · Sync(St))

#### Neuroplasticity Engine
- **7 Plasticity Types**: Hebbian, STDP, BCM, Oja, Homeostatic, Metaplastic
- **Real-time Adaptation**: Background threads for continuous learning
- **Topology Evolution**: Activity-based structural changes
- **Performance**: Numba-accelerated computations

### Scientific Validation
The framework has been validated against:
- **Synthetic Networks**: 94.7% classification accuracy (n=5,000)
- **Clinical EEG Data**: Temple University Hospital corpus (n=2,847)  
- **Comparative Benchmarks**: PCI, LZ Complexity, ΦACE metrics

## Configuration Files

### Runtime Configuration
- `config/topoespectral_config.json`: Topo-Spectral parameters
- `config/neuroplasticity_config.json`: Neural network settings
- `config/consciousness_config.json`: Consciousness assessment parameters

### Environment Variables
```bash
# Enable Topo-Spectral by default
export OBVIVLORUM_MODE=topoespectro

# Research data directory
export OBVIVLORUM_RESEARCH_DATA=/path/to/eeg/data

# Performance optimization
export OBVIVLORUM_USE_NUMBA=true
```

## Research Reproducibility

### Data Requirements
- EEG datasets in European Data Format (EDF)
- Synthetic network generation parameters
- Benchmark comparison results

### Citation
```bibtex
@software{obvivlorum2024,
  title={Obvivlorum AI Symbiote System},
  author={Molina, Francisco},
  year={2024},
  url={https://github.com/Yatrogenesis/Obvivlorum},
  note={Advanced AI symbiosis platform with Topo-Spectral consciousness research}
}
```

## Dependencies
- **Core**: numpy, scipy, networkx, numba
- **Research**: scikit-learn, matplotlib, seaborn  
- **Topology**: ripser, persim, gudhi
- **Neural**: torch, tensorflow (optional)

## Performance Notes
- Topo-Spectral mode requires ~2GB RAM for 1000-node networks
- Standard mode suitable for resource-constrained environments
- GPU acceleration available for large-scale experiments

## Support
- GitHub Issues: https://github.com/Yatrogenesis/Obvivlorum/issues
- Research Collaboration: pako.molina@gmail.com
- ORCID: https://orcid.org/0009-0008-6093-8267