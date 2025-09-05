#!/usr/bin/env python3
"""
Performance benchmarks for Obvivlorum core components
"""

import pytest
import time
import asyncio
import numpy as np
from memory_profiler import profile
import psutil
import os

# Import components to benchmark
from security_manager import SecurityManager
from structured_logger import StructuredLogger
from unified_launcher import UnifiedLauncher

try:
    from scientific.quantum_formalism import QuantumSymbolicProcessor, Concept
    from scientific.consciousness_metrics import ConsciousnessAssessment, create_test_network
    from scientific.neuroplasticity_engine import NeuroplasticNetwork, PlasticityRule, PlasticityType
    SCIENTIFIC_AVAILABLE = True
except ImportError:
    SCIENTIFIC_AVAILABLE = False


class BenchmarkSuite:
    """Comprehensive benchmark suite for Obvivlorum components"""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process(os.getpid())
    
    def measure_memory(self, func, *args, **kwargs):
        """Measure memory usage of a function"""
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = final_memory - initial_memory
        
        return result, memory_delta
    
    def measure_time(self, func, iterations=10, *args, **kwargs):
        """Measure execution time of a function"""
        times = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'iterations': iterations
        }


@pytest.mark.benchmark
class TestCoreBenchmarks(BenchmarkSuite):
    """Benchmarks for core system components"""
    
    def test_security_manager_threat_assessment_benchmark(self, benchmark):
        """Benchmark security manager threat assessment"""
        security = SecurityManager()
        
        async def assess_multiple_threats():
            commands = [
                "echo hello",
                "ls -la",
                "python script.py",
                "rm file.txt",
                "sudo apt install package",
                "wget http://example.com/file",
                "curl -X POST api.example.com",
                "docker run image",
                "git clone repo",
                "npm install package"
            ]
            
            results = []
            for cmd in commands:
                threat_level = await security.assess_threat_level(cmd, {})
                results.append(threat_level)
            return results
        
        result = benchmark(lambda: asyncio.run(assess_multiple_threats()))
        assert len(result) == 10
    
    def test_structured_logger_performance_benchmark(self, benchmark):
        """Benchmark structured logger performance"""
        logger = StructuredLogger("benchmark_test")
        
        def log_burst():
            for i in range(100):
                logger.info(f"Benchmark log message {i}", extra={
                    "message_id": i,
                    "batch": "performance_test",
                    "timestamp": time.time()
                })
        
        result = benchmark(log_burst)
        
        # Cleanup
        asyncio.run(logger.shutdown())
    
    def test_unified_launcher_mode_listing_benchmark(self, benchmark):
        """Benchmark unified launcher mode listing"""
        launcher = UnifiedLauncher()
        
        def list_modes_multiple():
            for _ in range(50):
                modes = list(launcher.modes.keys())
            return len(modes)
        
        result = benchmark(list_modes_multiple)
        assert result > 0


@pytest.mark.benchmark
@pytest.mark.skipif(not SCIENTIFIC_AVAILABLE, reason="Scientific modules not available")
class TestScientificBenchmarks(BenchmarkSuite):
    """Benchmarks for scientific components"""
    
    def test_quantum_processing_benchmark(self, benchmark):
        """Benchmark quantum symbolic processing"""
        processor = QuantumSymbolicProcessor(256)
        
        # Create test concepts
        concepts = [
            Concept(f"concept_{i}", np.random.random(128))
            for i in range(20)
        ]
        
        def quantum_operations():
            # Create superposition
            superposition_id = processor.create_concept_superposition(concepts)
            
            # Compute information metrics
            metrics = processor.compute_quantum_information(superposition_id)
            
            # Create entanglement
            entanglement_id = processor.entangle_concepts(concepts[0], concepts[1])
            entanglement_metrics = processor.measure_entanglement(entanglement_id)
            
            return metrics, entanglement_metrics
        
        result = benchmark(quantum_operations)
        assert len(result) == 2
    
    def test_consciousness_assessment_benchmark(self, benchmark):
        """Benchmark consciousness assessment pipeline"""
        network = create_test_network()
        assessor = ConsciousnessAssessment(network)
        
        def assess_multiple_states():
            states = [
                np.random.random(8),
                np.zeros(8),
                np.ones(8) * 0.5,
                np.array([1, 1, 1, 1, 0, 0, 0, 0]),
                np.array([0.8, 0.9, 0.7, 0.8, 0.2, 0.1, 0.3, 0.2])
            ]
            
            results = []
            for state in states:
                assessment = assessor.assess_consciousness_level(state)
                results.append(assessment)
            
            return results
        
        result = benchmark(assess_multiple_states)
        assert len(result) == 5
        
        # Verify all assessments have required keys
        for assessment in result:
            assert "phi" in assessment
            assert "consciousness_level" in assessment
    
    def test_neuroplasticity_simulation_benchmark(self, benchmark):
        """Benchmark neuroplasticity simulation"""
        network = NeuroplasticNetwork(n_neurons=100, connectivity_probability=0.15)
        
        # Add plasticity rules
        hebbian_rule = PlasticityRule(PlasticityType.HEBBIAN, learning_rate=0.005)
        stdp_rule = PlasticityRule(PlasticityType.STDP, learning_rate=0.5)
        network.add_plasticity_rule(hebbian_rule)
        network.add_plasticity_rule(stdp_rule)
        
        def simulate_batch():
            spike_counts = []
            
            for step in range(50):  # 50 timesteps
                # Create structured input
                external_input = np.zeros(100)
                if step % 10 < 3:  # Periodic stimulation
                    external_input[:20] = np.random.normal(8, 2, 20)
                external_input += np.random.normal(0, 1, 100)
                
                # Simulate timestep
                spikes = network.simulate_timestep(dt=1.0, external_input=external_input)
                spike_counts.append(np.sum(spikes))
            
            return spike_counts
        
        result = benchmark(simulate_batch)
        assert len(result) == 50
    
    def test_integrated_research_pipeline_benchmark(self, benchmark):
        """Benchmark integrated research pipeline"""
        def research_pipeline():
            # Step 1: Quantum processing
            processor = QuantumSymbolicProcessor(64)
            concepts = [Concept(f"concept_{i}", np.random.random(32)) for i in range(5)]
            superposition_id = processor.create_concept_superposition(concepts)
            quantum_metrics = processor.compute_quantum_information(superposition_id)
            
            # Step 2: Consciousness assessment
            network = create_test_network()
            assessor = ConsciousnessAssessment(network)
            consciousness_state = np.random.random(8)
            consciousness_assessment = assessor.assess_consciousness_level(consciousness_state)
            
            # Step 3: Neuroplasticity simulation
            neural_net = NeuroplasticNetwork(n_neurons=30, connectivity_probability=0.2)
            rule = PlasticityRule(PlasticityType.HEBBIAN, learning_rate=0.01)
            neural_net.add_plasticity_rule(rule)
            
            # Run short simulation
            for _ in range(10):
                input_signal = np.random.normal(2, 1, 30)
                neural_net.simulate_timestep(1.0, input_signal)
            
            stats = neural_net.get_network_statistics()
            
            return {
                'quantum_entropy': quantum_metrics.get('von_neumann_entropy', 0),
                'consciousness_phi': consciousness_assessment['phi'],
                'neural_connectivity': stats['connectivity']
            }
        
        result = benchmark(research_pipeline)
        
        # Verify pipeline completion
        assert 'quantum_entropy' in result
        assert 'consciousness_phi' in result
        assert 'neural_connectivity' in result


@pytest.mark.benchmark
class TestMemoryBenchmarks(BenchmarkSuite):
    """Memory usage benchmarks"""
    
    def test_security_manager_memory_usage(self):
        """Test SecurityManager memory footprint"""
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Create multiple security managers
        managers = []
        for i in range(10):
            manager = SecurityManager()
            managers.append(manager)
        
        peak_memory = self.process.memory_info().rss / 1024 / 1024
        memory_per_manager = (peak_memory - initial_memory) / 10
        
        print(f"Memory per SecurityManager: {memory_per_manager:.2f} MB")
        
        # Cleanup
        del managers
        
        assert memory_per_manager < 50  # Should use less than 50MB per manager
    
    @pytest.mark.skipif(not SCIENTIFIC_AVAILABLE, reason="Scientific modules not available")
    def test_quantum_processor_memory_scaling(self):
        """Test QuantumSymbolicProcessor memory scaling"""
        memory_usage = []
        hilbert_dims = [32, 64, 128, 256]
        
        for dim in hilbert_dims:
            initial_memory = self.process.memory_info().rss / 1024 / 1024
            
            processor = QuantumSymbolicProcessor(dim)
            
            # Create concepts and superpositions
            concepts = [Concept(f"concept_{i}", np.random.random(dim//2)) for i in range(5)]
            superposition_id = processor.create_concept_superposition(concepts)
            
            peak_memory = self.process.memory_info().rss / 1024 / 1024
            memory_delta = peak_memory - initial_memory
            memory_usage.append((dim, memory_delta))
            
            del processor  # Cleanup
        
        # Verify memory scaling is reasonable
        for dim, memory in memory_usage:
            print(f"Hilbert dimension {dim}: {memory:.2f} MB")
            assert memory < 1000  # Should use less than 1GB


class BenchmarkRunner:
    """Main benchmark runner"""
    
    def __init__(self):
        self.suite = BenchmarkSuite()
    
    def run_all_benchmarks(self):
        """Run all benchmark suites"""
        print("🚀 Starting Obvivlorum Performance Benchmarks")
        print("=" * 60)
        
        # System information
        print(f"Python version: {os.sys.version}")
        print(f"CPU count: {psutil.cpu_count()}")
        print(f"Available memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
        print(f"Process PID: {os.getpid()}")
        print()
        
        # Run pytest benchmarks
        pytest_args = [
            __file__,
            "-v",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-columns=min,max,mean,stddev,rounds,iterations",
            "--benchmark-group-by=group",
        ]
        
        pytest.main(pytest_args)


if __name__ == "__main__":
    runner = BenchmarkRunner()
    runner.run_all_benchmarks()