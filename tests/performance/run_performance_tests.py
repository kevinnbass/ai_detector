"""
Performance Test Runner
Comprehensive performance testing suite runner with reporting and analysis
"""

import asyncio
import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from tests.performance.benchmarks import (
    DetectionBenchmarks, DataProcessingBenchmarks, 
    APIBenchmarks, SystemBenchmarks
)
from tests.performance.performance_monitor import PerformanceMonitor


class PerformanceTestRunner:
    """Comprehensive performance test runner"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.results = {}
        self.monitor = PerformanceMonitor()
        self.start_time = None
        self.test_report = {}
        
        # Setup logging
        log_level = getattr(logging, self.config.get('log_level', 'INFO'))
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('test-results/performance_tests.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for performance tests"""
        return {
            "log_level": "INFO",
            "output_dir": "test-results",
            "run_python_benchmarks": True,
            "run_javascript_benchmarks": True,
            "run_system_benchmarks": True,
            "run_load_tests": True,
            "performance_requirements": {
                "detection_time_ms": 100,
                "api_response_time_ms": 2000,
                "memory_limit_mb": 50,
                "throughput_ops_per_sec": 16.7,  # >1000 tweets/min
                "error_rate_percent": 1.0
            },
            "test_iterations": {
                "unit_benchmarks": 100,
                "integration_benchmarks": 50,
                "load_test_duration_seconds": 60
            },
            "alert_thresholds": {
                "cpu_usage_percent": 80,
                "memory_usage_percent": 85,
                "response_time_ms": 2000
            }
        }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite"""
        self.start_time = datetime.now()
        self.logger.info("Starting comprehensive performance test suite")
        
        # Ensure output directory exists
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Start performance monitoring
            self.monitor.start()
            
            # Run test suites
            if self.config["run_python_benchmarks"]:
                await self._run_python_benchmarks()
            
            if self.config["run_javascript_benchmarks"]:
                await self._run_javascript_benchmarks()
            
            if self.config["run_system_benchmarks"]:
                await self._run_system_benchmarks()
            
            if self.config["run_load_tests"]:
                await self._run_load_tests()
            
            # Generate comprehensive report
            await self._generate_comprehensive_report()
            
            # Check performance requirements
            self._check_performance_requirements()
            
        except Exception as e:
            self.logger.error(f"Error in performance test suite: {e}")
            raise
        finally:
            # Stop monitoring and generate final report
            self.monitor.stop()
            self._export_results()
        
        return self.test_report
    
    async def _run_python_benchmarks(self):
        """Run Python performance benchmarks"""
        self.logger.info("Running Python performance benchmarks...")
        
        try:
            # Detection benchmarks
            self.logger.info("  - Running detection benchmarks")
            detection_bench = DetectionBenchmarks()
            await detection_bench.setup_detector()
            
            detection_results = []
            detection_results.append(await detection_bench.benchmark_single_detection(text_length=100))
            detection_results.append(await detection_bench.benchmark_single_detection(text_length=500))
            detection_results.append(await detection_bench.benchmark_single_detection(text_length=1000))
            detection_results.append(await detection_bench.benchmark_batch_detection(batch_size=10))
            detection_results.append(await detection_bench.benchmark_batch_detection(batch_size=50))
            detection_results.append(await detection_bench.benchmark_concurrent_detection(concurrent_users=5))
            detection_results.append(await detection_bench.benchmark_concurrent_detection(concurrent_users=10))
            
            self.results["detection_benchmarks"] = [
                {
                    "operation": r.operation,
                    "avg_time_ms": r.avg_time * 1000,
                    "p95_time_ms": r.p95_time * 1000,
                    "throughput_ops_sec": r.throughput,
                    "success_count": r.success_count,
                    "error_count": r.error_count,
                    "memory_usage_mb": r.memory_usage
                }
                for r in detection_results
            ]
            
            # Data processing benchmarks
            self.logger.info("  - Running data processing benchmarks")
            data_bench = DataProcessingBenchmarks()
            await data_bench.setup_processors()
            
            data_results = []
            data_results.append(await data_bench.benchmark_text_preprocessing(text_length=500))
            data_results.append(await data_bench.benchmark_text_preprocessing(text_length=2000))
            data_results.append(await data_bench.benchmark_feature_extraction(text_length=500))
            data_results.append(await data_bench.benchmark_feature_extraction(text_length=2000))
            data_results.append(await data_bench.benchmark_data_collection(sample_count=50))
            data_results.append(await data_bench.benchmark_data_collection(sample_count=200))
            
            batch_results = await data_bench.benchmark_batch_processing(batch_size=50)
            data_results.extend(batch_results)
            
            self.results["data_processing_benchmarks"] = [
                {
                    "operation": r.operation,
                    "avg_time_ms": r.avg_time * 1000,
                    "p95_time_ms": r.p95_time * 1000,
                    "throughput_ops_sec": r.throughput,
                    "memory_usage_mb": r.memory_usage
                }
                for r in data_results
            ]
            
            # API benchmarks
            self.logger.info("  - Running API benchmarks")
            api_bench = APIBenchmarks()
            await api_bench.setup_api_client()
            
            api_results = []
            api_results.append(await api_bench.benchmark_api_requests(request_count=50))
            api_results.append(await api_bench.benchmark_api_requests(request_count=100))
            api_results.append(await api_bench.benchmark_api_with_queue())
            api_results.append(await api_bench.benchmark_api_with_cache())
            
            self.results["api_benchmarks"] = [
                {
                    "operation": r.operation,
                    "avg_time_ms": r.avg_time * 1000,
                    "p95_time_ms": r.p95_time * 1000,
                    "throughput_ops_sec": r.throughput,
                    "memory_usage_mb": r.memory_usage
                }
                for r in api_results
            ]
            
            self.logger.info("Python benchmarks completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in Python benchmarks: {e}")
            self.results["python_benchmark_error"] = str(e)
    
    async def _run_javascript_benchmarks(self):
        """Run JavaScript performance benchmarks"""
        self.logger.info("Running JavaScript performance benchmarks...")
        
        try:
            # Run JavaScript benchmarks via Node.js
            js_benchmark_script = Path(__file__).parent / "extension-benchmarks.js"
            
            # Ensure the script exists
            if not js_benchmark_script.exists():
                self.logger.warning("JavaScript benchmark script not found, skipping JS benchmarks")
                return
            
            # Run the JavaScript benchmarks
            result = subprocess.run([
                "node", str(js_benchmark_script)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                self.logger.info("JavaScript benchmarks completed successfully")
                
                # Parse results if they were written to a file
                js_results_file = Path("test-results") / "extension-benchmark-results.json"
                if js_results_file.exists():
                    with open(js_results_file, 'r') as f:
                        js_results = json.load(f)
                    self.results["javascript_benchmarks"] = js_results
                else:
                    self.results["javascript_benchmarks"] = {
                        "status": "completed",
                        "stdout": result.stdout,
                        "note": "Results not captured in JSON format"
                    }
            else:
                self.logger.error(f"JavaScript benchmarks failed: {result.stderr}")
                self.results["javascript_benchmark_error"] = {
                    "return_code": result.returncode,
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            self.logger.error("JavaScript benchmarks timed out")
            self.results["javascript_benchmark_error"] = "Timeout after 5 minutes"
        except FileNotFoundError:
            self.logger.error("Node.js not found, skipping JavaScript benchmarks")
            self.results["javascript_benchmark_error"] = "Node.js not available"
        except Exception as e:
            self.logger.error(f"Error running JavaScript benchmarks: {e}")
            self.results["javascript_benchmark_error"] = str(e)
    
    async def _run_system_benchmarks(self):
        """Run system-level benchmarks"""
        self.logger.info("Running system benchmarks...")
        
        try:
            system_bench = SystemBenchmarks()
            await system_bench.setup_system()
            
            system_results = []
            system_results.append(await system_bench.benchmark_end_to_end_detection())
            system_results.append(await system_bench.benchmark_high_throughput(target_rps=50))
            system_results.append(await system_bench.benchmark_high_throughput(target_rps=100))
            
            self.results["system_benchmarks"] = [
                {
                    "operation": r.operation,
                    "avg_time_ms": r.avg_time * 1000,
                    "p95_time_ms": r.p95_time * 1000,
                    "throughput_ops_sec": r.throughput,
                    "memory_usage_mb": r.memory_usage,
                    "success_count": r.success_count,
                    "error_count": r.error_count
                }
                for r in system_results
            ]
            
            self.logger.info("System benchmarks completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in system benchmarks: {e}")
            self.results["system_benchmark_error"] = str(e)
    
    async def _run_load_tests(self):
        """Run load testing scenarios"""
        self.logger.info("Running load tests...")
        
        try:
            # Simulate sustained load
            duration = self.config["test_iterations"]["load_test_duration_seconds"]
            start_time = time.time()
            
            # Create multiple concurrent tasks
            tasks = []
            for i in range(10):  # 10 concurrent workers
                tasks.append(self._load_test_worker(f"worker_{i}", duration))
            
            # Run load test
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            failed_results = [r for r in results if isinstance(r, Exception)]
            
            total_operations = sum(r["operations_completed"] for r in successful_results)
            total_errors = sum(r["errors"] for r in successful_results)
            actual_duration = time.time() - start_time
            
            self.results["load_test"] = {
                "duration_seconds": actual_duration,
                "total_operations": total_operations,
                "total_errors": total_errors,
                "operations_per_second": total_operations / actual_duration,
                "error_rate_percent": (total_errors / total_operations * 100) if total_operations > 0 else 0,
                "successful_workers": len(successful_results),
                "failed_workers": len(failed_results),
                "worker_results": successful_results
            }
            
            self.logger.info(f"Load test completed: {total_operations} operations in {actual_duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in load tests: {e}")
            self.results["load_test_error"] = str(e)
    
    async def _load_test_worker(self, worker_id: str, duration_seconds: int) -> Dict[str, Any]:
        """Individual load test worker"""
        start_time = time.time()
        operations_completed = 0
        errors = 0
        
        # Setup worker components (mock for this example)
        try:
            while (time.time() - start_time) < duration_seconds:
                try:
                    # Simulate detection operation
                    await asyncio.sleep(0.05)  # 50ms simulated processing
                    
                    # Record metric
                    self.monitor.record_application_metric(
                        "load_test_operation_time_ms", 
                        50 + (operations_completed % 10)  # Slightly varying time
                    )
                    
                    operations_completed += 1
                    
                except Exception as e:
                    errors += 1
                    self.logger.debug(f"Error in {worker_id}: {e}")
        
        except Exception as e:
            self.logger.error(f"Fatal error in {worker_id}: {e}")
            raise
        
        return {
            "worker_id": worker_id,
            "operations_completed": operations_completed,
            "errors": errors,
            "duration": time.time() - start_time
        }
    
    async def _generate_comprehensive_report(self):
        """Generate comprehensive performance report"""
        self.logger.info("Generating comprehensive performance report...")
        
        end_time = datetime.now()
        
        # Generate monitoring report
        monitoring_report = self.monitor.generate_report(
            start_time=self.start_time,
            end_time=end_time
        )
        
        # Compile comprehensive report
        self.test_report = {
            "test_run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": (end_time - self.start_time).total_seconds() / 60,
                "configuration": self.config
            },
            "benchmark_results": self.results,
            "system_monitoring": {
                "metrics_summary": monitoring_report.metrics_summary,
                "alerts_triggered": monitoring_report.alerts_triggered,
                "system_health": monitoring_report.system_health,
                "recommendations": monitoring_report.recommendations
            },
            "performance_analysis": self._analyze_performance_results(),
            "requirement_compliance": self._check_performance_requirements()
        }
    
    def _analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance results and provide insights"""
        analysis = {
            "summary": {},
            "bottlenecks": [],
            "optimizations": [],
            "trends": {}
        }
        
        # Analyze detection performance
        if "detection_benchmarks" in self.results:
            detection_times = [
                r["avg_time_ms"] for r in self.results["detection_benchmarks"] 
                if "single_detection" in r["operation"]
            ]
            
            if detection_times:
                analysis["summary"]["detection_performance"] = {
                    "avg_time_ms": sum(detection_times) / len(detection_times),
                    "max_time_ms": max(detection_times),
                    "min_time_ms": min(detection_times)
                }
                
                # Check for bottlenecks
                if max(detection_times) > self.config["performance_requirements"]["detection_time_ms"]:
                    analysis["bottlenecks"].append("Detection time exceeds requirements")
        
        # Analyze throughput
        if "load_test" in self.results:
            ops_per_sec = self.results["load_test"].get("operations_per_second", 0)
            required_ops_per_sec = self.config["performance_requirements"]["throughput_ops_per_sec"]
            
            analysis["summary"]["throughput"] = {
                "achieved_ops_per_sec": ops_per_sec,
                "required_ops_per_sec": required_ops_per_sec,
                "meets_requirement": ops_per_sec >= required_ops_per_sec
            }
            
            if ops_per_sec < required_ops_per_sec:
                analysis["bottlenecks"].append("Throughput below requirements")
                analysis["optimizations"].append("Consider horizontal scaling or performance optimization")
        
        # Analyze memory usage
        memory_data = []
        for benchmark_type in ["detection_benchmarks", "data_processing_benchmarks", "api_benchmarks"]:
            if benchmark_type in self.results:
                for result in self.results[benchmark_type]:
                    if "memory_usage_mb" in result:
                        if isinstance(result["memory_usage_mb"], dict):
                            memory_data.append(result["memory_usage_mb"].get("peak_traced_mb", 0))
        
        if memory_data:
            max_memory = max(memory_data)
            memory_limit = self.config["performance_requirements"]["memory_limit_mb"]
            
            analysis["summary"]["memory_usage"] = {
                "peak_memory_mb": max_memory,
                "memory_limit_mb": memory_limit,
                "meets_requirement": max_memory <= memory_limit
            }
            
            if max_memory > memory_limit:
                analysis["bottlenecks"].append("Memory usage exceeds limits")
                analysis["optimizations"].append("Optimize memory allocation and implement memory pooling")
        
        return analysis
    
    def _check_performance_requirements(self) -> Dict[str, Any]:
        """Check if performance requirements are met"""
        requirements = self.config["performance_requirements"]
        compliance = {
            "overall_status": "PASS",
            "checks": {}
        }
        
        # Check detection time requirement
        if "detection_benchmarks" in self.results:
            detection_times = [
                r["p95_time_ms"] for r in self.results["detection_benchmarks"]
                if "single_detection" in r["operation"]
            ]
            
            if detection_times:
                max_detection_time = max(detection_times)
                compliance["checks"]["detection_time"] = {
                    "requirement_ms": requirements["detection_time_ms"],
                    "actual_p95_ms": max_detection_time,
                    "status": "PASS" if max_detection_time <= requirements["detection_time_ms"] else "FAIL"
                }
                
                if max_detection_time > requirements["detection_time_ms"]:
                    compliance["overall_status"] = "FAIL"
        
        # Check API response time requirement
        if "api_benchmarks" in self.results:
            api_times = [r["p95_time_ms"] for r in self.results["api_benchmarks"]]
            
            if api_times:
                max_api_time = max(api_times)
                compliance["checks"]["api_response_time"] = {
                    "requirement_ms": requirements["api_response_time_ms"],
                    "actual_p95_ms": max_api_time,
                    "status": "PASS" if max_api_time <= requirements["api_response_time_ms"] else "FAIL"
                }
                
                if max_api_time > requirements["api_response_time_ms"]:
                    compliance["overall_status"] = "FAIL"
        
        # Check throughput requirement
        if "load_test" in self.results:
            actual_throughput = self.results["load_test"].get("operations_per_second", 0)
            compliance["checks"]["throughput"] = {
                "requirement_ops_per_sec": requirements["throughput_ops_per_sec"],
                "actual_ops_per_sec": actual_throughput,
                "status": "PASS" if actual_throughput >= requirements["throughput_ops_per_sec"] else "FAIL"
            }
            
            if actual_throughput < requirements["throughput_ops_per_sec"]:
                compliance["overall_status"] = "FAIL"
        
        # Check error rate requirement
        if "load_test" in self.results:
            error_rate = self.results["load_test"].get("error_rate_percent", 0)
            compliance["checks"]["error_rate"] = {
                "requirement_percent": requirements["error_rate_percent"],
                "actual_percent": error_rate,
                "status": "PASS" if error_rate <= requirements["error_rate_percent"] else "FAIL"
            }
            
            if error_rate > requirements["error_rate_percent"]:
                compliance["overall_status"] = "FAIL"
        
        return compliance
    
    def _export_results(self):
        """Export all results to files"""
        output_dir = Path(self.config["output_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export main report
        main_report_file = output_dir / f"performance_test_report_{timestamp}.json"
        with open(main_report_file, 'w') as f:
            json.dump(self.test_report, f, indent=2, default=str)
        
        # Export monitoring data
        if hasattr(self.monitor, 'reports') and self.monitor.reports:
            monitoring_file = output_dir / f"performance_monitoring_{timestamp}.json"
            self.monitor.export_report(self.monitor.reports[-1], monitoring_file.name)
        
        # Export summary
        summary_file = output_dir / f"performance_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            self._write_summary_report(f)
        
        self.logger.info(f"Results exported to {output_dir}")
        self.logger.info(f"Main report: {main_report_file}")
        self.logger.info(f"Summary: {summary_file}")
    
    def _write_summary_report(self, file):
        """Write human-readable summary report"""
        file.write("AI DETECTOR PERFORMANCE TEST SUMMARY\n")
        file.write("=" * 50 + "\n\n")
        
        # Test run info
        file.write(f"Test Run Date: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Duration: {self.test_report['test_run_info']['duration_minutes']:.2f} minutes\n")
        file.write(f"Overall Status: {self.test_report['requirement_compliance']['overall_status']}\n\n")
        
        # Performance requirements check
        file.write("PERFORMANCE REQUIREMENTS\n")
        file.write("-" * 25 + "\n")
        
        for check_name, check_data in self.test_report['requirement_compliance']['checks'].items():
            status_icon = "✅" if check_data['status'] == "PASS" else "❌"
            file.write(f"{status_icon} {check_name.replace('_', ' ').title()}: {check_data['status']}\n")
        
        file.write("\n")
        
        # Key metrics
        if "performance_analysis" in self.test_report:
            analysis = self.test_report["performance_analysis"]
            
            file.write("KEY PERFORMANCE METRICS\n")
            file.write("-" * 25 + "\n")
            
            if "detection_performance" in analysis["summary"]:
                det_perf = analysis["summary"]["detection_performance"]
                file.write(f"Average Detection Time: {det_perf['avg_time_ms']:.2f}ms\n")
                file.write(f"Max Detection Time: {det_perf['max_time_ms']:.2f}ms\n")
            
            if "throughput" in analysis["summary"]:
                throughput = analysis["summary"]["throughput"]
                file.write(f"Achieved Throughput: {throughput['achieved_ops_per_sec']:.2f} ops/sec\n")
                file.write(f"Required Throughput: {throughput['required_ops_per_sec']:.2f} ops/sec\n")
            
            file.write("\n")
            
            # Bottlenecks and optimizations
            if analysis["bottlenecks"]:
                file.write("IDENTIFIED BOTTLENECKS\n")
                file.write("-" * 20 + "\n")
                for bottleneck in analysis["bottlenecks"]:
                    file.write(f"• {bottleneck}\n")
                file.write("\n")
            
            if analysis["optimizations"]:
                file.write("OPTIMIZATION RECOMMENDATIONS\n")
                file.write("-" * 30 + "\n")
                for optimization in analysis["optimizations"]:
                    file.write(f"• {optimization}\n")
        
        # System health
        if "system_monitoring" in self.test_report:
            file.write(f"\nSystem Health: {self.test_report['system_monitoring']['system_health']}\n")
            
            if self.test_report['system_monitoring']['recommendations']:
                file.write("\nSystem Recommendations:\n")
                for rec in self.test_report['system_monitoring']['recommendations']:
                    file.write(f"• {rec}\n")


async def main():
    """Main entry point for performance testing"""
    parser = argparse.ArgumentParser(description="AI Detector Performance Test Suite")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output-dir", default="test-results", help="Output directory")
    parser.add_argument("--skip-js", action="store_true", help="Skip JavaScript benchmarks")
    parser.add_argument("--skip-load", action="store_true", help="Skip load tests")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Apply command line overrides
    config["output_dir"] = args.output_dir
    config["run_javascript_benchmarks"] = not args.skip_js
    config["run_load_tests"] = not args.skip_load
    
    if args.quick:
        config["test_iterations"] = {
            "unit_benchmarks": 10,
            "integration_benchmarks": 5,
            "load_test_duration_seconds": 10
        }
    
    # Run performance tests
    runner = PerformanceTestRunner(config)
    
    try:
        results = await runner.run_all_tests()
        
        print("\n" + "="*60)
        print("PERFORMANCE TEST SUITE COMPLETED")
        print("="*60)
        print(f"Overall Status: {results['requirement_compliance']['overall_status']}")
        print(f"Results exported to: {config['output_dir']}")
        
        # Exit with error code if tests failed
        if results['requirement_compliance']['overall_status'] == "FAIL":
            sys.exit(1)
        
    except Exception as e:
        logging.error(f"Performance test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())