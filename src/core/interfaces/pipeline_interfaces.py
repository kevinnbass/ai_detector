"""
Pipeline Interface Definitions
Interfaces for processing pipelines and workflow management
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from .base_interfaces import IInitializable, IConfigurable, IMonitorable, IMetricsProvider


class PipelineStatus(Enum):
    """Pipeline status enumeration"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageType(Enum):
    """Pipeline stage type enumeration"""
    INPUT = "input"
    PROCESSING = "processing"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    OUTPUT = "output"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"


class ExecutionMode(Enum):
    """Execution mode enumeration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class PipelineResult:
    """Pipeline execution result"""
    success: bool
    output: Any
    execution_time: float
    stage_results: Dict[str, Any]
    errors: List[str]
    metadata: Dict[str, Any]


@dataclass
class StageResult:
    """Individual stage result"""
    stage_name: str
    success: bool
    output: Any
    execution_time: float
    input_data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IPipelineStage(IInitializable, IConfigurable, IMonitorable, ABC):
    """Interface for pipeline stages"""
    
    @abstractmethod
    async def execute(self, input_data: Any, context: Dict[str, Any]) -> StageResult:
        """Execute pipeline stage"""
        pass
    
    @abstractmethod
    def get_stage_name(self) -> str:
        """Get stage name"""
        pass
    
    @abstractmethod
    def get_stage_type(self) -> StageType:
        """Get stage type"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get stage dependencies"""
        pass
    
    @abstractmethod
    def can_process(self, input_data: Any) -> bool:
        """Check if stage can process input"""
        pass
    
    @abstractmethod
    def estimate_execution_time(self, input_data: Any) -> float:
        """Estimate execution time in seconds"""
        pass
    
    @abstractmethod
    def get_required_resources(self) -> Dict[str, Any]:
        """Get required computational resources"""
        pass


class IConditionalStage(IPipelineStage):
    """Interface for conditional pipeline stages"""
    
    @abstractmethod
    def evaluate_condition(self, input_data: Any, context: Dict[str, Any]) -> bool:
        """Evaluate stage condition"""
        pass
    
    @abstractmethod
    def get_condition_expression(self) -> str:
        """Get condition expression"""
        pass
    
    @abstractmethod
    def set_condition(self, condition: Union[str, Callable]) -> None:
        """Set stage condition"""
        pass


class IParallelStage(IPipelineStage):
    """Interface for parallel processing stages"""
    
    @abstractmethod
    async def execute_parallel(self, input_batches: List[Any], 
                              context: Dict[str, Any]) -> List[StageResult]:
        """Execute stage in parallel"""
        pass
    
    @abstractmethod
    def get_parallel_degree(self) -> int:
        """Get degree of parallelism"""
        pass
    
    @abstractmethod
    def set_parallel_degree(self, degree: int) -> None:
        """Set degree of parallelism"""
        pass
    
    @abstractmethod
    def split_input(self, input_data: Any, batch_size: int) -> List[Any]:
        """Split input for parallel processing"""
        pass
    
    @abstractmethod
    def merge_results(self, results: List[StageResult]) -> StageResult:
        """Merge parallel results"""
        pass


class IPipelineOrchestrator(IInitializable, IConfigurable, IMonitorable, IMetricsProvider, ABC):
    """Interface for pipeline orchestration"""
    
    @abstractmethod
    async def execute_pipeline(self, input_data: Any, 
                              pipeline_config: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Execute complete pipeline"""
        pass
    
    @abstractmethod
    def add_stage(self, stage: IPipelineStage, position: Optional[int] = None) -> None:
        """Add stage to pipeline"""
        pass
    
    @abstractmethod
    def remove_stage(self, stage_name: str) -> bool:
        """Remove stage from pipeline"""
        pass
    
    @abstractmethod
    def get_stages(self) -> List[IPipelineStage]:
        """Get all pipeline stages"""
        pass
    
    @abstractmethod
    def get_pipeline_graph(self) -> Dict[str, Any]:
        """Get pipeline dependency graph"""
        pass
    
    @abstractmethod
    async def pause_pipeline(self) -> bool:
        """Pause pipeline execution"""
        pass
    
    @abstractmethod
    async def resume_pipeline(self) -> bool:
        """Resume pipeline execution"""
        pass
    
    @abstractmethod
    async def cancel_pipeline(self) -> bool:
        """Cancel pipeline execution"""
        pass
    
    @abstractmethod
    def get_execution_status(self) -> PipelineStatus:
        """Get pipeline execution status"""
        pass


class IPipelineBuilder(ABC):
    """Interface for pipeline building"""
    
    @abstractmethod
    def create_pipeline(self, name: str) -> 'IPipelineBuilder':
        """Create new pipeline"""
        pass
    
    @abstractmethod
    def add_stage(self, stage: IPipelineStage) -> 'IPipelineBuilder':
        """Add stage to pipeline"""
        pass
    
    @abstractmethod
    def add_conditional_stage(self, stage: IConditionalStage, 
                             condition: Union[str, Callable]) -> 'IPipelineBuilder':
        """Add conditional stage"""
        pass
    
    @abstractmethod
    def add_parallel_stage(self, stage: IParallelStage, 
                          degree: int = 4) -> 'IPipelineBuilder':
        """Add parallel stage"""
        pass
    
    @abstractmethod
    def set_execution_mode(self, mode: ExecutionMode) -> 'IPipelineBuilder':
        """Set pipeline execution mode"""
        pass
    
    @abstractmethod
    def set_error_handling(self, strategy: str) -> 'IPipelineBuilder':
        """Set error handling strategy"""
        pass
    
    @abstractmethod
    def build(self) -> IPipelineOrchestrator:
        """Build pipeline orchestrator"""
        pass


class IPipelineMonitor(IMonitorable, IMetricsProvider, ABC):
    """Interface for pipeline monitoring"""
    
    @abstractmethod
    def monitor_execution(self, pipeline_id: str, 
                         callback: Optional[Callable] = None) -> str:
        """Monitor pipeline execution"""
        pass
    
    @abstractmethod
    def get_execution_metrics(self, pipeline_id: str) -> Dict[str, Any]:
        """Get execution metrics"""
        pass
    
    @abstractmethod
    def get_stage_metrics(self, pipeline_id: str, stage_name: str) -> Dict[str, Any]:
        """Get stage-specific metrics"""
        pass
    
    @abstractmethod
    def get_resource_usage(self, pipeline_id: str) -> Dict[str, Any]:
        """Get resource usage metrics"""
        pass
    
    @abstractmethod
    def set_alert_thresholds(self, thresholds: Dict[str, float]) -> None:
        """Set monitoring alert thresholds"""
        pass
    
    @abstractmethod
    def get_alerts(self, pipeline_id: str) -> List[Dict[str, Any]]:
        """Get pipeline alerts"""
        pass


class IPipelineConfig(ABC):
    """Interface for pipeline configuration"""
    
    @abstractmethod
    def get_pipeline_name(self) -> str:
        """Get pipeline name"""
        pass
    
    @abstractmethod
    def get_execution_mode(self) -> ExecutionMode:
        """Get execution mode"""
        pass
    
    @abstractmethod
    def get_stage_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get stage configurations"""
        pass
    
    @abstractmethod
    def get_error_handling_strategy(self) -> str:
        """Get error handling strategy"""
        pass
    
    @abstractmethod
    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration"""
        pass
    
    @abstractmethod
    def get_timeout_config(self) -> Dict[str, int]:
        """Get timeout configuration"""
        pass
    
    @abstractmethod
    def validate_config(self) -> tuple[bool, List[str]]:
        """Validate pipeline configuration"""
        pass


class IPipelineRegistry(ABC):
    """Interface for pipeline registry"""
    
    @abstractmethod
    def register_pipeline(self, name: str, orchestrator: IPipelineOrchestrator) -> bool:
        """Register pipeline"""
        pass
    
    @abstractmethod
    def get_pipeline(self, name: str) -> Optional[IPipelineOrchestrator]:
        """Get registered pipeline"""
        pass
    
    @abstractmethod
    def list_pipelines(self) -> List[str]:
        """List registered pipelines"""
        pass
    
    @abstractmethod
    def unregister_pipeline(self, name: str) -> bool:
        """Unregister pipeline"""
        pass
    
    @abstractmethod
    def get_pipeline_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get pipeline information"""
        pass


class IPipelineScheduler(ABC):
    """Interface for pipeline scheduling"""
    
    @abstractmethod
    def schedule_pipeline(self, pipeline_name: str, cron_expression: str,
                         input_data: Any = None) -> str:
        """Schedule pipeline execution"""
        pass
    
    @abstractmethod
    def unschedule_pipeline(self, schedule_id: str) -> bool:
        """Unschedule pipeline"""
        pass
    
    @abstractmethod
    def get_scheduled_pipelines(self) -> List[Dict[str, Any]]:
        """Get scheduled pipelines"""
        pass
    
    @abstractmethod
    def trigger_pipeline(self, pipeline_name: str, input_data: Any = None) -> str:
        """Manually trigger pipeline"""
        pass
    
    @abstractmethod
    def get_execution_history(self, pipeline_name: str, 
                             limit: int = 100) -> List[Dict[str, Any]]:
        """Get pipeline execution history"""
        pass


class IWorkflowEngine(IInitializable, IConfigurable, ABC):
    """Interface for workflow engines"""
    
    @abstractmethod
    async def execute_workflow(self, workflow_definition: Dict[str, Any],
                              input_data: Any) -> Dict[str, Any]:
        """Execute workflow from definition"""
        pass
    
    @abstractmethod
    def create_workflow_from_template(self, template_name: str,
                                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create workflow from template"""
        pass
    
    @abstractmethod
    def validate_workflow_definition(self, definition: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate workflow definition"""
        pass
    
    @abstractmethod
    def get_workflow_templates(self) -> List[Dict[str, Any]]:
        """Get available workflow templates"""
        pass
    
    @abstractmethod
    def save_workflow_template(self, name: str, definition: Dict[str, Any]) -> bool:
        """Save workflow template"""
        pass


class IDataPipeline(IPipelineOrchestrator):
    """Interface for data processing pipelines"""
    
    @abstractmethod
    async def process_stream(self, data_stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
        """Process streaming data"""
        pass
    
    @abstractmethod
    def set_batch_size(self, batch_size: int) -> None:
        """Set batch processing size"""
        pass
    
    @abstractmethod
    def get_throughput_metrics(self) -> Dict[str, float]:
        """Get data throughput metrics"""
        pass
    
    @abstractmethod
    def add_data_validator(self, validator: Callable[[Any], bool]) -> None:
        """Add data validation function"""
        pass
    
    @abstractmethod
    def add_data_transformer(self, transformer: Callable[[Any], Any]) -> None:
        """Add data transformation function"""
        pass