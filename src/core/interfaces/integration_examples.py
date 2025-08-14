"""
Interface Integration Examples
Shows how different interfaces work together in the system
"""

from typing import Dict, Any, List
from .detector_interfaces import IDetector, IDetectionResult
from .service_interfaces import IAnalysisService, IDetectionService
from .data_interfaces import IDataCollector, IDataProcessor, DataSample
from .api_interfaces import IAPIHandler, APIRequest, APIResponse
from .extension_interfaces import IContentScript, IBackgroundScript
from .pipeline_interfaces import IPipelineStage, IPipelineOrchestrator


class DetectionPipelineExample:
    """
    Example showing how detection components integrate through interfaces
    """
    
    def __init__(self, 
                 detector: IDetector,
                 analysis_service: IAnalysisService,
                 data_processor: IDataProcessor,
                 pipeline: IPipelineOrchestrator):
        self.detector = detector
        self.analysis_service = analysis_service
        self.data_processor = data_processor
        self.pipeline = pipeline
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Example of text processing through multiple interfaces
        
        Flow: Raw text -> Data Processing -> Detection -> Analysis -> Result
        """
        # 1. Create data sample
        sample = DataSample(
            id="sample_1",
            content=text,
            source=None,
            timestamp=None
        )
        
        # 2. Process through data processor
        processed_sample = await self.data_processor.process(sample)
        
        # 3. Detect using detector
        detection_result = await self.detector.detect(processed_sample.content)
        
        # 4. Analyze using service
        analysis_result = await self.analysis_service.analyze_text(
            processed_sample.content
        )
        
        # 5. Combine results
        return {
            "original_text": text,
            "processed_text": processed_sample.content,
            "detection": detection_result.to_dict(),
            "analysis": analysis_result.to_dict(),
            "metadata": {
                "processing_pipeline": "complete",
                "components_used": ["data_processor", "detector", "analysis_service"]
            }
        }


class APIToServiceIntegrationExample:
    """
    Example showing API layer to service layer integration
    """
    
    def __init__(self,
                 api_handler: IAPIHandler,
                 detection_service: IDetectionService,
                 analysis_service: IAnalysisService):
        self.api_handler = api_handler
        self.detection_service = detection_service
        self.analysis_service = analysis_service
    
    async def handle_detection_request(self, request: APIRequest) -> APIResponse:
        """
        Example of API request flowing through service layer
        
        Flow: API Request -> Validation -> Service Call -> Response Formatting
        """
        try:
            # 1. Extract text from request
            text = request.body.get("text", "")
            detector_type = request.body.get("detector_type", "ensemble")
            
            # 2. Call detection service (which uses detector interfaces)
            detection_result = await self.detection_service.detect(
                text, detector_type
            )
            
            # 3. Call analysis service for additional insights
            analysis_result = await self.analysis_service.analyze_text(text)
            
            # 4. Combine and format response
            response_data = {
                "success": True,
                "data": {
                    "detection": detection_result.to_dict(),
                    "analysis": analysis_result.to_dict(),
                    "request_id": request.headers.get("X-Request-ID"),
                    "processing_time": 0.0  # Would be calculated
                }
            }
            
            return APIResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body=response_data
            )
            
        except Exception as e:
            # Error handling through interfaces
            error_response = {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
            
            return APIResponse(
                status_code=500,
                headers={"Content-Type": "application/json"},
                body=error_response
            )


class ExtensionToAPIIntegrationExample:
    """
    Example showing Chrome extension to API integration
    """
    
    def __init__(self,
                 content_script: IContentScript,
                 background_script: IBackgroundScript,
                 api_client: 'IAPIClient'):
        self.content_script = content_script
        self.background_script = background_script
        self.api_client = api_client
    
    async def analyze_page_content(self) -> Dict[str, Any]:
        """
        Example of extension analyzing page content through API
        
        Flow: Content Script -> Background Script -> API Client -> Backend API
        """
        # 1. Content script scans page
        page_elements = await self.content_script.scan_page([
            'p', 'div[role="article"]', '.tweet-text'
        ])
        
        results = []
        
        # 2. For each element, request analysis
        for element in page_elements[:5]:  # Limit to 5 for demo
            try:
                # 3. Background script handles API communication
                api_response = await self.api_client.post(
                    "/api/v1/detect",
                    data={
                        "text": element["text"],
                        "options": {
                            "quick": True,
                            "include_explanation": False
                        }
                    }
                )
                
                if api_response.status_code == 200:
                    # 4. Update UI with results
                    self.content_script.add_ui_indicator(
                        element["elementId"], 
                        api_response.body["data"]
                    )
                    
                    results.append({
                        "element_id": element["elementId"],
                        "result": api_response.body["data"],
                        "success": True
                    })
                else:
                    results.append({
                        "element_id": element["elementId"],
                        "error": "API request failed",
                        "success": False
                    })
                    
            except Exception as e:
                results.append({
                    "element_id": element["elementId"],
                    "error": str(e),
                    "success": False
                })
        
        return {
            "page_url": self.content_script.get_page_info()["url"],
            "elements_found": len(page_elements),
            "elements_analyzed": len(results),
            "results": results
        }


class DataPipelineIntegrationExample:
    """
    Example showing data collection and processing pipeline integration
    """
    
    def __init__(self,
                 data_collector: IDataCollector,
                 data_processor: IDataProcessor,
                 pipeline: IPipelineOrchestrator,
                 detector: IDetector):
        self.data_collector = data_collector
        self.data_processor = data_processor
        self.pipeline = pipeline
        self.detector = detector
    
    async def process_collected_data(self, count: int = 100) -> Dict[str, Any]:
        """
        Example of end-to-end data processing pipeline
        
        Flow: Data Collection -> Processing -> Detection -> Results
        """
        # 1. Collect data samples
        samples = await self.data_collector.collect(
            count=count,
            filters={"source": "twitter", "min_length": 50}
        )
        
        # 2. Process through pipeline (using pipeline orchestrator)
        pipeline_results = []
        
        for sample in samples:
            try:
                # Process sample
                processed_sample = await self.data_processor.process(sample)
                
                # Run detection
                detection_result = await self.detector.detect(
                    processed_sample.content
                )
                
                pipeline_results.append({
                    "sample_id": sample.id,
                    "original_content": sample.content,
                    "processed_content": processed_sample.content,
                    "detection": detection_result.to_dict(),
                    "processing_successful": True
                })
                
            except Exception as e:
                pipeline_results.append({
                    "sample_id": sample.id,
                    "error": str(e),
                    "processing_successful": False
                })
        
        # 3. Generate summary statistics
        successful_count = sum(1 for r in pipeline_results if r.get("processing_successful", False))
        ai_detected_count = sum(1 for r in pipeline_results 
                               if r.get("processing_successful", False) and 
                               r.get("detection", {}).get("prediction") == "ai")
        
        return {
            "total_samples": len(samples),
            "processed_successfully": successful_count,
            "processing_success_rate": successful_count / len(samples) if samples else 0,
            "ai_detected": ai_detected_count,
            "ai_detection_rate": ai_detected_count / successful_count if successful_count else 0,
            "results": pipeline_results[:10],  # Include first 10 for inspection
            "collection_stats": self.data_collector.get_collection_stats(),
            "processing_stats": self.data_processor.get_processing_stats()
        }


class ServiceOrchestrationExample:
    """
    Example showing how services work together through interfaces
    """
    
    def __init__(self,
                 analysis_service: IAnalysisService,
                 detection_service: IDetectionService,
                 cache_service: 'ICacheService',
                 config_service: 'IConfigurationService'):
        self.analysis_service = analysis_service
        self.detection_service = detection_service
        self.cache_service = cache_service
        self.config_service = config_service
    
    async def intelligent_analysis(self, text: str) -> Dict[str, Any]:
        """
        Example of intelligent analysis using multiple services
        
        Flow: Cache Check -> Configuration -> Detection -> Analysis -> Caching
        """
        # 1. Generate cache key
        cache_key = f"analysis:{hash(text)}"
        
        # 2. Check cache first
        cached_result = await self.cache_service.get(cache_key)
        if cached_result:
            return {
                "result": cached_result,
                "source": "cache",
                "cache_hit": True
            }
        
        # 3. Get configuration
        detection_config = await self.config_service.get_config(
            "detection.default_config", 
            {"threshold": 0.7, "quick_mode": False}
        )
        
        # 4. Run detection with configuration
        detection_result = await self.detection_service.detect(text)
        
        # 5. Run additional analysis if confidence is low
        analysis_result = None
        if detection_result.get_score().confidence < 0.8:
            analysis_result = await self.analysis_service.analyze_text(text)
        
        # 6. Combine results
        combined_result = {
            "detection": detection_result.to_dict(),
            "additional_analysis": analysis_result.to_dict() if analysis_result else None,
            "config_used": detection_config,
            "confidence_boost_applied": analysis_result is not None
        }
        
        # 7. Cache result
        await self.cache_service.set(
            cache_key, 
            combined_result, 
            ttl=3600  # 1 hour
        )
        
        return {
            "result": combined_result,
            "source": "computed",
            "cache_hit": False
        }


# Interface Composition Examples
class CompositeDetectorExample:
    """
    Example showing how multiple detector interfaces can be composed
    """
    
    def __init__(self, 
                 pattern_detector: 'IPatternDetector',
                 llm_detector: 'ILLMDetector',
                 ml_detector: 'IMLDetector',
                 ensemble_detector: 'IEnsembleDetector'):
        self.pattern_detector = pattern_detector
        self.llm_detector = llm_detector
        self.ml_detector = ml_detector
        self.ensemble_detector = ensemble_detector
        
        # Configure ensemble with individual detectors
        self.ensemble_detector.add_detector(pattern_detector, weight=0.3)
        self.ensemble_detector.add_detector(llm_detector, weight=0.4)
        self.ensemble_detector.add_detector(ml_detector, weight=0.3)
    
    async def comprehensive_detection(self, text: str) -> Dict[str, Any]:
        """
        Run comprehensive detection using multiple detector types
        """
        # Individual detector results
        individual_results = {}
        
        # Pattern-based detection
        pattern_result = await self.pattern_detector.detect(text)
        individual_results["pattern"] = pattern_result.to_dict()
        
        # LLM-based detection
        llm_result = await self.llm_detector.detect(text)
        individual_results["llm"] = llm_result.to_dict()
        
        # ML-based detection
        ml_result = await self.ml_detector.detect(text)
        individual_results["ml"] = ml_result.to_dict()
        
        # Ensemble result
        ensemble_result = await self.ensemble_detector.detect(text)
        
        return {
            "ensemble_result": ensemble_result.to_dict(),
            "individual_results": individual_results,
            "confidence_comparison": {
                "pattern": pattern_result.get_score().confidence,
                "llm": llm_result.get_score().confidence,
                "ml": ml_result.get_score().confidence,
                "ensemble": ensemble_result.get_score().confidence
            },
            "recommendation": "Use ensemble result for final decision"
        }