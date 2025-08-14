"""
JSON Schema validation utilities for the AI Detector system.

This module provides centralized schema validation functionality for all components,
ensuring data consistency and type safety across the entire application.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from enum import Enum

try:
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
except ImportError:
    raise ImportError(
        "jsonschema package is required for schema validation. "
        "Install with: pip install jsonschema"
    )

logger = logging.getLogger(__name__)


class SchemaType(Enum):
    """Enumeration of available schema types."""
    
    DETECTION_REQUEST = "detection_request"
    DETECTION_RESPONSE = "detection_response"
    EXTENSION_MESSAGE = "extension_message"
    TRAINING_DATA = "training_data"
    API_ERROR = "api_error"
    PERFORMANCE_METRICS = "performance_metrics"
    LLM_ANALYSIS = "llm_analysis"
    CONFIG_SETTINGS = "config_settings"


@dataclass
class ValidationResult:
    """Result of schema validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    schema_version: Optional[str] = None
    
    def __bool__(self) -> bool:
        """Allow boolean evaluation of validation result."""
        return self.is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "schema_version": self.schema_version
        }


class SchemaValidator:
    """
    JSON Schema validator for AI Detector data formats.
    
    Provides validation, type checking, and format verification for all
    data structures used across the system components.
    """
    
    def __init__(self, schema_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the schema validator.
        
        Args:
            schema_dir: Directory containing JSON schema files.
                       Defaults to project root/schemas.
        """
        self.schema_dir = self._resolve_schema_directory(schema_dir)
        self._schema_cache: Dict[SchemaType, Dict[str, Any]] = {}
        self._validator_cache: Dict[SchemaType, Draft7Validator] = {}
        
        logger.info(f"Initialized SchemaValidator with schema directory: {self.schema_dir}")
    
    def _resolve_schema_directory(self, schema_dir: Optional[Union[str, Path]]) -> Path:
        """Resolve the schema directory path."""
        if schema_dir:
            return Path(schema_dir)
        
        # Default to schemas directory relative to project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # Go up to project root
        return project_root / "schemas"
    
    def _load_schema(self, schema_type: SchemaType) -> Dict[str, Any]:
        """
        Load a JSON schema from disk.
        
        Args:
            schema_type: Type of schema to load
            
        Returns:
            Loaded schema dictionary
            
        Raises:
            FileNotFoundError: If schema file doesn't exist
            ValueError: If schema is invalid JSON
        """
        if schema_type in self._schema_cache:
            return self._schema_cache[schema_type]
        
        schema_file = self.schema_dir / f"{schema_type.value}.json"
        
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # Validate the schema itself
            Draft7Validator.check_schema(schema)
            
            # Cache the loaded schema
            self._schema_cache[schema_type] = schema
            logger.debug(f"Loaded and cached schema: {schema_type.value}")
            
            return schema
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schema file {schema_file}: {e}")
        except jsonschema.exceptions.SchemaError as e:
            raise ValueError(f"Invalid schema in file {schema_file}: {e}")
    
    def _get_validator(self, schema_type: SchemaType) -> Draft7Validator:
        """
        Get a cached validator for the specified schema type.
        
        Args:
            schema_type: Type of schema to get validator for
            
        Returns:
            Draft7Validator instance
        """
        if schema_type in self._validator_cache:
            return self._validator_cache[schema_type]
        
        schema = self._load_schema(schema_type)
        validator = Draft7Validator(schema)
        
        # Cache the validator
        self._validator_cache[schema_type] = validator
        
        return validator
    
    def validate(
        self, 
        data: Any, 
        schema_type: SchemaType,
        strict: bool = True
    ) -> ValidationResult:
        """
        Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema_type: Type of schema to validate against
            strict: If True, treat warnings as errors
            
        Returns:
            ValidationResult with validation details
        """
        try:
            validator = self._get_validator(schema_type)
            errors = []
            warnings = []
            
            # Collect all validation errors
            for error in validator.iter_errors(data):
                error_msg = f"Path: {' -> '.join(str(p) for p in error.absolute_path)}, Error: {error.message}"
                
                # Classify errors vs warnings
                if self._is_warning(error):
                    warnings.append(error_msg)
                else:
                    errors.append(error_msg)
            
            # Determine if validation passed
            is_valid = len(errors) == 0 and (not strict or len(warnings) == 0)
            
            # Get schema version if available
            schema = self._schema_cache[schema_type]
            schema_version = schema.get('version', schema.get('$id', ''))
            
            result = ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                schema_version=schema_version
            )
            
            if not is_valid:
                logger.warning(f"Validation failed for {schema_type.value}: {len(errors)} errors, {len(warnings)} warnings")
            else:
                logger.debug(f"Validation passed for {schema_type.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error for {schema_type.value}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation exception: {str(e)}"],
                warnings=[]
            )
    
    def _is_warning(self, error: ValidationError) -> bool:
        """
        Determine if a validation error should be treated as a warning.
        
        Args:
            error: JSON schema validation error
            
        Returns:
            True if error should be treated as warning
        """
        # Treat certain validation failures as warnings rather than errors
        warning_keywords = [
            'additionalProperties',
            'format',
            'deprecated'
        ]
        
        return any(keyword in error.validator for keyword in warning_keywords)
    
    def validate_detection_request(self, data: Any) -> ValidationResult:
        """Validate detection request data."""
        return self.validate(data, SchemaType.DETECTION_REQUEST)
    
    def validate_detection_response(self, data: Any) -> ValidationResult:
        """Validate detection response data."""
        return self.validate(data, SchemaType.DETECTION_RESPONSE)
    
    def validate_extension_message(self, data: Any) -> ValidationResult:
        """Validate Chrome extension message data."""
        return self.validate(data, SchemaType.EXTENSION_MESSAGE)
    
    def validate_training_data(self, data: Any) -> ValidationResult:
        """Validate training dataset data."""
        return self.validate(data, SchemaType.TRAINING_DATA)
    
    def validate_api_error(self, data: Any) -> ValidationResult:
        """Validate API error response data."""
        return self.validate(data, SchemaType.API_ERROR)
    
    def validate_performance_metrics(self, data: Any) -> ValidationResult:
        """Validate performance metrics data."""
        return self.validate(data, SchemaType.PERFORMANCE_METRICS)
    
    def validate_llm_analysis(self, data: Any) -> ValidationResult:
        """Validate LLM analysis response data."""
        return self.validate(data, SchemaType.LLM_ANALYSIS)
    
    def validate_config_settings(self, data: Any) -> ValidationResult:
        """Validate configuration settings data."""
        return self.validate(data, SchemaType.CONFIG_SETTINGS)
    
    def get_schema(self, schema_type: SchemaType) -> Dict[str, Any]:
        """
        Get a loaded schema dictionary.
        
        Args:
            schema_type: Type of schema to retrieve
            
        Returns:
            Schema dictionary
        """
        return self._load_schema(schema_type)
    
    def get_example_data(self, schema_type: SchemaType) -> Optional[Any]:
        """
        Get example data from a schema.
        
        Args:
            schema_type: Type of schema to get examples from
            
        Returns:
            Example data if available, None otherwise
        """
        schema = self._load_schema(schema_type)
        examples = schema.get('examples', [])
        
        if examples:
            return examples[0] if len(examples) == 1 else examples
        
        return None
    
    def list_available_schemas(self) -> List[SchemaType]:
        """
        List all available schema types.
        
        Returns:
            List of available schema types
        """
        available = []
        
        for schema_type in SchemaType:
            schema_file = self.schema_dir / f"{schema_type.value}.json"
            if schema_file.exists():
                available.append(schema_type)
        
        return available
    
    def clear_cache(self):
        """Clear all cached schemas and validators."""
        self._schema_cache.clear()
        self._validator_cache.clear()
        logger.info("Cleared schema and validator cache")


# Global validator instance for convenience
_global_validator: Optional[SchemaValidator] = None


def get_validator(schema_dir: Optional[Union[str, Path]] = None) -> SchemaValidator:
    """
    Get the global schema validator instance.
    
    Args:
        schema_dir: Schema directory (only used for first call)
        
    Returns:
        SchemaValidator instance
    """
    global _global_validator
    
    if _global_validator is None:
        _global_validator = SchemaValidator(schema_dir)
    
    return _global_validator


# Convenience functions for common validations
def validate_detection_request(data: Any) -> ValidationResult:
    """Validate detection request data using global validator."""
    return get_validator().validate_detection_request(data)


def validate_detection_response(data: Any) -> ValidationResult:
    """Validate detection response data using global validator."""
    return get_validator().validate_detection_response(data)


def validate_extension_message(data: Any) -> ValidationResult:
    """Validate extension message data using global validator."""
    return get_validator().validate_extension_message(data)


def validate_config_settings(data: Any) -> ValidationResult:
    """Validate configuration settings using global validator."""
    return get_validator().validate_config_settings(data)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    validator = SchemaValidator()
    
    # Test detection request validation
    test_request = {
        "text": "This is a test message for AI detection.",
        "request_id": "test_req_001",
        "options": {
            "detection_method": "ensemble",
            "confidence_threshold": 0.7
        }
    }
    
    result = validator.validate_detection_request(test_request)
    print(f"Detection request validation: {result.is_valid}")
    
    if not result.is_valid:
        print("Errors:")
        for error in result.errors:
            print(f"  - {error}")
    
    # Test with invalid data
    invalid_request = {
        "text": "",  # Empty text should fail
        "request_id": "invalid id with spaces"  # Spaces not allowed
    }
    
    result = validator.validate_detection_request(invalid_request)
    print(f"Invalid request validation: {result.is_valid}")
    print(f"Errors: {result.errors}")