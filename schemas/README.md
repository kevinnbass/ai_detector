# JSON Schema Documentation

This directory contains JSON Schema definitions for all data formats used across the AI Detector system. These schemas provide structure validation, type safety, and documentation for data exchanged between components.

## Schema Files

### Core Detection Schemas

- **`detection_request.json`** - Schema for AI detection requests across all components
- **`detection_response.json`** - Schema for AI detection responses from all analyzers  
- **`llm_analysis.json`** - Schema for structured LLM analysis responses with quantified dimensions

### Extension Communication

- **`extension_message.json`** - Schema for messages between Chrome extension components (content script, background, popup)

### Training and Data

- **`training_data.json`** - Schema for training datasets used in AI detection models

### System Operations

- **`api_error.json`** - Standardized error response schema for all API endpoints
- **`performance_metrics.json`** - Schema for performance monitoring and benchmarking data
- **`config_settings.json`** - Schema for application configuration settings across all components

## Usage

### Python Validation

```python
import json
import jsonschema
from jsonschema import validate

# Load schema
with open('schemas/detection_request.json', 'r') as f:
    schema = json.load(f)

# Validate data
data = {
    "text": "This is a test message",
    "request_id": "req_123",
    "options": {"detection_method": "ensemble"}
}

try:
    validate(instance=data, schema=schema)
    print("Data is valid")
except jsonschema.exceptions.ValidationError as err:
    print(f"Validation error: {err}")
```

### JavaScript Validation

```javascript
const Ajv = require('ajv');
const schema = require('./schemas/detection_request.json');

const ajv = new Ajv();
const validate = ajv.compile(schema);

const data = {
    text: "This is a test message",
    request_id: "req_123",
    options: { detection_method: "ensemble" }
};

const valid = validate(data);
if (!valid) {
    console.log('Validation errors:', validate.errors);
}
```

### API Integration

```python
# Example integration with FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json

# Load schema and create Pydantic model
with open('schemas/detection_request.json', 'r') as f:
    schema = json.load(f)

class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    request_id: str = Field(..., regex=r'^[a-zA-Z0-9_-]+$')
    options: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
```

## Schema Validation in CI/CD

The schemas are automatically validated in our CI/CD pipeline:

```yaml
- name: Validate JSON schemas
  run: |
    python -c "
    import json
    import jsonschema
    from pathlib import Path
    
    schema_dir = Path('schemas')
    for schema_file in schema_dir.glob('*.json'):
        with open(schema_file) as f:
            schema = json.load(f)
        jsonschema.Draft7Validator.check_schema(schema)
        print(f'✓ {schema_file.name} is valid')
    "
```

## Schema Development Guidelines

### 1. Naming Conventions
- Use snake_case for property names
- Use clear, descriptive names
- Include units in names when applicable (e.g., `timeout_ms`, `memory_mb`)

### 2. Required vs Optional Fields
- Mark essential fields as `required`
- Use `default` values for optional fields when sensible
- Document when fields are conditionally required

### 3. Validation Rules
- Set appropriate `minimum` and `maximum` values
- Use `pattern` for string format validation
- Use `enum` for restricted value sets
- Include `minLength`/`maxLength` for strings

### 4. Documentation
- Always include `description` fields
- Provide clear examples in the schema
- Document edge cases and special behavior

### 5. Backwards Compatibility
- Use `additionalProperties: false` carefully
- Consider schema versioning for breaking changes
- Add new optional fields rather than modifying existing ones

## Integration Points

### Extension Components
- Content scripts validate messages using `extension_message.json`
- Background scripts use `detection_request.json` and `detection_response.json`
- Popup components use `config_settings.json` for user preferences

### API Layer
- REST endpoints validate requests/responses using appropriate schemas
- Error responses follow `api_error.json` format
- Performance metrics collected according to `performance_metrics.json`

### Training Pipeline
- Dataset validation using `training_data.json`
- LLM analysis output validated with `llm_analysis.json`
- Configuration managed through `config_settings.json`

## Tools and Libraries

### Recommended Libraries

**Python:**
- `jsonschema` - JSON Schema validation
- `pydantic` - Data validation using Python type hints
- `cerberus` - Lightweight validation library

**JavaScript:**
- `ajv` - The fastest JSON schema validator
- `joi` - Object schema validation
- `yup` - Schema validation with TypeScript support

**Development Tools:**
- `json-schema-faker` - Generate mock data from schemas
- `quicktype` - Generate types from JSON Schema
- `json-schema-to-typescript` - Generate TypeScript interfaces

## Maintenance

### Regular Tasks
1. Review schemas quarterly for completeness
2. Update examples with real-world data
3. Validate against actual API responses
4. Check for unused or deprecated fields

### Version Management
- Schemas follow semantic versioning
- Breaking changes increment major version
- New optional fields increment minor version
- Documentation fixes increment patch version

### Testing
All schemas are tested with:
- Valid example data
- Invalid data to verify error handling
- Edge cases and boundary conditions
- Real production data samples

For questions or contributions to the schema definitions, please refer to the main project documentation or create an issue in the repository.