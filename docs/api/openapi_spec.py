"""
Generate OpenAPI specification for AI Detector API
Creates comprehensive API documentation
"""

import json
from typing import Dict, Any
from pathlib import Path

# Import FastAPI app to generate spec
from src.api.server import app


def generate_openapi_spec() -> Dict[str, Any]:
    """Generate OpenAPI specification"""
    return app.openapi()


def save_openapi_spec(output_file: str = "openapi.json"):
    """Save OpenAPI spec to file"""
    spec = generate_openapi_spec()
    
    # Ensure output directory exists
    output_path = Path(__file__).parent / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    # Save spec
    with open(output_path, 'w') as f:
        json.dump(spec, f, indent=2)
    
    print(f"OpenAPI specification saved to {output_path}")
    return output_path


def generate_api_documentation():
    """Generate comprehensive API documentation"""
    spec = generate_openapi_spec()
    
    # Extract information for documentation
    info = spec.get("info", {})
    paths = spec.get("paths", {})
    components = spec.get("components", {})
    
    # Generate markdown documentation
    docs = []
    docs.append(f"# {info.get('title', 'API Documentation')}")
    docs.append(f"\n**Version:** {info.get('version', '1.0.0')}")
    docs.append(f"\n**Description:** {info.get('description', '')}")
    
    # Add base information
    docs.append("\n## Base Information")
    docs.append("- **Base URL:** `/api/v1`")
    docs.append("- **Authentication:** JWT Bearer Token or API Key")
    docs.append("- **Content Type:** `application/json`")
    
    # Add endpoints
    docs.append("\n## Endpoints")
    
    for path, methods in paths.items():
        docs.append(f"\n### {path}")
        
        for method, details in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                docs.append(f"\n#### {method.upper()}")
                docs.append(f"**Summary:** {details.get('summary', 'No summary')}")
                
                if 'description' in details:
                    docs.append(f"\n**Description:** {details['description']}")
                
                # Parameters
                if 'parameters' in details:
                    docs.append("\n**Parameters:**")
                    for param in details['parameters']:
                        required = " (required)" if param.get('required') else ""
                        docs.append(f"- `{param.get('name')}`{required}: {param.get('description', '')}")
                
                # Request body
                if 'requestBody' in details:
                    docs.append("\n**Request Body:** Required")
                    content = details['requestBody'].get('content', {})
                    if 'application/json' in content:
                        schema_ref = content['application/json'].get('schema', {}).get('$ref', '')
                        if schema_ref:
                            schema_name = schema_ref.split('/')[-1]
                            docs.append(f"Schema: `{schema_name}`")
                
                # Responses
                if 'responses' in details:
                    docs.append("\n**Responses:**")
                    for status_code, response_info in details['responses'].items():
                        docs.append(f"- `{status_code}`: {response_info.get('description', '')}")
    
    # Add schemas
    docs.append("\n## Data Models")
    
    schemas = components.get('schemas', {})
    for schema_name, schema_def in schemas.items():
        docs.append(f"\n### {schema_name}")
        
        if 'description' in schema_def:
            docs.append(f"{schema_def['description']}")
        
        if 'properties' in schema_def:
            docs.append("\n**Properties:**")
            required_fields = schema_def.get('required', [])
            
            for prop_name, prop_def in schema_def['properties'].items():
                required = " (required)" if prop_name in required_fields else ""
                prop_type = prop_def.get('type', 'unknown')
                prop_desc = prop_def.get('description', '')
                docs.append(f"- `{prop_name}` ({prop_type}){required}: {prop_desc}")
    
    # Add authentication section
    docs.append("\n## Authentication")
    docs.append("\nThe API supports two authentication methods:")
    docs.append("\n### JWT Bearer Token")
    docs.append("Include in header: `Authorization: Bearer <token>`")
    docs.append("\n### API Key")
    docs.append("Include in header: `X-API-Key: <api_key>`")
    
    # Add rate limiting section
    docs.append("\n## Rate Limiting")
    docs.append("\nAPI requests are rate-limited per endpoint:")
    docs.append("- `/detect`: 60 requests/minute")
    docs.append("- `/detect/batch`: 10 requests/minute") 
    docs.append("- `/train`: 5 requests/5 minutes")
    docs.append("- Default: 100 requests/minute")
    docs.append("\nRate limit headers are included in responses:")
    docs.append("- `X-RateLimit-Limit`: Request limit")
    docs.append("- `X-RateLimit-Remaining`: Remaining requests")
    docs.append("- `X-RateLimit-Reset`: Reset time")
    
    # Add WebSocket section
    docs.append("\n## WebSocket API")
    docs.append("\n### Connection")
    docs.append("Connect to: `ws://localhost:8000/ws?token=<jwt_token>`")
    docs.append("\n### Message Format")
    docs.append("```json")
    docs.append("{")
    docs.append('  "type": "message_type",')
    docs.append('  "data": {...},')
    docs.append('  "correlation_id": "optional_id"')
    docs.append("}")
    docs.append("```")
    docs.append("\n### Supported Message Types")
    docs.append("- `ping`: Health check")
    docs.append("- `detection_request`: Real-time text analysis")
    docs.append("- `subscribe`: Subscribe to events")
    
    # Add error handling section
    docs.append("\n## Error Handling")
    docs.append("\nAll API responses follow a consistent format:")
    docs.append("```json")
    docs.append("{")
    docs.append('  "success": boolean,')
    docs.append('  "message": "string",')
    docs.append('  "data": {},')
    docs.append('  "errors": [],')
    docs.append('  "timestamp": "ISO_8601_string"')
    docs.append("}")
    docs.append("```")
    
    # Add HTTP status codes
    docs.append("\n### HTTP Status Codes")
    docs.append("- `200`: Success")
    docs.append("- `201`: Created")
    docs.append("- `400`: Bad Request")
    docs.append("- `401`: Unauthorized")
    docs.append("- `403`: Forbidden")
    docs.append("- `404`: Not Found")
    docs.append("- `422`: Validation Error")
    docs.append("- `429`: Rate Limit Exceeded")
    docs.append("- `500`: Internal Server Error")
    
    return "\n".join(docs)


def save_api_documentation(output_file: str = "api_documentation.md"):
    """Save API documentation to markdown file"""
    docs = generate_api_documentation()
    
    output_path = Path(__file__).parent / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(docs)
    
    print(f"API documentation saved to {output_path}")
    return output_path


if __name__ == "__main__":
    # Generate both OpenAPI spec and documentation
    save_openapi_spec()
    save_api_documentation()
    
    print("API documentation generation complete!")
    print("- OpenAPI Spec: docs/api/openapi.json")
    print("- Markdown Docs: docs/api/api_documentation.md")
    print("- Interactive Docs: http://localhost:8000/docs")
    print("- ReDoc: http://localhost:8000/redoc")