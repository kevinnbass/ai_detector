# System Overview Diagrams

Visual representations of the AI Detector system architecture using ASCII diagrams and Mermaid charts for comprehensive system understanding.

## High-Level System Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CE[Chrome Extension]
        WEB[Web Interface]
        API_CLIENT[SDK/API Clients]
        CLI[CLI Tools]
    end
    
    subgraph "API Gateway Layer"
        LB[Load Balancer]
        AG[API Gateway]
        RL[Rate Limiter]
        AUTH[Authentication]
    end
    
    subgraph "Core Services"
        DS[Detection Service]
        PS[Processing Service]
        CS[Cache Service]
        MS[Monitoring Service]
    end
    
    subgraph "Detection Engine"
        PD[Pattern Detection]
        ML[ML Classification]
        LLM[LLM Analysis]
        ENS[Ensemble Methods]
    end
    
    subgraph "Data Layer"
        DB[(PostgreSQL)]
        CACHE[(Redis)]
        TS[(InfluxDB)]
        FILES[File Storage]
    end
    
    subgraph "External APIs"
        OR[OpenRouter API]
        GM[Gemini API]
        HF[HuggingFace]
    end
    
    CE --> LB
    WEB --> LB
    API_CLIENT --> LB
    CLI --> LB
    
    LB --> AG
    AG --> RL
    RL --> AUTH
    AUTH --> DS
    
    DS --> PS
    DS --> CS
    DS --> MS
    
    PS --> PD
    PS --> ML
    PS --> LLM
    PS --> ENS
    
    DS --> DB
    CS --> CACHE
    MS --> TS
    DS --> FILES
    
    LLM --> OR
    LLM --> GM
    ML --> HF
```

## Component Interaction Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant AG as API Gateway
    participant DS as Detection Service
    participant CE as Cache Engine
    participant DE as Detection Engine
    participant DB as Database
    participant EXT as External APIs
    
    C->>AG: POST /detect
    AG->>AG: Validate & Authenticate
    AG->>DS: Forward Request
    DS->>CE: Check Cache
    
    alt Cache Hit
        CE-->>DS: Return Cached Result
        DS-->>AG: Response
        AG-->>C: JSON Result
    else Cache Miss
        DS->>DE: Process Text
        DE->>DE: Extract Features
        DE->>DE: Pattern Analysis
        DE->>EXT: LLM Analysis
        EXT-->>DE: AI Assessment
        DE->>DE: Ensemble Vote
        DE-->>DS: Detection Result
        DS->>CE: Cache Result
        DS->>DB: Store Result
        DS-->>AG: Response
        AG-->>C: JSON Result
    end
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Input Sources"
        A[API Requests]
        B[Chrome Extension]
        C[Batch Files]
        D[Stream Data]
    end
    
    subgraph "Processing Pipeline"
        E[Input Validation]
        F[Text Preprocessing]
        G[Feature Extraction]
        H[Detection Methods]
        I[Result Aggregation]
    end
    
    subgraph "Output Destinations"
        J[API Response]
        K[Extension UI]
        L[Database Storage]
        M[Cache Layer]
        N[Monitoring System]
    end
    
    A --> E
    B --> E
    C --> E
    D --> E
    
    E --> F
    F --> G
    G --> H
    H --> I
    
    I --> J
    I --> K
    I --> L
    I --> M
    I --> N
```

## Chrome Extension Architecture

```mermaid
graph TB
    subgraph "Extension Components"
        popup[Popup UI]
        background[Background Script]
        content[Content Script]
        options[Options Page]
    end
    
    subgraph "Chrome APIs"
        storage[chrome.storage]
        runtime[chrome.runtime]
        tabs[chrome.tabs]
        scripting[chrome.scripting]
    end
    
    subgraph "Web Page"
        dom[DOM Elements]
        text[Text Content]
        indicators[Visual Indicators]
    end
    
    subgraph "AI Detector API"
        detect[/detect]
        batch[/detect/batch]
        health[/health]
    end
    
    popup <--> background
    content <--> background
    options <--> background
    
    background <--> storage
    background <--> runtime
    background <--> tabs
    background <--> scripting
    
    content <--> dom
    content <--> text
    content <--> indicators
    
    background <--> detect
    background <--> batch
    background <--> health
```

## High-Throughput Processing Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        A[API Gateway]
        B[Message Queue]
        C[Stream Processor]
    end
    
    subgraph "Processing Layer"
        D[Load Balancer]
        E[Worker Pool]
        F[GPU Processors]
        G[Vectorized Ops]
    end
    
    subgraph "Optimization Layer"
        H[Batch Optimizer]
        I[Memory Manager]
        J[Cache Controller]
        K[Performance Monitor]
    end
    
    subgraph "Output Layer"
        L[Result Aggregator]
        M[Response Formatter]
        N[Storage Writer]
        O[Event Publisher]
    end
    
    A --> D
    B --> D
    C --> D
    
    D --> E
    D --> F
    D --> G
    
    E --> H
    F --> I
    G --> J
    E --> K
    
    H --> L
    I --> L
    J --> L
    K --> L
    
    L --> M
    L --> N
    L --> O
```

## Detection Engine Workflow

```mermaid
flowchart TD
    Start([Input Text]) --> Validate{Validate Input}
    Validate -->|Valid| Preprocess[Text Preprocessing]
    Validate -->|Invalid| Error[Return Error]
    
    Preprocess --> Features[Feature Extraction]
    Features --> Methods{Detection Method}
    
    Methods -->|Pattern| Pattern[Pattern Detection]
    Methods -->|ML| ML[ML Classification]
    Methods -->|LLM| LLM[LLM Analysis]
    Methods -->|Ensemble| All[All Methods]
    
    Pattern --> PatternResult[Pattern Score]
    ML --> MLResult[ML Score]
    LLM --> LLMResult[LLM Score]
    All --> PatternResult
    All --> MLResult
    All --> LLMResult
    
    PatternResult --> Vote[Ensemble Voting]
    MLResult --> Vote
    LLMResult --> Vote
    
    Vote --> Confidence{Confidence > Threshold}
    Confidence -->|Yes| AI[AI Generated]
    Confidence -->|No| Human[Human Written]
    
    AI --> Format[Format Response]
    Human --> Format
    Format --> Cache[Cache Result]
    Cache --> End([Return Result])
```

## Database Schema Design

```mermaid
erDiagram
    DETECTION_REQUESTS {
        uuid id PK
        text content
        string method
        float confidence_threshold
        json options
        timestamp created_at
        string user_id FK
    }
    
    DETECTION_RESULTS {
        uuid id PK
        uuid request_id FK
        boolean is_ai_generated
        float confidence_score
        string detection_method
        float processing_time_ms
        json metadata
        json features
        timestamp created_at
    }
    
    USERS {
        uuid id PK
        string email
        string api_key
        json settings
        timestamp created_at
        timestamp last_active
    }
    
    API_USAGE {
        uuid id PK
        uuid user_id FK
        string endpoint
        int request_count
        timestamp period_start
        timestamp period_end
    }
    
    MODEL_PERFORMANCE {
        uuid id PK
        string model_name
        string version
        float accuracy
        float precision
        float recall
        float f1_score
        timestamp evaluated_at
    }
    
    DETECTION_REQUESTS ||--|| DETECTION_RESULTS : "produces"
    USERS ||--o{ DETECTION_REQUESTS : "makes"
    USERS ||--o{ API_USAGE : "tracks"
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Load Balancer"
        LB[NGINX/HAProxy]
    end
    
    subgraph "Kubernetes Cluster"
        subgraph "API Tier"
            API1[API Pod 1]
            API2[API Pod 2]
            API3[API Pod 3]
        end
        
        subgraph "Processing Tier"
            PROC1[Processor Pod 1]
            PROC2[Processor Pod 2]
            PROC3[GPU Processor Pod]
        end
        
        subgraph "Cache Tier"
            REDIS1[Redis Master]
            REDIS2[Redis Replica]
        end
        
        subgraph "Database Tier"
            DB1[PostgreSQL Primary]
            DB2[PostgreSQL Replica]
        end
    end
    
    subgraph "External Services"
        OPENROUTER[OpenRouter API]
        GEMINI[Gemini API]
        MONITORING[Monitoring Stack]
    end
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> PROC1
    API2 --> PROC2
    API3 --> PROC3
    
    API1 --> REDIS1
    API2 --> REDIS1
    API3 --> REDIS1
    
    REDIS1 --> REDIS2
    
    API1 --> DB1
    API2 --> DB1
    API3 --> DB1
    
    DB1 --> DB2
    
    PROC1 --> OPENROUTER
    PROC2 --> GEMINI
    
    API1 --> MONITORING
    PROC1 --> MONITORING
    REDIS1 --> MONITORING
    DB1 --> MONITORING
```

## Security Architecture

```mermaid
graph TB
    subgraph "Security Layers"
        subgraph "Edge Security"
            WAF[Web Application Firewall]
            DDOS[DDoS Protection]
            GEO[Geo Filtering]
        end
        
        subgraph "Application Security"
            AUTH[Authentication]
            AUTHZ[Authorization]
            VALID[Input Validation]
            RATE[Rate Limiting]
        end
        
        subgraph "Data Security"
            ENCRYPT[Encryption at Rest]
            TLS[TLS in Transit]
            MASK[Data Masking]
            AUDIT[Audit Logging]
        end
        
        subgraph "Infrastructure Security"
            VPC[VPC/Network Security]
            IAM[Identity Management]
            SECRETS[Secret Management]
            MONITOR[Security Monitoring]
        end
    end
    
    Internet --> WAF
    WAF --> DDOS
    DDOS --> GEO
    GEO --> AUTH
    
    AUTH --> AUTHZ
    AUTHZ --> VALID
    VALID --> RATE
    
    RATE --> ENCRYPT
    ENCRYPT --> TLS
    TLS --> MASK
    MASK --> AUDIT
    
    AUDIT --> VPC
    VPC --> IAM
    IAM --> SECRETS
    SECRETS --> MONITOR
```

## Monitoring and Observability

```mermaid
graph LR
    subgraph "Application"
        APP[AI Detector API]
        EXT[Chrome Extension]
        PROC[Processing Workers]
    end
    
    subgraph "Metrics Collection"
        PROM[Prometheus]
        OTEL[OpenTelemetry]
        LOGS[Log Aggregation]
    end
    
    subgraph "Visualization"
        GRAF[Grafana Dashboards]
        ALERT[Alert Manager]
        TRACE[Jaeger Tracing]
    end
    
    subgraph "Storage"
        TSDB[Time Series DB]
        ELASTIC[Elasticsearch]
        OBJECT[Object Storage]
    end
    
    APP --> PROM
    EXT --> OTEL
    PROC --> LOGS
    
    PROM --> GRAF
    OTEL --> TRACE
    LOGS --> ELASTIC
    
    GRAF --> ALERT
    TRACE --> GRAF
    ELASTIC --> GRAF
    
    PROM --> TSDB
    LOGS --> OBJECT
    TRACE --> TSDB
```

## Performance Optimization Flow

```mermaid
flowchart TD
    Input[Text Input] --> Cache{Check Cache}
    Cache -->|Hit| Return[Return Cached Result]
    Cache -->|Miss| Route{Route Request}
    
    Route -->|Small Batch| Fast[Fast Processing]
    Route -->|Large Batch| Parallel[Parallel Processing]
    Route -->|GPU Available| GPU[GPU Processing]
    
    Fast --> SingleCore[Single Core Processing]
    Parallel --> MultiCore[Multi-Core Processing]
    GPU --> CudaOps[CUDA Operations]
    
    SingleCore --> Validate[Validate Result]
    MultiCore --> Aggregate[Aggregate Results]
    CudaOps --> Validate
    
    Aggregate --> Validate
    Validate --> Store[Store in Cache]
    Store --> Monitor[Update Metrics]
    Monitor --> Return
```

These diagrams provide a comprehensive visual representation of the AI Detector system architecture, showing how components interact, data flows through the system, and how different architectural patterns are implemented for scalability, security, and performance.

---

*Last updated: January 15, 2025*