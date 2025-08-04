# shared/enums.py - Enumerações Unificadas
"""
Enumerações centralizadas que consolidam todos os enums duplicados
em um sistema unificado para o Team Agents.
"""

from enum import Enum, IntEnum


class ComplexityLevel(Enum):
    """Níveis de complexidade unificados para tarefas e operações"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXPERT = "expert"
    CRITICAL = "critical"
    
    @property
    def numeric_value(self) -> int:
        """Retorna valor numérico para comparação"""
        mapping = {
            self.SIMPLE: 1,
            self.MEDIUM: 2,
            self.COMPLEX: 3,
            self.EXPERT: 4,
            self.CRITICAL: 5
        }
        return mapping[self]
    
    def __lt__(self, other):
        if isinstance(other, ComplexityLevel):
            return self.numeric_value < other.numeric_value
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, ComplexityLevel):
            return self.numeric_value <= other.numeric_value
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, ComplexityLevel):
            return self.numeric_value > other.numeric_value
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, ComplexityLevel):
            return self.numeric_value >= other.numeric_value
        return NotImplemented


class AgentRole(Enum):
    """Papéis de agentes unificados"""
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    ANALYST = "analyst"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    MONITOR = "monitor"
    MANAGER = "manager"
    CONSULTANT = "consultant"
    RESEARCHER = "researcher"
    DEVELOPER = "developer"
    DESIGNER = "designer"
    TESTER = "tester"
    ARCHITECT = "architect"
    DATA_SCIENTIST = "data_scientist"
    SECURITY_EXPERT = "security_expert"
    DEVOPS_ENGINEER = "devops_engineer"
    PRODUCT_MANAGER = "product_manager"
    BUSINESS_ANALYST = "business_analyst"
    QUALITY_ASSURANCE = "quality_assurance"
    TECHNICAL_WRITER = "technical_writer"


class SeverityLevel(Enum):
    """Níveis de severidade unificados"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"
    
    @property
    def numeric_value(self) -> int:
        """Retorna valor numérico para comparação"""
        mapping = {
            self.INFO: 0,
            self.LOW: 1,
            self.MEDIUM: 2,
            self.HIGH: 3,
            self.CRITICAL: 4,
            self.EMERGENCY: 5
        }
        return mapping[self]
    
    def __lt__(self, other):
        if isinstance(other, SeverityLevel):
            return self.numeric_value < other.numeric_value
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, SeverityLevel):
            return self.numeric_value <= other.numeric_value
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, SeverityLevel):
            return self.numeric_value > other.numeric_value
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, SeverityLevel):
            return self.numeric_value >= other.numeric_value
        return NotImplemented


class TimeoutLevel(Enum):
    """Níveis de timeout unificados"""
    QUICK = "quick"        # 5-30 segundos
    SHORT = "short"        # 30 segundos - 2 minutos
    MEDIUM = "medium"      # 2-10 minutos
    LONG = "long"          # 10-30 minutos
    EXTENDED = "extended"  # 30+ minutos
    
    @property
    def default_seconds(self) -> int:
        """Retorna tempo padrão em segundos"""
        mapping = {
            self.QUICK: 30,
            self.SHORT: 120,
            self.MEDIUM: 600,
            self.LONG: 1800,
            self.EXTENDED: 3600
        }
        return mapping[self]


class AgentStatus(Enum):
    """Status de agentes unificados"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    INITIALIZING = "initializing"
    STOPPING = "stopping"
    STOPPED = "stopped"


class TaskStatus(Enum):
    """Status de tarefas unificados"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class OperationType(Enum):
    """Tipos de operação unificados"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    VALIDATE = "validate"
    ANALYZE = "analyze"
    PROCESS = "process"
    TRANSFORM = "transform"
    GENERATE = "generate"
    OPTIMIZE = "optimize"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    BACKUP = "backup"
    RESTORE = "restore"


class ResourceType(Enum):
    """Tipos de recurso unificados"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    API = "api"
    FILE = "file"
    SERVICE = "service"
    QUEUE = "queue"
    CACHE = "cache"
    MODEL = "model"
    TOOL = "tool"


class LogLevel(IntEnum):
    """Níveis de log unificados (compatível com logging padrão)"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class ExecutionMode(Enum):
    """Modos de execução unificados"""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"


class DataFormat(Enum):
    """Formatos de dados unificados"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    TEXT = "text"
    BINARY = "binary"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"
    EXCEL = "excel"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class EnvironmentType(Enum):
    """Tipos de ambiente unificados"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    EDGE = "edge"


class SecurityLevel(Enum):
    """Níveis de segurança unificados"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class CachePolicy(Enum):
    """Políticas de cache unificadas"""
    NO_CACHE = "no_cache"
    SHORT_TERM = "short_term"      # 5-15 minutos
    MEDIUM_TERM = "medium_term"    # 1-6 horas
    LONG_TERM = "long_term"        # 1-7 dias
    PERSISTENT = "persistent"      # Até invalidação manual
    
    @property
    def default_ttl_seconds(self) -> int:
        """Retorna TTL padrão em segundos"""
        mapping = {
            self.NO_CACHE: 0,
            self.SHORT_TERM: 900,      # 15 minutos
            self.MEDIUM_TERM: 21600,   # 6 horas
            self.LONG_TERM: 604800,    # 7 dias
            self.PERSISTENT: -1        # Sem expiração
        }
        return mapping[self]


class ValidationRule(Enum):
    """Regras de validação unificadas"""
    REQUIRED = "required"
    NOT_EMPTY = "not_empty"
    MIN_LENGTH = "min_length"
    MAX_LENGTH = "max_length"
    PATTERN = "pattern"
    EMAIL = "email"
    URL = "url"
    NUMERIC = "numeric"
    POSITIVE = "positive"
    RANGE = "range"
    DATE = "date"
    DATETIME = "datetime"
    JSON_SCHEMA = "json_schema"
    CUSTOM = "custom"


# Utility functions para conversão entre enums
def severity_to_complexity(severity: SeverityLevel) -> ComplexityLevel:
    """Converte nível de severidade para complexidade"""
    mapping = {
        SeverityLevel.INFO: ComplexityLevel.SIMPLE,
        SeverityLevel.LOW: ComplexityLevel.SIMPLE,
        SeverityLevel.MEDIUM: ComplexityLevel.MEDIUM,
        SeverityLevel.HIGH: ComplexityLevel.COMPLEX,
        SeverityLevel.CRITICAL: ComplexityLevel.EXPERT,
        SeverityLevel.EMERGENCY: ComplexityLevel.CRITICAL
    }
    return mapping.get(severity, ComplexityLevel.MEDIUM)


def complexity_to_timeout(complexity: ComplexityLevel) -> TimeoutLevel:
    """Converte complexidade para nível de timeout"""
    mapping = {
        ComplexityLevel.SIMPLE: TimeoutLevel.QUICK,
        ComplexityLevel.MEDIUM: TimeoutLevel.SHORT,
        ComplexityLevel.COMPLEX: TimeoutLevel.MEDIUM,
        ComplexityLevel.EXPERT: TimeoutLevel.LONG,
        ComplexityLevel.CRITICAL: TimeoutLevel.EXTENDED
    }
    return mapping.get(complexity, TimeoutLevel.MEDIUM)


def get_enum_values(enum_class) -> list:
    """Retorna lista de valores de um enum"""
    return [item.value for item in enum_class]


def get_enum_by_value(enum_class, value: str):
    """Obtém enum por valor string"""
    for item in enum_class:
        if item.value == value:
            return item
    return None