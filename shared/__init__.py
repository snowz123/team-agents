# shared/__init__.py - Módulos Base Compartilhados do Team Agents
"""
Módulos base compartilhados que consolidam classes duplicadas e fornecem
interfaces unificadas para todo o sistema Team Agents.
"""

from .exceptions import (
    ValidationError,
    TeamAgentsException,
    ErrorContext,
    ErrorSeverity
)

from .enums import (
    ComplexityLevel,
    AgentRole,
    SeverityLevel,
    TimeoutLevel
)

from .models import (
    ProjectContext,
    TaskRequest,
    ExecutionResult
)

from .logging import (
    LoggingContextManager,
    setup_unified_logging
)

from .config import (
    UnifiedConfig,
    TimeoutManager
)

__all__ = [
    # Exceptions
    'ValidationError',
    'TeamAgentsException', 
    'ErrorContext',
    'ErrorSeverity',
    
    # Enums
    'ComplexityLevel',
    'AgentRole',
    'SeverityLevel',
    'TimeoutLevel',
    
    # Models
    'ProjectContext',
    'TaskRequest',
    'ExecutionResult',
    
    # Logging
    'LoggingContextManager',
    'setup_unified_logging',
    
    # Config
    'UnifiedConfig',
    'TimeoutManager'
]