# shared/exceptions.py - Sistema Unificado de Exceções
"""
Sistema centralizado de exceções para o Team Agents que consolida
todas as classes de exceção duplicadas em uma hierarquia unificada.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
import traceback
import uuid


class ErrorSeverity(Enum):
    """Níveis de severidade de erro unificados"""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Contexto unificado de erro com informações detalhadas"""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    component: str
    operation: str
    details: Dict[str, Any]
    user_message: str
    technical_message: str
    suggestions: List[str]
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        if not self.error_id:
            self.error_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável"""
        return {
            'error_id': self.error_id,
            'timestamp': self.timestamp.isoformat(),
            'severity': self.severity.value,
            'component': self.component,
            'operation': self.operation,
            'details': self.details,
            'user_message': self.user_message,
            'technical_message': self.technical_message,
            'suggestions': self.suggestions,
            'stack_trace': self.stack_trace
        }
    
    def to_user_message(self) -> str:
        """Gera mensagem amigável para o usuário"""
        msg = f"[{self.severity.value.upper()}] {self.user_message}"
        if self.suggestions:
            msg += "\n\nSugestões:"
            for suggestion in self.suggestions:
                msg += f"\n- {suggestion}"
        return msg


class TeamAgentsException(Exception):
    """Exceção base do sistema Team Agents"""
    
    def __init__(self, message: str, context: Optional[ErrorContext] = None, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 component: str = "unknown", operation: str = "unknown",
                 details: Optional[Dict[str, Any]] = None,
                 suggestions: Optional[List[str]] = None):
        super().__init__(message)
        
        if context is None:
            context = ErrorContext(
                error_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                severity=severity,
                component=component,
                operation=operation,
                details=details or {},
                user_message=message,
                technical_message=message,
                suggestions=suggestions or [],
                stack_trace=traceback.format_exc()
            )
        
        self.context = context
    
    def __str__(self) -> str:
        return f"[{self.context.component}:{self.context.operation}] {self.context.technical_message}"
    
    def get_user_message(self) -> str:
        """Retorna mensagem amigável para o usuário"""
        return self.context.to_user_message()
    
    def get_context_dict(self) -> Dict[str, Any]:
        """Retorna contexto como dicionário"""
        return self.context.to_dict()


class ValidationError(TeamAgentsException):
    """Erro de validação unificado"""
    
    def __init__(self, message: str, field_name: str = None, 
                 field_value: Any = None, validation_rule: str = None,
                 **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'field_name': field_name,
            'field_value': str(field_value) if field_value is not None else None,
            'validation_rule': validation_rule
        })
        
        suggestions = kwargs.get('suggestions', [])
        if field_name and not suggestions:
            suggestions.append(f"Verifique o valor do campo '{field_name}'")
        
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            component=kwargs.get('component', 'validation'),
            operation=kwargs.get('operation', 'validate_field'),
            details=details,
            suggestions=suggestions
        )


class ConfigurationError(TeamAgentsException):
    """Erro de configuração"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({'config_key': config_key})
        
        suggestions = kwargs.get('suggestions', [])
        if config_key and not suggestions:
            suggestions.append(f"Verifique a configuração da chave '{config_key}'")
        
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            component=kwargs.get('component', 'configuration'),
            operation=kwargs.get('operation', 'load_config'),
            details=details,
            suggestions=suggestions
        )


class AgentExecutionError(TeamAgentsException):
    """Erro de execução de agente"""
    
    def __init__(self, message: str, agent_id: str = None, 
                 task_id: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'agent_id': agent_id,
            'task_id': task_id
        })
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Verifique se o agente está configurado corretamente",
                "Verifique se os recursos necessários estão disponíveis",
                "Considere tentar novamente com diferentes parâmetros"
            ])
        
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            component=kwargs.get('component', 'agent_execution'),
            operation=kwargs.get('operation', 'execute_task'),
            details=details,
            suggestions=suggestions
        )


class TimeoutError(TeamAgentsException):
    """Erro de timeout unificado"""
    
    def __init__(self, message: str, timeout_seconds: float = None,
                 operation_type: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'timeout_seconds': timeout_seconds,
            'operation_type': operation_type
        })
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Considere aumentar o tempo limite da operação",
                "Verifique se há problemas de conectividade",
                "Tente dividir a operação em partes menores"
            ])
        
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            component=kwargs.get('component', 'timeout'),
            operation=kwargs.get('operation', 'wait_operation'),
            details=details,
            suggestions=suggestions
        )


class DatabaseError(TeamAgentsException):
    """Erro de banco de dados"""
    
    def __init__(self, message: str, query: str = None, 
                 connection_string: str = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'query': query,
            'connection_string': connection_string
        })
        
        suggestions = kwargs.get('suggestions', [])
        if not suggestions:
            suggestions.extend([
                "Verifique se o banco de dados está acessível",
                "Verifique se as credenciais estão corretas",
                "Verifique se a query está bem formada"
            ])
        
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            component=kwargs.get('component', 'database'),
            operation=kwargs.get('operation', 'database_operation'),
            details=details,
            suggestions=suggestions
        )


class APIError(TeamAgentsException):
    """Erro de API"""
    
    def __init__(self, message: str, endpoint: str = None,
                 status_code: int = None, response_data: Any = None, **kwargs):
        details = kwargs.get('details', {})
        details.update({
            'endpoint': endpoint,
            'status_code': status_code,
            'response_data': str(response_data) if response_data is not None else None
        })
        
        suggestions = kwargs.get('suggestions', [])
        if status_code and not suggestions:
            if status_code == 401:
                suggestions.append("Verifique as credenciais de autenticação")
            elif status_code == 403:
                suggestions.append("Verifique as permissões de acesso")
            elif status_code == 404:
                suggestions.append("Verifique se o endpoint está correto")
            elif status_code >= 500:
                suggestions.append("Problema no servidor - tente novamente mais tarde")
        
        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            component=kwargs.get('component', 'api'),
            operation=kwargs.get('operation', 'api_call'),
            details=details,
            suggestions=suggestions
        )


# Funções utilitárias para criação de exceções
def create_validation_error(message: str, field_name: str = None, 
                          field_value: Any = None) -> ValidationError:
    """Cria erro de validação com contexto padrão"""
    return ValidationError(
        message=message,
        field_name=field_name,
        field_value=field_value,
        suggestions=[
            "Verifique se o valor está no formato correto",
            "Consulte a documentação para valores válidos"
        ]
    )


def create_agent_error(message: str, agent_id: str = None, 
                      task_id: str = None) -> AgentExecutionError:
    """Cria erro de agente com contexto padrão"""
    return AgentExecutionError(
        message=message,
        agent_id=agent_id,
        task_id=task_id
    )


def create_timeout_error(message: str, timeout_seconds: float = None,
                        operation: str = None) -> TimeoutError:
    """Cria erro de timeout com contexto padrão"""
    return TimeoutError(
        message=message,
        timeout_seconds=timeout_seconds,
        operation_type=operation
    )


# Exception handler decorator
def handle_team_agents_exceptions(component: str = "unknown"):
    """Decorator para capturar e converter exceções em TeamAgentsException"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except TeamAgentsException:
                raise  # Re-raise TeamAgentsException as-is
            except Exception as e:
                # Convert other exceptions to TeamAgentsException
                raise TeamAgentsException(
                    message=str(e),
                    component=component,
                    operation=func.__name__,
                    severity=ErrorSeverity.HIGH,
                    details={'original_exception': type(e).__name__},
                    suggestions=[
                        "Verifique o log detalhado para mais informações",
                        "Entre em contato com o suporte se o problema persistir"
                    ]
                )
        return wrapper
    return decorator