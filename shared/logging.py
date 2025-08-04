# shared/logging.py - Sistema de Logging Unificado
"""
Sistema centralizado de logging que consolida todas as classes de logging
duplicadas em uma implementação unificada para o Team Agents.
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import uuid
import threading
from pathlib import Path

from .enums import LogLevel, SeverityLevel
from .exceptions import TeamAgentsException


@dataclass
class LogContext:
    """Contexto unificado de log"""
    correlation_id: str
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}


class StructuredFormatter(logging.Formatter):
    """Formatador estruturado para logs JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Criar estrutura base do log
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process
        }
        
        # Adicionar contexto se disponível
        if hasattr(record, 'context') and record.context:
            log_entry['context'] = asdict(record.context)
        
        # Adicionar informações de exceção se disponível
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        # Adicionar campos customizados
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'context']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class LoggingContextManager:
    """Gerenciador de contexto unificado para logging"""
    
    _local = threading.local()
    
    def __init__(self):
        self.handlers = {}
        self.loggers = {}
        self._setup_default_logging()
    
    def _setup_default_logging(self):
        """Configura logging padrão"""
        # Criar diretório de logs se não existir
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configurar logger raiz
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remover handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Handler para console (desenvolvimento)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # Handler para arquivo JSON (produção)
        file_handler = logging.FileHandler(
            log_dir / "team_agents.log", 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
        
        # Handler para erros
        error_handler = logging.FileHandler(
            log_dir / "team_agents_errors.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(error_handler)
        
        self.handlers['console'] = console_handler
        self.handlers['file'] = file_handler
        self.handlers['error'] = error_handler
    
    @classmethod
    def get_context(cls) -> Optional[LogContext]:
        """Obtém contexto atual da thread"""
        return getattr(cls._local, 'context', None)
    
    @classmethod
    def set_context(cls, context: LogContext):
        """Define contexto para a thread atual"""
        cls._local.context = context
    
    @classmethod
    def clear_context(cls):
        """Limpa contexto da thread atual"""
        if hasattr(cls._local, 'context'):
            delattr(cls._local, 'context')
    
    @contextmanager
    def context(self, correlation_id: str = None, component: str = "unknown",
                operation: str = "unknown", user_id: str = None,
                session_id: str = None, request_id: str = None,
                metadata: Dict[str, Any] = None):
        """Context manager para logging com contexto"""
        context = LogContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            component=component,
            operation=operation,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            metadata=metadata or {}
        )
        
        old_context = self.get_context()
        self.set_context(context)
        
        try:
            yield context
        finally:
            if old_context:
                self.set_context(old_context)
            else:
                self.clear_context()
    
    def get_logger(self, name: str) -> logging.Logger:
        """Obtém logger com nome específico"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            
            # Adicionar método personalizado para log com contexto
            def log_with_context(level, message, *args, **kwargs):
                context = self.get_context()
                extra = kwargs.get('extra', {})
                if context:
                    extra['context'] = context
                kwargs['extra'] = extra
                logger.log(level, message, *args, **kwargs)
            
            logger.log_with_context = log_with_context
            self.loggers[name] = logger
        
        return self.loggers[name]
    
    def configure_component_logger(self, component: str, level: LogLevel = LogLevel.INFO,
                                 additional_handlers: List[logging.Handler] = None):
        """Configura logger específico para componente"""
        logger = self.get_logger(component)
        logger.setLevel(level.value)
        
        if additional_handlers:
            for handler in additional_handlers:
                logger.addHandler(handler)
        
        return logger
    
    def log_structured(self, level: LogLevel, message: str, component: str,
                      operation: str = "unknown", **kwargs):
        """Log estruturado com contexto automático"""
        logger = self.get_logger(component)
        
        # Obter ou criar contexto
        context = self.get_context()
        if not context:
            context = LogContext(
                correlation_id=str(uuid.uuid4()),
                component=component,
                operation=operation
            )
        
        # Adicionar dados extras
        extra = {
            'context': context,
            **kwargs
        }
        
        logger.log(level.value, message, extra=extra)
    
    def log_exception(self, exception: Exception, component: str,
                     operation: str = "unknown", additional_info: Dict[str, Any] = None):
        """Log especializado para exceções"""
        logger = self.get_logger(component)
        
        context = self.get_context() or LogContext(
            correlation_id=str(uuid.uuid4()),
            component=component,
            operation=operation
        )
        
        extra = {
            'context': context,
            'exception_type': type(exception).__name__,
            'exception_message': str(exception)
        }
        
        if additional_info:
            extra.update(additional_info)
        
        # Se for TeamAgentsException, usar contexto da exceção
        if isinstance(exception, TeamAgentsException):
            extra['error_context'] = exception.get_context_dict()
        
        logger.error(f"Exception in {operation}: {str(exception)}", 
                    exc_info=True, extra=extra)
    
    def log_performance(self, component: str, operation: str, 
                       duration_seconds: float, success: bool = True,
                       additional_metrics: Dict[str, float] = None):
        """Log de métricas de performance"""
        logger = self.get_logger(f"{component}.performance")
        
        context = self.get_context() or LogContext(
            correlation_id=str(uuid.uuid4()),
            component=component,
            operation=operation
        )
        
        metrics = {
            'duration_seconds': duration_seconds,
            'success': success,
            'operations_per_second': 1 / duration_seconds if duration_seconds > 0 else 0
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        extra = {
            'context': context,
            'performance_metrics': metrics
        }
        
        status = "completed" if success else "failed"
        message = f"Operation {operation} {status} in {duration_seconds:.3f}s"
        
        logger.info(message, extra=extra)
    
    def setup_file_rotation(self, max_bytes: int = 10 * 1024 * 1024,
                           backup_count: int = 5):
        """Configura rotação de arquivos de log"""
        from logging.handlers import RotatingFileHandler
        
        log_dir = Path("logs")
        
        # Substituir handler de arquivo por handler rotativo
        if 'file' in self.handlers:
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.handlers['file'])
        
        rotating_handler = RotatingFileHandler(
            log_dir / "team_agents.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        rotating_handler.setLevel(logging.DEBUG)
        rotating_handler.setFormatter(StructuredFormatter())
        
        root_logger = logging.getLogger()
        root_logger.addHandler(rotating_handler)
        self.handlers['file'] = rotating_handler
    
    def add_remote_handler(self, endpoint: str, api_key: str = None):
        """Adiciona handler para logging remoto"""
        # Implementação básica - pode ser estendida para diferentes provedores
        class RemoteHandler(logging.Handler):
            def __init__(self, endpoint: str, api_key: str = None):
                super().__init__()
                self.endpoint = endpoint
                self.api_key = api_key
            
            def emit(self, record):
                try:
                    import requests
                    log_entry = self.format(record)
                    headers = {'Content-Type': 'application/json'}
                    if self.api_key:
                        headers['Authorization'] = f'Bearer {self.api_key}'
                    
                    requests.post(self.endpoint, data=log_entry, headers=headers, timeout=5)
                except:
                    pass  # Falha silenciosa para não afetar aplicação
        
        remote_handler = RemoteHandler(endpoint, api_key)
        remote_handler.setFormatter(StructuredFormatter())
        
        root_logger = logging.getLogger()
        root_logger.addHandler(remote_handler)
        self.handlers['remote'] = remote_handler


# Instância global do gerenciador de logging
_global_logging_manager = None

def get_logging_manager() -> LoggingContextManager:
    """Obtém instância global do gerenciador de logging"""
    global _global_logging_manager
    if _global_logging_manager is None:
        _global_logging_manager = LoggingContextManager()
    return _global_logging_manager


def setup_unified_logging(log_level: LogLevel = LogLevel.INFO,
                         enable_console: bool = True,
                         enable_file: bool = True,
                         enable_rotation: bool = True,
                         log_dir: str = "logs") -> LoggingContextManager:
    """Configura sistema de logging unificado"""
    # Criar diretório de logs
    Path(log_dir).mkdir(exist_ok=True)
    
    # Obter gerenciador global
    manager = get_logging_manager()
    
    # Configurar nível de log
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.value)
    
    # Configurar rotação se solicitado
    if enable_rotation:
        manager.setup_file_rotation()
    
    return manager


# Funções de conveniência para logging
def log_info(message: str, component: str = "team_agents", **kwargs):
    """Log de informação"""
    manager = get_logging_manager()
    manager.log_structured(LogLevel.INFO, message, component, **kwargs)


def log_error(message: str, component: str = "team_agents", **kwargs):
    """Log de erro"""
    manager = get_logging_manager()
    manager.log_structured(LogLevel.ERROR, message, component, **kwargs)


def log_warning(message: str, component: str = "team_agents", **kwargs):
    """Log de aviso"""
    manager = get_logging_manager()
    manager.log_structured(LogLevel.WARNING, message, component, **kwargs)


def log_debug(message: str, component: str = "team_agents", **kwargs):
    """Log de debug"""
    manager = get_logging_manager()
    manager.log_structured(LogLevel.DEBUG, message, component, **kwargs)


def log_exception(exception: Exception, component: str = "team_agents", **kwargs):
    """Log de exceção"""
    manager = get_logging_manager()
    manager.log_exception(exception, component, **kwargs)