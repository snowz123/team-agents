# advanced_error_handling.py - Sistema Avançado de Tratamento de Erros
"""
Sistema robusto de tratamento de erros, logging inteligente e recuperação
automática para o Team Agents.
"""

import sys
import traceback
import logging
import functools
import inspect
from typing import Dict, List, Any, Optional, Callable, Type, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue
import json
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from shared.exceptions import TeamAgentsException, ErrorSeverity, ErrorContext
from shared.logging import get_logging_manager
from shared.enums import SeverityLevel


class ErrorCategory(Enum):
    """Categorias de erro do sistema"""
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"


class RecoveryStrategy(Enum):
    """Estratégias de recuperação de erro"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    IGNORE = "ignore"
    ESCALATE = "escalate"


@dataclass
class ErrorPattern:
    """Padrão de erro para detecção automática"""
    pattern_id: str
    error_types: List[Type[Exception]]
    keywords: List[str]
    category: ErrorCategory
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    escalation_threshold: int = 5
    description: str = ""


@dataclass
class ErrorOccurrence:
    """Ocorrência de erro registrada"""
    error_id: str
    timestamp: datetime
    exception: Exception
    context: ErrorContext
    pattern: Optional[ErrorPattern]
    stack_trace: str
    system_state: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    escalated: bool = False


@dataclass
class ErrorSummary:
    """Resumo de erros por período"""
    start_time: datetime
    end_time: datetime
    total_errors: int
    errors_by_category: Dict[ErrorCategory, int]
    errors_by_severity: Dict[ErrorSeverity, int]
    top_error_patterns: List[Dict[str, Any]]
    recovery_success_rate: float
    most_affected_components: List[str]


class CircuitBreaker:
    """Implementação de Circuit Breaker pattern"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func):
        """Decorator para aplicar circuit breaker"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == "OPEN":
                    if self.last_failure_time and \
                       (datetime.now() - self.last_failure_time).seconds < self.reset_timeout:
                        raise TeamAgentsException(
                            "Circuit breaker is OPEN - service unavailable",
                            severity=ErrorSeverity.HIGH
                        )
                    else:
                        self.state = "HALF_OPEN"
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Sucesso - resetar contador
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                    
                    return result
                    
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = datetime.now()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                    
                    raise
        
        return wrapper


class RetryHandler:
    """Gerenciador de retry com backoff exponencial"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def __call__(self, func):
        """Decorator para retry automático"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt < self.max_retries:
                        # Calcular delay com backoff exponencial
                        delay = min(
                            self.base_delay * (self.exponential_base ** attempt),
                            self.max_delay
                        )
                        
                        import time
                        time.sleep(delay)
                    else:
                        # Última tentativa falhou
                        break
            
            # Todas as tentativas falharam
            raise TeamAgentsException(
                f"Operation failed after {self.max_retries + 1} attempts",
                details={'last_error': str(last_exception), 'attempts': self.max_retries + 1}
            ) from last_exception
        
        return wrapper


class ErrorAnalyzer:
    """Analisador inteligente de erros"""
    
    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.error_history = []
        self.pattern_stats = {}
        self.logger = get_logging_manager().get_logger("error_analyzer")
    
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Inicializa padrões de erro conhecidos"""
        return [
            ErrorPattern(
                pattern_id="database_connection_error",
                error_types=[ConnectionError, TimeoutError],
                keywords=["connection", "database", "timeout", "refused"],
                category=ErrorCategory.DATABASE,
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.RETRY,
                max_retries=3,
                retry_delay=2.0,
                description="Erro de conexão com banco de dados"
            ),
            
            ErrorPattern(
                pattern_id="network_timeout",
                error_types=[TimeoutError, ConnectionError],
                keywords=["timeout", "network", "connection", "unreachable"],
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.RETRY,
                max_retries=2,
                retry_delay=1.0,
                description="Timeout de rede"
            ),
            
            ErrorPattern(
                pattern_id="memory_error",
                error_types=[MemoryError],
                keywords=["memory", "out of memory", "allocation"],
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.CRITICAL,
                recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                description="Erro de memória insuficiente"
            ),
            
            ErrorPattern(
                pattern_id="validation_error",
                error_types=[ValueError, TypeError],
                keywords=["validation", "invalid", "type", "format"],
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.FAIL_FAST,
                description="Erro de validação de dados"
            ),
            
            ErrorPattern(
                pattern_id="api_rate_limit",
                error_types=[Exception],
                keywords=["rate limit", "too many requests", "429", "quota"],
                category=ErrorCategory.EXTERNAL_API,
                severity=ErrorSeverity.MEDIUM,
                recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                retry_delay=60.0,
                description="Rate limit de API externa"
            ),
            
            ErrorPattern(
                pattern_id="authentication_error",
                error_types=[PermissionError],
                keywords=["authentication", "unauthorized", "403", "401", "forbidden"],
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.ESCALATE,
                description="Erro de autenticação"
            )
        ]
    
    def analyze_error(self, exception: Exception, context: ErrorContext = None) -> ErrorOccurrence:
        """Analisa erro e determina padrão e estratégia"""
        error_text = str(exception).lower()
        stack_trace = traceback.format_exc()
        
        # Buscar padrão correspondente
        matching_pattern = None
        for pattern in self.error_patterns:
            # Verificar tipo de exceção
            if any(isinstance(exception, error_type) for error_type in pattern.error_types):
                matching_pattern = pattern
                break
            
            # Verificar palavras-chave
            if any(keyword in error_text for keyword in pattern.keywords):
                matching_pattern = pattern
                break
        
        # Coletar estado do sistema
        system_state = self._collect_system_state()
        
        # Criar ocorrência de erro
        error_occurrence = ErrorOccurrence(
            error_id=f"error_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            exception=exception,
            context=context or ErrorContext(
                error_id="",
                timestamp=datetime.now(),
                severity=ErrorSeverity.MEDIUM,
                component="unknown",
                operation="unknown",
                details={},
                user_message=str(exception),
                technical_message=str(exception),
                suggestions=[]
            ),
            pattern=matching_pattern,
            stack_trace=stack_trace,
            system_state=system_state
        )
        
        # Registrar na história
        self.error_history.append(error_occurrence)
        
        # Atualizar estatísticas do padrão
        if matching_pattern:
            pattern_id = matching_pattern.pattern_id
            if pattern_id not in self.pattern_stats:
                self.pattern_stats[pattern_id] = {'count': 0, 'last_occurrence': None}
            
            self.pattern_stats[pattern_id]['count'] += 1
            self.pattern_stats[pattern_id]['last_occurrence'] = datetime.now()
        
        self.logger.info(f"Erro analisado: {error_occurrence.error_id}, padrão: {matching_pattern.pattern_id if matching_pattern else 'unknown'}")
        
        return error_occurrence
    
    def _collect_system_state(self) -> Dict[str, Any]:
        """Coleta estado atual do sistema"""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads(),
                'open_files': len(process.open_files()) if hasattr(process, 'open_files') else 0,
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {'error': 'Unable to collect system state'}
    
    def get_error_summary(self, hours: int = 24) -> ErrorSummary:
        """Gera resumo de erros do período"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e.timestamp > cutoff_time]
        
        if not recent_errors:
            return ErrorSummary(
                start_time=cutoff_time,
                end_time=datetime.now(),
                total_errors=0,
                errors_by_category={},
                errors_by_severity={},
                top_error_patterns=[],
                recovery_success_rate=0.0,
                most_affected_components=[]
            )
        
        # Contar por categoria
        errors_by_category = {}
        for error in recent_errors:
            if error.pattern:
                category = error.pattern.category
                errors_by_category[category] = errors_by_category.get(category, 0) + 1
        
        # Contar por severidade
        errors_by_severity = {}
        for error in recent_errors:
            severity = error.context.severity
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        # Top padrões
        pattern_counts = {}
        for error in recent_errors:
            if error.pattern:
                pattern_id = error.pattern.pattern_id
                pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1
        
        top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_error_patterns = [{'pattern': p, 'count': c} for p, c in top_patterns]
        
        # Taxa de recuperação
        recovery_attempts = len([e for e in recent_errors if e.recovery_attempted])
        successful_recoveries = len([e for e in recent_errors if e.recovery_successful])
        recovery_success_rate = (successful_recoveries / recovery_attempts) if recovery_attempts > 0 else 0.0
        
        # Componentes mais afetados
        component_counts = {}
        for error in recent_errors:
            component = error.context.component
            component_counts[component] = component_counts.get(component, 0) + 1
        
        most_affected = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        most_affected_components = [comp for comp, count in most_affected]
        
        return ErrorSummary(
            start_time=cutoff_time,
            end_time=datetime.now(),
            total_errors=len(recent_errors),
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            top_error_patterns=top_error_patterns,
            recovery_success_rate=recovery_success_rate,
            most_affected_components=most_affected_components
        )


class NotificationManager:
    """Gerenciador de notificações de erro"""
    
    def __init__(self):
        self.email_config = None
        self.slack_config = None
        self.notification_queue = queue.Queue()
        self.notification_thread = None
        self.running = False
        self.logger = get_logging_manager().get_logger("notifications")
    
    def configure_email(self, smtp_server: str, smtp_port: int, 
                       username: str, password: str, from_addr: str):
        """Configura notificações por email"""
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_addr': from_addr
        }
    
    def start_notification_processor(self):
        """Inicia processador de notificações"""
        if self.running:
            return
        
        self.running = True
        self.notification_thread = threading.Thread(
            target=self._process_notifications, 
            daemon=True
        )
        self.notification_thread.start()
        self.logger.info("Processador de notificações iniciado")
    
    def stop_notification_processor(self):
        """Para processador de notificações"""
        self.running = False
        if self.notification_thread:
            self.notification_thread.join(timeout=5)
        self.logger.info("Processador de notificações parado")
    
    def queue_notification(self, error_occurrence: ErrorOccurrence):
        """Adiciona notificação à fila"""
        # Verificar se deve notificar baseado na severidade
        if error_occurrence.context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.notification_queue.put(error_occurrence)
    
    def _process_notifications(self):
        """Loop de processamento de notificações"""
        while self.running:
            try:
                # Timeout para verificar se deve parar
                error_occurrence = self.notification_queue.get(timeout=1)
                
                self._send_notification(error_occurrence)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Erro ao processar notificação: {e}")
    
    def _send_notification(self, error_occurrence: ErrorOccurrence):
        """Envia notificação específica"""
        try:
            if self.email_config:
                self._send_email_notification(error_occurrence)
        except Exception as e:
            self.logger.error(f"Erro ao enviar notificação: {e}")
    
    def _send_email_notification(self, error_occurrence: ErrorOccurrence):
        """Envia notificação por email"""
        if not self.email_config:
            return
        
        # Criar mensagem
        msg = MIMEMultipart()
        msg['From'] = self.email_config['from_addr']
        msg['To'] = "admin@teamagents.com"  # Configurável
        msg['Subject'] = f"[Team Agents] Erro {error_occurrence.context.severity.value.upper()}"
        
        # Corpo da mensagem
        body = f"""
        Erro detectado no sistema Team Agents:
        
        ID do Erro: {error_occurrence.error_id}
        Timestamp: {error_occurrence.timestamp}
        Severidade: {error_occurrence.context.severity.value}
        Componente: {error_occurrence.context.component}
        Operação: {error_occurrence.context.operation}
        
        Mensagem: {error_occurrence.context.user_message}
        
        Detalhes Técnicos: {error_occurrence.context.technical_message}
        
        Stack Trace:
        {error_occurrence.stack_trace}
        
        Estado do Sistema:
        {json.dumps(error_occurrence.system_state, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Enviar email
        with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
        
        self.logger.info(f"Notificação enviada por email para erro {error_occurrence.error_id}")


class AdvancedErrorHandler:
    """Gerenciador principal de tratamento de erros"""
    
    def __init__(self):
        self.analyzer = ErrorAnalyzer()
        self.notification_manager = NotificationManager()
        self.recovery_handlers = {}
        self.circuit_breakers = {}
        self.logger = get_logging_manager().get_logger("error_handler")
        
        # Iniciar processador de notificações
        self.notification_manager.start_notification_processor()
    
    def handle_error(self, exception: Exception, context: ErrorContext = None, 
                    attempt_recovery: bool = True) -> bool:
        """Trata erro de forma inteligente"""
        
        # Analisar erro
        error_occurrence = self.analyzer.analyze_error(exception, context)
        
        # Log detalhado
        self.logger.error(
            f"Erro tratado: {error_occurrence.error_id}",
            extra={
                'error_id': error_occurrence.error_id,
                'pattern': error_occurrence.pattern.pattern_id if error_occurrence.pattern else None,
                'severity': error_occurrence.context.severity.value,
                'component': error_occurrence.context.component
            }
        )
        
        # Enviar notificação se necessário
        self.notification_manager.queue_notification(error_occurrence)
        
        # Tentar recuperação se solicitado
        recovery_successful = False
        if attempt_recovery and error_occurrence.pattern:
            recovery_successful = self._attempt_recovery(error_occurrence)
        
        return recovery_successful
    
    def _attempt_recovery(self, error_occurrence: ErrorOccurrence) -> bool:
        """Tenta recuperação automática baseada no padrão"""
        pattern = error_occurrence.pattern
        strategy = pattern.recovery_strategy
        
        error_occurrence.recovery_attempted = True
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                # Retry já seria implementado por decorator
                return True
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                # Circuit breaker previne chamadas futuras
                component = error_occurrence.context.component
                if component not in self.circuit_breakers:
                    self.circuit_breakers[component] = CircuitBreaker()
                return True
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                # Implementar degradação elegante
                self._enable_degraded_mode(error_occurrence.context.component)
                return True
            
            elif strategy == RecoveryStrategy.ESCALATE:
                # Escalar erro para nível superior
                self._escalate_error(error_occurrence)
                return False
            
            elif strategy == RecoveryStrategy.FAIL_FAST:
                # Falhar rapidamente sem tentar recuperar
                return False
            
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Erro durante tentativa de recuperação: {e}")
            return False
    
    def _enable_degraded_mode(self, component: str):
        """Habilita modo degradado para componente"""
        self.logger.warning(f"Modo degradado habilitado para componente: {component}")
        # Implementação específica dependeria do componente
    
    def _escalate_error(self, error_occurrence: ErrorOccurrence):
        """Escala erro para nível superior"""
        error_occurrence.escalated = True
        self.logger.critical(f"Erro escalado: {error_occurrence.error_id}")
        
        # Notificação imediata para erros escalados
        self.notification_manager.queue_notification(error_occurrence)
    
    def register_recovery_handler(self, error_pattern_id: str, handler: Callable):
        """Registra handler customizado de recuperação"""
        self.recovery_handlers[error_pattern_id] = handler
        self.logger.info(f"Handler de recuperação registrado para padrão: {error_pattern_id}")
    
    def get_error_dashboard_data(self) -> Dict[str, Any]:
        """Retorna dados para dashboard de erros"""
        summary_24h = self.analyzer.get_error_summary(hours=24)
        summary_1h = self.analyzer.get_error_summary(hours=1)
        
        return {
            'current_timestamp': datetime.now().isoformat(),
            'last_24h': {
                'total_errors': summary_24h.total_errors,
                'errors_by_category': {k.value: v for k, v in summary_24h.errors_by_category.items()},
                'errors_by_severity': {k.value: v for k, v in summary_24h.errors_by_severity.items()},
                'recovery_success_rate': summary_24h.recovery_success_rate,
                'most_affected_components': summary_24h.most_affected_components
            },
            'last_1h': {
                'total_errors': summary_1h.total_errors,
                'errors_by_category': {k.value: v for k, v in summary_1h.errors_by_category.items()}
            },
            'circuit_breakers': {
                component: {
                    'state': cb.state,
                    'failure_count': cb.failure_count
                } for component, cb in self.circuit_breakers.items()
            },
            'recent_errors': [
                {
                    'error_id': e.error_id,
                    'timestamp': e.timestamp.isoformat(),
                    'component': e.context.component,
                    'severity': e.context.severity.value,
                    'pattern': e.pattern.pattern_id if e.pattern else None,
                    'recovered': e.recovery_successful
                }
                for e in self.analyzer.error_history[-10:]  # Últimos 10 erros
            ]
        }
    
    def cleanup_and_shutdown(self):
        """Limpeza antes do shutdown"""
        self.logger.info("Parando sistema de tratamento de erros")
        self.notification_manager.stop_notification_processor()
        
        # Salvar relatório final
        dashboard_data = self.get_error_dashboard_data()
        with open('error_report.json', 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        self.logger.info("Sistema de tratamento de erros finalizado")


# Instância global
_global_error_handler = None

def get_error_handler() -> AdvancedErrorHandler:
    """Obtém instância global do tratador de erros"""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = AdvancedErrorHandler()
    return _global_error_handler


# Decorators principais
def handle_errors(component: str = "unknown", operation: str = "unknown"):
    """Decorator principal para tratamento de erros"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error_id="",
                    timestamp=datetime.now(),
                    severity=ErrorSeverity.MEDIUM,
                    component=component,
                    operation=operation,
                    details={'function': func.__name__},
                    user_message=str(e),
                    technical_message=str(e),
                    suggestions=[]
                )
                
                error_handler = get_error_handler()
                recovery_successful = error_handler.handle_error(e, context)
                
                if not recovery_successful:
                    raise
                
                return None  # ou valor padrão apropriado
        
        return wrapper
    return decorator


def circuit_breaker(failure_threshold: int = 5, reset_timeout: int = 60):
    """Decorator para circuit breaker"""
    return CircuitBreaker(failure_threshold, reset_timeout)


def retry_on_failure(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator para retry automático"""
    return RetryHandler(max_retries, base_delay)