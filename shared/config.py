# shared/config.py - Sistema de Configuração Unificado
"""
Sistema centralizado de configuração que consolida todas as classes de configuração
e timeout em uma implementação unificada para o Team Agents.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime, timedelta
import threading
import time
from enum import Enum

from .enums import TimeoutLevel, EnvironmentType, SecurityLevel
from .exceptions import ConfigurationError, TimeoutError


@dataclass
class TimeoutConfig:
    """Configuração unificada de timeouts"""
    default_timeout: int = 300  # 5 minutos
    quick_timeout: int = 30     # 30 segundos
    short_timeout: int = 120    # 2 minutos
    medium_timeout: int = 600   # 10 minutos
    long_timeout: int = 1800    # 30 minutos
    extended_timeout: int = 3600  # 1 hora
    
    # Timeouts específicos por operação
    database_timeout: int = 30
    api_timeout: int = 60
    file_operation_timeout: int = 120
    network_timeout: int = 90
    computation_timeout: int = 1800
    
    # Configurações de retry
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    def get_timeout(self, level: TimeoutLevel) -> int:
        """Obtém timeout baseado no nível"""
        mapping = {
            TimeoutLevel.QUICK: self.quick_timeout,
            TimeoutLevel.SHORT: self.short_timeout,
            TimeoutLevel.MEDIUM: self.medium_timeout,
            TimeoutLevel.LONG: self.long_timeout,
            TimeoutLevel.EXTENDED: self.extended_timeout
        }
        return mapping.get(level, self.default_timeout)


@dataclass
class DatabaseConfig:
    """Configuração unificada de banco de dados"""
    connection_string: str = "sqlite:///team_agents.db"
    pool_size: int = 10
    max_overflow: int = 20
    connection_timeout: int = 30
    query_timeout: int = 60
    enable_echo: bool = False
    enable_pool_pre_ping: bool = True
    
    # Configurações de backup
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 30
    backup_path: str = "backups/database"


@dataclass
class LoggingConfig:
    """Configuração unificada de logging"""
    level: str = "INFO"
    format: str = "structured"  # "structured" ou "simple"
    enable_console: bool = True
    enable_file: bool = True
    enable_remote: bool = False
    
    # Configurações de arquivo
    log_file: str = "logs/team_agents.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Configurações de logging remoto
    remote_endpoint: Optional[str] = None
    remote_api_key: Optional[str] = None
    
    # Filtros de logging
    exclude_components: List[str] = field(default_factory=list)
    include_performance_logs: bool = True


@dataclass
class SecurityConfig:
    """Configuração unificada de segurança"""
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_encryption: bool = True
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    # Configurações de autenticação
    jwt_secret_key: Optional[str] = None
    jwt_expiration_hours: int = 24
    password_min_length: int = 8
    require_password_complexity: bool = True
    
    # Configurações de criptografia
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_days: int = 90
    
    # Configurações de auditoria
    enable_audit_log: bool = True
    audit_log_retention_days: int = 365


@dataclass
class PerformanceConfig:
    """Configuração unificada de performance"""
    max_concurrent_tasks: int = 10
    max_memory_usage_mb: int = 1024
    max_cpu_usage_percent: float = 80.0
    
    # Cache
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    max_cache_size_mb: int = 512
    
    # Otimizações
    enable_lazy_loading: bool = True
    enable_batch_processing: bool = True
    batch_size: int = 100
    
    # Monitoramento
    enable_metrics: bool = True
    metrics_collection_interval: int = 60


@dataclass
class IntegrationConfig:
    """Configuração unificada de integrações"""
    # APIs externas
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Webhooks
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # Notificações
    email_enabled: bool = False
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    
    # Slack
    slack_enabled: bool = False
    slack_token: Optional[str] = None
    slack_channel: Optional[str] = None


class UnifiedConfig:
    """Sistema de configuração unificado"""
    
    def __init__(self, config_file: Optional[str] = None, 
                 environment: EnvironmentType = EnvironmentType.DEVELOPMENT):
        self.environment = environment
        self.config_file = config_file or self._get_default_config_file()
        self._config_data = {}
        self._lock = threading.Lock()
        self._watchers = []
        
        # Configurações por seção
        self.timeout = TimeoutConfig()
        self.database = DatabaseConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.integration = IntegrationConfig()
        
        # Carregar configuração
        self.load_config()
        
        # Configurar a partir de variáveis de ambiente
        self._load_from_environment()
    
    def _get_default_config_file(self) -> str:
        """Obtém arquivo de configuração padrão baseado no ambiente"""
        base_name = f"config_{self.environment.value}"
        
        # Tentar diferentes formatos
        for ext in ['.yaml', '.yml', '.json']:
            config_path = f"config/{base_name}{ext}"
            if Path(config_path).exists():
                return config_path
        
        # Retornar padrão se não encontrar
        return f"config/{base_name}.yaml"
    
    def load_config(self):
        """Carrega configuração do arquivo"""
        if not Path(self.config_file).exists():
            self._create_default_config()
            return
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.endswith('.json'):
                    self._config_data = json.load(f)
                else:
                    self._config_data = yaml.safe_load(f) or {}
            
            # Aplicar configurações carregadas
            self._apply_loaded_config()
            
        except Exception as e:
            raise ConfigurationError(
                f"Erro ao carregar configuração de {self.config_file}",
                config_key="config_file",
                details={'file_path': self.config_file, 'error': str(e)}
            )
    
    def _create_default_config(self):
        """Cria arquivo de configuração padrão"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        default_config = {
            'environment': self.environment.value,
            'timeout': asdict(self.timeout),
            'database': asdict(self.database),
            'logging': asdict(self.logging),
            'security': asdict(self.security),
            'performance': asdict(self.performance),
            'integration': asdict(self.integration)
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                if self.config_file.endswith('.json'):
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                else:
                    yaml.dump(default_config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
            
            self._config_data = default_config
            
        except Exception as e:
            raise ConfigurationError(
                f"Erro ao criar configuração padrão em {self.config_file}",
                config_key="config_file",
                details={'file_path': self.config_file, 'error': str(e)}
            )
    
    def _apply_loaded_config(self):
        """Aplica configuração carregada aos objetos de configuração"""
        try:
            if 'timeout' in self._config_data:
                for key, value in self._config_data['timeout'].items():
                    if hasattr(self.timeout, key):
                        setattr(self.timeout, key, value)
            
            if 'database' in self._config_data:
                for key, value in self._config_data['database'].items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            if 'logging' in self._config_data:
                for key, value in self._config_data['logging'].items():
                    if hasattr(self.logging, key):
                        setattr(self.logging, key, value)
            
            if 'security' in self._config_data:
                for key, value in self._config_data['security'].items():
                    if hasattr(self.security, key):
                        if key == 'security_level' and isinstance(value, str):
                            setattr(self.security, key, SecurityLevel(value))
                        else:
                            setattr(self.security, key, value)
            
            if 'performance' in self._config_data:
                for key, value in self._config_data['performance'].items():
                    if hasattr(self.performance, key):
                        setattr(self.performance, key, value)
            
            if 'integration' in self._config_data:
                for key, value in self._config_data['integration'].items():
                    if hasattr(self.integration, key):
                        setattr(self.integration, key, value)
                        
        except Exception as e:
            raise ConfigurationError(
                f"Erro ao aplicar configuração carregada",
                config_key="apply_config",
                details={'error': str(e)}
            )
    
    def _load_from_environment(self):
        """Carrega configurações de variáveis de ambiente"""
        env_mappings = {
            # Database
            'DATABASE_URL': ('database', 'connection_string'),
            'DB_POOL_SIZE': ('database', 'pool_size'),
            
            # Security  
            'JWT_SECRET_KEY': ('security', 'jwt_secret_key'),
            'ENCRYPTION_KEY': ('security', 'encryption_key'),
            
            # Integration
            'OPENAI_API_KEY': ('integration', 'openai_api_key'),
            'ANTHROPIC_API_KEY': ('integration', 'anthropic_api_key'),
            'GOOGLE_API_KEY': ('integration', 'google_api_key'),
            
            # Logging
            'LOG_LEVEL': ('logging', 'level'),
            'REMOTE_LOG_ENDPOINT': ('logging', 'remote_endpoint'),
            
            # Performance
            'MAX_CONCURRENT_TASKS': ('performance', 'max_concurrent_tasks'),
            'MAX_MEMORY_MB': ('performance', 'max_memory_usage_mb'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                
                # Converter tipo se necessário
                current_value = getattr(section_obj, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                
                setattr(section_obj, key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtém valor de configuração"""
        try:
            keys = key.split('.')
            value = self._config_data
            
            for k in keys:
                value = value[k]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Define valor de configuração"""
        with self._lock:
            keys = key.split('.')
            current = self._config_data
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
    
    def save_config(self):
        """Salva configuração atual no arquivo"""
        with self._lock:
            try:
                # Atualizar dados com configurações atuais
                self._config_data.update({
                    'timeout': asdict(self.timeout),
                    'database': asdict(self.database),
                    'logging': asdict(self.logging),
                    'security': asdict(self.security),
                    'performance': asdict(self.performance),
                    'integration': asdict(self.integration)
                })
                
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    if self.config_file.endswith('.json'):
                        json.dump(self._config_data, f, indent=2, ensure_ascii=False, default=str)
                    else:
                        yaml.dump(self._config_data, f, default_flow_style=False,
                                 allow_unicode=True, indent=2)
                        
            except Exception as e:
                raise ConfigurationError(
                    f"Erro ao salvar configuração em {self.config_file}",
                    config_key="save_config",
                    details={'file_path': self.config_file, 'error': str(e)}
                )
    
    def reload_config(self):
        """Recarrega configuração do arquivo"""
        self.load_config()
    
    def validate_config(self) -> List[str]:
        """Valida configuração atual"""
        errors = []
        
        # Validar timeouts
        if self.timeout.default_timeout <= 0:
            errors.append("timeout.default_timeout deve ser maior que 0")
        
        # Validar database
        if not self.database.connection_string:
            errors.append("database.connection_string é obrigatório")
        
        # Validar security
        if self.security.enable_authentication and not self.security.jwt_secret_key:
            errors.append("security.jwt_secret_key é obrigatório quando autenticação está habilitada")
        
        # Validar performance
        if self.performance.max_concurrent_tasks <= 0:
            errors.append("performance.max_concurrent_tasks deve ser maior que 0")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte configuração para dicionário"""
        return {
            'environment': self.environment.value,
            'timeout': asdict(self.timeout),
            'database': asdict(self.database),
            'logging': asdict(self.logging),
            'security': asdict(self.security),
            'performance': asdict(self.performance),
            'integration': asdict(self.integration)
        }


class TimeoutManager:
    """Gerenciador unificado de timeouts"""
    
    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TimeoutConfig()
        self._active_operations = {}
        self._lock = threading.Lock()
    
    def execute_with_timeout(self, func, timeout_level: TimeoutLevel = TimeoutLevel.MEDIUM,
                           custom_timeout: Optional[int] = None, *args, **kwargs):
        """Executa função com timeout"""
        timeout_seconds = custom_timeout or self.config.get_timeout(timeout_level)
        
        operation_id = str(id(func)) + str(time.time())
        
        def target():
            try:
                result = func(*args, **kwargs)
                with self._lock:
                    self._active_operations[operation_id] = {'status': 'completed', 'result': result}
            except Exception as e:
                with self._lock:
                    self._active_operations[operation_id] = {'status': 'failed', 'error': e}
        
        # Iniciar thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        
        with self._lock:
            self._active_operations[operation_id] = {'status': 'running', 'thread': thread}
        
        thread.start()
        thread.join(timeout_seconds)
        
        with self._lock:
            operation = self._active_operations.get(operation_id, {})
            
            if thread.is_alive():
                # Timeout ocorreu
                operation['status'] = 'timeout'
                raise TimeoutError(
                    f"Operação {func.__name__} excedeu timeout de {timeout_seconds} segundos",
                    timeout_seconds=timeout_seconds,
                    operation_type=func.__name__
                )
            
            elif operation.get('status') == 'completed':
                return operation.get('result')
            
            elif operation.get('status') == 'failed':
                raise operation.get('error')
            
            else:
                raise TimeoutError(
                    f"Operação {func.__name__} terminou em estado desconhecido",
                    timeout_seconds=timeout_seconds,
                    operation_type=func.__name__
                )
    
    def get_active_operations(self) -> Dict[str, Dict[str, Any]]:
        """Retorna operações ativas"""
        with self._lock:
            return {k: {
                'status': v.get('status'),
                'start_time': v.get('start_time')
            } for k, v in self._active_operations.items()}


# Instância global de configuração
_global_config = None

def get_config() -> UnifiedConfig:
    """Obtém instância global de configuração"""
    global _global_config
    if _global_config is None:
        _global_config = UnifiedConfig()
    return _global_config


def initialize_config(config_file: str = None, 
                     environment: EnvironmentType = EnvironmentType.DEVELOPMENT) -> UnifiedConfig:
    """Inicializa sistema de configuração"""
    global _global_config
    _global_config = UnifiedConfig(config_file, environment)
    return _global_config