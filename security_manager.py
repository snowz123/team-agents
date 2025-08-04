# security_manager.py - Gerenciador de Segurança Crítico
"""
Sistema de segurança que resolve vulnerabilidades críticas identificadas
na análise do sistema Team Agents.
"""

import re
import os
import hashlib
import secrets
import subprocess
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import shlex

from shared.exceptions import TeamAgentsException, SecurityError
from shared.enums import SecurityLevel


class CommandSafetyLevel(Enum):
    """Níveis de segurança para execução de comandos"""
    FORBIDDEN = "forbidden"  # Nunca permitir
    RESTRICTED = "restricted"  # Apenas com validação rigorosa
    SAFE = "safe"  # Permitir com validação básica
    TRUSTED = "trusted"  # Permitir sem restrições (admin apenas)


@dataclass
class SecurityPolicy:
    """Política de segurança"""
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_extensions: Set[str] = None
    forbidden_patterns: Set[str] = None
    max_input_length: int = 10000
    enable_command_execution: bool = False
    trusted_users: Set[str] = None
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = {
                '.txt', '.json', '.yaml', '.yml', '.csv', '.py', '.js', '.html', '.css'
            }
        
        if self.forbidden_patterns is None:
            self.forbidden_patterns = {
                # Comandos perigosos
                r'\b(rm\s+-rf|del\s+/[sqf]|format\s+c:)',
                r'\b(shutdown|reboot|halt)\b',
                r'\b(sudo|su\s+root)\b',
                r'\b(chmod\s+777|chown\s+root)\b',
                # Injeção de código
                r'[;&|`$(){}]',
                r'(exec|eval|system|shell_exec)\s*\(',
                r'__(import|eval)__',
                # SQL injection
                r"(union\s+select|drop\s+table|delete\s+from)",
                r"(insert\s+into|update\s+set|alter\s+table)",
                # Scripts maliciosos
                r'<script[^>]*>',
                r'javascript:',
                r'vbscript:',
            }
        
        if self.trusted_users is None:
            self.trusted_users = set()


class SecureInputValidator:
    """Validador seguro de inputs"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = logging.getLogger("security.validator")
        
        # Compilar padrões regex para melhor performance
        self.compiled_forbidden_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in policy.forbidden_patterns
        ]
    
    def validate_string_input(self, value: str, max_length: int = None) -> bool:
        """Valida input de string"""
        if not isinstance(value, str):
            raise SecurityError("Input deve ser string")
        
        # Verificar tamanho
        max_len = max_length or self.policy.max_input_length
        if len(value) > max_len:
            raise SecurityError(f"Input muito longo: {len(value)} > {max_len}")
        
        # Verificar padrões proibidos
        for pattern in self.compiled_forbidden_patterns:
            if pattern.search(value):
                raise SecurityError(f"Padrão proibido detectado: {pattern.pattern}")
        
        return True
    
    def validate_file_path(self, file_path: str) -> bool:
        """Valida caminho de arquivo"""
        self.validate_string_input(file_path)
        
        path = Path(file_path)
        
        # Verificar tentativas de path traversal
        if '..' in file_path or file_path.startswith('/'):
            if not file_path.startswith(str(Path.cwd())):
                raise SecurityError("Path traversal detectado")
        
        # Verificar extensão
        if path.suffix.lower() not in self.policy.allowed_file_extensions:
            raise SecurityError(f"Extensão não permitida: {path.suffix}")
        
        return True
    
    def validate_sql_query(self, query: str) -> bool:
        """Valida query SQL"""
        self.validate_string_input(query)
        
        # Padrões específicos para SQL
        dangerous_sql = [
            r'\b(drop|delete|truncate|alter)\s+',
            r'\bunion\s+select\b',
            r'--\s*$',
            r'/\*.*\*/',
            r"'\s*(or|and)\s+.*=",
        ]
        
        for pattern in dangerous_sql:
            if re.search(pattern, query, re.IGNORECASE):
                raise SecurityError(f"SQL perigoso detectado: {pattern}")
        
        return True
    
    def sanitize_input(self, value: str) -> str:
        """Sanitiza input removendo caracteres perigosos"""
        if not isinstance(value, str):
            return str(value)
        
        # Remover caracteres de controle
        sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        
        # Escapar caracteres especiais HTML
        html_escape = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
        }
        
        for char, escape in html_escape.items():
            sanitized = sanitized.replace(char, escape)
        
        return sanitized


class SecureCommandExecutor:
    """Executor seguro de comandos"""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.logger = logging.getLogger("security.executor")
        
        # Comandos permitidos por categoria
        self.safe_commands = {
            'ls', 'dir', 'pwd', 'whoami', 'date', 'echo',
            'cat', 'head', 'tail', 'grep', 'find', 'wc'
        }
        
        self.restricted_commands = {
            'python', 'node', 'npm', 'pip', 'git', 'docker'
        }
        
        self.forbidden_commands = {
            'rm', 'del', 'rmdir', 'format', 'fdisk',
            'shutdown', 'reboot', 'halt', 'poweroff',
            'sudo', 'su', 'chmod', 'chown', 'passwd',
            'useradd', 'userdel', 'groupadd', 'crontab'
        }
    
    def get_command_safety_level(self, command: str) -> CommandSafetyLevel:
        """Determina nível de segurança do comando"""
        cmd_parts = shlex.split(command.lower())
        if not cmd_parts:
            return CommandSafetyLevel.FORBIDDEN
        
        base_command = cmd_parts[0]
        
        if base_command in self.forbidden_commands:
            return CommandSafetyLevel.FORBIDDEN
        elif base_command in self.safe_commands:
            return CommandSafetyLevel.SAFE
        elif base_command in self.restricted_commands:
            return CommandSafetyLevel.RESTRICTED
        else:
            return CommandSafetyLevel.FORBIDDEN  # Por padrão, proibir
    
    def execute_safe_command(self, command: str, user_id: str = None,
                           timeout: int = 30) -> Dict[str, Any]:
        """Executa comando de forma segura"""
        
        if not self.policy.enable_command_execution:
            raise SecurityError("Execução de comandos desabilitada")
        
        # Verificar nível de segurança
        safety_level = self.get_command_safety_level(command)
        
        if safety_level == CommandSafetyLevel.FORBIDDEN:
            raise SecurityError(f"Comando proibido: {command}")
        
        if safety_level == CommandSafetyLevel.RESTRICTED:
            if user_id not in self.policy.trusted_users:
                raise SecurityError("Usuário não autorizado para comando restrito")
        
        # Validar comando
        validator = SecureInputValidator(self.policy)
        validator.validate_string_input(command, max_length=1000)
        
        # Log da execução
        self.logger.warning(f"Executando comando: {command} (usuário: {user_id})")
        
        try:
            # Executar com timeout e captura segura
            result = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path.cwd()),
                env=self._get_safe_environment()
            )
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout[:10000],  # Limitar output
                'stderr': result.stderr[:10000],
                'returncode': result.returncode,
                'command': command
            }
            
        except subprocess.TimeoutExpired:
            raise SecurityError(f"Comando excedeu timeout de {timeout}s")
        except Exception as e:
            raise SecurityError(f"Erro na execução: {str(e)}")
    
    def _get_safe_environment(self) -> Dict[str, str]:
        """Retorna ambiente seguro para execução"""
        # Ambiente mínimo e seguro
        safe_env = {
            'PATH': '/usr/bin:/bin',
            'HOME': str(Path.home()),
            'USER': os.getenv('USER', 'unknown'),
            'LANG': 'en_US.UTF-8'
        }
        
        return safe_env


class DatabaseSecurityManager:
    """Gerenciador de segurança para banco de dados"""
    
    def __init__(self):
        self.logger = logging.getLogger("security.database")
    
    def validate_connection_string(self, conn_string: str) -> bool:
        """Valida string de conexão"""
        if not conn_string:
            raise SecurityError("String de conexão vazia")
        
        # Verificar se contém credenciais em texto claro
        if re.search(r'password\s*=\s*[^;]+', conn_string, re.IGNORECASE):
            self.logger.warning("Credenciais em texto claro detectadas")
        
        # Verificar URLs suspeitas
        suspicious_hosts = ['localhost', '127.0.0.1', '0.0.0.0']
        for host in suspicious_hosts:
            if host in conn_string and 'production' in os.getenv('ENVIRONMENT', ''):
                self.logger.warning(f"Host suspeito em produção: {host}")
        
        return True
    
    def sanitize_query_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitiza parâmetros de query"""
        sanitized = {}
        validator = SecureInputValidator(SecurityPolicy())
        
        for key, value in params.items():
            if isinstance(value, str):
                sanitized[key] = validator.sanitize_input(value)
            else:
                sanitized[key] = value
        
        return sanitized


class SecurityManager:
    """Gerenciador principal de segurança"""
    
    def __init__(self, policy: SecurityPolicy = None):
        self.policy = policy or SecurityPolicy()
        self.validator = SecureInputValidator(self.policy)
        self.executor = SecureCommandExecutor(self.policy)
        self.db_security = DatabaseSecurityManager()
        self.logger = logging.getLogger("security.manager")
        
        # Audit log
        self._setup_audit_logging()
    
    def _setup_audit_logging(self):
        """Configura logging de auditoria"""
        audit_logger = logging.getLogger("security.audit")
        handler = logging.FileHandler("logs/security_audit.log")
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        audit_logger.addHandler(handler)
        audit_logger.setLevel(logging.INFO)
        
        self.audit_logger = audit_logger
    
    def validate_api_input(self, data: Dict[str, Any], endpoint: str = "unknown") -> Dict[str, Any]:
        """Valida input de API"""
        self.audit_logger.info(f"Validando input para endpoint: {endpoint}")
        
        sanitized = {}
        
        for key, value in data.items():
            try:
                if isinstance(value, str):
                    self.validator.validate_string_input(value)
                    sanitized[key] = self.validator.sanitize_input(value)
                elif isinstance(value, dict):
                    sanitized[key] = self.validate_api_input(value, f"{endpoint}.{key}")
                elif isinstance(value, list):
                    sanitized[key] = [
                        self.validator.sanitize_input(item) if isinstance(item, str) else item
                        for item in value
                    ]
                else:
                    sanitized[key] = value
                    
            except SecurityError as e:
                self.audit_logger.error(f"Validação falhou para {endpoint}.{key}: {e}")
                raise
        
        return sanitized
    
    def secure_file_operation(self, file_path: str, operation: str = "read") -> bool:
        """Valida operação de arquivo"""
        self.validator.validate_file_path(file_path)
        
        path = Path(file_path)
        
        # Verificar se arquivo existe e tamanho
        if path.exists() and operation == "read":
            if path.stat().st_size > self.policy.max_file_size:
                raise SecurityError(f"Arquivo muito grande: {path.stat().st_size}")
        
        # Log da operação
        self.audit_logger.info(f"Operação de arquivo: {operation} - {file_path}")
        
        return True
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Gera token seguro"""
        return secrets.token_urlsafe(length)
    
    def hash_password(self, password: str) -> str:
        """Hash seguro de senha"""
        salt = secrets.token_bytes(32)
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return salt.hex() + key.hex()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verifica senha hashada"""
        try:
            salt = bytes.fromhex(hashed[:64])
            key = bytes.fromhex(hashed[64:])
            new_key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
            return new_key == key
        except:
            return False
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100,
                        window_seconds: int = 3600) -> bool:
        """Verifica rate limiting (implementação básica)"""
        # Em produção, usar Redis ou similar
        # Por agora, apenas log
        self.audit_logger.info(f"Rate limit check: {identifier}")
        return True
    
    def get_security_headers(self) -> Dict[str, str]:
        """Retorna headers de segurança para HTTP"""
        return {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }


# Instância global de segurança
_global_security_manager = None

def get_security_manager() -> SecurityManager:
    """Obtém instância global do gerenciador de segurança"""
    global _global_security_manager
    if _global_security_manager is None:
        _global_security_manager = SecurityManager()
    return _global_security_manager


def initialize_security(policy: SecurityPolicy = None) -> SecurityManager:
    """Inicializa sistema de segurança"""
    global _global_security_manager
    _global_security_manager = SecurityManager(policy)
    return _global_security_manager


# Decorators de segurança
def require_input_validation(func):
    """Decorator para validação obrigatória de input"""
    def wrapper(*args, **kwargs):
        security = get_security_manager()
        
        # Validar argumentos
        for arg in args:
            if isinstance(arg, str):
                security.validator.validate_string_input(arg)
        
        for key, value in kwargs.items():
            if isinstance(value, str):
                security.validator.validate_string_input(value)
            elif isinstance(value, dict):
                security.validate_api_input(value, func.__name__)
        
        return func(*args, **kwargs)
    return wrapper


def require_safe_file_access(func):
    """Decorator para operações seguras de arquivo"""
    def wrapper(file_path, *args, **kwargs):
        security = get_security_manager()
        security.secure_file_operation(file_path, "read")
        return func(file_path, *args, **kwargs)
    return wrapper