# api_security.py - Sistema de Segurança para APIs
"""
Sistema de validação e segurança para todas as APIs do Team Agents,
implementando validação rigorosa de inputs e proteção contra ataques.
"""

from functools import wraps
from typing import Dict, Any, List, Optional, Callable, Union
import re
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from flask import request, jsonify, abort, g
import logging

from security_manager import get_security_manager, SecurityManager
from shared.exceptions import ValidationError, SecurityError, TeamAgentsException
from shared.enums import SeverityLevel


@dataclass
class APIEndpointConfig:
    """Configuração de segurança para endpoint de API"""
    endpoint: str
    methods: List[str] = field(default_factory=lambda: ['GET', 'POST'])
    require_auth: bool = True
    rate_limit: int = 100  # requests per hour
    max_payload_size: int = 1024 * 1024  # 1MB
    allowed_content_types: List[str] = field(default_factory=lambda: ['application/json'])
    input_validation: Dict[str, Any] = field(default_factory=dict)
    security_level: str = "medium"  # low, medium, high, critical
    
    def __post_init__(self):
        if not self.input_validation:
            self.input_validation = {
                'max_string_length': 10000,
                'max_array_length': 1000,
                'max_object_depth': 10,
                'allowed_html': False,
                'sanitize_inputs': True
            }


class RateLimiter:
    """Sistema de rate limiting simples"""
    
    def __init__(self):
        self.requests = {}  # client_id -> list of timestamps
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = datetime.now()
    
    def is_allowed(self, client_id: str, limit: int, window_seconds: int = 3600) -> bool:
        """Verifica se requisição é permitida"""
        now = datetime.now()
        
        # Cleanup periódico
        if (now - self.last_cleanup).total_seconds() > self.cleanup_interval:
            self._cleanup_old_requests(now, window_seconds)
            self.last_cleanup = now
        
        # Obter requisições do cliente
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        client_requests = self.requests[client_id]
        
        # Remover requisições antigas
        cutoff_time = now - timedelta(seconds=window_seconds)
        client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]
        
        # Verificar limite
        if len(client_requests) >= limit:
            return False
        
        # Adicionar nova requisição
        client_requests.append(now)
        return True
    
    def _cleanup_old_requests(self, now: datetime, window_seconds: int):
        """Remove requisições antigas de todos os clientes"""
        cutoff_time = now - timedelta(seconds=window_seconds)
        
        for client_id in list(self.requests.keys()):
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] 
                if req_time > cutoff_time
            ]
            
            # Remover clientes sem requisições recentes
            if not self.requests[client_id]:
                del self.requests[client_id]


class InputValidator:
    """Validador avançado de inputs de API"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security = security_manager
        self.logger = logging.getLogger("api_security.validator")
    
    def validate_request_data(self, data: Any, config: APIEndpointConfig) -> Dict[str, Any]:
        """Valida dados da requisição"""
        if data is None:
            return {}
        
        validation_config = config.input_validation
        
        # Validar tipo principal
        if isinstance(data, dict):
            return self._validate_object(data, validation_config, depth=0)
        elif isinstance(data, list):
            return self._validate_array(data, validation_config)
        elif isinstance(data, str):
            return self._validate_string(data, validation_config)
        else:
            return data
    
    def _validate_object(self, obj: Dict[str, Any], config: Dict[str, Any], depth: int = 0) -> Dict[str, Any]:
        """Valida objeto/dicionário"""
        max_depth = config.get('max_object_depth', 10)
        if depth > max_depth:
            raise ValidationError(f"Objeto muito profundo: profundidade > {max_depth}")
        
        validated = {}
        
        for key, value in obj.items():
            # Validar chave
            if not isinstance(key, str):
                raise ValidationError("Chaves do objeto devem ser strings")
            
            if len(key) > 100:
                raise ValidationError("Nome da chave muito longo")
            
            self.security.validator.validate_string_input(key, max_length=100)
            
            # Validar valor recursivamente
            if isinstance(value, dict):
                validated[key] = self._validate_object(value, config, depth + 1)
            elif isinstance(value, list):
                validated[key] = self._validate_array(value, config)
            elif isinstance(value, str):
                validated[key] = self._validate_string(value, config)
            else:
                validated[key] = value
        
        return validated
    
    def _validate_array(self, arr: List[Any], config: Dict[str, Any]) -> List[Any]:
        """Valida array/lista"""
        max_length = config.get('max_array_length', 1000)
        if len(arr) > max_length:
            raise ValidationError(f"Array muito longo: {len(arr)} > {max_length}")
        
        validated = []
        
        for item in arr:
            if isinstance(item, dict):
                validated.append(self._validate_object(item, config))
            elif isinstance(item, list):
                validated.append(self._validate_array(item, config))
            elif isinstance(item, str):
                validated.append(self._validate_string(item, config))
            else:
                validated.append(item)
        
        return validated
    
    def _validate_string(self, text: str, config: Dict[str, Any]) -> str:
        """Valida string"""
        max_length = config.get('max_string_length', 10000)
        allow_html = config.get('allowed_html', False)
        sanitize = config.get('sanitize_inputs', True)
        
        # Validações básicas
        self.security.validator.validate_string_input(text, max_length)
        
        # Verificar HTML se não permitido
        if not allow_html and re.search(r'<[^>]*>', text):
            raise ValidationError("HTML não permitido neste campo")
        
        # Sanitizar se necessário
        if sanitize:
            text = self.security.validator.sanitize_input(text)
        
        return text


class APISecurityManager:
    """Gerenciador principal de segurança para APIs"""
    
    def __init__(self):
        self.security = get_security_manager()
        self.rate_limiter = RateLimiter()
        self.validator = InputValidator(self.security)
        self.logger = logging.getLogger("api_security")
        
        # Configurações de endpoints
        self.endpoint_configs: Dict[str, APIEndpointConfig] = {}
        
        # Registrar endpoints padrão
        self._register_default_endpoints()
    
    def _register_default_endpoints(self):
        """Registra configurações padrão dos endpoints"""
        
        # Endpoints públicos (baixa segurança)
        self.register_endpoint(APIEndpointConfig(
            endpoint="/api/health",
            methods=["GET"],
            require_auth=False,
            rate_limit=1000,
            security_level="low"
        ))
        
        self.register_endpoint(APIEndpointConfig(
            endpoint="/api/info",
            methods=["GET"],
            require_auth=False,
            rate_limit=500,
            security_level="low"
        ))
        
        # Endpoints de agentes (alta segurança)
        self.register_endpoint(APIEndpointConfig(
            endpoint="/api/agents/execute",
            methods=["POST"],
            require_auth=True,
            rate_limit=50,
            max_payload_size=2 * 1024 * 1024,  # 2MB
            security_level="high",
            input_validation={
                'max_string_length': 50000,
                'max_array_length': 100,
                'max_object_depth': 5,
                'allowed_html': False,
                'sanitize_inputs': True
            }
        ))
        
        self.register_endpoint(APIEndpointConfig(
            endpoint="/api/agents/list",
            methods=["GET"],
            require_auth=True,
            rate_limit=200,
            security_level="medium"
        ))
        
        # Endpoints de dados (segurança crítica)
        self.register_endpoint(APIEndpointConfig(
            endpoint="/api/data/upload",
            methods=["POST"],
            require_auth=True,
            rate_limit=10,
            max_payload_size=10 * 1024 * 1024,  # 10MB
            security_level="critical",
            allowed_content_types=["application/json", "multipart/form-data"]
        ))
        
        # Endpoints administrativos (segurança crítica)
        self.register_endpoint(APIEndpointConfig(
            endpoint="/api/admin/*",
            methods=["GET", "POST", "PUT", "DELETE"],
            require_auth=True,
            rate_limit=20,
            security_level="critical"
        ))
    
    def register_endpoint(self, config: APIEndpointConfig):
        """Registra configuração de endpoint"""
        self.endpoint_configs[config.endpoint] = config
        self.logger.info(f"Endpoint registrado: {config.endpoint} (segurança: {config.security_level})")
    
    def get_endpoint_config(self, endpoint: str) -> Optional[APIEndpointConfig]:
        """Obtém configuração do endpoint"""
        # Busca exata primeiro
        if endpoint in self.endpoint_configs:
            return self.endpoint_configs[endpoint]
        
        # Busca por padrão (wildcard)
        for pattern, config in self.endpoint_configs.items():
            if pattern.endswith("*"):
                base_pattern = pattern[:-1]
                if endpoint.startswith(base_pattern):
                    return config
        
        return None
    
    def get_client_id(self, request) -> str:
        """Obtém identificador único do cliente"""
        # Usar header X-Client-ID se disponível
        client_id = request.headers.get('X-Client-ID')
        if client_id:
            return client_id
        
        # Usar IP + User-Agent como fallback
        ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        user_agent = request.headers.get('User-Agent', 'unknown')
        
        return f"{ip}:{hash(user_agent) % 10000}"
    
    def validate_request(self, endpoint: str, method: str) -> Dict[str, Any]:
        """Valida requisição completa"""
        config = self.get_endpoint_config(endpoint)
        if not config:
            # Configuração padrão para endpoints não registrados
            config = APIEndpointConfig(
                endpoint=endpoint,
                methods=[method],
                require_auth=True,
                security_level="high"
            )
        
        validation_result = {
            'allowed': True,
            'config': config,
            'client_id': None,
            'validated_data': None,
            'errors': []
        }
        
        try:
            # 1. Verificar método HTTP
            if method not in config.methods:
                raise SecurityError(f"Método {method} não permitido para {endpoint}")
            
            # 2. Verificar rate limiting
            client_id = self.get_client_id(request)
            validation_result['client_id'] = client_id
            
            if not self.rate_limiter.is_allowed(client_id, config.rate_limit):
                raise SecurityError(f"Rate limit excedido para cliente {client_id}")
            
            # 3. Verificar tamanho do payload
            if hasattr(request, 'content_length') and request.content_length:
                if request.content_length > config.max_payload_size:
                    raise SecurityError(f"Payload muito grande: {request.content_length} > {config.max_payload_size}")
            
            # 4. Verificar Content-Type
            if method in ['POST', 'PUT', 'PATCH']:
                content_type = request.content_type or 'application/json'
                if not any(ct in content_type for ct in config.allowed_content_types):
                    raise SecurityError(f"Content-Type não permitido: {content_type}")
            
            # 5. Validar dados da requisição
            request_data = None
            if method in ['POST', 'PUT', 'PATCH'] and request.is_json:
                try:
                    request_data = request.get_json()
                    validation_result['validated_data'] = self.validator.validate_request_data(request_data, config)
                except Exception as e:
                    raise ValidationError(f"Dados JSON inválidos: {str(e)}")
            
            # 6. Log da requisição
            self.logger.info(
                f"Requisição validada: {method} {endpoint}",
                extra={
                    'client_id': client_id,
                    'security_level': config.security_level,
                    'payload_size': request.content_length or 0
                }
            )
            
        except (SecurityError, ValidationError) as e:
            validation_result['allowed'] = False
            validation_result['errors'].append(str(e))
            
            # Audit log para tentativas suspeitas
            self.security.audit_logger.warning(
                f"Requisição bloqueada: {method} {endpoint} - {str(e)}",
                extra={
                    'client_id': client_id,
                    'endpoint': endpoint,
                    'error_type': type(e).__name__
                }
            )
        
        except Exception as e:
            validation_result['allowed'] = False
            validation_result['errors'].append(f"Erro interno: {str(e)}")
            self.logger.error(f"Erro na validação: {str(e)}")
        
        return validation_result


# Instância global
_global_api_security = None

def get_api_security() -> APISecurityManager:
    """Obtém instância global do gerenciador de segurança de API"""
    global _global_api_security
    if _global_api_security is None:
        _global_api_security = APISecurityManager()
    return _global_api_security


# Decorators para Flask
def secure_endpoint(endpoint_path: str = None, **kwargs):
    """Decorator para proteger endpoint Flask"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs_inner):
            api_security = get_api_security()
            
            # Usar path do endpoint ou deduzir da função
            endpoint = endpoint_path or request.endpoint or func.__name__
            method = request.method
            
            # Validar requisição
            validation = api_security.validate_request(endpoint, method)
            
            if not validation['allowed']:
                # Retornar erro de segurança
                return jsonify({
                    'error': 'Security validation failed',
                    'details': validation['errors'],
                    'timestamp': datetime.now().isoformat()
                }), 403
            
            # Adicionar dados validados ao contexto global
            g.validated_data = validation['validated_data']
            g.client_id = validation['client_id']
            g.security_config = validation['config']
            
            # Executar função original
            try:
                result = func(*args, **kwargs_inner)
                
                # Log de sucesso
                api_security.logger.info(
                    f"Endpoint executado com sucesso: {method} {endpoint}",
                    extra={'client_id': g.client_id}
                )
                
                return result
                
            except Exception as e:
                # Log de erro
                api_security.logger.error(
                    f"Erro na execução do endpoint: {method} {endpoint} - {str(e)}",
                    extra={'client_id': g.client_id}
                )
                raise
        
        return wrapper
    return decorator


def require_auth(func):
    """Decorator para requerer autenticação"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Verificar header de autorização
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Token de autenticação necessário'}), 401
        
        token = auth_header.split(' ')[1]
        
        # Validar token (implementação básica)
        if not _validate_auth_token(token):
            return jsonify({'error': 'Token inválido ou expirado'}), 401
        
        # Adicionar informações do usuário ao contexto
        g.user_id = _get_user_from_token(token)
        
        return func(*args, **kwargs)
    
    return wrapper


def _validate_auth_token(token: str) -> bool:
    """Valida token de autenticação (implementação básica)"""
    # Em produção, usar JWT ou sistema de autenticação adequado
    return len(token) > 10 and token.isalnum()


def _get_user_from_token(token: str) -> str:
    """Obtém ID do usuário do token (implementação básica)"""
    # Em produção, decodificar JWT ou consultar banco
    return f"user_{hash(token) % 1000}"


# Middleware para Flask
def setup_security_middleware(app):
    """Configura middleware de segurança para Flask app"""
    
    @app.before_request
    def security_check():
        """Verificação de segurança antes de cada requisição"""
        # Headers de segurança
        security_headers = get_security_manager().get_security_headers()
        
        # Pular verificação para arquivos estáticos
        if request.endpoint == 'static':
            return
        
        # Log da requisição
        logger = logging.getLogger("api_security.middleware")
        logger.info(
            f"Requisição recebida: {request.method} {request.path}",
            extra={
                'ip': request.environ.get('REMOTE_ADDR'),
                'user_agent': request.headers.get('User-Agent')
            }
        )
    
    @app.after_request
    def add_security_headers(response):
        """Adiciona headers de segurança à resposta"""
        security_headers = get_security_manager().get_security_headers()
        
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    @app.errorhandler(403)
    def handle_forbidden(error):
        """Handler para erros 403"""
        return jsonify({
            'error': 'Acesso negado',
            'message': 'Você não tem permissão para acessar este recurso',
            'timestamp': datetime.now().isoformat()
        }), 403
    
    @app.errorhandler(429)
    def handle_rate_limit(error):
        """Handler para rate limiting"""
        return jsonify({
            'error': 'Rate limit excedido',
            'message': 'Muitas requisições. Tente novamente mais tarde.',
            'timestamp': datetime.now().isoformat()
        }), 429