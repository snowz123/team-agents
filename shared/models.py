# shared/models.py - Modelos de Dados Unificados
"""
Dataclasses centralizadas que consolidam todos os modelos de dados
duplicados em estruturas unificadas para o Team Agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from enum import Enum
import uuid

from .enums import (
    ComplexityLevel, AgentRole, TaskStatus, AgentStatus, 
    SeverityLevel, OperationType, ExecutionMode
)


@dataclass
class ProjectContext:
    """Contexto unificado de projeto"""
    project_id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    owner: str
    collaborators: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    complexity_level: ComplexityLevel = ComplexityLevel.MEDIUM
    status: str = "active"
    deadline: Optional[datetime] = None
    budget: Optional[float] = None
    priority: SeverityLevel = SeverityLevel.MEDIUM
    requirements: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.project_id:
            self.project_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()
        if not self.updated_at:
            self.updated_at = datetime.now()
    
    def add_collaborator(self, user_id: str):
        """Adiciona colaborador ao projeto"""
        if user_id not in self.collaborators:
            self.collaborators.append(user_id)
            self.updated_at = datetime.now()
    
    def add_tag(self, tag: str):
        """Adiciona tag ao projeto"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.now()
    
    def update_metadata(self, key: str, value: Any):
        """Atualiza metadados do projeto"""
        self.metadata[key] = value
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável"""
        return {
            'project_id': self.project_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'owner': self.owner,
            'collaborators': self.collaborators,
            'tags': self.tags,
            'metadata': self.metadata,
            'complexity_level': self.complexity_level.value,
            'status': self.status,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'budget': self.budget,
            'priority': self.priority.value,
            'requirements': self.requirements,
            'deliverables': self.deliverables,
            'resources': self.resources,
            'constraints': self.constraints
        }


@dataclass 
class TaskRequest:
    """Solicitação unificada de tarefa"""
    task_id: str
    title: str
    description: str
    requested_by: str
    created_at: datetime
    complexity: ComplexityLevel
    priority: SeverityLevel
    operation_type: OperationType
    execution_mode: ExecutionMode = ExecutionMode.SYNCHRONOUS
    agent_requirements: List[AgentRole] = field(default_factory=list)
    input_data: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    expected_output: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None
    timeout_seconds: Optional[int] = None
    retry_attempts: int = 3
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    project_context: Optional[ProjectContext] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.now()
    
    def add_dependency(self, task_id: str):
        """Adiciona dependência de tarefa"""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
    
    def set_parameter(self, key: str, value: Any):
        """Define parâmetro da tarefa"""
        self.parameters[key] = value
    
    def set_constraint(self, key: str, value: Any):
        """Define restrição da tarefa"""
        self.constraints[key] = value
    
    def add_agent_requirement(self, role: AgentRole):
        """Adiciona requisito de agente"""
        if role not in self.agent_requirements:
            self.agent_requirements.append(role)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável"""
        return {
            'task_id': self.task_id,
            'title': self.title,
            'description': self.description,
            'requested_by': self.requested_by,
            'created_at': self.created_at.isoformat(),
            'complexity': self.complexity.value,
            'priority': self.priority.value,
            'operation_type': self.operation_type.value,
            'execution_mode': self.execution_mode.value,
            'agent_requirements': [role.value for role in self.agent_requirements],
            'input_data': self.input_data,
            'parameters': self.parameters,
            'constraints': self.constraints,
            'expected_output': self.expected_output,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'timeout_seconds': self.timeout_seconds,
            'retry_attempts': self.retry_attempts,
            'dependencies': self.dependencies,
            'tags': self.tags,
            'metadata': self.metadata,
            'project_context': self.project_context.to_dict() if self.project_context else None
        }


@dataclass
class ExecutionResult:
    """Resultado unificado de execução"""
    execution_id: str
    task_id: str
    agent_id: Optional[str]
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]
    success: bool
    result_data: Dict[str, Any] = field(default_factory=dict)
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    retry_count: int = 0
    partial_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())
        if not self.started_at:
            self.started_at = datetime.now()
        if self.completed_at and self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def mark_completed(self, success: bool = True):
        """Marca execução como completa"""
        self.completed_at = datetime.now()
        self.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        self.success = success
        if self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()
    
    def add_log(self, message: str, level: str = "info"):
        """Adiciona entrada de log"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] [{level.upper()}] {message}"
        self.logs.append(log_entry)
    
    def add_error(self, error: str):
        """Adiciona erro"""
        self.errors.append(error)
        self.add_log(error, "error")
    
    def add_warning(self, warning: str):
        """Adiciona aviso"""
        self.warnings.append(warning)
        self.add_log(warning, "warning")
    
    def set_metric(self, name: str, value: float):
        """Define métrica"""
        self.metrics[name] = value
    
    def add_output_file(self, file_path: str):
        """Adiciona arquivo de saída"""
        if file_path not in self.output_files:
            self.output_files.append(file_path)
    
    def set_resource_usage(self, resource: str, usage: Any):
        """Define uso de recurso"""
        self.resource_usage[resource] = usage
    
    def add_partial_result(self, result: Dict[str, Any]):
        """Adiciona resultado parcial"""
        result['timestamp'] = datetime.now().isoformat()
        self.partial_results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável"""
        return {
            'execution_id': self.execution_id,
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'success': self.success,
            'result_data': self.result_data,
            'output_files': self.output_files,
            'metrics': self.metrics,
            'logs': self.logs,
            'errors': self.errors,
            'warnings': self.warnings,
            'metadata': self.metadata,
            'resource_usage': self.resource_usage,
            'quality_score': self.quality_score,
            'confidence_score': self.confidence_score,
            'retry_count': self.retry_count,
            'partial_results': self.partial_results
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna resumo da execução"""
        return {
            'execution_id': self.execution_id,
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'status': self.status.value,
            'success': self.success,
            'duration_seconds': self.duration_seconds,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'quality_score': self.quality_score,
            'confidence_score': self.confidence_score
        }


@dataclass
class AgentInfo:
    """Informações unificadas de agente"""
    agent_id: str
    name: str
    role: AgentRole
    status: AgentStatus
    capabilities: List[str] = field(default_factory=list)
    specialties: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    description: str = ""
    max_concurrent_tasks: int = 1
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks_count: int = 0
    failed_tasks_count: int = 0
    average_execution_time: Optional[float] = None
    success_rate: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = str(uuid.uuid4())
    
    def add_capability(self, capability: str):
        """Adiciona capacidade"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def add_specialty(self, specialty: str):
        """Adiciona especialidade"""
        if specialty not in self.specialties:
            self.specialties.append(specialty)
    
    def update_status(self, status: AgentStatus):
        """Atualiza status do agente"""
        self.status = status
        self.last_active = datetime.now()
    
    def add_current_task(self, task_id: str):
        """Adiciona tarefa atual"""
        if task_id not in self.current_tasks and len(self.current_tasks) < self.max_concurrent_tasks:
            self.current_tasks.append(task_id)
            return True
        return False
    
    def complete_task(self, task_id: str, success: bool = True):
        """Completa tarefa"""
        if task_id in self.current_tasks:
            self.current_tasks.remove(task_id)
        
        if success:
            self.completed_tasks_count += 1
        else:
            self.failed_tasks_count += 1
        
        # Recalcular taxa de sucesso
        total_tasks = self.completed_tasks_count + self.failed_tasks_count
        if total_tasks > 0:
            self.success_rate = self.completed_tasks_count / total_tasks
        
        self.last_active = datetime.now()
    
    def is_available(self) -> bool:
        """Verifica se agente está disponível"""
        return (self.status == AgentStatus.IDLE and 
                len(self.current_tasks) < self.max_concurrent_tasks)
    
    def can_handle_complexity(self, complexity: ComplexityLevel) -> bool:
        """Verifica se agente pode lidar com complexidade"""
        # Lógica simples baseada na taxa de sucesso e especialidades
        if complexity == ComplexityLevel.SIMPLE:
            return True
        elif complexity == ComplexityLevel.MEDIUM:
            return self.success_rate >= 0.7
        elif complexity == ComplexityLevel.COMPLEX:
            return self.success_rate >= 0.8 and len(self.specialties) >= 2
        elif complexity == ComplexityLevel.EXPERT:
            return self.success_rate >= 0.9 and len(self.specialties) >= 3
        else:
            return self.success_rate >= 0.95 and len(self.specialties) >= 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'role': self.role.value,
            'status': self.status.value,
            'capabilities': self.capabilities,
            'specialties': self.specialties,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'version': self.version,
            'description': self.description,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'current_tasks': self.current_tasks,
            'completed_tasks_count': self.completed_tasks_count,
            'failed_tasks_count': self.failed_tasks_count,
            'average_execution_time': self.average_execution_time,
            'success_rate': self.success_rate,
            'metadata': self.metadata,
            'configuration': self.configuration
        }


# Factory functions para criação de modelos
def create_project_context(name: str, description: str, owner: str,
                          complexity: ComplexityLevel = ComplexityLevel.MEDIUM) -> ProjectContext:
    """Cria contexto de projeto com valores padrão"""
    return ProjectContext(
        project_id=str(uuid.uuid4()),
        name=name,
        description=description,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        owner=owner,
        complexity_level=complexity
    )


def create_task_request(title: str, description: str, requested_by: str,
                       complexity: ComplexityLevel = ComplexityLevel.MEDIUM,
                       priority: SeverityLevel = SeverityLevel.MEDIUM,
                       operation_type: OperationType = OperationType.EXECUTE) -> TaskRequest:
    """Cria solicitação de tarefa com valores padrão"""
    return TaskRequest(
        task_id=str(uuid.uuid4()),
        title=title,
        description=description,
        requested_by=requested_by,
        created_at=datetime.now(),
        complexity=complexity,
        priority=priority,
        operation_type=operation_type
    )


def create_execution_result(task_id: str, agent_id: str = None) -> ExecutionResult:
    """Cria resultado de execução com valores padrão"""
    return ExecutionResult(
        execution_id=str(uuid.uuid4()),
        task_id=task_id,
        agent_id=agent_id,
        status=TaskStatus.RUNNING,
        started_at=datetime.now(),
        success=False
    )


def create_agent_info(name: str, role: AgentRole, 
                     capabilities: List[str] = None) -> AgentInfo:
    """Cria informações de agente com valores padrão"""
    return AgentInfo(
        agent_id=str(uuid.uuid4()),
        name=name,
        role=role,
        status=AgentStatus.IDLE,
        capabilities=capabilities or [],
        created_at=datetime.now(),
        last_active=datetime.now()
    )