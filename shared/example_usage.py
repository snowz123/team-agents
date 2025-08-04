# shared/example_usage.py - Exemplos de Uso dos Módulos Compartilhados
"""
Exemplos de como usar os módulos base compartilhados do Team Agents.
Este arquivo demonstra a consolidação das classes duplicadas.
"""

from datetime import datetime, timedelta
from shared import (
    ValidationError, TeamAgentsException, ErrorContext, ErrorSeverity,
    ComplexityLevel, AgentRole, SeverityLevel, TimeoutLevel,
    ProjectContext, TaskRequest, ExecutionResult, AgentInfo,
    LoggingContextManager, get_logging_manager,
    UnifiedConfig, get_config, TimeoutManager
)


def example_exception_handling():
    """Exemplo de uso do sistema unificado de exceções"""
    print("=== Exemplo: Sistema de Exceções Unificado ===")
    
    try:
        # Criar exceção de validação
        raise ValidationError(
            message="E-mail inválido",
            field_name="email",
            field_value="invalid-email",
            validation_rule="email_format",
            suggestions=["Use formato nome@dominio.com", "Verifique caracteres especiais"]
        )
    except ValidationError as e:
        print(f"Erro capturado: {e}")
        print(f"Mensagem para usuário: {e.get_user_message()}")
        print(f"Contexto: {e.get_context_dict()}")
    
    print()


def example_enums_usage():
    """Exemplo de uso dos enums unificados"""
    print("=== Exemplo: Enums Unificados ===")
    
    # Comparação de complexidade
    simple = ComplexityLevel.SIMPLE
    expert = ComplexityLevel.EXPERT
    
    print(f"Simple < Expert: {simple < expert}")
    print(f"Expert numeric value: {expert.numeric_value}")
    
    # Timeout baseado em complexidade
    from shared.enums import complexity_to_timeout
    timeout_level = complexity_to_timeout(expert)
    print(f"Timeout level para Expert: {timeout_level}")
    print(f"Timeout em segundos: {timeout_level.default_seconds}")
    
    print()


def example_models_usage():
    """Exemplo de uso dos modelos de dados unificados"""
    print("=== Exemplo: Modelos de Dados Unificados ===")
    
    # Criar contexto de projeto
    project = ProjectContext(
        project_id="",  # Será gerado automaticamente
        name="Sistema de IA Avançado",
        description="Desenvolvimento de sistema de IA para automação",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        owner="equipe_dev",
        complexity_level=ComplexityLevel.EXPERT,
        priority=SeverityLevel.HIGH
    )
    
    project.add_collaborator("dev_001")
    project.add_tag("ia")
    project.add_tag("automacao")
    
    print(f"Projeto criado: {project.name}")
    print(f"ID: {project.project_id}")
    print(f"Colaboradores: {project.collaborators}")
    print(f"Tags: {project.tags}")
    
    # Criar solicitação de tarefa
    task = TaskRequest(
        task_id="",  # Será gerado automaticamente
        title="Implementar modelo de classificação",
        description="Desenvolver modelo ML para classificação de documentos",
        requested_by="product_manager",
        created_at=datetime.now(),
        complexity=ComplexityLevel.COMPLEX,
        priority=SeverityLevel.HIGH,
        project_context=project
    )
    
    task.add_agent_requirement(AgentRole.DATA_SCIENTIST)
    task.add_agent_requirement(AgentRole.DEVELOPER)
    task.set_parameter("model_type", "random_forest")
    task.set_constraint("max_training_time", 3600)
    
    print(f"\nTarefa criada: {task.title}")
    print(f"ID: {task.task_id}")
    print(f"Agentes necessários: {[role.value for role in task.agent_requirements]}")
    
    # Criar resultado de execução
    result = ExecutionResult(
        execution_id="",
        task_id=task.task_id,
        agent_id="agent_data_scientist_001",
        status=task.status if hasattr(task, 'status') else None,
        started_at=datetime.now(),
        success=False  # Ainda em execução
    )
    
    result.add_log("Iniciando treinamento do modelo", "info")
    result.set_metric("accuracy", 0.85)
    result.set_resource_usage("memory_mb", 512)
    result.add_partial_result({"epoch": 1, "loss": 0.3})
    
    # Completar execução
    result.mark_completed(success=True)
    result.quality_score = 0.92
    result.confidence_score = 0.88
    
    print(f"\nExecução completada: {result.execution_id}")
    print(f"Sucesso: {result.success}")
    print(f"Duração: {result.duration_seconds:.2f}s")
    print(f"Métricas: {result.metrics}")
    
    print()


def example_logging_usage():
    """Exemplo de uso do sistema de logging unificado"""
    print("=== Exemplo: Sistema de Logging Unificado ===")
    
    # Obter gerenciador de logging
    logging_manager = get_logging_manager()
    
    # Usar contexto de logging
    with logging_manager.context(
        component="example_system",
        operation="data_processing",
        user_id="user_123",
        metadata={"batch_size": 100}
    ) as context:
        
        # Logger específico do componente
        logger = logging_manager.get_logger("example_system")
        
        # Logs estruturados
        logging_manager.log_structured(
            level=logging_manager.LogLevel.INFO,
            message="Processamento iniciado",
            component="example_system",
            operation="data_processing",
            batch_id="batch_001"
        )
        
        # Simular operação com performance logging
        start_time = datetime.now()
        time.sleep(0.1)  # Simular trabalho
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logging_manager.log_performance(
            component="example_system",
            operation="data_processing",
            duration_seconds=duration,
            success=True,
            additional_metrics={"records_processed": 1000}
        )
        
        print(f"Context ID: {context.correlation_id}")
        print(f"Operação: {context.operation}")
    
    print()


def example_config_usage():
    """Exemplo de uso do sistema de configuração unificado"""
    print("=== Exemplo: Sistema de Configuração Unificado ===")
    
    # Obter configuração global
    config = get_config()
    
    print(f"Ambiente: {config.environment.value}")
    print(f"Timeout padrão: {config.timeout.default_timeout}s")
    print(f"String de conexão DB: {config.database.connection_string}")
    print(f"Nível de log: {config.logging.level}")
    print(f"Max tarefas concorrentes: {config.performance.max_concurrent_tasks}")
    
    # Usar timeout manager
    timeout_manager = TimeoutManager(config.timeout)
    
    def operacao_lenta():
        import time
        time.sleep(2)
        return "Operação concluída"
    
    try:
        resultado = timeout_manager.execute_with_timeout(
            operacao_lenta,
            timeout_level=TimeoutLevel.SHORT,
            custom_timeout=3
        )
        print(f"Resultado: {resultado}")
    except Exception as e:
        print(f"Erro na operação: {e}")
    
    # Configuração personalizada
    config.set("custom.feature_flag", True)
    custom_value = config.get("custom.feature_flag", False)
    print(f"Feature flag personalizada: {custom_value}")
    
    print()


def example_agent_info_usage():
    """Exemplo de uso das informações de agente"""
    print("=== Exemplo: Informações de Agente ===")
    
    # Criar informações de agente
    agent = AgentInfo(
        agent_id="",
        name="Data Science Specialist",
        role=AgentRole.DATA_SCIENTIST,
        status=agent.AgentStatus.IDLE if hasattr(agent, 'AgentStatus') else "idle",
        capabilities=["machine_learning", "data_analysis", "python", "sql"]
    )
    
    agent.add_specialty("deep_learning")
    agent.add_specialty("nlp")
    agent.add_capability("tensorflow")
    
    print(f"Agente: {agent.name}")
    print(f"ID: {agent.agent_id}")
    print(f"Role: {agent.role.value}")
    print(f"Capacidades: {agent.capabilities}")
    print(f"Especialidades: {agent.specialties}")
    print(f"Disponível: {agent.is_available()}")
    print(f"Pode lidar com complexidade EXPERT: {agent.can_handle_complexity(ComplexityLevel.EXPERT)}")
    
    # Simular execução de tarefa
    task_id = "task_123"
    if agent.add_current_task(task_id):
        print(f"Tarefa {task_id} atribuída ao agente")
        
        # Completar tarefa
        agent.complete_task(task_id, success=True)
        print(f"Taxa de sucesso atual: {agent.success_rate:.2%}")
    
    print()


def main():
    """Função principal com todos os exemplos"""
    print("🤖 EXEMPLOS DE USO - MÓDULOS COMPARTILHADOS TEAM AGENTS")
    print("=" * 60)
    
    example_exception_handling()
    example_enums_usage()
    example_models_usage()
    example_logging_usage()
    example_config_usage()
    example_agent_info_usage()
    
    print("✅ Todos os exemplos executados com sucesso!")
    print("Os módulos compartilhados estão funcionando corretamente.")


if __name__ == "__main__":
    main()