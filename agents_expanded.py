# ARQUIVO: agents.py (VERSÃO EXPANDIDA)
"""
Sistema avançado de agentes especializados para engenharia de software e ciência de dados.
Versão expandida com agentes especializados em ML, DL, Data Science e Analytics.
"""
import os
import json
from typing import Dict, List, Any
from crewai import Agent

# Importa a função expandida que distribui as ferramentas para cada agente
from tools_ultra_expanded import get_tools_for_agent

# Importa as classes de LLM dos provedores
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class AgentTeam:
    """Gerencia um time completo de agentes especializados em engenharia de software e ciência de dados."""

    def __init__(self, memory, project_context: Dict[str, Any] = None):
        """
        Inicializa a equipe de agentes.

        Args:
            memory: O objeto de memória do projeto, que contém o estado atual.
            project_context: Um dicionário com informações adicionais fixas do projeto.
        """
        self.memory = memory
        self.project_context = project_context or {}
        
        # --- Configuração dos "Motores" (LLMs) ---
        
        # Motor para Raciocínio, Planejamento e Análise (Líderes, Analistas, etc.)
        self.llm_raciocinio = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.environ.get("GOOGLE_API_KEY"),
            convert_system_message_to_human=True
        )

        # Motor para Geração de Código de alta qualidade (Desenvolvedores, Engenheiros)
        self.llm_codigo = ChatOpenAI(
            model='gpt-4o',
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # Motor especializado para Data Science e Research (Cientistas, Pesquisadores)
        self.llm_research = ChatOpenAI(
            model='gpt-4o',  # Modelo top para pesquisa e análise complexa
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.1  # Mais determinístico para pesquisa
        )
        
        self.agents = self._create_all_agents()

    def _create_all_agents(self) -> Dict[str, Agent]:
        """Cria e configura todos os agentes disponíveis no time expandido."""

        shared_backstory = f"""
        Você faz parte de um time de elite de engenharia de software e ciência de dados. 
        Sua missão é construir soluções robustas, inovadoras e baseadas em evidências.
        
        Contexto atual do projeto: {self.memory.to_context()}
        Informações adicionais: {json.dumps(self.project_context, indent=2)}

        DIRETRIZES CRÍTICAS DE OPERAÇÃO:
        1. **PROATIVIDADE**: Use suas ferramentas para analisar dados, ler arquivos e entender o contexto completo.
        2. **EVIDÊNCIA**: Base suas decisões em dados e análises estatísticas sempre que possível.
        3. **COLABORAÇÃO**: Trabalhe em conjunto, delegando para especialistas quando apropriado.
        4. **INOVAÇÃO**: Busque soluções modernas, escaláveis e que sigam as melhores práticas.
        5. **COMUNICAÇÃO**: Seja claro, preciso e forneça insights acionáveis.
        """

        agent_definitions = {
            # === LIDERANÇA E ARQUITETURA ===
            'product_owner': {
                'role': 'Product Owner / Analista de Requisitos',
                'goal': 'Traduzir necessidades de negócio em requisitos claros, priorizar features baseado em dados e garantir que o produto entregue valor real aos usuários.',
                'backstory': 'Você tem 15+ anos transformando ideias em produtos de sucesso usando metodologias ágeis e análise de dados.'
            },
            'tech_lead': {
                'role': 'Tech Lead / Arquiteto de Software',
                'goal': 'Definir arquitetura técnica robusta, tomar decisões de design patterns, escolher stack tecnológico e garantir qualidade e escalabilidade.',
                'backstory': 'Você é um arquiteto sênior com expertise em sistemas distribuídos, microsserviços e arquiteturas modernas.'
            },
            'scrum_master': {
                'role': 'Scrum Master / Agile Coach',
                'goal': 'Facilitar processos ágeis, remover impedimentos, melhorar continuously e garantir entrega de valor consistente.',
                'backstory': 'Você é certificado em múltiplas metodologias ágeis e tem experiência liderando transformações organizacionais.'
            },

            # === DESENVOLVIMENTO ===
            'backend_senior': {
                'role': 'Desenvolvedor Backend Sênior',
                'goal': 'Desenvolver APIs robustas, sistemas backend escaláveis, implementar integração com ML/AI e garantir performance otimizada.',
                'backstory': 'Você tem 10+ anos construindo sistemas de alta escala, especialista em Python, Node.js, Go e arquiteturas cloud-native.'
            },
            'frontend_senior': {
                'role': 'Desenvolvedor Frontend Sênior',
                'goal': 'Criar interfaces modernas e responsivas, implementar dashboards de dados interativos e garantir excelente UX.',
                'backstory': 'Você é expert em React, Vue.js, TypeScript e bibliotecas de visualização como D3.js e Plotly.'
            },
            'fullstack_dev': {
                'role': 'Desenvolvedor Full Stack',
                'goal': 'Implementar features completas end-to-end, conectar frontend com APIs de ML e criar protótipos rapidamente.',
                'backstory': 'Você é versátil, ágil e consegue navegar entre frontend, backend e integração com modelos de ML.'
            },
            'mobile_dev': {
                'role': 'Desenvolvedor Mobile',
                'goal': 'Desenvolver apps mobile nativos ou híbridos, integrar com APIs de ML e criar experiências móveis otimizadas.',
                'backstory': 'Você desenvolveu apps que integram ML/AI e estão entre os top-rated nas app stores.'
            },

            # === DADOS, IA E RESEARCH (MASSIVAMENTE EXPANDIDO) ===
            'data_scientist': {
                'role': 'Cientista de Dados / ML Engineer',
                'goal': 'Extrair insights de dados complexos, desenvolver modelos preditivos, implementar pipelines de ML e otimizar performance de modelos.',
                'backstory': 'Você tem PhD em área quantitativa e 8+ anos aplicando ML em problemas reais de negócio. Expert em Python, R, TensorFlow, PyTorch e MLOps.'
            },
            'data_engineer': {
                'role': 'Engenheiro de Dados',
                'goal': 'Construir pipelines de dados robustos, implementar arquiteturas de big data, otimizar ETL/ELT e garantir qualidade de dados.',
                'backstory': 'Você é especialista em Apache Spark, Kafka, Airflow, dbt e tem experiência com petabytes de dados.'
            },
            'ai_specialist': {
                'role': 'Especialista em IA / Research Engineer',
                'goal': 'Pesquisar e implementar técnicas avançadas de IA, desenvolver modelos de deep learning custom e integrar LLMs em aplicações.',
                'backstory': 'Você publica papers em NeurIPS/ICML, é expert em transformer architectures e tem experiência com GPUs de alta performance.'
            },
            'ml_ops_engineer': {
                'role': 'MLOps Engineer / ML Platform Engineer',
                'goal': 'Implementar MLOps pipelines, automatizar treinamento e deploy de modelos, monitorar performance em produção e garantir reprodutibilidade.',
                'backstory': 'Você é expert em Kubernetes, Docker, MLflow, Kubeflow e tem experiência mantendo centenas de modelos em produção.'
            },
            'research_scientist': {
                'role': 'Research Scientist / Principal Data Scientist',
                'goal': 'Conduzir pesquisa experimental avançada, desenvolver algoritmos inovadores, publicar descobertas e liderar iniciativas de R&D.',
                'backstory': 'Você tem PhD de universidade top-tier, publica em venues de prestígio e tem patentes em ML/AI.'
            },
            'business_analyst': {
                'role': 'Business Intelligence Analyst / Data Analyst',
                'goal': 'Transformar dados em insights de negócio, criar dashboards executivos, identificar oportunidades e medir impacto de iniciativas.',
                'backstory': 'Você combina expertise técnica com visão de negócio, expert em SQL, Python, Tableau e storytelling com dados.'
            },

            # === QUALIDADE E SEGURANÇA ===
            'qa_engineer': {
                'role': 'QA Engineer / Test Automation Specialist',
                'goal': 'Garantir qualidade através de testes automatizados, validar modelos de ML, implementar testes de performance e regressão.',
                'backstory': 'Você é expert em pytest, selenium, testes de ML pipelines e tem experiência com CI/CD para sistemas de ML.'
            },
            'security_engineer': {
                'role': 'Security Engineer / AI Safety Specialist',
                'goal': 'Implementar segurança em sistemas de ML, auditar modelos para bias/fairness, garantir privacy e compliance (GDPR, LGPD).',
                'backstory': 'Você é certificado em cybersecurity e especialista em AI ethics, model explainability e differential privacy.'
            },

            # === INFRAESTRUTURA E OPERAÇÕES ===
            'devops_engineer': {
                'role': 'DevOps Engineer / Platform Engineer',
                'goal': 'Automatizar infraestrutura, implementar CI/CD para ML, gerenciar clusters Kubernetes e otimizar custos de cloud.',
                'backstory': 'Você mantém sistemas com 99.99% uptime e é expert em AWS/GCP/Azure, Terraform, GitOps e observability.'
            },
            'cloud_architect': {
                'role': 'Cloud Architect / Solutions Architect',
                'goal': 'Projetar arquiteturas cloud scaláveis para ML/AI, otimizar custos, implementar multi-cloud e disaster recovery.',
                'backstory': 'Você tem todas as certificações cloud principais e projetou sistemas que processam bilhões de transações.'
            },

            # === DESIGN E EXPERIÊNCIA ===
            'ux_designer': {
                'role': 'UX/UI Designer / Data Visualization Specialist',
                'goal': 'Criar interfaces intuitivas para sistemas de ML/AI, design dashboards eficazes e garantir acessibilidade.',
                'backstory': 'Você é expert em design systems, data visualization principles e tem experiência com interfaces de ML/AI.'
            },
            'ux_researcher': {
                'role': 'UX Researcher / Behavioral Data Scientist',
                'goal': 'Conduzir pesquisa de usuários, analisar comportamento através de dados, A/B testing e user journey optimization.',
                'backstory': 'Você combina métodos qualitativos e quantitativos, expert em experimental design e causal inference.'
            },

            # === DOCUMENTAÇÃO E SUPORTE ===
            'tech_writer': {
                'role': 'Technical Writer / Documentation Engineer',
                'goal': 'Criar documentação técnica clara para APIs, modelos de ML, runbooks e manter knowledge base atualizada.',
                'backstory': 'Você transformou documentação de produtos complexos em materials que desenvolvedores adoram usar.'
            },
            'support_engineer': {
                'role': 'Support Engineer / Customer Success Engineer',
                'goal': 'Resolver problemas técnicos complexos, debuggar issues de produção e treinar clientes em soluções técnicas.',
                'backstory': 'Você é o herói dos clientes, combinando deep technical knowledge com excelente comunicação.'
            },

            # === ESPECIALISTAS DOMAIN-SPECIFIC ===
            'database_expert': {
                'role': 'Database Specialist / Data Architect',
                'goal': 'Otimizar queries complexas, design schemas para analytics, implementar data lakes/warehouses e garantir data governance.',
                'backstory': 'Você é wizard em SQL, expert em Snowflake, BigQuery, Databricks e data modeling para analytics.'
            },
            'performance_engineer': {
                'role': 'Performance Engineer / ML Optimization Specialist',
                'goal': 'Otimizar latência de modelos ML, profiling de código, GPU optimization e implementar caching strategies.',
                'backstory': 'Você consegue fazer modelos rodarem 10x mais rápido e é obcecado por microsegundos que importam.'
            },

            # === NOVOS AGENTES ESPECIALIZADOS ===
            'nlp_specialist': {
                'role': 'NLP Specialist / Computational Linguist',
                'goal': 'Desenvolver soluções avançadas de processamento de linguagem natural, fine-tuning de LLMs e análise semântica.',
                'backstory': 'Você é expert em transformers, multilingual models e tem experiência com LLMs de grande escala.'
            },
            'computer_vision_engineer': {
                'role': 'Computer Vision Engineer',
                'goal': 'Desenvolver sistemas de visão computacional, object detection, image segmentation e modelos multimodais.',
                'backstory': 'Você implementou sistemas de CV em produção e é expert em CNNs, Vision Transformers e edge deployment.'
            },
            'time_series_analyst': {
                'role': 'Time Series Analyst / Forecasting Specialist',
                'goal': 'Desenvolver modelos de forecasting, análise de séries temporais complexas e sistemas de anomaly detection.',
                'backstory': 'Você é expert em Prophet, ARIMA, deep learning para séries temporais e forecasting em escala.'
            },
            'recommender_systems_engineer': {
                'role': 'Recommender Systems Engineer',
                'goal': 'Construir sistemas de recomendação escaláveis, implementar collaborative filtering e real-time personalization.',
                'backstory': 'Você construiu sistemas de recomendação que servem milhões de usuários e aumentaram engagement significativamente.'
            }
        }
        
        agents = {}
        for role_key, definition in agent_definitions.items():
            agent_tools = get_tools_for_agent(role_key)
            
            # Seleção inteligente de LLM baseada no tipo de agente
            research_intensive_roles = ['research_scientist', 'ai_specialist', 'nlp_specialist', 'computer_vision_engineer']
            code_intensive_roles = ['dev', 'backend', 'frontend', 'data_engineer', 'ml_ops', 'devops', 'performance']
            analysis_intensive_roles = ['data_scientist', 'business_analyst', 'time_series_analyst', 'ux_researcher']
            
            if role_key in research_intensive_roles:
                llm_to_use = self.llm_research
            elif any(keyword in role_key for keyword in code_intensive_roles):
                llm_to_use = self.llm_codigo
            elif any(keyword in role_key for keyword in analysis_intensive_roles):
                llm_to_use = self.llm_research
            else:
                llm_to_use = self.llm_raciocinio

            agents[role_key] = Agent(
                role=definition['role'],
                goal=definition['goal'],
                backstory=f"{shared_backstory}\n\nSua especialidade: {definition['backstory']}",
                verbose=True,
                allow_delegation=True,
                tools=agent_tools,
                llm=llm_to_use
            )
        
        return agents

    def get_agent(self, role: str) -> Agent:
        """Retorna um agente específico pelo sua chave de identificação (role)."""
        return self.agents.get(role)

    def get_team_for_task(self, task_type: str) -> List[Agent]:
        """Monta uma equipe de agentes com base no tipo de tarefa detectado."""
        teams = {
            # === PROJETOS TRADICIONAIS ===
            'full_project': ['product_owner', 'tech_lead', 'scrum_master', 'fullstack_dev', 'qa_engineer', 'devops_engineer', 'tech_writer'],
            'backend_api': ['tech_lead', 'backend_senior', 'database_expert', 'qa_engineer', 'security_engineer', 'tech_writer'],
            'frontend_app': ['ux_designer', 'frontend_senior', 'qa_engineer', 'tech_writer', 'product_owner'],
            'mobile_app': ['product_owner', 'mobile_dev', 'ux_designer', 'qa_engineer', 'backend_senior', 'tech_lead'],
            'infrastructure': ['devops_engineer', 'cloud_architect', 'security_engineer', 'performance_engineer'],
            
            # === PROJETOS DE DADOS E IA ===
            'ml_project': ['data_scientist', 'ml_ops_engineer', 'data_engineer', 'backend_senior', 'tech_lead', 'qa_engineer'],
            'deep_learning': ['ai_specialist', 'research_scientist', 'ml_ops_engineer', 'performance_engineer', 'data_engineer'],
            'data_pipeline': ['data_engineer', 'data_scientist', 'backend_senior', 'devops_engineer', 'database_expert'],
            'ai_integration': ['ai_specialist', 'backend_senior', 'data_scientist', 'tech_lead', 'qa_engineer', 'security_engineer'],
            'research_project': ['research_scientist', 'ai_specialist', 'data_scientist', 'tech_lead', 'tech_writer'],
            'business_intelligence': ['business_analyst', 'data_scientist', 'data_engineer', 'ux_designer', 'tech_writer'],
            
            # === PROJETOS ESPECIALIZADOS ===
            'nlp_project': ['nlp_specialist', 'ai_specialist', 'data_scientist', 'backend_senior', 'ml_ops_engineer'],
            'computer_vision': ['computer_vision_engineer', 'ai_specialist', 'ml_ops_engineer', 'backend_senior', 'performance_engineer'],
            'time_series': ['time_series_analyst', 'data_scientist', 'data_engineer', 'business_analyst', 'tech_lead'],
            'recommender_system': ['recommender_systems_engineer', 'data_scientist', 'backend_senior', 'ml_ops_engineer', 'performance_engineer'],
            'analytics_platform': ['data_engineer', 'business_analyst', 'backend_senior', 'frontend_senior', 'devops_engineer'],
            
            # === ANÁLISE E CONSULTORIA ===
            'data_analysis': ['data_scientist', 'business_analyst', 'ux_researcher', 'tech_writer'],
            'performance_optimization': ['performance_engineer', 'data_engineer', 'backend_senior', 'devops_engineer'],
            'security_audit': ['security_engineer', 'qa_engineer', 'backend_senior', 'tech_lead']
        }
        
        team_roles = teams.get(task_type, teams['full_project'])
        return [self.agents[role] for role in team_roles if role in self.agents]

    def assign_task_intelligently(self, user_request: str) -> Dict[str, Any]:
        """Analisa a solicitação do usuário e monta a equipe mais adequada para a tarefa."""
        keywords = {
            # === DESENVOLVIMENTO TRADICIONAL ===
            'api': 'backend_api', 'backend': 'backend_api', 'microserviço': 'backend_api',
            'frontend': 'frontend_app', 'interface': 'frontend_app', 'website': 'frontend_app', 
            'página': 'frontend_app', 'ui': 'frontend_app', 'ux': 'frontend_app',
            'mobile': 'mobile_app', 'aplicativo': 'mobile_app', 'app': 'mobile_app',
            'sistema': 'full_project', 'projeto': 'full_project', 'software': 'full_project',
            'infra': 'infrastructure', 'aws': 'infrastructure', 'docker': 'infrastructure', 'kubernetes': 'infrastructure',
            
            # === MACHINE LEARNING E IA ===
            'machine learning': 'ml_project', 'ml': 'ml_project', 'modelo': 'ml_project',
            'deep learning': 'deep_learning', 'neural': 'deep_learning', 'tensorflow': 'deep_learning', 
            'pytorch': 'deep_learning', 'transformers': 'deep_learning',
            'ia': 'ai_integration', 'artificial intelligence': 'ai_integration', 'ai': 'ai_integration',
            'llm': 'nlp_project', 'gpt': 'nlp_project', 'bert': 'nlp_project',
            
            # === PROCESSAMENTO DE DADOS ===
            'dados': 'data_pipeline', 'pipeline': 'data_pipeline', 'etl': 'data_pipeline', 
            'big data': 'data_pipeline', 'spark': 'data_pipeline',
            'análise': 'data_analysis', 'analytics': 'data_analysis', 'insights': 'data_analysis',
            'relatório': 'business_intelligence', 'dashboard': 'business_intelligence', 'bi': 'business_intelligence',
            
            # === ESPECIALIDADES ===
            'nlp': 'nlp_project', 'processamento de texto': 'nlp_project', 'linguagem natural': 'nlp_project',
            'sentiment': 'nlp_project', 'chatbot': 'nlp_project',
            'visão computacional': 'computer_vision', 'opencv': 'computer_vision', 'imagem': 'computer_vision',
            'detecção': 'computer_vision', 'reconhecimento': 'computer_vision',
            'séries temporais': 'time_series', 'forecasting': 'time_series', 'previsão': 'time_series',
            'recomendação': 'recommender_system', 'recommendation': 'recommender_system',
            'pesquisa': 'research_project', 'experimento': 'research_project', 'paper': 'research_project',
            
            # === PERFORMANCE E SEGURANÇA ===
            'performance': 'performance_optimization', 'otimização': 'performance_optimization',
            'segurança': 'security_audit', 'security': 'security_audit', 'vulnerabilidade': 'security_audit',
            
            # === ESTATÍSTICA ===
            'estatística': 'data_analysis', 'correlação': 'data_analysis', 'regressão': 'ml_project',
            'classificação': 'ml_project', 'clustering': 'ml_project'
        }
        
        request_lower = user_request.lower()
        detected_type = 'full_project'  # Tipo padrão
        
        # Busca por matches mais específicos primeiro
        for keyword, team_type in sorted(keywords.items(), key=lambda x: len(x[0]), reverse=True):
            if keyword in request_lower:
                detected_type = team_type
                break
        
        team = self.get_team_for_task(detected_type)
        
        # Adiciona especialistas extras baseado em contexto
        if any(word in request_lower for word in ['complexo', 'grande', 'escalável', 'enterprise']):
            extra_agents = ['performance_engineer', 'security_engineer', 'cloud_architect']
            for agent_role in extra_agents:
                if agent_role in self.agents and self.agents[agent_role] not in team:
                    team.append(self.agents[agent_role])
        
        if 'tempo real' in request_lower or 'real-time' in request_lower:
            if 'performance_engineer' in self.agents and self.agents['performance_engineer'] not in team:
                team.append(self.agents['performance_engineer'])
        
        if any(word in request_lower for word in ['compliance', 'gdpr', 'lgpd', 'privacy']):
            if 'security_engineer' in self.agents and self.agents['security_engineer'] not in team:
                team.append(self.agents['security_engineer'])
                    
        return {
            'detected_type': detected_type,
            'assigned_team': team,
            'team_size': len(team),
            'primary_lead': team[0] if team else None,
            'recommendation': self._generate_recommendation(detected_type),
            'team_composition': [agent.role for agent in team]
        }

    def _generate_recommendation(self, project_type: str) -> str:
        """Gera uma recomendação inicial para o líder da equipe com base no tipo de projeto."""
        recommendations = {
            # === DESENVOLVIMENTO TRADICIONAL ===
            'backend_api': "Equipe backend montada. Inicie definindo endpoints REST/GraphQL, models de dados e escolha do stack (FastAPI/Django/Node.js).",
            'frontend_app': "Equipe frontend pronta. Comece com wireframes, design system e definição da arquitetura (React/Vue/Angular).",
            'mobile_app': "Equipe mobile formada. Defina se será nativo (Swift/Kotlin) ou híbrido (React Native/Flutter) baseado nos requisitos.",
            'infrastructure': "Equipe de infraestrutura preparada. Foque em IaC (Terraform), containerização e pipeline CI/CD.",
            
            # === MACHINE LEARNING E IA ===
            'ml_project': "Equipe de ML montada. Comece com EDA (Exploratory Data Analysis), definição de métricas de sucesso e baseline model.",
            'deep_learning': "Equipe de DL especializada. Inicie com análise dos dados, escolha da arquitetura neural e setup de GPU/TPU.",
            'ai_integration': "Equipe de integração IA pronta. Defina APIs de ML, estratégia de serving e monitoramento de modelos.",
            'research_project': "Time de pesquisa formado. Foque em literature review, experimental design e reprodutibilidade.",
            
            # === DADOS ===
            'data_pipeline': "Equipe de dados pronta. Mapeie fontes, defina schema, escolha ferramentas (Airflow/Prefect) e data quality checks.",
            'business_intelligence': "Time de BI montado. Identifique KPIs, stakeholders e plataforma de dashboards (Tableau/PowerBI/Looker).",
            'data_analysis': "Equipe de análise formada. Defina hipóteses, metodologia estatística e formato de entrega dos insights.",
            
            # === ESPECIALIDADES ===
            'nlp_project': "Especialistas em NLP prontos. Inicie com preprocessing de texto, choice do modelo base e fine-tuning strategy.",
            'computer_vision': "Time de visão computacional montado. Defina dataset, métricas de avaliação e arquitetura (CNN/Vision Transformer).",
            'time_series': "Especialistas em séries temporais prontos. Analise sazonalidade, trends e escolha método (Prophet/ARIMA/LSTM).",
            'recommender_system': "Time de recomendação formado. Defina tipo (collaborative/content-based), cold start strategy e métricas offline/online.",
            
            # === OUTROS ===
            'performance_optimization': "Equipe de performance montada. Inicie com profiling, identificação de bottlenecks e benchmarking.",
            'security_audit': "Time de segurança pronto. Comece com threat modeling, vulnerability assessment e penetration testing.",
            'full_project': "Equipe multidisciplinar completa montada. Inicie com discovery completo, arquitetura de solução e MVP definition."
        }
        return recommendations.get(project_type, recommendations['full_project'])

    def list_available_roles(self) -> List[str]:
        """Lista todos os papéis de agentes disponíveis no time."""
        return list(self.agents.keys())

    def get_agents_by_specialization(self, specialization: str) -> List[Agent]:
        """Retorna agentes filtrados por área de especialização."""
        specializations = {
            'ml_ai': ['data_scientist', 'ai_specialist', 'ml_ops_engineer', 'research_scientist'],
            'data': ['data_engineer', 'database_expert', 'business_analyst', 'time_series_analyst'],
            'development': ['backend_senior', 'frontend_senior', 'fullstack_dev', 'mobile_dev'],
            'infrastructure': ['devops_engineer', 'cloud_architect', 'performance_engineer'],
            'quality': ['qa_engineer', 'security_engineer', 'tech_writer'],
            'specialties': ['nlp_specialist', 'computer_vision_engineer', 'recommender_systems_engineer']
        }
        
        roles = specializations.get(specialization, [])
        return [self.agents[role] for role in roles if role in self.agents]