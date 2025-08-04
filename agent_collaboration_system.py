# agent_collaboration_system.py - Sistema Real de Compartilhamento entre Agentes
"""
Sistema avançado de colaboração que permite compartilhamento real de dados,
conhecimento e resultados entre agentes do Team Agents.
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import pickle
import base64
from pathlib import Path

class MessageType(Enum):
    """Tipos de mensagens entre agentes"""
    KNOWLEDGE_SHARE = "knowledge_share"
    REQUEST_HELP = "request_help"
    PROVIDE_DATA = "provide_data"
    TASK_DELEGATION = "task_delegation"
    RESULT_NOTIFICATION = "result_notification"
    LEARNING_UPDATE = "learning_update"
    COLLABORATION_REQUEST = "collaboration_request"

class Priority(Enum):
    """Prioridades de mensagens"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class AgentMessage:
    """Mensagem entre agentes"""
    id: str
    from_agent: str
    to_agent: str
    message_type: MessageType
    content: Dict[str, Any]
    priority: Priority
    timestamp: datetime
    expires_at: Optional[datetime] = None
    requires_response: bool = False
    response_to: Optional[str] = None

@dataclass
class SharedKnowledge:
    """Conhecimento compartilhado entre agentes"""
    id: str
    source_agent: str
    domain: str
    knowledge_type: str
    content: Dict[str, Any]
    confidence_score: float
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    tags: List[str] = None

@dataclass
class CollaborationTask:
    """Tarefa colaborativa entre agentes"""
    id: str
    title: str
    description: str
    initiator_agent: str
    participating_agents: List[str]
    status: str  # pending, in_progress, completed, failed
    created_at: datetime
    deadline: Optional[datetime]
    shared_data: Dict[str, Any]
    results: Dict[str, Any]

class AgentCollaborationSystem:
    """Sistema central de colaboração entre agentes"""
    
    def __init__(self, db_path: str = "agent_collaboration.db"):
        self.db_path = db_path
        self.active_agents = {}  # agent_id -> agent_info
        self.message_handlers = {}  # agent_id -> message_handler_function
        self.collaboration_callbacks = {}  # agent_id -> callback_functions
        self._lock = threading.Lock()
        
        # Inicializar banco de dados
        self._init_database()
        
        # Iniciar thread de processamento
        self._processing_thread = threading.Thread(target=self._process_messages_loop, daemon=True)
        self._processing_thread.start()
    
    def _init_database(self):
        """Inicializa o banco de dados SQLite"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de mensagens
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                from_agent TEXT NOT NULL,
                to_agent TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                priority INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                expires_at TEXT,
                requires_response BOOLEAN NOT NULL,
                response_to TEXT,
                processed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabela de conhecimento compartilhado
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shared_knowledge (
                id TEXT PRIMARY KEY,
                source_agent TEXT NOT NULL,
                domain TEXT NOT NULL,
                knowledge_type TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                tags TEXT
            )
        ''')
        
        # Tabela de tarefas colaborativas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collaboration_tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                initiator_agent TEXT NOT NULL,
                participating_agents TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                deadline TEXT,
                shared_data TEXT,
                results TEXT
            )
        ''')
        
        # Tabela de agentes ativos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_agents (
                agent_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                capabilities TEXT,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def register_agent(self, agent_id: str, agent_name: str, agent_type: str, 
                      capabilities: List[str], message_handler: Callable = None):
        """Registra um agente no sistema de colaboração"""
        with self._lock:
            self.active_agents[agent_id] = {
                'name': agent_name,
                'type': agent_type,
                'capabilities': capabilities,
                'last_seen': datetime.now(),
                'status': 'active'
            }
            
            if message_handler:
                self.message_handlers[agent_id] = message_handler
        
        # Salvar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO active_agents (agent_id, agent_name, agent_type, capabilities, last_seen, status) VALUES (?, ?, ?, ?, ?, ?)',
            (agent_id, agent_name, agent_type, json.dumps(capabilities), datetime.now().isoformat(), 'active')
        )
        conn.commit()
        conn.close()
        
        # Notificar outros agentes sobre novo agente
        self._broadcast_message(
            from_agent=agent_id,
            message_type=MessageType.RESULT_NOTIFICATION,
            content={
                'event': 'agent_joined',
                'agent_info': {
                    'id': agent_id,
                    'name': agent_name,
                    'type': agent_type,
                    'capabilities': capabilities
                }
            },
            exclude_agent=agent_id
        )
    
    def unregister_agent(self, agent_id: str):
        """Remove agente do sistema"""
        with self._lock:
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            if agent_id in self.message_handlers:
                del self.message_handlers[agent_id]
            if agent_id in self.collaboration_callbacks:
                del self.collaboration_callbacks[agent_id]
        
        # Atualizar status no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE active_agents SET status = ?, last_seen = ? WHERE agent_id = ?',
            ('offline', datetime.now().isoformat(), agent_id)
        )
        conn.commit()
        conn.close()
    
    def send_message(self, message: AgentMessage):
        """Envia mensagem entre agentes"""
        # Salvar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO messages (id, from_agent, to_agent, message_type, content, priority, 
                                timestamp, expires_at, requires_response, response_to)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.id,
            message.from_agent,
            message.to_agent,
            message.message_type.value,
            json.dumps(message.content),
            message.priority.value,
            message.timestamp.isoformat(),
            message.expires_at.isoformat() if message.expires_at else None,
            message.requires_response,
            message.response_to
        ))
        conn.commit()
        conn.close()
        
        # Processar imediatamente se for urgente
        if message.priority == Priority.URGENT:
            self._process_message(message)
    
    def share_knowledge(self, agent_id: str, domain: str, knowledge_type: str,
                       content: Dict[str, Any], confidence_score: float = 0.8,
                       tags: List[str] = None):
        """Compartilha conhecimento com outros agentes"""
        knowledge = SharedKnowledge(
            id=str(uuid.uuid4()),
            source_agent=agent_id,
            domain=domain,
            knowledge_type=knowledge_type,
            content=content,
            confidence_score=confidence_score,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or []
        )
        
        # Salvar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO shared_knowledge (id, source_agent, domain, knowledge_type, content,
                                        confidence_score, created_at, updated_at, access_count, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            knowledge.id,
            knowledge.source_agent,
            knowledge.domain,
            knowledge.knowledge_type,
            json.dumps(knowledge.content),
            knowledge.confidence_score,
            knowledge.created_at.isoformat(),
            knowledge.updated_at.isoformat(),
            0,
            json.dumps(knowledge.tags)
        ))
        conn.commit()
        conn.close()
        
        # Notificar agentes interessados
        self._notify_knowledge_shared(knowledge)
        
        return knowledge.id
    
    def query_knowledge(self, agent_id: str, domain: str = None, knowledge_type: str = None,
                       tags: List[str] = None, min_confidence: float = 0.5) -> List[SharedKnowledge]:
        """Busca conhecimento compartilhado"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM shared_knowledge WHERE confidence_score >= ?'
        params = [min_confidence]
        
        if domain:
            query += ' AND domain = ?'
            params.append(domain)
        
        if knowledge_type:
            query += ' AND knowledge_type = ?'
            params.append(knowledge_type)
        
        query += ' ORDER BY confidence_score DESC, created_at DESC'
        
        cursor.execute(query, params)
        results = []
        
        for row in cursor.fetchall():
            knowledge = SharedKnowledge(
                id=row[0],
                source_agent=row[1],
                domain=row[2],
                knowledge_type=row[3],
                content=json.loads(row[4]),
                confidence_score=row[5],
                created_at=datetime.fromisoformat(row[6]),
                updated_at=datetime.fromisoformat(row[7]),
                access_count=row[8],
                tags=json.loads(row[9]) if row[9] else []
            )
            
            # Filtrar por tags se especificado
            if tags and not any(tag in knowledge.tags for tag in tags):
                continue
            
            results.append(knowledge)
            
            # Incrementar contador de acesso
            cursor.execute('UPDATE shared_knowledge SET access_count = access_count + 1 WHERE id = ?', (knowledge.id,))
        
        conn.commit()
        conn.close()
        
        return results
    
    def request_collaboration(self, initiator_agent: str, task_title: str, 
                            task_description: str, required_capabilities: List[str],
                            deadline: Optional[datetime] = None) -> str:
        """Solicita colaboração de outros agentes"""
        
        # Encontrar agentes com capacidades necessárias
        suitable_agents = []
        for agent_id, agent_info in self.active_agents.items():
            if agent_id == initiator_agent:
                continue
            
            agent_capabilities = agent_info.get('capabilities', [])
            if any(cap in agent_capabilities for cap in required_capabilities):
                suitable_agents.append(agent_id)
        
        # Criar tarefa colaborativa
        task = CollaborationTask(
            id=str(uuid.uuid4()),
            title=task_title,
            description=task_description,
            initiator_agent=initiator_agent,
            participating_agents=suitable_agents,
            status='pending',
            created_at=datetime.now(),
            deadline=deadline,
            shared_data={},
            results={}
        )
        
        # Salvar no banco
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO collaboration_tasks (id, title, description, initiator_agent,
                                           participating_agents, status, created_at, deadline, shared_data, results)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.id,
            task.title,
            task.description,
            task.initiator_agent,
            json.dumps(task.participating_agents),
            task.status,
            task.created_at.isoformat(),
            task.deadline.isoformat() if task.deadline else None,
            json.dumps(task.shared_data),
            json.dumps(task.results)
        ))
        conn.commit()
        conn.close()
        
        # Enviar solicitações de colaboração
        for agent_id in suitable_agents:
            message = AgentMessage(
                id=str(uuid.uuid4()),
                from_agent=initiator_agent,
                to_agent=agent_id,
                message_type=MessageType.COLLABORATION_REQUEST,
                content={
                    'task_id': task.id,
                    'title': task_title,
                    'description': task_description,
                    'required_capabilities': required_capabilities,
                    'deadline': deadline.isoformat() if deadline else None
                },
                priority=Priority.HIGH,
                timestamp=datetime.now(),
                requires_response=True
            )
            self.send_message(message)
        
        return task.id
    
    def accept_collaboration(self, agent_id: str, task_id: str):
        """Aceita participação em tarefa colaborativa"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Atualizar status da tarefa se necessário
        cursor.execute('SELECT * FROM collaboration_tasks WHERE id = ?', (task_id,))
        task_row = cursor.fetchone()
        
        if task_row and task_row[5] == 'pending':  # status
            cursor.execute('UPDATE collaboration_tasks SET status = ? WHERE id = ?', ('in_progress', task_id))
        
        conn.commit()
        conn.close()
        
        # Notificar iniciador
        if task_row:
            initiator = task_row[3]  # initiator_agent
            message = AgentMessage(
                id=str(uuid.uuid4()),
                from_agent=agent_id,
                to_agent=initiator,
                message_type=MessageType.RESULT_NOTIFICATION,
                content={
                    'event': 'collaboration_accepted',
                    'task_id': task_id,
                    'agent_id': agent_id
                },
                priority=Priority.MEDIUM,
                timestamp=datetime.now()
            )
            self.send_message(message)
    
    def contribute_to_task(self, agent_id: str, task_id: str, contribution: Dict[str, Any]):
        """Contribui com dados/resultados para tarefa colaborativa"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Obter tarefa atual
        cursor.execute('SELECT shared_data, results FROM collaboration_tasks WHERE id = ?', (task_id,))
        row = cursor.fetchone()
        
        if row:
            shared_data = json.loads(row[0]) if row[0] else {}
            results = json.loads(row[1]) if row[1] else {}
            
            # Adicionar contribuição
            results[agent_id] = {
                'contribution': contribution,
                'timestamp': datetime.now().isoformat()
            }
            
            # Atualizar no banco
            cursor.execute(
                'UPDATE collaboration_tasks SET results = ? WHERE id = ?',
                (json.dumps(results), task_id)
            )
            conn.commit()
        
        conn.close()
        
        # Notificar outros participantes
        self._notify_task_update(task_id, agent_id, 'contribution_added')
    
    def get_collaboration_tasks(self, agent_id: str) -> List[CollaborationTask]:
        """Obtém tarefas colaborativas do agente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM collaboration_tasks 
            WHERE initiator_agent = ? OR participating_agents LIKE ?
            ORDER BY created_at DESC
        ''', (agent_id, f'%{agent_id}%'))
        
        tasks = []
        for row in cursor.fetchall():
            task = CollaborationTask(
                id=row[0],
                title=row[1],
                description=row[2],
                initiator_agent=row[3],
                participating_agents=json.loads(row[4]),
                status=row[5],
                created_at=datetime.fromisoformat(row[6]),
                deadline=datetime.fromisoformat(row[7]) if row[7] else None,
                shared_data=json.loads(row[8]) if row[8] else {},
                results=json.loads(row[9]) if row[9] else {}
            )
            tasks.append(task)
        
        conn.close()
        return tasks
    
    def _process_messages_loop(self):
        """Loop principal de processamento de mensagens"""
        while True:
            try:
                # Processar mensagens pendentes
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM messages 
                    WHERE processed = FALSE AND (expires_at IS NULL OR expires_at > ?)
                    ORDER BY priority DESC, timestamp ASC
                    LIMIT 50
                ''', (datetime.now().isoformat(),))
                
                messages = cursor.fetchall()
                conn.close()
                
                for msg_row in messages:
                    message = AgentMessage(
                        id=msg_row[0],
                        from_agent=msg_row[1],
                        to_agent=msg_row[2],
                        message_type=MessageType(msg_row[3]),
                        content=json.loads(msg_row[4]),
                        priority=Priority(msg_row[5]),
                        timestamp=datetime.fromisoformat(msg_row[6]),
                        expires_at=datetime.fromisoformat(msg_row[7]) if msg_row[7] else None,
                        requires_response=msg_row[8],
                        response_to=msg_row[9]
                    )
                    
                    self._process_message(message)
                
                # Limpar mensagens expiradas
                self._cleanup_expired_messages()
                
                time.sleep(1)  # Pausa entre processamentos
                
            except Exception as e:
                print(f"Erro no processamento de mensagens: {e}")
                time.sleep(5)
    
    def _process_message(self, message: AgentMessage):
        """Processa uma mensagem individual"""
        try:
            # Verificar se agente destinatário está ativo
            if message.to_agent not in self.active_agents:
                return
            
            # Executar handler do agente se disponível
            if message.to_agent in self.message_handlers:
                handler = self.message_handlers[message.to_agent]
                handler(message)
            
            # Marcar como processada
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE messages SET processed = TRUE WHERE id = ?', (message.id,))
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Erro ao processar mensagem {message.id}: {e}")
    
    def _broadcast_message(self, from_agent: str, message_type: MessageType,
                          content: Dict[str, Any], priority: Priority = Priority.MEDIUM,
                          exclude_agent: str = None):
        """Envia mensagem para todos os agentes ativos"""
        for agent_id in self.active_agents.keys():
            if agent_id == exclude_agent:
                continue
            
            message = AgentMessage(
                id=str(uuid.uuid4()),
                from_agent=from_agent,
                to_agent=agent_id,
                message_type=message_type,
                content=content,
                priority=priority,
                timestamp=datetime.now()
            )
            self.send_message(message)
    
    def _notify_knowledge_shared(self, knowledge: SharedKnowledge):
        """Notifica agentes sobre novo conhecimento compartilhado"""
        content = {
            'event': 'knowledge_shared',
            'knowledge_id': knowledge.id,
            'domain': knowledge.domain,
            'knowledge_type': knowledge.knowledge_type,
            'source_agent': knowledge.source_agent,
            'confidence_score': knowledge.confidence_score,
            'tags': knowledge.tags
        }
        
        self._broadcast_message(
            from_agent='system',
            message_type=MessageType.LEARNING_UPDATE,
            content=content,
            exclude_agent=knowledge.source_agent
        )
    
    def _notify_task_update(self, task_id: str, agent_id: str, event_type: str):
        """Notifica sobre atualizações em tarefas colaborativas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT participating_agents FROM collaboration_tasks WHERE id = ?', (task_id,))
        row = cursor.fetchone()
        
        if row:
            participants = json.loads(row[0])
            for participant in participants:
                if participant != agent_id:
                    message = AgentMessage(
                        id=str(uuid.uuid4()),
                        from_agent=agent_id,
                        to_agent=participant,
                        message_type=MessageType.RESULT_NOTIFICATION,
                        content={
                            'event': event_type,
                            'task_id': task_id,
                            'updated_by': agent_id
                        },
                        priority=Priority.MEDIUM,
                        timestamp=datetime.now()
                    )
                    self.send_message(message)
        
        conn.close()
    
    def _cleanup_expired_messages(self):
        """Remove mensagens expiradas"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE expires_at IS NOT NULL AND expires_at <= ?', 
                      (datetime.now().isoformat(),))
        conn.commit()
        conn.close()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema de colaboração"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Contar mensagens
        cursor.execute('SELECT COUNT(*) FROM messages')
        total_messages = cursor.fetchone()[0]
        
        # Contar conhecimento compartilhado
        cursor.execute('SELECT COUNT(*) FROM shared_knowledge')
        total_knowledge = cursor.fetchone()[0]
        
        # Contar tarefas colaborativas
        cursor.execute('SELECT COUNT(*) FROM collaboration_tasks')
        total_tasks = cursor.fetchone()[0]
        
        # Agentes ativos
        cursor.execute('SELECT COUNT(*) FROM active_agents WHERE status = "active"')
        active_agents = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'active_agents': active_agents,
            'total_messages': total_messages,
            'shared_knowledge_items': total_knowledge,
            'collaboration_tasks': total_tasks,
            'system_uptime': datetime.now().isoformat()
        }

# Função de conveniência para criar sistema global
_global_collaboration_system = None

def get_collaboration_system() -> AgentCollaborationSystem:
    """Obtém instância global do sistema de colaboração"""
    global _global_collaboration_system
    if _global_collaboration_system is None:
        _global_collaboration_system = AgentCollaborationSystem()
    return _global_collaboration_system

def initialize_collaboration_system(db_path: str = "agent_collaboration.db") -> AgentCollaborationSystem:
    """Inicializa sistema de colaboração com configurações específicas"""
    global _global_collaboration_system
    _global_collaboration_system = AgentCollaborationSystem(db_path)
    return _global_collaboration_system