# performance_optimizer.py - Otimizador de Performance
"""
Sistema de otimização de performance que identifica e resolve
gargalos de memória, CPU e I/O no Team Agents.
"""

import gc
import os
import sys
import time
import psutil
import threading
import functools
import weakref
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import logging
from contextlib import contextmanager
import json

from shared.logging import get_logging_manager


@dataclass
class PerformanceMetrics:
    """Métricas de performance do sistema"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_io_read: int
    disk_io_write: int
    network_sent: int
    network_recv: int
    active_threads: int
    open_files: int
    database_connections: int = 0
    cache_hit_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'disk_io_read': self.disk_io_read,
            'disk_io_write': self.disk_io_write,
            'network_sent': self.network_sent,
            'network_recv': self.network_recv,
            'active_threads': self.active_threads,
            'open_files': self.open_files,
            'database_connections': self.database_connections,
            'cache_hit_rate': self.cache_hit_rate
        }


@dataclass
class OptimizationResult:
    """Resultado de otimização"""
    optimization_type: str
    description: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_percent: float
    applied: bool = False
    error: Optional[str] = None


class MemoryProfiler:
    """Profiler de uso de memória"""
    
    def __init__(self):
        self.snapshots = []
        self.object_counts = {}
        self.logger = get_logging_manager().get_logger("performance.memory")
    
    def take_snapshot(self, label: str = "snapshot") -> Dict[str, Any]:
        """Captura snapshot da memória"""
        gc.collect()  # Força coleta de lixo
        
        # Informações do processo
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Contagem de objetos Python
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        snapshot = {
            'label': label,
            'timestamp': datetime.now(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'object_counts': object_counts,
            'total_objects': len(gc.get_objects()),
            'gc_collections': gc.get_stats()
        }
        
        self.snapshots.append(snapshot)
        self.logger.info(f"Snapshot '{label}': {snapshot['rss_mb']:.1f}MB RAM, {snapshot['total_objects']} objetos")
        
        return snapshot
    
    def compare_snapshots(self, before_label: str, after_label: str) -> Dict[str, Any]:
        """Compara dois snapshots"""
        before = next((s for s in self.snapshots if s['label'] == before_label), None)
        after = next((s for s in self.snapshots if s['label'] == after_label), None)
        
        if not before or not after:
            return {'error': 'Snapshots não encontrados'}
        
        # Diferenças de memória
        memory_diff = after['rss_mb'] - before['rss_mb']
        object_diff = after['total_objects'] - before['total_objects']
        
        # Tipos de objeto que mais cresceram
        object_growth = {}
        for obj_type in set(list(before['object_counts'].keys()) + list(after['object_counts'].keys())):
            before_count = before['object_counts'].get(obj_type, 0)
            after_count = after['object_counts'].get(obj_type, 0)
            diff = after_count - before_count
            if diff != 0:
                object_growth[obj_type] = diff
        
        # Top crescimentos
        top_growth = sorted(object_growth.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        
        return {
            'memory_diff_mb': memory_diff,
            'object_diff': object_diff,
            'top_object_growth': top_growth,
            'before_memory': before['rss_mb'],
            'after_memory': after['rss_mb'],
            'percent_change': (memory_diff / before['rss_mb']) * 100 if before['rss_mb'] > 0 else 0
        }
    
    def detect_memory_leaks(self) -> List[Dict[str, Any]]:
        """Detecta possíveis vazamentos de memória"""
        if len(self.snapshots) < 3:
            return []
        
        leaks = []
        recent_snapshots = self.snapshots[-3:]
        
        # Verificar crescimento constante de tipos específicos
        for obj_type in ['dict', 'list', 'tuple', 'str', 'function']:
            counts = [s['object_counts'].get(obj_type, 0) for s in recent_snapshots]
            
            # Se há crescimento consistente
            if len(counts) >= 3 and counts[1] > counts[0] and counts[2] > counts[1]:
                growth_rate = (counts[-1] - counts[0]) / len(counts)
                if growth_rate > 100:  # Mais de 100 objetos por snapshot
                    leaks.append({
                        'object_type': obj_type,
                        'growth_rate': growth_rate,
                        'current_count': counts[-1],
                        'severity': 'high' if growth_rate > 1000 else 'medium'
                    })
        
        return leaks


class ResourceMonitor:
    """Monitor de recursos do sistema"""
    
    def __init__(self, interval_seconds: int = 5):
        self.interval = interval_seconds
        self.metrics_history = []
        self.monitoring = False
        self.monitor_thread = None
        self.logger = get_logging_manager().get_logger("performance.monitor")
        
        # Thresholds de alerta
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_io_mb_per_sec': 100.0,
            'open_files': 1000
        }
    
    def start_monitoring(self):
        """Inicia monitoramento contínuo"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Monitoramento de recursos iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Monitoramento de recursos parado")
    
    def _monitor_loop(self):
        """Loop principal de monitoramento"""
        last_disk_io = None
        last_network = None
        
        while self.monitoring:
            try:
                metrics = self._collect_metrics(last_disk_io, last_network)
                self.metrics_history.append(metrics)
                
                # Manter apenas últimas 1000 métricas
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                
                # Verificar thresholds
                self._check_thresholds(metrics)
                
                # Atualizar baselines
                last_disk_io = (metrics.disk_io_read, metrics.disk_io_write)
                last_network = (metrics.network_sent, metrics.network_recv)
                
                time.sleep(self.interval)
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(self.interval)
    
    def _collect_metrics(self, last_disk_io: Optional[Tuple[int, int]] = None,
                        last_network: Optional[Tuple[int, int]] = None) -> PerformanceMetrics:
        """Coleta métricas atuais do sistema"""
        process = psutil.Process()
        
        # CPU e Memória
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        # I/O de disco
        disk_io = process.io_counters()
        disk_read = disk_io.read_bytes
        disk_write = disk_io.write_bytes
        
        # Network I/O (aproximação)
        try:
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent
            net_recv = net_io.bytes_recv
        except:
            net_sent = net_recv = 0
        
        # Threads e arquivos
        active_threads = process.num_threads()
        try:
            open_files = len(process.open_files())
        except:
            open_files = 0
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_sent=net_sent,
            network_recv=net_recv,
            active_threads=active_threads,
            open_files=open_files
        )
    
    def _check_thresholds(self, metrics: PerformanceMetrics):
        """Verifica se métricas excedem thresholds"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            alerts.append(f"Alto uso de CPU: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds['memory_percent']:
            alerts.append(f"Alto uso de memória: {metrics.memory_percent:.1f}%")
        
        if metrics.open_files > self.thresholds['open_files']:
            alerts.append(f"Muitos arquivos abertos: {metrics.open_files}")
        
        for alert in alerts:
            self.logger.warning(alert)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Retorna métricas atuais"""
        return self._collect_metrics()
    
    def get_metrics_summary(self, minutes: int = 30) -> Dict[str, Any]:
        """Retorna resumo das métricas dos últimos N minutos"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {'error': 'Nenhuma métrica encontrada'}
        
        # Calcular estatísticas
        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_mb for m in recent_metrics]
        
        return {
            'period_minutes': minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory': {
                'avg_mb': sum(memory_values) / len(memory_values),
                'max_mb': max(memory_values),
                'min_mb': min(memory_values),
                'current_mb': recent_metrics[-1].memory_mb
            },
            'threads': {
                'current': recent_metrics[-1].active_threads,
                'max': max(m.active_threads for m in recent_metrics)
            }
        }


class DatabaseOptimizer:
    """Otimizador de performance de banco de dados"""
    
    def __init__(self):
        self.connection_pool = {}
        self.query_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.logger = get_logging_manager().get_logger("performance.database")
    
    def optimize_database_file(self, db_path: str) -> OptimizationResult:
        """Otimiza arquivo de banco específico"""
        before_size = Path(db_path).stat().st_size if Path(db_path).exists() else 0
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # VACUUM para compactar
                cursor.execute("VACUUM")
                
                # ANALYZE para atualizar estatísticas
                cursor.execute("ANALYZE")
                
                # Configurações de performance
                cursor.execute("PRAGMA optimize")
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.execute("PRAGMA synchronous = NORMAL")
                cursor.execute("PRAGMA cache_size = 10000")
                cursor.execute("PRAGMA temp_store = MEMORY")
                
                conn.commit()
            
            after_size = Path(db_path).stat().st_size
            size_reduction = before_size - after_size
            improvement = (size_reduction / before_size * 100) if before_size > 0 else 0
            
            return OptimizationResult(
                optimization_type="database_vacuum",
                description=f"Otimização do banco {db_path}",
                before_metrics={'size_bytes': before_size},
                after_metrics={'size_bytes': after_size},
                improvement_percent=improvement,
                applied=True
            )
            
        except Exception as e:
            return OptimizationResult(
                optimization_type="database_vacuum",
                description=f"Erro na otimização do banco {db_path}",
                before_metrics={'size_bytes': before_size},
                after_metrics={'size_bytes': before_size},
                improvement_percent=0,
                applied=False,
                error=str(e)
            )
    
    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do pool de conexões"""
        return {
            'active_connections': len(self.connection_pool),
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cached_queries': len(self.query_cache)
        }


class PerformanceOptimizer:
    """Otimizador principal de performance"""
    
    def __init__(self):
        self.memory_profiler = MemoryProfiler()
        self.resource_monitor = ResourceMonitor()
        self.db_optimizer = DatabaseOptimizer()
        self.logger = get_logging_manager().get_logger("performance.optimizer")
        
        # Cache global com TTL
        self._cache = {}
        self._cache_ttl = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Iniciar monitoramento
        self.resource_monitor.start_monitoring()
    
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Executa otimização completa do sistema"""
        self.logger.info("Iniciando otimização completa do sistema")
        
        optimization_results = {
            'started_at': datetime.now().isoformat(),
            'memory_optimization': None,
            'database_optimization': [],
            'cache_optimization': None,
            'gc_optimization': None,
            'overall_improvement': 0.0
        }
        
        # 1. Snapshot inicial
        self.memory_profiler.take_snapshot("before_optimization")
        initial_metrics = self.resource_monitor.get_current_metrics()
        
        try:
            # 2. Otimização de memória
            optimization_results['memory_optimization'] = self._optimize_memory()
            
            # 3. Otimização de banco de dados
            optimization_results['database_optimization'] = self._optimize_databases()
            
            # 4. Otimização de cache
            optimization_results['cache_optimization'] = self._optimize_cache()
            
            # 5. Garbage collection
            optimization_results['gc_optimization'] = self._optimize_garbage_collection()
            
            # 6. Snapshot final
            self.memory_profiler.take_snapshot("after_optimization")
            final_metrics = self.resource_monitor.get_current_metrics()
            
            # 7. Calcular melhoria geral
            if initial_metrics and final_metrics:
                memory_improvement = (initial_metrics.memory_mb - final_metrics.memory_mb) / initial_metrics.memory_mb * 100
                optimization_results['overall_improvement'] = memory_improvement
                optimization_results['memory_saved_mb'] = initial_metrics.memory_mb - final_metrics.memory_mb
            
            optimization_results['completed_at'] = datetime.now().isoformat()
            optimization_results['success'] = True
            
        except Exception as e:
            optimization_results['error'] = str(e)
            optimization_results['success'] = False
            self.logger.error(f"Erro na otimização: {e}")
        
        return optimization_results
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """Otimiza uso de memória"""
        self.logger.info("Otimizando uso de memória")
        
        # Detectar vazamentos
        leaks = self.memory_profiler.detect_memory_leaks()
        
        # Forçar garbage collection
        collected = gc.collect()
        
        # Limpar caches
        cache_cleared = len(self._cache)
        self._cache.clear()
        self._cache_ttl.clear()
        
        return {
            'garbage_collected': collected,
            'cache_entries_cleared': cache_cleared,
            'memory_leaks_detected': len(leaks),
            'leaks': leaks
        }
    
    def _optimize_databases(self) -> List[Dict[str, Any]]:
        """Otimiza bancos de dados"""
        self.logger.info("Otimizando bancos de dados")
        
        results = []
        
        # Encontrar arquivos de banco
        db_files = list(Path('.').rglob('*.db'))
        
        for db_file in db_files:
            if db_file.stat().st_size > 1024:  # Apenas arquivos > 1KB
                result = self.db_optimizer.optimize_database_file(str(db_file))
                results.append({
                    'file': str(db_file),
                    'optimization': result.__dict__
                })
        
        return results
    
    def _optimize_cache(self) -> Dict[str, Any]:
        """Otimiza sistema de cache"""
        self.logger.info("Otimizando cache")
        
        # Limpar entradas expiradas
        now = time.time()
        expired_keys = [key for key, ttl in self._cache_ttl.items() if ttl < now]
        
        for key in expired_keys:
            del self._cache[key]
            del self._cache_ttl[key]
        
        # Estatísticas do cache
        hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        
        return {
            'expired_entries_removed': len(expired_keys),
            'current_cache_size': len(self._cache),
            'hit_rate': hit_rate,
            'total_hits': self._cache_hits,
            'total_misses': self._cache_misses
        }
    
    def _optimize_garbage_collection(self) -> Dict[str, Any]:
        """Otimiza garbage collection"""
        self.logger.info("Otimizando garbage collection")
        
        # Configurar GC para ser mais agressivo com objetos temporários
        gc.set_threshold(700, 10, 10)  # Mais agressivo que padrão (700, 10, 10)
        
        # Múltiplas passadas de coleta
        collections = []
        for generation in range(3):
            collected = gc.collect(generation)
            collections.append(collected)
        
        # Estatísticas do GC
        stats = gc.get_stats()
        
        return {
            'collections_by_generation': collections,
            'total_collected': sum(collections),
            'gc_stats': stats,
            'gc_thresholds': gc.get_threshold()
        }
    
    @contextmanager
    def performance_context(self, operation_name: str):
        """Context manager para monitorar performance de operações"""
        start_time = time.time()
        start_memory = self.resource_monitor.get_current_metrics()
        
        self.logger.info(f"Iniciando operação: {operation_name}")
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.resource_monitor.get_current_metrics()
            duration = end_time - start_time
            
            memory_diff = (end_memory.memory_mb - start_memory.memory_mb) if start_memory and end_memory else 0
            
            self.logger.info(
                f"Operação concluída: {operation_name} "
                f"(duração: {duration:.2f}s, memória: {memory_diff:+.1f}MB)"
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Gera relatório completo de performance"""
        current_metrics = self.resource_monitor.get_current_metrics()
        metrics_summary = self.resource_monitor.get_metrics_summary(minutes=30)
        db_stats = self.db_optimizer.get_connection_pool_stats()
        
        # Memória detalhada
        memory_comparison = None
        if len(self.memory_profiler.snapshots) >= 2:
            memory_comparison = self.memory_profiler.compare_snapshots(
                self.memory_profiler.snapshots[-2]['label'],
                self.memory_profiler.snapshots[-1]['label']
            )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics.to_dict() if current_metrics else None,
            'metrics_summary_30min': metrics_summary,
            'database_stats': db_stats,
            'memory_analysis': memory_comparison,
            'cache_stats': {
                'size': len(self._cache),
                'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
            },
            'gc_stats': gc.get_stats()
        }
    
    def cached_operation(self, ttl_seconds: int = 3600):
        """Decorator para cache com TTL"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Criar chave do cache
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                
                # Verificar cache
                now = time.time()
                if cache_key in self._cache and self._cache_ttl.get(cache_key, 0) > now:
                    self._cache_hits += 1
                    return self._cache[cache_key]
                
                # Cache miss - executar função
                self._cache_misses += 1
                result = func(*args, **kwargs)
                
                # Armazenar no cache
                self._cache[cache_key] = result
                self._cache_ttl[cache_key] = now + ttl_seconds
                
                return result
            return wrapper
        return decorator
    
    def cleanup_and_shutdown(self):
        """Limpeza antes do shutdown"""
        self.logger.info("Executando limpeza final")
        
        # Parar monitoramento
        self.resource_monitor.stop_monitoring()
        
        # Forçar GC final
        gc.collect()
        
        # Salvar relatório final
        report = self.get_performance_report()
        with open('final_performance_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info("Limpeza concluída, relatório salvo")


# Instância global
_global_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Obtém instância global do otimizador"""
    global _global_performance_optimizer
    if _global_performance_optimizer is None:
        _global_performance_optimizer = PerformanceOptimizer()
    return _global_performance_optimizer


# Decorators de conveniência
def monitor_performance(operation_name: str = None):
    """Decorator para monitorar performance de funções"""
    def decorator(func):
        name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_performance_optimizer()
            with optimizer.performance_context(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def cached_result(ttl_seconds: int = 3600):
    """Decorator para cache de resultados"""
    def decorator(func):
        optimizer = get_performance_optimizer()
        return optimizer.cached_operation(ttl_seconds)(func)
    return decorator