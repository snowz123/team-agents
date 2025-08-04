# intelligent_data_analytics.py - Sistema Avançado de Análise Inteligente de Dados
"""
Sistema completo de análise inteligente de dados com visualização avançada,
insights automáticos, machine learning e relatórios executivos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, List, Any, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Imports condicionais
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import io
import base64
from datetime import datetime, timedelta
import tempfile
import os

class DataInsight:
    """Representa um insight descoberto nos dados"""
    def __init__(self, insight_type: str, title: str, description: str, 
                 importance: float, data: Dict[str, Any] = None):
        self.insight_type = insight_type
        self.title = title
        self.description = description
        self.importance = importance  # 0.0 - 1.0
        self.data = data or {}
        self.timestamp = datetime.now()

class IntelligentDataAnalytics:
    """Sistema avançado de análise inteligente de dados"""
    
    def __init__(self):
        self.data = None
        self.insights = []
        self.visualizations = {}
        self.ml_models = {}
        self.reports = {}
        
    def load_data(self, data_source, **kwargs) -> bool:
        """Carrega dados de diferentes fontes"""
        try:
            if isinstance(data_source, str):
                # Arquivo
                if data_source.endswith('.csv'):
                    self.data = pd.read_csv(data_source, **kwargs)
                elif data_source.endswith('.json'):
                    self.data = pd.read_json(data_source, **kwargs)
                elif data_source.endswith('.xlsx') or data_source.endswith('.xls'):
                    self.data = pd.read_excel(data_source, **kwargs)
                elif data_source.endswith('.parquet'):
                    self.data = pd.read_parquet(data_source, **kwargs)
                else:
                    # Tentar SQL ou URL
                    try:
                        import sqlite3
                        conn = sqlite3.connect(data_source)
                        self.data = pd.read_sql_query("SELECT * FROM main_table", conn)
                        conn.close()
                    except:
                        # Tentar como URL
                        self.data = pd.read_csv(data_source, **kwargs)
            
            elif isinstance(data_source, pd.DataFrame):
                self.data = data_source.copy()
            
            elif isinstance(data_source, dict):
                self.data = pd.DataFrame(data_source)
            
            elif isinstance(data_source, list):
                self.data = pd.DataFrame(data_source)
            
            else:
                return False
            
            # Análise inicial automática
            self._perform_initial_analysis()
            return True
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
    
    def _perform_initial_analysis(self):
        """Análise inicial automática dos dados"""
        if self.data is None:
            return
        
        # Detectar tipos de dados
        self._detect_data_types()
        
        # Detectar problemas de qualidade
        self._detect_data_quality_issues()
        
        # Análise estatística básica
        self._basic_statistical_analysis()
        
        # Detectar padrões iniciais
        self._detect_initial_patterns()
    
    def _detect_data_types(self):
        """Detecta e otimiza tipos de dados"""
        insights = []
        
        for col in self.data.columns:
            original_type = self.data[col].dtype
            
            # Tentar converter para tipos mais eficientes
            if self.data[col].dtype == 'object':
                # Verificar se é data
                try:
                    pd.to_datetime(self.data[col].head(100))
                    self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                    insights.append(DataInsight(
                        "data_type",
                        f"Coluna {col} convertida para datetime",
                        f"A coluna '{col}' foi automaticamente convertida de {original_type} para datetime",
                        0.6
                    ))
                    continue
                except:
                    pass
                
                # Verificar se é numérico
                if self.data[col].str.replace(',', '').str.replace('.', '').str.replace('-', '').str.isdigit().all():
                    try:
                        self.data[col] = pd.to_numeric(self.data[col].str.replace(',', ''))
                        insights.append(DataInsight(
                            "data_type",
                            f"Coluna {col} convertida para numérico",
                            f"A coluna '{col}' foi automaticamente convertida para tipo numérico",
                            0.7
                        ))
                    except:
                        pass
                
                # Verificar se é categoria
                unique_ratio = self.data[col].nunique() / len(self.data)
                if unique_ratio < 0.05:  # Menos de 5% de valores únicos
                    self.data[col] = self.data[col].astype('category')
                    insights.append(DataInsight(
                        "data_type",
                        f"Coluna {col} convertida para categoria",
                        f"A coluna '{col}' foi convertida para categoria (otimização de memória)",
                        0.5
                    ))
        
        self.insights.extend(insights)
    
    def _detect_data_quality_issues(self):
        """Detecta problemas de qualidade dos dados"""
        if self.data is None:
            return
        
        insights = []
        
        # Valores ausentes
        missing_data = self.data.isnull().sum()
        missing_percentages = (missing_data / len(self.data)) * 100
        
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = missing_percentages[col]
                if percentage > 50:
                    importance = 0.9
                    severity = "crítico"
                elif percentage > 20:
                    importance = 0.7
                    severity = "alto"
                elif percentage > 5:
                    importance = 0.5
                    severity = "médio"
                else:
                    importance = 0.3
                    severity = "baixo"
                
                insights.append(DataInsight(
                    "data_quality",
                    f"Valores ausentes em {col}",
                    f"A coluna '{col}' tem {missing_count} valores ausentes ({percentage:.1f}%) - Severidade: {severity}",
                    importance,
                    {"column": col, "missing_count": missing_count, "percentage": percentage}
                ))
        
        # Duplicatas
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            insights.append(DataInsight(
                "data_quality",
                "Registros duplicados encontrados",
                f"Encontrados {duplicates} registros duplicados no dataset ({(duplicates/len(self.data)*100):.1f}%)",
                0.6,
                {"duplicate_count": duplicates}
            ))
        
        # Outliers (apenas para colunas numéricas)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > 0:
                outlier_percentage = (outliers / len(self.data)) * 100
                insights.append(DataInsight(
                    "data_quality",
                    f"Outliers detectados em {col}",
                    f"A coluna '{col}' tem {outliers} possíveis outliers ({outlier_percentage:.1f}%)",
                    0.4,
                    {"column": col, "outlier_count": outliers, "percentage": outlier_percentage}
                ))
        
        self.insights.extend(insights)
    
    def _basic_statistical_analysis(self):
        """Análise estatística básica automática"""
        if self.data is None:
            return
        
        insights = []
        
        # Estatísticas descritivas para colunas numéricas
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            desc_stats = self.data[numeric_cols].describe()
            
            for col in numeric_cols:
                mean_val = desc_stats.loc['mean', col]
                std_val = desc_stats.loc['std', col]
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                
                if cv > 100:
                    insights.append(DataInsight(
                        "statistical",
                        f"Alta variabilidade em {col}",
                        f"A coluna '{col}' tem coeficiente de variação de {cv:.1f}%, indicando alta dispersão",
                        0.6,
                        {"column": col, "cv": cv, "mean": mean_val, "std": std_val}
                    ))
                
                # Verificar distribuição
                skewness = self.data[col].skew()
                if abs(skewness) > 2:
                    direction = "positivamente" if skewness > 0 else "negativamente"
                    insights.append(DataInsight(
                        "statistical",
                        f"Assimetria em {col}",
                        f"A coluna '{col}' está {direction} assimétrica (skewness: {skewness:.2f})",
                        0.5,
                        {"column": col, "skewness": skewness}
                    ))
        
        # Análise de correlações
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            
            # Encontrar correlações altas
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            for pair in high_corr_pairs:
                direction = "positiva" if pair['correlation'] > 0 else "negativa"
                insights.append(DataInsight(
                    "correlation",
                    f"Correlação {direction} forte",
                    f"'{pair['var1']}' e '{pair['var2']}' têm correlação {direction} forte ({pair['correlation']:.3f})",
                    0.7,
                    pair
                ))
        
        self.insights.extend(insights)
    
    def _detect_initial_patterns(self):
        """Detecta padrões iniciais nos dados"""
        if self.data is None:
            return
        
        insights = []
        
        # Padrões temporais (se houver colunas de data)
        date_cols = self.data.select_dtypes(include=['datetime64']).columns
        for col in date_cols:
            # Detectar tendências temporais
            if len(self.data) > 10:
                self.data['_temp_year'] = self.data[col].dt.year
                self.data['_temp_month'] = self.data[col].dt.month
                
                # Padrão sazonal
                seasonal_pattern = self.data.groupby('_temp_month').size().std()
                if seasonal_pattern > self.data.groupby('_temp_month').size().mean() * 0.5:
                    insights.append(DataInsight(
                        "temporal",
                        f"Padrão sazonal detectado",
                        f"Os dados mostram variação sazonal significativa ao longo do ano",
                        0.6,
                        {"column": col, "seasonal_std": seasonal_pattern}
                    ))
                
                # Limpar colunas temporárias
                self.data.drop(['_temp_year', '_temp_month'], axis=1, inplace=True)
        
        # Padrões categóricos
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = self.data[col].value_counts()
            
            # Detectar dominância de categoria
            if len(value_counts) > 1:
                dominance_ratio = value_counts.iloc[0] / len(self.data)
                if dominance_ratio > 0.8:
                    insights.append(DataInsight(
                        "categorical",
                        f"Categoria dominante em {col}",
                        f"A categoria '{value_counts.index[0]}' representa {dominance_ratio*100:.1f}% dos dados",
                        0.5,
                        {"column": col, "dominant_category": value_counts.index[0], "ratio": dominance_ratio}
                    ))
        
        self.insights.extend(insights)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Gera relatório abrangente dos dados"""
        if self.data is None:
            return {"error": "Nenhum dado carregado"}
        
        report = {
            "summary": {
                "rows": len(self.data),
                "columns": len(self.data.columns),
                "memory_usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                "data_types": self.data.dtypes.value_counts().to_dict()
            },
            "data_quality": {
                "missing_data": self.data.isnull().sum().to_dict(),
                "duplicate_rows": self.data.duplicated().sum(),
                "completeness_score": (1 - self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
            },
            "insights": [
                {
                    "type": insight.insight_type,
                    "title": insight.title,
                    "description": insight.description,
                    "importance": insight.importance,
                    "data": insight.data
                }
                for insight in sorted(self.insights, key=lambda x: x.importance, reverse=True)
            ],
            "recommendations": self._generate_recommendations()
        }
        
        # Adicionar estatísticas por tipo de dado
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report["numeric_analysis"] = self.data[numeric_cols].describe().to_dict()
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            report["categorical_analysis"] = {}
            for col in categorical_cols:
                report["categorical_analysis"][col] = {
                    "unique_values": self.data[col].nunique(),
                    "most_frequent": self.data[col].mode().iloc[0] if not self.data[col].mode().empty else None,
                    "frequency_distribution": self.data[col].value_counts().head(10).to_dict()
                }
        
        self.reports["comprehensive"] = report
        return report
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Gera recomendações baseadas nos insights"""
        recommendations = []
        
        # Recomendações baseadas em qualidade de dados
        for insight in self.insights:
            if insight.insight_type == "data_quality":
                if "valores ausentes" in insight.title.lower():
                    percentage = insight.data.get("percentage", 0)
                    if percentage > 50:
                        recommendations.append({
                            "priority": "high",
                            "action": "remove_column",
                            "description": f"Considere remover a coluna '{insight.data['column']}' devido ao alto percentual de valores ausentes",
                            "column": insight.data["column"]
                        })
                    elif percentage > 20:
                        recommendations.append({
                            "priority": "medium",
                            "action": "impute_values",
                            "description": f"Implemente estratégia de imputação para a coluna '{insight.data['column']}'",
                            "column": insight.data["column"]
                        })
                
                elif "duplicados" in insight.title.lower():
                    recommendations.append({
                        "priority": "medium",
                        "action": "remove_duplicates",
                        "description": "Remova registros duplicados para melhorar a qualidade dos dados"
                    })
        
        # Recomendações baseadas em correlações
        for insight in self.insights:
            if insight.insight_type == "correlation" and insight.importance > 0.6:
                recommendations.append({
                    "priority": "medium",
                    "action": "investigate_correlation",
                    "description": f"Investigue a relação entre '{insight.data['var1']}' e '{insight.data['var2']}' - possível multicolinearidade"
                })
        
        return recommendations
    
    def create_advanced_visualizations(self) -> Dict[str, str]:
        """Cria visualizações avançadas e interativas"""
        if self.data is None:
            return {}
        
        visualizations = {}
        
        # 1. Dashboard Overview
        overview_fig = self._create_overview_dashboard()
        visualizations["overview_dashboard"] = self._fig_to_html(overview_fig)
        
        # 2. Análise de Correlação
        if len(self.data.select_dtypes(include=[np.number]).columns) > 1:
            corr_fig = self._create_correlation_heatmap()
            visualizations["correlation_heatmap"] = self._fig_to_html(corr_fig)
        
        # 3. Análise de Distribuições
        dist_fig = self._create_distribution_analysis()
        visualizations["distribution_analysis"] = self._fig_to_html(dist_fig)
        
        # 4. Análise Temporal (se houver dados de data)
        date_cols = self.data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            temporal_fig = self._create_temporal_analysis(date_cols[0])
            visualizations["temporal_analysis"] = self._fig_to_html(temporal_fig)
        
        # 5. Análise de Outliers
        outlier_fig = self._create_outlier_analysis()
        visualizations["outlier_analysis"] = self._fig_to_html(outlier_fig)
        
        self.visualizations.update(visualizations)
        return visualizations
    
    def _create_overview_dashboard(self):
        """Cria dashboard de overview"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Tipos de Dados", "Valores Ausentes", "Estatísticas Básicas", "Qualidade dos Dados"),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator", "colspan": 1}]]
        )
        
        # Gráfico de pizza - tipos de dados
        dtype_counts = self.data.dtypes.value_counts()
        fig.add_trace(
            go.Pie(labels=dtype_counts.index.astype(str), values=dtype_counts.values, name="Tipos"),
            row=1, col=1
        )
        
        # Gráfico de barras - valores ausentes
        missing_data = self.data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            fig.add_trace(
                go.Bar(x=missing_data.index, y=missing_data.values, name="Valores Ausentes"),
                row=1, col=2
            )
        
        # Estatísticas básicas
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_data = self.data[numeric_cols].describe().loc[['mean', 'std']].T
            fig.add_trace(
                go.Bar(x=stats_data.index, y=stats_data['mean'], name="Média"),
                row=2, col=1
            )
        
        # Indicador de qualidade
        completeness = (1 - self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=completeness,
                title={"text": "Qualidade dos Dados (%)"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "green"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90
                    }
                }
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Dashboard de Análise de Dados",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_correlation_heatmap(self):
        """Cria heatmap de correlação interativo"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return go.Figure()
        
        corr_matrix = self.data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Matriz de Correlação",
            xaxis_title="Variáveis",
            yaxis_title="Variáveis",
            height=600
        )
        
        return fig
    
    def _create_distribution_analysis(self):
        """Cria análise de distribuições"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return go.Figure()
        
        n_cols = min(len(numeric_cols), 4)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols[:n_rows*n_cols]
        )
        
        for i, col in enumerate(numeric_cols[:n_rows*n_cols]):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1
            
            fig.add_trace(
                go.Histogram(x=self.data[col], name=col, nbinsx=30),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title="Análise de Distribuições",
            height=300*n_rows,
            showlegend=False
        )
        
        return fig
    
    def _create_temporal_analysis(self, date_col):
        """Cria análise temporal"""
        if date_col not in self.data.columns:
            return go.Figure()
        
        # Agrupar por período
        temporal_data = self.data.set_index(date_col).resample('M').size()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=temporal_data.index,
            y=temporal_data.values,
            mode='lines+markers',
            name='Contagem por Mês',
            line=dict(width=2)
        ))
        
        # Adicionar tendência
        from scipy.stats import linregress
        x_numeric = np.arange(len(temporal_data))
        slope, intercept, r_value, p_value, std_err = linregress(x_numeric, temporal_data.values)
        trend_line = slope * x_numeric + intercept
        
        fig.add_trace(go.Scatter(
            x=temporal_data.index,
            y=trend_line,
            mode='lines',
            name=f'Tendência (R²={r_value**2:.3f})',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title="Análise Temporal",
            xaxis_title="Data",
            yaxis_title="Frequência",
            height=400
        )
        
        return fig
    
    def _create_outlier_analysis(self):
        """Cria análise de outliers"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return go.Figure()
        
        fig = go.Figure()
        
        for col in numeric_cols:
            fig.add_trace(go.Box(
                y=self.data[col],
                name=col,
                boxpoints='outliers'
            ))
        
        fig.update_layout(
            title="Análise de Outliers",
            yaxis_title="Valores",
            height=500
        )
        
        return fig
    
    def _fig_to_html(self, fig) -> str:
        """Converte figura Plotly para HTML"""
        return pyo.plot(fig, output_type='div', include_plotlyjs=True)
    
    def perform_machine_learning_analysis(self, target_column: str = None) -> Dict[str, Any]:
        """Realiza análise de machine learning automática"""
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn não disponível"}
        
        if self.data is None:
            return {"error": "Nenhum dado carregado"}
        
        results = {}
        
        # Clustering automático
        clustering_result = self._perform_clustering_analysis()
        results["clustering"] = clustering_result
        
        # Detecção de anomalias
        anomaly_result = self._perform_anomaly_detection()
        results["anomaly_detection"] = anomaly_result
        
        # Análise preditiva (se target especificado)
        if target_column and target_column in self.data.columns:
            prediction_result = self._perform_prediction_analysis(target_column)
            results["prediction"] = prediction_result
        
        # Análise de componentes principais
        pca_result = self._perform_pca_analysis()
        results["pca"] = pca_result
        
        self.ml_models.update(results)
        return results
    
    def _perform_clustering_analysis(self) -> Dict[str, Any]:
        """Análise de clustering automática"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "Dados insuficientes para clustering"}
        
        # Preparar dados
        X = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means com diferentes números de clusters
        results = {"kmeans": {}, "dbscan": {}}
        
        # K-means
        inertias = []
        silhouette_scores = []
        
        from sklearn.metrics import silhouette_score
        
        for k in range(2, min(11, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, labels))
        
        # Melhor número de clusters (método do cotovelo + silhouette)
        best_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Clustering final
        final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(X_scaled)
        
        results["kmeans"] = {
            "best_k": best_k,
            "silhouette_score": max(silhouette_scores),
            "cluster_centers": final_kmeans.cluster_centers_.tolist(),
            "labels": cluster_labels.tolist(),
            "inertias": inertias,
            "silhouette_scores": silhouette_scores
        }
        
        # DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_noise = list(dbscan_labels).count(-1)
        
        results["dbscan"] = {
            "n_clusters": n_clusters_dbscan,
            "n_noise_points": n_noise,
            "labels": dbscan_labels.tolist()
        }
        
        return results
    
    def _perform_anomaly_detection(self) -> Dict[str, Any]:
        """Detecção de anomalias"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"error": "Nenhuma coluna numérica para análise de anomalias"}
        
        X = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.score_samples(X)
        
        n_anomalies = (anomaly_labels == -1).sum()
        
        return {
            "method": "Isolation Forest",
            "n_anomalies": int(n_anomalies),
            "anomaly_percentage": float(n_anomalies / len(X) * 100),
            "anomaly_labels": anomaly_labels.tolist(),
            "anomaly_scores": anomaly_scores.tolist(),
            "threshold": float(np.percentile(anomaly_scores, 10))
        }
    
    def _perform_prediction_analysis(self, target_column: str) -> Dict[str, Any]:
        """Análise preditiva automática"""
        if target_column not in self.data.columns:
            return {"error": f"Coluna {target_column} não encontrada"}
        
        # Preparar dados
        y = self.data[target_column].dropna()
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_column]
        
        if len(numeric_cols) == 0:
            return {"error": "Nenhuma feature numérica disponível"}
        
        X = self.data[numeric_cols].loc[y.index].fillna(self.data[numeric_cols].mean())
        
        # Determinar se é classificação ou regressão
        is_classification = y.dtype == 'object' or y.nunique() < 10
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {"task_type": "classification" if is_classification else "regression"}
        
        if is_classification:
            # Codificar labels se necessário
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_test_encoded = label_encoder.transform(y_test)
            else:
                y_train_encoded = y_train
                y_test_encoded = y_test
                label_encoder = None
            
            # Random Forest Classifier
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train_scaled, y_train_encoded)
            y_pred = rf_classifier.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_test_encoded, y_pred)
            
            results.update({
                "model": "Random Forest Classifier",
                "accuracy": float(accuracy),
                "feature_importance": dict(zip(X.columns, rf_classifier.feature_importances_)),
                "classification_report": classification_report(y_test_encoded, y_pred, output_dict=True)
            })
        
        else:
            # Linear Regression
            linear_reg = LinearRegression()
            linear_reg.fit(X_train_scaled, y_train)
            y_pred = linear_reg.predict(X_test_scaled)
            
            from sklearn.metrics import mean_squared_error, r2_score
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.update({
                "model": "Linear Regression",
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2_score": float(r2),
                "coefficients": dict(zip(X.columns, linear_reg.coef_))
            })
        
        return results
    
    def _perform_pca_analysis(self) -> Dict[str, Any]:
        """Análise de componentes principais"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return {"error": "Dados insuficientes para PCA"}
        
        X = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Variância explicada
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Número de componentes para 95% da variância
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        return {
            "n_components": len(explained_variance_ratio),
            "explained_variance_ratio": explained_variance_ratio.tolist(),
            "cumulative_variance": cumulative_variance.tolist(),
            "n_components_95_variance": int(n_components_95),
            "feature_loadings": pca.components_[:3].tolist() if len(pca.components_) >= 3 else pca.components_.tolist()
        }
    
    def export_analysis_report(self, format: str = "html") -> str:
        """Exporta relatório completo de análise"""
        if format.lower() == "html":
            return self._generate_html_report()
        elif format.lower() == "json":
            return json.dumps(self.generate_comprehensive_report(), indent=2)
        else:
            return "Formato não suportado. Use 'html' ou 'json'"
    
    def _generate_html_report(self) -> str:
        """Gera relatório HTML completo"""
        report_data = self.generate_comprehensive_report()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Análise de Dados - Team Agents</title>
            <meta charset="utf-8">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 10px; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .insight {{ margin: 10px 0; padding: 10px; background: #f8f9fa; border-left: 4px solid #007bff; }}
                .high-importance {{ border-left-color: #dc3545; }}
                .medium-importance {{ border-left-color: #ffc107; }}
                .low-importance {{ border-left-color: #28a745; }}
                .recommendation {{ margin: 10px 0; padding: 10px; background: #e7f3ff; border-left: 4px solid #17a2b8; }}
                table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Relatório de Análise Inteligente de Dados</h1>
                <p>Gerado automaticamente pelo Team Agents em {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Resumo dos Dados</h2>
                <table>
                    <tr><th>Métrica</th><th>Valor</th></tr>
                    <tr><td>Linhas</td><td>{report_data['summary']['rows']:,}</td></tr>
                    <tr><td>Colunas</td><td>{report_data['summary']['columns']}</td></tr>
                    <tr><td>Uso de Memória</td><td>{report_data['summary']['memory_usage']}</td></tr>
                    <tr><td>Score de Qualidade</td><td>{report_data['data_quality']['completeness_score']:.1f}%</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🎯 Principais Insights</h2>
        """
        
        # Adicionar insights
        for insight in report_data['insights'][:10]:  # Top 10 insights
            importance_class = ""
            if insight['importance'] > 0.7:
                importance_class = "high-importance"
            elif insight['importance'] > 0.4:
                importance_class = "medium-importance"
            else:
                importance_class = "low-importance"
            
            html_content += f"""
                <div class="insight {importance_class}">
                    <h4>{insight['title']}</h4>
                    <p>{insight['description']}</p>
                    <small>Importância: {insight['importance']:.2f} | Tipo: {insight['type']}</small>
                </div>
            """
        
        # Adicionar recomendações
        html_content += """
            </div>
            
            <div class="section">
                <h2>💡 Recomendações</h2>
        """
        
        for rec in report_data['recommendations']:
            html_content += f"""
                <div class="recommendation">
                    <h4>Prioridade: {rec['priority'].upper()}</h4>
                    <p>{rec['description']}</p>
                </div>
            """
        
        # Adicionar visualizações se disponíveis
        if self.visualizations:
            html_content += """
                </div>
                
                <div class="section">
                    <h2>📈 Visualizações</h2>
            """
            
            for viz_name, viz_html in self.visualizations.items():
                html_content += f"""
                    <h3>{viz_name.replace('_', ' ').title()}</h3>
                    {viz_html}
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content

# Função de conveniência
def analyze_data(data_source, target_column: str = None) -> IntelligentDataAnalytics:
    """Função de conveniência para análise rápida de dados"""
    analyzer = IntelligentDataAnalytics()
    
    if analyzer.load_data(data_source):
        # Realizar análises automáticas
        analyzer.generate_comprehensive_report()
        analyzer.create_advanced_visualizations()
        
        if target_column:
            analyzer.perform_machine_learning_analysis(target_column)
    
    return analyzer