# advanced_multilang_tools.py - Ferramentas Avançadas Multi-linguagem
"""
Sistema avançado de ferramentas que suporta múltiplas linguagens de programação
e fornece capacidades reais de automação, análise e desenvolvimento.
"""

import os
import subprocess
import tempfile
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re
import ast

class ProgrammingLanguage(Enum):
    """Linguagens de programação suportadas"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    BASH = "bash"
    POWERSHELL = "powershell"

@dataclass
class CodeResult:
    """Resultado da execução de código"""
    success: bool
    output: str
    error: str = ""
    execution_time: float = 0.0
    language: ProgrammingLanguage = None
    
class AdvancedMultiLangTools:
    """Ferramentas avançadas multi-linguagem para agentes"""
    
    def __init__(self):
        self.supported_languages = {
            ProgrammingLanguage.PYTHON: {
                'extension': '.py',
                'runner': 'python',
                'compiler': None,
                'template': self._get_python_template()
            },
            ProgrammingLanguage.JAVASCRIPT: {
                'extension': '.js',
                'runner': 'node',
                'compiler': None,
                'template': self._get_js_template()
            },
            ProgrammingLanguage.TYPESCRIPT: {
                'extension': '.ts',
                'runner': 'ts-node',
                'compiler': 'tsc',
                'template': self._get_ts_template()
            },
            ProgrammingLanguage.JAVA: {
                'extension': '.java',
                'runner': 'java',
                'compiler': 'javac',
                'template': self._get_java_template()
            },
            ProgrammingLanguage.CSHARP: {
                'extension': '.cs',
                'runner': 'dotnet run',
                'compiler': 'dotnet build',
                'template': self._get_csharp_template()
            },
            ProgrammingLanguage.CPP: {
                'extension': '.cpp',
                'runner': None,
                'compiler': 'g++',
                'template': self._get_cpp_template()
            },
            ProgrammingLanguage.GO: {
                'extension': '.go',
                'runner': 'go run',
                'compiler': 'go build',
                'template': self._get_go_template()
            },
            ProgrammingLanguage.RUST: {
                'extension': '.rs',
                'runner': None,
                'compiler': 'rustc',
                'template': self._get_rust_template()
            }
        }
        
    def _get_python_template(self) -> str:
        return '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generated Python code by Team Agents
"""

{imports}

def main():
    {code}

if __name__ == "__main__":
    main()
'''

    def _get_js_template(self) -> str:
        return '''/**
 * Generated JavaScript code by Team Agents
 */

{imports}

function main() {
    {code}
}

main();
'''

    def _get_ts_template(self) -> str:
        return '''/**
 * Generated TypeScript code by Team Agents
 */

{imports}

function main(): void {
    {code}
}

main();
'''

    def _get_java_template(self) -> str:
        return '''/**
 * Generated Java code by Team Agents
 */

{imports}

public class GeneratedCode {
    public static void main(String[] args) {
        {code}
    }
}
'''

    def _get_csharp_template(self) -> str:
        return '''/**
 * Generated C# code by Team Agents
 */

using System;
{imports}

namespace TeamAgents
{
    class GeneratedCode
    {
        static void Main(string[] args)
        {
            {code}
        }
    }
}
'''

    def _get_cpp_template(self) -> str:
        return '''/**
 * Generated C++ code by Team Agents
 */

#include <iostream>
{imports}

using namespace std;

int main() {
    {code}
    return 0;
}
'''

    def _get_go_template(self) -> str:
        return '''/**
 * Generated Go code by Team Agents
 */

package main

import (
    "fmt"
    {imports}
)

func main() {
    {code}
}
'''

    def _get_rust_template(self) -> str:
        return '''/**
 * Generated Rust code by Team Agents
 */

{imports}

fn main() {
    {code}
}
'''

    def generate_code(self, 
                     language: ProgrammingLanguage, 
                     task_description: str,
                     code_requirements: List[str] = None,
                     imports: List[str] = None) -> str:
        """Gera código na linguagem especificada"""
        
        if language not in self.supported_languages:
            raise ValueError(f"Linguagem {language} não suportada")
        
        template = self.supported_languages[language]['template']
        
        # Gerar código baseado na descrição da tarefa
        generated_code = self._generate_code_logic(language, task_description, code_requirements)
        
        # Formattar imports
        imports_str = self._format_imports(language, imports or [])
        
        # Substituir placeholders no template
        final_code = template.format(
            imports=imports_str,
            code=generated_code
        )
        
        return final_code
    
    def _generate_code_logic(self, 
                           language: ProgrammingLanguage, 
                           task_description: str,
                           requirements: List[str] = None) -> str:
        """Gera a lógica do código baseada na descrição"""
        
        # Análise inteligente da tarefa para gerar código adequado
        if 'api' in task_description.lower():
            return self._generate_api_code(language, task_description)
        elif 'database' in task_description.lower():
            return self._generate_database_code(language, task_description)
        elif 'web' in task_description.lower():
            return self._generate_web_code(language, task_description)
        elif 'data analysis' in task_description.lower():
            return self._generate_data_analysis_code(language, task_description)
        elif 'machine learning' in task_description.lower():
            return self._generate_ml_code(language, task_description)
        else:
            return self._generate_generic_code(language, task_description)
    
    def _generate_api_code(self, language: ProgrammingLanguage, description: str) -> str:
        """Gera código para APIs"""
        if language == ProgrammingLanguage.PYTHON:
            return '''# API Flask/FastAPI
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "API funcionando!", "data": []})

@app.route('/api/data', methods=['POST'])
def create_data():
    data = request.get_json()
    return jsonify({"message": "Dados recebidos", "received": data})

if __name__ == '__main__':
    app.run(debug=True)'''
        
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return '''// API Express.js
const express = require('express');
const app = express();

app.use(express.json());

app.get('/api/data', (req, res) => {
    res.json({ message: 'API funcionando!', data: [] });
});

app.post('/api/data', (req, res) => {
    const data = req.body;
    res.json({ message: 'Dados recebidos', received: data });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Servidor rodando na porta ${PORT}`);
});'''
        
        elif language == ProgrammingLanguage.GO:
            return '''// API Go
package main

import (
    "encoding/json"
    "net/http"
    "github.com/gorilla/mux"
)

type Response struct {
    Message string      `json:"message"`
    Data    interface{} `json:"data,omitempty"`
}

func getData(w http.ResponseWriter, r *http.Request) {
    response := Response{Message: "API funcionando!", Data: []string{}}
    json.NewEncoder(w).Encode(response)
}

func createData(w http.ResponseWriter, r *http.Request) {
    var data map[string]interface{}
    json.NewDecoder(r.Body).Decode(&data)
    
    response := Response{Message: "Dados recebidos", Data: data}
    json.NewEncoder(w).Encode(response)
}

func main() {
    r := mux.NewRouter()
    r.HandleFunc("/api/data", getData).Methods("GET")
    r.HandleFunc("/api/data", createData).Methods("POST")
    
    http.ListenAndServe(":8000", r)
}'''
        
        return "// Código API genérico"
    
    def _generate_database_code(self, language: ProgrammingLanguage, description: str) -> str:
        """Gera código para operações de banco de dados"""
        if language == ProgrammingLanguage.PYTHON:
            return '''# Operações de banco de dados
import sqlite3
import pandas as pd

class DatabaseManager:
    def __init__(self, db_path='data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def insert_data(self, name, value):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO data (name, value) VALUES (?, ?)", (name, value))
        conn.commit()
        conn.close()
    
    def get_all_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM data", conn)
        conn.close()
        return df

# Exemplo de uso
db = DatabaseManager()
db.insert_data("exemplo", 123.45)
data = db.get_all_data()
print(data)'''
        
        elif language == ProgrammingLanguage.JAVA:
            return '''// Operações de banco de dados Java
import java.sql.*;

public class DatabaseManager {
    private String dbUrl = "jdbc:sqlite:data.db";
    
    public void initDatabase() throws SQLException {
        Connection conn = DriverManager.getConnection(dbUrl);
        String sql = "CREATE TABLE IF NOT EXISTS data (" +
                    "id INTEGER PRIMARY KEY AUTOINCREMENT," +
                    "name TEXT NOT NULL," +
                    "value REAL," +
                    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)";
        
        Statement stmt = conn.createStatement();
        stmt.execute(sql);
        conn.close();
    }
    
    public void insertData(String name, double value) throws SQLException {
        Connection conn = DriverManager.getConnection(dbUrl);
        String sql = "INSERT INTO data (name, value) VALUES (?, ?)";
        
        PreparedStatement pstmt = conn.prepareStatement(sql);
        pstmt.setString(1, name);
        pstmt.setDouble(2, value);
        pstmt.executeUpdate();
        conn.close();
    }
    
    public void getAllData() throws SQLException {
        Connection conn = DriverManager.getConnection(dbUrl);
        String sql = "SELECT * FROM data";
        
        Statement stmt = conn.createStatement();
        ResultSet rs = stmt.executeQuery(sql);
        
        while (rs.next()) {
            System.out.println(rs.getInt("id") + " - " + 
                             rs.getString("name") + " - " + 
                             rs.getDouble("value"));
        }
        conn.close();
    }
}'''
        
        return "// Código de banco de dados genérico"
    
    def _generate_web_code(self, language: ProgrammingLanguage, description: str) -> str:
        """Gera código para aplicações web"""
        if language == ProgrammingLanguage.HTML:
            return '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aplicação Web - Team Agents</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #007bff; color: white; padding: 20px; border-radius: 8px; }
        .content { padding: 20px; background: #f8f9fa; margin-top: 20px; border-radius: 8px; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Team Agents - Aplicação Web</h1>
        </div>
        <div class="content">
            <h2>Dashboard</h2>
            <p>Aplicação web gerada automaticamente pelos agentes.</p>
            <button onclick="loadData()">Carregar Dados</button>
            <div id="results"></div>
        </div>
    </div>
    
    <script>
        function loadData() {
            document.getElementById('results').innerHTML = '<p>Carregando dados...</p>';
            // Simular carregamento de dados
            setTimeout(() => {
                document.getElementById('results').innerHTML = '<p>Dados carregados com sucesso!</p>';
            }, 1000);
        }
    </script>
</body>
</html>'''
        
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return '''// Aplicação web interativa
document.addEventListener('DOMContentLoaded', function() {
    
    class WebApp {
        constructor() {
            this.data = [];
            this.init();
        }
        
        init() {
            this.setupEventListeners();
            this.loadInitialData();
        }
        
        setupEventListeners() {
            document.getElementById('loadBtn').addEventListener('click', () => {
                this.loadData();
            });
            
            document.getElementById('processBtn').addEventListener('click', () => {
                this.processData();
            });
        }
        
        async loadData() {
            try {
                // Simular chamada API
                const response = await fetch('/api/data');
                this.data = await response.json();
                this.renderData();
            } catch (error) {
                console.error('Erro ao carregar dados:', error);
            }
        }
        
        processData() {
            // Processar dados
            const processed = this.data.map(item => ({
                ...item,
                processed: true,
                timestamp: new Date().toISOString()
            }));
            
            console.log('Dados processados:', processed);
        }
        
        renderData() {
            const container = document.getElementById('dataContainer');
            container.innerHTML = this.data.map(item => 
                `<div class="data-item">${JSON.stringify(item)}</div>`
            ).join('');
        }
        
        loadInitialData() {
            this.data = [
                { id: 1, name: 'Item 1', value: 100 },
                { id: 2, name: 'Item 2', value: 200 }
            ];
            this.renderData();
        }
    }
    
    // Inicializar aplicação
    const app = new WebApp();
});'''
        
        return "// Código web genérico"
    
    def _generate_data_analysis_code(self, language: ProgrammingLanguage, description: str) -> str:
        """Gera código para análise de dados"""
        if language == ProgrammingLanguage.PYTHON:
            return '''# Análise de dados avançada
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class DataAnalyzer:
    def __init__(self, data_path=None):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, path):
        """Carrega dados de diferentes formatos"""
        if path.endswith('.csv'):
            self.data = pd.read_csv(path)
        elif path.endswith('.json'):
            self.data = pd.read_json(path)
        elif path.endswith('.xlsx'):
            self.data = pd.read_excel(path)
        else:
            raise ValueError("Formato não suportado")
    
    def exploratory_analysis(self):
        """Análise exploratória dos dados"""
        if self.data is None:
            return "Nenhum dado carregado"
        
        analysis = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'statistics': self.data.describe().to_dict()
        }
        
        return analysis
    
    def create_visualizations(self):
        """Cria visualizações dos dados"""
        if self.data is None:
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histograma
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.data[numeric_cols[0]].hist(ax=axes[0, 0], bins=30)
            axes[0, 0].set_title(f'Distribuição de {numeric_cols[0]}')
        
        # Boxplot
        if len(numeric_cols) > 1:
            self.data.boxplot(column=numeric_cols[:4], ax=axes[0, 1])
            axes[0, 1].set_title('Boxplots das Variáveis Numéricas')
        
        # Correlação
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, ax=axes[1, 0])
            axes[1, 0].set_title('Matriz de Correlação')
        
        # Scatter plot
        if len(numeric_cols) > 1:
            axes[1, 1].scatter(self.data[numeric_cols[0]], self.data[numeric_cols[1]])
            axes[1, 1].set_xlabel(numeric_cols[0])
            axes[1, 1].set_ylabel(numeric_cols[1])
            axes[1, 1].set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def build_predictive_model(self, target_column):
        """Constrói modelo preditivo"""
        if self.data is None or target_column not in self.data.columns:
            return "Dados ou coluna alvo inválidos"
        
        # Preparar dados
        X = self.data.select_dtypes(include=[np.number]).drop(columns=[target_column])
        y = self.data[target_column]
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Normalizar dados
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Treinar modelo
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results = {
            'mse': mse,
            'r2_score': r2,
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
        }
        
        return results
    
    def generate_insights(self):
        """Gera insights automaticamente"""
        if self.data is None:
            return []
        
        insights = []
        
        # Insights sobre valores ausentes
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            insights.append(f"Dataset tem {missing.sum()} valores ausentes em {(missing > 0).sum()} colunas")
        
        # Insights sobre correlações
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.data[numeric_cols].corr()
            high_corr = np.where(np.abs(corr_matrix) > 0.8)
            for i, j in zip(high_corr[0], high_corr[1]):
                if i != j:
                    insights.append(f"Alta correlação entre {numeric_cols[i]} e {numeric_cols[j]}: {corr_matrix.iloc[i, j]:.2f}")
        
        # Insights sobre outliers
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[col] < (Q1 - 1.5 * IQR)) | (self.data[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                insights.append(f"Coluna {col} tem {outliers} possíveis outliers")
        
        return insights

# Exemplo de uso
analyzer = DataAnalyzer()
# analyzer.load_data('dados.csv')
# insights = analyzer.generate_insights()
# print("Insights encontrados:", insights)'''
        
        elif language == ProgrammingLanguage.R:
            return '''# Análise de dados em R
library(tidyverse)
library(ggplot2)
library(corrplot)
library(randomForest)
library(caret)

# Classe para análise de dados
DataAnalyzer <- R6Class("DataAnalyzer",
  public = list(
    data = NULL,
    model = NULL,
    
    initialize = function(data_path = NULL) {
      if (!is.null(data_path)) {
        self$load_data(data_path)
      }
    },
    
    load_data = function(path) {
      if (str_ends(path, ".csv")) {
        self$data <- read.csv(path)
      } else if (str_ends(path, ".xlsx")) {
        self$data <- readxl::read_excel(path)
      } else {
        stop("Formato não suportado")
      }
    },
    
    exploratory_analysis = function() {
      if (is.null(self$data)) {
        return("Nenhum dado carregado")
      }
      
      list(
        dimensions = dim(self$data),
        columns = colnames(self$data),
        summary = summary(self$data),
        missing_values = sapply(self$data, function(x) sum(is.na(x)))
      )
    },
    
    create_visualizations = function() {
      if (is.null(self$data)) return()
      
      numeric_cols <- select_if(self$data, is.numeric)
      
      # Histograma
      p1 <- ggplot(self$data, aes_string(x = names(numeric_cols)[1])) +
        geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
        theme_minimal() +
        ggtitle(paste("Distribuição de", names(numeric_cols)[1]))
      
      # Boxplot
      p2 <- self$data %>%
        select_if(is.numeric) %>%
        pivot_longer(everything()) %>%
        ggplot(aes(x = name, y = value)) +
        geom_boxplot(fill = "lightblue") +
        theme_minimal() +
        ggtitle("Boxplots das Variáveis Numéricas") +
        theme(axis.text.x = element_text(angle = 45))
      
      # Matriz de correlação
      if (ncol(numeric_cols) > 1) {
        corr_matrix <- cor(numeric_cols, use = "complete.obs")
        corrplot(corr_matrix, method = "color", type = "upper", 
                 tl.cex = 0.8, tl.col = "black")
      }
      
      print(p1)
      print(p2)
    },
    
    build_predictive_model = function(target_column) {
      if (is.null(self$data) || !target_column %in% colnames(self$data)) {
        return("Dados ou coluna alvo inválidos")
      }
      
      # Preparar dados
      numeric_data <- select_if(self$data, is.numeric)
      formula_str <- paste(target_column, "~ .")
      
      # Dividir dados
      set.seed(42)
      train_indices <- createDataPartition(numeric_data[[target_column]], p = 0.8, list = FALSE)
      train_data <- numeric_data[train_indices, ]
      test_data <- numeric_data[-train_indices, ]
      
      # Treinar modelo
      self$model <- randomForest(as.formula(formula_str), data = train_data, ntree = 100)
      
      # Avaliar modelo
      predictions <- predict(self$model, test_data)
      mse <- mean((test_data[[target_column]] - predictions)^2)
      r2 <- cor(test_data[[target_column]], predictions)^2
      
      list(
        mse = mse,
        r2_score = r2,
        feature_importance = importance(self$model)
      )
    },
    
    generate_insights = function() {
      if (is.null(self$data)) return(c())
      
      insights <- c()
      
      # Valores ausentes
      missing_counts <- sapply(self$data, function(x) sum(is.na(x)))
      total_missing <- sum(missing_counts)
      if (total_missing > 0) {
        insights <- c(insights, paste("Dataset tem", total_missing, "valores ausentes"))
      }
      
      # Correlações altas
      numeric_cols <- select_if(self$data, is.numeric)
      if (ncol(numeric_cols) > 1) {
        corr_matrix <- cor(numeric_cols, use = "complete.obs")
        high_corr <- which(abs(corr_matrix) > 0.8 & corr_matrix != 1, arr.ind = TRUE)
        if (nrow(high_corr) > 0) {
          for (i in 1:nrow(high_corr)) {
            insights <- c(insights, paste("Alta correlação entre", 
                                        colnames(corr_matrix)[high_corr[i, 1]], "e", 
                                        colnames(corr_matrix)[high_corr[i, 2]]))
          }
        }
      }
      
      insights
    }
  )
)

# Exemplo de uso
# analyzer <- DataAnalyzer$new()
# analyzer$load_data("dados.csv")
# insights <- analyzer$generate_insights()
# print(insights)'''
        
        return "# Código de análise de dados genérico"
    
    def _generate_ml_code(self, language: ProgrammingLanguage, description: str) -> str:
        """Gera código para machine learning"""
        if language == ProgrammingLanguage.PYTHON:
            return '''# Sistema de Machine Learning avançado
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

class MLPipeline:
    def __init__(self, task_type='classification'):
        self.task_type = task_type  # 'classification' ou 'regression'
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, X, y):
        """Prepara os dados para treinamento"""
        # Tratar valores ausentes
        X = X.fillna(X.mean() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        
        # Codificar variáveis categóricas
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Codificar target se necessário
        if self.task_type == 'classification' and y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
        
        return X_scaled, y
    
    def train_models(self, X_train, y_train):
        """Treina múltiplos modelos"""
        if self.task_type == 'classification':
            self.models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42),
                'SVM': SVC(random_state=42)
            }
        else:
            self.models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'GradientBoosting': GradientBoostingRegressor(random_state=42),
                'SVR': SVR()
            }
        
        # Treinar cada modelo
        trained_models = {}
        for name, model in self.models.items():
            print(f"Treinando {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Avalia todos os modelos"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            if self.task_type == 'classification':
                # Métricas de classificação
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted')
                }
            else:
                # Métricas de regressão
                results[name] = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2_score': r2_score(y_test, y_pred)
                }
        
        return results
    
    def hyperparameter_tuning(self, X_train, y_train, model_name):
        """Otimização de hiperparâmetros"""
        if model_name == 'RandomForest':
            if self.task_type == 'classification':
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
        
        elif model_name == 'SVM' or model_name == 'SVR':
            model = SVC(random_state=42) if self.task_type == 'classification' else SVR()
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        
        else:
            return None
        
        # Grid search
        grid_search = GridSearchCV(model, param_grid, cv=5, 
                                 scoring='accuracy' if self.task_type == 'classification' else 'r2',
                                 n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def select_best_model(self, results):
        """Seleciona o melhor modelo baseado nas métricas"""
        if self.task_type == 'classification':
            best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
            best_score = results[best_model_name]['f1_score']
        else:
            best_model_name = max(results.keys(), key=lambda k: results[k]['r2_score'])
            best_score = results[best_model_name]['r2_score']
        
        self.best_model = self.models[best_model_name]
        return best_model_name, best_score
    
    def save_model(self, filepath):
        """Salva o melhor modelo"""
        if self.best_model is not None:
            joblib.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder if hasattr(self, 'label_encoder') else None,
                'task_type': self.task_type
            }, filepath)
            print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath):
        """Carrega um modelo salvo"""
        saved_data = joblib.load(filepath)
        self.best_model = saved_data['model']
        self.scaler = saved_data['scaler']
        if saved_data['label_encoder']:
            self.label_encoder = saved_data['label_encoder']
        self.task_type = saved_data['task_type']
        print(f"Modelo carregado de: {filepath}")
    
    def predict(self, X):
        """Faz predições com o melhor modelo"""
        if self.best_model is None:
            raise ValueError("Nenhum modelo treinado. Execute fit primeiro.")
        
        # Preparar dados da mesma forma que no treinamento
        X_processed = self.scaler.transform(X)
        predictions = self.best_model.predict(X_processed)
        
        # Decodificar se necessário
        if (self.task_type == 'classification' and 
            hasattr(self, 'label_encoder') and 
            self.label_encoder is not None):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def feature_importance(self):
        """Retorna importância das features"""
        if (self.best_model is not None and 
            hasattr(self.best_model, 'feature_importances_')):
            return self.best_model.feature_importances_
        return None
    
    def full_pipeline(self, X, y, test_size=0.2):
        """Pipeline completo de ML"""
        print("Iniciando pipeline completo de Machine Learning...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Preparar dados
        print("Preparando dados...")
        X_train_processed, y_train_processed = self.prepare_data(X_train, y_train)
        X_test_processed = self.scaler.transform(X_test)
        if self.task_type == 'classification' and hasattr(self, 'label_encoder'):
            y_test_processed = self.label_encoder.transform(y_test)
        else:
            y_test_processed = y_test
        
        # Treinar modelos
        print("Treinando modelos...")
        self.train_models(X_train_processed, y_train_processed)
        
        # Avaliar modelos
        print("Avaliando modelos...")
        results = self.evaluate_models(X_test_processed, y_test_processed)
        
        # Selecionar melhor modelo
        best_name, best_score = self.select_best_model(results)
        print(f"Melhor modelo: {best_name} com score: {best_score:.4f}")
        
        # Otimizar hiperparâmetros do melhor modelo
        print("Otimizando hiperparâmetros...")
        optimized_model, best_params = self.hyperparameter_tuning(X_train_processed, y_train_processed, best_name)
        if optimized_model:
            self.best_model = optimized_model
            print(f"Melhores parâmetros: {best_params}")
        
        return {
            'results': results,
            'best_model': best_name,
            'best_score': best_score,
            'best_params': best_params if 'best_params' in locals() else None
        }

# Exemplo de uso
# ml_pipeline = MLPipeline(task_type='classification')
# results = ml_pipeline.full_pipeline(X, y)
# ml_pipeline.save_model('modelo_treinado.pkl')'''
        
        return "# Código de ML genérico"
    
    def _generate_generic_code(self, language: ProgrammingLanguage, description: str) -> str:
        """Gera código genérico baseado na descrição"""
        return f"""// Código gerado automaticamente para: {description}
// Linguagem: {language.value}

// TODO: Implementar lógica específica baseada nos requisitos
console.log("Código gerado com sucesso!");
"""
    
    def _format_imports(self, language: ProgrammingLanguage, imports: List[str]) -> str:
        """Formata imports específicos para cada linguagem"""
        if not imports:
            return ""
        
        if language == ProgrammingLanguage.PYTHON:
            return "\n".join(imports)
        elif language == ProgrammingLanguage.JAVASCRIPT:
            return "\n".join(f"const {imp} = require('{imp}');" for imp in imports)
        elif language == ProgrammingLanguage.JAVA:
            return "\n".join(f"import {imp};" for imp in imports)
        elif language == ProgrammingLanguage.CSHARP:
            return "\n".join(f"using {imp};" for imp in imports)
        elif language == ProgrammingLanguage.CPP:
            return "\n".join(f"#include <{imp}>" for imp in imports)
        elif language == ProgrammingLanguage.GO:
            return '    "' + '"\n    "'.join(imports) + '"' if imports else ""
        
        return "\n".join(imports)
    
    def execute_code(self, 
                    code: str, 
                    language: ProgrammingLanguage,
                    timeout: int = 30) -> CodeResult:
        """Executa código na linguagem especificada"""
        
        if language not in self.supported_languages:
            return CodeResult(
                success=False,
                output="",
                error=f"Linguagem {language} não suportada",
                language=language
            )
        
        lang_config = self.supported_languages[language]
        
        try:
            # Criar arquivo temporário
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=lang_config['extension'], 
                delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Compilar se necessário
            if lang_config['compiler']:
                compile_cmd = f"{lang_config['compiler']} {temp_file_path}"
                compile_result = subprocess.run(
                    compile_cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True,
                    timeout=timeout
                )
                
                if compile_result.returncode != 0:
                    return CodeResult(
                        success=False,
                        output="",
                        error=f"Erro de compilação: {compile_result.stderr}",
                        language=language
                    )
            
            # Executar código
            if lang_config['runner']:
                if language == ProgrammingLanguage.CPP:
                    # Para C++, executar o binário compilado
                    executable = temp_file_path.replace('.cpp', '.exe' if os.name == 'nt' else '')
                    run_cmd = executable
                else:
                    run_cmd = f"{lang_config['runner']} {temp_file_path}"
                
                result = subprocess.run(
                    run_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                return CodeResult(
                    success=result.returncode == 0,
                    output=result.stdout,
                    error=result.stderr,
                    language=language
                )
            
            else:
                return CodeResult(
                    success=False,
                    output="",
                    error="Execução não suportada para esta linguagem",
                    language=language
                )
        
        except subprocess.TimeoutExpired:
            return CodeResult(
                success=False,
                output="",
                error=f"Timeout: Execução excedeu {timeout} segundos",
                language=language
            )
        
        except Exception as e:
            return CodeResult(
                success=False,
                output="",
                error=f"Erro na execução: {str(e)}",
                language=language
            )
        
        finally:
            # Limpar arquivos temporários
            try:
                if 'temp_file_path' in locals():
                    os.unlink(temp_file_path)
                    # Limpar executável se existir
                    if language == ProgrammingLanguage.CPP:
                        exe_path = temp_file_path.replace('.cpp', '.exe' if os.name == 'nt' else '')
                        if os.path.exists(exe_path):
                            os.unlink(exe_path)
            except:
                pass
    
    def get_supported_languages(self) -> List[ProgrammingLanguage]:
        """Retorna lista de linguagens suportadas"""
        return list(self.supported_languages.keys())
    
    def get_language_info(self, language: ProgrammingLanguage) -> Dict[str, Any]:
        """Retorna informações sobre uma linguagem específica"""
        if language in self.supported_languages:
            return self.supported_languages[language]
        return {}
    
    def suggest_language(self, task_description: str) -> ProgrammingLanguage:
        """Sugere a melhor linguagem para uma tarefa"""
        task_lower = task_description.lower()
        
        # Mapeamento de palavras-chave para linguagens
        language_keywords = {
            ProgrammingLanguage.PYTHON: [
                'data science', 'machine learning', 'ai', 'analytics', 
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch'
            ],
            ProgrammingLanguage.JAVASCRIPT: [
                'web', 'frontend', 'react', 'vue', 'angular', 'node',
                'web development', 'browser', 'dom'
            ],
            ProgrammingLanguage.JAVA: [
                'enterprise', 'spring', 'android', 'backend',
                'microservices', 'scalable'
            ],
            ProgrammingLanguage.CSHARP: [
                'microsoft', '.net', 'windows', 'enterprise',
                'desktop application'
            ],
            ProgrammingLanguage.CPP: [
                'performance', 'system programming', 'game engine',
                'embedded', 'real-time'
            ],
            ProgrammingLanguage.GO: [
                'microservices', 'concurrent', 'api', 'backend',
                'scalable', 'cloud'
            ],
            ProgrammingLanguage.RUST: [
                'system programming', 'performance', 'memory safe',
                'concurrent', 'low-level'
            ],
            ProgrammingLanguage.R: [
                'statistics', 'data analysis', 'research',
                'statistical modeling', 'visualization'
            ]
        }
        
        # Calcular scores para cada linguagem
        scores = {}
        for language, keywords in language_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                scores[language] = score
        
        # Retornar linguagem com maior score ou Python como padrão
        if scores:
            return max(scores, key=scores.get)
        
        return ProgrammingLanguage.PYTHON  # Padrão