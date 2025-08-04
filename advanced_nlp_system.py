# advanced_nlp_system.py - Sistema Avançado de Processamento de Linguagem Natural
"""
Sistema completo de NLP que inclui resumos automáticos, análise de sentimento,
tradução, extração de entidades, análise semântica e processamento de texto avançado.
"""

import re
import json
import unicodedata
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import math
from collections import Counter, defaultdict
import string

# Imports condicionais para bibliotecas de NLP
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    from nltk.sentiment import SentimentIntensityAnalyzer
    
    # Download de recursos necessários (apenas uma vez)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    
    try:
        nltk.data.find('averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    
    try:
        nltk.data.find('maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker', quiet=True)
    
    try:
        nltk.data.find('words')
    except LookupError:
        nltk.download('words', quiet=True)
    
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    # Verificar se modelo está disponível
    try:
        nlp = spacy.load("pt_core_news_sm")
        SPACY_PT_AVAILABLE = True
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
            SPACY_EN_AVAILABLE = True
            SPACY_PT_AVAILABLE = False
        except OSError:
            SPACY_EN_AVAILABLE = False
            SPACY_PT_AVAILABLE = False
except ImportError:
    SPACY_EN_AVAILABLE = False
    SPACY_PT_AVAILABLE = False

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

class Language(Enum):
    """Idiomas suportados"""
    PORTUGUESE = "pt"
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    AUTO = "auto"

class SentimentType(Enum):
    """Tipos de sentimento"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"

@dataclass
class TextSummary:
    """Resultado de resumo de texto"""
    original_length: int
    summary_length: int
    compression_ratio: float
    summary: str
    key_sentences: List[str]
    keywords: List[str]
    sentiment: str
    reading_level: str

@dataclass
class SentimentAnalysis:
    """Resultado de análise de sentimento"""
    overall_sentiment: SentimentType
    confidence: float
    positive_score: float
    negative_score: float
    neutral_score: float
    emotion_breakdown: Dict[str, float]

@dataclass
class EntityExtraction:
    """Resultado de extração de entidades"""
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    dates: List[str]
    numbers: List[str]
    emails: List[str]
    urls: List[str]
    phones: List[str]

@dataclass
class TextAnalytics:
    """Resultado completo de análise de texto"""
    language: str
    word_count: int
    sentence_count: int
    paragraph_count: int
    avg_sentence_length: float
    avg_word_length: float
    readability_score: float
    keywords: List[Tuple[str, float]]
    sentiment: SentimentAnalysis
    entities: EntityExtraction
    summary: TextSummary

class AdvancedNLPSystem:
    """Sistema avançado de processamento de linguagem natural"""
    
    def __init__(self):
        self.stemmer = PorterStemmer() if NLTK_AVAILABLE else None
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if NLTK_AVAILABLE else None
        
        # Stopwords para diferentes idiomas
        self.stopwords_dict = {}
        if NLTK_AVAILABLE:
            try:
                self.stopwords_dict['pt'] = set(stopwords.words('portuguese'))
                self.stopwords_dict['en'] = set(stopwords.words('english'))
                self.stopwords_dict['es'] = set(stopwords.words('spanish'))
                self.stopwords_dict['fr'] = set(stopwords.words('french'))
                self.stopwords_dict['de'] = set(stopwords.words('german'))
                self.stopwords_dict['it'] = set(stopwords.words('italian'))
            except:
                # Fallback para stopwords básicas
                self.stopwords_dict['pt'] = {'a', 'o', 'de', 'da', 'do', 'e', 'em', 'um', 'uma', 'com', 'para', 'por', 'se', 'que', 'não', 'na', 'no'}
                self.stopwords_dict['en'] = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        # Padrões regex para extração de entidades
        self.regex_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'phone': r'(\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
            'number': r'\b\d+(?:\.\d+)?\b',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{2,4}[/-]\d{1,2}[/-]\d{1,2}\b'
        }
        
        # Dicionários de tradução básica (para casos sem bibliotecas especializadas)
        self.basic_translations = {
            'pt_to_en': {
                'olá': 'hello', 'mundo': 'world', 'casa': 'house', 'carro': 'car',
                'água': 'water', 'comida': 'food', 'livro': 'book', 'trabalho': 'work',
                'família': 'family', 'amigo': 'friend', 'amor': 'love', 'tempo': 'time'
            },
            'en_to_pt': {
                'hello': 'olá', 'world': 'mundo', 'house': 'casa', 'car': 'carro',
                'water': 'água', 'food': 'comida', 'book': 'livro', 'work': 'trabalho',
                'family': 'família', 'friend': 'amigo', 'love': 'amor', 'time': 'tempo'
            }
        }
    
    def detect_language(self, text: str) -> str:
        """Detecta o idioma do texto"""
        if not text or len(text.strip()) < 10:
            return Language.ENGLISH.value
        
        # Análise simples baseada em características do idioma
        text_lower = text.lower()
        
        # Palavras características por idioma
        language_indicators = {
            'pt': ['que', 'não', 'com', 'uma', 'para', 'esse', 'isso', 'mais', 'muito', 'como', 'também', 'ser', 'ter', 'fazer'],
            'en': ['the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but', 'his', 'from', 'they', 'she'],
            'es': ['que', 'de', 'no', 'la', 'el', 'en', 'y', 'a', 'ser', 'se', 'te', 'todo', 'le', 'da'],
            'fr': ['que', 'de', 'et', 'le', 'la', 'les', 'des', 'un', 'une', 'dans', 'pour', 'avec', 'sur', 'par'],
            'de': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist'],
            'it': ['che', 'di', 'la', 'il', 'e', 'in', 'un', 'è', 'per', 'con', 'non', 'una', 'su', 'le']
        }
        
        scores = {}
        words = text_lower.split()
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for word in words if word in indicators)
            scores[lang] = score / len(words) if words else 0
        
        # Retornar idioma com maior score
        detected_lang = max(scores, key=scores.get) if scores else 'en'
        return detected_lang
    
    def clean_text(self, text: str, language: str = None) -> str:
        """Limpa e normaliza texto"""
        if not text:
            return ""
        
        # Normalizar unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remover caracteres de controle
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Normalizar espaços em branco
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remover caracteres especiais desnecessários (manter pontuação básica)
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'']', ' ', text)
        
        # Normalizar pontuação
        text = re.sub(r'\.{2,}', '...', text)  # Múltiplos pontos
        text = re.sub(r'\!{2,}', '!', text)    # Múltiplas exclamações
        text = re.sub(r'\?{2,}', '?', text)    # Múltiplas interrogações
        
        return text.strip()
    
    def extract_keywords(self, text: str, language: str = None, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """Extrai palavras-chave usando TF-IDF simplificado"""
        if not text:
            return []
        
        if language is None:
            language = self.detect_language(text)
        
        # Tokenizar e limpar
        if NLTK_AVAILABLE:
            words = word_tokenize(text.lower())
        else:
            words = text.lower().split()
        
        # Remover stopwords e pontuação
        stopwords_set = self.stopwords_dict.get(language, set())
        cleaned_words = [
            word for word in words 
            if word not in stopwords_set 
            and word not in string.punctuation 
            and len(word) > 2
            and word.isalpha()
        ]
        
        if not cleaned_words:
            return []
        
        # Calcular frequências
        word_freq = Counter(cleaned_words)
        
        # Calcular TF-IDF simplificado
        total_words = len(cleaned_words)
        tfidf_scores = {}
        
        for word, freq in word_freq.items():
            tf = freq / total_words
            # IDF simplificado baseado na raridade da palavra
            idf = math.log(total_words / freq)
            tfidf_scores[word] = tf * idf
        
        # Retornar top keywords
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:max_keywords]
    
    def extract_entities(self, text: str, language: str = None) -> EntityExtraction:
        """Extrai entidades nomeadas do texto"""
        entities = EntityExtraction(
            persons=[], organizations=[], locations=[],
            dates=[], numbers=[], emails=[], urls=[], phones=[]
        )
        
        if not text:
            return entities
        
        # Extrair usando regex
        entities.emails = re.findall(self.regex_patterns['email'], text)
        entities.urls = re.findall(self.regex_patterns['url'], text)
        entities.phones = re.findall(self.regex_patterns['phone'], text)
        entities.numbers = re.findall(self.regex_patterns['number'], text)
        entities.dates = re.findall(self.regex_patterns['date'], text)
        
        # Usar NLTK se disponível
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text)
                pos_tags = pos_tag(tokens)
                named_entities = ne_chunk(pos_tags)
                
                for chunk in named_entities:
                    if hasattr(chunk, 'label'):
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        if chunk.label() == 'PERSON':
                            entities.persons.append(entity_text)
                        elif chunk.label() == 'ORGANIZATION':
                            entities.organizations.append(entity_text)
                        elif chunk.label() in ['GPE', 'LOCATION']:
                            entities.locations.append(entity_text)
            except:
                pass
        
        # Usar spaCy se disponível
        if SPACY_PT_AVAILABLE or SPACY_EN_AVAILABLE:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        entities.persons.append(ent.text)
                    elif ent.label_ == 'ORG':
                        entities.organizations.append(ent.text)
                    elif ent.label_ in ['GPE', 'LOC']:
                        entities.locations.append(ent.text)
            except:
                pass
        
        # Remover duplicatas
        entities.persons = list(set(entities.persons))
        entities.organizations = list(set(entities.organizations))
        entities.locations = list(set(entities.locations))
        entities.emails = list(set(entities.emails))
        entities.urls = list(set(entities.urls))
        entities.phones = list(set(entities.phones))
        
        return entities
    
    def analyze_sentiment(self, text: str, language: str = None) -> SentimentAnalysis:
        """Analisa sentimento do texto"""
        if not text:
            return SentimentAnalysis(
                overall_sentiment=SentimentType.NEUTRAL,
                confidence=0.0,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                emotion_breakdown={}
            )
        
        # Usar VADER (NLTK) se disponível
        if NLTK_AVAILABLE and self.sentiment_analyzer:
            try:
                scores = self.sentiment_analyzer.polarity_scores(text)
                
                # Determinar sentimento geral
                if scores['compound'] >= 0.05:
                    overall = SentimentType.POSITIVE
                elif scores['compound'] <= -0.05:
                    overall = SentimentType.NEGATIVE
                else:
                    overall = SentimentType.NEUTRAL
                
                return SentimentAnalysis(
                    overall_sentiment=overall,
                    confidence=abs(scores['compound']),
                    positive_score=scores['pos'],
                    negative_score=scores['neg'],
                    neutral_score=scores['neu'],
                    emotion_breakdown={
                        'joy': scores['pos'] * 0.8,
                        'anger': scores['neg'] * 0.6,
                        'sadness': scores['neg'] * 0.4,
                        'fear': scores['neg'] * 0.3,
                        'surprise': abs(scores['compound']) * 0.2
                    }
                )
            except:
                pass
        
        # Análise de sentimento simplificada
        positive_words = {
            'pt': ['bom', 'ótimo', 'excelente', 'maravilhoso', 'perfeito', 'feliz', 'amor', 'alegria', 'sucesso', 'positivo'],
            'en': ['good', 'great', 'excellent', 'wonderful', 'perfect', 'happy', 'love', 'joy', 'success', 'positive'],
            'es': ['bueno', 'excelente', 'maravilloso', 'perfecto', 'feliz', 'amor', 'alegría', 'éxito', 'positivo'],
            'fr': ['bon', 'excellent', 'merveilleux', 'parfait', 'heureux', 'amour', 'joie', 'succès', 'positif'],
            'de': ['gut', 'ausgezeichnet', 'wunderbar', 'perfekt', 'glücklich', 'liebe', 'freude', 'erfolg', 'positiv'],
            'it': ['buono', 'eccellente', 'meraviglioso', 'perfetto', 'felice', 'amore', 'gioia', 'successo', 'positivo']
        }
        
        negative_words = {
            'pt': ['ruim', 'péssimo', 'terrível', 'horrível', 'triste', 'ódio', 'raiva', 'fracasso', 'negativo', 'problema'],
            'en': ['bad', 'terrible', 'horrible', 'awful', 'sad', 'hate', 'anger', 'failure', 'negative', 'problem'],
            'es': ['malo', 'terrible', 'horrible', 'espantoso', 'triste', 'odio', 'ira', 'fracaso', 'negativo', 'problema'],
            'fr': ['mauvais', 'terrible', 'horrible', 'affreux', 'triste', 'haine', 'colère', 'échec', 'négatif', 'problème'],
            'de': ['schlecht', 'schrecklich', 'furchtbar', 'traurig', 'hass', 'wut', 'versagen', 'negativ', 'problem'],
            'it': ['cattivo', 'terribile', 'orribile', 'triste', 'odio', 'rabbia', 'fallimento', 'negativo', 'problema']
        }
        
        if language is None:
            language = self.detect_language(text)
        
        text_lower = text.lower()
        words = text_lower.split()
        
        pos_words = positive_words.get(language, positive_words['en'])
        neg_words = negative_words.get(language, negative_words['en'])
        
        pos_count = sum(1 for word in words if word in pos_words)
        neg_count = sum(1 for word in words if word in neg_words)
        
        total_sentiment_words = pos_count + neg_count
        
        if total_sentiment_words == 0:
            return SentimentAnalysis(
                overall_sentiment=SentimentType.NEUTRAL,
                confidence=0.5,
                positive_score=0.0,
                negative_score=0.0,
                neutral_score=1.0,
                emotion_breakdown={}
            )
        
        pos_score = pos_count / len(words)
        neg_score = neg_count / len(words)
        
        if pos_score > neg_score:
            overall = SentimentType.POSITIVE
            confidence = pos_score / (pos_score + neg_score)
        elif neg_score > pos_score:
            overall = SentimentType.NEGATIVE
            confidence = neg_score / (pos_score + neg_score)
        else:
            overall = SentimentType.NEUTRAL
            confidence = 0.5
        
        return SentimentAnalysis(
            overall_sentiment=overall,
            confidence=confidence,
            positive_score=pos_score,
            negative_score=neg_score,
            neutral_score=1.0 - pos_score - neg_score,
            emotion_breakdown={
                'joy': pos_score * 0.8,
                'anger': neg_score * 0.6,
                'sadness': neg_score * 0.4
            }
        )
    
    def summarize_text(self, text: str, max_sentences: int = 3, language: str = None) -> TextSummary:
        """Cria resumo automático do texto"""
        if not text or len(text.strip()) < 50:
            return TextSummary(
                original_length=len(text),
                summary_length=len(text),
                compression_ratio=1.0,
                summary=text,
                key_sentences=[text] if text else [],
                keywords=[],
                sentiment="neutral",
                reading_level="unknown"
            )
        
        if language is None:
            language = self.detect_language(text)
        
        # Tokenizar sentenças
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Fallback simples para separação de sentenças
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            keywords = self.extract_keywords(text, language)
            sentiment = self.analyze_sentiment(text, language)
            
            return TextSummary(
                original_length=len(text),
                summary_length=len(text),
                compression_ratio=1.0,
                summary=text,
                key_sentences=sentences,
                keywords=[kw[0] for kw in keywords[:5]],
                sentiment=sentiment.overall_sentiment.value,
                reading_level=self._calculate_reading_level(text)
            )
        
        # Calcular pontuação das sentenças
        sentence_scores = {}
        keywords = self.extract_keywords(text, language, max_keywords=20)
        keyword_dict = dict(keywords)
        
        for i, sentence in enumerate(sentences):
            if NLTK_AVAILABLE:
                words = word_tokenize(sentence.lower())
            else:
                words = sentence.lower().split()
            
            # Score baseado em palavras-chave
            keyword_score = sum(keyword_dict.get(word, 0) for word in words)
            
            # Score baseado na posição (primeiras e últimas sentenças são mais importantes)
            position_score = 0
            if i < len(sentences) * 0.3:  # Primeiro terço
                position_score = 0.3
            elif i > len(sentences) * 0.7:  # Último terço
                position_score = 0.2
            
            # Score baseado no comprimento (sentenças muito curtas ou longas são penalizadas)
            length_score = 0
            word_count = len(words)
            if 10 <= word_count <= 30:
                length_score = 0.2
            elif 5 <= word_count <= 50:
                length_score = 0.1
            
            sentence_scores[i] = keyword_score + position_score + length_score
        
        # Selecionar melhores sentenças
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:max_sentences]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Ordenar por posição original
        
        summary_sentences = [sentences[i] for i, _ in top_sentences]
        summary = ' '.join(summary_sentences)
        
        sentiment = self.analyze_sentiment(summary, language)
        
        return TextSummary(
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=len(summary) / len(text),
            summary=summary,
            key_sentences=summary_sentences,
            keywords=[kw[0] for kw in keywords[:10]],
            sentiment=sentiment.overall_sentiment.value,
            reading_level=self._calculate_reading_level(text)
        )
    
    def _calculate_reading_level(self, text: str) -> str:
        """Calcula nível de leitura do texto"""
        if TEXTSTAT_AVAILABLE:
            try:
                ease_score = flesch_reading_ease(text)
                if ease_score >= 90:
                    return "muito_facil"
                elif ease_score >= 80:
                    return "facil"
                elif ease_score >= 70:
                    return "medio_facil"  
                elif ease_score >= 60:
                    return "medio"
                elif ease_score >= 50:
                    return "medio_dificil"
                elif ease_score >= 30:
                    return "dificil"
                else:
                    return "muito_dificil"
            except:
                pass
        
        # Cálculo simplificado
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+', text)
            words = text.split()
        
        if not sentences or not words:
            return "unknown"
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Heurística simples
        if avg_sentence_length < 15 and avg_word_length < 5:
            return "facil"
        elif avg_sentence_length < 20 and avg_word_length < 6:
            return "medio"
        else:
            return "dificil"
    
    def translate_text(self, text: str, target_language: str, source_language: str = None) -> Dict[str, Any]:
        """Traduz texto (implementação básica)"""
        if not text:
            return {"translated_text": "", "confidence": 0.0, "detected_language": "unknown"}
        
        if source_language is None:
            source_language = self.detect_language(text)
        
        # Tradução básica para demonstração
        translation_key = f"{source_language}_to_{target_language}"
        
        if translation_key in self.basic_translations:
            words = text.lower().split()
            translated_words = []
            
            for word in words:
                clean_word = word.strip(string.punctuation)
                if clean_word in self.basic_translations[translation_key]:
                    translated_words.append(self.basic_translations[translation_key][clean_word])
                else:
                    translated_words.append(word)  # Manter palavra original se não encontrar tradução
            
            translated_text = ' '.join(translated_words)
            confidence = 0.6  # Confiança básica
        else:
            # Sem tradução disponível
            translated_text = text
            confidence = 0.0
        
        return {
            "translated_text": translated_text,
            "confidence": confidence,
            "detected_language": source_language,
            "target_language": target_language,
            "translation_method": "basic_dictionary"
        }
    
    def analyze_text_comprehensive(self, text: str, language: str = None) -> TextAnalytics:
        """Análise completa de texto"""
        if not text:
            return TextAnalytics(
                language="unknown",
                word_count=0,
                sentence_count=0,
                paragraph_count=0,
                avg_sentence_length=0.0,
                avg_word_length=0.0,
                readability_score=0.0,
                keywords=[],
                sentiment=SentimentAnalysis(SentimentType.NEUTRAL, 0.0, 0.0, 0.0, 1.0, {}),
                entities=EntityExtraction([], [], [], [], [], [], [], []),
                summary=TextSummary(0, 0, 1.0, "", [], [], "neutral", "unknown")
            )
        
        if language is None:
            language = self.detect_language(text)
        
        # Análises básicas
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
        else:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            words = text.split()
        
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Análises avançadas
        keywords = self.extract_keywords(text, language)
        sentiment = self.analyze_sentiment(text, language)
        entities = self.extract_entities(text, language)
        summary = self.summarize_text(text, language=language)
        
        # Score de legibilidade
        readability_score = 0.0
        if TEXTSTAT_AVAILABLE:
            try:
                readability_score = flesch_reading_ease(text)
            except:
                pass
        
        return TextAnalytics(
            language=language,
            word_count=word_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            avg_sentence_length=avg_sentence_length,
            avg_word_length=avg_word_length,
            readability_score=readability_score,
            keywords=keywords,
            sentiment=sentiment,
            entities=entities,
            summary=summary
        )
    
    def generate_text_report(self, analysis: TextAnalytics) -> str:
        """Gera relatório em texto da análise"""
        report = f"""
# 📄 RELATÓRIO DE ANÁLISE DE TEXTO
*Gerado automaticamente pelo Team Agents NLP System*

## 📊 ESTATÍSTICAS BÁSICAS
- **Idioma Detectado**: {analysis.language.upper()}
- **Palavras**: {analysis.word_count:,}
- **Sentenças**: {analysis.sentence_count:,}
- **Parágrafos**: {analysis.paragraph_count:,}
- **Tamanho Médio das Sentenças**: {analysis.avg_sentence_length:.1f} palavras
- **Tamanho Médio das Palavras**: {analysis.avg_word_length:.1f} caracteres

## 📈 LEGIBILIDADE
- **Score de Legibilidade**: {analysis.readability_score:.1f}
- **Nível de Leitura**: {analysis.summary.reading_level.replace('_', ' ').title()}

## 🔑 PALAVRAS-CHAVE PRINCIPAIS
"""
        
        for i, (keyword, score) in enumerate(analysis.keywords[:10], 1):
            report += f"{i}. **{keyword}** (relevância: {score:.3f})\n"
        
        report += f"""
## 😊 ANÁLISE DE SENTIMENTO
- **Sentimento Geral**: {analysis.sentiment.overall_sentiment.value.upper()}
- **Confiança**: {analysis.sentiment.confidence:.1%}
- **Score Positivo**: {analysis.sentiment.positive_score:.1%}
- **Score Negativo**: {analysis.sentiment.negative_score:.1%}
- **Score Neutro**: {analysis.sentiment.neutral_score:.1%}

"""
        
        if analysis.sentiment.emotion_breakdown:
            report += "### Detalhamento Emocional:\n"
            for emotion, score in analysis.sentiment.emotion_breakdown.items():
                if score > 0.1:
                    report += f"- **{emotion.title()}**: {score:.1%}\n"
        
        report += f"""
## 🏷️ ENTIDADES EXTRAÍDAS
"""
        
        if analysis.entities.persons:
            report += f"**Pessoas**: {', '.join(analysis.entities.persons[:10])}\n"
        if analysis.entities.organizations:
            report += f"**Organizações**: {', '.join(analysis.entities.organizations[:10])}\n"
        if analysis.entities.locations:
            report += f"**Locais**: {', '.join(analysis.entities.locations[:10])}\n"
        if analysis.entities.dates:
            report += f"**Datas**: {', '.join(analysis.entities.dates[:5])}\n"
        if analysis.entities.emails:
            report += f"**E-mails**: {', '.join(analysis.entities.emails[:5])}\n"
        if analysis.entities.urls:
            report += f"**URLs**: {', '.join(analysis.entities.urls[:3])}\n"
        
        report += f"""
## 📝 RESUMO AUTOMÁTICO
**Taxa de Compressão**: {analysis.summary.compression_ratio:.1%}

{analysis.summary.summary}

### Sentenças-Chave:
"""
        
        for i, sentence in enumerate(analysis.summary.key_sentences, 1):
            report += f"{i}. {sentence}\n"
        
        report += f"""
---
*Análise realizada em {datetime.now().strftime('%d/%m/%Y às %H:%M')}*
"""
        
        return report
    
    def process_document(self, document_path: str = None, text: str = None, 
                        language: str = None, output_format: str = "json") -> Dict[str, Any]:
        """Processa documento completo"""
        if document_path:
            try:
                with open(document_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            except Exception as e:
                return {"error": f"Erro ao ler arquivo: {e}"}
        
        if not text:
            return {"error": "Texto não fornecido"}
        
        # Limpar texto
        cleaned_text = self.clean_text(text, language)
        
        # Análise completa
        analysis = self.analyze_text_comprehensive(cleaned_text, language)
        
        if output_format.lower() == "json":
            # Converter para formato JSON serializável
            return {
                "language": analysis.language,
                "statistics": {
                    "word_count": analysis.word_count,
                    "sentence_count": analysis.sentence_count,
                    "paragraph_count": analysis.paragraph_count,
                    "avg_sentence_length": analysis.avg_sentence_length,
                    "avg_word_length": analysis.avg_word_length,
                    "readability_score": analysis.readability_score
                },
                "keywords": [{"word": kw[0], "score": kw[1]} for kw in analysis.keywords],
                "sentiment": {
                    "overall": analysis.sentiment.overall_sentiment.value,
                    "confidence": analysis.sentiment.confidence,
                    "scores": {
                        "positive": analysis.sentiment.positive_score,
                        "negative": analysis.sentiment.negative_score,
                        "neutral": analysis.sentiment.neutral_score
                    },
                    "emotions": analysis.sentiment.emotion_breakdown
                },
                "entities": {
                    "persons": analysis.entities.persons,
                    "organizations": analysis.entities.organizations,
                    "locations": analysis.entities.locations,
                    "dates": analysis.entities.dates,
                    "emails": analysis.entities.emails,
                    "urls": analysis.entities.urls,
                    "phones": analysis.entities.phones
                },
                "summary": {
                    "text": analysis.summary.summary,
                    "compression_ratio": analysis.summary.compression_ratio,
                    "key_sentences": analysis.summary.key_sentences,
                    "keywords": analysis.summary.keywords,
                    "sentiment": analysis.summary.sentiment,
                    "reading_level": analysis.summary.reading_level
                }
            }
        
        elif output_format.lower() == "report":
            return {"report": self.generate_text_report(analysis)}
        
        else:
            return {"error": "Formato de saída não suportado. Use 'json' ou 'report'"}

# Função de conveniência para análise rápida
def analyze_text(text: str, language: str = None) -> Dict[str, Any]:
    """Função de conveniência para análise rápida de texto"""
    nlp_system = AdvancedNLPSystem()
    return nlp_system.process_document(text=text, language=language)

def summarize_text(text: str, max_sentences: int = 3, language: str = None) -> str:
    """Função de conveniência para resumo rápido"""
    nlp_system = AdvancedNLPSystem()
    summary = nlp_system.summarize_text(text, max_sentences, language)
    return summary.summary

def detect_sentiment(text: str, language: str = None) -> str:
    """Função de conveniência para análise de sentimento"""
    nlp_system = AdvancedNLPSystem()
    sentiment = nlp_system.analyze_sentiment(text, language)
    return sentiment.overall_sentiment.value