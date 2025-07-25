# bio_inspired_skeptical_ai_russian.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any
import random
import uuid
import re

# Настройка логирования
# Уменьшаем verbosity для transformers
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. РЕАЛЬНЫЕ НЕЙРОБИОЛОГИЧЕСКИЕ ДАННЫЕ ---
# Эти данные основаны на исследованиях.

FREQUENCY_BANDS = {
    'delta': {'range_hz': (0.5, 4), 'description': 'Сон, глубокая медитация'},
    'theta': {'range_hz': (4, 8), 'description': 'Память, обучение, креативность'},
    'alpha': {'range_hz': (8, 13), 'description': 'Расслабление, состояние покоя'},
    'beta': {'range_hz': (13, 30), 'description': 'Активное мышление, концентрация'},
    'gamma': {'range_hz': (30, 100), 'description': 'Высокий когнитивный контроль'},
    'high_gamma': {'range_hz': (100, 200), 'description': 'Связано с нейронными ансамблями'}
}

BRAIN_REGIONS = {
    'dmPFC': {'description': 'Дорсомедиальная префронтальная кора', 'typical_bands': {'gamma': 0.8, 'theta': 0.6}, 'functions': ['executive_control', 'decision_making']},
    'lPFC': {'description': 'Латеральная префронтальная кора', 'typical_bands': {'gamma': 0.7, 'beta': 0.8}, 'functions': ['working_memory', 'planning']},
    'PPC': {'description': 'Задняя теменная кора', 'typical_bands': {'alpha': 0.5, 'gamma': 0.9}, 'functions': ['spatial_attention']},
    'MTL': {'description': 'Медиальная височная доля', 'typical_bands': {'theta': 0.9, 'delta': 0.7}, 'functions': ['memory_encoding']},
    'LHA': {'description': 'Латеральная гипоталамическая область', 'typical_bands': {'low_gamma': 0.6}, 'functions': ['motivation']}
}

BEHAVIORAL_PHASES = {
    'desire': {'description': 'Фаза желания', 'regions': ['LHA'], 'bands': {'low_gamma': 0.7}},
    'search': {'description': 'Фаза поиска', 'regions': ['PPC', 'lPFC'], 'bands': {'beta': 0.8}},
    'consumption': {'description': 'Фаза потребления', 'regions': ['MTL', 'dmPFC'], 'bands': {'gamma': 0.9}}
}

class CognitiveMode(Enum):
    LOGICAL_CHAIN = "logical"
    CREATIVE_IMAGINATION = "creative"
    EMOTIONAL_REASONING = "emotional"
    SOCIAL_COGNITION = "social"

@dataclass
class NeuralPattern:
    """Структура для хранения реалистичных нейронных паттернов."""
    pattern_id: str
    brain_region: str
    frequency_band: str
    cognitive_function: str
    power: float
    coherence: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

# --- 2. СИМУЛЯТОР НЕЙРОННЫХ ПАТТЕРНОВ ---
class NeuralPatternSimulator:
    def __init__(self):
        self.brain_regions = BRAIN_REGIONS
        self.frequency_bands = FREQUENCY_BANDS
        self.behavioral_phases = BEHAVIORAL_PHASES

    def generate_patterns(self, num_patterns: int = 20) -> List[NeuralPattern]:
        patterns = []
        for _ in range(num_patterns):
            region_name = random.choice(list(self.brain_regions.keys()))
            region_data = self.brain_regions[region_name]
            
            if region_data['typical_bands']:
                band_name = random.choices(list(region_data['typical_bands'].keys()), 
                                         weights=list(region_data['typical_bands'].values()))[0]
                base_power = region_data['typical_bands'][band_name]
            else:
                band_name = random.choice(list(self.frequency_bands.keys()))
                base_power = 0.5

            func = random.choice(region_data['functions'])
            power = np.clip(np.random.normal(base_power, 0.15), 0.0, 1.0)
            coherence = np.clip(np.random.beta(2, 5) + 0.2, 0.0, 1.0)

            metadata = {}
            if random.random() < 0.3:
                phase_name = random.choice(list(self.behavioral_phases.keys()))
                metadata['behavioral_phase'] = phase_name

            pattern = NeuralPattern(
                pattern_id=f"pat_{uuid.uuid4().hex[:6]}",
                brain_region=region_name,
                frequency_band=band_name,
                cognitive_function=func,
                power=power,
                coherence=coherence,
                timestamp=datetime.now(),
                metadata=metadata
            )
            patterns.append(pattern)
        return patterns

# --- 3. АНАЛИЗАТОР СКЕПТИЦИЗМА ---
class SkepticismAnalyzer:
    def analyze(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 5:
            return {
                "authenticity_score": 0.1,
                "red_flags": ["Слишком короткий или пустой текст"],
                "positive_indicators": [],
                "detailed_scores": {}
            }

        words = text.split()
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # --- Метрики ---
        # 1. Лексическое разнообразие (TTR)
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0
        ttr_score = 1.0 - abs(ttr - 0.65) / 0.35
        ttr_score = np.clip(ttr_score, 0.0, 1.0)

        # 2. Длина предложений
        sent_var_score = 0.5
        if sentences and len(sentences) > 1:
            sent_lengths = [len(s.split()) for s in sentences]
            var_sent_len = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
            sent_var_score = 1.0 - abs(var_sent_len - 50) / 50
            sent_var_score = np.clip(sent_var_score, 0.0, 1.0)

        # 3. Эмоциональные слова
        emotional_words = ['чувствую', 'думаю', 'верю', 'боюсь', 'рад', 'грустно', 'волнуюсь', 'люблю', 'ненавижу', 'нравится', 'удивлен']
        emot_count = sum(text.lower().count(w) for w in emotional_words)
        emot_density = emot_count / total_words if total_words > 0 else 0
        emot_score = 1.0 - abs(emot_density - 0.03) / 0.05
        emot_score = np.clip(emot_score, 0.0, 1.0)

        # 4. Личные местоимения
        personal_pronouns = ['я', 'меня', 'мне', 'мной', 'мое', 'моя', 'мои', 'мой']
        pers_count = sum(text.lower().count(w) for w in personal_pronouns)
        pers_density = pers_count / total_words if total_words > 0 else 0
        pers_score = 1.0 - abs(pers_density - 0.02) / 0.03
        pers_score = np.clip(pers_score, 0.0, 1.0)

        # 5. Логические связки
        logic_words = ['потому что', 'следовательно', 'однако', 'поэтому', 'если', 'то', 'таким образом', 'во-первых', 'во-вторых']
        logic_count = sum(text.lower().count(w) for w in logic_words)
        logic_density = logic_count / max(len(sentences), 1)
        logic_score = 1.0 - abs(logic_density - 0.1) / 0.2
        logic_score = np.clip(logic_score, 0.0, 1.0)

        # --- Комбинированная оценка ---
        weights = {'ttr': 0.2, 'sentence_var': 0.2, 'emotion': 0.2, 'personal': 0.2, 'logic': 0.2}
        authenticity_score = (ttr_score * weights['ttr'] + 
                             sent_var_score * weights['sentence_var'] + 
                             emot_score * weights['emotion'] + 
                             pers_score * weights['personal'] +
                             logic_score * weights['logic'])

        # --- Красные флаги и положительные индикаторы ---
        red_flags = []
        if not (0.3 < ttr < 0.9): red_flags.append("Ненормальное лексическое разнообразие")
        if len(words) < 10: red_flags.append("Слишком короткий текст")
        if emot_density < 0.005: red_flags.append("Отсутствие эмоциональных выражений")
        # Проверка на бессмыслицу (очень простая эвристика)
        if len(words) > 5 and all(len(w) < 2 or not w.isalpha() for w in words[:5]):
            red_flags.append("Возможно бессмысленный текст")

        positive_indicators = []
        if 0.4 < ttr < 0.8: positive_indicators.append("Хорошее лексическое разнообразие")
        if len(sentences) > 1 and np.var([len(s.split()) for s in sentences]) > 10: 
            positive_indicators.append("Хорошая вариативность длины предложений")
        if emot_density > 0.01: positive_indicators.append("Найдены эмоциональные выражения")
        if pers_density > 0.005: positive_indicators.append("Найдены личные ссылки")
        if logic_density > 0.05: positive_indicators.append("Найдены логические связки")

        return {
            "authenticity_score": round(float(authenticity_score), 3),
            "red_flags": red_flags,
            "positive_indicators": positive_indicators,
            "detailed_scores": {
                "lexical_diversity": round(float(ttr_score), 3),
                "sentence_variance": round(float(sent_var_score), 3),
                "emotional_expression": round(float(emot_score), 3),
                "personal_reference": round(float(pers_score), 3),
                "logical_coherence": round(float(logic_score), 3)
            }
        }

# --- 4. ОСНОВНОЙ ГЕНЕРАТОР ---
class BioInspiredSkepticalTextGenerator:
    MODEL_NAME = "sberbank-ai/rugpt3large_based_on_gpt2"
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Используется устройство: {self.device}")

        logger.info(f"Загрузка модели '{self.MODEL_NAME}' и токенизатора...")
        # Для моделей, предобученных на специальных токенах, важно загрузить их конфигурацию
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.MODEL_NAME)
        # Убедимся, что pad_token установлен
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        self.model = GPT2LMHeadModel.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Модель '{self.MODEL_NAME}' загружена. Параметров: {self.model.num_parameters()}")

        self.neural_simulator = NeuralPatternSimulator()
        self.skepticism_analyzer = SkepticismAnalyzer()
        self.neural_patterns = self.neural_simulator.generate_patterns()
        logger.info("Генератор инициализирован.")

    def _get_mode_influence(self, mode: CognitiveMode) -> Dict[str, float]:
        """Определяет параметры генерации на основе когнитивного режима и нейронных паттернов."""
        relevant_patterns = [p for p in self.neural_patterns 
                           if mode.value in p.cognitive_function or 
                              (p.metadata and p.metadata.get('behavioral_phase') == mode.value)]
        
        if not relevant_patterns:
            # Если нет релевантных паттернов, используем базовые параметры
            return {"temperature": 0.8, "top_p": 0.9}

        total_power = sum(p.power for p in relevant_patterns)
        avg_coherence = np.mean([p.coherence for p in relevant_patterns])

        # Простое правило влияния паттернов на параметры
        if mode == CognitiveMode.CREATIVE_IMAGINATION:
            # Для креатива: выше температура, чуть ниже top_p для баланса
            temperature = np.clip(0.8 + (total_power / len(relevant_patterns)) * 0.3, 0.8, 1.2)
            top_p = np.clip(0.9 - (avg_coherence * 0.1), 0.85, 0.95)
        elif mode == CognitiveMode.LOGICAL_CHAIN:
            # Для логики: ниже температура, выше top_p для фокусировки
            temperature = np.clip(0.7 - (total_power / len(relevant_patterns)) * 0.1, 0.6, 0.8)
            top_p = np.clip(0.92 + (avg_coherence * 0.05), 0.92, 0.95)
        else: # EMOTIONAL_REASONING, SOCIAL_COGNITION
            # Средние значения
            temperature = np.clip(0.75 + (total_power / len(relevant_patterns)) * 0.15, 0.7, 1.0)
            top_p = np.clip(0.9 - (avg_coherence * 0.05), 0.88, 0.92)
            
        logger.info(f"Режим {mode.value}: температура={temperature:.2f}, top_p={top_p:.2f} на основе {len(relevant_patterns)} паттернов")
        return {"temperature": temperature, "top_p": top_p}

    def generate_text(self, prompt: str, 
                      cognitive_mode: CognitiveMode = CognitiveMode.LOGICAL_CHAIN,
                      max_length: int = 150, 
                      min_length: int = 30) -> Dict[str, Any]:
        """
        Генерация текста с учетом когнитивного режима и анализа скептицизма.
        """
        try:
            logger.info(f"Генерация текста в режиме '{cognitive_mode.value}' для промпта: '{prompt}'")
            
            # 1. Получаем параметры, "вдохновленные" нейронными паттернами
            mode_params = self._get_mode_influence(cognitive_mode)
            temperature = mode_params.get("temperature", 0.8)
            top_p = mode_params.get("top_p", 0.9)

            # 2. Подготовка входных данных
            # ВАЖНО: Не добавляем специальные токены в промпт для этой модели,
            # так как она не была обучена на них.
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # 3. Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=min(inputs.shape[1] + max_length, 512),
                    min_length=inputs.shape[1] + min_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3, # Увеличил для лучшего качества
                    top_k=50,
                    top_p=top_p,
                    num_return_sequences=1,
                    early_stopping=True
                )
            
            # 4. Постобработка
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Корректное удаление промпта
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            # 5. Анализ достоверности
            analysis = self.skepticism_analyzer.analyze(generated_text)

            return {
                "generated_text": generated_text,
                "input_prompt": prompt,
                "cognitive_mode": cognitive_mode.value,
                "authenticity_analysis": analysis,
                "neural_patterns_used": len(self.neural_patterns),
                "generation_params": {
                    "model": self.MODEL_NAME,
                    "applied_temperature": round(float(temperature), 3),
                    "applied_top_p": round(float(top_p), 3),
                    "max_length": max_length,
                    "min_length": min_length
                },
                "metadata": {
                    "timestamp": str(datetime.now())
                }
            }
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}

# --- 5. ДЕМОНСТРАЦИЯ ---
def main():
    logger.info("=== Bio-Inspired Skeptical Text Generator (Russian) ===")
    logger.info("Используется модель: sberbank-ai/rugpt3large_based_on_gpt2")
    
    try:
        # Инициализация генератора
        generator = BioInspiredSkepticalTextGenerator()
    except Exception as e:
        logger.error(f"Не удалось загрузить модель: {e}")
        logger.info("Убедитесь, что вы установили необходимые библиотеки: pip install transformers torch")
        return

    # Примеры запросов
    examples = [
        {
            "prompt": "Объясните процесс фотосинтеза шаг за шагом.",
            "mode": CognitiveMode.LOGICAL_CHAIN
        },
        {
            "prompt": "Представьте себе мир, где люди могут читать мысли друг друга.",
            "mode": CognitiveMode.CREATIVE_IMAGINATION
        },
        {
            "prompt": "Опишите свои чувства при виде заката.",
            "mode": CognitiveMode.EMOTIONAL_REASONING
        },
        {
            "prompt": "Как вы думаете, что чувствует человек, когда его не понимают?",
            "mode": CognitiveMode.SOCIAL_COGNITION
        }
    ]
    
    for i, example in enumerate(examples):
        print(f"\n--- Пример {i+1}: {example['mode'].value.upper()} ---")
        print(f"Промпт: {example['prompt']}")
        
        result = generator.generate_text(
            prompt=example['prompt'],
            cognitive_mode=example['mode'],
            max_length=120,
            min_length=40
        )
        
        if "error" not in result:
            print(f"Сгенерированный текст: {result['generated_text']}")
            print(f"Длина: {len(result['generated_text'].split())} слов")
            print(f"Достоверность: {result['authenticity_analysis']['authenticity_score']}")
            if result['authenticity_analysis']['red_flags']:
                print(f"Красные флаги: {', '.join(result['authenticity_analysis']['red_flags'])}")
            else:
                print("Красные флаги: Нет")
            print(f"Параметры: T={result['generation_params']['applied_temperature']}, P={result['generation_params']['applied_top_p']}")
        else:
            print(f"Ошибка: {result['error']}")

if __name__ == "__main__":
    main()