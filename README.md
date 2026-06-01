# FrauEdeltraud - Перевод видео в английский

**Система для перевода русского видео/текста в простой английский (A2+) с озвучкой.**

## 🚀 Быстрый старт

### 1. Запусти тест для нужного сервиса

```bash
python tests/test_extractor_manual.py      # Извлечение текста
python tests/test_translator_manual.py     # Перевод
python tests/test_critic_manual.py         # Проверка качества
python tests/test_video_manual.py          # Генерация видео
python tests/test_pipeline_manual.py       # Всё вместе
```

### 2. Отредактируй данные в тесте

Открой файл теста, найди функцию:

```python
def test_something():
    service = SomeService()
    
    # РЕДАКТИРУЙ ЗДЕСЬ:
    source = "Твой текст или URL"
    
    result = service.process(source)
    print(result)
```

### 3. Запусти тест

```bash
python tests/test_extractor_manual.py
```

Смотри логи от каждого сервиса в консоли.

## 📁 Структура тестов

```
tests/
├── test_extractor_manual.py      - 3 теста ExtractorService
├── test_translator_manual.py     - 3 теста TranslatorService
├── test_critic_manual.py         - 3 теста CriticService
├── test_video_manual.py          - 3 теста VideoGeneratorService
└── test_pipeline_manual.py       - 3 теста всего pipeline
```

## 🔧 Сервисы

- **ExtractorService** - извлечение текста (YouTube, видео, текст)
- **TranslatorService** - перевод на английский A2+
- **CriticService** - проверка качества перевода
- **VideoGeneratorService** - генерация видео с озвучкой

## 📝 Примеры тестов

### ExtractorService

```python
# В файле test_extractor_manual.py

source = "Привет! Как дела?"  # или YouTube URL или путь к видео
text, language, video_path, audio_path = service.process(source)
```

### TranslatorService

```python
source = "Привет! Как дела?"
translated, stats = service.process(source)
```

### CriticService

```python
source = "Привет! Как дела?"
translation = "Hello! How are you?"
is_approved, final, details = service.critique(source, translation, stats)
```

### VideoGeneratorService

```python
text = "Hello! How are you?"
video_path, duration = service.process(text)
```

## 💬 Логирование

Каждый сервис выводит логи с префиксом:

```
[ExtractorService] Инициализирован
[ExtractorService] Обработка: Привет ...
✓ Извлечено (ru): 50 символов

[TranslatorService] Инициализирован
[TranslatorService] Начало перевода...
✓ Перевод завершён: 70 символов

[CriticService] Проверка перевода...
✓ Оценка: 8/10, Одобрено: True

[VideoGeneratorService] Генерация видео...
✓ Видео готово: 5.2 сек
```

## 🎯 Как работает

1. Открываешь файл теста
2. Редактируешь входные данные (текст, URL или путь к видео)
3. Запускаешь: `python tests/test_*.py`
4. Смотришь логи в консоли
5. Результат в `./temp/` папке

**Всё просто и ручное. Никаких автотестов.**

## 📋 Требования

```bash
pip install -r requirements.txt
```

Нужен `.env` файл с `GOOGLE_API_KEY`.

## ✅ Что было

- ✓ 5 ручных тестов для каждого сервиса
- ✓ Каждый тест - это Python файл который ты редактируешь
- ✓ Логи от каждого сервиса отдельно
- ✓ Никаких автотестов
- ✓ Чистый и простой код
