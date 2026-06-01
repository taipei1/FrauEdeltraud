# FrauEdeltraud — Translation Video System

Мультиагентная система (LangGraph) для перевода видео/текста с английскую речь
и публикацией коротких вертикальных видео со словами и подсветкой.

## Как это работает

```
вход (текст или YouTube URL)
        │
        ▼
  EXTRACTOR     ←  субтитры → Gemini audio → Gemini video
        │
        ▼
  TRANSLATOR    ←  Gemini + словарь из PostgreSQL (pgvector)
        │
        ▼
  VIDEO GEN     ←  edge-tts + тайминги слов + ffmpeg
        │
        ▼
  TELEGRAM      ←  отправка mp4 + caption со статистикой
```

## Запуск

```bash
# С текстом
python main.py "Привет! Как у тебя дела?"

# С YouTube URL
python main.py "https://www.youtube.com/watch?v=..."
```

## Структура проекта

```
FrauEdeltraud/
├── main.py                          # точка входа
├── requirements.txt                 # зависимости (librosa/soundfile удалены)
├── pytest.ini                       # маркеры для автотестов
├── run_tests.bat / run_tests.ps1    # скрипты запуска тестов
│
├── src/
│   ├── config.py                    # настройки (TEMP_DIR, VIDEO_*, GEMINI_*)
│   ├── langgraph_system/
│   │   ├── state.py                 # AgentState (TypedDict)
│   │   ├── agents.py                # 3 агента
│   │   └── workflow.py              # граф
│   ├── services/
│   │   ├── extractor.py             # текст / субтитры / аудио / видео
│   │   ├── translator.py            # Gemini + словарь
│   │   ├── tts_service.py           # edge-tts + WordBoundary тайминги
│   │   ├── video_service.py         # кадры PIL + ffmpeg
│   │   ├── telegram_service.py      # отправка в Telegram
│   │   ├── db_service.py            # PostgreSQL + pgvector
│   │   └── llm_client.py            # Gemini → Groq fallback
│   └── utils/logger.py
│
└── tests/
    ├── conftest.py                  # фикстуры (temp_output_dir, sample_text)
    ├── test_unit_*.py               # быстрые, без сети/БД
    ├── test_integration_*.py        # нужны API-ключи и сеть
    └── test_outputs/                # сюда пишутся артефакты
```

## Автотесты — как запускать

### Маркеры

| маркер           | что делает                                                      |
|------------------|-----------------------------------------------------------------|
| `unit`           | чистые тесты, без сети, без БД, без API-ключей (по умолчанию)   |
| `integration`    | нужны сеть и API-ключи в `.env`                                 |
| `slow`           | долгие (YouTube, рендер полного видео)                          |
| `requires_db`    | нужен PostgreSQL на `DATABASE_URL`                              |

### Команды

```bash
# ВСЕ unit-тесты — самые быстрые, не нужны ключи (~15 сек)
python -m pytest tests/ -m unit -v

# unit + integration (без slow/db) — нужны ключи и интернет (~30 сек)
python -m pytest tests/ -m "not slow and not requires_db" -v

# Только интеграция (нужны ключи)
python -m pytest tests/ -m integration -v

# Только долгие (полный workflow с рендером видео)
python -m pytest tests/ -m slow -v

# Только то, что требует БД
python -m pytest tests/ -m requires_db -v

# Всё подряд (часть упадёт без ключей/БД)
python -m pytest tests/ -v
```

### Скрипты

```bat
run_tests.bat unit
run_tests.bat "not slow and not requires_db"
run_tests.bat slow
```

```powershell
.\run_tests.ps1 -Mark "unit"
.\run_tests.ps1 -Mark "integration"
.\run_tests.ps1                 # всё подряд
```

### Что покрыто

| файл                              | что проверяет                                                |
|-----------------------------------|--------------------------------------------------------------|
| `test_unit_config.py`             | VIDEO_* кратны 16, TTS_VOICE задан, workflow компилируется    |
| `test_unit_edge_tts_boundary.py`  | **регрессионный**: edge-tts молчит без `boundary=WordBoundary`|
| `test_unit_tts_service.py`        | TTS возвращает непустые `word_timings` (фикс главного бага)  |
| `test_unit_video_service.py`      | `render_frame` рисует белый текст и жёлтую подсветку         |
| `test_unit_extractor.py`          | определение языка, очистка VTT/SRT                           |
| `test_unit_translator.py`         | `_word_in_text` (без подстрок, регистронезависимо)           |
| `test_integration_video.py`       | end-to-end `VideoService.process()` с ffmpeg                 |
| `test_integration_translator.py`  | реальный перевод через LLM                                   |
| `test_integration_db.py`          | подключение к PostgreSQL                                     |
| `test_integration_workflow.py`    | полный E2E (text → video → telegram)                         |

## Настройка окружения

```env
# .env
GOOGLE_API_KEY=...                  # обязательно для перевода
GROQ_API_KEY=...                    # опционально, fallback
DATABASE_URL=postgresql://postgres:mysecretpassword@localhost:5432/postgres
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Системные зависимости: **ffmpeg** (для склейки видео и аудио; imageio-ffmpeg как fallback), **PostgreSQL + pgvector** (для словаря).

## Частые ошибки

- **«ffmpeg не найден»** — установите ffmpeg или используется imageio-ffmpeg fallback
- **«429 Quota exceeded»** — дневная квота Gemini; подождите или настройте GROQ_API_KEY
- **«Видео без субтитров»** — этот баг исправлен (см. commit history). Регрессионный тест: `pytest tests/test_unit_edge_tts_boundary.py -v`
