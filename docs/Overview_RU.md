# Industrial Docs RAG Chatbot — обзор проекта (RU)


**1. Идея**

Этот репозиторий — «скелет» системы Retrieval-Augmented Generation (RAG)
для работы с внутренней документацией (инструкции, регламенты, технические описания).

**Цель:**

- собрать документы в одну индексируемую базу;

- дать простой API, который:

- по запросу находит релевантные фрагменты текста;

- в дальнейшем сможет передавать эти фрагменты в LLM для генерации ответа.

Текущая версия не привязана к конкретной компании и не содержит реальной документации —
это шаблон, который можно адаптировать под любую предметную область.


**2. Архитектура (в двух словах)**
**Конфигурация (src/config.py, configs/default.yaml)**

- Описывает базовые пути проекта:

- data/raw — сырые тексты;

- data/index — FAISS-индекс и метаданные;

- configs/default.yaml — параметры эмбеддингов, ретривера, LLM-клиента.

- Даёт типизированный AppConfig (embedding / retrieval / llm) и пути:

- config.INDEX_PATH

- config.METADATA_PATH


**Индексация (два варианта)**

**1. Лёгкий скрипт** scripts/build_index.py

- Читает текстовые файлы (.txt, .md);

- режет их на чанки по символам;

- строит простые hash-эмбеддинги через VectorRetriever.encode_texts;

- собирает FAISS-индекс и JSONL-метаданные (doc_id, chunk_id, source_path, text).

**2. Более тяжёлый** ingest-pipeline src/ingest.py

- Делает разбиение текстов на параграфы + перекрывающиеся чанки;

- использует sentence-transformers для настоящих эмбеддингов;

- пишет индекс и метаданные в те же файлы, что и scripts/build_index.py
(пути берутся из config.INDEX_PATH и config.METADATA_PATH).

- Оба варианта формируют файловый формат, который понимает ретривер и API.


**Векторный ретривер** (src/retriever.py)

Класс VectorRetriever:

- умеет:

- кодировать тексты в эмбеддинги (hash-based заглушка);

- строить FAISS-индекс;

- сохранять / загружать индекс и метаданные;

- выполнять search(query, top_k, with_text) и отдавать список результатов с полями:

- rank, score, doc_id, chunk_id, source_path, metadata, а при with_text=True — text.

- фабрики:

- VectorRetriever.from_config() / from_default() — загрузка индекса и метаданных по путям из config;

- VectorRetriever.for_index_building() — режим «только строим индекс».


**HTTP-API** (src/api/app.py)

Минимальное FastAPI-приложение с эндпоинтами:

- GET /health — проверка живости;

- POST /search — векторный поиск по индексу;

- POST /chat — заглушка будущего RAG-чат-бота:

- возвращает шаблонный ответ;

- прикладывает список найденного контекста (список SearchResult).

Ретривер создаётся лениво через get_retriever() и использует VectorRetriever.from_config().


**CLI-интерфейс** (src/cli.py)

Простой CLI для ручной проверки поиска:


python -m src.cli search "How to harden SSH on Ubuntu?" --top-k 5


- поднимает VectorRetriever.from_default();

- вызывает поиск;

- печатает ранжированные результаты со скором, doc_id, chunk_id и коротким сниппетом текста.



**Тесты**

- tests/test_api_smoke.py — smoke-тесты HTTP-API:

- монкипатчит get_retriever, подставляя DummyRetriever;

- проверяет форму ответов /health, /search, /chat.

- tests/test_build_index_script.py — лёгкий end-to-end тест scripts/build_index.py:

- создаёт временную директорию с текстовыми файлами;

- временно переназначает config.INDEX_PATH и config.METADATA_PATH;

- запускает скрипт и проверяет, что индекс и метаданные созданы.

- tests/test_retriever_encode_and_search.py:

- тестирует VectorRetriever.encode_texts (размерность, нормировка, детерминированность);

- проверяет полный цикл: encode → build index → save → load → search.



**3. Структура проекта**

project-root/
├─ configs/
│  └─ default.yaml          # параметры эмбеддингов, ретривера, LLM
├─ data/
│  ├─ raw/                  # сырые документы (.txt, .md)
│  └─ index/                # faiss_index.bin + metadata.jsonl (генерируются)
├─ src/
│  ├─ __init__.py
│  ├─ config.py             # центральные настройки: пути, AppConfig, INDEX_PATH/METADATA_PATH
│  ├─ retriever.py          # VectorRetriever: загрузка/поиск/работа с FAISS
│  ├─ cli.py                # CLI: python -m src.cli search "..." --top-k 5
│  ├─ ingest.py             # ingestion-пайплайн на sentence-transformers
│  └─ api/
│     └─ app.py             # FastAPI-приложение: /health, /search, /chat
├─ scripts/
│  └─ build_index.py        # лёгкий скрипт сборки индекса (hash-эмбеддинги)
├─ tests/
│  ├─ test_api_smoke.py             # smoke-тесты API с DummyRetriever
│  ├─ test_build_index_script.py    # e2e-тест scripts/build_index.py
│  └─ test_retriever_encode_and_search.py  # e2e-тест ретривера и encode_texts
├─ docs/
│  ├─ Overview_EN.md        # обзор на английском
│  └─ Overview_RU.md        # этот файл
│  └─ images/
│     └─ cli_search_example.png
│     └─ swagger_search_example.png
│     └─ uvicorn_run_example.png
├─ requirements.txt         # зависимости для запуска сервиса
├─ requirements-dev.txt     # dev-зависимости (pytest, black, mypy и т.п.)
├─ pytest.ini               # конфигурация pytest
├─ .gitignore               # игнорирование venv, кэшей, временных файлов
└─ README.md                # основное описание проекта (EN)



**4. Типичные сценарии использования**

**4.1. Подготовка данных**

1. Собрать документы в data/raw/ в виде простых файлов:

.txt

.md

2. При необходимости предварительно конвертировать PDF / DOCX → текст (наружу этого репозитория).


**4.2. Построение индекса**

**Вариант A — лёгкий, без sentence-transformers**

python -m scripts.build_index --input-dir data/raw


Скрипт:

- читает все .txt/.md в data/raw;

- режет на чанки по символам;

- считает простые hash-эмбеддинги;

- строит FAISS-индекс;

- сохраняет:

data/index/faiss_index.bin

data/index/metadata.jsonl

**Вариант B — ingest-pipeline с sentence-transformers**


python -m src.ingest


Скрипт:

- читает все документы из data/raw;

- строит параграфные чанки с перекрытием;

- использует модель из configs/default.yaml (embedding.model_name) на CPU/GPU;

- строит FAISS-индекс и сохраняет в те же файлы, что и вариант A.

Оба варианта совместимы с VectorRetriever и API.


**4.3. Запуск HTTP-API**

Из корня проекта:

uvicorn src.api.app:app --reload


Далее можно:

- GET /health — проверить, что сервис жив;

- POST /search — выполнять запросы вида:


{
  "query": "How to harden SSH on Ubuntu?",
  "top_k": 5
}


- POST /chat — пока заглушка:

- возвращает текст вида "This is a stub chat endpoint response...";

- включает context — список найденных чанков.


**4.4. CLI-поиск**

Для быстрой проверки без HTTP:

python -m src.cli search "How to harden SSH on Ubuntu?" --top-k 5


Выводится:

- ранги;

- score;

- doc_id, chunk_id;

- короткий сниппет текста.


**5. Статус проекта**

- Индекс строится в двух вариантах:

- лёгкий hash-based (scripts/build_index.py);

- «тяжёлый» на sentence-transformers (src/ingest.py).

- RAG-часть /chat пока работает как заглушка:

- возвращает фиксированный ответ;

- прикладывает контекст, полученный от ретривера.

- Основной фокус — на чистой архитектуре и разделении ответственности:

- отдельная конфигурация (configs/default.yaml, src/config.py);

- изолированный ретривер (src/retriever.py);

- тонкое HTTP-API (src/api/app.py);

- CLI и скрипты для индексации (src/cli.py, scripts/build_index.py, src/ingest.py);

- простые, но показательные тесты (tests/…).


**Проект можно использовать как стартовую точку для:**

- внутреннего чатбота по документации;

- прототипа системы поддержки инженеров / операторов;

- учебного RAG-проекта для студентов и джуниор-разработчиков;

- демонстрации базовой архитектуры «индекс → ретривер → API → клиент (CLI/HTTP)».

