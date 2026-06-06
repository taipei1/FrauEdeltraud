# SRS Application - Design Document

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         React Frontend                          │
│  (TypeScript + Vite, Black & White UI, Keyboard Navigation)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                    HTTP REST API
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                    FastAPI Backend                              │
│                   (Python + SQLAlchemy)                         │
├─────────────────────────────────────────────────────────────────┤
│  Routers:                                                       │
│  ├─ /api/cards/          (Language module CRUD)                │
│  ├─ /api/reviews/        (FSRS review logging)                 │
│  ├─ /api/obsidian/       (Knowledge base endpoints)            │
│  └─ /api/stats/          (Time tracking & analytics)           │
├─────────────────────────────────────────────────────────────────┤
│  Services:                                                      │
│  ├─ FSRSService          (Algorithm calculations)              │
│  ├─ GeminiService        (LLM integration)                     │
│  ├─ TTSService           (Audio generation)                    │
│  └─ ObsidianSyncService  (File monitoring)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
    ┌─────────────────────┐  ┌──────────────────┐
    │  PostgreSQL + PGVector  │  Google APIs    │
    │  (Data Storage)      │  (Gemini, TTS)  │
    └─────────────────────┘  └──────────────────┘
```

## Database Schema

### Core Tables

#### users
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
**Purpose**: Single-user app, but table prepared for future multi-user support.

---

#### cards (Language Module)
```sql
CREATE TABLE cards (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    front VARCHAR(500) NOT NULL,           -- The word/phrase to learn
    back VARCHAR(500) NOT NULL,            -- Translation
    hint TEXT,                             -- Example context
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],   -- Tags for filtering
    language VARCHAR(10) NOT NULL,         -- 'en' or 'sk'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- FSRS state
    stability FLOAT DEFAULT 1.0,           -- How stable the memory is
    difficulty FLOAT DEFAULT 5.0,          -- Perceived difficulty (1-10)
    last_reviewed TIMESTAMP,               -- When last reviewed
    
    UNIQUE(user_id, front, language),      -- Prevent duplicates per language
    INDEX idx_language (user_id, language),
    INDEX idx_tags (user_id, tags),
    INDEX idx_due (user_id, last_reviewed, stability)
);
```
**Purpose**: Store vocabulary cards for language learning with FSRS parameters.

---

#### reviews (FSRS History)
```sql
CREATE TABLE reviews (
    id SERIAL PRIMARY KEY,
    card_id INT REFERENCES cards(id) ON DELETE CASCADE,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    rating INT CHECK (rating IN (1, 2, 3, 4)) NOT NULL,  -- User grade
    review_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- FSRS state after review
    stability_after FLOAT,
    difficulty_after FLOAT,
    interval_days INT,                     -- Days until next review
    elapsed_days INT,                      -- Days since last review
    
    INDEX idx_card (card_id),
    INDEX idx_user_date (user_id, review_time)
);
```
**Purpose**: Log all review sessions for FSRS algorithm and analytics.

---

#### obsidian_notes (Knowledge Base)
```sql
CREATE TABLE obsidian_notes (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    file_path VARCHAR(1000) NOT NULL,      -- /path/to/note.md
    content TEXT NOT NULL,                 -- Full markdown content
    tags TEXT[] DEFAULT ARRAY[]::TEXT[],   -- Tags from markdown
    created_at TIMESTAMP,                  -- File creation date
    updated_at TIMESTAMP,                  -- File modification date
    last_reviewed TIMESTAMP,               -- When last reviewed
    
    -- FSRS state for entire note
    stability FLOAT DEFAULT 1.0,
    difficulty FLOAT DEFAULT 5.0,
    
    UNIQUE(user_id, file_path),
    INDEX idx_user (user_id),
    INDEX idx_tags (user_id, tags),
    INDEX idx_due (user_id, last_reviewed, stability)
);
```
**Purpose**: Track Markdown notes from Obsidian vault for review.

---

#### note_embeddings (Vector Search)
```sql
CREATE TABLE note_embeddings (
    id SERIAL PRIMARY KEY,
    note_id INT REFERENCES obsidian_notes(id) ON DELETE CASCADE,
    embedding vector(1536),                -- Gemini embedding vector
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_note (note_id)
);
```
**Purpose**: Store embeddings for semantic search and duplicate detection.

---

#### session_stats (Time Tracking)
```sql
CREATE TABLE session_stats (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE CASCADE,
    session_date DATE NOT NULL,
    module_type VARCHAR(20) NOT NULL,     -- 'language' or 'obsidian'
    category VARCHAR(255),                 -- Tag or language code
    minutes_spent INT DEFAULT 0,
    card_count INT DEFAULT 0,             -- Cards reviewed in session
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(user_id, session_date, module_type, category),
    INDEX idx_user_date (user_id, session_date)
);
```
**Purpose**: Track time spent learning per day, per category.

---

## API Endpoints

### Authentication (Minimal Single-User)

```
POST /api/auth/login
  - Body: { "username": "user", "password": "pass" }
  - Response: { "token": "...", "user_id": 1 }

POST /api/auth/logout
  - Response: { "success": true }
```

---

### Language Module (Cards)

#### Import & Manage Cards

```
POST /api/cards/import
  - Body: { "csv_content": "...", "language": "en" }
  - Purpose: Import cards from CSV/Markdown table
  - Steps:
    1. Parse CSV
    2. Check duplicates (show conflicts)
    3. Auto-tag via Gemini
    4. Seed FSRS weights based on publishedAt date
    5. Insert into DB
  - Response: { "imported": 150, "duplicates": 5, "conflicts": [...] }

GET /api/cards/search
  - Query: ?language=en&tags=verbs,advanced&limit=50
  - Purpose: Search cards by language and tags
  - Response: [{ id, front, back, hint, tags, language }, ...]

GET /api/cards/{id}
  - Purpose: Get single card details
  - Response: { id, front, back, hint, tags, language, stability, difficulty }

POST /api/cards
  - Body: { "front": "baffle", "back": "наводнить", "hint": "...", "language": "en", "tags": ["verbs"] }
  - Purpose: Add single card
  - Checks: Duplicate detection, auto-tagging optional
  - Response: { id, ... }

DELETE /api/cards/{id}
  - Purpose: Delete card permanently
  - Response: { "success": true }

PATCH /api/cards/{id}
  - Body: { "back": "updated translation", "tags": [...] }
  - Purpose: Update card
  - Response: { id, ... }
```

---

#### Review Session (FSRS)

```
GET /api/cards/due?language=en&limit=20
  - Purpose: Get cards due for review (FSRS scheduling)
  - Algorithm:
    - Due if: NOW > last_reviewed + interval_days
    - Sort by: stability ASC (weaker cards first)
    - Limit: 20 cards per batch
  - Response: [{ id, front, back, hint, tags, stability, difficulty }, ...]

POST /api/reviews
  - Body: { 
      "card_id": 123,
      "rating": 3,           // 1=Again, 2=Hard, 3=Good, 4=Easy
      "elapsed_days": 10,    // Days since last review
      "time_spent_seconds": 5
    }
  - Purpose: Log review and update FSRS state
  - Steps:
    1. Calculate new stability/difficulty via py-fsrs
    2. Calculate next interval_days
    3. Update card.stability, card.difficulty, card.last_reviewed
    4. Insert review log
    5. Track session time
  - Response: { "next_review_in_days": 14, "stability": 42.5 }
```

---

### Obsidian Module (Knowledge Base)

#### Synchronization

```
POST /api/obsidian/sync
  - Body: { "folder_path": "/path/to/obsidian-test/" }
  - Purpose: Scan folder, detect changes, update DB
  - Steps:
    1. List all .md files in folder
    2. For each file:
       - Check if already in DB
       - Extract metadata (tags, created, modified)
       - Generate embedding via Gemini
       - Store in DB
    3. Handle deleted files (soft delete?)
  - Response: { "synced": 45, "updated": 12, "deleted": 3 }

GET /api/obsidian/notes
  - Query: ?tags=philosophy,math&limit=50
  - Purpose: List notes with optional filtering
  - Response: [{ id, file_path, tags, created_at, stability, difficulty }, ...]

GET /api/obsidian/notes/{id}
  - Purpose: Get note content
  - Response: { id, file_path, content, tags, stability, difficulty }
```

---

#### Dynamic Question Generation

```
GET /api/obsidian/due?limit=5
  - Purpose: Get notes due for review (FSRS scheduling)
  - Same logic as cards/due
  - Response: [{ id, file_path, tags, stability }, ...]

POST /api/obsidian/questions
  - Body: { 
      "note_id": 456,
      "num_questions": 1,    // How many to generate
      "question_type": "standard"  // or "star" for advanced
    }
  - Purpose: Generate questions dynamically via Gemini
  - Steps:
    1. Fetch note.content
    2. Call Gemini with prompt:
       "Generate N atomic, specific questions from this text.
        Questions must be answerable in 1-2 sentences.
        Provide answer for each question."
    3. Parse LLM response (questions + answers)
    4. Return to frontend (NOT stored in DB)
  - Response: [
      { 
        "question": "What is...?",
        "answer": "The answer is..."
      },
      ...
    ]

POST /api/obsidian/reviews
  - Body: { 
      "note_id": 456,
      "rating": 3,           // 1-4 grade
      "elapsed_days": 5,
      "time_spent_seconds": 45
    }
  - Purpose: Log review of entire note (not individual questions)
  - Steps:
    1. Update note.stability, note.difficulty, note.last_reviewed
    2. Insert review log
    3. Track session time
  - Response: { "next_review_in_days": 21 }
```

---

### Statistics

```
GET /api/stats/daily?date=2026-05-28
  - Purpose: Get time spent on given date (total + by category)
  - Response: {
      "date": "2026-05-28",
      "total_minutes": 45,
      "by_category": {
        "en": 20,
        "sk": 15,
        "obsidian": 10
      }
    }

GET /api/stats/summary?days=30
  - Purpose: Get summary for last N days
  - Response: {
      "period_days": 30,
      "total_minutes": 1245,
      "avg_per_day": 41.5,
      "by_module": {
        "language": 900,
        "obsidian": 345
      },
      "by_category": { "en": 600, "sk": 300, ... }
    }

GET /api/stats/progress?module=language
  - Purpose: Learning progress (cards completed, due, etc.)
  - Response: {
      "total_cards": 1000,
      "cards_due": 50,
      "cards_reviewed": 800,
      "avg_stability": 25.3,
      "by_language": {
        "en": { "total": 600, "due": 30, "stability": 28 },
        "sk": { "total": 400, "due": 20, "stability": 22 }
      }
    }
```

---

## Services Layer

### FSRSService

**Purpose**: Encapsulate FSRS algorithm calculations.

```python
class FSRSService:
    def calculate_review(
        self,
        rating: int,           # 1-4
        stability: float,
        difficulty: float,
        elapsed_days: int
    ) -> Dict[str, float]:
        """
        Calculate new stability, difficulty, and interval.
        Uses py-fsrs library internally.
        
        Returns:
            {
                "stability": 42.5,
                "difficulty": 6.2,
                "interval_days": 14
            }
        """
        ...
    
    def seed_initial_weights(
        self,
        created_date: datetime,
        current_date: datetime
    ) -> Tuple[float, float]:
        """
        For imported cards, seed FSRS weights based on age.
        Ensures old cards don't repeat infinitely.
        
        Example:
            - Card created 6 months ago → stability=50, difficulty=5
            - Card created 1 month ago → stability=20, difficulty=5
            - Card created today → stability=1, difficulty=5
        """
        ...
```

---

### GeminiService

**Purpose**: Wrapper for Google Gemini API calls.

```python
class GeminiService:
    def generate_tags_and_translation(
        self,
        word: str,
        context: str,
        language: str
    ) -> Dict[str, Any]:
        """
        Auto-tag and translate a word.
        
        Prompt:
            "Analyze this English word: '{word}'
             Context: {context}
             Provide:
             1. Tags (comma-separated, e.g., 'verbs,advanced,phrasal')
             2. Translation to Russian
             3. Example usage"
        """
        ...
    
    def generate_questions(
        self,
        content: str,
        num_questions: int = 1,
        advanced: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate atomic questions from note content.
        
        Prompt (standard):
            "Generate {num_questions} atomic, specific questions
             answerable in 1-2 sentences from this text:
             {content}
             
             Format: Q1: ... A1: ... Q2: ... A2: ..."
        
        Prompt (advanced/star):
            "Based on the concepts in this text, generate
             a deep, thought-provoking question that extends
             the content and might not have a direct answer.
             {content}
             
             Provide: Q: ... A: ..."
        """
        ...
    
    def generate_embeddings(
        self,
        text: str
    ) -> List[float]:
        """
        Generate vector embedding for semantic search.
        """
        ...
```

---

### TTSService

**Purpose**: Google Text-to-Speech integration.

```python
class TTSService:
    def synthesize_speech(
        self,
        text: str,
        language: str,
        voice_name: str = None
    ) -> bytes:
        """
        Generate audio from text.
        
        Args:
            language: 'en', 'sk', etc.
            voice_name: e.g., 'en-US-Neural2-C' (male), 
                       'en-US-Neural2-E' (female)
        
        Returns:
            MP3 audio bytes
        """
        ...
    
    def get_language_voice(self, language: str) -> str:
        """
        Map language code to recommended Google TTS voice.
        """
        ...
```

---

### ObsidianSyncService

**Purpose**: File watcher and Markdown parsing.

```python
class ObsidianSyncService:
    def watch_folder(self, folder_path: str):
        """
        Continuously monitor folder for changes.
        Uses watchdog library (cross-platform file monitoring).
        """
        ...
    
    def parse_markdown(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata and content from .md file.
        
        Returns:
            {
                "file_path": "/path/to/note.md",
                "content": "...",
                "tags": ["tag1", "tag2"],
                "created_at": datetime,
                "updated_at": datetime,
                "frontmatter": {...}  # YAML front matter if present
            }
        """
        ...
    
    def sync_database(self, folder_path: str) -> Dict[str, int]:
        """
        Full sync: scan all files, update DB.
        
        Returns:
            {"synced": 45, "updated": 12, "deleted": 3}
        """
        ...
```

---

## Frontend Architecture

### Page Structure

```
App.tsx
├── Navigation / Module Selector
│   ├── Language Learning
│   ├── Obsidian
│   └── Statistics
│
├── LanguagePage.tsx
│   ├── CardImport.tsx (Add cards)
│   ├── LanguageSelector.tsx (en / sk)
│   ├── TagFilter.tsx
│   └── ReviewSession.tsx
│       └── CardDisplay.tsx
│
├── ObsidianPage.tsx
│   ├── NoteSyncButton.tsx
│   ├── TagFilter.tsx
│   └── ReviewSession.tsx
│       └── QuestionDisplay.tsx
│
└── StatsPage.tsx
    └── StatsView.tsx (Charts, time tracking)
```

---

### Component Examples

#### CardDisplay.tsx (Language Module)
```typescript
interface Card {
  id: number;
  front: string;
  back: string;
  hint?: string;
  language: string;
}

interface CardDisplayProps {
  card: Card;
  onGrade: (rating: 1 | 2 | 3 | 4) => void;
  onDelete: () => void;
  onReplayAudio: () => void;
}

export function CardDisplay({
  card,
  onGrade,
  onDelete,
  onReplayAudio
}: CardDisplayProps) {
  const [isFlipped, setIsFlipped] = useState(false);

  // Hotkey handling
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        setIsFlipped(!isFlipped);
      }
      if (e.code === 'Digit1') onGrade(1);
      if (e.code === 'Digit2') onGrade(2);
      if (e.code === 'Digit3') onGrade(3);
      if (e.code === 'Digit4') onGrade(4);
      if (e.code === 'KeyD') onDelete();
      if (e.code === 'KeyR') onReplayAudio();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isFlipped]);

  return (
    <div className="card-container">
      <div className={`card ${isFlipped ? 'flipped' : ''}`}>
        <div className="card-front">
          <h1>{card.front}</h1>
          <button onClick={onReplayAudio}>🔊 Repeat (R)</button>
        </div>
        <div className="card-back">
          <p>{card.back}</p>
          {card.hint && <p className="hint">{card.hint}</p>}
        </div>
      </div>

      <div className="controls">
        <button onClick={() => onGrade(1)}>1: Again</button>
        <button onClick={() => onGrade(2)}>2: Hard</button>
        <button onClick={() => onGrade(3)}>3: Good</button>
        <button onClick={() => onGrade(4)}>4: Easy</button>
        <button onClick={onDelete} className="delete">D: Delete</button>
      </div>
    </div>
  );
}
```

---

#### QuestionDisplay.tsx (Obsidian Module)
```typescript
interface Question {
  question: string;
  answer: string;
}

interface QuestionDisplayProps {
  question: Question;
  onGrade: (rating: 1 | 2 | 3 | 4) => void;
  onRejectQuestion: () => void;
  onAskAnother: () => void;
  onStarQuestion: () => void;
}

export function QuestionDisplay({
  question,
  onGrade,
  onRejectQuestion,
  onAskAnother,
  onStarQuestion
}: QuestionDisplayProps) {
  const [isRevealed, setIsRevealed] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        e.preventDefault();
        setIsRevealed(!isRevealed);
      }
      if (e.code === 'Digit1') onGrade(1);
      if (e.code === 'Digit2') onGrade(2);
      if (e.code === 'Digit3') onGrade(3);
      if (e.code === 'Digit4') onGrade(4);
      if (e.code === 'KeyD') onRejectQuestion();
      if (e.code === 'KeyE') onAskAnother();
      if (e.code === 'KeyS') onStarQuestion();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isRevealed]);

  return (
    <div className="question-container">
      <h2>{question.question}</h2>

      {isRevealed && (
        <div className="answer">
          <p>{question.answer}</p>
        </div>
      )}

      {!isRevealed && (
        <p className="hint">(Press Space to reveal answer)</p>
      )}

      <div className="controls">
        <button onClick={() => onGrade(1)}>1: Again</button>
        <button onClick={() => onGrade(2)}>2: Hard</button>
        <button onClick={() => onGrade(3)}>3: Good</button>
        <button onClick={() => onGrade(4)}>4: Easy</button>
        <button onClick={onRejectQuestion}>D: Reject</button>
        <button onClick={onAskAnother}>E: Ask Another</button>
        <button onClick={onStarQuestion}>S: Star Question</button>
      </div>
    </div>
  );
}
```

---

## Data Flow Diagrams

### Language Module - Import Flow

```
User uploads cards.csv
    ↓
CSV Parser
    ├─ Validate format
    ├─ Check duplicates
    └─ Extract: front, back, hint, tags, language
    ↓
Gemini Auto-Tag
    ├─ Translate if needed
    ├─ Auto-assign tags
    └─ Add context
    ↓
FSRS Seeding
    ├─ Calculate stability from publishedAt
    ├─ Set difficulty=5 (neutral)
    └─ Set last_reviewed=now
    ↓
Database Insert
    ├─ cards table
    └─ Handle duplicates
    ↓
Response: Import summary
```

---

### Obsidian Module - Review Flow

```
User requests due notes
    ↓
FSRS Scheduling
    ├─ Find notes where: NOW > last_reviewed + interval
    ├─ Sort by stability (weakest first)
    └─ Limit: 5 notes per batch
    ↓
Display first note (title, tags)
    ↓
User presses Space (show questions)
    ↓
Gemini Question Generation
    ├─ Get note.content
    ├─ LLM generates N atomic questions
    └─ Include answers
    ↓
User grades question (1-4)
    ↓
FSRS Update (for entire note, not question)
    ├─ Calculate new stability/difficulty
    ├─ Update note.stability, difficulty, last_reviewed
    └─ Record in reviews table
    ↓
Next question or next note
```

---

## Integration Points

### Google Gemini API

- **Auto-tagging cards**: Analyze word, suggest tags
- **Dynamic question generation**: Create questions from note text
- **Embeddings**: Generate 1536-dim vectors for pgvector
- **Advanced questions (star)**: Generate deep questions beyond content

**Rate Limiting**: Max 5 requests/min during MVP to avoid costs.

---

### Google Text-to-Speech

- **Language support**: English (en-US, en-GB), Slovak (sk-SK)
- **Voice selection**: Male/female options
- **Output**: MP3 audio, cached on frontend localStorage

**Cost optimization**: Cache audio by hash(text + language).

---

### PostgreSQL + pgvector

- **Primary data storage**: All cards, notes, reviews
- **Semantic search**: Find similar notes via embedding similarity
- **Indexing**: IVFFLAT index for fast vector search

---

## Error Handling & Validation

### Input Validation

```python
# Pydantic schemas for all endpoints

class CardCreate(BaseModel):
    front: str = Field(..., max_length=500)
    back: str = Field(..., max_length=500)
    hint: Optional[str] = Field(None, max_length=2000)
    tags: List[str] = Field(default_factory=list)
    language: str = Field(..., pattern='^(en|sk)$')
    
    @validator('front', 'back')
    def non_empty(cls, v):
        if not v.strip():
            raise ValueError('Cannot be empty')
        return v.strip()

class ReviewCreate(BaseModel):
    card_id: int
    rating: int = Field(..., ge=1, le=4)
    elapsed_days: int = Field(default=0, ge=0)
    time_spent_seconds: int = Field(default=0, ge=0)
```

---

### Error Responses

```json
{
  "error": "Card not found",
  "status": 404,
  "timestamp": "2026-05-28T10:30:00Z"
}
```

---

## Security Considerations

1. **CORS**: Allow only localhost:3000 during dev
2. **Rate limiting**: 100 requests/min per endpoint
3. **Input validation**: Pydantic schemas for all inputs
4. **SQL injection**: Use SQLAlchemy ORM (parameterized queries)
5. **API keys**: Stored in `.env`, never in code
6. **HTTPS**: Enforce in production (Docker nginx)

---

## Performance Optimization

### Database

- **Indexing**: User + language, user + tags, user + due_date
- **Pagination**: Always limit query results (default 50)
- **Connection pooling**: SQLAlchemy engine with pool_size=10

### Caching

- **TTS audio**: localStorage (browser cache)
- **Embeddings**: In-memory cache with TTL (1 hour)
- **Note content**: Cache in-memory while file unchanged

### LLM

- **Request batching**: Generate 3 questions at once, not 1-by-1
- **Prompt caching**: Reuse prompts for similar content
- **Rate limiting**: Queue requests, max 5/min

---

## Deployment Architecture (Docker)

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:latest
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: mysecretpassword
      POSTGRES_DB: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: ./backend
    environment:
      DATABASE_URL: postgresql://postgres:mysecretpassword@postgres:5432/postgres
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
    depends_on:
      - postgres
    ports:
      - "8000:8000"
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      VITE_API_URL: http://localhost:8000

volumes:
  postgres_data:
```

---

## Next Steps

1. Review this design for feedback/changes
2. Proceed to `tasks.md` for detailed implementation steps
3. Execute PHASE 1 (infrastructure setup)

