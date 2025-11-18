-- Vibe Photos Phase Final database design
-- Compatible with SQLite and PostgreSQL

-- ==========================================
-- Core tables
-- ==========================================

-- Photos table
CREATE TABLE photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- File information
    path TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_hash TEXT,  -- SHA256 for deduplication
    file_size INTEGER,

    -- Image attributes
    width INTEGER,
    height INTEGER,
    format TEXT,  -- jpg, png, etc.

    -- Timestamps
    taken_at TIMESTAMP,  -- EXIF timestamp
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- AI recognition results
    ai_category TEXT,
    ai_confidence REAL,
    ai_subcategory TEXT,
    ai_brand TEXT,
    ai_model TEXT,
    ai_attributes JSON,  -- {color, size, style, etc.}

    -- OCR results
    ocr_text TEXT,
    ocr_language TEXT,

    -- User metadata
    user_label TEXT,
    user_tags TEXT,
    user_notes TEXT,
    is_favorite BOOLEAN DEFAULT FALSE,
    is_hidden BOOLEAN DEFAULT FALSE,

    -- Processing status
    process_status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed
    needs_review BOOLEAN DEFAULT FALSE,
    review_reason TEXT,

    -- Vector embedding for similarity search
    embedding BLOB,

    -- Validation hints
    CHECK (ai_confidence >= 0 AND ai_confidence <= 1),
    CHECK (process_status IN ('pending', 'processing', 'completed', 'failed'))
);

-- ==========================================
-- Detection results
-- ==========================================

CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,

    -- Detection data
    object_class TEXT NOT NULL,
    confidence REAL NOT NULL,

    -- Bounding box
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,

    -- Additional attributes
    attributes JSON,

    -- Model metadata
    model_name TEXT,
    model_version TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CHECK (confidence >= 0 AND confidence <= 1)
);

-- ==========================================
-- Annotation management
-- ==========================================

CREATE TABLE annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,

    -- AI predictions
    ai_prediction TEXT,
    ai_confidence REAL,
    ai_suggestions JSON,  -- [{label, score}, ...]

    -- Human labels
    user_label TEXT NOT NULL,
    user_confirmed BOOLEAN DEFAULT TRUE,

    -- Batch tagging
    batch_applied BOOLEAN DEFAULT FALSE,
    batch_group_id TEXT,

    -- Training metadata
    used_for_training BOOLEAN DEFAULT FALSE,
    training_batch_id TEXT,

    -- Audit data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,

    CHECK (ai_confidence >= 0 AND ai_confidence <= 1)
);

-- ==========================================
-- Few-shot learning tables
-- ==========================================

CREATE TABLE custom_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Product metadata
    name TEXT UNIQUE NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,
    brand TEXT,

    -- Learning statistics
    sample_count INTEGER DEFAULT 0,
    min_confidence_threshold REAL DEFAULT 0.7,

    -- Model data
    prototype_vector BLOB,
    model_path TEXT,

    -- Performance metrics
    accuracy REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Extra attributes
    description TEXT,
    attributes JSON,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT,

    CHECK (min_confidence_threshold >= 0 AND min_confidence_threshold <= 1),
    CHECK (accuracy >= 0 AND accuracy <= 1)
);

CREATE TABLE product_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id INTEGER NOT NULL REFERENCES custom_products(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,

    -- Sample metadata
    is_positive BOOLEAN DEFAULT TRUE,
    feature_vector BLOB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(product_id, photo_id)
);

-- ==========================================
-- Search and organization
-- ==========================================

CREATE TABLE collections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    name TEXT UNIQUE NOT NULL,
    description TEXT,

    -- Collection type
    type TEXT DEFAULT 'manual',  -- manual, smart, timeline

    -- Smart collection rules
    smart_rules JSON,

    -- Counters
    photo_count INTEGER DEFAULT 0,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE collection_photos (
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,

    -- Ordering metadata
    sort_order INTEGER,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (collection_id, photo_id)
);

CREATE TABLE search_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    query TEXT NOT NULL,
    filters JSON,
    result_count INTEGER,

    -- Performance metrics
    response_time_ms INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id TEXT
);

-- ==========================================
-- System management
-- ==========================================

CREATE TABLE processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE,

    task_type TEXT NOT NULL,  -- detect, ocr, embed, thumbnail
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',  -- pending, processing, completed, failed

    -- Retry metadata
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,

    -- Error information
    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,

    CHECK (priority >= 1 AND priority <= 10),
    CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);

CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value TEXT,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==========================================
-- Indexes
-- ==========================================

CREATE INDEX idx_photos_path ON photos(path);
CREATE INDEX idx_photos_hash ON photos(file_hash);
CREATE INDEX idx_photos_category ON photos(ai_category);
CREATE INDEX idx_photos_user_label ON photos(user_label);
CREATE INDEX idx_photos_taken_at ON photos(taken_at);
CREATE INDEX idx_photos_imported_at ON photos(imported_at);
CREATE INDEX idx_photos_needs_review ON photos(needs_review);
CREATE INDEX idx_photos_status ON photos(process_status);

CREATE INDEX idx_detections_photo_id ON detections(photo_id);
CREATE INDEX idx_detections_class ON detections(object_class);

CREATE INDEX idx_annotations_photo_id ON annotations(photo_id);
CREATE INDEX idx_annotations_user_label ON annotations(user_label);
CREATE INDEX idx_annotations_training ON annotations(used_for_training);

CREATE INDEX idx_collection_photos_collection ON collection_photos(collection_id);
CREATE INDEX idx_collection_photos_photo ON collection_photos(photo_id);

CREATE INDEX idx_queue_status ON processing_queue(status, priority);
CREATE INDEX idx_queue_photo_id ON processing_queue(photo_id);

-- ==========================================
-- Views
-- ==========================================

CREATE VIEW photos_need_review AS
SELECT
    p.id,
    p.path,
    p.filename,
    p.ai_category,
    p.ai_confidence,
    p.review_reason,
    p.imported_at
FROM photos p
WHERE p.needs_review = TRUE
  AND p.process_status = 'completed'
ORDER BY p.imported_at DESC;

CREATE VIEW photo_statistics AS
SELECT
    COUNT(*) AS total_photos,
    COUNT(CASE WHEN ai_category IS NOT NULL THEN 1 END) AS categorized_photos,
    COUNT(CASE WHEN user_label IS NOT NULL THEN 1 END) AS labeled_photos,
    COUNT(CASE WHEN needs_review = TRUE THEN 1 END) AS review_needed,
    COUNT(CASE WHEN is_favorite = TRUE THEN 1 END) AS favorites,
    AVG(ai_confidence) AS avg_confidence
FROM photos;

CREATE VIEW category_statistics AS
SELECT
    ai_category AS category,
    COUNT(*) AS photo_count,
    AVG(ai_confidence) AS avg_confidence,
    COUNT(CASE WHEN user_label IS NOT NULL THEN 1 END) AS verified_count
FROM photos
WHERE ai_category IS NOT NULL
GROUP BY ai_category
ORDER BY photo_count DESC;

-- ==========================================
-- Triggers
-- ==========================================

CREATE TRIGGER update_photo_modified_time
AFTER UPDATE ON photos
BEGIN
    UPDATE photos SET modified_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER update_collection_count_insert
AFTER INSERT ON collection_photos
BEGIN
    UPDATE collections
    SET photo_count = photo_count + 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.collection_id;
END;

CREATE TRIGGER update_collection_count_delete
AFTER DELETE ON collection_photos
BEGIN
    UPDATE collections
    SET photo_count = photo_count - 1,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = OLD.collection_id;
END;

-- ==========================================
-- Seed data
-- ==========================================

INSERT INTO config (key, value, description) VALUES
('auto_classify', 'true', 'Automatically classify newly imported photos'),
('confidence_threshold', '0.7', 'Confidence threshold for AI classification'),
('enable_ocr', 'true', 'Enable OCR text extraction'),
('enable_few_shot', 'true', 'Enable few-shot learning workflow'),
('batch_size', '16', 'Batch size for processing jobs'),
('thumbnail_size', '512', 'Generated thumbnail size'),
('model_name', 'siglip2-base-patch16-224', 'Default perception model');

INSERT INTO collections (name, description, type) VALUES
('Recently Imported', 'Photos added within the last 7 days', 'smart'),
('Needs Review', 'Photos that require manual confirmation', 'smart'),
('Favorites', 'Photos flagged as favorites', 'smart');
