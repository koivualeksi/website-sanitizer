CREATE TABLE IF NOT EXISTS pages (
    id          SERIAL PRIMARY KEY,
    url         TEXT NOT NULL UNIQUE,
    html        TEXT,
    markdown    TEXT,
    status      TEXT,
    error       TEXT,
    scraped_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT now(),
    dataset     TEXT
);

CREATE TABLE IF NOT EXISTS annotations (
    id          SERIAL PRIMARY KEY,
    page_id     INTEGER NOT NULL REFERENCES pages(id) UNIQUE,
    ranges      JSONB NOT NULL DEFAULT '[]',
    source      TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual', 'llm')),
    validated   BOOLEAN NOT NULL DEFAULT FALSE,
    skipped     BOOLEAN NOT NULL DEFAULT FALSE,
    has_cookies BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);
