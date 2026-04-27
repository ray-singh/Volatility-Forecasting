-- Run this once in the Supabase SQL editor.

create table if not exists ingest_log (
    date         date        primary key,
    gcs_path     text        not null default '',
    row_count    integer     not null default 0,
    status       text        not null check (status in ('success', 'error')),
    error        text        not null default '',
    ingested_at  timestamptz not null default now()
);

-- Index for quick status lookups
create index if not exists ingest_log_status_idx on ingest_log (status);
