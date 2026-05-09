create table if not exists ingest_log (
    date date primary key,
    gcs_path text not null default '',
    row_count integer not null default 0,
    status text not null check (status in ('success', 'error')),
    error text not null default '',
    ingested_at timestamptz not null default now()
);

-- Index for quick status lookups
create index if not exists ingest_log_status_idx on ingest_log (status);

create table if not exists retrain_log (
    id bigint generated always as identity primary key,
    retrained_at timestamptz not null default now(),
    days_used integer not null default 0,
    lgbm_rmse double precision,
    status text not null check (status in ('success', 'error', 'skipped')),
    error text not null default ''
);

create index if not exists retrain_log_retrained_at_idx on retrain_log (retrained_at desc);
