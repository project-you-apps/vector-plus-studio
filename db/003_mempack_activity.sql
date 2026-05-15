-- =============================================================================
-- 003 — Mempack activity log: append-only event stream + per-Mempack
--                              activity-logging on/off toggle.
-- =============================================================================
-- Run in Supabase SQL editor after 002_mempacks_schema.sql.
-- Idempotent: `create table if not exists`, `add column if not exists`,
-- `drop policy if exists` before recreating. Re-running is safe.
--
-- Concept: when an agent imprints a passage, edits Pattern I, or otherwise
-- mutates a Mempack, membot appends one row to public.mempack_activity. The
-- /app dashboard polls GET /api/mempack/<id>/activity?since=<ts> to surface
-- a "what did my agent do" feed to the user.
--
-- Pull-only model: agent writes, user reads when their tab is open. No push,
-- no notifications. The log is a property of the hosting environment (VPS) —
-- when a Mempack moves to a different host, its activity stays behind.
--
-- Per-Mempack opt-out: mempacks.activity_logging_enabled flag (default true
-- for VPS-hosted Mempacks; the /app UI exposes it as a checkbox). Imprint
-- code checks the flag before inserting. Local installs would default it
-- false — that's app-level policy, not enforced in SQL.
--
-- Andy + Claude 2026-05-14.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- mempacks: per-cart logging toggle
-- ---------------------------------------------------------------------------
alter table public.mempacks
  add column if not exists activity_logging_enabled boolean not null default true;


-- ---------------------------------------------------------------------------
-- mempack_activity — one row per logged event. Append-only from the user
-- API perspective (no update/delete policy); only service role writes.
-- ---------------------------------------------------------------------------
create table if not exists public.mempack_activity (
  id              uuid primary key default gen_random_uuid(),
  mempack_id      uuid references public.mempacks(id) on delete cascade not null,

  -- 'imprint' | 'pattern_i_update' | 'mount' | 'copy_in' | 'copy_out'
  -- | 'pattern_update' | 'pattern_delete' | 'create' | other
  event_type      text not null,

  -- Human-readable one-liner for the feed UI ("imprinted 412B at idx 47")
  summary         text not null,

  -- Affected pattern index, when applicable (imprint, pattern_update, etc.)
  pattern_idx     int,

  -- Structured payload for the UI to render rich rows: src_cart, src_idx,
  -- byte_size, agent_user_agent, etc. Open schema by event_type.
  metadata        jsonb,

  -- Optional identifier for which agent did the thing (mcp client name,
  -- session id prefix, "browser" for /app UI actions). Nullable.
  agent_label     text,

  created_at      timestamptz default now()
);

-- Primary access pattern: "give me events for this Mempack since timestamp X"
create index if not exists mempack_activity_mempack_idx
  on public.mempack_activity (mempack_id, created_at desc);

-- Secondary: cross-Mempack recency for admin/diagnostic dashboards
create index if not exists mempack_activity_recent_idx
  on public.mempack_activity (created_at desc);

-- Filter by event type within a Mempack (e.g. "show only imprints")
create index if not exists mempack_activity_type_idx
  on public.mempack_activity (mempack_id, event_type, created_at desc);


-- ---------------------------------------------------------------------------
-- RLS — users read their own Mempacks' activity. Only service role writes.
-- ---------------------------------------------------------------------------
alter table public.mempack_activity enable row level security;

drop policy if exists "users see own activity" on public.mempack_activity;
create policy "users see own activity" on public.mempack_activity
  for select using (exists (
    select 1 from public.mempacks m
    where m.id = mempack_activity.mempack_id and m.user_id = auth.uid()
  ));

-- No insert/update/delete policies — append-only from the user API surface.
-- Service role bypasses RLS for membot's append_activity() writes.


-- =============================================================================
-- Smoke checks (run after the above to confirm everything is in place)
-- =============================================================================
-- -- New table exists + RLS on?
-- select tablename, rowsecurity from pg_tables
--   where schemaname='public' and tablename='mempack_activity';
--   -- expect: mempack_activity, t
--
-- -- Logging toggle column added?
-- select column_name, data_type, column_default from information_schema.columns
--   where table_schema='public' and table_name='mempacks'
--     and column_name='activity_logging_enabled';
--   -- expect: activity_logging_enabled, boolean, true
--
-- -- Policy registered?
-- select tablename, policyname from pg_policies
--   where schemaname='public' and tablename='mempack_activity';
--   -- expect 1: "users see own activity"
--
-- -- Indexes present?
-- select indexname from pg_indexes
--   where schemaname='public' and tablename='mempack_activity';
--   -- expect 4 (pkey + 3 secondary)
