-- =============================================================================
-- 001 — OAuth schema: extend Heartbeat's profiles + add user_carts + saved_searches
-- =============================================================================
-- Run in Supabase SQL editor (one-shot paste).
-- Idempotent: uses `add column if not exists`, `create ... if not exists`, and
-- `drop policy if exists` before recreating. Re-running is safe.
--
-- Source: docs/vps-internal/OAUTH-PRE-SCOPE.md, Blocks B0.3 + E.
--
-- IMPORTANT — Path B (Andy 2026-05-11):
-- A `public.profiles` table already exists from Heartbeat dev with shape:
--   id, username, full_name, avatar_url, updated_at, email
-- and an existing `handle_new_user` trigger that auto-creates profile rows.
-- This migration EXTENDS that table; it does NOT recreate it or replace
-- Heartbeat's trigger. New columns are additive + nullable so Heartbeat's
-- existing flow keeps working untouched.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- profiles -- additive extension of Heartbeat's existing table
-- ---------------------------------------------------------------------------
-- New columns we need for the Waving Cat product stack:
--   display_name   -- preferred chat-label; falls back to full_name when null
--   gravatar_email -- email to MD5-hash for the gravatar fallback (client-side)
--   apps_list      -- which Waving Cat apps the user has touched: {'vps','membot',...}
--   created_at     -- profile-row creation timestamp (Heartbeat only tracked updated_at)
alter table public.profiles add column if not exists display_name    text;
alter table public.profiles add column if not exists gravatar_email  text;
alter table public.profiles add column if not exists apps_list       text[] default '{}';
alter table public.profiles add column if not exists created_at      timestamptz default now();

-- Backfill created_at for existing rows that pre-date this column. For rows
-- where updated_at exists, copy it as a best-effort creation timestamp.
update public.profiles
  set created_at = coalesce(updated_at, now())
  where created_at is null;

-- NOTE: Heartbeat's existing RLS policies govern profiles. We do NOT add or
-- modify policies here. If we eventually need a "public read profile basics"
-- policy (for shared-cart owner display etc.), add it in a follow-up migration.


-- ---------------------------------------------------------------------------
-- handle_new_user -- upgrade Heartbeat's trigger to also populate the OAuth
-- metadata fields (full_name, display_name, avatar_url). Heartbeat's existing
-- behavior (insert id + email) is preserved; this is purely additive.
-- ---------------------------------------------------------------------------
-- Heartbeat's original was:
--   insert into public.profiles (id, email) values (new.id, new.email);
-- which left full_name/avatar_url NULL on every new user. This version pulls
-- them from auth.users.raw_user_meta_data where the OAuth provider populated
-- them (Google uses 'full_name', GitHub uses 'name', both use 'avatar_url').
--
-- on conflict (id) do nothing matches Heartbeat's implicit behavior: if a
-- profile row already exists for this user id, leave it alone (don't clobber
-- a user's manual edits to display_name etc. on a re-signin event).
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email, full_name, display_name, avatar_url)
  values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name'),
    coalesce(new.raw_user_meta_data->>'full_name', new.raw_user_meta_data->>'name'),
    new.raw_user_meta_data->>'avatar_url'
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

-- NOTE: existing trigger `on_auth_user_created` (or whatever Heartbeat named
-- it) already points at `handle_new_user`. The CREATE OR REPLACE FUNCTION
-- above swaps the body underneath the existing trigger — no DROP/RECREATE
-- of the trigger itself needed.


-- ---------------------------------------------------------------------------
-- One-shot backfill for existing rows that pre-date the trigger upgrade.
-- Non-destructive: only fills NULLs, never overwrites existing values.
-- ---------------------------------------------------------------------------
update public.profiles p
  set
    full_name    = coalesce(p.full_name,    u.raw_user_meta_data->>'full_name', u.raw_user_meta_data->>'name'),
    display_name = coalesce(p.display_name, p.full_name,                         u.raw_user_meta_data->>'full_name', u.raw_user_meta_data->>'name'),
    avatar_url   = coalesce(p.avatar_url,   u.raw_user_meta_data->>'avatar_url')
  from auth.users u
  where p.id = u.id
    and (p.full_name is null or p.display_name is null or p.avatar_url is null);


-- Convenience helper: append an app identifier to the current user's apps_list
-- if not already there. Apps call this on first authenticated access so we
-- get cross-app usage analytics and can power the "your apps" dashboard.
--   select public.track_app_usage('vps');
create or replace function public.track_app_usage(app_name text)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  update public.profiles
    set apps_list = array_append(coalesce(apps_list, '{}'), app_name)
    where id = auth.uid()
      and not (app_name = any(coalesce(apps_list, '{}')));
end;
$$;


-- ---------------------------------------------------------------------------
-- user_carts -- VPS-specific: which private carts a user owns
-- ---------------------------------------------------------------------------
create table if not exists public.user_carts (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid references auth.users(id) on delete cascade not null,
  cart_filename   text not null,            -- name on disk in user's private dir
  display_name    text not null,            -- user-facing label
  size_bytes      bigint not null,
  pattern_count   int,
  created_at      timestamptz default now(),
  unique (user_id, cart_filename)
);

alter table public.user_carts enable row level security;

drop policy if exists "users see own carts"     on public.user_carts;
drop policy if exists "users insert own carts"  on public.user_carts;
drop policy if exists "users update own carts"  on public.user_carts;
drop policy if exists "users delete own carts"  on public.user_carts;

create policy "users see own carts"     on public.user_carts for select using (auth.uid() = user_id);
create policy "users insert own carts"  on public.user_carts for insert with check (auth.uid() = user_id);
create policy "users update own carts"  on public.user_carts for update using (auth.uid() = user_id);
create policy "users delete own carts"  on public.user_carts for delete using (auth.uid() = user_id);


-- ---------------------------------------------------------------------------
-- saved_searches -- VPS-specific: queries a user has bookmarked
-- ---------------------------------------------------------------------------
create table if not exists public.saved_searches (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid references auth.users(id) on delete cascade not null,
  cart_id         uuid references public.user_carts(id) on delete cascade,
                  -- nullable: a saved search can reference a bundled or sandbox
                  -- cart by name in metadata rather than by uuid.
  query_text      text not null,
  search_mode     text not null,
                  -- 'smart', 'pure_brain', 'fast', 'associate', 'hamming_blend', etc.
  created_at      timestamptz default now()
);

alter table public.saved_searches enable row level security;

drop policy if exists "users see own searches"     on public.saved_searches;
drop policy if exists "users insert own searches"  on public.saved_searches;
drop policy if exists "users delete own searches"  on public.saved_searches;

create policy "users see own searches"     on public.saved_searches for select using (auth.uid() = user_id);
create policy "users insert own searches"  on public.saved_searches for insert with check (auth.uid() = user_id);
create policy "users delete own searches"  on public.saved_searches for delete using (auth.uid() = user_id);


-- =============================================================================
-- Smoke checks (run after the above to confirm everything is in place)
-- =============================================================================
-- -- New profile columns present?
-- select column_name, data_type, column_default
--   from information_schema.columns
--   where table_schema='public' and table_name='profiles'
--   order by ordinal_position;
--   -- expect: id, username, full_name, avatar_url, updated_at, email,
--   --         display_name, gravatar_email, apps_list, created_at
--
-- -- New tables exist + RLS on?
-- select tablename, rowsecurity from pg_tables
--   where schemaname='public' and tablename in ('user_carts','saved_searches');
--   -- both should show rowsecurity=t
--
-- -- Policies registered?
-- select policyname, tablename from pg_policies
--   where schemaname='public' and tablename in ('user_carts','saved_searches')
--   order by tablename, policyname;
--   -- expect 4 rows for user_carts, 3 for saved_searches
--
-- -- track_app_usage callable?
-- select public.track_app_usage('vps');
--   -- (no error = function exists; check your own row's apps_list afterward)
