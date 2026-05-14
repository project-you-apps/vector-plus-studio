-- =============================================================================
-- 002 — Mempack schema: cart-row + per-pattern H-block normalized table
--                       + auto-provision audit log + Storage-bucket policies
-- =============================================================================
-- Run in Supabase SQL editor after 001_oauth_schema.sql.
-- Idempotent: uses `create table if not exists`, `drop policy if exists` before
-- recreating, and `create or replace function` throughout. Re-running is safe.
--
-- Concept: a Mempack is a per-agent writable brain cartridge. Its lightweight
-- queryable metadata lives in Postgres (this migration); the heavy binary
-- content (embeddings + texts + manifest sidecar) lives in Supabase Storage
-- as a `.cart.npz` blob, addressed by `storage_path`.
--
-- Per-pattern H-block fields (the cart-format hippocampus struct, distinct
-- from the physics-layer H-row) are exploded into columns on `mempack_patterns`
-- so Postgres can index/query across them without unblobbing the cart.
--
-- See docs/PATTERN-ANATOMY.md §3 "The Hippocampus: H-row and H-block" for the
-- canonical 64-byte H-block layout this table mirrors.
--
-- Andy + Claude 2026-05-13.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- mempacks — one row per cart. Points at a Storage blob holding the .cart.npz.
-- ---------------------------------------------------------------------------
create table if not exists public.mempacks (
  id                 uuid primary key default gen_random_uuid(),
  user_id            uuid references auth.users(id) on delete cascade not null,

  -- Identity
  name               text not null,                       -- e.g. "primary"
  cart_type          text not null default 'agent-memory',-- 'agent-memory' | 'knowledge' | etc.

  -- Storage blob pointer
  storage_bucket     text not null default 'mempacks',
  storage_path       text not null,                       -- <user_id>/<name>.cart.npz
  storage_status     text not null default 'pending',     -- pending | ready | orphaned
                                                          -- pending = row inserted, blob upload not yet confirmed
                                                          -- ready   = blob present, mountable
                                                          -- orphaned= blob delete pending GC; row kept for audit

  -- Cart-level metadata (mirrored from the cart's manifest sidecar)
  pattern_count      int not null default 0,              -- N patterns in the cart
  size_bytes         bigint not null default 0,           -- blob byte size
  briefing           text,                                -- briefing template (Pattern 0 field)
  pattern_i_text     text,                                -- Pattern I (agent behavioral instructions)
  manifest           jsonb,                               -- full Pattern 0 manifest snapshot
  format_version     smallint default 1,                  -- H-block format version (see PATTERN-ANATOMY)

  -- Lifecycle
  created_at         timestamptz default now(),
  updated_at         timestamptz default now(),
  last_mounted_at    timestamptz,
  mount_count        int not null default 0,

  unique (user_id, name)
);

create index if not exists mempacks_user_idx     on public.mempacks (user_id);
create index if not exists mempacks_status_idx   on public.mempacks (storage_status) where storage_status != 'ready';
create index if not exists mempacks_lastmount_idx on public.mempacks (last_mounted_at desc);


-- ---------------------------------------------------------------------------
-- mempack_patterns — one row per pattern in a Mempack. H-block fields
-- exploded into native Postgres columns for SQL-side indexing/filtering.
--
-- The cart blob remains authoritative for embedding + full text; this table
-- is a queryable projection of the cart's H-block array + a small text preview.
-- ---------------------------------------------------------------------------
create table if not exists public.mempack_patterns (
  id              uuid primary key default gen_random_uuid(),
  mempack_id      uuid references public.mempacks(id) on delete cascade not null,
  pattern_idx     int  not null,                          -- 0-based position in cart arrays

  -- H-block fields (canonical 12-field 64-byte format, mirroring
  -- vector-plus-studio-repo/api/cartridge_io.py:HIPPO_FORMAT)
  pattern_id      bigint,                                 -- uint32, 1-based; 0 = none
  format_version  smallint,                               -- uint8
  cartridge_type  smallint,                               -- uint8
  parent_ptr      bigint,                                 -- uint32 backward link (1-based)
  child_ptr       bigint,                                 -- uint32 forward link (1-based)
  sibling_ptr     bigint,                                 -- uint32 sideways link (0 for linear carts)
  source_hash     bigint,                                 -- uint32 hash of source filename
  sequence_num    int,                                    -- uint16
  ts_unix         bigint,                                 -- uint32 Unix seconds
  flags           smallint not null default 0,            -- packed 8-bit flags
  perms_byte      smallint not null default 0,            -- packed RWX (R=1, W=2, X=4)

  -- Generated columns over the bit-packed `flags` byte (auto-computed, indexable).
  -- See docs/PATTERN-ANATOMY.md §3 for the canonical bit layout.
  tombstone       boolean generated always as ((flags & 1)  <> 0) stored,  -- bit 0
  pinned          boolean generated always as ((flags & 2)  <> 0) stored,  -- bit 1
  has_parent      boolean generated always as ((flags & 4)  <> 0) stored,  -- bit 2
  has_child       boolean generated always as ((flags & 8)  <> 0) stored,  -- bit 3
  has_sibling     boolean generated always as ((flags & 16) <> 0) stored,  -- bit 4
  -- perish_class = bits 5-6: 0=volatile, 1=replaceable, 2=archival, 3=reserved
  perish_class    smallint generated always as ((flags >> 5) & 3) stored,

  -- Generated columns over the bit-packed `perms_byte`
  perm_r          boolean generated always as ((perms_byte & 1) <> 0) stored,
  perm_w          boolean generated always as ((perms_byte & 2) <> 0) stored,
  perm_x          boolean generated always as ((perms_byte & 4) <> 0) stored,

  -- SQL-side text projection (full text + embedding stay in the cart blob)
  text_preview    text,                                   -- first ~200 chars for keyword pre-filter
  text_length     int,                                    -- full text byte length

  -- Original 64-byte H-block preserved verbatim. Lets membot round-trip read
  -- without reconstructing from columns. Authoritative source of truth for
  -- any field we haven't exploded yet.
  hippocampus_raw bytea,

  created_at      timestamptz default now(),
  updated_at      timestamptz default now(),

  unique (mempack_id, pattern_idx)
);

-- Indexes for likely query patterns
create index if not exists mempack_patterns_mempack_idx on public.mempack_patterns (mempack_id, pattern_idx);
create index if not exists mempack_patterns_active_idx  on public.mempack_patterns (mempack_id) where not tombstone;
create index if not exists mempack_patterns_source_idx  on public.mempack_patterns (source_hash);
create index if not exists mempack_patterns_ts_idx      on public.mempack_patterns (ts_unix desc);
create index if not exists mempack_patterns_perish_idx  on public.mempack_patterns (mempack_id, perish_class);
create index if not exists mempack_patterns_pinned_idx  on public.mempack_patterns (mempack_id) where pinned;


-- ---------------------------------------------------------------------------
-- mempack_provisions_log — audit trail for auto-provision events.
-- One row per provisioning attempt (success or failure) so we can answer
-- "did this user get auto-provisioned and when?" without fishing in app logs.
-- ---------------------------------------------------------------------------
create table if not exists public.mempack_provisions_log (
  id              uuid primary key default gen_random_uuid(),
  user_id         uuid references auth.users(id) on delete cascade not null,
  mempack_id      uuid references public.mempacks(id) on delete set null,
  trigger_source  text not null,                          -- 'lazy_list' | 'manual' | 'signup_trigger' | etc.
  outcome         text not null,                          -- 'created' | 'already_existed' | 'failed'
  error_message   text,                                   -- populated on failure
  fired_at        timestamptz default now()
);

create index if not exists mempack_provisions_user_idx on public.mempack_provisions_log (user_id, fired_at desc);


-- ---------------------------------------------------------------------------
-- Per-user storage caps. Enforced via BEFORE INSERT/UPDATE trigger.
-- Cap tiers are stored on the profile (apps_list pattern or a dedicated tier
-- field — for v1 we hardcode the free-tier default and gate Pro/Enterprise
-- via app-level checks).
-- ---------------------------------------------------------------------------
-- v1 defaults (tune later via a `mempack_tiers` table when product policy firms):
--   Free:       1 Mempack, 10 MB per Mempack
--   Pro:        5 Mempacks, 100 MB per Mempack
--   Enterprise: unlimited
--
-- For this migration we only enforce the FREE-tier cap. Higher tiers will land
-- with a `profiles.mempack_tier` column + tier-keyed cap lookup.
create or replace function public.check_mempack_size_cap()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  v_count   int;
  v_total   bigint;
  v_count_cap int := 1;            -- free-tier Mempack count
  v_size_cap  bigint := 10485760;  -- free-tier 10 MB per Mempack
begin
  -- Count cap: enforced only on INSERT (size cap fires on both)
  if tg_op = 'INSERT' then
    select count(*) into v_count from public.mempacks
      where user_id = new.user_id and id != new.id;
    if v_count >= v_count_cap then
      raise exception 'Mempack count cap exceeded for user % (% Mempacks, cap %)',
        new.user_id, v_count, v_count_cap
        using errcode = '23514';
    end if;
  end if;

  -- Size cap: per-Mempack
  if new.size_bytes > v_size_cap then
    raise exception 'Mempack size cap exceeded (% bytes > % cap)',
      new.size_bytes, v_size_cap
      using errcode = '23514';
  end if;

  return new;
end;
$$;

drop trigger if exists trg_check_mempack_size_cap on public.mempacks;
create trigger trg_check_mempack_size_cap
  before insert or update on public.mempacks
  for each row execute function public.check_mempack_size_cap();


-- ---------------------------------------------------------------------------
-- RLS — users see/edit only their own Mempacks. Membot's service role
-- bypasses RLS (service role inherits from postgres role, ignores policies).
-- ---------------------------------------------------------------------------
alter table public.mempacks              enable row level security;
alter table public.mempack_patterns      enable row level security;
alter table public.mempack_provisions_log enable row level security;

-- mempacks
drop policy if exists "users see own mempacks"     on public.mempacks;
drop policy if exists "users insert own mempacks"  on public.mempacks;
drop policy if exists "users update own mempacks"  on public.mempacks;
drop policy if exists "users delete own mempacks"  on public.mempacks;

create policy "users see own mempacks"    on public.mempacks for select using (auth.uid() = user_id);
create policy "users insert own mempacks" on public.mempacks for insert with check (auth.uid() = user_id);
create policy "users update own mempacks" on public.mempacks for update using (auth.uid() = user_id);
create policy "users delete own mempacks" on public.mempacks for delete using (auth.uid() = user_id);

-- mempack_patterns — access gated through parent Mempack's user_id
drop policy if exists "users see own patterns"     on public.mempack_patterns;
drop policy if exists "users insert own patterns"  on public.mempack_patterns;
drop policy if exists "users update own patterns"  on public.mempack_patterns;
drop policy if exists "users delete own patterns"  on public.mempack_patterns;

create policy "users see own patterns" on public.mempack_patterns
  for select using (exists (
    select 1 from public.mempacks m
    where m.id = mempack_patterns.mempack_id and m.user_id = auth.uid()
  ));

create policy "users insert own patterns" on public.mempack_patterns
  for insert with check (exists (
    select 1 from public.mempacks m
    where m.id = mempack_patterns.mempack_id and m.user_id = auth.uid()
  ));

create policy "users update own patterns" on public.mempack_patterns
  for update using (exists (
    select 1 from public.mempacks m
    where m.id = mempack_patterns.mempack_id and m.user_id = auth.uid()
  ));

create policy "users delete own patterns" on public.mempack_patterns
  for delete using (exists (
    select 1 from public.mempacks m
    where m.id = mempack_patterns.mempack_id and m.user_id = auth.uid()
  ));

-- mempack_provisions_log — users see only their own log rows; only service role inserts
drop policy if exists "users see own provisions" on public.mempack_provisions_log;
create policy "users see own provisions" on public.mempack_provisions_log
  for select using (auth.uid() = user_id);


-- =============================================================================
-- Storage bucket setup (run as a separate one-shot in Supabase Storage UI
-- or via the `storage` schema directly — Postgres can't enforce all of
-- Supabase Storage's bucket-level policies in a normal migration).
-- =============================================================================
--
-- 1. Create the bucket:
--      In Supabase Studio → Storage → New bucket
--      Name:   mempacks
--      Public: false (private — accessed only via signed URLs or service role)
--
-- 2. Bucket-level RLS policies (paste into the Storage policy editor):
--
--    -- Users can read their own Mempack blobs (path prefixed with their UUID)
--    create policy "users read own mempack blobs"
--      on storage.objects for select
--      using ( bucket_id = 'mempacks' and (storage.foldername(name))[1] = auth.uid()::text );
--
--    -- Users can write to their own folder
--    create policy "users write own mempack blobs"
--      on storage.objects for insert
--      with check ( bucket_id = 'mempacks' and (storage.foldername(name))[1] = auth.uid()::text );
--
--    -- Users can update their own blobs (rewrites on cart update)
--    create policy "users update own mempack blobs"
--      on storage.objects for update
--      using ( bucket_id = 'mempacks' and (storage.foldername(name))[1] = auth.uid()::text );
--
--    -- Users can delete their own blobs
--    create policy "users delete own mempack blobs"
--      on storage.objects for delete
--      using ( bucket_id = 'mempacks' and (storage.foldername(name))[1] = auth.uid()::text );
--
-- 3. Membot uses the SERVICE ROLE key (bypasses bucket RLS) to read/write any
--    user's blob, since it's acting on behalf of the agent at mount time.
--    Service-role key lives in /opt/membot/.env on the droplet.


-- =============================================================================
-- Smoke checks (run after the above to confirm everything is in place)
-- =============================================================================
-- -- New tables exist + RLS on?
-- select tablename, rowsecurity from pg_tables
--   where schemaname='public' and tablename in ('mempacks','mempack_patterns','mempack_provisions_log');
--   -- all three should show rowsecurity=t
--
-- -- Policies registered?
-- select tablename, policyname from pg_policies
--   where schemaname='public' and tablename in ('mempacks','mempack_patterns','mempack_provisions_log')
--   order by tablename, policyname;
--   -- expect 4 on mempacks, 4 on mempack_patterns, 1 on mempack_provisions_log
--
-- -- Size-cap trigger present?
-- select tgname from pg_trigger
--   where tgrelid = 'public.mempacks'::regclass and not tgisinternal;
--   -- expect: trg_check_mempack_size_cap
--
-- -- Generated columns wired correctly? Insert a test row and check derived fields:
-- with test_mp as (
--   insert into public.mempacks (user_id, name, storage_path)
--   values (auth.uid(), 'cap_test', auth.uid()::text || '/cap_test.cart.npz')
--   returning id
-- )
-- insert into public.mempack_patterns (mempack_id, pattern_idx, flags, perms_byte)
-- values ((select id from test_mp), 0, 0x62, 0x03)  -- pinned + archival + RW
-- returning pattern_idx, tombstone, pinned, perish_class, perm_r, perm_w, perm_x;
--   -- expect: 0, false, true, 3, true, true, false (perish_class=3 = (0x62>>5)&3 = 11b = 3)
--   -- Actually 0x62 = 0110 0010 → bit 1 (pinned) + bits 5-6 = 11 = perish=3 (reserved).
--   -- Use 0x42 for archival (bit 6 only) + pinned: 0100 0010 → pinned=t, perish=2.
