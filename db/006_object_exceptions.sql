/*
-- =============================================================================
-- 006 — per-document exceptions, and the per-cart inheritance toggle
-- =============================================================================
-- Run in Supabase SQL editor (one-shot paste).
-- Idempotent: `add column if not exists`, `create table if not exists`,
-- `drop policy if exists` before each `create policy`. Re-running is safe.
--
-- WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT
-- --------------------------------------------------
-- 004 answers "may this seat open this cart". This answers "...and may they
-- read THIS DOCUMENT inside it", which is the thing Andy asked for on 08-05 and
-- Dennis asked about for the fleet on 08-08.
--
-- It stores ONLY the per-seat exceptions. Document EXPOSURE — "this document
-- requires editor, whoever you are" — is deliberately NOT here: that is a
-- property of the document, true on any server, so it belongs in the cart
-- (Pattern-0), not in one deployment's database. The split we settled on
-- 08-10: THE CART CARRIES PROPERTIES OF THE DOCUMENT, THE DATABASE CARRIES
-- RELATIONSHIPS BETWEEN PEOPLE AND DOCUMENTS. Copy a cart to another company and
-- "this is a compensation record" should survive the trip; "Betty may not read
-- it" should not, because there is no Betty there and that uuid either means
-- nothing or means somebody else.
--
-- WHAT A DOCUMENT IS
-- -------------------
-- `source_hash`: the uint32 at h-block offset 18, a deterministic md5 of the
-- source filename, written by membot's cartridge_builder since long before any
-- of this. Verified 2026-08-10: each Redwood office cart carries exactly 300
-- distinct values over ~1,400 passages and all 300 resolve back to real
-- filenames. So this needs no format change, no rebuild, and no migration.
--
-- Stored as BIGINT, not INT. A uint32 reaches 4,294,967,295 and Postgres `int`
-- stops at 2,147,483,647 — roughly half of all real source hashes would
-- overflow. Not hypothetical: `redwood-company` already contains hashes above
-- 3.1e9.
--
-- WHY THIS TABLE IS STRICTER THAN 004
-- ------------------------------------
-- In 004 a grantee may SELECT their own grant row. Here they may not, and the
-- difference is deliberate: a `deny` row on a document is itself evidence that
-- the document exists. Letting Betty read "you are denied on document 9a3f…"
-- would leak the existence — and, with the cart in hand, the identity — of
-- exactly the material the row is protecting. Resolution goes through the
-- SECURITY DEFINER function below, so no client ever needs to read this table.
--
-- The same reasoning is why the function returns only the CALLER's own rows and
-- never says how many other exceptions exist.
-- =============================================================================
*/


/*
-- ---------------------------------------------------------------------------
-- The per-cart inheritance toggle (Andy, 2026-08-09)
-- ---------------------------------------------------------------------------
-- "Toggle On, documents inherit. Toggle off, documents require explicit
--  assignment. This puts it on the system admin to decide how secure they want
--  their carts. It might get annoying to users to have to set permissions on
--  every file in a cart so inheriting is good. But when doing employee records
--  or say a health records cart and a new patient's file is added then since
--  it's got PII they'll want it default explicit-add."
--
-- Backfills TRUE, which is exactly today's behaviour: every existing document in
-- every existing cart is visible to everyone who can open the cart. Backfilling
-- FALSE would black out every cart we have on the next deploy.
--
-- ⚠ THE DEFAULT HERE IS FOR BACKFILL ONLY. New carts must ASK. A buried default
-- is discovered during the incident, so the API should require an explicit value
-- at cart creation — one question, in the user's terms, at the moment they have
-- the most context about what the cart is for.
*/
alter table public.user_carts
  add column if not exists inherit_new_documents boolean not null default true;

/*
-- When the toggle was last flipped. This is what makes a grandfathered document
-- identifiable: access granted by inheritance BEFORE this timestamp was arrived
-- at under the previous policy. Recorded rather than derived — deriving "was
-- this deliberate?" from today's toggle is the event-time-vs-ingest-time bug in
-- another costume, and this project has now shipped that bug in three systems.
*/
alter table public.user_carts
  add column if not exists inherit_changed_at timestamptz;


/*
-- ---------------------------------------------------------------------------
-- The exceptions themselves — sparse, one row per (cart, document, seat)
-- ---------------------------------------------------------------------------
-- A 39,000-passage cart with three restricted documents costs three rows, not
-- 39,000. That sparseness is the reason permissions attach to documents rather
-- than passages: a control nobody can maintain is a control that does not exist.
*/
create table if not exists public.cart_object_exceptions (
  id            uuid primary key default gen_random_uuid(),
  cart_id       uuid   references public.user_carts(id) on delete cascade not null,
  source_hash   bigint not null,
  grantee_id    uuid   references auth.users(id)        on delete cascade not null,
  /*
  -- 'deny' shares this column with the grantable levels on purpose. An ABSENT
  -- row means "nobody has said anything about this seat and this document";
  -- 'deny' means "somebody decided no". If those were spelled the same way the
  -- table could never distinguish undecided from decided-against — the same
  -- fail-open collapse that made `cart_access_for()` necessary one layer up.
  */
  access_level  text not null check (access_level in ('viewer','commenter','editor','deny')),
  granted_by    uuid   references auth.users(id)        on delete set null,
  /*
  -- Why the exception exists, in the owner's words. Optional, and worth having:
  -- an access-control row with no rationale is the thing nobody dares remove two
  -- years later, so the table slowly accretes restrictions no one can justify.
  */
  note          text,
  created_at    timestamptz default now(),
  updated_at    timestamptz default now(),
  unique (cart_id, source_hash, grantee_id)
);

create index if not exists cart_object_exceptions_seat_idx
  on public.cart_object_exceptions (grantee_id, cart_id);
create index if not exists cart_object_exceptions_cart_idx
  on public.cart_object_exceptions (cart_id);


alter table public.cart_object_exceptions enable row level security;

/*
-- SELECT: OWNER ONLY. See "why this table is stricter than 004" above — a deny
-- row is evidence the document exists, so the subject of a restriction must not
-- be able to read it. Seats learn their own access through the function below,
-- which tells them what they MAY see and never enumerates what they may not.
*/
drop policy if exists "owner sees object exceptions" on public.cart_object_exceptions;
create policy "owner sees object exceptions" on public.cart_object_exceptions
  for select using (public.owns_cart(cart_id));

/*
-- INSERT / UPDATE / DELETE: OWNER ONLY, and never about themselves.
--
-- `grantee_id <> auth.uid()` mirrors 004. Here it also prevents an owner from
-- locking themselves out of their own cart one document at a time — a state
-- with no recovery path through the UI, since fixing it would require the very
-- access they just removed.
*/
drop policy if exists "owner sets object exceptions"    on public.cart_object_exceptions;
drop policy if exists "owner updates object exceptions" on public.cart_object_exceptions;
drop policy if exists "owner clears object exceptions"  on public.cart_object_exceptions;
create policy "owner sets object exceptions" on public.cart_object_exceptions
  for insert with check (public.owns_cart(cart_id) and grantee_id <> auth.uid());
create policy "owner updates object exceptions" on public.cart_object_exceptions
  for update using (public.owns_cart(cart_id));
create policy "owner clears object exceptions" on public.cart_object_exceptions
  for delete using (public.owns_cart(cart_id));


/*
-- ---------------------------------------------------------------------------
-- object_access_for(): the caller's whole picture for ONE cart, in ONE call
-- ---------------------------------------------------------------------------
-- Returns jsonb:
--   { "inherit": bool,
--     "inherit_changed_at": timestamptz | null,
--     "exceptions": { "<source_hash>": "<level|deny>", ... } }
--
-- ONE CALL PER (SEAT, CART), NOT ONE PER DOCUMENT. A search returns passages
-- from many documents; asking per document would put a network round-trip
-- inside the result loop. The exception set is sparse by construction, so the
-- whole of it is smaller than a single page of results, and the caller caches it
-- exactly like `cart_guard` caches the mount decision.
--
-- SECURITY DEFINER for the same reason as 005: RLS on this table is owner-only,
-- so a caller-scoped client asking about its own restrictions would get zero
-- rows and read that as "unrestricted" — fail-open, and invisible in any test
-- run against a live database. The function is narrow by construction: it
-- returns the CALLER's own exceptions and nothing about anyone else's.
--
-- Returns `inherit: true` with no exceptions for an unregistered cart, which is
-- the legacy-cart rule from 004 carried down a level: a cart nobody has claimed
-- governs nothing, and every document in it behaves as it always has.
*/
create or replace function public.object_access_for(p_filename text)
returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
  v_seat    uuid := auth.uid();
  v_cart    uuid;
  v_inherit boolean;
  v_changed timestamptz;
  v_exc     jsonb;
begin
  if p_filename is null or length(trim(p_filename)) = 0 then
    return jsonb_build_object('inherit', true, 'exceptions', '{}'::jsonb);
  end if;

  /*
  -- Resolve the name to the CALLER's best claim on it — owner row first, then
  -- any cart they hold a grant on. Inherits 005's known ambiguity verbatim:
  -- `user_carts` is unique on (user_id, cart_filename), NOT on cart_filename, so
  -- two firms may each own a `payroll.pkl`. Keyed by name today because mounts
  -- are; the day we host two tenants with the same cart name, this and 005 move
  -- to cart ids together.
  */
  select uc.id, uc.inherit_new_documents, uc.inherit_changed_at
    into v_cart, v_inherit, v_changed
    from public.user_carts uc
   where uc.cart_filename = p_filename
     and (uc.user_id = v_seat
          or exists (select 1 from public.cart_grants g
                      where g.cart_id = uc.id and g.grantee_id = v_seat))
   order by (uc.user_id = v_seat) desc
   limit 1;

  if v_cart is null then
    /*
    -- Either unregistered, or registered and this seat has no claim. Both mean
    -- "no document-level policy applies to you here" — and in the second case
    -- 004 has already refused the mount, so nothing downstream can act on this.
    */
    return jsonb_build_object('inherit', true, 'exceptions', '{}'::jsonb);
  end if;

  if v_seat is null then
    return jsonb_build_object('inherit', coalesce(v_inherit, true),
                              'exceptions', '{}'::jsonb);
  end if;

  select coalesce(jsonb_object_agg(e.source_hash::text, e.access_level), '{}'::jsonb)
    into v_exc
    from public.cart_object_exceptions e
   where e.cart_id = v_cart
     and e.grantee_id = v_seat;

  return jsonb_build_object(
    'inherit',            coalesce(v_inherit, true),
    'inherit_changed_at', v_changed,
    'exceptions',         coalesce(v_exc, '{}'::jsonb));
end;
$$;

/*
-- `anon` included so an anonymous caller gets an honest empty answer rather than
-- a permission error the API would have to interpret. Same reasoning as 005.
*/
grant execute on function public.object_access_for(text) to authenticated, anon;


/*
-- ---------------------------------------------------------------------------
-- verification
-- ---------------------------------------------------------------------------
-- Expect {"inherit": true, "exceptions": {}} for a name nobody has registered:
--
--   select public.object_access_for('definitely-not-a-real-cart.pkl');
--
-- Expect the toggle to exist and read true for every existing cart:
--
--   select cart_filename, inherit_new_documents from public.user_carts limit 10;
--
-- After inserting a deny for Betty on one finance document, as its OWNER:
--
--   select public.object_access_for('redwood-finance.cart.npz');
*/
