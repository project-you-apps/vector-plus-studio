-- =============================================================================
-- 004 — cart_grants: who else can see a cart, and at what access level
-- =============================================================================
-- Run in Supabase SQL editor (one-shot paste).
-- Idempotent: `create table if not exists`, `drop policy if exists` before each
-- `create policy`. Re-running is safe.
--
-- WHY A NEW TABLE INSTEAD OF WIDENING user_carts
-- ----------------------------------------------
-- `public.user_carts` means "carts I own": unique (user_id, cart_filename), RLS
-- `auth.uid() = user_id` on all four verbs, and its columns (size_bytes,
-- pattern_count) are properties of the CART, not of a grant. Adding grantee rows
-- there would duplicate cart metadata per grantee and overload one table with two
-- meanings — the same collision that forced `role` → `access_level` on 2026-08-01.
--
-- So: user_carts stays the OWNER's registry, untouched. cart_grants records who
-- else may reach it. Existing rows and policies are unaffected.
--
-- VOCABULARY IS SHARED WITH THE CODE, NOT RE-INVENTED HERE
-- --------------------------------------------------------
-- viewer / commenter / editor exactly matches ACCESS_LEVELS in api/access.py and
-- the `.permissions.json` sidecar key announced to the fleet in forum post
-- 67144643. Three spellings of one concept is how you get a cart that grants
-- write in one layer and denies it in another.
--
-- OWNER IS NOT AN ACCESS LEVEL, AND THE CHECK CONSTRAINT ENFORCES IT
-- ------------------------------------------------------------------
-- Ownership is proven by owning the row in user_carts (and, on disk, by holding
-- the key that signs the cart) — never by a string someone can write. If 'owner'
-- were grantable here, a compromised grant row would be an ownership transfer.
-- `share` and `sign` are unreachable from any value in this table by construction.
-- =============================================================================


create table if not exists public.cart_grants (
  id            uuid primary key default gen_random_uuid(),
  cart_id       uuid references public.user_carts(id) on delete cascade not null,
  grantee_id    uuid references auth.users(id)        on delete cascade not null,
  access_level  text not null check (access_level in ('viewer','commenter','editor')),
  granted_by    uuid references auth.users(id)        on delete set null,
  created_at    timestamptz default now(),
  updated_at    timestamptz default now(),
  unique (cart_id, grantee_id)
);

create index if not exists cart_grants_grantee_idx on public.cart_grants (grantee_id);
create index if not exists cart_grants_cart_idx    on public.cart_grants (cart_id);


-- ---------------------------------------------------------------------------
-- helper: does the current user own this cart?
-- ---------------------------------------------------------------------------
-- SECURITY DEFINER so the policies below can consult user_carts without being
-- blocked by user_carts' own RLS. Kept to a single boolean so it cannot be used
-- to read anything else.
create or replace function public.owns_cart(p_cart_id uuid)
returns boolean
language sql
security definer
set search_path = public
as $$
  select exists (
    select 1 from public.user_carts uc
    where uc.id = p_cart_id and uc.user_id = auth.uid()
  );
$$;


alter table public.cart_grants enable row level security;

-- SELECT: you can see a grant if it is YOURS, or if you own the cart it is on.
drop policy if exists "grantee sees own grants"      on public.cart_grants;
drop policy if exists "owner sees grants on carts"   on public.cart_grants;
create policy "grantee sees own grants"    on public.cart_grants
  for select using (auth.uid() = grantee_id);
create policy "owner sees grants on carts" on public.cart_grants
  for select using (public.owns_cart(cart_id));

-- INSERT / UPDATE / DELETE: OWNER ONLY. A grantee must never be able to widen
-- their own access, which is the whole point of separating grants from levels.
drop policy if exists "owner grants access"  on public.cart_grants;
drop policy if exists "owner updates access" on public.cart_grants;
drop policy if exists "owner revokes access" on public.cart_grants;
create policy "owner grants access"  on public.cart_grants
  for insert with check (public.owns_cart(cart_id) and grantee_id <> auth.uid());
create policy "owner updates access" on public.cart_grants
  for update using (public.owns_cart(cart_id));
create policy "owner revokes access" on public.cart_grants
  for delete using (public.owns_cart(cart_id));


-- ---------------------------------------------------------------------------
-- user_carts -- ADDITIVE select policy so a grantee can see the cart row itself
-- ---------------------------------------------------------------------------
-- Without this, a grant points at a row the grantee cannot read, and every
-- shared-cart list comes back empty with no error to explain it. The existing
-- owner policies are left exactly as they are; this only ADDS a second way to
-- pass SELECT. Insert/update/delete remain owner-only and untouched.
drop policy if exists "grantees see shared carts" on public.user_carts;
create policy "grantees see shared carts" on public.user_carts
  for select using (
    exists (
      select 1 from public.cart_grants g
      where g.cart_id = user_carts.id and g.grantee_id = auth.uid()
    )
  );


-- ---------------------------------------------------------------------------
-- effective_access -- one place that answers "what may this user do with cart X"
-- ---------------------------------------------------------------------------
-- Returns 'owner' for the owner, the granted level for a grantee, and NULL for
-- everyone else. NULL is the fail-closed answer: absence of a grant is a denial,
-- not a default. Two fail-OPEN paths were found in the sidecar reader on
-- 2026-08-01 for exactly this reason — an empty permissions file returned
-- writable — so the database-side answer is explicit about it.
create or replace function public.effective_access(p_cart_id uuid)
returns text
language sql
stable
security definer
set search_path = public
as $$
  select case
    when exists (select 1 from public.user_carts uc
                 where uc.id = p_cart_id and uc.user_id = auth.uid())
      then 'owner'
    else (select g.access_level from public.cart_grants g
          where g.cart_id = p_cart_id and g.grantee_id = auth.uid())
  end;
$$;


-- ---------------------------------------------------------------------------
-- updated_at maintenance
-- ---------------------------------------------------------------------------
create or replace function public.touch_cart_grants_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists cart_grants_touch_updated_at on public.cart_grants;
create trigger cart_grants_touch_updated_at
  before update on public.cart_grants
  for each row execute function public.touch_cart_grants_updated_at();
