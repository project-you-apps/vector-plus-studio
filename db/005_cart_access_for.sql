-- =============================================================================
-- 005 — cart_access_for(): the caller's effective access to a cart BY FILENAME
-- =============================================================================
-- Run in Supabase SQL editor (one-shot paste).
-- Idempotent: `create or replace`. Re-running is safe. Creates no tables, alters
-- no rows, grants no new access — it only answers a question that could not be
-- asked correctly from the client side.
--
-- WHY THIS EXISTS: RLS MAKES "UNREGISTERED" AND "DENIED" LOOK IDENTICAL
-- ---------------------------------------------------------------------
-- The API's Supabase client carries the CALLER's token so RLS applies to them
-- (api/profile_routes.py:69 — a service-role key there would bypass every policy
-- in 004). Correct for reads. Fatal for this check.
--
-- `user_carts` SELECT is `auth.uid() = user_id` plus 004's additive grantee
-- policy. So a seat with no grant selecting a cart it may not reach gets ZERO
-- rows — exactly what a cart that was never registered returns. The mount gate
-- treats unregistered as a legacy cart and ALLOWS it (Andy's rule, 2026-08-03),
-- so without this function the gate would allow everyone, and every test run
-- against a live database would pass while doing so.
--
-- SECURITY DEFINER, returning ONE STRING
-- ---------------------------------------
-- Same shape and same reasoning as `owns_cart()` in 004: it sees past RLS, and
-- it is deliberately narrow enough that it cannot be turned into a reader of
-- anything else. It reveals whether a NAME is registered and what the CALLER's
-- own level is. It never reveals who the owner is, what the cart contains, or
-- who else holds a grant.
--
-- ⚠ KNOWN AMBIGUITY, SURFACED NOT SILENTLY RESOLVED
-- --------------------------------------------------
-- `user_carts` is unique on (user_id, cart_filename), NOT on cart_filename. Two
-- users may each own a different cart named `company-brain.pkl`. This function
-- resolves a name to the CALLER's best claim on it: owner first, then any grant.
-- That is right for a local studio where the filesystem is the caller's own, and
-- it is NOT right for a hosted multi-tenant deployment, where mounts should be
-- keyed by cart id rather than by name. Recorded here rather than papered over,
-- because the day we host two firms with a `payroll.pkl` each is the day it
-- matters, and it will not announce itself.
-- =============================================================================

create or replace function public.cart_access_for(p_filename text)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
  v_seat  uuid := auth.uid();
  v_level text;
begin
  if p_filename is null or length(trim(p_filename)) = 0 then
    return null;
  end if;

  -- Registered at all? Any owner's row counts: the question here is whether the
  -- NAME is under management, which is what separates a legacy cart from a
  -- governed one.
  if not exists (select 1 from public.user_carts uc
                  where uc.cart_filename = p_filename) then
    return 'unregistered';
  end if;

  if v_seat is null then
    return null;                     -- registered cart, anonymous caller: denied
  end if;

  if exists (select 1 from public.user_carts uc
              where uc.cart_filename = p_filename and uc.user_id = v_seat) then
    return 'owner';
  end if;

  select g.access_level into v_level
    from public.cart_grants g
    join public.user_carts uc on uc.id = g.cart_id
   where uc.cart_filename = p_filename
     and g.grantee_id = v_seat
   order by case g.access_level
              when 'editor'    then 3
              when 'commenter' then 2
              when 'viewer'    then 1
              else 0
            end desc
   limit 1;

  return v_level;                    -- null when registered and ungranted: denied
end;
$$;

-- `authenticated` covers signed-in seats; `anon` is included so an anonymous
-- caller receives an honest "denied" from the function rather than a permission
-- error the API would have to guess the meaning of.
grant execute on function public.cart_access_for(text) to authenticated, anon;


-- ---------------------------------------------------------------------------
-- verification (expect: 'unregistered' for a name no one has registered)
-- ---------------------------------------------------------------------------
-- select public.cart_access_for('definitely-not-a-real-cart.pkl');
