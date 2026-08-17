-- Register the reddwood-* demo carts by MIRRORING their redwood-* twins.
--
-- WHY A MIRROR AND NOT A UUID LIST. `cartridges/grants.sql` and `office.json` carry
-- DETERMINISTIC placeholder seats (`bound_to_auth: false`) that were never the live identities
-- -- Betty's real seat appears in the droplet log as 2301149e-..., not the 39525aff-... in that
-- file. So the on-disk grant script is stale, and hand-copying UUIDs out of it would grant the
-- carts to three accounts that do not exist.
--
-- This asks the database who actually owns and may read the redwood carts, and gives the
-- reddwood twins exactly the same answer. It is correct without anyone knowing a UUID, and it
-- stays correct if the real seats are re-bound later.
--
-- SAFE TO RUN TWICE. Both statements are ON CONFLICT DO NOTHING and neither touches a
-- redwood-* row. Nothing here revokes, updates or deletes anything.
--
-- ⚠ ROWS FIRST, THEN THE FILES. This is the opposite of what it first seemed.
--
-- An UNREGISTERED cart is visible to everyone, including signed-out callers -- that is the
-- legacy fall-through in `cart_access.decide` ("legacy carts readable by anyone"). So dropping
-- the reddwood files into the cartridge directory before these rows exist would leave them
-- publicly listed AND mountable, carrying the same content their redwood twins restrict.
--
-- Registering first fails the safe way instead: the carts are owned and granted, so anonymous
-- cannot see them at all, and the only symptom is that a granted user clicking one gets a
-- refusal until the files land.
--
-- The files are therefore parked OUTSIDE the cartridge dir at
-- /opt/vector-plus-studio/_staging-reddwood (2026-08-17), where nginx and `list_cartridges`
-- cannot reach them. After running this, move them in -- no restart needed, the cart list
-- rescans the directory on every request:
--
--   mv /opt/vector-plus-studio/_staging-reddwood/* /opt/vector-plus-studio/cartridges/
--   rmdir /opt/vector-plus-studio/_staging-reddwood
--
-- Context: 2026-08-17. reddwood-* are the same 300 source files per cart as redwood-*, rebuilt
-- with the line-aware chunker so passages keep their markdown. Measured retrieval cost of that
-- change: none distinguishable from noise (docs/RESEARCH-chunking-strategies-2026-08-16.md).

begin;

-- 1. Ownership. `user_carts` IS ownership (db/004: ownership is a row, never a grant), so this
--    single insert makes each reddwood cart belong to whoever owns its redwood twin.
--    pattern_count is deliberately NOT copied -- the line-aware chunker produces a different
--    number of passages (company 1712 -> 1593) and a wrong count here is a small lie the UI
--    would repeat. Left null; nothing reads it for access.
insert into public.user_carts (user_id, cart_filename, display_name, size_bytes, pattern_count)
select uc.user_id,
       replace(uc.cart_filename, 'redwood-', 'reddwood-'),
       replace(uc.display_name,  'redwood-', 'reddwood-'),
       uc.size_bytes,
       null
  from public.user_carts uc
 where uc.cart_filename like 'redwood-%'
on conflict (user_id, cart_filename) do nothing;

-- 2. Grants. Every (grantee, level) on a redwood cart is reproduced on its reddwood twin, so
--    Betty still has no finance and the denial the demo turns on survives the copy.
--    Joined on user_id as well as filename so a cart owned by two different accounts under the
--    same name cannot cross-wire.
insert into public.cart_grants (cart_id, grantee_id, access_level, granted_by)
select nc.id, g.grantee_id, g.access_level, g.granted_by
  from public.cart_grants g
  join public.user_carts oc on oc.id = g.cart_id
  join public.user_carts nc on nc.cart_filename = replace(oc.cart_filename, 'redwood-', 'reddwood-')
                           and nc.user_id       = oc.user_id
 where oc.cart_filename like 'redwood-%'
on conflict (cart_id, grantee_id) do nothing;

commit;

-- VERIFY -- expect one reddwood row per redwood row, and matching grant counts per pair.
--
-- select cart_filename, count(*) over () as carts
--   from public.user_carts where cart_filename like 'reddwood-%' order by 1;
--
-- select oc.cart_filename as original, count(og.*) as grants_before,
--        nc.cart_filename as mirrored, count(ng.*) as grants_after
--   from public.user_carts oc
--   left join public.cart_grants og on og.cart_id = oc.id
--   left join public.user_carts nc on nc.cart_filename = replace(oc.cart_filename,'redwood-','reddwood-')
--   left join public.cart_grants ng on ng.cart_id = nc.id
--  where oc.cart_filename like 'redwood-%'
--  group by 1, 3 order by 1;
