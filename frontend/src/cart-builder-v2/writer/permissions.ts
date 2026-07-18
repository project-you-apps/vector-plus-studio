// Step 2a permissions sidecar generator.
// Schema mirrors api/cartridge_io.py:save_cart_permissions.
//
// Browser-built default: { default: "rw", version: "1.0" }
// Read+write by default — the user just built this cart in their own
// browser, so they own it and should be able to Edit Carts on it
// without an extra unlock step. Matches the pattern-level perms_byte
// default (PERM_R | PERM_W) in hippocampus.ts..
//
// To narrow permissions before sharing/uploading, the user can edit
// the sidecar manually or use bin/set_cart_permissions.py to set
// default="r" (or any other allowed string).

export type DefaultPermsString = 'r' | 'rw' | 'rwx'

export interface CartPermissionsSpec {
  default?: DefaultPermsString
  owner?: string
  description?: string
  version?: string
}

export interface CartPermissionsPayload {
  default: DefaultPermsString
  version: string
  owner?: string
  description?: string
}

export const PERMISSIONS_SCHEMA_VERSION = '1.0'

export function buildPermissions(
  spec: CartPermissionsSpec = {}
): CartPermissionsPayload {
  const payload: CartPermissionsPayload = {
    default: spec.default ?? 'rw',
    version: spec.version ?? PERMISSIONS_SCHEMA_VERSION,
  }
  if (spec.owner) payload.owner = spec.owner
  if (spec.description) payload.description = spec.description
  return payload
}
