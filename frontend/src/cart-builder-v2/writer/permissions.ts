// Step 2a permissions sidecar generator.
// Schema mirrors api/cartridge_io.py:save_cart_permissions.
//
// Browser-built default: { default: "r", version: "1.0" }
// Read-only by default — safe for sharing/uploading. User can edit the
// sidecar to widen permissions before deploy.

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
    default: spec.default ?? 'r',
    version: spec.version ?? PERMISSIONS_SCHEMA_VERSION,
  }
  if (spec.owner) payload.owner = spec.owner
  if (spec.description) payload.description = spec.description
  return payload
}
