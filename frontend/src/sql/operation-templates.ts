// SQL Editor operation templates (VPS wave-1 shell).
//
// Ten canonical operations mirror the CC_sql-over-lattice mapping — SQL keywords
// map to lattice/substrate operations, not literal SQL execution. See
// docs/vps-internal/SQL Editor Design 2026-07-10.md §3 for the design rationale
// (why `similarity(text)` isn't a SQL function, why DELETE tombstones, etc.).
//
// Wave-1 is UI shell only; clicking an operation button appends the snippet to
// the editor. The interpreter that translates these into real lattice ops is
// v1.5 work.

export type SQLOperation =
  | 'SELECT'
  | 'WHERE'
  | 'JOIN'
  | 'ORDER BY'
  | 'GROUP BY'
  | 'LIKE'
  | 'INSERT'
  | 'UPDATE'
  | 'DELETE'
  | 'LIMIT'

// Ordered so the toolbar renders read/filter ops first (SELECT..LIMIT top row
// in the design mock), CRUD ops second. Keep this array as the source of truth
// for both the button list and the templates map.
export const SQL_OPERATIONS: SQLOperation[] = [
  'SELECT', 'WHERE', 'JOIN', 'ORDER BY', 'GROUP BY',
  'LIKE', 'INSERT', 'UPDATE', 'DELETE', 'LIMIT',
]

export const OPERATION_TEMPLATES: Record<SQLOperation, string> = {
  'SELECT':   `SELECT * FROM cart WHERE `,
  'WHERE':    `WHERE source LIKE '%.pdf' AND relevance > 0.7\n`,
  'JOIN':     `LEFT JOIN cart 'other-cart-id' ON similarity(text) > 0.8\n`,
  'ORDER BY': `ORDER BY relevance DESC, date ASC\n`,
  'GROUP BY': `GROUP BY source, MONTH(date)\n`,
  'LIKE':     `LIKE '%coconut milk%'`,
  'INSERT':   `INSERT INTO cart (text, source, tags)\n  VALUES ('new passage', 'manual-entry', 'important');\n`,
  'UPDATE':   `UPDATE cart SET tags = 'reviewed' WHERE pattern_id = 142;\n`,
  'DELETE':   `DELETE FROM cart WHERE pattern_id = 142;  -- tombstones\n`,
  'LIMIT':    `LIMIT 10 OFFSET 0`,
}
