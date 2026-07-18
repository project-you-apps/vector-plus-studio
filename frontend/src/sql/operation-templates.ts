// SQL Editor operation templates (v1 shell).
//
// Ten canonical operations — SQL keywords map to lattice/substrate operations,
// not literal SQL execution — e.g. `similarity(text)` is a semantic reranker
// not a scalar function; DELETE tombstones rather than removes rows so the
// substrate can replay history.
//
// v1 is UI shell only; clicking an operation button appends the snippet to
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

// Every snippet ends with `\n` so successive pill clicks (or user typing) land
// on a fresh line instead of jamming against the previous snippet's tail. If
// a snippet naturally ends mid-clause (SELECT ... WHERE ), the trailing
// newline lets the user compose the next clause below it and click another
// pill without manual line breaks.
export const OPERATION_TEMPLATES: Record<SQLOperation, string> = {
  'SELECT':   `SELECT * FROM cart WHERE \n`,
  'WHERE':    `WHERE source LIKE '%.pdf' AND relevance > 0.7\n`,
  'JOIN':     `LEFT JOIN cart 'other-cart-id' ON similarity(text) > 0.8\n`,
  'ORDER BY': `ORDER BY relevance DESC, date ASC\n`,
  'GROUP BY': `GROUP BY source, MONTH(date)\n`,
  'LIKE':     `LIKE '%coconut milk%'\n`,
  'INSERT':   `INSERT INTO cart (text, source, tags)\n  VALUES ('new passage', 'manual-entry', 'important');\n`,
  'UPDATE':   `UPDATE cart SET tags = 'reviewed' WHERE pattern_id = 142;\n`,
  'DELETE':   `DELETE FROM cart WHERE pattern_id = 142;  -- tombstones\n`,
  'LIMIT':    `LIMIT 10 OFFSET 0\n`,
}
