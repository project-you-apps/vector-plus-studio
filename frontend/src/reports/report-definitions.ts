// Report definitions — the 9 generic report types for VPS, sourced from
// docs/vps-internal/Report Types Design 2026-07-10.md. Coverage Report
// (Wave-1c, 2026-07-12) rounds the set out to 9 to match the pitch copy.
//
// These are the FRONTEND shells only — the backend Report interface + registry
// (see design doc §0.1) is future work. Each definition here mirrors the
// design doc's per-report input schema so the auto-generated form matches
// what the backend module will eventually expect.
//
// LLM dependency is currently true only for Executive TL;DR; Wave-2 reports
// (Timeline, Trend, Financial) will pick up optional LLM fallback later but
// stay false for this shell so the sparkles badge stays honest.

export type FieldType =
  | 'text'
  | 'number'
  | 'select'
  | 'date-range'
  | 'regex'
  | 'textarea'

export interface FieldSchema {
  name: string
  label: string
  type: FieldType
  required: boolean
  default?: string | number
  options?: string[]      // for type: 'select'
  placeholder?: string
  helpText?: string
}

export interface ReportDefinition {
  name: string            // slug, e.g. "summary"
  displayName: string     // "Summary"
  description: string     // one-line description
  icon: string            // lucide-react icon name (see ReportCard iconMap)
  llmDependency: boolean
  inputSchema: FieldSchema[]
}

// Ordered per Report Types Design §1-8 so the card grid mirrors the doc.
// Wave-1 (no LLM) first, TL;DR (LLM-dep) last.
export const REPORT_DEFINITIONS: ReportDefinition[] = [
  {
    name: 'summary',
    displayName: 'Summary',
    description: 'Cart orientation — what\'s in this cart, top themes, sources.',
    icon: 'FileText',
    llmDependency: false,
    inputSchema: [
      {
        name: 'top_themes',
        label: 'Top themes',
        type: 'number',
        required: false,
        default: 5,
        helpText: 'How many top themes to surface (default 5).',
      },
      {
        name: 'date_range',
        label: 'Date range',
        type: 'date-range',
        required: false,
        helpText: 'Optional — restrict summary to a date window.',
      },
    ],
  },
  {
    name: 'timeline',
    displayName: 'Timeline',
    description: 'Chronological view of events + mentions clustered by period.',
    icon: 'Calendar',
    llmDependency: false,
    inputSchema: [
      {
        name: 'entity_filter',
        label: 'Entity filter',
        type: 'text',
        required: false,
        placeholder: 'e.g. Sysco Portland',
        helpText: 'Optional — restrict to mentions of one entity.',
      },
      {
        name: 'granularity',
        label: 'Granularity',
        type: 'select',
        required: true,
        default: 'week',
        options: ['day', 'week', 'month'],
        helpText: 'Cluster events at this level.',
      },
    ],
  },
  {
    name: 'trend',
    displayName: 'Trend',
    description: 'Numeric metric evolution over time — fuel surcharge, invoice totals, etc.',
    icon: 'TrendingUp',
    llmDependency: false,
    inputSchema: [
      {
        name: 'metric_pattern',
        label: 'Metric',
        type: 'select',
        required: true,
        default: 'fuel_surcharge',
        options: ['fuel_surcharge', 'invoice_total', 'state_fee', 'custom'],
        helpText: 'Named preset or "custom" to supply your own regex.',
      },
      {
        name: 'custom_regex',
        label: 'Custom regex (if metric = custom)',
        type: 'regex',
        required: false,
        placeholder: 'e.g. \\$\\s*(\\d+\\.\\d{2})',
        helpText: 'Only used when Metric is set to "custom".',
      },
      {
        name: 'grouping',
        label: 'Grouping',
        type: 'select',
        required: true,
        default: 'month',
        options: ['week', 'month', 'invoice'],
        helpText: 'Aggregate the series at this level.',
      },
    ],
  },
  {
    name: 'comparison',
    displayName: 'Comparison',
    description: 'Two subsets of a cart side-by-side. "Q1 vs Q2", "vendor A vs B".',
    icon: 'Scale',
    llmDependency: false,
    inputSchema: [
      {
        name: 'subset_a_name',
        label: 'Subset A name',
        type: 'text',
        required: true,
        placeholder: 'e.g. May invoices',
      },
      {
        name: 'subset_a_query',
        label: 'Subset A query',
        type: 'text',
        required: true,
        placeholder: 'Membot query string that defines subset A',
      },
      {
        name: 'subset_b_name',
        label: 'Subset B name',
        type: 'text',
        required: true,
        placeholder: 'e.g. June invoices',
      },
      {
        name: 'subset_b_query',
        label: 'Subset B query',
        type: 'text',
        required: true,
        placeholder: 'Membot query string that defines subset B',
      },
    ],
  },
  {
    name: 'entity_rollup',
    displayName: 'Entity Rollup',
    description: 'All mentions of X across the cart, chronologically.',
    icon: 'Tag',
    llmDependency: false,
    inputSchema: [
      {
        name: 'entity_name',
        label: 'Entity name',
        type: 'text',
        required: true,
        placeholder: 'e.g. Sysco Portland',
      },
      {
        name: 'aliases',
        label: 'Aliases',
        type: 'text',
        required: false,
        placeholder: 'comma-separated alternate spellings',
        helpText: 'Optional — e.g. "SP, Sysco PDX, Sysco-Portland".',
      },
    ],
  },
  {
    name: 'financial_rollup',
    displayName: 'Financial Rollup',
    description: 'Numeric extraction + summation for money-shaped carts.',
    icon: 'DollarSign',
    llmDependency: false,
    inputSchema: [
      {
        name: 'extraction_preset',
        label: 'Extraction preset',
        type: 'select',
        required: true,
        default: 'invoice_total',
        options: ['invoice_total', 'line_item_total', 'expense_category'],
        helpText: 'Preset currency-extraction pattern to apply.',
      },
      {
        name: 'grouping_dim',
        label: 'Grouping dimension',
        type: 'select',
        required: true,
        default: 'period',
        options: ['vendor', 'category', 'period', 'invoice'],
        helpText: 'Aggregate totals by this dimension.',
      },
      {
        name: 'date_range',
        label: 'Date range',
        type: 'date-range',
        required: false,
        helpText: 'Optional — restrict rollup to a date window.',
      },
    ],
  },
  {
    name: 'change_log',
    displayName: 'Change Log',
    description: 'What changed between two cart snapshots — added / removed / modified.',
    icon: 'GitCompareArrows',
    llmDependency: false,
    inputSchema: [
      {
        name: 'cart_id_old',
        label: 'Old cart',
        type: 'text',
        required: true,
        placeholder: 'cart name or path (older snapshot)',
        helpText: 'The "before" cart to diff from.',
      },
      {
        name: 'cart_id_new',
        label: 'New cart',
        type: 'text',
        required: true,
        placeholder: 'cart name or path (newer snapshot)',
        helpText: 'The "after" cart to diff to.',
      },
      {
        name: 'diff_strategy',
        label: 'Diff strategy',
        type: 'select',
        required: true,
        default: 'semantic',
        options: ['exact', 'semantic'],
        helpText: 'exact = string-match passages; semantic = embedding cosine >= 0.92.',
      },
    ],
  },
  {
    name: 'coverage',
    displayName: 'Coverage Report',
    description: 'Diagnose gaps in the cart — underrepresented themes, orphan entities, source imbalance, and time-range holes.',
    icon: 'ShieldAlert',
    llmDependency: false,
    inputSchema: [
      {
        name: 'min_theme_items',
        label: 'Min items per theme',
        type: 'number',
        required: false,
        default: 3,
        helpText: 'Themes represented by fewer than this many items are flagged as underrepresented.',
      },
      {
        name: 'gap_threshold_days',
        label: 'Gap threshold (days)',
        type: 'number',
        required: false,
        default: 30,
        helpText: 'Time gaps longer than this between consecutive dates are surfaced.',
      },
      {
        name: 'max_orphan_entities',
        label: 'Max orphan entities',
        type: 'number',
        required: false,
        default: 20,
        helpText: 'Cap on how many orphan entities to display.',
      },
      {
        name: 'source_coverage_min',
        label: 'Source coverage min',
        type: 'number',
        required: false,
        default: 5,
        helpText: 'Sources contributing fewer than this many items are flagged as under-utilized.',
      },
    ],
  },
  {
    name: 'tldr',
    displayName: 'Executive TL;DR',
    description: 'LLM-synthesized 5-bullet summary. The "just tell me what I\'m looking at" report.',
    icon: 'Sparkles',
    llmDependency: true,
    inputSchema: [
      {
        name: 'focus',
        label: 'Focus',
        type: 'text',
        required: false,
        placeholder: 'e.g. fuel surcharges, vendor breakdown',
        helpText: 'Optional — narrow the summary to a specific angle.',
      },
      {
        name: 'length',
        label: 'Length',
        type: 'select',
        required: true,
        default: '5-bullet',
        options: ['5-bullet', '1-paragraph', '1-page'],
        helpText: 'Shape of the synthesis.',
      },
    ],
  },
]
