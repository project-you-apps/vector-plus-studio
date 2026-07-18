// Agent definitions — the 4 v1 agents for VPS.
//
// Backend equivalent lives at api/agents/{auto_briefing,qa,professor,cart_curator}.py
// with matching input_schema. Keep the two shapes in sync — the smoke test on
// the backend asserts each registered agent's schema is well-formed; the
// frontend uses these slugs verbatim to POST /api/agents/run.
//
// Field types are the SAME enum as the Reports engine
// (frontend/src/reports/report-definitions.ts) so ReportInputPane's
// schema-driven form renderer can be reused verbatim by AgentInputPane.

import type { FieldSchema } from '../reports/report-definitions'

export type { FieldSchema } from '../reports/report-definitions'

export interface AgentDefinition {
  name: string            // slug, e.g. "qa" — matches backend Agent.name
  displayName: string     // "Q&A"
  description: string     // one-line description
  icon: string            // lucide-react icon name (see AgentCard iconMap)
  llmDependency: boolean  // true for all v1 agents
  inputSchema: FieldSchema[]
}

// Ordered to match the "Four v1 agents" table so the card
// grid reads left-to-right in the same order the doc lists them.
export const AGENT_DEFINITIONS: AgentDefinition[] = [
  {
    name: 'auto_briefing',
    displayName: 'Auto-Briefing',
    description:
      "Daily briefing on this cart — recent additions, key themes, notable material.",
    icon: 'Newspaper',
    llmDependency: true,
    inputSchema: [
      {
        name: 'focus',
        label: 'Focus (optional)',
        type: 'text',
        required: false,
        placeholder: 'e.g. compliance, new hires, product launches',
        helpText: 'Optional — narrow the briefing to a specific topic.',
      },
      {
        name: 'tone',
        label: 'Tone',
        type: 'select',
        required: true,
        default: 'executive',
        options: ['executive', 'casual', 'technical'],
        helpText: 'Voice register for the briefing narrative.',
      },
    ],
  },
  {
    name: 'qa',
    displayName: 'Q&A',
    description:
      "Ask a natural-language question — retrieves the most relevant passages and synthesizes a cited answer.",
    icon: 'MessageCircleQuestion',
    llmDependency: true,
    inputSchema: [
      {
        name: 'question',
        label: 'Your question',
        type: 'textarea',
        required: true,
        placeholder: 'e.g. What themes appear across the poems in this cart?',
        helpText: "Ask anything the cart's material could answer.",
      },
      {
        name: 'answer_style',
        label: 'Answer style',
        type: 'select',
        required: true,
        default: 'concise',
        options: ['concise', 'detailed', 'bulleted'],
        helpText: 'Shape of the response.',
      },
    ],
  },
  {
    name: 'professor',
    displayName: 'Professor',
    description:
      "Generate a study quiz from the cart. Great for onboarding new hires.",
    icon: 'GraduationCap',
    llmDependency: true,
    inputSchema: [
      {
        name: 'num_questions',
        label: 'Number of questions',
        type: 'number',
        required: true,
        default: 5,
        helpText: 'How many quiz questions to generate (1-20).',
      },
      {
        name: 'difficulty',
        label: 'Difficulty',
        type: 'select',
        required: true,
        default: 'medium',
        options: ['easy', 'medium', 'hard'],
        helpText: 'Shapes both retrieval width and question style.',
      },
      {
        name: 'topic',
        label: 'Topic filter (optional)',
        type: 'text',
        required: false,
        placeholder: 'e.g. Sysco Portland, fuel surcharges, poetry themes',
        helpText: 'Optional — narrow the quiz to a specific topic.',
      },
    ],
  },
  {
    name: 'cart_curator',
    displayName: 'Cart Curator',
    description:
      "Recommendations for improving this cart — thin coverage, under-represented sources, orphan material.",
    icon: 'ClipboardList',
    llmDependency: true,
    inputSchema: [
      {
        name: 'focus_area',
        label: 'Focus area (optional)',
        type: 'text',
        required: false,
        placeholder: 'e.g. financial docs, vendor invoices, meeting notes',
        helpText: "Optional — tell the curator what you're building this cart for.",
      },
      {
        name: 'source_min',
        label: 'Under-represented source threshold',
        type: 'number',
        required: false,
        default: 5,
        helpText: 'Sources with fewer than this many patterns are flagged.',
      },
    ],
  },
  // Free Agent — LAST in the grid on purpose. It's the catch-all fallback
  // when none of the four specialized recipes above fit the task. Positioning
  // signals "start with a specialized recipe; drop here if nothing fits."
  // Name is a baseball pun (2026-07-14, All-Star week) — may rename if it
  // doesn't stick. Icon uses 'Bot' which is already in AgentCard's ICON_MAP
  // + thematically appropriate as the "generic agent" glyph.
  {
    name: 'free_agent',
    displayName: 'Free Agent',
    description:
      "When none of the specialized recipes fit — describe your task or question, and Free Agent will do its best with your cart as context.",
    icon: 'Bot',
    llmDependency: true,
    inputSchema: [
      {
        name: 'user_input',
        label: 'What would you like the agent to do?',
        type: 'textarea',
        required: true,
        placeholder:
          'Summarize the last month of additions... rewrite this passage for a general audience... compare the two most-cited papers...',
        helpText: "Any task or question the cart's material could help with.",
      },
    ],
  },
]
