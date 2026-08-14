# hhDesign.md — VoiceRAG White × Deep Green Product Dashboard

**Product:** VoiceRAG — Low-Latency Voice-Enabled RAG System  
**Reference Direction:** Hacker House Goa + the supplied modern product-page reference  
**Theme:** White + `#1F7335`  
**Design Type:** Editorial Product Website + Interactive AI Dashboard  
**Primary Platform:** Desktop-first Web, responsive to mobile

---

## 1. Design Vision

VoiceRAG should look like a **real AI product**, not a generic admin dashboard.

Combine:

```text
Modern Product Website
        +
RAG / AI Observability
        +
Live Interactive Demo
```

The visual language should use:

- strong typography
- large whitespace
- concise section headings
- big product-focused visual blocks
- restrained cards
- clear CTAs
- long-scroll storytelling
- real technical metrics

Avoid:

- neon/cyberpunk AI styling
- purple/blue gradient SaaS themes
- cartoon robots
- excessive rounded cards
- excessive shadows
- "Welcome back" admin-dashboard patterns

Core identity:

> **VOICE IN. KNOWLEDGE OUT.**

Secondary identity:

> **FAST. GROUNDED. MEASURABLE.**

---

# 2. Color System

The primary visual identity is **white + deep green**.

| Role | Hex |
|---|---|
| Main Background | `#FFFFFF` |
| Soft Background | `#FAFCFA` |
| Light Green | `#EAF4ED` |
| Very Light Green | `#F3F8F4` |
| Primary Green | `#1F7335` |
| Dark Green | `#15552A` |
| Primary Text | `#111111` |
| Secondary Text | `#667085` |
| Muted Text | `#98A2B3` |
| Border | `#E1E7E3` |
| Warning | `#C98A20` |
| Error | `#C93636` |

### Color ratio

```text
White / off-white      75–85%
Light green surfaces    10–15%
Deep green               5–10%
```

`#1F7335` is a **signal color**, not a page-fill color.

Use it for:

- primary CTA
- active navigation
- microphone active state
- pipeline progress
- important metrics
- selected retrieval result
- grounding state
- health status
- progress bars
- links

---

# 3. Typography

## Primary

**Inter**

Use for:

- navigation
- body
- cards
- buttons
- forms
- metrics

## Display

**Inter Tight**

Alternatives:

- Geist
- Space Grotesk

Use for major editorial headings only.

## Technical

**JetBrains Mono**

Use for:

- latency
- request IDs
- HNSW parameters
- vector dimensions
- logs
- trace IDs
- raw technical values

### Scale

```text
Hero                72–104px
Major statement     48–64px
Page heading        32–40px
Section heading     24–32px
Metric              32–56px
Card heading        15–18px
Body                14–16px
Metadata            11–13px
Technical           11–13px
```

Hero:

```text
font-weight: 750–800
line-height: 0.90–0.98
letter-spacing: -0.055em
```

---

# 4. Global Layout

The main dashboard should behave like a product website with operational sections.

```text
HEADER
  ↓
HERO
  ↓
LIVE QUERY
  ↓
RAG PIPELINE
  ↓
PERFORMANCE
  ↓
RETRIEVAL / EVIDENCE
  ↓
ANSWER
  ↓
DATASET / VECTOR INDEX
  ↓
BENCHMARKS
  ↓
FAQ
  ↓
FOOTER
```

Desktop:

```text
Max width: 1440–1600px
Page padding: 32–56px
Section spacing: 80–140px
```

Mobile:

```text
Page padding: 20px
Section spacing: 56–80px
```

---

# 5. Header

Minimal white header.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ ◇ VoiceRAG       Dashboard   Live Query   Retrieval   Benchmarks   System │
│                                                               ● Healthy   │
└──────────────────────────────────────────────────────────────────────────┘
```

Rules:

- white background
- thin bottom border
- no heavy shadow
- green active state
- compact labels

Active navigation:

```text
background: #EAF4ED
color: #1F7335
```

---

# 6. Hero

The hero should immediately explain the product.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  VOICE IN.                              ┌────────────────────────────────┐ │
│  KNOWLEDGE OUT.                         │                                │ │
│                                         │             ◉                  │ │
│  Low-latency multilingual RAG          │                                │ │
│  for fast, grounded answers.           │       HOLD TO SPEAK             │ │
│                                         │                                │ │
│  [ RUN LIVE QUERY → ]                   │       ● READY                   │ │
│  [ VIEW SYSTEM ]                         │                                │ │
│                                         └────────────────────────────────┘ │
│                                                                            │
│  142 MS                                                                   │
│  END-TO-END LATENCY                                                       │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

Use a large light-green visual block:

```text
background: #EAF4ED
```

Inside it show:

- microphone
- waveform
- system status
- optional mini pipeline

Do not use stock photography.

---

# 7. Hero Copy

Primary:

```text
VOICE IN.
KNOWLEDGE OUT.
```

Supporting:

```text
Ask your knowledge base through voice
and get fast, grounded answers backed by retrieved evidence.
```

CTA:

```text
[ RUN LIVE QUERY → ]
```

Secondary:

```text
[ VIEW SYSTEM ]
```

---

# 8. Hero Performance Signal

Treat latency like a product metric.

```text
142
MS

END-TO-END LATENCY

TARGET
< 200 MS
```

Use green for the active/status portion.

Production UI must show actual telemetry. Numbers in this design are wireframe examples only.

---

# 9. Intro Section

Create a short editorial transition:

```text
ONE PIPELINE.
ONE ANSWER.
```

Supporting:

```text
Speech is converted to text,
semantic meaning is retrieved from
the indexed corpus, and the answer
is generated from evidence.
```

Keep it short.

---

# 10. Three-Step Product Story

```text
01
LISTEN

Sarvam converts voice
into text.

02
RETRIEVE

Multilingual embeddings
search Qdrant + HNSW.

03
ANSWER

A fast generation model
produces a grounded response.
```

Use giant numbers with green accents.

No cards required; use whitespace and separators.

---

# 11. Live Query Section

Heading:

```text
ASK THE SYSTEM.
```

Subheading:

```text
Speak naturally. See exactly how the system processes your request.
```

Layout:

```text
┌─────────────────────────────────────────────────────────────────────┐
│ ASK THE SYSTEM.                         LIVE PIPELINE                │
│                                                                     │
│ ┌──────────────────────────────┐       ✓ Speech recognized 48ms    │
│ │                              │       ✓ Query embedded    12ms    │
│ │             ◉                │       ✓ Retrieved           9ms   │
│ │                              │       ✓ Context             3ms   │
│ │       HOLD TO SPEAK          │       ✓ Generated          68ms   │
│ │                              │                                     
│ │          Ready               │       TOTAL              140ms   │
│ └──────────────────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

# 12. Voice Interaction States

## Idle

```text
◉

HOLD TO SPEAK
Ready
```

## Listening

```text
●

LISTENING...

▂▅▇▆▃▂▅▇▃
```

## Processing

```text
◌

PROCESSING QUERY...
```

## Complete

```text
✓

ANSWER READY
```

Active color:

```text
#1F7335
```

Animation:

```text
120–250ms
```

Keep it subtle.

---

# 13. Query Transcription

After STT show:

```text
YOU SAID

"What causes earthquakes?"
```

Actions:

```text
[ Edit Query ]
```

If transcription fails:

```text
We couldn't understand the audio.
Please try again.
```

---

# 14. RAG Pipeline

Heading:

```text
THE PIPELINE.
```

Subheading:

```text
Every request passes through the same measurable path.
```

Desktop:

```text
VOICE
  │
  ▼
SARVAM STT
48 ms
  │
  ▼
MULTILINGUAL EMBEDDING
12 ms
  │
  ▼
QDRANT
HNSW
9 ms
  │
  ▼
TOP-K
5 RESULTS
  │
  ▼
CONTEXT
3 ms
  │
  ▼
FAST LLM
68 ms
  │
  ▼
GROUNDED ANSWER
```

Mobile should become a vertical timeline.

---

# 15. Pipeline Node

Example:

```text
┌─────────────────────────────┐
│ QDRANT                      │
│ HNSW                        │
│                             │
│ 9 ms                        │
│ ● Operational               │
└─────────────────────────────┘
```

Default:

```text
background: #FFFFFF
border: 1px solid #E1E7E3
```

Active:

```text
background: #EAF4ED
border: 1px solid #1F7335
```

Completed:

```text
green check
```

---

# 16. Performance Section

Major statement:

```text
STAY
UNDER
200MS.
```

Supporting:

```text
Measured across the full user-to-response pipeline.
```

Metrics:

```text
P50                 P70                 P100

142 ms              159 ms              188 ms

TARGET
<200 ms
```

Use a strong typographic layout rather than three identical dashboard cards.

---

# 17. Latency Chart

```text
200 ───────────────────────────────────── TARGET

180 ─────────╮
             ╰────╮
160               ╰──────╮
                          ╰─────
140 ───────────────────────────────

     12:00     12:15     12:30     12:45
```

Use:

```text
Latency line: #1F7335
Target line:  #D7E8DB
Grid:         #E9EFEB
```

Filters:

```text
[15m] [1h] [6h] [24h] [7d]
```

---

# 18. Performance Breakdown

```text
STT            48ms
EMBEDDING      12ms
QDRANT          9ms
CONTEXT         3ms
LLM            68ms
OTHER          15ms
─────────────────────
TOTAL         155ms
```

Visualize with one horizontal bar.

The largest section should naturally reveal the current bottleneck.

---

# 19. Retrieval Section

Heading:

```text
WHAT DID WE FIND?
```

Subheading:

```text
Inspect the passages retrieved before generation.
```

Use wide result blocks, not crowded cards.

---

# 20. Retrieval Result

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ #01                                                     SCORE 0.923       │
│                                                                          │
│ Maintaining a consistent sleep schedule can improve sleep quality...    │
│                                                                          │
│ CHUNK 8921     LANGUAGE EN     SOURCE MSMARCO-XI     8.4 ms              │
│                                                                          │
│ [ VIEW SOURCE → ]                                                       │
└──────────────────────────────────────────────────────────────────────────┘
```

Use green for score and active links.

---

# 21. Retrieval Score

Display:

```text
0.923
```

Font:

```text
JetBrains Mono
```

Color:

```text
#1F7335
```

Ranking:

```text
#01
#02
#03
```

---

# 22. Evidence Section

Heading:

```text
THE EVIDENCE.
```

Show Top-K evidence:

```text
01 / 0.923

Retrieved passage...

MSMARCO-XI
CHUNK 8921
```

Use thin separators.

Avoid excessive cards.

---

# 23. Answer Section

Heading:

```text
THE ANSWER.
```

Large answer surface:

```text
Regular exercise, maintaining a
consistent sleep schedule, and
reducing caffeine before bedtime
can improve sleep quality.
```

Below:

```text
✓ GROUNDED
4 SUPPORTING PASSAGES
CONFIDENCE: HIGH

[ VIEW EVIDENCE ]
```

The answer should look like a research result, not a chat bubble.

---

# 24. Evidence → Answer Relationship

Show the flow:

```text
RETRIEVED EVIDENCE
        │
        ├─────────┐
        ├─────────┤
        ├─────────┤
        └─────────┘
            │
            ▼
         ANSWER
```

The interface should make it obvious that evidence comes before generation.

---

# 25. No-Evidence State

```text
NO SIGNAL.

We couldn't retrieve enough relevant
evidence to answer this question.

RETRIEVAL CONFIDENCE

0.31

[ TRY ANOTHER QUERY ]
```

Use:

```text
background: #F7F9F8
border: #E1E7E3
```

Do not use an aggressive red screen for a normal retrieval miss.

---

# 26. Grounding States

```text
✓ GROUNDED
```

```text
! LOW EVIDENCE
```

```text
× UNSUPPORTED
```

Colors:

```text
Grounded      #1F7335
Low evidence  #C98A20
Unsupported   #C93636
```

Never display "AI verified" unless that verification is actually implemented.

---

# 27. Dataset Section

Heading:

```text
THE CORPUS.
```

Large title:

```text
MSMARCO-XI
```

Stats:

```text
12M+
RECORDS

~52 GB
DATA

14+
LANGUAGES
```

Production UI must source exact values from the actual indexing pipeline.

---

# 28. Dataset Visual Block

Use a strong green panel inspired by the supplied reference's large feature blocks.

```text
┌────────────────────────────────────────────┐
│                                            │
│              MSMARCO-XI                   │
│                                            │
│               12M+                         │
│              RECORDS                       │
│                                            │
│                52 GB                       │
│                                            │
└────────────────────────────────────────────┘
```

Background:

```text
#1F7335
```

Text:

```text #FFFFFF
```

Place a white data-stat panel beside it.

---

# 29. Vector Index Section

Heading:

```text
THE INDEX.
```

Primary:

```text
QDRANT
```

Secondary:

```text
HNSW
Dense Vector Search
```

Technical values:

```text
Vectors         12,482,193
Dimension       768
Distance        Cosine
ef_search       32
M               16
Memory          31.4 GB
```

Only show actual runtime values in production.

---

# 30. Vector Index Visual

```text
┌──────────────────────────────────────────┐
│                                          │
│               QDRANT                     │
│                                          │
│                HNSW                      │
│                                          │
│               12M+                       │
│              VECTORS                     │
│                                          │
└──────────────────────────────────────────┘
```

Background:

```text
#1F7335
```

Use white technical text.

---

# 31. Architecture Section

Heading:

```text
UNDER THE HOOD.
```

Architecture:

```text
USER VOICE
    ↓
SARVAM STT
    ↓
TEXT QUERY
    ↓
MULTILINGUAL EMBEDDING
    ↓
QDRANT + HNSW
    ↓
TOP-K
    ↓
CONTEXT
    ↓
FAST LLM
    ↓
GROUNDED RESPONSE
```

Keep the diagram clean.

No dense architecture clutter on the main page.

---

# 32. Technology Feature Section

Use a layout similar to the supplied reference's feature cards.

```text
┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
│ SARVAM             │ │ MULTILINGUAL       │ │ QDRANT             │
│                    │ │                    │ │                    │
│ Speech → Text      │ │ Query → Vector     │ │ HNSW Retrieval     │
│                    │ │                    │ │                    │
│ Learn more →       │ │ Learn more →       │ │ Learn more →       │
└────────────────────┘ └────────────────────┘ └────────────────────┘

┌────────────────────┐ ┌────────────────────┐
│ FAST LLM           │ │ GROUNDING          │
│                    │ │                    │
│ Evidence → Answer  │ │ Evidence-first     │
│                    │ │ responses          │
│ Learn more →       │ │ Learn more →       │
└────────────────────┘ └────────────────────┘
```

Use mostly white cards with one light-green highlighted feature.

---

# 33. Benchmark Section

Heading:

```text
WHAT ACTUALLY WORKS?
```

Subheading:

```text
Compare retrieval quality against latency.
```

Table:

```text
CONFIGURATION      RECALL@5    P50     P100     RESULT

Baseline            89.2%      138ms   191ms    PASS
Model B             92.4%      146ms   198ms    PASS
Quantized           91.8%      121ms   176ms    ★
```

Recommended configuration:

```text
★ RECOMMENDED
```

---

# 34. Dense vs Hybrid

The UI must clearly distinguish the baseline architecture from experiments.

## Baseline

```text
DENSE VECTOR RAG
● ACTIVE
```

Flow:

```text
Query
 ↓
Embedding
 ↓
Qdrant + HNSW
 ↓
Top-K
```

## Experimental

```text
HYBRID
○ EXPERIMENTAL
```

Only activate Hybrid if it is actually implemented and benchmarked.

---

# 35. System Health

Heading:

```text
SYSTEM STATUS.
```

Rows:

```text
Sarvam STT        ● Operational      48 ms
Embedding         ● Operational      12 ms
Qdrant            ● Operational       9 ms
LLM               ● Operational      68 ms
```

Use green dots with text.

Never rely only on color.

---

# 36. Request Trace

Use a vertical editorial timeline.

```text
REQUEST / 8F21A91

13:31:42.001
● REQUEST RECEIVED

13:31:42.049
● SARVAM STT
  +48 ms

13:31:42.061
● EMBEDDING
  +12 ms

13:31:42.070
● QDRANT
  +9 ms

13:31:42.073
● CONTEXT
  +3 ms

13:31:42.141
● FIRST TOKEN
  +68 ms

13:31:42.156
● RESPONSE COMPLETE
  +15 ms

TOTAL
155 ms
```

Use JetBrains Mono for timestamps and durations.

---

# 37. FAQ Section

A FAQ near the bottom follows the long-scroll product-page pattern of the supplied reference.

Heading:

```text
QUESTIONS.
ANSWERS.
```

Items:

```text
What is VoiceRAG?
+
```

```text
Why use dense vector retrieval?
+
```

```text
Why Qdrant and HNSW?
+
```

```text
How does the system target <200ms?
+
```

```text
How are answers grounded?
+
```

```text
What happens when retrieval finds insufficient evidence?
+
```

Interaction:

- closed by default
- one item open at a time
- 150–200ms expansion
- green plus/chevron

---

# 38. Footer

```text
┌─────────────────────────────────────────────────────────────────────┐
│ ◇ VoiceRAG                                                          │
│                                                                     │
│ Voice-enabled dense retrieval system.                              │
│                                                                     │
│ Dashboard   Retrieval   Benchmarks   System                        │
│                                                                     │
│ © 2026 VoiceRAG                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

White background.

Top border:

```text
#E1E7E3
```

---

# 39. Card Rules

Do not make every section a card.

Use three visual layers.

### Major Surface

Large white or soft-green region.

### Metric Card

Small bordered surface.

### Editorial Data

Text plus separators; no card.

Recommended rhythm:

```text
BIG TEXT
    ↓
WHITE SPACE
    ↓
GREEN BLOCK
    ↓
DATA
    ↓
WHITE SPACE
    ↓
NEXT SECTION
```

---

# 40. Border and Shadow Rules

Prefer borders.

```text
border: 1px solid #E1E7E3
```

Optional subtle shadow:

```text
0 2px 8px rgba(16, 24, 40, 0.04)
```

Never use:

- large dark shadows
- neon glow
- glassmorphism

---

# 41. Radius

```text
Buttons:        8px
Inputs:         8px
Cards:          10–12px
Panels:         14–16px
Hero blocks:    16–20px
```

Avoid excessive pill-shaped components.

---

# 42. Icons

Use **Lucide** consistently.

Recommended:

```text
Mic
Search
Database
Activity
ShieldCheck
Clock
Check
AlertTriangle
ArrowUpRight
ChevronDown
Settings
BarChart3
```

---

# 43. Charts

Use:

- line chart for end-to-end latency
- horizontal bar for stage latency
- bar chart for Recall@K
- benchmark comparison table
- optional area chart for query volume

Avoid:

- pie charts
- 3D charts
- decorative charts
- multiple confusing color palettes

---

# 44. Motion

Motion should communicate state.

### Good

- microphone waveform
- pipeline stage activation
- number count-up
- progress
- request trace
- FAQ expansion

### Avoid

- particle backgrounds
- floating blobs
- animated gradients
- glowing borders
- constant movement

Timing:

```text
120–250ms
```

---

# 45. Responsive Behavior

## Desktop

```text
1440px+
```

Layout:

```text
2-column hero
3-column feature blocks
full-width analytics
```

## Tablet

```text
768–1439px
```

Layout:

```text
2-column hero
2-column features
stacked evidence
```

## Mobile

```text
<768px
```

Layout:

```text
1-column
stacked pipeline
full-width CTA
collapsed navigation
```

---

# 46. Mobile Hero

```text
VOICE IN.

KNOWLEDGE OUT.

Low-latency multilingual RAG.

[ RUN LIVE QUERY → ]

┌──────────────────────┐
│          ◉           │
│     HOLD TO SPEAK    │
└──────────────────────┘

142ms
END-TO-END
```

Do not simply shrink the desktop layout. Recompose it.

---

# 47. Accessibility

Requirements:

- WCAG AA contrast where practical
- keyboard-accessible microphone controls
- visible focus states
- text labels with status colors
- charts with textual summary values
- readable error messages

Example:

Bad:

```text
●
```

Good:

```text
● Operational
```

---

# 48. Primary Navigation

Keep it short:

```text
Dashboard
Live Query
Retrieval
Benchmarks
System
```

Secondary technical navigation:

```text
Dataset
Vector Index
Services
Logs
Settings
```

---

# 49. Dashboard vs Technical Pages

The Overview page should remain editorial and visual.

Technical pages can become denser.

```text
Overview
    ↓
Product-style presentation

Retrieval
    ↓
Data-dense inspection

Benchmarks
    ↓
Experiment console

Vector Index
    ↓
Infrastructure details

Logs
    ↓
Technical trace
```

This prevents the whole product from becoming a boring admin panel.

---

# 50. Main Dashboard Wireframe

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│ ◇ VoiceRAG       Dashboard   Live Query   Retrieval   Benchmarks   ● Healthy │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  VOICE IN.                                ┌────────────────────────────────┐ │
│  KNOWLEDGE OUT.                           │                                │ │
│                                           │             ◉                  │ │
│  Low-latency multilingual RAG            │                                │ │
│  for fast, grounded answers.             │       HOLD TO SPEAK             │ │
│                                           │       ● READY                   │ │
│  [ RUN LIVE QUERY → ]                    │                                │ │
│                                           └────────────────────────────────┘ │
│                                                                               │
│  142 MS                                                                     │
│  END-TO-END LATENCY                                                         │
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  FROM SPEECH TO ANSWER.                                                      │
│                                                                               │
│  01 LISTEN       02 RETRIEVE        03 ANSWER                              │
│  Sarvam STT      Qdrant + HNSW      Fast LLM                                │
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  THE PIPELINE.                                                              │
│                                                                               │
│  VOICE → EMBEDDING → QDRANT → CONTEXT → LLM                                │
│   48ms      12ms       9ms       3ms      68ms                             │
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  STAY UNDER 200MS.                                                           │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │ END-TO-END LATENCY                                                     │ │
│  │ 200ms ─────────────────────────────── TARGET                           │ │
│  │         ╭──╮                                                           │ │
│  │ 150ms ──╯  ╰────╮                                                      │ │
│  │                  ╰────                                                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  WHAT DID WE FIND?                                                           │
│                                                                               │
│  #01  0.923     Retrieved passage...                                         │
│  #02  0.891     Retrieved passage...                                         │
│  #03  0.842     Retrieved passage...                                         │
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  THE ANSWER.                                                                 │
│                                                                               │
│  Your grounded answer appears here...                                       │
│                                                                               │
│  ✓ GROUNDED      4 SUPPORTING PASSAGES      142 MS                         │
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  THE CORPUS.                  THE INDEX.                                     │
│                                                                               │
│  MSMARCO-XI                   QDRANT                                        │
│  12M+ records                 HNSW                                          │
│  ~52 GB                       12M+ vectors                                  │
│  14+ languages                cosine similarity                             │
│                                                                               │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  WHAT ACTUALLY WORKS?                                                        │
│                                                                               │
│  Benchmark results                                                           │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

# 51. Demo Flow

The demo should feel like one story.

### Step 1

Landing:

```text
VOICE IN.
KNOWLEDGE OUT.
```

### Step 2

Click:

```text
RUN LIVE QUERY
```

### Step 3

Speak.

### Step 4

Pipeline activates:

```text
STT
↓
Embedding
↓
Qdrant
↓
Context
↓
LLM
```

### Step 5

Retrieved evidence appears.

### Step 6

Answer appears.

### Step 7

Latency appears.

```text
142 ms
```

### Step 8

Open:

```text
VIEW EVIDENCE
```

### Step 9

Inspect supporting passages.

### Step 10

Open:

```text
BENCHMARKS
```

and show measured performance.

Narrative:

> **Input → Retrieval → Evidence → Answer → Performance → Proof**

---

# 52. Technical Truthfulness

All metrics in wireframes are examples.

Production UI must display actual telemetry.

Examples:

```text
142 ms
91.4% Recall
12M vectors
31.4 GB memory
```

must come from the actual runtime when implemented.

Likewise:

- "Grounded" must correspond to the implemented grounding logic.
- "Operational" must come from actual service health.
- vector counts must come from Qdrant.
- P50/P70/P100 must come from benchmark runs.

Never use fake metrics in a final demonstration unless clearly labeled as demo data.

---

# 53. Acceptance Criteria

The final design is approved when:

- White is the dominant surface.
- `#1F7335` is the main accent.
- The interface feels like a product, not an admin template.
- Hero typography is strong.
- Live voice interaction is obvious.
- The complete RAG pipeline is understandable.
- `<200 ms` is visible as a central performance target.
- Retrieval evidence is inspectable.
- Grounding status is visible.
- Qdrant/HNSW information is available.
- Dataset information is available.
- Benchmarks are easy to compare.
- FAQ/system information is accessible.
- Desktop and mobile layouts work.
- Status is communicated through text and icons, not color alone.
- Technical values use real telemetry in production.

---

# 54. Final Product Identity

The final experience should feel like:

```text
MODERN AI PRODUCT
       +
RAG OBSERVABILITY
       +
LIVE TECHNICAL DEMO
```

The visual hierarchy:

```text
1. PRODUCT
2. LIVE QUERY
3. EVIDENCE
4. ANSWER
5. LATENCY
6. PIPELINE
7. INFRASTRUCTURE
8. BENCHMARKS
```

The final visual signature:

```text
WHITE
   +
#1F7335
   +
BOLD TYPOGRAPHY
   +
LARGE WHITESPACE
   +
REAL-TIME RAG PIPELINE
   +
TECHNICAL EVIDENCE
```

Final product statement:

> **VOICE IN. KNOWLEDGE OUT.**
>
> **FAST. GROUNDED. MEASURABLE.**
