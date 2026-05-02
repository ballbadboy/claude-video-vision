# Gemini Audio Chunking — Design Spec

**Date:** 2026-05-02
**Author:** FarhadGSRX (with Claude)
**Status:** Approved for implementation
**Repo:** FarhadGSRX/claude-video-vision (fork of jordanrendric/claude-video-vision)

## Problem

The plugin's Gemini audio backend fails on long videos. A 36-minute lecture
returned `Unterminated string in JSON at position 220951` — Gemini truncated
its structured-output JSON response mid-string. Root causes in
`mcp-server/src/backends/gemini-api.ts`:

1. Model hardcoded to `gemini-2.5-flash` with no config knob.
2. No `maxOutputTokens` set on the `generateContent` call; SDK default applies.
3. Entire audio file sent as one request expecting one structured-JSON response.
   No chunking. Long lectures generate transcripts that exceed the output cap
   the model uses for structured generation.

## Goals

- Lift the output-cap ceiling so videos in the 30-50 min range transcribe
  successfully in a single call.
- For longer videos, chunk audio along silence boundaries and transcribe
  chunks in parallel with retry-and-continue resilience.
- Expose model and chunking knobs via `video_configure` so tuning doesn't
  require a rebuild.

## Non-Goals

- Whisper backend changes. OpenAI and local backends are untouched.
- Frame extraction or visual-perception logic.
- Cross-chunk transcript dedup / continuity-smoothing post-processor. The
  overlap *mechanism* ships in this spec; the Haiku post-processor that
  consumes the overlap region is a follow-on design.
- Live integration tests against the Gemini API. CI uses mocks; live
  verification is manual per the checkpoints below.

## Architecture

The single public function `analyzeWithGeminiApi` becomes an orchestrator
around an extracted private worker `transcribeChunk`. For short videos the
orchestrator calls the worker once with offset 0 — current behavior. For long
videos it plans chunk boundaries, extracts per-chunk wav files via the
existing `extractAudio` extractor, fires `transcribeChunk` calls in parallel,
and stitches the results.

```
analyzeWithGeminiApi(videoPath, config)
  ├─ duration <= chunk_trigger:
  │    extractAudio(videoPath) → transcribeChunk(wav, offset=0)
  └─ duration > chunk_trigger:
       planChunks(videoPath, config) → ChunkPlan[]
       Promise.allSettled(chunks.map(c =>
         extractAudio(videoPath, {start, end, filename})
         → transcribeChunkWithRetry(wav, c.start, config, retries=1)
       ))
       stitchResults(plan, results) → AudioResult { transcription, audio_tags, warnings }
```

### What stays the same

- `extractAudio` extractor (already accepts `startTime`/`endTime` — fits
  per-chunk extraction with one small change for filename).
- `parseGeminiAudioResponse` and the Gemini structured-output JSON schema.
- The contract callers see — `analyzeWithGeminiApi` returns `AudioResult`
  populated with `transcription`, `audio_tags`, `full_analysis`. New optional
  `warnings` field added.

### What's new

- `mcp-server/src/extractors/audio-chunker.ts` — silence detection, threshold
  ladder, boundary planning.
- `transcribeChunk(wavPath, offsetSec, config)` — private worker in
  `gemini-api.ts`. Single Gemini call with `maxOutputTokens` set. Returns
  segments and tags with timestamps already offset to absolute.
- `transcribeChunkWithRetry` — wraps `transcribeChunk` with one retry on
  failure and emits `ChunkWarning` events.
- `stitchResults(plan, results)` — concatenates transcripts, fills failed
  chunks with sentinel segments, threads warnings through.

## Audio Chunking Module

### `detectSilences(videoPath, threshold) → Interval[]`

Wraps the existing `silencedetect` ffmpeg call from `analyzers.ts`. Uses
`parseSilenceOutput` already present in that file. Single audio-filter pass.

### `planChunks(videoPath, durationSec, config) → ChunkPlan[]`

Threshold-ladder logic for finding chunk boundaries:

```
ideal = [config.audio_chunk_size_seconds, 2*size, 3*size, ...] up to duration
silences = detectSilences(default_threshold = -40dB / 0.5s)

for each ideal boundary t:
  pick silence within [t - tolerance, t + tolerance]  # tolerance = 30s
  match if found

if any boundaries unmatched:
  silences = detectSilences(loose_threshold = -30dB / 0.2s)
  retry boundary matching for unmatched only
  emit ChunkWarning(event: "loose_threshold") for each match found this way

for any still-unmatched boundary:
  use exact ideal time t (hard cut)
  emit ChunkWarning(event: "hard_cut")

return chunks where each chunk's
  actualStart = max(0, start - config.audio_chunk_overlap_seconds)
```

The `actualStart` field is what gets passed to `extractAudio`; the logical
`start` field is what gets passed to `transcribeChunk` as `offsetSec`.

### `ChunkPlan` type (added to `types.ts`)

```ts
interface ChunkPlan {
  start: number;        // logical start (seconds, absolute)
  actualStart: number;  // extraction start (= max(0, start - overlap))
  end: number;          // seconds, absolute
  index: number;        // 0-based
  total: number;        // total chunks
  cleanCut: boolean;    // true if start boundary snapped to silence
}
```

### Why no overlap by default

Default `audio_chunk_overlap_seconds = 0`. Adding overlap is asymmetric: chunk
N's `actualStart = max(0, start - overlap)`, while `end` stays at the logical
boundary. Chunks N-1 and N share `overlap` seconds of audio, which gets
transcribed twice — once at the tail of chunk N-1 and once at the head of
chunk N (with offset added).

**Footgun if overlap > 0 without a dedup post-processor:** the returned
transcript will contain duplicated segments around each chunk boundary. This
spec ships the field but does not add the dedup pass — that lands as a
follow-on Haiku-driven commit. Until then, leave the default at 0 unless
manually post-processing the result.

## Backend Refactor (`gemini-api.ts`)

### Public

```ts
analyzeWithGeminiApi(
  videoPath: string,    // changed from wavPath — needs duration + chunking
  config: Config,       // new param — reads model, chunk size, overlap, etc.
): Promise<AudioResult>
```

### Private worker

```ts
transcribeChunk(
  wavPath: string,
  offsetSec: number,    // added to every returned timestamp
  config: Config,       // for model name + maxOutputTokens
): Promise<{ segments: TranscriptionSegment[]; tags: AudioTag[] }>
```

Implementation: same Gemini Files-API + `generateContent` flow as today, with
two changes:

1. `model: config.audio_model` (was hardcoded `"gemini-2.5-flash"`).
2. Add `maxOutputTokens: config.audio_max_output_tokens` to `config` block of
   the `generateContent` call.

After parsing, every segment's `start`/`end` timestamps get `offsetSec` added
before returning. Same for `audio_tags`.

### Retry wrapper

```ts
transcribeChunkWithRetry(wavPath, offsetSec, config, retries = 1)
  → ChunkResult { ok: boolean, attempt: number, ... }
```

Loops up to `retries + 1` total attempts. On exception, emits a
`ChunkWarning(event: "retry")` between attempts. On final failure, returns
`{ ok: false, error }` — caller (orchestrator) constructs the sentinel
segment in `stitchResults`.

### `extractAudio` change

Add optional `filename` param to `ExtractAudioOptions` (defaults to
`"audio.wav"`). Currently the output path is hardcoded — chunked extraction
needs distinct filenames in the same temp directory. Roughly two lines.

## Concurrency + Resilience

### Parallelism

Orchestrator uses `Promise.allSettled` over all chunks. For a 36-min lecture
at default 10-min chunks that's 4 concurrent calls. Free-tier Gemini Flash
RPM allows it (≥10 RPM, well above 4 in flight). [Inference]

If future use needs >10 chunks (>100-min video at 10-min chunks), a `p-limit`
queue would be added. Out of scope here — flagged as future work.

### Sentinel for failed chunk

When `transcribeChunkWithRetry` returns `ok: false`, `stitchResults` inserts:

```ts
{
  start: HMS(plan.start),
  end: HMS(plan.end),
  text: "[transcription failed for this segment after retry]"
}
```

Keeps the timeline contiguous. Downstream consumers (the user, Claude
reading the transcript) see exactly which time range is missing. `audio_tags`
for that range stays empty.

### Warnings surface

`AudioResult` extended with optional `warnings: ChunkWarning[]`. Set only when
the chunked path runs. The name `warnings` is an umbrella for chunk events
worth surfacing — `failed` and `retry` are problems; `hard_cut` and
`loose_threshold` are informational notes about boundary-selection quality
that the user may or may not care about.

```ts
interface ChunkWarning {
  chunk_index: number;
  chunk_total: number;
  time_range: string;            // "06:00-12:00"
  event: "retry" | "failed" | "hard_cut" | "loose_threshold";
  detail?: string;               // e.g., "Gemini 500" or "no silence within ±30s"
}
```

Sources:
- `planChunks` emits `loose_threshold` and `hard_cut` events during boundary
  selection.
- `transcribeChunkWithRetry` emits `retry` and `failed` events.

The MCP tool description for `video_watch`/`video_analyze` should mention
that `result.warnings`, when present, indicates partial transcription
quality that should be surfaced to the user.

### File cleanup

Each `transcribeChunk` call uploads its wav to Gemini Files API. The existing
`try { ... } finally { ai.files.delete(...) }` pattern stays inside
`transcribeChunk` so per-chunk uploads are individually cleaned up even when
some chunks fail.

## Config Surface

### New fields in `Config` (`types.ts`)

```ts
audio_model: string;                    // default: "gemini-3-flash-preview"
audio_max_output_tokens: number;        // default: 65536
audio_chunk_trigger_seconds: number;    // default: 1200 (20 min)
audio_chunk_size_seconds: number;       // default: 600 (10 min)
audio_chunk_overlap_seconds: number;    // default: 0
```

### `video_configure` schema additions

```ts
audio_model: z.string().min(1).optional(),
audio_max_output_tokens: z.number().min(1024).max(200000).optional(),
audio_chunk_trigger_seconds: z.number().min(60).optional(),
audio_chunk_size_seconds: z.number().min(60).optional(),
audio_chunk_overlap_seconds: z.number().min(0).max(60).optional(),
```

### What's not exposed (intentionally)

Hardcoded for now:
- Silence-detection thresholds (`-40dB / 0.5s` default, `-30dB / 0.2s` loose).
- Boundary tolerance window (±30s).
- Retry count (1).
- Parallelism cap (unbounded; relevant only for future >10-chunk videos).

These are promoted to config only when a real use case demands tuning. YAGNI.

### `audio_model` as free string (not enum)

Reason: Gemini ships preview models on rolling cadence. Enum requires a code
change every release. Free string lets the user point at any current or
future model name without rebuild. A typo fails at API call time with a
"model not found" error from Gemini, which is clear enough.

### Backwards compatibility for existing config files

`loadConfig` already does `{ ...defaultConfig, ...raw }`. Existing user
config files without these fields inherit new defaults on next load. No
migration needed.

## Tier Cut Lines

The implementation ships in two commits.

### Commit 1 — Tier 1 (model + output cap + config fields)

- Add all 5 config fields to `types.ts` and `defaultConfig` in `config.ts`.
- Add the 5 fields to the `video_configure` zod schema.
- Extract `transcribeChunk(wavPath, offsetSec, config)` from existing
  `analyzeWithGeminiApi` body.
- In `transcribeChunk`: use `config.audio_model` and add
  `maxOutputTokens: config.audio_max_output_tokens` to the `generateContent`
  call. Apply `offsetSec` to returned timestamps (offset is 0 in Tier 1, but
  the math runs).
- `analyzeWithGeminiApi` signature changes from `(wavPath)` to
  `(videoPath, config)`. Body in Tier 1: extract audio from videoPath,
  call `transcribeChunk(wav, 0, config)`, return result. No duration check
  yet, no chunk branch yet — that arrives in Tier 2.
- Update callers in `tools/video-analyze.ts` and any other tool that calls
  `analyzeWithGeminiApi` to pass `videoPath` + `config` (they previously
  pre-extracted audio and passed the wav path; that pre-extraction moves
  inside the function).

Verification: `video_analyze --transcription` on a ≤15-min test video
succeeds. Then on the 36-min PH lecture: success here would mean the
gemini-3-flash-preview output budget alone handled the lecture, but Tier 2/3
still ships for longer-video resilience and the model-swap-only fix is not
durable to videos longer than the model's output cap.

### Commit 2 — Tier 2/3 (chunking + resilience + warnings)

- New file `extractors/audio-chunker.ts` with `detectSilences` and
  `planChunks`.
- `extractAudio` gains optional `filename` param.
- `transcribeChunkWithRetry` wrapper.
- `stitchResults` function.
- `analyzeWithGeminiApi` orchestrator branches on duration vs
  `audio_chunk_trigger_seconds`.
- `AudioResult.warnings` field added to types.
- `MCP tool descriptions` updated to mention `warnings`.

Verification: 36-min PH lecture transcribes end-to-end; 4 entries in
`warnings` reflect chunk-boundary decisions; transcript timestamps span
00:00:00 to ~36:24 with no gaps; no sentinel segments unless transient
API issues occurred.

## Testing

| Unit | Test | Why |
|------|------|-----|
| `transcribeChunk` | mock `@google/genai`, assert correct model + maxOutputTokens passed; assert offset added to all timestamps | Tier 1 correctness, offset arithmetic is a likely bug source |
| `transcribeChunkWithRetry` | inject failing-then-succeeding mock → asserts retry once + warning emitted; inject always-failing → asserts `ok: false` returned | Resilience contract |
| `planChunks` | fixture: 36-min duration with 0 silences → all hard cuts; fixture with silences at all targets → all clean cuts; mixed → mix | Threshold ladder branches |
| `stitchResults` | 4 chunks, mark 1 + 3 as failed → sentinel segments at right times, warnings include both, audio_tags concatenated correctly | Most error-prone integration point |
| `extractAudio` | new `filename` param respected; defaults to `"audio.wav"` when omitted | Don't break existing callers |

No integration test against live Gemini in CI. Live verification is manual
per the Tier checkpoints above.

## Open Items / Future Work

- Cross-chunk transcript dedup post-processor (Haiku) — needs its own design
  once the overlap mechanism has been tested with a non-zero default.
- `p-limit` queue for >10 concurrent chunks (videos >100 min at default
  10-min chunks).
- Promote silence thresholds / tolerance / retry count to config once a
  use case demands tuning.
- The "fallback after retry exhaustion" — currently a sentinel segment.
  Future: try a smaller chunk size for that segment, or flip backend to
  local Whisper for that one chunk. Out of scope.
