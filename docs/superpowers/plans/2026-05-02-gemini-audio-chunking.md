# Gemini Audio Chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Gemini audio transcription truncation on long videos (e.g., 36-min lectures) in the `claude-video-vision` plugin fork by swapping to a configurable model with a higher `maxOutputTokens`, then adding silence-aware chunking with parallel transcription and retry resilience.

**Architecture:** Two-phase ship. Phase 1 (Tier 1) extracts a `transcribeChunk` private worker, swaps the hardcoded model and adds `maxOutputTokens`, and exposes 5 new config fields via `video_configure`. Phase 2 (Tier 2/3) adds `audio-chunker.ts` for silence-aware boundary planning and orchestrates parallel chunk transcription with retry-once + sentinel-on-failure resilience inside `analyzeWithGeminiApi`.

**Tech Stack:** TypeScript, Node.js 20+, Vitest, `@google/genai` SDK, `zod`, `ffmpeg`. Plugin lives at `C:/_fg2/Code/F-Claude-Code-Vision/mcp-server/`.

**Spec:** `docs/superpowers/specs/2026-05-02-gemini-audio-chunking-design.md`

---

## File Structure

**Files to create:**
- `mcp-server/src/extractors/audio-chunker.ts` — silence detection + boundary planning (Phase 2)
- `mcp-server/tests/extractors/audio-chunker.test.ts` — unit tests for chunker (Phase 2)

**Files to modify:**
- `mcp-server/src/types.ts` — Phase 1: 5 new `Config` fields. Phase 2: `ChunkPlan`, `ChunkWarning`, `AudioResult.warnings`.
- `mcp-server/src/config.ts` — Phase 1: 5 new defaults in `defaultConfig`.
- `mcp-server/src/extractors/audio.ts` — Phase 1: optional `filename` param.
- `mcp-server/src/backends/gemini-api.ts` — Phase 1: extract `transcribeChunk`, swap model, add `maxOutputTokens`. Phase 2: orchestrator branch + retry wrapper + stitcher.
- `mcp-server/src/tools/video-configure.ts` — Phase 1: 5 zod schema entries.
- `mcp-server/src/tools/video-analyze.ts` — Phase 1: caller signature update.
- `mcp-server/src/tools/video-watch.ts` — Phase 1: caller signature update.
- `mcp-server/tests/config.test.ts` — Phase 1: defaults coverage.
- `mcp-server/tests/extractors/audio.test.ts` — Phase 1: `filename` param coverage.
- `mcp-server/tests/backends/gemini-api.test.ts` — Phase 1: `transcribeChunk` tests. Phase 2: orchestrator + retry tests.

---

# Phase 1 — Tier 1 (Commit 1)

## Task 1: Add new config fields

**Files:**
- Modify: `mcp-server/src/types.ts`
- Modify: `mcp-server/src/config.ts`
- Test: `mcp-server/tests/config.test.ts`

- [ ] **Step 1: Write the failing test**

Add this test to `mcp-server/tests/config.test.ts` (append inside the `describe("config", ...)` block, before the closing `});`):

```ts
  it("returns defaults for new audio fields", () => {
    const config = loadConfig(join(TEST_DIR, "config.json"));
    expect(config.audio_model).toBe("gemini-3-flash-preview");
    expect(config.audio_max_output_tokens).toBe(65536);
    expect(config.audio_chunk_trigger_seconds).toBe(1200);
    expect(config.audio_chunk_size_seconds).toBe(600);
    expect(config.audio_chunk_overlap_seconds).toBe(0);
  });

  it("preserves audio_model override when set", () => {
    const configPath = join(TEST_DIR, "config.json");
    writeFileSync(configPath, JSON.stringify({ audio_model: "gemini-3.1-pro-preview" }));
    const loaded = loadConfig(configPath);
    expect(loaded.audio_model).toBe("gemini-3.1-pro-preview");
    expect(loaded.audio_max_output_tokens).toBe(65536);
  });
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision/mcp-server
npm test -- config.test.ts
```

Expected: FAIL with TypeScript errors about unknown properties `audio_model` etc.

- [ ] **Step 3: Add fields to `Config` interface in `types.ts`**

In `mcp-server/src/types.ts`, modify the `Config` interface (currently lines 7-19) by appending the 5 fields before the closing brace:

```ts
export interface Config {
  backend: Backend;
  whisper_engine: WhisperEngine;
  whisper_model: WhisperModel;
  whisper_at: boolean;
  frame_mode: FrameMode;
  frame_resolution: number;
  default_fps: number | "auto";
  max_frames: number;
  frame_describer_model: DescriberModel;
  enable_index: boolean;
  session_max_age_days: number;
  audio_model: string;
  audio_max_output_tokens: number;
  audio_chunk_trigger_seconds: number;
  audio_chunk_size_seconds: number;
  audio_chunk_overlap_seconds: number;
}
```

- [ ] **Step 4: Add defaults to `defaultConfig` in `config.ts`**

In `mcp-server/src/config.ts`, modify `defaultConfig` (currently lines 6-18) by appending the 5 fields before the closing brace:

```ts
export const defaultConfig: Config = {
  backend: "unconfigured",
  whisper_engine: "cpp",
  whisper_model: "auto",
  whisper_at: false,
  frame_mode: "images",
  frame_resolution: 512,
  default_fps: "auto",
  max_frames: 100,
  frame_describer_model: "sonnet",
  enable_index: false,
  session_max_age_days: 7,
  audio_model: "gemini-3-flash-preview",
  audio_max_output_tokens: 65536,
  audio_chunk_trigger_seconds: 1200,
  audio_chunk_size_seconds: 600,
  audio_chunk_overlap_seconds: 0,
};
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
npm test -- config.test.ts
```

Expected: PASS for all `config` tests including the 2 new ones.

- [ ] **Step 6: Commit**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision
git add mcp-server/src/types.ts mcp-server/src/config.ts mcp-server/tests/config.test.ts
git commit -m "feat(config): add audio_model, max_output_tokens, and chunk fields with defaults"
```

---

## Task 2: Add `filename` param to `extractAudio`

**Files:**
- Modify: `mcp-server/src/extractors/audio.ts`
- Test: `mcp-server/tests/extractors/audio.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `mcp-server/tests/extractors/audio.test.ts` inside the `describe("audio extraction", ...)` block:

```ts
  it("respects custom filename param", async () => {
    const wavPath = await extractAudio(FIXTURE, OUT_DIR, {
      filename: "chunk-3.wav",
    });
    expect(wavPath.endsWith("chunk-3.wav")).toBe(true);
    expect(existsSync(wavPath)).toBe(true);
  });

  it("defaults to audio.wav when filename omitted", async () => {
    const wavPath = await extractAudio(FIXTURE, OUT_DIR);
    expect(wavPath.endsWith("audio.wav")).toBe(true);
  });
```

- [ ] **Step 2: Run test to verify the new test fails**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision/mcp-server
npm test -- audio.test.ts
```

Expected: FAIL on "respects custom filename param" — TypeScript error about unknown property `filename`.

- [ ] **Step 3: Add `filename` to `ExtractAudioOptions` and use it**

Replace `mcp-server/src/extractors/audio.ts` with:

```ts
import { execFile } from "child_process";
import { promisify } from "util";
import { mkdirSync, existsSync } from "fs";
import { join } from "path";

const execFileAsync = promisify(execFile);

export interface ExtractAudioOptions {
  startTime?: string;
  endTime?: string;
  filename?: string;
}

export async function extractAudio(
  videoPath: string,
  outputDir: string,
  options: ExtractAudioOptions = {},
): Promise<string> {
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir, { recursive: true });
  }

  const filename = options.filename ?? "audio.wav";
  const outputPath = join(outputDir, filename);
  const args: string[] = [];

  if (options.startTime) {
    args.push("-ss", options.startTime);
  }

  args.push("-i", videoPath);

  if (options.endTime) {
    args.push("-to", options.endTime);
  }

  args.push(
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "16000",
    "-ac", "1",
    "-y",
    outputPath,
  );

  await execFileAsync("ffmpeg", args);
  return outputPath;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test -- audio.test.ts
```

Expected: PASS for all `audio extraction` tests including the 2 new ones.

- [ ] **Step 5: Commit**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision
git add mcp-server/src/extractors/audio.ts mcp-server/tests/extractors/audio.test.ts
git commit -m "feat(audio): add optional filename param to extractAudio"
```

---

## Task 3: Add zod entries to `video-configure` schema

**Files:**
- Modify: `mcp-server/src/tools/video-configure.ts`

(No unit test — the existing repo has no tests for `video-configure.ts`; the schema is a thin pass-through validated by build + manual reload.)

- [ ] **Step 1: Add the 5 zod entries**

In `mcp-server/src/tools/video-configure.ts`, modify the `server.tool` call's schema object (currently lines 14-27) by appending the 5 new entries before the closing brace:

```ts
    {
      backend: z.enum(["gemini-api", "local", "openai"]).optional(),
      whisper_engine: z.enum(["cpp", "python"]).optional(),
      whisper_model: z.enum(["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3", "auto"]).optional(),
      whisper_at: z.boolean().optional(),
      frame_mode: z.enum(["images", "descriptions"]).optional(),
      frame_resolution: z.number().min(128).max(2048).optional(),
      default_fps: z.union([z.number().positive(), z.literal("auto")]).optional(),
      max_frames: z.number().min(1).max(1000).optional(),
      frame_describer_model: z.enum(["opus", "sonnet", "haiku"]).optional(),
      enable_index: z.boolean().optional(),
      session_max_age_days: z.number().min(1).optional(),
      clear_sessions: z.boolean().optional(),
      audio_model: z.string().min(1).optional(),
      audio_max_output_tokens: z.number().min(1024).max(200000).optional(),
      audio_chunk_trigger_seconds: z.number().min(60).optional(),
      audio_chunk_size_seconds: z.number().min(60).optional(),
      audio_chunk_overlap_seconds: z.number().min(0).max(60).optional(),
    },
```

- [ ] **Step 2: Build to verify TypeScript passes**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision/mcp-server
npm run build
```

Expected: build succeeds with no errors.

- [ ] **Step 3: Commit**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision
git add mcp-server/src/tools/video-configure.ts
git commit -m "feat(configure): expose audio_model and chunk fields via video_configure"
```

---

## Task 4: Refactor `gemini-api.ts` — extract `transcribeChunk`, swap model, add `maxOutputTokens`

**Files:**
- Modify: `mcp-server/src/backends/gemini-api.ts`
- Test: `mcp-server/tests/backends/gemini-api.test.ts`

- [ ] **Step 1: Write failing tests for `transcribeChunk`**

Append to `mcp-server/tests/backends/gemini-api.test.ts` (after the closing `});` of `describe("waitForFileActive", ...)`):

```ts
import { transcribeChunk } from "../../src/backends/gemini-api.js";
import type { Config } from "../../src/types.js";
import { defaultConfig } from "../../src/config.js";

function makeConfig(overrides: Partial<Config> = {}): Config {
  return { ...defaultConfig, ...overrides };
}

describe("transcribeChunk", () => {
  let originalEnv: string | undefined;

  beforeEach(() => {
    originalEnv = process.env.GEMINI_API_KEY;
    process.env.GEMINI_API_KEY = "test-key";
  });

  afterEach(() => {
    if (originalEnv === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalEnv;
    vi.restoreAllMocks();
  });

  it("passes config.audio_model and audio_max_output_tokens to generateContent", async () => {
    const generateContent = vi.fn(async () => ({
      text: JSON.stringify({ transcription: [], audio_tags: [] }),
    }));
    const upload = vi.fn(async () => ({ name: "files/x", uri: "gs://x", mimeType: "audio/wav", state: "ACTIVE" }));
    const fakeAi = {
      files: {
        upload,
        get: vi.fn(async (a: { name: string }) => ({ name: a.name, state: "ACTIVE", uri: "gs://x", mimeType: "audio/wav" })),
        delete: vi.fn(async () => {}),
      },
      models: { generateContent },
    };

    vi.doMock("@google/genai", () => ({
      GoogleGenAI: vi.fn(() => fakeAi),
      createPartFromUri: vi.fn(() => ({})),
      createUserContent: vi.fn((parts: unknown[]) => parts),
      Type: { OBJECT: "OBJECT", ARRAY: "ARRAY", STRING: "STRING" },
    }));

    const { transcribeChunk: fresh } = await import("../../src/backends/gemini-api.js?mock1");
    await fresh("/tmp/x.wav", 0, makeConfig({ audio_model: "gemini-3-flash-preview", audio_max_output_tokens: 65536 }));

    expect(generateContent).toHaveBeenCalledTimes(1);
    const callArgs = generateContent.mock.calls[0][0];
    expect(callArgs.model).toBe("gemini-3-flash-preview");
    expect(callArgs.config.maxOutputTokens).toBe(65536);
  });

  it("adds offsetSec to all returned timestamps", async () => {
    const fakeResponse = {
      text: JSON.stringify({
        transcription: [
          { start: "00:00:05", end: "00:00:10", text: "hello" },
          { start: "00:01:00", end: "00:01:05", text: "world" },
        ],
        audio_tags: [{ start: "00:00:30", end: "00:00:32", tag: "music" }],
      }),
    };
    const fakeAi = {
      files: {
        upload: vi.fn(async () => ({ name: "files/x", uri: "gs://x", mimeType: "audio/wav", state: "ACTIVE" })),
        get: vi.fn(async (a: { name: string }) => ({ name: a.name, state: "ACTIVE", uri: "gs://x", mimeType: "audio/wav" })),
        delete: vi.fn(async () => {}),
      },
      models: { generateContent: vi.fn(async () => fakeResponse) },
    };

    vi.doMock("@google/genai", () => ({
      GoogleGenAI: vi.fn(() => fakeAi),
      createPartFromUri: vi.fn(() => ({})),
      createUserContent: vi.fn((parts: unknown[]) => parts),
      Type: { OBJECT: "OBJECT", ARRAY: "ARRAY", STRING: "STRING" },
    }));

    const { transcribeChunk: fresh } = await import("../../src/backends/gemini-api.js?mock2");
    const offsetSec = 600; // chunk starting at 10 minutes
    const result = await fresh("/tmp/x.wav", offsetSec, makeConfig());

    expect(result.segments[0].start).toBe("00:10:05");
    expect(result.segments[0].end).toBe("00:10:10");
    expect(result.segments[1].start).toBe("00:11:00");
    expect(result.segments[1].end).toBe("00:11:05");
    expect(result.tags[0].start).toBe("00:10:30");
    expect(result.tags[0].end).toBe("00:10:32");
  });

  it("throws when GEMINI_API_KEY missing", async () => {
    delete process.env.GEMINI_API_KEY;
    vi.doMock("@google/genai", () => ({
      GoogleGenAI: vi.fn(),
      createPartFromUri: vi.fn(),
      createUserContent: vi.fn(),
      Type: {},
    }));
    const { transcribeChunk: fresh } = await import("../../src/backends/gemini-api.js?mock3");
    await expect(fresh("/tmp/x.wav", 0, makeConfig())).rejects.toThrow(/GEMINI_API_KEY/);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision/mcp-server
npm test -- gemini-api.test.ts
```

Expected: FAIL on `transcribeChunk` tests — function not exported yet.

- [ ] **Step 3: Refactor `gemini-api.ts`**

Replace the `analyzeWithGeminiApi` function (currently lines 91-180) and add a new `transcribeChunk` export. Add this import at the top of `mcp-server/src/backends/gemini-api.ts`:

```ts
import { extractAudio } from "../extractors/audio.js";
import { parseHMS, formatHMS } from "../utils/timestamps.js";
import type { Config } from "../types.js";
```

Then replace the existing `analyzeWithGeminiApi` function (the one starting `export async function analyzeWithGeminiApi(audioPath: string)`) with:

```ts
function offsetTimestamp(hms: string, offsetSec: number): string {
  return formatHMS(parseHMS(hms) + offsetSec);
}

export async function transcribeChunk(
  wavPath: string,
  offsetSec: number,
  config: Config,
): Promise<{ segments: TranscriptionSegment[]; tags: AudioTag[] }> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error("GEMINI_API_KEY environment variable is not set. Run video_setup to configure.");
  }

  const { GoogleGenAI, createPartFromUri, createUserContent, Type } = await import("@google/genai");
  const ai = new GoogleGenAI({ apiKey });

  const uploaded = await ai.files.upload({
    file: wavPath,
    config: { mimeType: getMimeType(wavPath) },
  });

  await waitForFileActive(ai as unknown as GenAiClient, uploaded);

  try {
    const response = await ai.models.generateContent({
      model: config.audio_model,
      contents: createUserContent([
        createPartFromUri(uploaded.uri!, uploaded.mimeType!),
        `Analyze this audio track and return structured JSON.

Produce two arrays:
1. "transcription": one entry per contiguous speech segment, with start and end timestamps as "HH:MM:SS" strings and the spoken text verbatim.
2. "audio_tags": one entry per non-speech audio event (music, sound effects, ambient sounds) with start and end timestamps as "HH:MM:SS" strings and a short lowercase label.

Use "00:00:00" if you cannot determine a timestamp. Return empty arrays if no speech or no non-speech events are present.`,
      ]),
      config: {
        thinkingConfig: { thinkingBudget: 0 },
        maxOutputTokens: config.audio_max_output_tokens,
        responseMimeType: "application/json",
        responseJsonSchema: {
          type: Type.OBJECT,
          properties: {
            transcription: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  start: { type: Type.STRING, description: "HH:MM:SS start timestamp" },
                  end: { type: Type.STRING, description: "HH:MM:SS end timestamp" },
                  text: { type: Type.STRING, description: "Verbatim spoken text for this segment" },
                },
                propertyOrdering: ["start", "end", "text"],
                required: ["start", "end", "text"],
              },
            },
            audio_tags: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  start: { type: Type.STRING, description: "HH:MM:SS start timestamp" },
                  end: { type: Type.STRING, description: "HH:MM:SS end timestamp" },
                  tag: { type: Type.STRING, description: "Short lowercase label for the audio event" },
                },
                propertyOrdering: ["start", "end", "tag"],
                required: ["start", "end", "tag"],
              },
            },
          },
          propertyOrdering: ["transcription", "audio_tags"],
          required: ["transcription", "audio_tags"],
        },
      },
    });

    const parsed = parseGeminiAudioResponse(response.text ?? "");
    const segments = parsed.transcription.map(s => ({
      start: offsetTimestamp(s.start, offsetSec),
      end: offsetTimestamp(s.end, offsetSec),
      text: s.text,
    }));
    const tags = parsed.audio_tags.map(t => ({
      start: offsetTimestamp(t.start, offsetSec),
      end: offsetTimestamp(t.end, offsetSec),
      tag: t.tag,
    }));

    return { segments, tags };
  } finally {
    await ai.files.delete({ name: uploaded.name! }).catch(() => {});
  }
}

export interface AudioSlice {
  startTime?: string;
  endTime?: string;
}

export async function analyzeWithGeminiApi(
  videoPath: string,
  config: Config,
  slice?: AudioSlice,
): Promise<AudioResult> {
  const { mkdtempSync, rmSync } = await import("fs");
  const { tmpdir } = await import("os");
  const { join } = await import("path");
  const tmpDir = mkdtempSync(join(tmpdir(), "cvv-gemini-"));

  try {
    const wavPath = await extractAudio(videoPath, tmpDir, {
      startTime: slice?.startTime,
      endTime: slice?.endTime,
    });
    const { segments, tags } = await transcribeChunk(wavPath, 0, config);
    return {
      backend: "gemini-api",
      transcription: segments,
      audio_tags: tags,
      full_analysis: null,
    };
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
}
```

Verify `mcp-server/src/utils/timestamps.ts` exports both `formatHMS` and `parseHMS`. If `parseHMS` is missing, add it to that file:

```ts
export function parseHMS(hms: string): number {
  const parts = hms.split(":").map(Number);
  if (parts.length !== 3) throw new Error(`Invalid HMS string: ${hms}`);
  const [h, m, s] = parts;
  return h * 3600 + m * 60 + s;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test -- gemini-api.test.ts
```

Expected: PASS for `waitForFileActive` (existing) AND `transcribeChunk` (new).

- [ ] **Step 5: Commit**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision
git add mcp-server/src/backends/gemini-api.ts mcp-server/src/utils/timestamps.ts mcp-server/tests/backends/gemini-api.test.ts
git commit -m "refactor(gemini): extract transcribeChunk worker, swap model, add maxOutputTokens"
```

---

## Task 5: Update callers — `video-analyze.ts` and `video-watch.ts`

**Files:**
- Modify: `mcp-server/src/tools/video-analyze.ts`
- Modify: `mcp-server/src/tools/video-watch.ts`

(No new unit tests — these are call-site updates verified by `npm run build` succeeding.)

- [ ] **Step 1: Update `video-analyze.ts` caller**

In `mcp-server/src/tools/video-analyze.ts` line 213-216, the existing block is:

```ts
          if (config.backend === "gemini-api") {
            const audioDir = join(workDir, "audio");
            const wavPath = await extractAudio(safePath, audioDir, {});
            audioResult = await analyzeWithGeminiApi(wavPath);
          } else if (config.backend === "openai") {
```

Replace with:

```ts
          if (config.backend === "gemini-api") {
            audioResult = await analyzeWithGeminiApi(safePath, config);
          } else if (config.backend === "openai") {
```

- [ ] **Step 2: Update `video-watch.ts` caller**

In `mcp-server/src/tools/video-watch.ts` lines 124-129, the existing block is:

```ts
      } else if (config.backend === "gemini-api") {
        const audioDir = join(workDir, "audio");
        audioPromise = extractAudio(safePath, audioDir, {
          startTime: params.start_time,
          endTime: params.end_time,
        }).then((wavPath) => analyzeWithGeminiApi(wavPath));
      } else if (config.backend === "openai") {
```

Replace with:

```ts
      } else if (config.backend === "gemini-api") {
        audioPromise = analyzeWithGeminiApi(safePath, config, {
          startTime: params.start_time,
          endTime: params.end_time,
        });
      } else if (config.backend === "openai") {
```

- [ ] **Step 3: Build to verify**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision/mcp-server
npm run build
```

Expected: build succeeds.

- [ ] **Step 4: Run full test suite**

```bash
npm test
```

Expected: all existing tests + new tests pass.

- [ ] **Step 5: Commit**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision
git add mcp-server/src/tools/video-analyze.ts mcp-server/src/tools/video-watch.ts
git commit -m "refactor(tools): pass videoPath + config to analyzeWithGeminiApi"
```

---

## Task 6: Tier 1 Verification Checkpoint (manual)

After Task 5 commits, the `transcribeChunk` worker is in place, the model is swapped, and `maxOutputTokens` is set. This task is manual integration verification before Phase 2 begins.

- [ ] **Step 1: Build production output**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision/mcp-server
npm install        # if not already
npm run build
```

- [ ] **Step 2: Patch `.mcp.json` locally for testing (do not commit)**

The repo's `.mcp.json` runs `npx claude-video-vision@latest` from the npm registry. To use the local fork:

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision
cat > .mcp.json <<'EOF'
{
  "claude-video-vision": {
    "command": "node",
    "args": ["C:/_fg2/Code/F-Claude-Code-Vision/mcp-server/dist/index.js"]
  }
}
EOF
```

This change is local-only — do NOT commit it. Verify with `git status` that `.mcp.json` shows as modified.

- [ ] **Step 3: Install local plugin in Claude Code**

In a Claude Code session:

```
/plugin uninstall claude-video-vision
```

Then exit Claude Code and start a new session pointing at the fork:

```bash
claude --plugin-dir C:/_fg2/Code/F-Claude-Code-Vision
```

In the new session, verify the plugin loaded:

```
/plugin
```

Expected: lists `claude-video-vision` as enabled.

- [ ] **Step 4: Verify config picked up new defaults**

```
/setup-video-vision
```

Or call `video_configure` with no args via the MCP tool. Expected output: config JSON now contains `audio_model: "gemini-3-flash-preview"`, `audio_max_output_tokens: 65536`, etc.

- [ ] **Step 5: Smoke test on a short video (≤15 min)**

Find or record a short test video (use the 3-second fixture if nothing else available, or a YouTube clip download). Run via the `video_analyze` MCP tool with `transcription: true`. Expected: completes without truncation error, returns transcription segments.

- [ ] **Step 6: Test on the 36-min PH lecture**

```
video_analyze C:/Users/farha/Downloads/"Pulmonary Hypertension Review of Diagnosis and Treatment.mp4"
  with filters { transcription: true }
```

Expected behavior — two possible outcomes:
- **Pass:** model + maxOutputTokens alone solved the problem. Phase 2 still ships for resilience on longer videos.
- **Fail with similar truncation error:** confirms Phase 2 chunking is required. Note the new error position; it'll inform whether 65536 tokens was even close.

- [ ] **Step 7: Document the Phase 1 result**

Append a note to the spec at `docs/superpowers/specs/2026-05-02-gemini-audio-chunking-design.md` under a new `## Phase 1 Verification` section: paste the test outcome (pass/fail), the date, and any error messages observed. Commit:

```bash
git add docs/superpowers/specs/2026-05-02-gemini-audio-chunking-design.md
git commit -m "docs(spec): record Phase 1 verification result"
```

---

# Phase 2 — Tier 2/3 (Commit 2 logical milestone)

## Task 7: Add `ChunkPlan`, `ChunkWarning`, and `AudioResult.warnings` types

**Files:**
- Modify: `mcp-server/src/types.ts`
- Test: `mcp-server/tests/types.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `mcp-server/tests/types.test.ts`:

```ts
import type { ChunkPlan, ChunkWarning, AudioResult } from "../src/types.js";

describe("chunking types", () => {
  it("ChunkPlan has expected shape", () => {
    const plan: ChunkPlan = {
      start: 0,
      actual_start: 0,
      end: 600,
      index: 0,
      total: 4,
      clean_cut: true,
    };
    expect(plan.start).toBe(0);
    expect(plan.clean_cut).toBe(true);
  });

  it("ChunkWarning has expected event types", () => {
    const w: ChunkWarning = {
      chunk_index: 0,
      chunk_total: 4,
      time_range: "00:00-10:00",
      event: "retry",
      detail: "Gemini 500",
    };
    expect(w.event).toBe("retry");
  });

  it("AudioResult.warnings is optional", () => {
    const r: AudioResult = {
      backend: "gemini-api",
      transcription: [],
      audio_tags: [],
      full_analysis: null,
    };
    expect(r.warnings).toBeUndefined();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npm test -- types.test.ts
```

Expected: FAIL — `ChunkPlan`, `ChunkWarning` not exported.

- [ ] **Step 3: Add the new types to `types.ts`**

Append at the end of `mcp-server/src/types.ts`:

```ts
export interface ChunkPlan {
  start: number;
  actual_start: number;
  end: number;
  index: number;
  total: number;
  clean_cut: boolean;
}

export interface ChunkWarning {
  chunk_index: number;
  chunk_total: number;
  time_range: string;
  event: "retry" | "failed" | "hard_cut" | "loose_threshold";
  detail?: string;
}
```

And modify the existing `AudioResult` interface (currently lines 51-56) to add the optional `warnings` field:

```ts
export interface AudioResult {
  backend: Backend;
  transcription: TranscriptionSegment[];
  audio_tags: AudioTag[];
  full_analysis: string | null;
  warnings?: ChunkWarning[];
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test -- types.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp-server/src/types.ts mcp-server/tests/types.test.ts
git commit -m "feat(types): add ChunkPlan, ChunkWarning, AudioResult.warnings"
```

---

## Task 8: Create `audio-chunker.ts` with `detectSilences` and `planChunks`

**Files:**
- Create: `mcp-server/src/extractors/audio-chunker.ts`
- Test: `mcp-server/tests/extractors/audio-chunker.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `mcp-server/tests/extractors/audio-chunker.test.ts`:

```ts
import { describe, it, expect, vi } from "vitest";
import { planChunks, type SilenceDetector } from "../../src/extractors/audio-chunker.js";
import { defaultConfig } from "../../src/config.js";

describe("planChunks", () => {
  it("returns single chunk when duration <= chunk_size", async () => {
    const fakeDetector: SilenceDetector = vi.fn(async () => []);
    const { chunks, warnings } = await planChunks("/x.mp4", 500, defaultConfig, fakeDetector);
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toMatchObject({ start: 0, end: 500, index: 0, total: 1 });
    expect(warnings).toEqual([]);
  });

  it("uses clean silence cuts when silences fall near ideal boundaries", async () => {
    const fakeDetector: SilenceDetector = vi.fn(async () => [
      { start: "00:09:55", end: "00:10:05", duration: 10 },
      { start: "00:19:50", end: "00:20:00", duration: 10 },
      { start: "00:29:58", end: "00:30:02", duration: 4 },
    ]);
    const { chunks } = await planChunks("/x.mp4", 2400, defaultConfig, fakeDetector);
    expect(chunks).toHaveLength(4);
    expect(chunks.map(p => p.clean_cut)).toEqual([true, true, true, true]);
    expect(fakeDetector).toHaveBeenCalledTimes(1);
  });

  it("retries with looser threshold when default returns insufficient silences", async () => {
    let callCount = 0;
    const fakeDetector: SilenceDetector = vi.fn(async (_path, threshold) => {
      callCount++;
      if (threshold === "default") return [];
      return [{ start: "00:09:50", end: "00:10:10", duration: 20 }];
    });
    const { chunks, warnings } = await planChunks("/x.mp4", 1300, defaultConfig, fakeDetector);
    expect(callCount).toBe(2);
    expect(chunks).toHaveLength(2);
    expect(chunks[0].clean_cut).toBe(true);
    expect(warnings.some(w => w.event === "loose_threshold")).toBe(true);
  });

  it("falls back to hard cut and emits hard_cut warning when no silence found", async () => {
    const fakeDetector: SilenceDetector = vi.fn(async () => []);
    const { chunks, warnings } = await planChunks("/x.mp4", 1300, defaultConfig, fakeDetector);
    expect(chunks).toHaveLength(2);
    expect(chunks[0].clean_cut).toBe(false);
    expect(chunks[0].end).toBe(600); // exact ideal boundary
    expect(warnings.some(w => w.event === "hard_cut")).toBe(true);
  });

  it("respects audio_chunk_overlap_seconds in actual_start", async () => {
    const fakeDetector: SilenceDetector = vi.fn(async () => []);
    const config = { ...defaultConfig, audio_chunk_overlap_seconds: 5 };
    const { chunks } = await planChunks("/x.mp4", 1300, config, fakeDetector);
    expect(chunks[0].actual_start).toBe(0); // first chunk: max(0, 0-5) = 0
    expect(chunks[1].start).toBe(600);
    expect(chunks[1].actual_start).toBe(595); // 600 - 5
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
npm test -- audio-chunker.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Create `audio-chunker.ts`**

Create `mcp-server/src/extractors/audio-chunker.ts`:

```ts
import { execFile } from "child_process";
import { promisify } from "util";
import { join } from "path";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import type { ChunkPlan, ChunkWarning, Config, Interval } from "../types.js";
import { parseSilenceOutput } from "./analyzers.js";
import { parseHMS } from "../utils/timestamps.js";

const execFileAsync = promisify(execFile);

export type SilenceThreshold = "default" | "loose";
export type SilenceDetector = (videoPath: string, threshold: SilenceThreshold) => Promise<Interval[]>;

const TOLERANCE_SECONDS = 30;

const SILENCE_PARAMS: Record<SilenceThreshold, string> = {
  default: "silencedetect=n=-40dB:d=0.5",
  loose: "silencedetect=n=-30dB:d=0.2",
};

export async function detectSilencesReal(
  videoPath: string,
  threshold: SilenceThreshold,
): Promise<Interval[]> {
  const tmp = mkdtempSync(join(tmpdir(), "cvv-silence-"));
  try {
    const args = ["-i", videoPath, "-af", SILENCE_PARAMS[threshold], "-f", "null", "-"];
    let stderr = "";
    try {
      const r = await execFileAsync("ffmpeg", args, { maxBuffer: 100 * 1024 * 1024 });
      stderr = r.stderr;
    } catch (err: any) {
      stderr = err.stderr || "";
    }
    return parseSilenceOutput(stderr);
  } finally {
    rmSync(tmp, { recursive: true, force: true });
  }
}

interface BoundaryMatch {
  boundary: number;
  silenceMidpoint: number | null;
  clean_cut: boolean;
  threshold: SilenceThreshold | null;
}

function findNearestSilence(boundary: number, silences: Interval[]): number | null {
  let nearestMidpoint: number | null = null;
  let nearestDistance = Infinity;
  for (const s of silences) {
    const mid = (parseHMS(s.start) + parseHMS(s.end)) / 2;
    const distance = Math.abs(mid - boundary);
    if (distance <= TOLERANCE_SECONDS && distance < nearestDistance) {
      nearestDistance = distance;
      nearestMidpoint = mid;
    }
  }
  return nearestMidpoint;
}

export async function planChunks(
  videoPath: string,
  durationSec: number,
  config: Config,
  detector: SilenceDetector = detectSilencesReal,
): Promise<{ chunks: ChunkPlan[]; warnings: ChunkWarning[] }> {
  const chunkSize = config.audio_chunk_size_seconds;
  const overlap = config.audio_chunk_overlap_seconds;

  // Single-chunk path
  if (durationSec <= chunkSize) {
    return {
      chunks: [{ start: 0, actual_start: 0, end: durationSec, index: 0, total: 1, clean_cut: true }],
      warnings: [],
    };
  }

  // Compute ideal boundaries (multiples of chunkSize, less than durationSec)
  const idealBoundaries: number[] = [];
  for (let t = chunkSize; t < durationSec; t += chunkSize) {
    idealBoundaries.push(t);
  }

  // Pass 1: default threshold
  const defaultSilences = await detector(videoPath, "default");
  const matches: BoundaryMatch[] = idealBoundaries.map(b => {
    const mid = findNearestSilence(b, defaultSilences);
    return {
      boundary: b,
      silenceMidpoint: mid,
      clean_cut: mid !== null,
      threshold: mid !== null ? "default" : null,
    };
  });

  // Pass 2: loose threshold for unmatched
  const unmatched = matches.filter(m => !m.clean_cut);
  if (unmatched.length > 0) {
    const looseSilences = await detector(videoPath, "loose");
    for (const m of unmatched) {
      const mid = findNearestSilence(m.boundary, looseSilences);
      if (mid !== null) {
        m.silenceMidpoint = mid;
        m.clean_cut = true;
        m.threshold = "loose";
      }
    }
  }

  // Build chunks + warnings
  const warnings: ChunkWarning[] = [];
  const total = matches.length + 1;
  const chunks: ChunkPlan[] = [];
  let prevEnd = 0;
  matches.forEach((m, i) => {
    const end = m.silenceMidpoint ?? m.boundary;
    const start = prevEnd;
    chunks.push({
      start,
      actual_start: Math.max(0, start - overlap),
      end,
      index: i,
      total,
      clean_cut: m.clean_cut,
    });
    if (m.threshold === "loose") {
      warnings.push({
        chunk_index: i,
        chunk_total: total,
        time_range: `${formatRange(start, end)}`,
        event: "loose_threshold",
        detail: "matched silence using loose threshold",
      });
    }
    if (!m.clean_cut) {
      warnings.push({
        chunk_index: i,
        chunk_total: total,
        time_range: `${formatRange(start, end)}`,
        event: "hard_cut",
        detail: `no silence within ±${TOLERANCE_SECONDS}s of target boundary`,
      });
    }
    prevEnd = end;
  });
  // Final chunk to end of video
  chunks.push({
    start: prevEnd,
    actual_start: Math.max(0, prevEnd - overlap),
    end: durationSec,
    index: matches.length,
    total,
    clean_cut: true, // trailing chunk has no boundary to cut at
  });

  return { chunks, warnings };
}

function formatRange(startSec: number, endSec: number): string {
  const m = (s: number) => {
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  };
  return `${m(startSec)}-${m(endSec)}`;
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test -- audio-chunker.test.ts
```

Expected: PASS for all 5 tests.

- [ ] **Step 5: Commit**

```bash
git add mcp-server/src/extractors/audio-chunker.ts mcp-server/tests/extractors/audio-chunker.test.ts
git commit -m "feat(chunker): silence-aware boundary planning with threshold ladder"
```

---

## Task 9: Add `transcribeChunkWithRetry` wrapper

**Files:**
- Modify: `mcp-server/src/backends/gemini-api.ts`
- Test: `mcp-server/tests/backends/gemini-api.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `mcp-server/tests/backends/gemini-api.test.ts`:

```ts
import { transcribeChunkWithRetry } from "../../src/backends/gemini-api.js";

describe("transcribeChunkWithRetry", () => {
  it("returns ok=true on first-try success", async () => {
    const worker = vi.fn(async () => ({ segments: [], tags: [] }));
    const result = await transcribeChunkWithRetry("/x.wav", 0, makeConfig(), 1, worker);
    expect(result.ok).toBe(true);
    expect(result.attempt).toBe(0);
    expect(worker).toHaveBeenCalledTimes(1);
  });

  it("retries once and returns ok=true with attempt=1 + retry warning", async () => {
    const worker = vi.fn()
      .mockRejectedValueOnce(new Error("Gemini 500"))
      .mockResolvedValueOnce({ segments: [], tags: [] });
    const onWarning = vi.fn();
    const result = await transcribeChunkWithRetry("/x.wav", 0, makeConfig(), 1, worker, onWarning);
    expect(result.ok).toBe(true);
    expect(result.attempt).toBe(1);
    expect(worker).toHaveBeenCalledTimes(2);
    expect(onWarning).toHaveBeenCalledWith(expect.objectContaining({ event: "retry" }));
  });

  it("returns ok=false after retries exhausted", async () => {
    const worker = vi.fn().mockRejectedValue(new Error("persistent fail"));
    const onWarning = vi.fn();
    const result = await transcribeChunkWithRetry("/x.wav", 0, makeConfig(), 1, worker, onWarning);
    expect(result.ok).toBe(false);
    expect(result.attempt).toBe(2);
    expect(worker).toHaveBeenCalledTimes(2);
    expect(onWarning).toHaveBeenCalledTimes(1);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npm test -- gemini-api.test.ts
```

Expected: FAIL — `transcribeChunkWithRetry` not exported.

- [ ] **Step 3: Add `transcribeChunkWithRetry` to `gemini-api.ts`**

Append to `mcp-server/src/backends/gemini-api.ts`:

```ts
export interface ChunkResult {
  ok: boolean;
  attempt: number;
  segments?: TranscriptionSegment[];
  tags?: AudioTag[];
  error?: string;
}

export type TranscribeWorker = (
  wavPath: string,
  offsetSec: number,
  config: Config,
) => Promise<{ segments: TranscriptionSegment[]; tags: AudioTag[] }>;

export type WarningEmitter = (w: { event: "retry"; attempt: number; error: string }) => void;

export async function transcribeChunkWithRetry(
  wavPath: string,
  offsetSec: number,
  config: Config,
  retries: number,
  worker: TranscribeWorker = transcribeChunk,
  onWarning?: WarningEmitter,
): Promise<ChunkResult> {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const r = await worker(wavPath, offsetSec, config);
      return { ok: true, attempt, segments: r.segments, tags: r.tags };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      if (attempt < retries) {
        if (onWarning) onWarning({ event: "retry", attempt, error: msg });
        continue;
      }
      return { ok: false, attempt: attempt + 1, error: msg };
    }
  }
  return { ok: false, attempt: retries + 1, error: "unreachable" };
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test -- gemini-api.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mcp-server/src/backends/gemini-api.ts mcp-server/tests/backends/gemini-api.test.ts
git commit -m "feat(gemini): add transcribeChunkWithRetry wrapper"
```

---

## Task 10: Wire orchestrator into `analyzeWithGeminiApi`

**Files:**
- Modify: `mcp-server/src/backends/gemini-api.ts`
- Test: `mcp-server/tests/backends/gemini-api.test.ts`

- [ ] **Step 1: Write the failing tests**

The orchestrator design uses dependency injection — the `analyzeWithGeminiApi` function accepts an optional `deps` param so tests can inject stubs without `vi.doMock` gymnastics. Append to `mcp-server/tests/backends/gemini-api.test.ts`:

```ts
import { analyzeWithGeminiApi } from "../../src/backends/gemini-api.js";

describe("analyzeWithGeminiApi orchestrator", () => {
  const baseConfig = { ...defaultConfig, audio_chunk_trigger_seconds: 1200, audio_chunk_size_seconds: 600 };

  function metaStub(seconds: number) {
    return vi.fn(async () => ({
      duration: "x", duration_seconds: seconds,
      resolution: "x", width: 0, height: 0, codec: "h264",
      original_fps: 30, file_size: "x", has_audio: true,
    }));
  }

  beforeEach(() => {
    process.env.GEMINI_API_KEY = "test-key";
  });

  it("uses single-call path when duration <= chunk_trigger", async () => {
    const worker = vi.fn(async () => ({
      segments: [{ start: "00:00:00", end: "00:00:05", text: "short" }],
      tags: [],
    }));
    const extract = vi.fn(async () => "/tmp/audio.wav");
    const result = await analyzeWithGeminiApi("/x.mp4", baseConfig, undefined, {
      getMetadata: metaStub(600),
      extract,
      worker,
    });
    expect(worker).toHaveBeenCalledTimes(1);
    expect(result.transcription).toHaveLength(1);
    expect(result.warnings).toBeUndefined();
  });

  it("uses chunked path when duration > chunk_trigger", async () => {
    const worker = vi.fn(async (_wav, offset) => ({
      segments: [{ start: "00:00:00", end: "00:00:05", text: `at ${offset}` }],
      tags: [],
    }));
    const extract = vi.fn(async (_v: string, _d: string, opts?: { filename?: string }) =>
      `/tmp/${opts?.filename ?? "audio.wav"}`,
    );
    const silenceDetector = vi.fn(async () => []);
    const result = await analyzeWithGeminiApi("/x.mp4", baseConfig, undefined, {
      getMetadata: metaStub(2400),
      extract,
      worker,
      silenceDetector,
    });
    expect(worker).toHaveBeenCalledTimes(4);
    expect(result.transcription).toHaveLength(4);
    expect(result.warnings).toBeDefined();
    expect(result.warnings!.filter(w => w.event === "hard_cut")).toHaveLength(3);
  });

  it("emits sentinel segment when chunk fails after retry", async () => {
    // Chunk 2 (offset=1200) fails on every call. Other chunks succeed first try.
    // Failure is keyed by offset (not by callCount) so it's deterministic under
    // Promise.all parallelism.
    const FAIL_OFFSET = 1200;
    const worker = vi.fn(async (_wav: string, offset: number) => {
      if (offset === FAIL_OFFSET) throw new Error("Gemini 500");
      return { segments: [{ start: "00:00:00", end: "00:00:05", text: "ok" }], tags: [] };
    });
    const extract = vi.fn(async (_v: string, _d: string, opts?: { filename?: string }) =>
      `/tmp/${opts?.filename ?? "audio.wav"}`,
    );
    const silenceDetector = vi.fn(async () => []);
    const result = await analyzeWithGeminiApi("/x.mp4", baseConfig, undefined, {
      getMetadata: metaStub(2400),
      extract,
      worker,
      silenceDetector,
    });
    const failedSegments = result.transcription.filter(t => t.text.includes("transcription failed"));
    expect(failedSegments).toHaveLength(1);
    expect(result.warnings!.filter(w => w.event === "failed")).toHaveLength(1);
    expect(result.warnings!.filter(w => w.event === "retry")).toHaveLength(1);
  });
});
```

These tests will fail because the current `analyzeWithGeminiApi` (from Task 4) doesn't accept a `deps` param and doesn't have a chunked branch.

- [ ] **Step 2: Run test to verify it fails**

```bash
npm test -- gemini-api.test.ts
```

Expected: FAIL — orchestrator doesn't yet support chunked path.

- [ ] **Step 3: Replace `analyzeWithGeminiApi` body with orchestrator + DI hooks**

In `mcp-server/src/backends/gemini-api.ts`, add this import:

```ts
import { planChunks, type SilenceDetector } from "../extractors/audio-chunker.js";
import { getVideoMetadata } from "../extractors/frames.js";
```

And replace the existing `analyzeWithGeminiApi` (created in Task 4) with:

```ts
export interface AnalyzeDeps {
  getMetadata?: typeof getVideoMetadata;
  extract?: typeof extractAudio;
  worker?: TranscribeWorker;
  silenceDetector?: SilenceDetector;
}

export async function analyzeWithGeminiApi(
  videoPath: string,
  config: Config,
  slice?: AudioSlice,
  deps: AnalyzeDeps = {},
): Promise<AudioResult> {
  const getMetadata = deps.getMetadata ?? getVideoMetadata;
  const extract = deps.extract ?? extractAudio;
  const worker = deps.worker ?? transcribeChunk;
  const silenceDetector = deps.silenceDetector;

  const { mkdtempSync, rmSync } = await import("fs");
  const { tmpdir } = await import("os");
  const { join } = await import("path");
  const tmpDir = mkdtempSync(join(tmpdir(), "cvv-gemini-"));

  try {
    // Slice mode (start/end set) always uses single-call path
    if (slice?.startTime || slice?.endTime) {
      const wavPath = await extract(videoPath, tmpDir, {
        startTime: slice.startTime,
        endTime: slice.endTime,
      });
      const { segments, tags } = await worker(wavPath, 0, config);
      return {
        backend: "gemini-api",
        transcription: segments,
        audio_tags: tags,
        full_analysis: null,
      };
    }

    const metadata = await getMetadata(videoPath);

    if (metadata.duration_seconds <= config.audio_chunk_trigger_seconds) {
      const wavPath = await extract(videoPath, tmpDir);
      const { segments, tags } = await worker(wavPath, 0, config);
      return {
        backend: "gemini-api",
        transcription: segments,
        audio_tags: tags,
        full_analysis: null,
      };
    }

    // Chunked path
    const { chunks, warnings: planWarnings } = await planChunks(
      videoPath,
      metadata.duration_seconds,
      config,
      silenceDetector,
    );

    const wavPaths = await Promise.all(
      chunks.map(c =>
        extract(videoPath, tmpDir, {
          startTime: secondsToHMS(c.actual_start),
          endTime: secondsToHMS(c.end),
          filename: `chunk-${c.index}.wav`,
        }),
      ),
    );

    const allWarnings: ChunkWarning[] = [...planWarnings];

    const results = await Promise.all(
      chunks.map((c, i) =>
        transcribeChunkWithRetry(
          wavPaths[i],
          c.start,
          config,
          1,
          worker,
          (w) => {
            allWarnings.push({
              chunk_index: c.index,
              chunk_total: c.total,
              time_range: hmsRange(c.start, c.end),
              event: w.event,
              detail: w.error,
            });
          },
        ),
      ),
    );

    const transcription: TranscriptionSegment[] = [];
    const audio_tags: AudioTag[] = [];
    chunks.forEach((c, i) => {
      const r = results[i];
      if (r.ok) {
        transcription.push(...(r.segments ?? []));
        audio_tags.push(...(r.tags ?? []));
      } else {
        transcription.push({
          start: secondsToHMS(c.start),
          end: secondsToHMS(c.end),
          text: "[transcription failed for this segment after retry]",
        });
        allWarnings.push({
          chunk_index: c.index,
          chunk_total: c.total,
          time_range: hmsRange(c.start, c.end),
          event: "failed",
          detail: r.error,
        });
      }
    });

    return {
      backend: "gemini-api",
      transcription,
      audio_tags,
      full_analysis: null,
      warnings: allWarnings,
    };
  } finally {
    rmSync(tmpDir, { recursive: true, force: true });
  }
}

function secondsToHMS(sec: number): string {
  return formatHMS(sec);
}

function hmsRange(startSec: number, endSec: number): string {
  return `${secondsToHMS(startSec)}-${secondsToHMS(endSec)}`;
}
```

**Note on Promise.all vs Promise.allSettled:** the spec mentions `Promise.allSettled`, but `transcribeChunkWithRetry` (Task 9) catches all worker exceptions internally and always resolves to a `ChunkResult`. So `Promise.all` here never sees a rejected promise — semantically equivalent to `Promise.allSettled` for this flow, with a cleaner result shape (no `.value` / `.reason` unwrapping).

**Note on slice + chunked path:** when the caller passes `slice` (start/end timestamps), the orchestrator always uses the single-call path regardless of full-video duration. Rationale: a sliced range is by definition a focused subset; chunking it adds complexity for a use case that's already narrowed. The chunker's `audio_chunk_trigger_seconds` only applies to whole-video transcription.

- [ ] **Step 4: Run tests to verify they pass**

```bash
npm test -- gemini-api.test.ts
```

Expected: PASS for all tests in this file (the 7 existing `waitForFileActive` tests plus the new 3 `transcribeChunk`, 3 `transcribeChunkWithRetry`, and 3 orchestrator tests).

- [ ] **Step 5: Commit**

```bash
git add mcp-server/src/backends/gemini-api.ts mcp-server/tests/backends/gemini-api.test.ts
git commit -m "feat(gemini): orchestrate parallel chunks with retry + sentinel + warnings"
```

---

## Task 11: Update MCP tool descriptions to mention warnings

**Files:**
- Modify: `mcp-server/src/tools/video-watch.ts`
- Modify: `mcp-server/src/tools/video-analyze.ts`

(No test — descriptions are documentation strings.)

- [ ] **Step 1: Update `video-analyze.ts` tool description**

In `mcp-server/src/tools/video-analyze.ts`, the tool description (currently passed as the second arg to `server.tool` around line 46) currently reads:

```ts
"Analyze video structure using ffmpeg filters. Returns scene changes, silence intervals, motion levels, and more. Use this before video_watch to plan which segments need detailed frame extraction. Does not extract frames.",
```

Replace with:

```ts
"Analyze video structure using ffmpeg filters. Returns scene changes, silence intervals, motion levels, and more. Use this before video_watch to plan which segments need detailed frame extraction. Does not extract frames. When transcription is enabled and the video is longer than the configured chunk trigger, the audio is chunked and transcribed in parallel; the result may include a `warnings` array describing chunk-boundary decisions, retries, or failures — surface these to the user when present.",
```

- [ ] **Step 2: Update `video-watch.ts` tool description**

Find the `server.tool("video_watch", ...)` description string and append a similar sentence about `warnings`. Exact existing text varies; the addition is:

```
When the video is long enough to trigger audio chunking, the result's audio.warnings array (if present) reports chunk-boundary decisions and any per-chunk retries or failures — surface these to the user.
```

- [ ] **Step 3: Build to verify**

```bash
npm run build
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add mcp-server/src/tools/video-watch.ts mcp-server/src/tools/video-analyze.ts
git commit -m "docs(tools): document audio.warnings in video_watch and video_analyze descriptions"
```

---

## Task 12: Tier 2/3 Verification Checkpoint (manual)

- [ ] **Step 1: Build and run all tests**

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision/mcp-server
npm run build
npm test
```

Expected: all tests pass.

- [ ] **Step 2: Restart Claude Code with the local fork (if not already)**

Same procedure as Task 6 Step 3. If `.mcp.json` is still patched locally, just restart Claude Code with `--plugin-dir`.

- [ ] **Step 3: Run on the 36-min PH lecture**

```
video_analyze C:/Users/farha/Downloads/"Pulmonary Hypertension Review of Diagnosis and Treatment.mp4"
  with filters { transcription: true }
```

Expected:
- Result completes successfully (no truncation error).
- `analysis.transcription` spans `00:00:00` to ~`00:36:24` continuously, no large gaps.
- If `warnings` is populated, inspect: should see ~3 `hard_cut` events (silence detection returned empty earlier, so all chunk boundaries default to hard cut after the loose-threshold pass also fails) and 0 `failed` events.
- Time-to-result: roughly 30s-2min (4 chunks transcribed in parallel).

- [ ] **Step 4: Document the Phase 2 result**

Append to the spec at `docs/superpowers/specs/2026-05-02-gemini-audio-chunking-design.md` under `## Phase 2 Verification`: paste the test outcome, the date, observed `warnings` array, and time-to-result. Commit:

```bash
git add docs/superpowers/specs/2026-05-02-gemini-audio-chunking-design.md
git commit -m "docs(spec): record Phase 2 verification result"
```

- [ ] **Step 5: Final cleanup**

If the local `.mcp.json` patch is no longer needed for ongoing development, restore it:

```bash
cd C:/_fg2/Code/F-Claude-Code-Vision
git checkout -- .mcp.json
```

Or keep it out for ongoing local testing and add `.mcp.json` to a personal `.git/info/exclude` to silence its dirty status.

---

## Summary

| Phase | Commits | Net effect |
|-------|---------|------------|
| Phase 1 (Tier 1) | 5 | Model swap to `gemini-3-flash-preview`, `maxOutputTokens=65536`, 5 new config fields, refactor to single-call path with extracted worker |
| Phase 1 verify | 1 (docs) | Manual integration verified |
| Phase 2 (Tier 2/3) | 5 | Silence-aware chunker, parallel orchestrator, retry-once + sentinel, warnings surface |
| Phase 2 verify | 1 (docs) | Manual integration verified on 36-min lecture |

Total: ~12 commits, two phases with explicit verification gates between them.
