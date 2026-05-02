import type { AudioResult, AudioTag, TranscriptionSegment } from "../types.js";
import { extractAudio } from "../extractors/audio.js";
import { parseHMS, formatHMS } from "../utils/timestamps.js";
import type { Config } from "../types.js";

interface GenAiFile {
  name?: string;
  state?: string;
  uri?: string;
  mimeType?: string;
}

interface GenAiFilesApi {
  get(args: { name: string }): Promise<GenAiFile>;
  delete(args: { name: string }): Promise<void>;
}

interface GenAiClient {
  files: GenAiFilesApi;
}

interface WaitForFileActiveOptions {
  timeoutMs?: number;
  pollIntervalMs?: number;
}

const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_POLL_INTERVAL_MS = 2_000;

export async function waitForFileActive(
  ai: GenAiClient,
  file: GenAiFile,
  options: WaitForFileActiveOptions = {},
): Promise<GenAiFile> {
  if (!file.name) {
    throw new Error("Cannot poll Gemini file state: file.name is missing");
  }

  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const pollIntervalMs = options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
  const deadline = Date.now() + timeoutMs;

  let current = file;
  while (current.state !== "ACTIVE") {
    if (current.state === "FAILED") {
      throw new Error(
        `Gemini file ${current.name} processing failed`,
      );
    }

    if (Date.now() > deadline) {
      throw new Error(
        `Gemini file ${current.name} stuck in state ${current.state ?? "unknown"} after ${timeoutMs}ms`,
      );
    }

    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    current = await ai.files.get({ name: current.name! });
  }

  return current;
}

interface ParsedGeminiAudio {
  transcription: TranscriptionSegment[];
  audio_tags: AudioTag[];
}

export function parseGeminiAudioResponse(raw: string): ParsedGeminiAudio {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    throw new Error(
      `Gemini returned non-JSON response despite structured output config: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (typeof parsed !== "object" || parsed === null) {
    throw new Error("Gemini JSON response is not an object");
  }

  const obj = parsed as Record<string, unknown>;
  const transcription = Array.isArray(obj.transcription)
    ? (obj.transcription as TranscriptionSegment[])
    : [];
  const audio_tags = Array.isArray(obj.audio_tags)
    ? (obj.audio_tags as AudioTag[])
    : [];

  return { transcription, audio_tags };
}

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

function getMimeType(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  const mimeTypes: Record<string, string> = {
    wav: "audio/wav",
    mp3: "audio/mp3",
    aac: "audio/aac",
    flac: "audio/flac",
    ogg: "audio/ogg",
    aiff: "audio/aiff",
  };
  return mimeTypes[ext || ""] || "audio/wav";
}
