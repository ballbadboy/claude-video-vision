import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { waitForFileActive } from "../../src/backends/gemini-api.js";

interface FakeFile {
  name?: string;
  state?: string;
  uri?: string;
  mimeType?: string;
}

function fakeClient(states: string[]): {
  client: { files: { get: ReturnType<typeof vi.fn>; delete: ReturnType<typeof vi.fn> } };
  calls: { get: number };
} {
  let index = 0;
  const calls = { get: 0 };
  return {
    client: {
      files: {
        get: vi.fn(async ({ name }: { name: string }) => {
          calls.get++;
          const state = states[Math.min(index, states.length - 1)];
          index++;
          return { name, state, uri: `gs://fake/${name}`, mimeType: "video/mp4" };
        }),
        delete: vi.fn(async () => {}),
      },
    },
    calls,
  };
}

describe("waitForFileActive", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("returns immediately when file is already ACTIVE", async () => {
    const { client, calls } = fakeClient(["ACTIVE"]);
    const file: FakeFile = { name: "files/abc", state: "ACTIVE" };

    const result = await waitForFileActive(client, file);

    expect(result.state).toBe("ACTIVE");
    expect(calls.get).toBe(0);
  });

  it("polls while PROCESSING then returns when ACTIVE", async () => {
    const { client, calls } = fakeClient(["PROCESSING", "ACTIVE"]);
    const file: FakeFile = { name: "files/abc", state: "PROCESSING" };

    const promise = waitForFileActive(client, file, {
      pollIntervalMs: 100,
      timeoutMs: 10_000,
    });
    await vi.advanceTimersByTimeAsync(300);
    const result = await promise;

    expect(result.state).toBe("ACTIVE");
    expect(calls.get).toBe(2);
  });

  it("throws on FAILED state after upload", async () => {
    const { client } = fakeClient(["FAILED"]);
    const file: FakeFile = { name: "files/abc", state: "FAILED" };

    await expect(waitForFileActive(client, file)).rejects.toThrow(
      /processing failed/,
    );
  });

  it("throws on FAILED state detected during polling", async () => {
    const { client } = fakeClient(["PROCESSING", "FAILED"]);
    const file: FakeFile = { name: "files/abc", state: "PROCESSING" };

    const promise = waitForFileActive(client, file, {
      pollIntervalMs: 100,
      timeoutMs: 10_000,
    });
    promise.catch(() => {});
    await vi.advanceTimersByTimeAsync(300);

    await expect(promise).rejects.toThrow(/processing failed/);
  });

  it("throws after timeout when stuck in PROCESSING", async () => {
    const { client } = fakeClient(["PROCESSING"]);
    const file: FakeFile = { name: "files/abc", state: "PROCESSING" };

    const promise = waitForFileActive(client, file, {
      pollIntervalMs: 50,
      timeoutMs: 200,
    });
    promise.catch(() => {});
    await vi.advanceTimersByTimeAsync(500);

    await expect(promise).rejects.toThrow(/stuck in state PROCESSING after 200ms/);
  });

  it("handles STATE_UNSPECIFIED by polling until ACTIVE", async () => {
    const { client, calls } = fakeClient(["STATE_UNSPECIFIED", "ACTIVE"]);
    const file: FakeFile = { name: "files/abc", state: "STATE_UNSPECIFIED" };

    const promise = waitForFileActive(client, file, {
      pollIntervalMs: 100,
      timeoutMs: 10_000,
    });
    await vi.advanceTimersByTimeAsync(300);
    const result = await promise;

    expect(result.state).toBe("ACTIVE");
    expect(calls.get).toBe(2);
  });

  it("throws when file.name is missing", async () => {
    const { client } = fakeClient(["PROCESSING"]);
    const file: FakeFile = { state: "PROCESSING" };

    await expect(waitForFileActive(client, file)).rejects.toThrow(
      /file\.name is missing/,
    );
  });

  it("uses default timeout and poll interval when options omitted", async () => {
    const { client } = fakeClient(["ACTIVE"]);
    const file: FakeFile = { name: "files/abc", state: "ACTIVE" };

    const result = await waitForFileActive(client, file);
    expect(result.state).toBe("ACTIVE");
  });
});

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
    vi.resetModules();
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
      GoogleGenAI: vi.fn(function () { return fakeAi; }),
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
      GoogleGenAI: vi.fn(function () { return fakeAi; }),
      createPartFromUri: vi.fn(() => ({})),
      createUserContent: vi.fn((parts: unknown[]) => parts),
      Type: { OBJECT: "OBJECT", ARRAY: "ARRAY", STRING: "STRING" },
    }));

    const { transcribeChunk: fresh } = await import("../../src/backends/gemini-api.js?mock2");
    const offsetSec = 600;
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
