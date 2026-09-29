/**
 * codegpt-pi — a pi provider extension for CodeGPT.
 *
 * Registers the `codegpt` provider against CodeGPT's cloud API. It does NOT
 * require a local CodeGPT install: authentication is CodeGPT's own device flow
 * (a browser URL plus a server-push SSE channel), and inference goes straight to
 * api.codegpt.co.
 *
 * Why a loopback proxy: pi's built-in `openai-completions` client is pointed at
 * a small HTTP server started here. The proxy rewrites the OpenAI request into
 * the shape CodeGPT's `/chat/tools` route expects (agentId, requestId, bare
 * tool definitions) and pipes CodeGPT's OpenAI-format SSE straight back, so pi
 * keeps its own message/tool conversion, usage accounting, and cancellation.
 *
 * Usage inside a normal pi install:
 *   pi -e /path/to/codegpt-pi
 *   /login codegpt        # device flow — open the printed URL on any device
 *   /model                # pick a CodeGPT model
 *
 * Headless: the login URL can be opened on another machine; tokens arrive over
 * the SSE connection this process holds. Or set CODEGPT_SESSION_TOKEN to a
 * token copied from a machine that already signed in.
 *
 * Env overrides (useful for staging): CODEGPT_BACKEND_URL, CODEGPT_PLAYGROUND_URL.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type { OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";

// =============================================================================
// Constants
// =============================================================================

const PROVIDER_ID = "codegpt";

/** CodeGPT session tokens arrive as `session-<token>` and never expire. */
const SESSION_TOKEN_ENV = "CODEGPT_SESSION_TOKEN";

/**
 * CodeGPT's base agent. Its `/chat/tools` route requires a UUID agentId; the
 * model is carried separately on the wire, so one base agent serves the whole
 * credits roster. Matches BASE_AGENT_ID in the CodeGPT bridge session route.
 */
const BASE_AGENT_ID = "2385a3fe-c991-4070-bac2-ecbad0840b70";

/** Channel label the backend accepts for this bridge. */
const CHANNEL = "codegpt-pi";

/** A far-future expiry (seconds) means "does not expire". */
const NEVER_EXPIRES_MS = 253402300799000;

/**
 * The model picker row and --list-models print the model id, not the name, and
 * pi exposes no tier/badge field — so the tier is appended to the id here and
 * stripped by the proxy before the request goes to CodeGPT. Never appears in a
 * real CodeGPT model id.
 */
const TIER_SEPARATOR = "/";

/** Matches the tier suffix appended to a model id; stripped before the wire. */
const TIER_SUFFIX_RE = /\/(economy|free|premium)$/i;

const backend = () =>
	(process.env.CODEGPT_BACKEND_URL ?? "https://api.codegpt.co/api/v1").replace(/\/+$/, "");
const playground = () =>
	(process.env.CODEGPT_PLAYGROUND_URL ?? "https://app.codegpt.co").replace(/\/+$/, "");

/** Stable-ish conversation id; only used for CodeGPT-side session grouping. */
const SESSION_ID = `pi-${randomUUID().slice(0, 24)}`;

// =============================================================================
// Wire helpers (ported from codegpt-nextjs core/harness)
// =============================================================================

/**
 * Normalize OpenAI tool definitions to CodeGPT's bare `{name,description,parameters}`
 * shape, keeping only portable JSON-Schema keywords.
 */
function toolsForBackend(value: unknown): unknown[] | undefined {
	if (!Array.isArray(value)) return undefined;
	return value.flatMap((tool) => {
		const fn = (tool as any)?.type === "function" ? (tool as any).function : tool;
		if (!fn || typeof fn.name !== "string" || typeof fn.description !== "string") return [];
		const parameters: Record<string, unknown> = { ...portableSchema(fn.parameters), type: "object" };
		if (!parameters.properties) parameters.properties = {};
		return [{ name: fn.name, description: fn.description, parameters }];
	});
}

function portableSchema(value: unknown): Record<string, unknown> {
	if (!value || typeof value !== "object" || Array.isArray(value)) return {};
	const schema = value as Record<string, unknown>;
	const out: Record<string, unknown> = {};
	for (const key of ["type", "description", "format", "nullable", "minimum", "maximum", "minItems", "maxItems"]) {
		if (schema[key] !== undefined) out[key] = schema[key];
	}
	if (Array.isArray(schema.enum)) out.enum = schema.enum;
	if (Array.isArray(schema.required)) out.required = schema.required.filter((v) => typeof v === "string");
	if (schema.items) out.items = portableSchema(schema.items);
	if (Array.isArray(schema.anyOf)) out.anyOf = schema.anyOf.map(portableSchema);
	if (schema.properties && typeof schema.properties === "object" && !Array.isArray(schema.properties)) {
		out.properties = Object.fromEntries(
			Object.entries(schema.properties as Record<string, unknown>).map(([name, child]) => [name, portableSchema(child)]),
		);
	}
	return out;
}

// Strip JSON-invalid control characters (strict model servers reject raw NUL/ESC).
// Tool results are held to NUL only so file bytes edit matches on are preserved.
const JSON_UNSAFE_CONTROL = /[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g;
const NUL_ONLY = /\u0000/g;

function sanitizeValue(value: unknown, fileContent: boolean): unknown {
	if (typeof value === "string") {
		const re = fileContent ? NUL_ONLY : JSON_UNSAFE_CONTROL;
		re.lastIndex = 0;
		return re.test(value) ? value.replace(re, "") : value;
	}
	if (Array.isArray(value)) return value.map((v) => sanitizeValue(v, fileContent));
	if (value && typeof value === "object") {
		const out: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(value)) out[k] = sanitizeValue(v, fileContent);
		return out;
	}
	return value;
}

function sanitizeMessages(messages: unknown): unknown {
	if (!Array.isArray(messages)) return messages;
	return messages.map((m) => sanitizeValue(m, (m as any)?.role === "tool"));
}

// =============================================================================
// Authentication (independent of any local CodeGPT install)
// =============================================================================

function credentialFromToken(token: string): OAuthCredentials {
	return { access: token, refresh: token, expires: NEVER_EXPIRES_MS };
}

function normalizeExpiry(value: unknown): number {
	if (typeof value === "number" && Number.isFinite(value)) {
		return value < 1e12 ? value * 1000 : value; // seconds vs ms
	}
	if (typeof value === "string") {
		const parsed = Date.parse(value);
		if (!Number.isNaN(parsed)) return parsed;
	}
	return NEVER_EXPIRES_MS;
}

/**
 * CodeGPT device login. Mints a connection id, sends the user's browser to the
 * playground login carrying it, then waits on the backend's SSE channel until
 * the completed login pushes the tokens. No local callback server, so it works
 * on a headless machine — the URL is opened elsewhere and the tokens arrive here.
 */
async function loginCodeGPT(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
	const envToken = process.env[SESSION_TOKEN_ENV];
	if (envToken) return credentialFromToken(envToken);

	const connectionId = randomUUID();
	const params = new URLSearchParams({
		source: "vscode",
		distinct_id: "codegpt-pi",
		connection_id: connectionId,
		flow_version: "1.1",
	});
	callbacks.onAuth({
		url: `${playground()}/login?${params.toString()}`,
		instructions: "Open this URL on any device with a browser, sign in, and return here.",
	});
	callbacks.onProgress?.("Waiting for CodeGPT sign-in to complete…");

	const res = await fetch(`${backend()}/extensions/connection/${connectionId}`, {
		headers: { accept: "text/event-stream" },
		signal: callbacks.signal,
	});
	if (!res.ok || !res.body) throw new Error(`CodeGPT login channel failed: HTTP ${res.status}`);

	const reader = res.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	for (;;) {
		const { done, value } = await reader.read();
		if (done) throw new Error("CodeGPT login channel closed before tokens arrived");
		buffer += decoder.decode(value, { stream: true });
		let newline: number;
		while ((newline = buffer.indexOf("\n")) > -1) {
			const line = buffer.slice(0, newline).trim();
			buffer = buffer.slice(newline + 1);
			const match = line.match(/^data:\s*(.*)$/);
			if (!match || !match[1]) continue;
			let payload: any;
			try {
				payload = JSON.parse(match[1]);
			} catch {
				continue; // keep-alive / comment frame
			}
			if (!payload?.access_token) continue;
			try {
				await reader.cancel();
			} catch {
				/* stream already closing */
			}
			return {
				access: payload.access_token,
				refresh: payload.refresh_token ?? payload.access_token,
				expires: normalizeExpiry(payload.expires_at),
			};
		}
	}
}

/**
 * CodeGPT session tokens do not expire (expires_at is year 9999), so refresh is
 * an identity operation. Kept so pi's credential store has a refresh hook.
 */
async function refreshCodeGPTToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
	if (!credentials.access) throw new Error("No CodeGPT session token; run /login codegpt again.");
	return credentials;
}

// =============================================================================
// Model catalog
// =============================================================================

type CatalogModel = {
	id: string;
	name: string;
	vision: boolean;
	contextWindow: number;
	maxTokens: number;
};

let cachedModels: ProviderModelConfig[] | null = null;

/**
 * CodeGPT's Credits menu carries the tier as `badge` (Economy/Free/New!/Strong)
 * plus a `pro` flag (premium). Pi has no tier field, so collapse it to one word
 * that answers "what does this cost": Economy, Free, or Premium.
 */
function tierOf(entry: any): "Economy" | "Free" | "Premium" {
	const badge = typeof entry?.badge === "string" ? entry.badge.toLowerCase() : "";
	if (badge === "free") return "Free";
	if (badge === "economy") return "Economy";
	return entry?.pro === true ? "Premium" : "Economy";
}

/**
 * The served catalog (`/vscode/catalog`) carries the credits menu plus per-model
 * facts. Public, no auth required.
 */
async function fetchCatalogModels(signal?: AbortSignal): Promise<ProviderModelConfig[]> {
	const res = await fetch(`${backend()}/vscode/catalog`, { signal });
	if (!res.ok) throw new Error(`CodeGPT catalog → HTTP ${res.status}`);
	const feed: any = await res.json();
	const menu: any[] = Array.isArray(feed?.creditsMenu) ? feed.creditsMenu : [];
	const aliases: Record<string, string> = feed?.aliases ?? {};
	const facts: Record<string, any> = feed?.models ?? {};

	const models: ProviderModelConfig[] = menu
		.filter((entry) => entry && typeof entry.id === "string")
		.map((entry): ProviderModelConfig => {
			const identity = aliases[entry.id] ?? entry.id;
			const fact = facts[identity] ?? facts[entry.id] ?? {};
			const tier = tierOf(entry);
			return {
				// Tier lives in the id so the picker row and --list-models show it;
				// the proxy strips it before the wire (see TIER_SEPARATOR).
				id: `${entry.id}${TIER_SEPARATOR}${tier.toLowerCase()}`,
				name: typeof entry.name === "string" && entry.name ? entry.name : entry.id,
				reasoning: false,
				input: fact.vision === true ? ["text", "image"] : ["text"],
				// CodeGPT bills the serve against the signed-in account; a second,
				// per-token cost in pi's readout would be wrong.
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
				contextWindow: typeof fact.contextWindow === "number" ? fact.contextWindow : 128000,
				maxTokens: typeof fact.maxOutput === "number" ? fact.maxOutput : 32000,
			};
		});

	if (!models.length) throw new Error("CodeGPT catalog returned no models");
	cachedModels = models;
	return models;
}

async function refreshCodeGPTModels(context: { signal?: AbortSignal }): Promise<ProviderModelConfig[]> {
	try {
		return await fetchCatalogModels(context?.signal);
	} catch (error) {
		if (cachedModels) return cachedModels;
		throw error;
	}
}

// =============================================================================
// Loopback proxy: OpenAI Chat Completions -> CodeGPT /chat/tools
// =============================================================================

function readBody(req: IncomingMessage): Promise<Buffer> {
	return new Promise((resolve, reject) => {
		const chunks: Buffer[] = [];
		req.on("data", (chunk) => chunks.push(chunk as Buffer));
		req.on("end", () => resolve(Buffer.concat(chunks)));
		req.on("error", reject);
	});
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
	if (res.headersSent) {
		res.end();
		return;
	}
	res.writeHead(status, { "content-type": "application/json" });
	res.end(JSON.stringify(body));
}

async function handleCompletion(req: IncomingMessage, res: ServerResponse): Promise<void> {
	const auth = req.headers.authorization ?? "";
	if (!auth.startsWith("Bearer ")) {
		sendJson(res, 401, { error: { message: "Missing CodeGPT session token; run /login codegpt." } });
		return;
	}

	let body: any;
	try {
		body = JSON.parse((await readBody(req)).toString("utf8") || "{}");
	} catch {
		sendJson(res, 400, { error: { message: "Invalid JSON body" } });
		return;
	}
	if (!Array.isArray(body.messages)) {
		sendJson(res, 400, { error: { message: "messages is required" } });
		return;
	}

	const controller = new AbortController();
	res.on("close", () => {
		if (!res.writableEnded) controller.abort();
	});

	const wireModel = typeof body.model === "string" ? body.model.replace(TIER_SUFFIX_RE, "") : body.model;
	const codegptMessages = sanitizeMessages(body.messages);
	const promptTokens = Math.ceil(JSON.stringify(codegptMessages).length / 4);

	const upstream = await fetch(`${backend()}/chat/tools`, {
		method: "POST",
		headers: {
			authorization: auth,
			"content-type": "application/json",
			tokens: "true",
			source: "api",
			channel: CHANNEL,
			// Only needed for organization-scoped accounts (enterprise/team).
			...(process.env.CODEGPT_ORG_ID ? { "CodeGPT-Org-Id": process.env.CODEGPT_ORG_ID } : {}),
		},
		body: JSON.stringify({
			requestId: randomUUID(),
			session_id: SESSION_ID,
			agentId: BASE_AGENT_ID,
			model: wireModel,
			messages: codegptMessages,
			tools: toolsForBackend(body.tools),
			tool_choice: typeof body.tool_choice === "string" ? body.tool_choice : undefined,
			client_tools_only: true,
			stream: body.stream !== false,
			format: "json",
		}),
		signal: controller.signal,
	});

	res.writeHead(upstream.status, {
		"content-type": upstream.headers.get("content-type") ?? "text/event-stream",
		"cache-control": "no-cache",
	});
	if (!upstream.body) {
		res.end();
		return;
	}

	// CodeGPT's stream omits token usage (it bills server-side but never emits a
	// usage chunk), which leaves pi's usage.output at 0 and breaks rate/token
	// readouts. Count the streamed output and append an OpenAI usage chunk before
	// [DONE]. Token counts are approximate (chars/4); cost stays zero.
	if (body.stream === false) {
		const reader = upstream.body.getReader();
		for (;;) {
			const { done, value } = await reader.read();
			if (done) break;
			res.write(Buffer.from(value));
		}
		res.end();
		return;
	}

	const reader = upstream.body.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	let outputChars = 0;
	const writeUsage = () => {
		const completion = Math.max(1, Math.ceil(outputChars / 4));
		res.write(
			`data: ${JSON.stringify({
				id: "chatcmpl-codegpt",
				object: "chat.completion.chunk",
				created: Math.floor(Date.now() / 1000),
				model: wireModel,
				choices: [],
				usage: { prompt_tokens: promptTokens, completion_tokens: completion, total_tokens: promptTokens + completion },
			})}\n\n`,
		);
		res.write("data: [DONE]\n\n");
	};
	for (;;) {
		const { done, value } = await reader.read();
		if (done) break;
		buffer += decoder.decode(value, { stream: true });
		let newline: number;
		while ((newline = buffer.indexOf("\n")) > -1) {
			const line = buffer.slice(0, newline);
			buffer = buffer.slice(newline + 1);
			const trimmed = line.trim();
			if (trimmed === "data: [DONE]") continue;
			if (trimmed.startsWith("data: ")) {
				try {
					const delta = JSON.parse(trimmed.slice(6))?.choices?.[0]?.delta;
					if (typeof delta?.content === "string") outputChars += delta.content.length;
					if (Array.isArray(delta?.tool_calls)) {
						for (const call of delta.tool_calls) outputChars += call?.function?.arguments?.length ?? 0;
					}
				} catch {
					/* non-JSON frame */
				}
			}
			res.write(line + "\n");
		}
	}
	if (buffer) res.write(buffer);
	writeUsage();
	res.end();
}

function startProxy(): Promise<string> {
	const server = createServer((req, res) => {
		const path = (req.url ?? "").split("?")[0];
		if (req.method !== "POST" || path !== "/chat/completions") {
			sendJson(res, 404, { error: { message: "Not found" } });
			return;
		}
		handleCompletion(req, res).catch((error) => {
			sendJson(res, 502, { error: { message: error instanceof Error ? error.message : String(error) } });
		});
	});
	return new Promise((resolve, reject) => {
		server.once("error", reject);
		server.listen(0, "127.0.0.1", () => {
			const address = server.address();
			const port = typeof address === "object" && address ? address.port : 0;
			server.unref();
			resolve(`http://127.0.0.1:${port}`);
		});
	});
}

// =============================================================================
// Extension entry point
// =============================================================================

export default async function codegptPi(pi: ExtensionAPI) {
	const baseUrl = await startProxy();

	// Seed the static list in the (awaited) factory so `--model codegpt/<id>`
	// resolves on a cold start; refreshModels only updates it afterwards.
	let seed: ProviderModelConfig[] = [];
	try {
		seed = await fetchCatalogModels();
	} catch {
		seed = cachedModels ?? [];
	}

	pi.registerProvider(PROVIDER_ID, {
		name: "CodeGPT",
		baseUrl,
		api: "openai-completions",
		// Fallback for headless/CI: a session token copied from elsewhere.
		apiKey: `$${SESSION_TOKEN_ENV}`,
		authHeader: true,
		models: seed,
		refreshModels: refreshCodeGPTModels,
		oauth: {
			name: "CodeGPT",
			isSubscription: true,
			login: loginCodeGPT,
			refreshToken: refreshCodeGPTToken,
			getApiKey: (credentials) => credentials.access,
		},
	});
}

// Named exports kept for testing without loading pi.
export { fetchCatalogModels, loginCodeGPT };
