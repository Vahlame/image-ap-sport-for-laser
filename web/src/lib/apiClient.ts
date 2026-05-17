/**
 * Cliente HTTP tipado al backend FastAPI (scripts/api_server.py).
 *
 * Endpoints:
 *  - GET  /api/health       liveness + estado modelo
 *  - GET  /api/materials    builtins + JSON custom
 *  - GET  /api/algorithms   agrupados por familia
 *  - POST /api/preview      procesa con max-side=400 (rapido)
 *  - POST /api/process      procesa full-res, devuelve PNG
 *
 * Base URL configurable via VITE_API_BASE_URL (default http://127.0.0.1:8000).
 */

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://127.0.0.1:18765';

export interface ProcessParams {
	preset: string;
	material: string;
	output_mm_short: number;
	output_dpi: number;
	algorithm: string;
	threshold: number;
	contrast: number;
	brightness: number;
	gamma: number;
	autocontrast: number;
	sharpen: number;
	sharpen_radius_mm: number;
	invert: boolean;
	preprocess_mode: 'none' | 'sauvola' | 'niblack' | 'clahe' | 'sauvola_clahe' | 'grabcut' | 'chanvese' | 'sam2';
	max_side: number;
	/** Si true (default), detecta zonas localmente uniformes (cielo, fondos planos) y
	 *  las clampea a blanco/negro puro antes del dither — reduce ruido moteado y ahorra
	 *  pulsos del láser. Desactivar para texturas sutiles intencionales. */
	simplify_plain_regions?: boolean;
	/** v2.0: S-curve tonal (0=sin, 0.5=suave, 1.0=agresiva). Aclara midtones, oscurece
	 *  sombras. Técnica estándar Photoshop/PhotoGrav para look fotorrealista. */
	s_curve_strength?: number;
	/** v2.0: Local contrast (Clarity) en %. Unsharp con radius grande + amount bajo.
	 *  Default 0=sin, 10-20 típico. Aumenta "punch" sin amplificar ruido como CLAHE. */
	local_contrast_amount?: number;
	/** v2.0: Auto-mirror para material *_back_engrave. Default true. Desactivar si
	 *  ya estás aplicando MirrorX en LightBurn (evita doble-mirror). */
	auto_mirror_back_engrave?: boolean;
	/** Métrica de calidad usada en HQ refine:
	 *  - 'v5' (default): no-reference, CPU rápido (FFT + tone match), recomendado para grabado láser.
	 *  - 'v4': perceptual con LPIPS+AlexNet; usa GPU si torch CUDA está disponible. Más lento.
	 */
	score_version?: 'v5' | 'v4';
}

/** Defaults seguros: el servidor aplica preset=='auto' por encima cuando se pasa. */
export const DEFAULT_PARAMS: ProcessParams = {
	preset: 'auto',
	material: '',
	output_mm_short: 0,
	output_dpi: 0,
	algorithm: 'jarvis_serpentine',
	threshold: 128,
	contrast: 1.15,
	brightness: 0.0,
	gamma: 1.0,
	autocontrast: 1.5,
	sharpen: 70.0,
	sharpen_radius_mm: 0.1,
	invert: true,
	preprocess_mode: 'sauvola',
	max_side: 0,
	score_version: 'v5',
	simplify_plain_regions: true
};

/** Mi-configuracion persistida en localStorage para flujo Express. */
export interface MyConfig {
	material: string;
	output_mm_short: number;
	output_dpi: number;
	sharpen_radius_mm: number;
	/** Métrica de calidad: 'v5' (CPU, default) o 'v4' (LPIPS GPU si disponible). */
	score_version?: 'v5' | 'v4';
}

const MY_CONFIG_KEY = 'laser_app_my_config_v1';

export function loadMyConfig(): MyConfig {
	const fallback: MyConfig = {
		material: 'acrylic_funsun_9060_back_engrave',
		output_mm_short: 80,
		output_dpi: 115,
		sharpen_radius_mm: 0.1,
		score_version: 'v5'
	};
	if (typeof localStorage === 'undefined') return fallback;
	try {
		const raw = localStorage.getItem(MY_CONFIG_KEY);
		if (!raw) return fallback;
		const parsed = JSON.parse(raw) as Partial<MyConfig>;
		return { ...fallback, ...parsed };
	} catch {
		return fallback;
	}
}

export function saveMyConfig(c: MyConfig): void {
	if (typeof localStorage === 'undefined') return;
	try {
		localStorage.setItem(MY_CONFIG_KEY, JSON.stringify(c));
	} catch {
		/* quota or disabled — silenciar */
	}
}

/** Preset legacy: alta calidad agricultor (validado experimentalmente Sesion 3).
 *  Equivalente al preset 'poster_back_engrave' del backend. */
export const PRESET_AGRICULTOR_HIGH_CONTRAST: ProcessParams = {
	...DEFAULT_PARAMS,
	preset: 'poster_back_engrave',
	algorithm: 'floyd',
	threshold: 75,
	contrast: 1.0,
	brightness: 10.0,
	gamma: 1.2,
	autocontrast: 2.0,
	sharpen: 60.0,
	invert: true,
	preprocess_mode: 'sauvola'
};

export interface RecommendedSettings {
	material: string;
	machine_compat: string;
	spot_mm: number;
	focus_mm: number;
	dpi: number;
	interval_mm: number;
	power_pct_min: number;
	power_pct_max: number;
	speed_mm_s_min: number;
	speed_mm_s_max: number;
	pass_through: boolean;
	mirror_x_required: boolean;
	lightburn_invert: boolean;
	tone_response: string;
	notes: string;
}

export interface PresetInfo {
	name: string;
	label: string;
	description: string;
	params: Record<string, unknown>;
	suggested_material: string;
}

export interface RecommendationResult {
	preset_name: string;
	preset_label: string;
	reason: string;
	stats: {
		mean: number;
		std: number;
		extreme_ratio: number;
		edge_density: number;
	};
}

export interface HealthResponse {
	status: string;
	model_loaded: boolean;
	cuda_available: boolean;
	repo_root: string;
}

export interface MaterialInfo {
	name: string;
	spot_mm: number;
	default_dpi: number;
	tone_response: 'monotonic' | 'non_monotonic' | 'linear';
	power_pct_range: [number, number];
	notes: string;
	source: 'builtin' | 'custom';
}

export interface AlgorithmGroup {
	family: 'ordered_dither' | 'error_diffusion' | 'burkes_blue_variants' | 'mix_multipass';
	algorithms: string[];
}

export interface ProcessMeta {
	width: number;
	height: number;
	whiteRatio: number;
	processTimeMs: number;
	sharpenRadiusPx: number;
	material: string;
}

/** Entrada del log estructurado del worker (t es segundos relativos al inicio del job). */
export interface JobLogLine {
	t: number;
	msg: string;
	kind: 'info' | 'warn' | 'error';
}

/** Snapshot del progreso de un job async (emitido por SSE en /api/jobs/{id}/stream). */
export interface JobProgress {
	job_id: string;
	status: 'queued' | 'running' | 'done' | 'error' | 'cancelled';
	current: number;
	total: number;
	progress_pct: number;
	best_score: number | null;
	elapsed_seconds: number;
	eta_seconds: number | null;
	stage: string;
	error_message: string | null;
	// Telemetría adicional (v1.4.1+)
	memory_mb: number | null;
	cpu_pct: number | null;
	seconds_per_candidate: number | null;
	last_candidate_seconds: number | null;
	score_history: number[];
	log_lines: JobLogLine[];
}

export interface RenderAsyncOptions {
	onJobStart?: (jobId: string) => void;
	onProgress?: (p: JobProgress) => void;
	/** Si pasás un AbortSignal y se aborta, se llama a /cancel en el servidor. */
	signal?: AbortSignal;
}

export class ApiError extends Error {
	constructor(public status: number, public detail: string) {
		super(`API ${status}: ${detail}`);
	}
}

async function fetchJson<T>(path: string): Promise<T> {
	const res = await fetch(`${API_BASE_URL}${path}`);
	if (!res.ok) {
		const detail = await res.text().catch(() => res.statusText);
		throw new ApiError(res.status, detail);
	}
	return res.json() as Promise<T>;
}

export interface SimulateMeta {
	sigmaPx: number;
	spotMm: number;
	dpi: number;
	material: string;
}

export const apiClient = {
	baseUrl: API_BASE_URL,

	async health(): Promise<HealthResponse> {
		return fetchJson<HealthResponse>('/api/health');
	},

	async materials(): Promise<MaterialInfo[]> {
		return fetchJson<MaterialInfo[]>('/api/materials');
	},

	async algorithms(): Promise<AlgorithmGroup[]> {
		return fetchJson<AlgorithmGroup[]>('/api/algorithms');
	},

	async presets(): Promise<PresetInfo[]> {
		return fetchJson<PresetInfo[]>('/api/presets');
	},

	async recommendedSettings(materialName: string): Promise<RecommendedSettings> {
		return fetchJson<RecommendedSettings>(`/api/recommended_settings/${encodeURIComponent(materialName)}`);
	},

	async recommendPreset(imageBlob: Blob): Promise<RecommendationResult> {
		const form = new FormData();
		form.append('image', imageBlob, 'input.jpg');
		const res = await fetch(`${API_BASE_URL}/api/recommend_preset`, { method: 'POST', body: form });
		if (!res.ok) {
			const detail = await res.text().catch(() => res.statusText);
			throw new ApiError(res.status, detail);
		}
		return res.json() as Promise<RecommendationResult>;
	},

	/**
	 * Simula el grabado físico a partir del PNG 1-bit (salida de /api/process).
	 * Aplica blur gaussiano del spot + respuesta tonal por material.
	 */
	async simulate(
		binaryBlob: Blob,
		opts: { material: string; outputDpi: number; appearance?: string }
	): Promise<{ blob: Blob; meta: SimulateMeta }> {
		const form = new FormData();
		form.append('image', binaryBlob, 'binary.png');
		form.append('material', opts.material);
		form.append('output_dpi', String(opts.outputDpi || 169));
		if (opts.appearance) form.append('appearance', opts.appearance);
		const res = await fetch(`${API_BASE_URL}/api/simulate`, { method: 'POST', body: form });
		if (!res.ok) {
			const detail = await res.text().catch(() => res.statusText);
			let message = detail;
			try {
				const parsed = JSON.parse(detail);
				message = parsed.detail ?? message;
			} catch {
				/* keep raw */
			}
			throw new ApiError(res.status, message);
		}
		const blob = await res.blob();
		return {
			blob,
			meta: {
				sigmaPx: parseFloat(res.headers.get('X-Sim-Sigma-Px') ?? '0'),
				spotMm: parseFloat(res.headers.get('X-Sim-Spot-Mm') ?? '0'),
				dpi: parseInt(res.headers.get('X-Sim-Dpi') ?? '0', 10),
				material: res.headers.get('X-Material') ?? ''
			}
		};
	},

	/**
	 * Procesa una imagen. Devuelve `{ blob, meta }`:
	 * - `blob`: PNG 1-bit listo para descargar/mostrar.
	 * - `meta`: parámetros del procesamiento extraídos de los headers.
	 *
	 * @param endpoint 'preview' (max_side=400) o 'process' (full-res).
	 */
	async render(
		endpoint: 'preview' | 'process',
		imageBlob: Blob,
		params: ProcessParams,
		filename = 'input.jpg'
	): Promise<{ blob: Blob; meta: ProcessMeta }> {
		const form = new FormData();
		form.append('image', imageBlob, filename);
		form.append('params_json', JSON.stringify(params));
		const res = await fetch(`${API_BASE_URL}/api/${endpoint}`, { method: 'POST', body: form });
		if (!res.ok) {
			const detail = await res.text().catch(() => res.statusText);
			let message = detail;
			try {
				const parsed = JSON.parse(detail);
				message = parsed.detail ?? message;
			} catch {
				/* keep raw */
			}
			throw new ApiError(res.status, message);
		}
		const blob = await res.blob();
		const meta: ProcessMeta = {
			width: parseInt(res.headers.get('X-Output-Width') ?? '0', 10),
			height: parseInt(res.headers.get('X-Output-Height') ?? '0', 10),
			whiteRatio: parseFloat(res.headers.get('X-White-Ratio') ?? '0'),
			processTimeMs: parseFloat(res.headers.get('X-Process-Time-Ms') ?? '0'),
			sharpenRadiusPx: parseFloat(res.headers.get('X-Sharpen-Radius-Px') ?? '0'),
			material: res.headers.get('X-Material') ?? ''
		};
		return { blob, meta };
	},

	/** Pide al backend cancelar un job en progreso. No bloquea si ya terminó. */
	async cancelJob(jobId: string): Promise<void> {
		await fetch(`${API_BASE_URL}/api/jobs/${jobId}/cancel`, { method: 'POST' }).catch(() => {
			/* el cancel es best-effort */
		});
	},

	/** Snapshot puntual del estado del job, usado como fallback si SSE falla. */
	async jobStatus(jobId: string): Promise<JobProgress> {
		return fetchJson<JobProgress>(`/api/jobs/${encodeURIComponent(jobId)}`);
	},

	/**
	 * Procesa imagen en modo async con progreso por SSE. Flujo:
	 *   1) POST /api/process_async → { job_id }
	 *   2) EventSource /api/jobs/{id}/stream → onProgress en cada update
	 *   3) Cuando status='done', GET /api/jobs/{id}/result → PNG bytes
	 *
	 * Si SSE falla (red, proxy, browser legacy), cae a polling cada 750ms.
	 * Si signal se aborta, manda /cancel y rechaza el promise.
	 */
	async renderAsync(
		imageBlob: Blob,
		params: ProcessParams | Partial<ProcessParams>,
		opts: RenderAsyncOptions = {},
		filename = 'input.jpg'
	): Promise<{ blob: Blob; meta: ProcessMeta; jobId: string }> {
		// 1. Encolar el job
		const form = new FormData();
		form.append('image', imageBlob, filename);
		form.append('params_json', JSON.stringify(params));
		const submitRes = await fetch(`${API_BASE_URL}/api/process_async`, {
			method: 'POST',
			body: form,
			signal: opts.signal
		});
		if (!submitRes.ok) {
			const detail = await submitRes.text().catch(() => submitRes.statusText);
			let message = detail;
			try {
				const parsed = JSON.parse(detail);
				message = parsed.detail ?? message;
			} catch {
				/* keep raw */
			}
			throw new ApiError(submitRes.status, message);
		}
		const { job_id: jobId } = (await submitRes.json()) as { job_id: string };
		opts.onJobStart?.(jobId);

		// 2. Esperar a que termine via SSE (con fallback a polling)
		const terminal: JobProgress['status'][] = ['done', 'error', 'cancelled'];
		const finalState = await new Promise<JobProgress>((resolve, reject) => {
			let settled = false;
			const onAbort = () => {
				if (settled) return;
				settled = true;
				// Best-effort: pedirle al server que cancele; rechazar localmente
				void apiClient.cancelJob(jobId);
				cleanup();
				reject(new ApiError(499, 'Cancelado por el cliente'));
			};

			// Polling como fallback / backup. Lo arrancamos SIEMPRE para no quedar
			// colgados si el SSE no llega (proxy bufferea, browser cierra conexión, etc.).
			let pollTimer: ReturnType<typeof setInterval> | null = null;
			let evtSource: EventSource | null = null;
			let lastStatusSeen: JobProgress['status'] | null = null;

			function cleanup() {
				if (pollTimer) clearInterval(pollTimer);
				pollTimer = null;
				if (evtSource) {
					try {
						evtSource.close();
					} catch {
						/* ignore */
					}
					evtSource = null;
				}
				if (opts.signal) opts.signal.removeEventListener('abort', onAbort);
			}

			function handleProgress(p: JobProgress) {
				if (settled) return;
				lastStatusSeen = p.status;
				opts.onProgress?.(p);
				if (terminal.includes(p.status)) {
					settled = true;
					cleanup();
					if (p.status === 'done') resolve(p);
					else if (p.status === 'cancelled')
						reject(new ApiError(499, 'Job cancelado'));
					else reject(new ApiError(500, p.error_message ?? 'error desconocido'));
				}
			}

			if (opts.signal) {
				if (opts.signal.aborted) {
					onAbort();
					return;
				}
				opts.signal.addEventListener('abort', onAbort);
			}

			// SSE
			try {
				evtSource = new EventSource(`${API_BASE_URL}/api/jobs/${jobId}/stream`);
				evtSource.onmessage = (ev) => {
					try {
						const p = JSON.parse(ev.data) as JobProgress;
						handleProgress(p);
					} catch {
						/* ignore garbage */
					}
				};
				evtSource.addEventListener('gone', () => {
					if (settled) return;
					settled = true;
					cleanup();
					reject(new ApiError(410, 'El job expiró antes de completarse'));
				});
				evtSource.onerror = () => {
					// Si ya tenemos terminal, el cleanup ya cerró todo. Si no,
					// dejamos que el polling termine de manejar.
				};
			} catch {
				/* sin SSE — polling se hace cargo */
			}

			// Polling backup cada 750 ms
			pollTimer = setInterval(() => {
				if (settled) return;
				apiClient
					.jobStatus(jobId)
					.then(handleProgress)
					.catch((err) => {
						// Si el job desapareció (404/410), abortar
						if (err instanceof ApiError && (err.status === 404 || err.status === 410)) {
							if (settled) return;
							settled = true;
							cleanup();
							reject(err);
						}
						// Otro error de red puntual: ignorar (siguiente tick reintenta)
					});
			}, 750);
		});

		// 3. Bajar el PNG final y headers
		const resultRes = await fetch(`${API_BASE_URL}/api/jobs/${jobId}/result`);
		if (!resultRes.ok) {
			const detail = await resultRes.text().catch(() => resultRes.statusText);
			let message = detail;
			try {
				const parsed = JSON.parse(detail);
				message = parsed.detail ?? message;
			} catch {
				/* keep raw */
			}
			throw new ApiError(resultRes.status, message);
		}
		const blob = await resultRes.blob();
		const meta: ProcessMeta = {
			width: parseInt(resultRes.headers.get('X-Output-Width') ?? '0', 10),
			height: parseInt(resultRes.headers.get('X-Output-Height') ?? '0', 10),
			whiteRatio: parseFloat(resultRes.headers.get('X-White-Ratio') ?? '0'),
			processTimeMs: parseFloat(resultRes.headers.get('X-Process-Time-Ms') ?? '0'),
			sharpenRadiusPx: parseFloat(resultRes.headers.get('X-Sharpen-Radius-Px') ?? '0'),
			material: resultRes.headers.get('X-Material') ?? ''
		};
		return { blob, meta, jobId };
	}
};
