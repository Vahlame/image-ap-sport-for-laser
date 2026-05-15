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
	preprocess_mode: 'none' | 'sauvola' | 'niblack' | 'grabcut' | 'chanvese' | 'sam2';
	max_side: number;
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
	max_side: 0
};

/** Mi-configuracion persistida en localStorage para flujo Express. */
export interface MyConfig {
	material: string;
	output_mm_short: number;
	output_dpi: number;
	sharpen_radius_mm: number;
}

const MY_CONFIG_KEY = 'laser_app_my_config_v1';

export function loadMyConfig(): MyConfig {
	const fallback: MyConfig = {
		material: 'acrylic_funsun_9060_back_engrave',
		output_mm_short: 80,
		output_dpi: 115,
		sharpen_radius_mm: 0.1
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
	}
};
