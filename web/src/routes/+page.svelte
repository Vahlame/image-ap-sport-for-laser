<script lang="ts">
	import { onMount, tick } from 'svelte';
	import CropStage from '$lib/components/CropStage.svelte';
	import {
		apiClient,
		DEFAULT_PARAMS,
		PRESET_AGRICULTOR_HIGH_CONTRAST,
		loadMyConfig,
		saveMyConfig,
		type AlgorithmGroup,
		type JobProgress,
		type MaterialInfo,
		type MyConfig,
		type ProcessMeta,
		type ProcessParams,
		type RecommendedSettings,
		type SimulateMeta
	} from '$lib/apiClient';

	type Step = 1 | 2 | 3 | 4 | 5;

	// --- Estado del wizard ---
	let step = $state<Step>(1);
	let originalUrl = $state<string | null>(null);
	let croppedBlob = $state<Blob | null>(null);
	let croppedUrl = $state<string | null>(null);

	// --- Conexión backend ---
	let backendOnline = $state<boolean | null>(null);
	let cudaAvailable = $state(false);
	let materials = $state<MaterialInfo[]>([]);
	let algorithms = $state<AlgorithmGroup[]>([]);
	let connectionError = $state<string | null>(null);

	// --- Params del pipeline ---
	let params = $state<ProcessParams>({ ...PRESET_AGRICULTOR_HIGH_CONTRAST });
	let preset = $state<'agricultor' | 'manual'>('agricultor');

	// --- Preview/result ---
	let previewBlobUrl = $state<string | null>(null);
	let previewMeta = $state<ProcessMeta | null>(null);
	let previewBusy = $state(false);
	let previewError = $state<string | null>(null);

	let finalBlob = $state<Blob | null>(null);
	let finalBlobUrl = $state<string | null>(null);
	let finalMeta = $state<ProcessMeta | null>(null);
	let processingFull = $state(false);

	let simBlobUrl = $state<string | null>(null);
	let simMeta = $state<SimulateMeta | null>(null);
	let simBusy = $state(false);
	let simError = $state<string | null>(null);

	let comparePos = $state(50);

	// --- Mi configuracion persistida (Express mode) ---
	let myConfig = $state<MyConfig>(loadMyConfig());
	let showMyConfig = $state(false);
	let expressMode = $state(true);  // default ON: subir foto → procesa automatico
	let expressProgressMsg = $state<string>('');
	let expressError = $state<string | null>(null);

	// --- Recomendaciones LightBurn (cargadas tras procesar) ---
	let recommendedSettings = $state<RecommendedSettings | null>(null);

	const steps = [
		{ n: 1 as const, label: 'Subir' },
		{ n: 2 as const, label: 'Recortar' },
		{ n: 3 as const, label: 'Ajustes' },
		{ n: 4 as const, label: 'Resultado' },
		{ n: 5 as const, label: 'Descargar' }
	];

	function revoke(url: string | null) {
		if (url) URL.revokeObjectURL(url);
	}

	function formatSeconds(s: number): string {
		if (!Number.isFinite(s) || s < 0) return '–';
		if (s < 60) return `${s.toFixed(0)}s`;
		const m = Math.floor(s / 60);
		const r = Math.round(s - m * 60);
		return `${m}m ${r.toString().padStart(2, '0')}s`;
	}

	// --- Bootstrap: conectar al backend ---
	onMount(async () => {
		try {
			const health = await apiClient.health();
			backendOnline = health.status === 'ok';
			cudaAvailable = health.cuda_available;
			[materials, algorithms] = await Promise.all([
				apiClient.materials(),
				apiClient.algorithms()
			]);
			// Aplicar mi config persistida al material del pipeline si está disponible
			if (myConfig.material && materials.some((m) => m.name === myConfig.material)) {
				params.material = myConfig.material;
			} else if (!params.material && materials.some((m) => m.name === 'acrylic_funsun_9060_back_engrave')) {
				params.material = 'acrylic_funsun_9060_back_engrave';
				myConfig.material = 'acrylic_funsun_9060_back_engrave';
				saveMyConfig(myConfig);
			}
		} catch (err) {
			backendOnline = false;
			connectionError = err instanceof Error ? err.message : String(err);
		}
	});

	function persistMyConfig() {
		saveMyConfig(myConfig);
	}

	async function loadRecommendedSettings(materialName: string) {
		if (!materialName) {
			recommendedSettings = null;
			return;
		}
		try {
			recommendedSettings = await apiClient.recommendedSettings(materialName);
		} catch (err) {
			console.error('No se pudo cargar recommended_settings:', err);
			recommendedSettings = null;
		}
	}

	// --- Progreso async (SSE) ---
	let expressJobId = $state<string | null>(null);
	let expressProgress = $state<JobProgress | null>(null);
	let expressCancelled = $state(false);
	let showProgressLog = $state(false);

	/** Construye el d="M..." de un sparkline SVG dado un arreglo de scores (menor=mejor). */
	function buildSparklinePath(values: number[], width = 220, height = 36): string {
		if (!values.length) return '';
		const n = values.length;
		const min = Math.min(...values);
		const max = Math.max(...values);
		const range = max - min || 1;
		const dx = n > 1 ? width / (n - 1) : 0;
		const pts = values.map((v, i) => {
			const x = i * dx;
			// Invertimos Y para que menor score (mejor) quede arriba
			const y = height - ((v - min) / range) * (height - 4) - 2;
			return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
		});
		return pts.join(' ');
	}

	function resetExpressProgress() {
		expressProgress = null;
		expressJobId = null;
		expressCancelled = false;
	}

	async function cancelExpressJob() {
		if (!expressJobId) return;
		expressCancelled = true;
		try {
			await apiClient.cancelJob(expressJobId);
		} catch (err) {
			console.warn('cancel job:', err);
		}
	}

	/** Modo Express: subir foto → procesa con MyConfig + preset=auto + HQ → result inmediato.
	 *  IMPORTANT: para que preset='auto' aplique completo, solo mandamos los campos
	 *  que el usuario controla (material/mm/dpi/sharpen). Los demás los completa el
	 *  backend con su default y luego el merger los reemplaza con los del preset elegido.
	 */
	async function expressProcess(file: File | Blob) {
		expressError = null;
		resetExpressProgress();
		expressProgressMsg = 'Subiendo imagen…';

		const imageBlob = file;
		revoke(croppedUrl);
		revoke(finalBlobUrl);
		revoke(simBlobUrl);
		croppedBlob = imageBlob;
		croppedUrl = URL.createObjectURL(imageBlob);
		finalBlob = null;
		finalBlobUrl = null;
		simBlobUrl = null;
		simMeta = null;

		// Params mínimos: el preset auto + las decisiones físicas del usuario.
		// NO spread DEFAULT_PARAMS — eso enviaría algorithm/threshold/etc. con valores
		// distintos al schema default y rompería el auto-merge del preset en el backend.
		const expressParams: Partial<ProcessParams> = {
			preset: 'auto',
			material: myConfig.material,
			output_mm_short: myConfig.output_mm_short,
			output_dpi: myConfig.output_dpi,
			sharpen_radius_mm: myConfig.sharpen_radius_mm,
			score_version: myConfig.score_version ?? 'v5'
		};
		// Reflejar también en `params` para que el wizard manual pueda continuar
		// si el usuario salta a modo manual desde el resultado.
		params = { ...DEFAULT_PARAMS, ...expressParams } as ProcessParams;

		try {
			processingFull = true;
			expressProgressMsg = 'Procesando con HQ refinement…';

			const { blob, meta } = await apiClient.renderAsync(imageBlob, expressParams, {
				onJobStart: (id) => {
					expressJobId = id;
				},
				onProgress: (p) => {
					expressProgress = p;
				}
			});

			revoke(finalBlobUrl);
			finalBlob = blob;
			finalBlobUrl = URL.createObjectURL(blob);
			finalMeta = meta;
			expressProgressMsg = '';
			resetExpressProgress();

			await loadRecommendedSettings(myConfig.material);
			await tick();
			step = 4;
		} catch (err) {
			if (expressCancelled) {
				expressProgressMsg = '';
				expressError = 'Procesamiento cancelado por el usuario.';
			} else {
				expressError = err instanceof Error ? err.message : String(err);
				expressProgressMsg = '';
			}
			resetExpressProgress();
		} finally {
			processingFull = false;
		}
	}

	function onExpressDropzoneChange(e: Event) {
		const input = e.currentTarget as HTMLInputElement;
		const f = input.files?.[0];
		// v2.1: reset input.value PRIMERO (no al final). Esto permite re-subir
		// la MISMA imagen después de un error o cancelación (HTML spec: si value
		// no cambia, no se dispara onchange, así que hay que limpiarlo antes).
		input.value = '';
		if (!f) return;
		if (!/^image\//.test(f.type)) {
			expressError = 'Formato no soportado. Usa PNG, JPG o WebP.';
			return;
		}
		// Validación temprana de tamaño (frontend) para feedback rápido
		if (f.size > 100 * 1024 * 1024) {
			expressError = `Imagen muy grande (${Math.round(f.size / 1024 / 1024)} MB). Máximo: 100 MB.`;
			return;
		}
		revoke(originalUrl);
		originalUrl = URL.createObjectURL(f);
		void expressProcess(f);
	}

	function applyPreset(name: 'agricultor' | 'manual') {
		preset = name;
		if (name === 'agricultor') {
			params = { ...PRESET_AGRICULTOR_HIGH_CONTRAST, material: params.material };
		}
		// manual no resetea, deja al usuario tunear
	}

	// --- Flow ---
	function onDropzoneChange(e: Event) {
		const input = e.currentTarget as HTMLInputElement;
		const f = input.files?.[0];
		// v2.1: reset input.value PRIMERO para permitir re-subir misma imagen
		input.value = '';
		if (!f) return;
		if (!/^image\//.test(f.type)) {
			alert('Formato no soportado. Usa PNG, JPG o WebP.');
			return;
		}
		if (f.size > 100 * 1024 * 1024) {
			alert(`Imagen muy grande (${Math.round(f.size / 1024 / 1024)} MB). Máximo: 100 MB.`);
			return;
		}
		revoke(originalUrl);
		revoke(croppedUrl);
		revoke(previewBlobUrl);
		revoke(finalBlobUrl);
		croppedBlob = null;
		croppedUrl = null;
		previewBlobUrl = null;
		previewMeta = null;
		finalBlob = null;
		finalBlobUrl = null;
		finalMeta = null;
		originalUrl = URL.createObjectURL(f);
		step = 2;
	}

	function onCropped(blob: Blob) {
		revoke(croppedUrl);
		revoke(previewBlobUrl);
		revoke(finalBlobUrl);
		previewBlobUrl = null;
		previewMeta = null;
		finalBlob = null;
		finalBlobUrl = null;
		finalMeta = null;
		croppedBlob = blob;
		croppedUrl = URL.createObjectURL(blob);
		step = 3;
	}

	// --- Preview con debounce (cuando cambian params en step 3) ---
	let previewTimer: ReturnType<typeof setTimeout> | null = null;
	$effect(() => {
		// dependencias: croppedBlob + cualquiera de params clave
		const _ = [
			croppedBlob, params.algorithm, params.threshold, params.contrast, params.brightness,
			params.gamma, params.autocontrast, params.sharpen, params.invert,
			params.preprocess_mode, params.material, step
		];
		if (!croppedBlob || step < 3 || !backendOnline) return;
		if (previewTimer) clearTimeout(previewTimer);
		previewTimer = setTimeout(runPreview, 350);
		return () => {
			if (previewTimer) clearTimeout(previewTimer);
		};
	});

	async function runPreview() {
		if (!croppedBlob) return;
		previewBusy = true;
		previewError = null;
		try {
			const { blob, meta } = await apiClient.render('preview', croppedBlob, params);
			revoke(previewBlobUrl);
			previewBlobUrl = URL.createObjectURL(blob);
			previewMeta = meta;
		} catch (err) {
			previewError = err instanceof Error ? err.message : String(err);
		} finally {
			previewBusy = false;
		}
	}

	async function processFullRes() {
		if (!croppedBlob) return;
		processingFull = true;
		previewError = null;
		resetExpressProgress();
		// Propagar la elección de métrica desde MyConfig hacia los params del Manual.
		const paramsWithMetric = { ...params, score_version: myConfig.score_version ?? params.score_version ?? 'v5' };
		try {
			const { blob, meta } = await apiClient.renderAsync(croppedBlob, paramsWithMetric, {
				onJobStart: (id) => {
					expressJobId = id;
				},
				onProgress: (p) => {
					expressProgress = p;
				}
			});
			revoke(finalBlobUrl);
			finalBlob = blob;
			finalBlobUrl = URL.createObjectURL(blob);
			finalMeta = meta;
			if (simBlobUrl) URL.revokeObjectURL(simBlobUrl);
			simBlobUrl = null;
			simMeta = null;
			simError = null;
			resetExpressProgress();
			await tick();
			step = 4;
		} catch (err) {
			if (expressCancelled) {
				previewError = 'Procesamiento cancelado por el usuario.';
			} else {
				previewError = err instanceof Error ? err.message : String(err);
			}
			resetExpressProgress();
		} finally {
			processingFull = false;
		}
	}

	async function runSimulation(appearanceOverride?: string) {
		if (!finalBlob) return;
		simBusy = true;
		simError = null;
		try {
			const { blob, meta } = await apiClient.simulate(finalBlob, {
				material: params.material,
				outputDpi: params.output_dpi > 0 ? params.output_dpi : selectedMaterial?.default_dpi ?? 169,
				appearance: appearanceOverride
			});
			if (simBlobUrl) URL.revokeObjectURL(simBlobUrl);
			simBlobUrl = URL.createObjectURL(blob);
			simMeta = meta;
		} catch (err) {
			simError = err instanceof Error ? err.message : String(err);
		} finally {
			simBusy = false;
		}
	}

	function downloadFinal() {
		// Bug fix v2.1: hay que (a) usar el finalBlob directo, no finalBlobUrl
		// (que puede estar siendo usado por <img> ya), (b) appendChild al DOM antes del
		// click (Firefox/Safari bloquean clicks en elementos no-DOM), (c) revoke después
		// para liberar memoria.
		if (!finalBlob) {
			console.warn('[downloadFinal] finalBlob es null/undefined');
			return;
		}
		const matLabel = (params.material || 'generic').replace(/[^a-z0-9_-]/gi, '_');
		const algoLabel = (params.algorithm || 'auto').replace(/[^a-z0-9_-]/gi, '_');
		const dimsLabel =
			params.output_mm_short > 0 ? `_${params.output_mm_short}mm_${params.output_dpi}dpi` : '';
		const filename = `laser_ready_${matLabel}_${algoLabel}${dimsLabel}.png`;

		const url = URL.createObjectURL(finalBlob);
		const a = document.createElement('a');
		a.href = url;
		a.download = filename;
		a.style.display = 'none';
		a.rel = 'noopener';
		document.body.appendChild(a);
		try {
			a.click();
		} finally {
			// setTimeout para que el browser termine de procesar el click antes de cleanup
			setTimeout(() => {
				if (a.parentNode) a.parentNode.removeChild(a);
				URL.revokeObjectURL(url);
			}, 100);
		}
	}

	function backToUpload() {
		// v2.1 fix: reset COMPLETO. Antes podía dejar processingFull=true si había
		// race con un error async, bloqueando futuros uploads. También cancela job
		// pendiente si hay uno.
		if (processingFull && expressJobId) {
			void cancelExpressJob();  // best-effort cancel
		}
		revoke(originalUrl);
		revoke(croppedUrl);
		revoke(previewBlobUrl);
		revoke(finalBlobUrl);
		revoke(simBlobUrl);
		originalUrl = null;
		croppedBlob = null;
		croppedUrl = null;
		previewBlobUrl = null;
		previewMeta = null;
		previewError = null;
		finalBlob = null;
		finalBlobUrl = null;
		finalMeta = null;
		simBlobUrl = null;
		simMeta = null;
		simError = null;
		processingFull = false;
		expressProgressMsg = '';
		expressError = null;
		resetExpressProgress();
		step = 1;
	}

	// Materiales: spot maximo DPI util
	const selectedMaterial = $derived(materials.find((m) => m.name === params.material));
	const dpiWarning = $derived.by(() => {
		if (!selectedMaterial || params.output_dpi <= 0) return null;
		const maxDpi = Math.round(25.4 / selectedMaterial.spot_mm);
		if (params.output_dpi > maxDpi) {
			return `DPI ${params.output_dpi} excede el límite físico para spot ${selectedMaterial.spot_mm.toFixed(2)} mm (máximo recomendado: ${maxDpi}).`;
		}
		return null;
	});
</script>

<div class="app-shell">
	<header class="topbar">
		<div class="brand">
			<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
				<circle cx="12" cy="12" r="3" />
				<path d="M12 2v3M12 19v3M2 12h3M19 12h3M5.2 5.2l2.1 2.1M16.7 16.7l2.1 2.1M5.2 18.8l2.1-2.1M16.7 7.3l2.1-2.1" />
			</svg>
			<span>Image AP — Laser Prep</span>
		</div>
		<div class="status">
			{#if backendOnline === null}
				<span class="dot pending"></span> Conectando…
			{:else if backendOnline}
				<span class="dot ok"></span>
				<span>API <code>{apiClient.baseUrl}</code></span>
				{#if cudaAvailable}<span class="badge cuda">CUDA</span>{:else}<span class="badge cpu">CPU</span>{/if}
			{:else}
				<span class="dot err"></span> Backend offline
				<span class="hint-error">{connectionError}</span>
			{/if}
		</div>
	</header>

	<nav class="steps" aria-label="Pasos del asistente">
		{#each steps as s (s.n)}
			<button
				type="button"
				class="step-pill"
				class:active={step === s.n}
				class:done={step > s.n}
				disabled={s.n > step + 1 || (s.n === 2 && !originalUrl) || (s.n >= 3 && !croppedBlob) || (s.n >= 4 && !finalBlob)}
				onclick={() => {
					if (s.n <= step || (s.n === step + 1 && (s.n === 2 ? originalUrl : croppedBlob))) {
						step = s.n;
					}
				}}>
				<span class="step-n">{s.n}</span>
				<span class="step-l">{s.label}</span>
			</button>
		{/each}
	</nav>

	<main>
		{#if !backendOnline && backendOnline !== null}
			<div class="panel panel-warn">
				<h2>Backend no disponible</h2>
				<p>Levantá la API con:</p>
				<pre><code>uvicorn scripts.api_server:app --host 127.0.0.1 --port 8000</code></pre>
				<p class="hint">Por defecto se espera en <code>http://127.0.0.1:8000</code>. Editá <code>VITE_API_BASE_URL</code> si lo movés.</p>
				{#if connectionError}
					<p class="hint hint-error">Detalle: <code>{connectionError}</code></p>
				{/if}
			</div>
		{:else if step === 1}
			<div class="panel">
				<div class="mode-toggle">
					<button class:active={expressMode} onclick={() => (expressMode = true)}>
						⚡ Modo Express
						<small>una foto → resultado, usa tu configuración</small>
					</button>
					<button class:active={!expressMode} onclick={() => (expressMode = false)}>
						🛠 Modo Manual
						<small>recortás, ajustás sliders, previews en vivo</small>
					</button>
				</div>

				<details class="my-config-panel" open={!myConfig.material}>
					<summary>
						<span>⚙ Mi configuración</span>
						<code class="my-config-summary">
							{myConfig.material || 'sin material'} · {myConfig.output_mm_short}mm @ {myConfig.output_dpi}DPI · {(myConfig.score_version ?? 'v5').toUpperCase()}
						</code>
					</summary>
					<p class="hint">
						Estos valores se guardan en tu navegador. Se usan en modo Express y como defaults en Manual.
					</p>
					<div class="grid2">
						<div class="field">
							<label for="mc-mat">Material por defecto</label>
							<select id="mc-mat" bind:value={myConfig.material} onchange={persistMyConfig}>
								<option value="">— elegir —</option>
								{#each materials as m (m.name)}
									<option value={m.name}>{m.name}</option>
								{/each}
							</select>
						</div>
						<div class="field">
							<label for="mc-mm">Lado corto (mm)</label>
							<input id="mc-mm" type="number" step="0.5" min="5" bind:value={myConfig.output_mm_short} onchange={persistMyConfig} />
						</div>
						<div class="field">
							<label for="mc-dpi">DPI</label>
							<input id="mc-dpi" type="number" step="1" min="50" max="600" bind:value={myConfig.output_dpi} onchange={persistMyConfig} />
						</div>
						<div class="field">
							<label for="mc-sr">Sharpen radius (mm)</label>
							<input id="mc-sr" type="number" step="0.01" min="0.05" max="2" bind:value={myConfig.sharpen_radius_mm} onchange={persistMyConfig} />
						</div>
					</div>

					<!-- Toggle métrica de calidad (v5 CPU / v4 GPU) -->
					<div class="metric-toggle">
						<div class="metric-toggle-head">
							<span class="metric-toggle-label">Métrica de calidad (HQ refine)</span>
							<span class="metric-toggle-badge" class:active={cudaAvailable}>
								{cudaAvailable ? '⚡ GPU disponible' : '🖥 CPU only'}
							</span>
						</div>
						<div class="metric-toggle-options">
							<label class="metric-opt" class:selected={(myConfig.score_version ?? 'v5') === 'v5'}>
								<input
									type="radio"
									name="score_version"
									value="v5"
									checked={(myConfig.score_version ?? 'v5') === 'v5'}
									onchange={() => { myConfig.score_version = 'v5'; persistMyConfig(); }} />
								<div class="metric-opt-body">
									<strong>Estándar — v5 (CPU, no-reference)</strong>
									<span class="metric-opt-desc">
										HVS-MSE + spectral blue-noise + tone match post-LUT. ~5-15 ms / candidato.
										Diseñada específicamente para grabado láser. <em>Recomendada.</em>
									</span>
								</div>
							</label>
							<label class="metric-opt" class:selected={myConfig.score_version === 'v4'}>
								<input
									type="radio"
									name="score_version"
									value="v4"
									checked={myConfig.score_version === 'v4'}
									onchange={() => { myConfig.score_version = 'v4'; persistMyConfig(); }} />
								<div class="metric-opt-body">
									<strong>Perceptual — v4 (LPIPS + AlexNet)</strong>
									<span class="metric-opt-desc">
										{#if cudaAvailable}
											Comparación perceptual contra auto-target. ~50-200 ms / candidato (acelera con GPU).
											Útil para A/B contra v5 si querés contrastar resultados.
										{:else}
											<span class="warn-inline">⚠ Tu PyTorch es <code>+cpu</code> — v4 correrá en CPU (lento ~5-10×).</span>
											Para acelerar reinstalá con:
											<code class="snippet">pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121</code>
										{/if}
									</span>
								</div>
							</label>
						</div>
					</div>
				</details>

				{#if expressMode}
					<h2 class="title">Modo Express</h2>
					<p class="lede">
						Subí una foto y la app la procesa automáticamente con tu configuración guardada
						(material <code>{myConfig.material || '—'}</code>, {myConfig.output_mm_short}mm @ {myConfig.output_dpi}DPI)
						+ auto-detección de preset + HQ refinement. Tarda ~2–3 minutos por imagen full-res.
					</p>
					<label class="dropzone dropzone-express">
						<input type="file" accept="image/png,image/jpeg,image/webp"
						       onchange={onExpressDropzoneChange}
						       disabled={processingFull || !myConfig.material} />
						<span class="dropzone-icon" aria-hidden="true">
							<svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
								<path d="M13 2L3 14h7l-1 8 10-12h-7l1-8z" />
							</svg>
						</span>
						<span class="dropzone-title">{processingFull ? 'Procesando…' : 'Arrastrá tu foto'}</span>
						<span class="dropzone-sub">
							{#if !myConfig.material}
								Primero seleccioná un material en "Mi configuración" ↑
							{:else}
								PNG · JPG · WebP — procesado automático con HQ refinement
							{/if}
						</span>
					</label>
					{#if processingFull || expressProgressMsg}
						<div class="progress-card">
							<div class="progress-head">
								<span class="progress-msg">
									⏳ {expressProgressMsg || (expressProgress?.stage === 'refine'
										? `Buscando mejor candidato HQ (${expressProgress.current}/${expressProgress.total})`
										: expressProgress?.stage
											? `Etapa: ${expressProgress.stage}`
											: 'Procesando…')}
								</span>
								<button type="button" class="btn btn-ghost btn-cancel" onclick={cancelExpressJob} disabled={expressCancelled}>
									{expressCancelled ? 'Cancelando…' : 'Cancelar'}
								</button>
							</div>
							{#if expressProgress}
								<div class="progress-bar" role="progressbar" aria-valuenow={expressProgress.progress_pct} aria-valuemin="0" aria-valuemax="100">
									<div class="progress-fill" style:width={`${Math.min(100, Math.max(0, expressProgress.progress_pct))}%`}></div>
								</div>
								<div class="progress-stats">
									<span><strong>{expressProgress.progress_pct.toFixed(0)}%</strong></span>
									<span>•</span>
									<span>{expressProgress.current}/{expressProgress.total}</span>
									<span>•</span>
									<span>transcurrido {formatSeconds(expressProgress.elapsed_seconds)}</span>
									{#if expressProgress.eta_seconds !== null && expressProgress.eta_seconds > 0}
										<span>•</span>
										<span>ETA {formatSeconds(expressProgress.eta_seconds)}</span>
									{/if}
								</div>

								<!-- Telemetría extendida -->
								<div class="telemetry-grid">
									{#if expressProgress.best_score !== null}
										<div class="tm-cell">
											<span class="tm-label">Mejor score (menor=mejor)</span>
											<span class="tm-val">{expressProgress.best_score.toFixed(4)}</span>
										</div>
									{/if}
									{#if expressProgress.seconds_per_candidate !== null}
										<div class="tm-cell">
											<span class="tm-label">Avg / candidato</span>
											<span class="tm-val">{expressProgress.seconds_per_candidate.toFixed(2)}s</span>
										</div>
									{/if}
									{#if expressProgress.last_candidate_seconds !== null}
										<div class="tm-cell">
											<span class="tm-label">Último candidato</span>
											<span class="tm-val">{expressProgress.last_candidate_seconds.toFixed(2)}s</span>
										</div>
									{/if}
									{#if expressProgress.memory_mb !== null}
										<div class="tm-cell">
											<span class="tm-label">RAM (RSS)</span>
											<span class="tm-val">{expressProgress.memory_mb.toFixed(0)} MB</span>
										</div>
									{/if}
									{#if expressProgress.cpu_pct !== null && expressProgress.cpu_pct > 0}
										<div class="tm-cell">
											<span class="tm-label">CPU</span>
											<span class="tm-val">{expressProgress.cpu_pct.toFixed(0)}%</span>
										</div>
									{/if}
									<div class="tm-cell">
										<span class="tm-label">Etapa</span>
										<span class="tm-val">{expressProgress.stage || '—'}</span>
									</div>
								</div>

								<!-- Sparkline evolución del best score -->
								{#if expressProgress.score_history.length > 1}
									<div class="sparkline-wrap">
										<div class="sparkline-head">
											<span class="tm-label">Evolución del mejor score</span>
											<span class="sparkline-range">
												{Math.min(...expressProgress.score_history).toFixed(4)}
												→
												{Math.max(...expressProgress.score_history).toFixed(4)}
											</span>
										</div>
										<svg class="sparkline" viewBox="0 0 220 36" preserveAspectRatio="none">
											<path
												d={buildSparklinePath(expressProgress.score_history)}
												fill="none"
												stroke="var(--accent)"
												stroke-width="1.8"
												stroke-linejoin="round"
												stroke-linecap="round" />
										</svg>
									</div>
								{/if}

								<!-- Log scrollable -->
								<details class="log-details" bind:open={showProgressLog}>
									<summary>
										Log del worker ({expressProgress.log_lines.length})
									</summary>
									<div class="log-scroll">
										{#each expressProgress.log_lines.slice().reverse() as ln, i (i)}
											<div class="log-line log-{ln.kind}">
												<span class="log-t">{ln.t.toFixed(2)}s</span>
												<span class="log-msg">{ln.msg}</span>
											</div>
										{/each}
									</div>
								</details>
							{:else}
								<div class="progress-bar"><div class="progress-fill indeterminate"></div></div>
							{/if}
						</div>
					{/if}
					{#if expressError}
						<div class="error-card">
							<p class="warn">⚠ {expressError}</p>
							<div class="actions">
								<!-- v2.1: botón retry para reintentar sin tener que volver a subir -->
								{#if croppedBlob}
									<button type="button" class="btn btn-secondary" onclick={() => { expressError = null; void expressProcess(croppedBlob!); }}>
										🔄 Reintentar con la misma imagen
									</button>
								{/if}
								<button type="button" class="btn btn-ghost" onclick={() => { expressError = null; }}>
									Cerrar mensaje
								</button>
							</div>
						</div>
					{/if}
				{:else}
					<h2 class="title">Modo Manual</h2>
					<p class="lede">
						Subí una foto nítida. En el siguiente paso elegís el recorte y después tunear con sliders.
					</p>
					<label class="dropzone">
						<input type="file" accept="image/png,image/jpeg,image/webp" onchange={onDropzoneChange} />
						<span class="dropzone-icon" aria-hidden="true">
							<svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
								<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
								<polyline points="17 8 12 3 7 8" />
								<line x1="12" y1="3" x2="12" y2="15" />
							</svg>
						</span>
						<span class="dropzone-title">Arrastrá o hacé clic</span>
						<span class="dropzone-sub">PNG · JPG · WebP — hasta ~8000×8000 px</span>
					</label>
					<p class="hint">
						Tip: subí el archivo de mayor calidad/resolución que tengas. El recorte y resize al tamaño físico mm×DPI lo hace el pipeline.
					</p>
				{/if}
			</div>
		{:else if step === 2 && originalUrl}
			{#key originalUrl}
				<CropStage src={originalUrl} onBack={backToUpload} onContinue={onCropped} />
			{/key}
		{:else if step === 3 && croppedUrl}
			<div class="step3-layout">
				<aside class="panel panel-controls">
					<h2 class="title">Ajustes</h2>

					<div class="preset-toggle">
						<button class:active={preset === 'agricultor'} onclick={() => applyPreset('agricultor')}>
							Preset óptimo
						</button>
						<button class:active={preset === 'manual'} onclick={() => applyPreset('manual')}>
							Manual
						</button>
					</div>

					<details class="group" open>
						<summary>Material y output físico</summary>
						<div class="field">
							<label for="mat">Material</label>
							<select id="mat" bind:value={params.material}>
								<option value="">— sin LUT —</option>
								{#each materials as m (m.name)}
									<option value={m.name}>{m.name} (spot {m.spot_mm.toFixed(2)} mm)</option>
								{/each}
							</select>
							{#if selectedMaterial}
								<p class="hint">
									{selectedMaterial.notes || `Respuesta tonal: ${selectedMaterial.tone_response}`}
								</p>
							{/if}
						</div>
						<div class="grid2">
							<div class="field">
								<label for="mmShort">Lado corto (mm)</label>
								<input id="mmShort" type="number" step="0.1" min="0" bind:value={params.output_mm_short} />
							</div>
							<div class="field">
								<label for="dpi">DPI grabado</label>
								<input id="dpi" type="number" step="1" min="0" max="2400" bind:value={params.output_dpi} />
							</div>
						</div>
						<div class="field">
							<label for="sharpenRadius">Sharpen radius físico (mm)</label>
							<input id="sharpenRadius" type="number" step="0.01" min="0.01" max="2" bind:value={params.sharpen_radius_mm} />
							<p class="hint">El radius USM real se escala al output. Default 0.10 ≈ ½ spot.</p>
						</div>
						{#if dpiWarning}
							<p class="warn">⚠ {dpiWarning}</p>
						{/if}
					</details>

					<details class="group" open={preset === 'manual'}>
						<summary>Algoritmo y tono</summary>
						<div class="field">
							<label for="algo">Algoritmo</label>
							<select id="algo" bind:value={params.algorithm}>
								{#each algorithms as g (g.family)}
									<optgroup label={g.family.replace('_', ' ')}>
										{#each g.algorithms as a (a)}
											<option value={a}>{a}</option>
										{/each}
									</optgroup>
								{/each}
							</select>
						</div>
						<div class="field">
							<label for="prep">Pre-procesado</label>
							<select id="prep" bind:value={params.preprocess_mode}>
								<option value="none">Ninguno</option>
								<option value="sauvola">Sauvola (contraste local)</option>
								<option value="niblack">Niblack (umbral local)</option>
								<option value="clahe">CLAHE (revela detalles en zonas claras)</option>
								<option value="sauvola_clahe">CLAHE + Sauvola (combo agresivo)</option>
							</select>
							{#if params.preprocess_mode === 'clahe' || params.preprocess_mode === 'sauvola_clahe'}
								<p class="hint">
									CLAHE redistribuye contraste local: ideal cuando hay detalles (logos, letras chicas)
									en zonas casi blancas o casi negras que el dither pierde.
								</p>
							{/if}
						</div>
						<div class="field-row">
							<label class="check"><input type="checkbox" bind:checked={params.invert} /> Invertir</label>
							<label class="check" title="Convierte zonas localmente uniformes (cielo, fondos planos) a blanco/negro puro antes del dither. Reduce ruido moteado y pulsos del láser.">
								<input type="checkbox" bind:checked={params.simplify_plain_regions} /> Simplificar fondos planos
							</label>
						</div>
						<div class="slider">
							<label for="thr">Umbral <code>{params.threshold}</code></label>
							<input id="thr" type="range" min="1" max="254" bind:value={params.threshold} />
						</div>
						<div class="slider">
							<label for="ct">Contraste <code>{params.contrast.toFixed(2)}</code></label>
							<input id="ct" type="range" min="0.3" max="2.5" step="0.05" bind:value={params.contrast} />
						</div>
						<div class="slider">
							<label for="br">Brillo <code>{params.brightness > 0 ? '+' : ''}{params.brightness.toFixed(0)}</code></label>
							<input id="br" type="range" min="-60" max="60" step="1" bind:value={params.brightness} />
						</div>
						<div class="slider">
							<label for="gm">Gamma <code>{params.gamma.toFixed(2)}</code></label>
							<input id="gm" type="range" min="0.4" max="2.5" step="0.05" bind:value={params.gamma} />
						</div>
						<div class="slider">
							<label for="ac">Autocontraste <code>{params.autocontrast.toFixed(1)}</code></label>
							<input id="ac" type="range" min="0" max="5" step="0.1" bind:value={params.autocontrast} />
						</div>
						<div class="slider">
							<label for="sh">Sharpen % <code>{params.sharpen.toFixed(0)}</code></label>
							<input id="sh" type="range" min="0" max="150" step="5" bind:value={params.sharpen} />
						</div>
					</details>

					{#if processingFull}
						<div class="progress-card progress-card-compact">
							<div class="progress-head">
								<span class="progress-msg">
									⏳ {expressProgress?.stage === 'refine'
										? `HQ (${expressProgress.current}/${expressProgress.total})`
										: expressProgress?.stage
											? `Etapa: ${expressProgress.stage}`
											: 'Procesando…'}
								</span>
								<button type="button" class="btn btn-ghost btn-cancel" onclick={cancelExpressJob} disabled={expressCancelled}>
									{expressCancelled ? 'Cancelando…' : 'Cancelar'}
								</button>
							</div>
							{#if expressProgress}
								<div class="progress-bar" role="progressbar" aria-valuenow={expressProgress.progress_pct} aria-valuemin="0" aria-valuemax="100">
									<div class="progress-fill" style:width={`${Math.min(100, Math.max(0, expressProgress.progress_pct))}%`}></div>
								</div>
								<div class="progress-stats">
									<span><strong>{expressProgress.progress_pct.toFixed(0)}%</strong></span>
									<span>• {formatSeconds(expressProgress.elapsed_seconds)}</span>
									{#if expressProgress.eta_seconds !== null && expressProgress.eta_seconds > 0}
										<span>• ETA {formatSeconds(expressProgress.eta_seconds)}</span>
									{/if}
									{#if expressProgress.memory_mb !== null}
										<span>• {expressProgress.memory_mb.toFixed(0)} MB</span>
									{/if}
									{#if expressProgress.cpu_pct !== null && expressProgress.cpu_pct > 0}
										<span>• CPU {expressProgress.cpu_pct.toFixed(0)}%</span>
									{/if}
									{#if expressProgress.best_score !== null}
										<span>• score <code>{expressProgress.best_score.toFixed(4)}</code></span>
									{/if}
								</div>
								{#if expressProgress.score_history.length > 2}
									<svg class="sparkline sparkline-compact" viewBox="0 0 220 24" preserveAspectRatio="none">
										<path
											d={buildSparklinePath(expressProgress.score_history, 220, 24)}
											fill="none"
											stroke="var(--accent)"
											stroke-width="1.6"
											stroke-linejoin="round"
											stroke-linecap="round" />
									</svg>
								{/if}
							{:else}
								<div class="progress-bar"><div class="progress-fill indeterminate"></div></div>
							{/if}
						</div>
					{/if}
					<div class="actions">
						<button type="button" class="btn btn-secondary" onclick={() => (step = 2)}>← Recorte</button>
						<button type="button" class="btn btn-primary" disabled={processingFull || !backendOnline} onclick={processFullRes}>
							{#if processingFull}Procesando…{:else}Procesar full-res →{/if}
						</button>
					</div>
				</aside>

				<section class="panel panel-preview">
					<header class="preview-head">
						<h2 class="title">Preview</h2>
						{#if previewBusy}
							<span class="badge busy">Actualizando…</span>
						{:else if previewMeta}
							<span class="meta-line">
								<code>{previewMeta.width}×{previewMeta.height}</code>
								· <code>{previewMeta.processTimeMs.toFixed(0)} ms</code>
								· wr <code>{(previewMeta.whiteRatio * 100).toFixed(1)}%</code>
								{#if previewMeta.material}· LUT <code>{previewMeta.material}</code>{/if}
							</span>
						{/if}
					</header>
					{#if previewError}
						<p class="warn">⚠ {previewError}</p>
					{/if}
					<div class="compare-wrap">
						<div class="compare" style={`--p:${comparePos}%`}>
							<div class="half before">
								<img src={croppedUrl} alt="Antes" />
								<span class="tag">Antes</span>
							</div>
							<div class="divider" aria-hidden="true"></div>
							<div class="half after">
								{#if previewBlobUrl}
									<img src={previewBlobUrl} alt="Preview laser" />
									<span class="tag">Laser preview</span>
								{:else if previewBusy}
									<div class="placeholder">Renderizando preview…</div>
								{:else}
									<div class="placeholder">Mové un slider para generar preview</div>
								{/if}
							</div>
						</div>
						<label class="slider-compare">
							<span class="sr-only">Posición comparador</span>
							<input type="range" min="0" max="100" bind:value={comparePos} />
						</label>
					</div>
				</section>
			</div>
		{:else if step === 4 && finalBlobUrl && finalMeta}
			<div class="panel">
				<h2 class="title">Resultado full-res</h2>
				<p class="meta-line">
					<code>{finalMeta.width}×{finalMeta.height}</code>
					· wr <code>{(finalMeta.whiteRatio * 100).toFixed(1)}%</code>
					· <code>{finalMeta.processTimeMs.toFixed(0)} ms</code>
					· sharpen radius <code>{finalMeta.sharpenRadiusPx.toFixed(2)} px</code>
					{#if finalMeta.material}· LUT <code>{finalMeta.material}</code>{/if}
				</p>

				<div class="result-grid">
					<figure>
						<figcaption>PNG laser-ready (1-bit)</figcaption>
						<img src={finalBlobUrl} alt="Salida laser-ready" class="out-img" />
					</figure>
					<figure>
						<figcaption>
							Simulación de grabado físico
							{#if simMeta}
								<span class="meta-line">
									· σ <code>{simMeta.sigmaPx.toFixed(2)} px</code>
									· spot <code>{simMeta.spotMm.toFixed(2)} mm</code>
									{#if simMeta.material}· <code>{simMeta.material}</code>{/if}
								</span>
							{/if}
						</figcaption>
						{#if simBlobUrl}
							<img src={simBlobUrl} alt="Simulación grabado" class="out-img" />
						{:else}
							<div class="sim-placeholder">
								<p>Aproximación de cómo se verá fotografiado tras grabar.</p>
								<div class="sim-actions">
									<button class="btn btn-ghost" disabled={simBusy} onclick={() => runSimulation()}>
										{#if simBusy}Simulando…{:else}Simular grabado{/if}
									</button>
									<button class="btn btn-ghost" disabled={simBusy} onclick={() => runSimulation('wood_burn')}>
										Simular madera
									</button>
								</div>
							</div>
						{/if}
						{#if simError}<p class="warn">⚠ {simError}</p>{/if}
					</figure>
				</div>

				{#if recommendedSettings}
					<section class="lightburn-card">
						<h3>📋 Configuración recomendada para LightBurn</h3>
						<p class="hint">
							Material <code>{recommendedSettings.material}</code>
							{#if recommendedSettings.machine_compat}· {recommendedSettings.machine_compat}{/if}
						</p>
						<div class="lb-grid">
							<div><span class="lb-label">DPI</span><span class="lb-val">{recommendedSettings.dpi}</span></div>
							<div><span class="lb-label">Interval mm</span><span class="lb-val">{recommendedSettings.interval_mm.toFixed(3)}</span></div>
							<div><span class="lb-label">Power %</span><span class="lb-val">{recommendedSettings.power_pct_min}–{recommendedSettings.power_pct_max}</span></div>
							<div><span class="lb-label">Speed mm/s</span><span class="lb-val">{recommendedSettings.speed_mm_s_min.toFixed(0)}–{recommendedSettings.speed_mm_s_max.toFixed(0)}</span></div>
							<div><span class="lb-label">Pass-Through</span><span class="lb-val">{recommendedSettings.pass_through ? '✓ ON' : '✗ OFF'}</span></div>
							<div><span class="lb-label">MirrorX</span><span class="lb-val">{recommendedSettings.mirror_x_required ? '✓ ON (back-engrave)' : '— no necesario'}</span></div>
							<div><span class="lb-label">Invert en LightBurn</span><span class="lb-val">{recommendedSettings.lightburn_invert ? '✓ ON' : '✗ OFF (el PNG ya está listo)'}</span></div>
							<div><span class="lb-label">Focus mm</span><span class="lb-val">{recommendedSettings.focus_mm > 0 ? recommendedSettings.focus_mm.toFixed(1) : '—'}</span></div>
						</div>
						{#if recommendedSettings.notes}
							<p class="hint lb-notes">💡 {recommendedSettings.notes}</p>
						{/if}
					</section>
				{/if}
				<div class="actions">
					<button type="button" class="btn btn-secondary" onclick={() => (step = 3)}>← Ajustar</button>
					<!-- v2.1: 2 botones — descarga directa rápida + step 5 para checklist completo -->
					<button type="button" class="btn btn-primary" onclick={downloadFinal} title="Descarga directa sin pasar por step 5">
						⬇ Descargar PNG ahora
					</button>
					<button type="button" class="btn btn-secondary" onclick={() => { step = 5; }}>
						Ver checklist →
					</button>
				</div>
			</div>
		{:else if step === 5 && finalBlobUrl}
			<div class="panel">
				<h2 class="title">Descargar PNG laser-ready</h2>
				<p class="lede">
					Importá el PNG en LightBurn (u otro CAM). Usá <strong>Pass-Through</strong> + threshold para que envíe el bitmap intacto.
				</p>
				<div class="final-preview">
					<img src={finalBlobUrl} alt="Salida final" class="out-img" />
				</div>
				<div class="actions">
					<button type="button" class="btn btn-secondary" onclick={() => (step = 4)}>← Volver</button>
					<button type="button" class="btn btn-primary" onclick={downloadFinal}>Descargar</button>
					<button type="button" class="btn btn-ghost" onclick={backToUpload}>Nueva imagen</button>
				</div>
				<details class="checklist">
					<summary>Checklist pre-grabado (recomendado)</summary>
					<ul>
						<li>Espejar (MirrorX) en el CAM si vas a grabar la cara posterior (back-engrave acrílico).</li>
						<li>Configurar interval = <code>25.4 / DPI</code> mm en la capa imagen.</li>
						<li>Pass-Through activado en LightBurn (no re-dither el PNG).</li>
						<li>Probar 9–12% potencia en acrílico antes de grabar el archivo definitivo.</li>
					</ul>
				</details>
			</div>
		{/if}
	</main>
</div>

<style>
	:global(html, body) {
		margin: 0;
		padding: 0;
		min-height: 100%;
		font-family: 'Inter', system-ui, -apple-system, sans-serif;
		background:
			radial-gradient(circle at 15% 0%, rgba(80, 140, 60, 0.18), transparent 50%),
			radial-gradient(circle at 85% 100%, rgba(40, 120, 80, 0.16), transparent 55%),
			#0c1410;
		color: #e6f0e0;
	}
	:global(:root) {
		--bg-panel: rgba(20, 30, 22, 0.78);
		--bg-panel-strong: rgba(28, 40, 30, 0.92);
		--border: rgba(120, 200, 120, 0.18);
		--border-strong: rgba(160, 220, 140, 0.34);
		--text: #e6f0e0;
		--text-muted: #a8baa3;
		--text-faint: #7a8a78;
		--accent: #8ee06b;
		--accent-dark: #5ea342;
		--accent-glow: rgba(142, 224, 107, 0.32);
		--warn: #f0c64a;
		--err: #f07a6e;
	}
	.app-shell {
		max-width: 1320px;
		margin: 0 auto;
		padding: 1.5rem 1.5rem 4rem;
	}
	.topbar {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 0.6rem 0 1rem;
		border-bottom: 1px solid var(--border);
		margin-bottom: 1rem;
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-weight: 600;
		letter-spacing: 0.02em;
		color: var(--accent);
	}
	.status {
		display: flex;
		align-items: center;
		gap: 0.6rem;
		font-size: 0.82rem;
		color: var(--text-muted);
	}
	.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
	.dot.ok { background: var(--accent); box-shadow: 0 0 8px var(--accent-glow); }
	.dot.err { background: var(--err); }
	.dot.pending { background: var(--warn); animation: pulse 1s ease-in-out infinite; }
	@keyframes pulse { 50% { opacity: 0.5; } }
	.badge {
		font-size: 0.7rem;
		padding: 0.15rem 0.5rem;
		border-radius: 99px;
		font-weight: 600;
		letter-spacing: 0.04em;
	}
	.badge.cuda { background: rgba(142, 224, 107, 0.18); color: var(--accent); border: 1px solid var(--border-strong); }
	.badge.cpu { background: rgba(170, 170, 170, 0.12); color: var(--text-muted); border: 1px solid var(--border); }
	.badge.busy { background: rgba(240, 198, 74, 0.18); color: var(--warn); border: 1px solid rgba(240, 198, 74, 0.3); padding: 0.3rem 0.7rem; }
	.hint-error { color: var(--err); }

	.steps {
		display: flex;
		gap: 0.4rem;
		justify-content: center;
		flex-wrap: wrap;
		margin-bottom: 1.4rem;
	}
	.step-pill {
		display: inline-flex;
		align-items: center;
		gap: 0.45rem;
		padding: 0.45rem 0.9rem;
		border: 1px solid var(--border);
		background: var(--bg-panel);
		color: var(--text-muted);
		border-radius: 99px;
		font-size: 0.85rem;
		cursor: pointer;
		font-family: inherit;
		transition: all 0.18s ease;
	}
	.step-pill:hover:not(:disabled) {
		border-color: var(--border-strong);
		color: var(--text);
	}
	.step-pill:disabled { opacity: 0.4; cursor: not-allowed; }
	.step-pill .step-n {
		width: 22px;
		height: 22px;
		border-radius: 50%;
		display: inline-flex;
		align-items: center;
		justify-content: center;
		font-size: 0.74rem;
		font-weight: 700;
		background: rgba(255, 255, 255, 0.05);
	}
	.step-pill.active {
		background: linear-gradient(135deg, rgba(142, 224, 107, 0.18), rgba(94, 163, 66, 0.1));
		border-color: var(--accent);
		color: var(--text);
		box-shadow: 0 0 0 1px var(--accent-glow), 0 0 20px var(--accent-glow);
	}
	.step-pill.active .step-n { background: var(--accent); color: #0c1410; }
	.step-pill.done { color: var(--accent); }
	.step-pill.done .step-n { background: rgba(94, 163, 66, 0.55); color: #0c1410; }

	.panel {
		background: var(--bg-panel);
		border: 1px solid var(--border);
		border-radius: 14px;
		padding: 1.6rem;
		backdrop-filter: blur(8px);
	}
	.panel-warn { border-color: rgba(240, 198, 74, 0.4); }
	.panel pre { background: rgba(0, 0, 0, 0.4); padding: 0.6rem 0.8rem; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; }

	.title { font-size: 1.25rem; margin: 0 0 0.4rem; color: var(--text); }
	.lede { color: var(--text-muted); margin: 0 0 1.4rem; }
	.hint { color: var(--text-faint); font-size: 0.82rem; margin: 0.4rem 0 0; }
	.warn { color: var(--warn); background: rgba(240, 198, 74, 0.06); border-left: 3px solid var(--warn); padding: 0.5rem 0.7rem; border-radius: 4px; font-size: 0.85rem; margin: 0.5rem 0; }

	/* v2.1: tarjeta de error con botón retry */
	.error-card {
		margin-top: 0.8rem;
		padding: 0.9rem 1rem;
		background: rgba(240, 122, 110, 0.08);
		border: 1px solid rgba(240, 122, 110, 0.3);
		border-radius: 8px;
	}
	.error-card .warn { margin: 0 0 0.7rem; }
	.error-card .actions { margin-top: 0.5rem; }

	.dropzone {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 0.4rem;
		border: 2px dashed var(--border-strong);
		border-radius: 12px;
		padding: 3rem 1rem;
		cursor: pointer;
		transition: all 0.2s;
		text-align: center;
		background: rgba(0, 0, 0, 0.15);
	}
	.dropzone:hover {
		border-color: var(--accent);
		background: rgba(142, 224, 107, 0.04);
	}
	.dropzone input { display: none; }
	.dropzone-icon { color: var(--accent); }
	.dropzone-title { font-weight: 600; color: var(--text); }
	.dropzone-sub { color: var(--text-muted); font-size: 0.85rem; }

	/* Step 3 — layout split */
	.step3-layout {
		display: grid;
		grid-template-columns: minmax(280px, 380px) 1fr;
		gap: 1.2rem;
		align-items: start;
	}
	@media (max-width: 900px) {
		.step3-layout { grid-template-columns: 1fr; }
	}
	.panel-controls { padding: 1rem 1.1rem; }
	.panel-preview { padding: 1rem 1.1rem; }

	.preset-toggle {
		display: flex;
		gap: 0.3rem;
		background: rgba(0, 0, 0, 0.25);
		padding: 0.25rem;
		border-radius: 10px;
		margin-bottom: 1rem;
	}
	.preset-toggle button {
		flex: 1;
		padding: 0.5rem;
		background: transparent;
		border: 0;
		color: var(--text-muted);
		border-radius: 7px;
		font-size: 0.85rem;
		cursor: pointer;
		font-family: inherit;
		transition: all 0.15s;
	}
	.preset-toggle button.active {
		background: var(--accent);
		color: #0c1410;
		font-weight: 600;
	}

	.group { border-top: 1px solid var(--border); padding: 0.5rem 0; }
	.group summary {
		cursor: pointer;
		font-size: 0.85rem;
		font-weight: 600;
		letter-spacing: 0.02em;
		text-transform: uppercase;
		color: var(--text-muted);
		padding: 0.6rem 0;
	}
	.group summary:hover { color: var(--text); }
	.group[open] summary { color: var(--accent); }

	.field { margin: 0.6rem 0; }
	.field label, .slider label {
		display: block;
		font-size: 0.78rem;
		color: var(--text-muted);
		margin-bottom: 0.3rem;
	}
	.field input, .field select {
		width: 100%;
		padding: 0.5rem 0.7rem;
		background: rgba(0, 0, 0, 0.25);
		color: var(--text);
		border: 1px solid var(--border);
		border-radius: 7px;
		font-size: 0.9rem;
		font-family: inherit;
	}
	.field input:focus, .field select:focus { outline: none; border-color: var(--accent); }
	.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; }
	.field-row { display: flex; gap: 0.6rem; }
	.check { display: flex; align-items: center; gap: 0.4rem; font-size: 0.85rem; color: var(--text); cursor: pointer; }

	.slider { margin: 0.5rem 0; }
	.slider input[type=range] {
		width: 100%;
		accent-color: var(--accent);
	}
	.slider label code {
		background: rgba(142, 224, 107, 0.12);
		color: var(--accent);
		padding: 0.05rem 0.4rem;
		border-radius: 4px;
		font-family: 'JetBrains Mono', 'Consolas', monospace;
		font-size: 0.78rem;
		float: right;
	}

	.actions {
		display: flex;
		gap: 0.5rem;
		margin-top: 1rem;
		flex-wrap: wrap;
	}
	.btn {
		padding: 0.6rem 1.1rem;
		border-radius: 8px;
		border: 1px solid var(--border-strong);
		background: var(--bg-panel-strong);
		color: var(--text);
		cursor: pointer;
		font-family: inherit;
		font-size: 0.9rem;
		font-weight: 500;
		transition: all 0.15s;
	}
	.btn:hover:not(:disabled) { border-color: var(--accent); }
	.btn:disabled { opacity: 0.4; cursor: not-allowed; }
	.btn-primary { background: linear-gradient(135deg, var(--accent), var(--accent-dark)); border-color: var(--accent); color: #0c1410; font-weight: 600; }
	.btn-primary:hover:not(:disabled) { box-shadow: 0 0 14px var(--accent-glow); }
	.btn-secondary { background: rgba(0, 0, 0, 0.2); }
	.btn-ghost { background: transparent; border-color: var(--border); color: var(--text-muted); }

	.preview-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 1rem;
		margin-bottom: 0.8rem;
		flex-wrap: wrap;
	}
	.meta-line {
		font-size: 0.78rem;
		color: var(--text-muted);
		font-family: 'JetBrains Mono', 'Consolas', monospace;
	}
	.meta-line code {
		color: var(--accent);
		background: rgba(0, 0, 0, 0.3);
		padding: 0.1rem 0.4rem;
		border-radius: 4px;
		margin: 0 0.1rem;
	}

	.compare-wrap { margin-top: 0.5rem; }
	.compare {
		position: relative;
		border-radius: 10px;
		overflow: hidden;
		border: 1px solid var(--border);
		min-height: 380px;
		background: repeating-conic-gradient(#0f1612 0 25%, #131c16 0 50%) 50% / 16px 16px;
	}
	.half {
		position: absolute;
		inset: 0;
		overflow: hidden;
	}
	.half.before {
		clip-path: inset(0 calc(100% - var(--p, 50%)) 0 0);
	}
	.half.after {
		clip-path: inset(0 0 0 var(--p, 50%));
	}
	.half img {
		width: 100%;
		height: 100%;
		object-fit: contain;
		display: block;
	}
	.placeholder {
		width: 100%;
		height: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--text-faint);
		font-size: 0.9rem;
		padding: 1rem;
		text-align: center;
	}
	.divider {
		position: absolute;
		left: var(--p, 50%);
		top: 0;
		bottom: 0;
		width: 2px;
		margin-left: -1px;
		background: linear-gradient(180deg, var(--accent), var(--accent-dark));
		box-shadow: 0 0 12px var(--accent-glow);
		z-index: 2;
		pointer-events: none;
	}
	.tag {
		position: absolute;
		bottom: 10px;
		left: 10px;
		font-size: 0.7rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		padding: 0.3rem 0.55rem;
		border-radius: 6px;
		background: rgba(12, 20, 16, 0.85);
		border: 1px solid var(--border);
		color: var(--accent);
		z-index: 3;
	}
	.half.after .tag { left: auto; right: 10px; }
	.slider-compare {
		display: block;
		margin-top: 0.5rem;
	}
	.slider-compare input[type=range] {
		width: 100%;
		accent-color: var(--accent);
	}
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		border: 0;
	}

	.final-preview {
		display: flex;
		justify-content: center;
		margin: 1rem 0;
	}
	.out-img {
		max-width: 100%;
		max-height: 540px;
		object-fit: contain;
		border: 1px solid var(--border);
		border-radius: 8px;
		background: repeating-conic-gradient(#0f1612 0 25%, #131c16 0 50%) 50% / 16px 16px;
	}

	.checklist {
		margin-top: 1.4rem;
		border-top: 1px solid var(--border);
		padding-top: 1rem;
	}
	.checklist summary {
		cursor: pointer;
		color: var(--text-muted);
		font-size: 0.9rem;
	}
	.checklist ul {
		margin: 0.6rem 0 0;
		padding-left: 1.2rem;
		color: var(--text);
	}
	.checklist li { margin: 0.3rem 0; font-size: 0.9rem; }
	.checklist code { background: rgba(0, 0, 0, 0.3); padding: 0.05rem 0.35rem; border-radius: 4px; color: var(--accent); }

	.result-grid {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 1rem;
		margin: 0.8rem 0 1rem;
	}
	@media (max-width: 720px) {
		.result-grid { grid-template-columns: 1fr; }
	}
	.result-grid figure {
		margin: 0;
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.result-grid figcaption {
		font-size: 0.85rem;
		color: var(--text-muted);
		font-weight: 600;
	}
	.sim-placeholder {
		border: 1px dashed var(--border);
		border-radius: 8px;
		padding: 1rem;
		display: flex;
		flex-direction: column;
		gap: 0.6rem;
		min-height: 220px;
		justify-content: center;
		align-items: center;
		background: rgba(0, 0, 0, 0.18);
	}
	.sim-placeholder p {
		color: var(--text-muted);
		font-size: 0.85rem;
		text-align: center;
		margin: 0;
	}
	.sim-actions {
		display: flex;
		gap: 0.4rem;
		flex-wrap: wrap;
		justify-content: center;
	}

	/* Mode toggle (Express vs Manual) */
	.mode-toggle {
		display: grid;
		grid-template-columns: 1fr 1fr;
		gap: 0.6rem;
		margin-bottom: 1.4rem;
	}
	.mode-toggle button {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		align-items: flex-start;
		padding: 1rem 1.2rem;
		background: rgba(0, 0, 0, 0.25);
		border: 1px solid var(--border);
		border-radius: 12px;
		color: var(--text-muted);
		font-family: inherit;
		font-size: 0.95rem;
		font-weight: 600;
		cursor: pointer;
		text-align: left;
		transition: all 0.18s ease;
	}
	.mode-toggle button:hover { border-color: var(--border-strong); color: var(--text); }
	.mode-toggle button.active {
		background: linear-gradient(135deg, rgba(142, 224, 107, 0.18), rgba(94, 163, 66, 0.1));
		border-color: var(--accent);
		color: var(--text);
		box-shadow: 0 0 14px var(--accent-glow);
	}
	.mode-toggle small {
		font-size: 0.75rem;
		font-weight: 400;
		color: var(--text-faint);
		letter-spacing: 0;
	}
	.mode-toggle button.active small { color: var(--text-muted); }

	/* Mi configuración */
	.my-config-panel {
		border: 1px solid var(--border);
		border-radius: 10px;
		margin-bottom: 1.2rem;
		padding: 0.4rem 0.9rem;
		background: rgba(0, 0, 0, 0.18);
	}
	.my-config-panel summary {
		cursor: pointer;
		padding: 0.5rem 0;
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 0.6rem;
		font-size: 0.9rem;
		font-weight: 600;
		color: var(--text);
		list-style: none;
	}
	.my-config-panel summary::-webkit-details-marker { display: none; }
	.my-config-panel[open] summary { color: var(--accent); }
	.my-config-summary {
		font-size: 0.78rem;
		font-weight: 400;
		color: var(--text-muted);
		font-family: 'JetBrains Mono', 'Consolas', monospace;
		background: rgba(0, 0, 0, 0.3);
		padding: 0.2rem 0.5rem;
		border-radius: 6px;
	}

	.dropzone-express {
		border-color: var(--accent);
		background: rgba(142, 224, 107, 0.06);
	}
	.dropzone-express:hover {
		background: rgba(142, 224, 107, 0.12);
		box-shadow: 0 0 14px var(--accent-glow);
	}

	/* Toggle de métrica (v5 CPU / v4 GPU) en Mi configuración */
	.metric-toggle {
		margin-top: 0.8rem;
		padding-top: 0.8rem;
		border-top: 1px dashed var(--border);
	}
	.metric-toggle-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.5rem;
	}
	.metric-toggle-label {
		font-size: 0.85rem;
		color: var(--text);
		font-weight: 600;
	}
	.metric-toggle-badge {
		font-size: 0.72rem;
		padding: 0.18rem 0.55rem;
		border-radius: 99px;
		background: rgba(170, 170, 170, 0.12);
		color: var(--text-muted);
		border: 1px solid var(--border);
		font-family: 'JetBrains Mono', 'Consolas', monospace;
	}
	.metric-toggle-badge.active {
		background: rgba(142, 224, 107, 0.18);
		color: var(--accent);
		border-color: var(--border-strong);
	}
	.metric-toggle-options {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}
	.metric-opt {
		display: flex;
		gap: 0.55rem;
		padding: 0.6rem 0.8rem;
		border: 1px solid var(--border);
		border-radius: 8px;
		background: rgba(0, 0, 0, 0.18);
		cursor: pointer;
		transition: all 0.15s;
	}
	.metric-opt:hover { border-color: var(--border-strong); }
	.metric-opt.selected {
		border-color: var(--accent);
		background: rgba(142, 224, 107, 0.06);
		box-shadow: 0 0 0 1px var(--accent-glow);
	}
	.metric-opt input[type='radio'] {
		margin-top: 0.25rem;
		accent-color: var(--accent);
	}
	.metric-opt-body {
		display: flex;
		flex-direction: column;
		gap: 0.18rem;
		flex: 1;
	}
	.metric-opt-body strong { font-size: 0.88rem; color: var(--text); }
	.metric-opt-desc {
		font-size: 0.78rem;
		color: var(--text-muted);
		line-height: 1.4;
	}
	.metric-opt-desc em { color: var(--accent); font-style: normal; }
	.warn-inline { color: var(--warn); }
	.metric-opt-desc code, .snippet {
		font-family: 'JetBrains Mono', 'Consolas', monospace;
		font-size: 0.72rem;
		background: rgba(0, 0, 0, 0.35);
		padding: 0.05rem 0.35rem;
		border-radius: 4px;
		color: var(--accent);
	}
	.snippet {
		display: block;
		margin-top: 0.3rem;
		padding: 0.35rem 0.5rem;
		overflow-x: auto;
		white-space: nowrap;
	}

	/* Progress card (Express + Manual) */
	.progress-card {
		margin-top: 1rem;
		padding: 0.9rem 1rem;
		background: rgba(142, 224, 107, 0.06);
		border: 1px solid var(--border-strong);
		border-radius: 10px;
	}
	.progress-head {
		display: flex;
		justify-content: space-between;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.6rem;
	}
	.progress-msg {
		color: var(--text);
		font-size: 0.9rem;
		font-weight: 500;
	}
	.btn-cancel {
		padding: 0.35rem 0.75rem;
		font-size: 0.78rem;
	}
	.progress-bar {
		height: 8px;
		width: 100%;
		background: rgba(0, 0, 0, 0.35);
		border-radius: 99px;
		overflow: hidden;
		position: relative;
	}
	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent-dark), var(--accent));
		border-radius: 99px;
		transition: width 0.25s ease;
		box-shadow: 0 0 8px var(--accent-glow);
	}
	.progress-fill.indeterminate {
		width: 30% !important;
		animation: indet 1.4s cubic-bezier(0.4, 0, 0.6, 1) infinite;
	}
	@keyframes indet {
		0% { transform: translateX(-100%); }
		50% { transform: translateX(120%); }
		100% { transform: translateX(420%); }
	}
	.progress-stats {
		display: flex;
		gap: 0.4rem;
		margin-top: 0.45rem;
		font-size: 0.78rem;
		color: var(--text-muted);
		font-family: 'JetBrains Mono', 'Consolas', monospace;
		flex-wrap: wrap;
	}
	.progress-stats code {
		color: var(--accent);
		background: rgba(0, 0, 0, 0.3);
		padding: 0.05rem 0.4rem;
		border-radius: 4px;
	}
	.progress-stats strong {
		color: var(--text);
	}

	.progress-card-compact { padding: 0.65rem 0.85rem; }

	/* Telemetría grid + cells */
	.telemetry-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
		gap: 0.45rem;
		margin-top: 0.7rem;
	}
	.tm-cell {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding: 0.4rem 0.55rem;
		background: rgba(0, 0, 0, 0.28);
		border-radius: 6px;
		border: 1px solid var(--border);
	}
	.tm-label {
		font-size: 0.7rem;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.tm-val {
		font-family: 'JetBrains Mono', 'Consolas', monospace;
		font-size: 0.88rem;
		color: var(--accent);
		font-weight: 600;
	}

	/* Sparkline */
	.sparkline-wrap {
		margin-top: 0.8rem;
		padding: 0.55rem 0.65rem;
		background: rgba(0, 0, 0, 0.28);
		border-radius: 6px;
		border: 1px solid var(--border);
	}
	.sparkline-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.5rem;
		margin-bottom: 0.3rem;
	}
	.sparkline-range {
		font-size: 0.72rem;
		color: var(--text-faint);
		font-family: 'JetBrains Mono', 'Consolas', monospace;
	}
	.sparkline {
		display: block;
		width: 100%;
		height: 36px;
	}
	.sparkline-compact { height: 24px; margin-top: 0.45rem; opacity: 0.85; }

	/* Log scrollable */
	.log-details {
		margin-top: 0.7rem;
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 0.4rem 0.6rem;
		background: rgba(0, 0, 0, 0.28);
	}
	.log-details summary {
		cursor: pointer;
		font-size: 0.78rem;
		color: var(--text-muted);
		padding: 0.2rem 0;
		font-weight: 500;
		text-transform: uppercase;
		letter-spacing: 0.04em;
	}
	.log-details[open] summary { color: var(--accent); }
	.log-scroll {
		margin-top: 0.4rem;
		max-height: 200px;
		overflow-y: auto;
		font-family: 'JetBrains Mono', 'Consolas', monospace;
		font-size: 0.76rem;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		padding-right: 0.2rem;
	}
	.log-line {
		display: flex;
		gap: 0.6rem;
		padding: 0.18rem 0.35rem;
		border-radius: 3px;
		color: var(--text);
	}
	.log-line:hover { background: rgba(255, 255, 255, 0.03); }
	.log-t {
		color: var(--text-faint);
		flex-shrink: 0;
		min-width: 3.5rem;
		text-align: right;
	}
	.log-msg {
		flex: 1;
		word-break: break-word;
	}
	.log-warn .log-msg { color: var(--warn); }
	.log-error .log-msg { color: var(--err); }

	/* Scrollbar styling for log */
	.log-scroll::-webkit-scrollbar { width: 6px; }
	.log-scroll::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); border-radius: 3px; }
	.log-scroll::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
	.log-scroll::-webkit-scrollbar-thumb:hover { background: var(--accent); }

	/* LightBurn recommendations card */
	.lightburn-card {
		margin: 1.4rem 0 0;
		padding: 1.1rem 1.2rem;
		background: rgba(142, 224, 107, 0.06);
		border: 1px solid var(--border-strong);
		border-radius: 12px;
	}
	.lightburn-card h3 {
		margin: 0 0 0.4rem;
		color: var(--accent);
		font-size: 1.05rem;
	}
	.lb-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
		gap: 0.5rem 1rem;
		margin: 0.6rem 0;
	}
	.lb-grid > div {
		display: flex;
		justify-content: space-between;
		gap: 0.5rem;
		padding: 0.4rem 0.6rem;
		background: rgba(0, 0, 0, 0.25);
		border-radius: 6px;
	}
	.lb-label { color: var(--text-muted); font-size: 0.82rem; }
	.lb-val {
		color: var(--accent);
		font-family: 'JetBrains Mono', 'Consolas', monospace;
		font-size: 0.85rem;
		font-weight: 600;
	}
	.lb-notes { margin-top: 0.8rem; }
</style>
