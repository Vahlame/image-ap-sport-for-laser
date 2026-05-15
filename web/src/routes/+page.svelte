<script lang="ts">
	import { onMount, tick } from 'svelte';
	import CropStage from '$lib/components/CropStage.svelte';
	import {
		apiClient,
		DEFAULT_PARAMS,
		PRESET_AGRICULTOR_HIGH_CONTRAST,
		type AlgorithmGroup,
		type MaterialInfo,
		type ProcessMeta,
		type ProcessParams,
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
			// Si no hay material seleccionado, sugerimos acrylic por defecto
			if (!params.material && materials.some((m) => m.name === 'acrylic_back_engrave')) {
				params.material = 'acrylic_back_engrave';
			}
		} catch (err) {
			backendOnline = false;
			connectionError = err instanceof Error ? err.message : String(err);
		}
	});

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
		if (!f) return;
		if (!/^image\//.test(f.type)) {
			alert('Formato no soportado. Usa PNG, JPG o WebP.');
			input.value = '';
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
		input.value = '';
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
		try {
			const { blob, meta } = await apiClient.render('process', croppedBlob, params);
			revoke(finalBlobUrl);
			finalBlob = blob;
			finalBlobUrl = URL.createObjectURL(blob);
			finalMeta = meta;
			// limpiar simulación previa
			if (simBlobUrl) URL.revokeObjectURL(simBlobUrl);
			simBlobUrl = null;
			simMeta = null;
			simError = null;
			await tick();
			step = 4;
		} catch (err) {
			previewError = err instanceof Error ? err.message : String(err);
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
		if (!finalBlobUrl) return;
		const matLabel = params.material || 'generic';
		const dimsLabel = params.output_mm_short > 0 ? `_${params.output_mm_short}mm_${params.output_dpi}dpi` : '';
		const a = document.createElement('a');
		a.href = finalBlobUrl;
		a.download = `laser_ready_${matLabel}_${params.algorithm}${dimsLabel}.png`;
		a.click();
	}

	function backToUpload() {
		revoke(originalUrl);
		revoke(croppedUrl);
		revoke(previewBlobUrl);
		revoke(finalBlobUrl);
		originalUrl = null;
		croppedBlob = null;
		croppedUrl = null;
		previewBlobUrl = null;
		previewMeta = null;
		finalBlob = null;
		finalBlobUrl = null;
		finalMeta = null;
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
				<h2 class="title">Subir imagen</h2>
				<p class="lede">
					Subí una foto nítida. En el siguiente paso elegís solo el recorte que va al láser.
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
							</select>
						</div>
						<div class="field-row">
							<label class="check"><input type="checkbox" bind:checked={params.invert} /> Invertir</label>
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

				<div class="actions">
					<button type="button" class="btn btn-secondary" onclick={() => (step = 3)}>← Ajustar</button>
					<button type="button" class="btn btn-primary" onclick={() => { step = 5; }}>Descargar PNG →</button>
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
</style>
