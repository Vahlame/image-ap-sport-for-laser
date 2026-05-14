<script lang="ts">
	import { tick } from 'svelte';
	import CropStage from '$lib/components/CropStage.svelte';
	import { canvasToPngBlob, grayscaleThresholdCanvas, imageToCanvas } from '$lib/laserPreview';

	type Step = 1 | 2 | 3 | 4 | 5;

	let step = $state<Step>(1);
	let originalUrl = $state<string | null>(null);
	let croppedUrl = $state<string | null>(null);

	let material = $state('madera');
	let widthMm = $state(100);
	let heightMm = $state(100);
	let dpi = $state(300);
	let ditherLabel = $state('jarvis');
	let threshold = $state(128);
	let contrast = $state(1);
	let brightness = $state(0);

	/** PNG de preview (umbral simple) generado en el navegador */
	let laserBlobUrl = $state<string | null>(null);

	/** Feedback: el preview actual es O(n) píxeles — en GPU/CPU moderna parece “instantáneo”. */
	let previewProcessing = $state(false);
	let previewMeta = $state({ ms: 0, width: 0, height: 0, megapixels: 0 });

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
		revoke(laserBlobUrl);
		laserBlobUrl = null;
		croppedUrl = null;
		previewMeta = { ms: 0, width: 0, height: 0, megapixels: 0 };
		originalUrl = URL.createObjectURL(f);
		step = 2;
		input.value = '';
	}

	function onCropped(blob: Blob) {
		revoke(croppedUrl);
		revoke(laserBlobUrl);
		laserBlobUrl = null;
		previewMeta = { ms: 0, width: 0, height: 0, megapixels: 0 };
		croppedUrl = URL.createObjectURL(blob);
		step = 3;
	}

	function backToUpload() {
		revoke(originalUrl);
		revoke(croppedUrl);
		revoke(laserBlobUrl);
		originalUrl = null;
		croppedUrl = null;
		laserBlobUrl = null;
		previewMeta = { ms: 0, width: 0, height: 0, megapixels: 0 };
		step = 1;
	}

	function backToCrop() {
		step = 2;
	}

	function goResult() {
		step = 4;
	}

	function goDownload() {
		step = 5;
	}

	$effect(() => {
		const src = croppedUrl;
		const th = threshold;
		const ct = contrast;
		const br = brightness;
		const st = step;
		if (!src || st < 3) {
			previewProcessing = false;
			return;
		}

		let cancelled = false;
		previewProcessing = true;

		const tid = setTimeout(() => {
			if (cancelled) return;
			const t0 = performance.now();
			const img = new Image();
			img.decoding = 'async';
			img.onload = () => {
				if (cancelled) return;
				const w = img.naturalWidth;
				const h = img.naturalHeight;
				const base = imageToCanvas(img);
				const out = grayscaleThresholdCanvas(base, th, ct, br);
				void canvasToPngBlob(out).then((blob) => {
					if (cancelled || !blob) {
						previewProcessing = false;
						return;
					}
					const elapsed = performance.now() - t0;
					void tick().then(() => {
						if (cancelled) return;
						revoke(laserBlobUrl);
						laserBlobUrl = URL.createObjectURL(blob);
						previewMeta = {
							ms: elapsed,
							width: w,
							height: h,
							megapixels: (w * h) / 1_000_000
						};
						previewProcessing = false;
					});
				});
			};
			img.onerror = () => {
				if (!cancelled) previewProcessing = false;
			};
			img.src = src;
		}, 160);

		return () => {
			cancelled = true;
			clearTimeout(tid);
			previewProcessing = false;
		};
	});

	function downloadPng() {
		if (!laserBlobUrl) return;
		const a = document.createElement('a');
		a.href = laserBlobUrl;
		a.download = `laser-preview-${material}-${widthMm}x${heightMm}mm-${dpi}dpi.png`;
		a.click();
	}
</script>

<div class="steps" aria-label="Pasos del asistente">
	{#each steps as s (s.n)}
		<span
			class="step-pill"
			class:active={step === s.n}
			class:done={step > s.n}
			data-step={s.n}>{s.label}</span>
	{/each}
</div>

{#if step === 1}
	<div class="panel">
		<div class="panel-upload-inner">
			<div class="panel-intro">
				<h2 class="title">Subir imagen</h2>
				<p class="lede">
					Subí una foto nítida del trabajo. En el siguiente paso elegís solo el recorte que va al láser.
				</p>
			</div>
			<label class="dropzone">
				<input type="file" accept="image/png,image/jpeg,image/webp" onchange={onDropzoneChange} />
				<span class="dropzone-icon" aria-hidden="true">
					<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
						<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
						<polyline points="17 8 12 3 7 8" />
						<line x1="12" y1="3" x2="12" y2="15" />
					</svg>
				</span>
				<span class="dropzone-title">Arrastrá o hacé clic</span>
				<span class="dropzone-sub">Soltá el archivo en esta zona</span>
				<span class="dropzone-formats">PNG · JPG · WebP</span>
			</label>
		</div>
	</div>
{:else if step === 2 && originalUrl}
	{#key originalUrl}
		<CropStage src={originalUrl} onBack={backToUpload} onContinue={onCropped} />
	{/key}
{:else if step === 3}
	<div class="panel">
		<h2 class="title">Ajustes de grabado</h2>
		<p class="subtitle">Parámetros de taller (el pipeline final irá en FastAPI).</p>

		<div class="grid2">
			<div class="field">
				<label for="mat">Material (preset)</label>
				<select id="mat" bind:value={material}>
					<option value="madera">Madera</option>
					<option value="acrilico">Acrílico</option>
					<option value="cuero">Cuero</option>
					<option value="pizarra">Pizarra negra</option>
				</select>
			</div>
			<div class="field">
				<label for="dith">Dither (referencia)</label>
				<select id="dith" bind:value={ditherLabel}>
					<option value="jarvis">Jarvis</option>
					<option value="floyd">Floyd-Steinberg</option>
					<option value="atkinson">Atkinson</option>
					<option value="stucki">Stucki</option>
				</select>
			</div>
			<div class="field">
				<label for="wmm">Ancho (mm)</label>
				<input id="wmm" type="number" min="1" step="0.1" bind:value={widthMm} />
			</div>
			<div class="field">
				<label for="hmm">Alto (mm)</label>
				<input id="hmm" type="number" min="1" step="0.1" bind:value={heightMm} />
			</div>
			<div class="field">
				<label for="dpi">DPI</label>
				<input id="dpi" type="number" min="72" max="1200" step="1" bind:value={dpi} />
			</div>
			<div class="field">
				<label for="thr">Umbral preview (0–255)</label>
				<input id="thr" type="range" min="0" max="255" bind:value={threshold} />
				<span class="mono">{threshold}</span>
			</div>
			<div class="field">
				<label for="ct">Contraste preview</label>
				<input id="ct" type="range" min="0.5" max="2.5" step="0.05" bind:value={contrast} />
				<span class="mono">{contrast.toFixed(2)}</span>
			</div>
			<div class="field">
				<label for="br">Brillo preview</label>
				<input id="br" type="range" min="-60" max="60" step="1" bind:value={brightness} />
				<span class="mono">{brightness}</span>
			</div>
		</div>

		<p class="preview-status" class:busy={previewProcessing}>
			{#if previewProcessing}
				Recalculando preview (gris + contraste + brillo + umbral)…
			{:else if previewMeta.ms > 0}
				Último preview: <span class="mono">{previewMeta.ms.toFixed(1)} ms</span> · {previewMeta.width}×{previewMeta.height} px
				(<span class="mono">{previewMeta.megapixels.toFixed(2)} MP</span>). Un solo barrido sobre los píxeles; en una CPU moderna suele sentirse “instantáneo”.
			{:else}
				Al mover los deslizadores se vuelve a pintar el canvas en el navegador (sin servidor).
			{/if}
		</p>
		<p class="hint hint-preview">
			El pipeline final (resize mm/DPI real, nitidez, dither tipo Jarvis, etc.) hará más trabajo; puede ir a Web Worker o a FastAPI.
		</p>

		<div class="actions">
			<button type="button" class="btn btn-secondary" onclick={backToCrop}>← Volver a recorte</button>
			<button type="button" class="btn" onclick={goResult}>Ver resultado</button>
		</div>
	</div>
{:else if step === 4 && croppedUrl && laserBlobUrl}
	<div class="panel">
		<h2 class="title">Antes / después (preview local)</h2>
		<p class="subtitle">
			Izquierda: recorte a color. Derecha: escala de grises + umbral simple (no es el dither Jarvis real).
		</p>
		{#if previewMeta.ms > 0}
			<p class="preview-status preview-status-inline">
				Tiempo del último preview: <span class="mono">{previewMeta.ms.toFixed(1)} ms</span> · {previewMeta.width}×{previewMeta.height} px.
			</p>
		{/if}

		<div class="compare-wrap">
			<div class="compare" style={`--p:${comparePos}%`}>
				<div class="half before">
					<img src={croppedUrl} alt="Antes" />
					<span class="tag">Antes</span>
				</div>
				<div class="divider" aria-hidden="true"></div>
				<div class="half after">
					<img src={laserBlobUrl} alt="Después" />
					<span class="tag">Después</span>
				</div>
			</div>
			<label class="slider-compare">
				<span class="sr-only">Posición comparador</span>
				<input type="range" min="0" max="100" bind:value={comparePos} />
			</label>
		</div>

		<div class="actions">
			<button type="button" class="btn btn-secondary" onclick={() => (step = 3)}>← Ajustes</button>
			<button type="button" class="btn" onclick={goDownload}>Descargar PNG →</button>
		</div>
	</div>
{:else if step === 4 && croppedUrl && !laserBlobUrl}
	<div class="panel panel-loading">Generando preview…</div>
{:else if step === 5 && laserBlobUrl}
	<div class="panel">
		<h2 class="title">Descargar PNG</h2>
		<p class="subtitle">
			Archivo generado en el navegador (preview). Cuando exista el backend, aquí irá el PNG del pipeline Python.
		</p>

		<div class="final-preview">
			<img src={laserBlobUrl} alt="Salida preview" class="out-img" />
		</div>

		<div class="actions">
			<button type="button" class="btn btn-secondary" onclick={() => (step = 4)}>← Resultado</button>
			<button type="button" class="btn" onclick={downloadPng}>Descargar PNG</button>
			<button type="button" class="btn btn-secondary" onclick={backToUpload}>Nueva imagen</button>
		</div>

		<p class="hint">
			Importá el PNG en tu CAM y configurá Pass-Through / umbral según tu máquina.
		</p>
	</div>
{/if}

<style>
	.preview-status {
		margin: 1rem 0 0;
		font-size: 0.9rem;
		line-height: 1.45;
		color: var(--text-muted, #94a3b8);
	}
	.preview-status.busy {
		color: var(--accent, #38bdf8);
	}
	.preview-status-inline {
		margin-top: 0.35rem;
	}
	.hint-preview {
		margin-top: 0.5rem;
	}
	.compare-wrap {
		margin-top: 0.5rem;
	}
	.compare {
		display: grid;
		grid-template-columns: 1fr 0 1fr;
		position: relative;
		border-radius: 8px;
		overflow: hidden;
		border: 1px solid var(--border);
		min-height: 280px;
	}
	.half {
		position: relative;
		overflow: hidden;
		min-height: 280px;
		background: rgba(7, 11, 18, 0.85);
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
	.divider {
		position: absolute;
		left: var(--p, 50%);
		top: 0;
		bottom: 0;
		width: 3px;
		margin-left: -1.5px;
		background: linear-gradient(180deg, #38bdf8, #a78bfa);
		z-index: 2;
		pointer-events: none;
	}
	.tag {
		position: absolute;
		bottom: 10px;
		left: 10px;
		font-size: 0.72rem;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		padding: 0.35rem 0.6rem;
		border-radius: 8px;
		background: rgba(15, 23, 42, 0.75);
		backdrop-filter: blur(8px);
		border: 1px solid rgba(255, 255, 255, 0.1);
		color: #f1f5f9;
		z-index: 1;
	}
	.half.after .tag {
		left: auto;
		right: 8px;
	}
	.final-preview {
		display: flex;
		justify-content: center;
		margin: 1rem 0;
	}
	.out-img {
		max-width: 100%;
		max-height: 480px;
		object-fit: contain;
		border: 1px solid var(--border);
		border-radius: 8px;
		background: repeating-conic-gradient(#1e293b 0% 25%, #0f172a 0% 50%) 50% / 16px 16px;
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
</style>
