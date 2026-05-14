<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import Cropper from 'cropperjs';
	import 'cropperjs/dist/cropper.css';

	let {
		src,
		onBack,
		onContinue
	}: {
		src: string;
		onBack: () => void;
		onContinue: (blob: Blob) => void;
	} = $props();

	let imgEl = $state<HTMLImageElement | null>(null);
	let cropper = $state<Cropper | null>(null);
	let error = $state<string | null>(null);

	onMount(() => {
		const el = imgEl;
		if (!el) return;

		const init = () => {
			try {
				cropper?.destroy();
				cropper = new Cropper(el, {
					viewMode: 1,
					dragMode: 'move',
					autoCropArea: 0.9,
					responsive: true
				});
			} catch (e) {
				error = e instanceof Error ? e.message : 'No se pudo iniciar el recorte';
			}
		};

		if (el.complete) init();
		else el.addEventListener('load', init, { once: true });
	});

	onDestroy(() => {
		cropper?.destroy();
		cropper = null;
	});

	function handleContinue() {
		if (!cropper) return;
		const canvas = cropper.getCroppedCanvas();
		canvas.toBlob(
			(blob: Blob | null) => {
				if (blob) onContinue(blob);
				else error = 'No se pudo generar el recorte';
			},
			'image/png',
			1
		);
	}
</script>

<div class="panel">
	<h2 class="title">Recortar imagen</h2>
	<p class="subtitle">Arrastra la imagen y usa las esquinas para ajustar el área.</p>

	{#if error}
		<p class="err" role="alert">{error}</p>
	{/if}

	<div class="crop-wrap">
		<img bind:this={imgEl} {src} alt="" crossorigin="anonymous" class="crop-img" />
	</div>

	<div class="actions">
		<button type="button" class="btn btn-secondary" onclick={onBack}>← Cambiar imagen</button>
		<button type="button" class="btn" onclick={handleContinue} disabled={!cropper}>Continuar →</button>
	</div>
</div>

<style>
	.title {
		margin: 0 0 0.25rem;
		font-size: 1.1rem;
	}
	.subtitle {
		margin: 0 0 1rem;
		font-size: 0.9rem;
		color: var(--text-muted);
	}
	.err {
		color: var(--danger);
		font-size: 0.9rem;
	}
	.crop-wrap {
		max-height: min(70vh, 640px);
	}

	:global(.cropper-container) {
		max-height: min(70vh, 640px);
	}

	.crop-img {
		display: block;
		max-width: 100%;
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.75rem;
		margin-top: 1rem;
	}
</style>
