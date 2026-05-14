<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Preparación láser — Easy local</title>
	<link rel="preconnect" href="https://fonts.googleapis.com" />
	<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous" />
	<link
		href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..700&family=JetBrains+Mono:wght@400;500;600&display=swap"
		rel="stylesheet"
	/>
</svelte:head>

<div class="app-shell">
	<header class="app-header">
		<div class="app-header-inner">
			<div class="app-header-titles">
				<h1>Preparación imagen láser</h1>
				<p>
					Flujo guiado: subir → recortar → ajustes → vista previa → PNG. Ahora todo corre en el navegador;
					el pipeline Python se conectará después.
				</p>
			</div>
			<span class="app-badge" title="Versión de desarrollo">MVP · preview local</span>
		</div>
	</header>
	<main class="app-main">
		{@render children()}
	</main>
</div>
