/**
 * Client-side preview only (not the final Python pipeline).
 * Converts image to grayscale + simple global threshold for UI feedback.
 */
export function imageToCanvas(img: HTMLImageElement): HTMLCanvasElement {
	const canvas = document.createElement('canvas');
	canvas.width = img.naturalWidth;
	canvas.height = img.naturalHeight;
	const ctx = canvas.getContext('2d');
	if (!ctx) throw new Error('2d context unavailable');
	ctx.drawImage(img, 0, 0);
	return canvas;
}

export function grayscaleThresholdCanvas(
	source: HTMLCanvasElement,
	threshold: number,
	contrast = 1,
	brightness = 0
): HTMLCanvasElement {
	const out = document.createElement('canvas');
	out.width = source.width;
	out.height = source.height;
	const sctx = source.getContext('2d');
	const octx = out.getContext('2d');
	if (!sctx || !octx) throw new Error('2d context unavailable');

	const id = sctx.getImageData(0, 0, source.width, source.height);
	const d = id.data;
	for (let i = 0; i < d.length; i += 4) {
		let y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
		y = (y - 128) * contrast + 128 + brightness;
		y = Math.max(0, Math.min(255, y));
		const v = y >= threshold ? 255 : 0;
		d[i] = v;
		d[i + 1] = v;
		d[i + 2] = v;
		d[i + 3] = 255;
	}
	octx.putImageData(id, 0, 0);
	return out;
}

export function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob | null> {
	return new Promise((resolve) => {
		canvas.toBlob((b) => resolve(b), 'image/png');
	});
}
