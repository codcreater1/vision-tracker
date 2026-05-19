class ValidationError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = "ValidationError";
    this.statusCode = statusCode;
  }
}

async function apiFetch(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (_) {
    throw new Error("Sunucuya ulaşılamıyor. Backend çalışıyor mu?");
  }

  if (!response.ok) {
    let errorDetail = `Sunucu hatası (HTTP ${response.status})`;
    try {
      const body = await response.json();
      errorDetail = body.detail || errorDetail;
    } catch (_) {}

    if (response.status === 400 || response.status === 422) {
      throw new ValidationError(errorDetail, response.status);
    }
    throw new Error(errorDetail);
  }

  return response;
}

async function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await apiFetch(CONFIG.ENDPOINTS.UPLOAD(), { method: "POST", body: form });
  return res.json();
}

async function signPdf({ taskId, imageFile, page, x, y, w, h }) {
  const form = new FormData();
  form.append("task_id", taskId);
  form.append("image", imageFile);
  form.append("page", String(page));
  form.append("x", String(x));
  form.append("y", String(y));
  form.append("w", String(w));
  form.append("h", String(h));
  const res = await apiFetch(CONFIG.ENDPOINTS.SIGN(), { method: "POST", body: form });
  return res.json();
}

async function downloadPdf(taskId) {
  const res = await apiFetch(CONFIG.ENDPOINTS.DOWNLOAD(taskId));
  return res.blob();
}
