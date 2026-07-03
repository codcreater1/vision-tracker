const state = {
  pdfFile: null,
  sigFile: null,
  taskId: null,
  pages: [],        // [{ index, width, height }] in PDF points
  currentPage: 0,
  pdfDoc: null,     // pdf.js document object
  sigPlacement: null, // { x_pt, y_pt, w_pt, h_pt }
};

// DOM
const pdfDropzone   = document.getElementById("pdf-dropzone");
const sigDropzone   = document.getElementById("sig-dropzone");
const pdfInput      = document.getElementById("pdf-input");
const sigInput      = document.getElementById("sig-input");
const uploadBtn     = document.getElementById("upload-btn");
const uploadSection = document.getElementById("upload-section");
const previewSection= document.getElementById("preview-section");
const downloadSection=document.getElementById("download-section");
const pdfCanvas     = document.getElementById("pdf-canvas");
const sigOverlay    = document.getElementById("sig-overlay");
const signBtn       = document.getElementById("sign-btn");
const downloadBtn   = document.getElementById("download-btn");
const errorBanner   = document.getElementById("error-banner");
const pdfFileName   = document.getElementById("pdf-file-name");
const sigFileName   = document.getElementById("sig-file-name");

// Helpers

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.style.display = "block";
  setTimeout(() => { errorBanner.style.display = "none"; }, 5000);
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  btn.dataset.originalText = btn.dataset.originalText || btn.textContent;
  btn.textContent = loading ? "Loading…" : btn.dataset.originalText;
}

// Drag & drop

function setupDropzone(dropzone, input, onFile) {
  dropzone.addEventListener("click", () => input.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  input.addEventListener("change", () => {
    if (input.files[0]) onFile(input.files[0]);
  });
}

function onPdfSelected(file) {
  const err = VALIDATION.validatePdf(file);
  if (err) { showError(err); return; }
  state.pdfFile = file;
  pdfDropzone.classList.add("has-file");
  pdfFileName.textContent = file.name;
}

function onSigSelected(file) {
  const err = VALIDATION.validateImage(file);
  if (err) { showError(err); return; }
  state.sigFile = file;
  sigDropzone.classList.add("has-file");
  sigFileName.textContent = file.name;
}

setupDropzone(pdfDropzone, pdfInput, onPdfSelected);
setupDropzone(sigDropzone, sigInput, onSigSelected);

// Upload

uploadBtn.addEventListener("click", async () => {
  if (!state.pdfFile) { showError("Please select a PDF file first."); return; }
  if (!state.sigFile) { showError("Please select a signature image first."); return; }

  setLoading(uploadBtn, true);
  try {
    const data = await uploadPdf(state.pdfFile);
    state.taskId = data.task_id;
    state.pages  = data.pages;

    uploadSection.style.display  = "none";
    previewSection.style.display = "block";
    await renderPage(state.currentPage);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(uploadBtn, false);
  }
});

// PDF preview

async function renderPage(pageIndex) {
  const pdfjsLib = window["pdfjs-dist/build/pdf"];
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";

  if (!state.pdfDoc) {
    const fileUrl = URL.createObjectURL(state.pdfFile);
    state.pdfDoc  = await pdfjsLib.getDocument(fileUrl).promise;
  }

  const page    = await state.pdfDoc.getPage(pageIndex + 1); // pdf.js is 1-indexed
  const viewport= page.getViewport({ scale: 1.5 });

  pdfCanvas.width  = viewport.width;
  pdfCanvas.height = viewport.height;

  await page.render({ canvasContext: pdfCanvas.getContext("2d"), viewport }).promise;

  sigOverlay.style.width  = pdfCanvas.width  + "px";
  sigOverlay.style.height = pdfCanvas.height + "px";
  state.sigPlacement = null;
  renderSigOverlay();
}

// Signature placement

const SIG_W_PX = 240; // signature box width in canvas pixels
const SIG_H_PX =  90;

sigOverlay.addEventListener("click", (e) => {
  const rect = sigOverlay.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  // pixel → PDF point conversion
  const pageInfo = state.pages[state.currentPage];
  const scaleX   = pageInfo.width  / pdfCanvas.width;
  const scaleY   = pageInfo.height / pdfCanvas.height;

  state.sigPlacement = {
    x_pt: clickX * scaleX,
    y_pt: clickY * scaleY,
    w_pt: SIG_W_PX * scaleX,
    h_pt: SIG_H_PX * scaleY,
    x_px: clickX,
    y_px: clickY,
  };
  renderSigOverlay();
});

function renderSigOverlay() {
  const ctx = sigOverlay.getContext("2d");
  ctx.clearRect(0, 0, sigOverlay.width, sigOverlay.height);

  if (!state.sigPlacement) {
    ctx.fillStyle = "rgba(79,70,229,0.10)";
    ctx.fillRect(0, 0, sigOverlay.width, sigOverlay.height);
    ctx.fillStyle = "#4f46e5";
    ctx.font = "16px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Click to place your signature", sigOverlay.width / 2, sigOverlay.height / 2);
    return;
  }

  const { x_px, y_px } = state.sigPlacement;
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, sigOverlay.width, sigOverlay.height);
    ctx.drawImage(img, x_px, y_px, SIG_W_PX, SIG_H_PX);
    ctx.strokeStyle = "#4f46e5";
    ctx.lineWidth   = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x_px, y_px, SIG_W_PX, SIG_H_PX);
  };
  img.src = URL.createObjectURL(state.sigFile);
}

// Keep overlay dimensions in sync with the PDF canvas
Object.defineProperty(sigOverlay, "width", {
  get() { return pdfCanvas.width; },
  set(v) { this.setAttribute("width", v); }
});
Object.defineProperty(sigOverlay, "height", {
  get() { return pdfCanvas.height; },
  set(v) { this.setAttribute("height", v); }
});

// Sign

signBtn.addEventListener("click", async () => {
  if (!state.sigPlacement) { showError("Please click on the PDF to place your signature first."); return; }

  setLoading(signBtn, true);
  try {
    await signPdf({
      taskId:    state.taskId,
      imageFile: state.sigFile,
      page:      state.currentPage,
      x:         state.sigPlacement.x_pt,
      y:         state.sigPlacement.y_pt,
      w:         state.sigPlacement.w_pt,
      h:         state.sigPlacement.h_pt,
    });

    previewSection.style.display  = "none";
    downloadSection.style.display = "block";
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(signBtn, false);
  }
});

// Download

downloadBtn.addEventListener("click", async () => {
  setLoading(downloadBtn, true);
  try {
    const blob = await downloadPdf(state.taskId);
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href     = url;
    a.download = "signed.pdf";
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(downloadBtn, false);
  }
});
