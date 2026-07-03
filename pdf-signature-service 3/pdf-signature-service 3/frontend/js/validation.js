const VALIDATION = {
  MAX_PDF_BYTES:   15 * 1024 * 1024,  // backend limit
  MAX_IMAGE_BYTES:  5 * 1024 * 1024,  // backend limit

  validatePdf(file) {
    if (!file) return "Please select a PDF file.";
    if (file.type !== "application/pdf") return "Only PDF files are accepted.";
    if (file.size > this.MAX_PDF_BYTES)
      return `PDF exceeds the 15 MB limit (selected: ${(file.size / 1024 / 1024).toFixed(1)} MB).`;
    return null;
  },

  validateImage(file) {
    if (!file) return "Please select a signature image.";
    if (!["image/png", "image/jpeg"].includes(file.type))
      return "Only PNG or JPEG images are accepted.";
    if (file.size > this.MAX_IMAGE_BYTES)
      return `Image exceeds the 5 MB limit (selected: ${(file.size / 1024 / 1024).toFixed(1)} MB).`;
    return null;
  },
};
