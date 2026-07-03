const CONFIG = {
  API_BASE_URL: window.ENV_API_BASE_URL || "http://localhost:8000/api/v1",

  ENDPOINTS: {
    UPLOAD:   () => `${CONFIG.API_BASE_URL}/pdf/upload`,
    SIGN:     () => `${CONFIG.API_BASE_URL}/pdf/sign`,
    DOWNLOAD: (taskId) => `${CONFIG.API_BASE_URL}/pdf/download/${taskId}`,
  },
};
