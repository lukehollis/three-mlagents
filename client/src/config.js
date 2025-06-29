// Configuration for API endpoints
const config = {
  // Update these URLs to match your deployed backend
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  WS_BASE_URL: import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000',
};

export default config; 