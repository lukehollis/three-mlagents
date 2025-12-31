import { createRoot } from 'react-dom/client';
import { GeistProvider, CssBaseline, Themes } from '@geist-ui/core';
import { Provider } from 'react-redux';
import '@fontsource-variable/geist';
import App from './App.jsx';
import store from './store.js';

const tacticalTheme = Themes.createFromDark({
  type: 'tactical',
  layout: {
    radius: '0px',
  },
  font: {
    sans: '"Orbitron", "Roboto Mono", monospace',
    mono: '"Roboto Mono", monospace',
  },
});

const container = document.getElementById('root');
createRoot(container).render(
  <Provider store={store}>
    <GeistProvider themes={[tacticalTheme]} themeType="tactical">
      <CssBaseline />
      <App />
    </GeistProvider>
  </Provider>
); 