import { createRoot } from 'react-dom/client';
import { GeistProvider, CssBaseline } from '@geist-ui/core';
import { Provider } from 'react-redux';
import '@fontsource-variable/geist';
import App from './App.jsx';
import store from './store.js';

const container = document.getElementById('root');
createRoot(container).render(
  <Provider store={store}>
    <GeistProvider themeType="dark">
      <CssBaseline />
      <App />
    </GeistProvider>
  </Provider>
); 