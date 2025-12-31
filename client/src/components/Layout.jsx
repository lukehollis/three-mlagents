import { Page, useMediaQuery } from '@geist-ui/core';

export default function Layout({ children }) {
  const isXs = useMediaQuery('xs');

  return (
    <Page style={{ 
      backgroundColor: 'transparent', 
      minHeight: '100vh', 
      padding: isXs ? '0 1rem' : undefined, 
      margin: '0 auto', 
      maxWidth: '1200px', 
      width: '100vw' 
    }}>
      {children}
    </Page>
  );
}
