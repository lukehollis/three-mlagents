import { useState, useEffect } from 'react';

const breakpoints = {
  xs: '480px',
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
};

export const useResponsive = () => {
  const [width, setWidth] = useState(window.innerWidth);

  useEffect(() => {
    const handleResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return {
    isXs: width <= parseInt(breakpoints.xs),
    isSm: width <= parseInt(breakpoints.sm),
    isMd: width <= parseInt(breakpoints.md),
    isLg: width <= parseInt(breakpoints.lg),
    isXl: width <= parseInt(breakpoints.xl),
    isMobile: width < parseInt(breakpoints.md),
    isTablet: width >= parseInt(breakpoints.md) && width < parseInt(breakpoints.lg),
    isDesktop: width >= parseInt(breakpoints.lg),
  };
}; 