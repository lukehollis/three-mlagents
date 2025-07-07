import React, { useState } from 'react';
import { Button, useMediaQuery } from '@geist-ui/core';
import { BlockMath } from 'react-katex';
import { ChevronsDown, ChevronsUp } from '@geist-ui/icons';
import { Function } from '@geist-ui/icons'


export default function EquationPanel({ equation, description }) {
  let defaultOpen = true;
  if (window.innerWidth < 768) {
    defaultOpen = false;
  }

  const [isOpen, setIsOpen] = useState(defaultOpen);
  const isMediumOrSmaller = useMediaQuery('md');



  if (!isOpen) {
    return (
      <div style={{ position: 'absolute', bottom: 10, left: 10, zIndex: 1 }}>
        <Button
          auto
          icon={<Function />}
          onClick={() => setIsOpen(true)}
          type="secondary"
        >
          Equation
        </Button>
      </div>
    );
  }

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 10,
        left: 10,
        width: isMediumOrSmaller ? '45%' : 'calc(100vw - 54px)',
        background: 'rgba(0,0,0,0.95)',
        color: '#fff',
        padding: '12px 16px',
        fontSize: isMediumOrSmaller ? 12 : 14,
        borderRadius: '8px',
        border: '1px solid rgba(255,255,255,0.2)',
        zIndex: 10,
        overflow: 'hidden',
      }}
    >
      <Button
        auto
        scale={0.5}
        icon={<ChevronsDown />}
        onClick={() => setIsOpen(false)}
        style={{
          position: 'absolute',
          top: 5,
          right: 5,
          zIndex: 2,
        }}
        type="ghost"
        ghost
      />

      <div style={{ overflowX: 'scroll' }}>
        <BlockMath
          math={equation}
        />
      </div>
      <div style={{ fontSize: isMediumOrSmaller ? 8 : 10, fontFamily: 'monospace', marginTop: 4 }}>
        {description}
      </div>
    </div>
  );
} 