import React, { useRef, useEffect } from 'react';
import { Text } from '@geist-ui/core';

const MessagePanel = ({ 
  messages, 
  title = null, 
  position = { position: 'absolute', bottom: '10px', left: '10px' },
  width = '450px',
  height = null,
  maxHeight = '40vh',
  style = {}
}) => {
  const containerRef = useRef(null);
  
  useEffect(() => {
    const el = containerRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages]);

  const codeStyle = { color: '#f81ce5', fontFamily: 'monospace' };
  
  const defaultStyle = {
    ...position,
    width,
    ...(height && { height }),
    maxHeight,
    overflowY: 'auto',
    background: 'rgba(0,0,0,0.6)',
    color: '#fff',
    border: '1px solid #444',
    padding: '16px',
    borderRadius: '5px',
    ...style
  };

  return (
    <div ref={containerRef} style={defaultStyle}>
      {title && (
        <Text h5 style={{ margin: '0 0 8px 0', color: '#37F5EB' }}>
          {title}
        </Text>
      )}
      {(!messages || messages.length === 0) && (
        <Text p style={{ margin: 0, fontSize: '12px' }}>[No messages]</Text>
      )}
      {messages && messages.map((msg, i) => {
        let content;
        if (msg.recipient_id !== null && msg.recipient_id !== undefined) {
          content = (
            <Text p style={{ margin: 0 }}>
              <span style={codeStyle}>[DM to {msg.recipient_id}]</span> {msg.message}
            </Text>
          );
        } else {
          content = (
            <Text p style={{ margin: 0 }}>
              <span style={codeStyle}>[Broadcast]</span> {msg.message}
            </Text>
          );
        }
        
        return (
          <div 
            key={i} 
            style={{ 
              marginBottom: '12px', 
              padding: '8px', 
              background: 'rgba(255,255,255,0.05)', 
              borderRadius: '4px', 
              fontSize: '12px' 
            }}
          >
            <Text p style={{ margin: 0, fontWeight: 'bold' }}>
              <span style={codeStyle}>[Step {msg.step}] Agent {msg.sender_id}</span>
            </Text>
            {content}
          </div>
        );
      })}
    </div>
  );
};

export default MessagePanel;
