import React from 'react';
import config from '../config.js';

export default function ModelInfoPanel({ modelInfo }) {
  if (!modelInfo) {
    return null;
  }

  return (
    <div
      style={{
        marginTop: '8px',
        fontSize: '12px',
        background: 'rgba(0,255,0,0.1)',
        padding: '4px 8px',
        borderRadius: '4px',
        border: '1px solid rgba(0,255,0,0.3)',
      }}
    >
      <div>
        <strong>Model:</strong> {modelInfo.filename}
      </div>
      <div>
        <strong>Trained:</strong>{' '}
        {modelInfo.timestamp.replace(
          /(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/,
          '$1-$2-$3 $4:$5:$6'
        )}
      </div>
      <div>
        <strong>Session:</strong> {modelInfo.sessionUuid}
      </div>
      <a
        href={`${config.API_BASE_URL}${modelInfo.fileUrl}`}
        download
        style={{ color: '#4CAF50', textDecoration: 'underline' }}
      >
        Download Policy
      </a>
    </div>
  );
} 