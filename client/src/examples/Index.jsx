import React from 'react';
import { Link } from 'react-router-dom';

export default function ExamplesIndex() {
  return (
    <div style={{ padding: 24 }}>
      <h1>RL-Agents Examples</h1>
      <p>Select a scene:</p>
      <ul>
        <li>
          <Link to="/basic">/basic – 1-D Move-To-Goal</Link>
        </li>
      </ul>
      <p>
        Controls and reward structure replicate the Unity&nbsp;ML-Agents <code>BasicController</code> scene.
      </p>
    </div>
  );
} 