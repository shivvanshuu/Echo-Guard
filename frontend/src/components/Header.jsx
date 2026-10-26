import React from 'react';
import { useTelemetry } from '../context/TelemetryContext';
import { Volume2, VolumeX } from 'lucide-react';
import DemoMenu from './DemoMenu';

export default function Header({ title, description, onOpenDispatch }) {
  const { wsConnected, audioEnabled, setAudioEnabled } = useTelemetry();

  return (
    <header className="app-header">
      <div className="topbar-title">
        <h2>{title}</h2>
        <p>{description}</p>
      </div>

      <div className="header-status-group">
        <div className="status-pill" role="status">
          <span className={`status-indicator ${wsConnected ? 'online' : 'offline'}`} aria-hidden="true" />
          <span>{wsConnected ? 'Live' : 'Reconnecting'}</span>
        </div>

        <button
          type="button"
          className="icon-btn"
          onClick={() => setAudioEnabled(!audioEnabled)}
          aria-label={audioEnabled ? 'Mute alert sound' : 'Unmute alert sound'}
          aria-pressed={!audioEnabled}
          title={audioEnabled ? 'Mute alert sound' : 'Unmute alert sound'}
        >
          {audioEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
        </button>

        <DemoMenu onOpenDispatch={onOpenDispatch} />
      </div>
    </header>
  );
}
