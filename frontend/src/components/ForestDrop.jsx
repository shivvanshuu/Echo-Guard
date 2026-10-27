import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Upload, TriangleAlert, LoaderCircle, Check } from 'lucide-react';
import { useTelemetry } from '../context/TelemetryContext';
import { dropAudioInForest, fetchListeningRange } from '../services/api';
import { threatLabel } from '../utils/threat';

const W = 1000;
const H = 620;
const ACCEPTED = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.mp4', '.webm', '.mov', '.mkv'];
const MAX_MB = 50;

const LOUDNESS = [
  { id: 'quiet', label: 'Quiet' },
  { id: 'normal', label: 'Normal' },
  { id: 'loud', label: 'Loud' },
];

const BASE_DETECTION_RADIUS_M = 2200;
const DEFAULT_DETECTION_RADII_M = {
  quiet: 1000,
  normal: 2200,
  loud: 5500,
};

const THREAT_COLOR = {
  CHAINSAW: 'var(--threat-chainsaw)',
  GUNSHOT: 'var(--threat-gunshot)',
  VEHICLE_ENGINE: 'var(--threat-vehicle)',
  ILLEGAL_VESSEL: 'var(--threat-vessel)',
  BLAST_FISHING: 'var(--threat-blast)',
};

const STATUS_TEXT = {
  detected: 'Threat detected',
  heard_no_threat: 'Heard clearly, no threat detected',
  too_faint: 'Too faint to hear',
};

function getNodeStatus(node) {
  if (!node) return 'too_faint';
  if (node.status === 'detected') {
    return 'detected';
  }
  // UI fallback logic: If calculated SNR is strictly positive (> 0 dB) but classification is unclassified (0%),
  // do NOT set alert status to "Too faint to hear". Correctly label as "Heard clearly, no threat detected".
  if (node.snr_db > 0 && (node.status === 'too_faint' || node.status === 'heard_no_threat' || !node.confidence || node.confidence === 0)) {
    return 'heard_no_threat';
  }
  return node.status || 'too_faint';
}

// ---- Forest scenery: fixed so it does not change between renders ---------------------------
function mulberry32(seed) {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const CANOPY_FILL = ['#e8efe3', '#e0e9da', '#d8e4d1'];
const CANOPY = (() => {
  const rand = mulberry32(7);
  return Array.from({ length: 120 }, () => ({
    x: rand() * W,
    y: rand() * H,
    r: 12 + rand() * 30,
    tone: Math.floor(rand() * 3),
  }));
})();

// ---- Map projection: sensor positions to canvas pixels and back ----------------------------
function useProjection(nodes) {
  return useMemo(() => {
    const lat0 = nodes.reduce((sum, n) => sum + n.lat, 0) / nodes.length;
    const lon0 = nodes.reduce((sum, n) => sum + n.lon, 0) / nodes.length;
    const metresPerLat = 110574;
    const metresPerLon = 111320 * Math.cos((lat0 * Math.PI) / 180);

    const xs = nodes.map((n) => (n.lon - lon0) * metresPerLon);
    const ys = nodes.map((n) => (lat0 - n.lat) * metresPerLat);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1000);
    const pad = Math.max(1500, span * 0.2);
    const minX = Math.min(...xs) - pad;
    const maxX = Math.max(...xs) + pad;
    const minY = Math.min(...ys) - pad;
    const maxY = Math.max(...ys) + pad;
    const scale = Math.min(W / (maxX - minX), H / (maxY - minY));
    const offX = (W - (maxX - minX) * scale) / 2;
    const offY = (H - (maxY - minY) * scale) / 2;
    const pixelsPerKm = scale * 1000;

    return {
      scale,
      pixelsPerKm,
      toPx: (lat, lon) => ({
        x: offX + ((lon - lon0) * metresPerLon - minX) * scale,
        y: offY + ((lat0 - lat) * metresPerLat - minY) * scale,
      }),
      toGeo: (x, y) => ({
        lon: lon0 + ((x - offX) / scale + minX) / metresPerLon,
        lat: lat0 - ((y - offY) / scale + minY) / metresPerLat,
      }),
    };
  }, [nodes]);
}

// ---- Small helpers -------------------------------------------------------------------------
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const formatDistance = (m) => (m < 1000 ? `${Math.round(m)} m` : `${(m / 1000).toFixed(1)} km`);
const formatDb = (db) => `${db > 0 ? '+' : ''}${db.toFixed(1)} dB`;
const formatSeconds = (s) => `${s.toFixed(1)} s`;
const percent = (value) => `${Math.round(value * 100)}%`;
const shortName = (id) => id.replace(/^NODE-/, '');
const formatCoord = (value, pos, neg) => `${Math.abs(value).toFixed(4)}° ${value >= 0 ? pos : neg}`;

function checkFile(file) {
  const name = file.name.toLowerCase();
  if (!ACCEPTED.some((ext) => name.endsWith(ext))) {
    return `This file type is not supported. Use one of: ${ACCEPTED.join(', ')}.`;
  }
  if (file.size > MAX_MB * 1024 * 1024) {
    return `This file is larger than ${MAX_MB} MB.`;
  }
  return '';
}

// ---- Clip waveform with the per-window verdicts underneath ---------------------------------
function ClipTimeline({ clip }) {
  const width = 600;
  const bars = clip.envelope;
  const barWidth = width / bars.length;
  const duration = clip.duration_s || 1;
  const peak = clip.verdict_at_source;

  return (
    <div>
      <svg
        viewBox="0 0 600 132"
        width="100%"
        role="img"
        aria-label={`Waveform of the clip, ${formatSeconds(duration)} long, with what the classifier found in each two-second window`}
      >
        {bars.map((value, i) => {
          const h = Math.max(1.5, value * 38);
          return (
            <rect
              key={i}
              x={i * barWidth + 0.5}
              y={46 - h}
              width={Math.max(1, barWidth - 1)}
              height={h * 2}
              rx="1"
              style={{ fill: 'var(--text-dim)' }}
            />
          );
        })}

        {clip.windows.map((w, i) => {
          const x = (w.start_s / duration) * width;
          const wWidth = ((w.end_s - w.start_s) / duration) * width;
          const isThreat = w.threat_type !== 'UNKNOWN' && w.threat_type !== 'SILENCE';
          const label =
            w.threat_type === 'SILENCE'
              ? 'silence'
              : isThreat
                ? `${threatLabel(w.threat_type)}, ${percent(w.confidence)}`
                : 'no threat';
          return (
            <rect
              key={i}
              x={x}
              y="96"
              width={Math.max(1, wWidth - 1)}
              height="16"
              rx="2"
              style={{
                fill: isThreat ? THREAT_COLOR[w.threat_type] || 'var(--threat-unknown)' : 'var(--bg-tertiary)',
                opacity: isThreat ? 0.3 + 0.7 * w.confidence : 1,
                stroke: peak.at_s === w.start_s ? 'var(--text-primary)' : 'none',
                strokeWidth: 1.5,
              }}
            >
              <title>{`${formatSeconds(w.start_s)} to ${formatSeconds(w.end_s)}: ${label}`}</title>
            </rect>
          );
        })}

        <text x="0" y="128" fontSize="11" style={{ fill: 'var(--text-muted)' }}>0 s</text>
        <text x={width} y="128" fontSize="11" textAnchor="end" style={{ fill: 'var(--text-muted)' }}>
          {formatSeconds(duration)}
        </text>
      </svg>
      <p className="fd-help">
        Top: loudness over time. Bottom: what the classifier found in each two-second window; darker means more
        confident. The outlined window is the strongest threat moment.
      </p>
    </div>
  );
}

// ---- Results -------------------------------------------------------------------------------
function Verdict({ result }) {
  const nodes = result.nodes;
  const strongest = nodes.find((n) => n.id === result.strongest_node);
  const source = result.clip.verdict_at_source;

  if (strongest) {
    return (
      <>
        <h3>
          {strongest.name || strongest.id} heard it best: {threatLabel(strongest.threat_type).toLowerCase()},{' '}
          {percent(strongest.confidence)} confident
        </h3>
        <p>
          {strongest.name ? `${strongest.id}, ` : ''}
          {formatDistance(strongest.distance_m)} from the sound. {result.alerts_sent}{' '}
          {result.alerts_sent === 1 ? 'alert was' : 'alerts were'} sent to the dashboard.
          {result.alerts_sent === 0 && ' (Alerts were switched off for this run.)'}
        </p>
      </>
    );
  }
  if (!source || !source.threat_type || source.threat_type === 'UNKNOWN' || source.threat_type === 'NONE') {
    return (
      <>
        <h3>No threat in this clip</h3>
        <p>The classifier found nothing in the audio that should raise an alert, so no sensor reported anything.</p>
      </>
    );
  }
  return (
    <>
      <h3>The clip has a threat, but every sensor was too far away</h3>
      <p>
        Close to the source the classifier hears {threatLabel(source.threat_type).toLowerCase()} (
        {percent(source.confidence)}), but the sound was quieter than the background noise at every sensor. Try
        dropping it closer to a sensor, or choose a louder setting.
      </p>
    </>
  );
}

function NodeTable({ result }) {
  return (
    <div className="fd-table-wrap">
      <table className="fd-table">
        <thead>
          <tr>
            <th>Sensor</th>
            <th>Distance</th>
            <th title="How loud the sound is compared with the background noise at this sensor">Sound vs noise</th>
            <th>Result</th>
            <th>Confidence</th>
            <th>Alert</th>
          </tr>
        </thead>
        <tbody>
          {result.nodes.map((n) => {
            const isStrongest = n.id === result.strongest_node;
            const effectiveStatus = getNodeStatus(n);
            return (
              <tr key={n.id} data-strongest={isStrongest}>
                <td>
                  <strong>{n.name || n.id}</strong>
                  {isStrongest && <span className="fd-badge">Strongest</span>}
                  <div className="fd-mono">{n.id}</div>
                </td>
                <td>{formatDistance(n.distance_m)}</td>
                <td>{formatDb(n.snr_db)}</td>
                <td>
                  {effectiveStatus === 'detected' && n.threat_type ? (
                    <span className={`threat-tag ${n.threat_type}`}>{threatLabel(n.threat_type)}</span>
                  ) : (
                    <span className="fd-muted">{STATUS_TEXT[effectiveStatus] || STATUS_TEXT.heard_no_threat}</span>
                  )}
                </td>
                <td>
                  {effectiveStatus === 'too_faint' ? (
                    <span className="fd-muted">n/a</span>
                  ) : (
                    <div className="confidence-bar-container">
                      <div className="confidence-track" aria-hidden="true">
                        <div
                          className="confidence-fill"
                          style={{
                            width: percent(n.confidence || 0),
                            backgroundColor: n.heard ? 'var(--threat-chainsaw)' : 'var(--text-dim)',
                          }}
                        />
                      </div>
                      <span className="confidence-label" style={{ minWidth: '2.6rem' }}>{percent(n.confidence || 0)}</span>
                    </div>
                  )}
                </td>
                <td>
                  {n.alert_id ? (
                    <span className="fd-sent"><Check size={14} aria-hidden="true" /> Sent</span>
                  ) : n.heard ? (
                    <span className="fd-muted">Not sent</span>
                  ) : (
                    <span className="fd-muted">None</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function PeakClasses({ clip, modelIsReal }) {
  const peakWindow = clip.windows.find((w) => w.start_s === clip.verdict_at_source.at_s) ||
    [...clip.windows].sort((a, b) => b.confidence - a.confidence)[0];
  const classes = (peakWindow && peakWindow.top_classes) || [];
  if (classes.length === 0) return null;

  return (
    <div>
      <div className="fd-label">
        What the classifier heard at {formatSeconds(peakWindow.start_s)}
        {!modelIsReal && ' (heuristic scores)'}
      </div>
      <ul className="fd-classes">
        {classes.map((c) => (
          <li key={c.name}>
            <span>{c.name}</span>
            <div className="confidence-track" aria-hidden="true" style={{ width: '100%' }}>
              <div className="confidence-fill" style={{ width: percent(Math.min(1, c.score)), backgroundColor: 'var(--accent-cyan)' }} />
            </div>
            <span className="fd-mono">{percent(c.score)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

// ---- Main view -----------------------------------------------------------------------------
export default function ForestDrop({ active }) {
  const { nodes } = useTelemetry();
  const { scale, pixelsPerKm, toPx, toGeo } = useProjection(nodes);

  const svgRef = useRef(null);
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const mediaUrl = useMemo(() => (file ? URL.createObjectURL(file) : ''), [file]);
  const [point, setPoint] = useState(null);
  const [hover, setHover] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [loudness, setLoudness] = useState('normal');
  const [sendAlerts, setSendAlerts] = useState(true);
  const [phase, setPhase] = useState('idle'); // idle | running | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [range, setRange] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetchListeningRange().then((data) => {
      if (!cancelled) setRange(data);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!mediaUrl) return undefined;
    return () => URL.revokeObjectURL(mediaUrl);
  }, [mediaUrl]);

  // A file dropped outside the forest would make the browser open it. Stop that while this view is open.
  useEffect(() => {
    if (!active) return undefined;
    const block = (e) => e.preventDefault();
    window.addEventListener('dragover', block);
    window.addEventListener('drop', block);
    return () => {
      window.removeEventListener('dragover', block);
      window.removeEventListener('drop', block);
    };
  }, [active]);

  const pointFromEvent = useCallback(
    (e) => {
      const rect = svgRef.current.getBoundingClientRect();
      const x = clamp(((e.clientX - rect.left) / rect.width) * W, 0, W);
      const y = clamp(((e.clientY - rect.top) / rect.height) * H, 0, H);
      return { x, y, ...toGeo(x, y) };
    },
    [toGeo]
  );

  const run = useCallback(
    async (fileArg, pointArg) => {
      if (!fileArg || !pointArg) return;
      setPhase('running');
      setError('');
      setResult(null);
      try {
        const data = await dropAudioInForest({
          file: fileArg,
          nodes,
          sourceLat: pointArg.lat,
          sourceLon: pointArg.lon,
          loudness,
          sendAlerts,
        });
        setResult(data);
        setPhase('done');
      } catch (err) {
        setError(err.message);
        setPhase('error');
      }
    },
    [nodes, loudness, sendAlerts]
  );

  const acceptFile = (candidate) => {
    const problem = checkFile(candidate);
    if (problem) {
      setError(problem);
      setPhase('error');
      return false;
    }
    setFile(candidate);
    setResult(null);
    setError('');
    setPhase('idle');
    return true;
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    setHover(null);
    const dropped = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!dropped || phase === 'running') return;
    const where = pointFromEvent(e);
    if (acceptFile(dropped)) {
      setPoint(where);
      run(dropped, where);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragging(true);
    setHover(pointFromEvent(e));
  };

  const handleCanvasClick = (e) => {
    if (phase === 'running') return;
    setPoint(pointFromEvent(e));
    setResult(null);
    if (phase === 'done') setPhase('idle');
  };

  const handlePick = (e) => {
    const picked = e.target.files && e.target.files[0];
    e.target.value = '';
    if (picked) acceptFile(picked);
  };

  const resultById = useMemo(() => {
    const map = {};
    if (result) result.nodes.forEach((n) => { map[n.id] = n; });
    return map;
  }, [result]);

  const radiusM =
    (range && range.detection_radius_m && range.detection_radius_m[loudness]) ||
    DEFAULT_DETECTION_RADII_M[loudness] ||
    BASE_DETECTION_RADIUS_M;
  const running = phase === 'running';
  const canRun = Boolean(file && point) && !running;
  const isVideo = file && file.type.startsWith('video/');

  const nodeColor = (n) => {
    const r = resultById[n.id];
    if (!r) return 'var(--accent-cyan)';
    if (r.status === 'detected') return THREAT_COLOR[r.threat_type] || 'var(--threat-unknown)';
    if (r.status === 'heard_no_threat') return 'var(--accent-amber)';
    return 'var(--text-dim)';
  };

  return (
    <div hidden={!active} className="view-enter">
      <div className="fd-layout">
        <section className="glass-panel" aria-label="Forest">
          <div className="panel-header">
            <div>
              <div className="panel-title">Forest</div>
              <div className="panel-subtitle">
                Drop an audio or video file where the sound happens. Sensors nearby will listen to it.
              </div>
            </div>
          </div>

          <div className="fd-canvas-wrap" data-dragging={dragging}>
            {/* Visual Terrain Overlays (SVG): absolute-positioned behind sensor nodes, above forest background */}
            <svg
              className="fd-terrain-overlay"
              viewBox={`0 0 ${W} ${H}`}
              aria-hidden="true"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 1,
              }}
            >
              {/* Distinct, winding River flowing across the bottom third */}
              <path
                d="M -20 445 C 160 435, 260 430, 370 435 C 440 438, 500 440, 560 438 C 650 432, 740 435, 840 440 S 960 445, 1030 445"
                fill="none"
                stroke="#bae6fd"
                strokeWidth="40"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.6"
              />
              <path
                d="M -20 445 C 160 435, 260 430, 370 435 C 440 438, 500 440, 560 438 C 650 432, 740 435, 840 440 S 960 445, 1030 445"
                fill="none"
                stroke="#7dd3fc"
                strokeWidth="16"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.3"
              />
              {/* Dirt Trail / Road entering from top-left into center of forest (dead-end logging camp landing at north bank) */}
              <path
                d="M -20 60 C 80 120, 170 170, 240 215 C 310 260, 360 295, 410 325 C 445 350, 470 375, 495 398"
                fill="none"
                stroke="#78350f"
                strokeWidth="8"
                strokeDasharray="10, 10"
                strokeLinecap="round"
                opacity="0.7"
              />
            </svg>

            <svg
              ref={svgRef}
              className="fd-svg"
              viewBox={`0 0 ${W} ${H}`}
              role="img"
              aria-label="Top-down view of the forest with the sensor nodes. Click to place a sound, or drop a file on it."
              onClick={handleCanvasClick}
              onDragOver={handleDragOver}
              onDragLeave={() => { setDragging(false); setHover(null); }}
              onDrop={handleDrop}
              style={{ position: 'relative', zIndex: 2 }}
            >
              <rect width={W} height={H} fill="#eef3ea" />
              {CANOPY.map((t, i) => (
                <circle key={i} cx={t.x} cy={t.y} r={t.r} fill={CANOPY_FILL[t.tone]} />
              ))}

              {/* Visual Terrain Overlays: River and Logging Trail directly above canopy */}
              <g pointerEvents="none" className="fd-terrain-layer">
                {/* Winding River across the bottom third */}
                <path
                  d="M -20 445 C 160 435, 260 430, 370 435 C 440 438, 500 440, 560 438 C 650 432, 740 435, 840 440 S 960 445, 1030 445"
                  fill="none"
                  stroke="#bae6fd"
                  strokeWidth="40"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.6"
                />
                <path
                  d="M -20 445 C 160 435, 260 430, 370 435 C 440 438, 500 440, 560 438 C 650 432, 740 435, 840 440 S 960 445, 1030 445"
                  fill="none"
                  stroke="#7dd3fc"
                  strokeWidth="16"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.3"
                />
                {/* Dirt Trail / Road entering from top-left terminating at north bank logging landing */}
                <path
                  d="M -20 60 C 80 120, 170 170, 240 215 C 310 260, 360 295, 410 325 C 445 350, 470 375, 495 398"
                  fill="none"
                  stroke="#78350f"
                  strokeWidth="8"
                  strokeDasharray="10, 10"
                  strokeLinecap="round"
                  opacity="0.7"
                />
              </g>

              {/* Visual Helpers: Dashed listening circles around nodes (e.g. 2.2 km base detection range) */}
              {nodes.map((n) => {
                const p = toPx(n.lat, n.lon);
                return (
                  <circle
                    key={`range-${n.id}`}
                    cx={p.x}
                    cy={p.y}
                    r={radiusM * scale}
                    fill="none"
                    style={{ stroke: 'var(--accent-cyan)', opacity: 0.4 }}
                    strokeWidth="1.5"
                    strokeDasharray="5 6"
                  />
                );
              })}

              {point &&
                nodes.map((n) => {
                  const r = resultById[n.id];
                  const effectiveStatus = getNodeStatus(r);
                  if (!r || effectiveStatus === 'too_faint') return null;
                  const p = toPx(n.lat, n.lon);
                  return (
                    <line
                      key={`line-${n.id}`}
                      x1={point.x}
                      y1={point.y}
                      x2={p.x}
                      y2={p.y}
                      strokeWidth={effectiveStatus === 'detected' ? 2 : 1.5}
                      strokeDasharray={effectiveStatus === 'detected' ? undefined : '4 5'}
                      style={{ stroke: nodeColor(n), opacity: 0.8 }}
                    />
                  );
                })}

              {nodes.map((n) => {
                const p = toPx(n.lat, n.lon);
                const r = resultById[n.id];
                const isStrongest = result && result.strongest_node === n.id;
                const effectiveStatus = getNodeStatus(r);
                return (
                  <g key={n.id} className="fd-node-group" style={{ cursor: 'pointer', pointerEvents: 'auto' }}>
                    {isStrongest && (
                      <circle cx={p.x} cy={p.y} r="16" fill="none" strokeWidth="2.5" style={{ stroke: nodeColor(n) }} />
                    )}
                    <circle cx={p.x} cy={p.y} r="8" strokeWidth="3" style={{ fill: nodeColor(n), stroke: '#ffffff' }} />
                    <text
                      x={p.x}
                      y={p.y + 27}
                      textAnchor="middle"
                      fontSize="13"
                      fontWeight="600"
                      style={{ fill: 'var(--text-primary)', paintOrder: 'stroke', stroke: '#eef3ea', strokeWidth: 4 }}
                    >
                      {shortName(n.id)}
                    </text>
                    {r && effectiveStatus !== 'too_faint' && (
                      <text
                        x={p.x}
                        y={p.y + 43}
                        textAnchor="middle"
                        fontSize="12"
                        style={{ fill: 'var(--text-secondary)', paintOrder: 'stroke', stroke: '#eef3ea', strokeWidth: 4 }}
                      >
                        {percent(r.confidence || 0)}
                      </text>
                    )}
                  </g>
                );
              })}

              {dragging && hover && (
                <g pointerEvents="none">
                  <circle cx={hover.x} cy={hover.y} r="14" fill="none" strokeWidth="2" strokeDasharray="4 4" style={{ stroke: 'var(--text-primary)' }} />
                  <circle cx={hover.x} cy={hover.y} r="3" style={{ fill: 'var(--text-primary)' }} />
                </g>
              )}

              {point && (
                <g pointerEvents="none">
                  <circle className="fd-pulse" cx={point.x} cy={point.y} r="14" fill="none" strokeWidth="2.5" style={{ stroke: 'var(--threat-chainsaw)' }} />
                  <circle cx={point.x} cy={point.y} r="7" strokeWidth="3" style={{ fill: 'var(--threat-chainsaw)', stroke: '#ffffff' }} />
                  <text
                    x={point.x}
                    y={point.y - 16}
                    textAnchor="middle"
                    fontSize="12"
                    fontWeight="600"
                    style={{ fill: 'var(--text-primary)', paintOrder: 'stroke', stroke: '#eef3ea', strokeWidth: 4 }}
                  >
                    Sound
                  </text>
                </g>
              )}
            </svg>
          </div>

          <div className="fd-caption">
            {`Dashed circles show how far a sensor can pick out a ${loudness} sound (about ${formatDistance(radiusM)}). `}
            Distances use each sensor’s real position. How far sound carries is a simulated assumption, not a field
            measurement.
          </div>
        </section>

        <section className="glass-panel" aria-label="Your sound">
          <div className="panel-header">
            <div>
              <div className="panel-title">Your sound</div>
              <div className="panel-subtitle">Pure wildlife audio works best, with or without a threat in it</div>
            </div>
          </div>

          <div className="fd-body">
            <div>
              <input
                ref={fileInputRef}
                type="file"
                accept={`${ACCEPTED.join(',')},audio/*,video/*`}
                onChange={handlePick}
                hidden
                aria-label="Choose an audio or video file"
              />
              <button type="button" className="btn-secondary" onClick={() => fileInputRef.current.click()} disabled={running}>
                <Upload size={15} aria-hidden="true" /> {file ? 'Choose a different file' : 'Choose a file'}
              </button>
              <p className="fd-help">
                Or drag it onto the forest. mp3, wav, mp4 and similar, up to {MAX_MB} MB. The first 60 seconds are used.
              </p>
            </div>

            {file && (
              <div>
                <div className="fd-file" title={file.name}>
                  <span>{file.name}</span>
                  <span className="fd-mono" style={{ flex: 'none' }}>{(file.size / (1024 * 1024)).toFixed(1)} MB</span>
                </div>
                {mediaUrl &&
                  (isVideo ? (
                    <video className="fd-media" src={mediaUrl} controls style={{ marginTop: '0.6rem' }} />
                  ) : (
                    <audio className="fd-media" src={mediaUrl} controls style={{ marginTop: '0.6rem' }} />
                  ))}
              </div>
            )}

            <div>
              <span className="fd-label" id="fd-loudness-label">How loud is it at the source?</span>
              <div className="fd-segmented" role="group" aria-labelledby="fd-loudness-label">
                {LOUDNESS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    aria-pressed={loudness === option.id}
                    onClick={() => setLoudness(option.id)}
                    disabled={running}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
              <p className="fd-help">Louder sounds carry further, so more sensors can hear them.</p>
            </div>

            <label className="fd-check">
              <input type="checkbox" checked={sendAlerts} onChange={(e) => setSendAlerts(e.target.checked)} disabled={running} />
              <span>Send alerts to the dashboard for sensors that detect a threat</span>
            </label>

            <div className="fd-status" aria-live="polite">
              {point
                ? `Sound placed at ${formatCoord(point.lat, 'N', 'S')}, ${formatCoord(point.lon, 'E', 'W')}.`
                : 'Click the forest to place the sound, or drop a file straight onto it.'}
            </div>

            <button type="button" className="btn-primary" onClick={() => run(file, point)} disabled={!canRun}>
              {running ? (
                <>
                  <LoaderCircle size={16} className="fd-spin" aria-hidden="true" /> Listening…
                </>
              ) : phase === 'done' ? (
                'Run again'
              ) : (
                'Listen at this spot'
              )}
            </button>

            {phase === 'error' && (
              <div className="fd-error" role="alert">
                <TriangleAlert size={16} aria-hidden="true" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </section>
      </div>

      {result && phase === 'done' && (
        <>
          {!result.model_is_real_yamnet && (
            <div className="fd-warning" role="status">
              <TriangleAlert size={16} aria-hidden="true" />
              <span>
                Real YAMNet is not loaded, so these results come from a simple signal heuristic ({result.model}), not
                the neural network. Confidence will not behave like the real model.
              </span>
            </div>
          )}

          <div className="fd-results">
            <section className="glass-panel" aria-label="What the sensors heard">
              <div className="fd-verdict">
                <Verdict result={result} />
              </div>
              <NodeTable result={result} />
            </section>

            <section className="glass-panel" aria-label="The clip">
              <div className="panel-header">
                <div>
                  <div className="panel-title">The clip</div>
                  <div className="panel-subtitle">
                    {result.file_info.filename}, {formatSeconds(result.clip.duration_s)} analysed
                  </div>
                </div>
              </div>
              <div className="fd-timeline">
                <ClipTimeline clip={result.clip} />
                <PeakClasses clip={result.clip} modelIsReal={result.model_is_real_yamnet} />
                <p className="fd-help fd-note">
                  Model: {result.model}. Sensors below{' '}
                  {result.assumptions.minimum_snr_db_to_detect} dB (sound quieter than the background noise) cannot
                  pick the sound out. Sound drops about 6 dB each time the distance doubles, plus{' '}
                  {result.assumptions.forest_absorption_db_per_km} dB per km of forest.
                </p>
              </div>
            </section>
          </div>
        </>
      )}
    </div>
  );
}
