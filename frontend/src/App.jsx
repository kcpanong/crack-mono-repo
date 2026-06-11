import { useState, useRef, useEffect } from 'react';
import { Upload, Columns, Square, Folder } from 'lucide-react';

const INITIAL_DIRECTORY = [];
const API_BASE = import.meta.env.VITE_APP_URL || '';
const MASK_API = import.meta.env.VITE_MASK_API_URL || `${API_BASE}/api/analyze-session`;

export default function App() {
  const [loading, setLoading] = useState(false);
  const [inspections, setInspections] = useState(INITIAL_DIRECTORY);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [selectedOriginalIdx, setSelectedOriginalIdx] = useState(0);
  const [isDualWindow, setIsDualWindow] = useState(false);
  const [activeCrack, setActiveCrack] = useState(null);
  const activeSession = inspections[selectedIdx] || null;
  const activeInspection = activeSession?.originals?.[selectedOriginalIdx] || null;

  const computeGlobalStats = () => {
    if (!activeInspection || !activeInspection.crack_data?.bounding_boxes) {
      return { totalLength: "0.00mm", overallAvgWidth: "0.00mm", orientations: ["None"] };
    }

    const boxes = activeInspection.crack_data.bounding_boxes;
    if (boxes.length === 0) return { totalLength: "0.00mm", overallAvgWidth: "0.00mm", orientations: ["None"] };

    let runningTotalLength = 0;
    let runningWidthSum = 0;
    const uniqueOrientations = new Set();

    boxes.forEach(b => {
      const len = parseFloat(b.crackLength) || 0;
      const wid = parseFloat(b.avgWidth) || 0;

      runningTotalLength += len;
      runningWidthSum += wid;
      if (b.orientation) uniqueOrientations.add(b.orientation);
    });

    return {
      totalLength: `${runningTotalLength.toFixed(2)}mm`,
      overallAvgWidth: `${(runningWidthSum / boxes.length).toFixed(2)}mm`,
      orientations: uniqueOrientations.size > 0 ? Array.from(uniqueOrientations) : ["Mixed / Unknown"]
    };
  };

  const macroStats = computeGlobalStats();

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData
      });
      const newInspectionItem = await res.json();

      setInspections((prevList) => {
        const updatedList = [...prevList, newInspectionItem];
        setSelectedIdx(updatedList.length - 1);
        return updatedList;
      });
    } catch (err) {
      console.error("Error communicating with the backend", err);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzedSession = async (session) => {
    if (!session || loading) return;

    setLoading(true);
    try {
      // Force frontend -> backend orchestration. Backend will call the external
      // DAMAGE_MASK_API_URL (if configured) and return processed data.
      const res = await fetch(`${API_BASE}/api/analyze-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(session)
      });

      const updateSessionPayload = await res.json();

      setInspections((prevList) =>
        prevList.map((item) =>
          item.sessionId === session.sessionId ? updateSessionPayload : item
        )
      );
    } catch (err) {
      console.error("Error running session processing pipeline script", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchDirectoryHistory = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/inspections`);
        const json = await res.json();
        if (json && json.length > 0) {
          setInspections(json);
        }
      } catch (err) {
        console.error("Failed to read historical directory data node", err);
      }
    };
    fetchDirectoryHistory();
  }, []);

  // Session summary state and helpers (moved outside JSX)
  const [sessionSummaryText, setSessionSummaryText] = useState(null);

  const getOrientationLabel = (orientation) => {
    if (!orientation) return 'N/A';
    const s = String(orientation);
    if (/curve/i.test(s)) return `Curve (${s})`;
    const m = s.match(/(-?\d+(?:\.\d+)?)/);
    if (!m) return s;
    const deg = parseFloat(m[1]);
    if (Number.isNaN(deg)) return s;
    // Normalize to [0,180)
    const norm = ((deg % 180) + 180) % 180;
    const near = (target, tol = 15) => Math.abs(norm - target) <= tol;
    let label = 'Diagonal';
    if (near(0) || near(180)) label = 'Horizontal';
    else if (near(90)) label = 'Vertical';
    return `${label} (${deg.toFixed(1)}°)`;
  };

  const handleCreateAssessment = async (session, original) => {
    if (!session || !original) return;
    setLoading(true);
    try {
      const payload = {
        id: `img_${original.id}`,
        sessionId: session.sessionId,
        type: original.type || 'original',
        name: original.name || `img_${original.id}`,
        storageUrl: original.url,
        masks3Url: original.mask_url || null,
        is_assessed: true,
        assessed_at: new Date().toISOString(),
        crack_data: {
          bounding_boxes: original.crack_data?.bounding_boxes || [],
          contours: original.crack_data?.contours || []
        }
      };

      const res = await fetch(`${API_BASE}/api/assessments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const saved = await res.json();

      // update local state: attach assessment to the original
      setInspections(prev => prev.map(s => {
        if (s.sessionId !== session.sessionId) return s;
        return {
          ...s,
          originals: s.originals.map(o => o.id === original.id ? { ...o, crack_data: { ...(o.crack_data||{}), assessment: saved.crack_data?.assessment || {} } } : o)
        };
      }));
    } catch (e) {
      console.error('Failed to create assessment', e);
    } finally {
      setLoading(false);
    }
  };

  const generateSessionSummary = () => {
    if (!activeSession) return;
    const counts = { remedial: 0, months_1_2: 0, not_safe: 0 };
    const allBoxes = [];
    activeSession.originals?.forEach(o => {
      (o.crack_data?.bounding_boxes || []).forEach(b => allBoxes.push(b));
    });

    allBoxes.forEach(b => {
      const len = parseFloat(String(b.crackLength || '').split(' ')[0]) || 0;
      if (len > 20) counts.not_safe += 1;
      else if (len > 15) counts.months_1_2 += 1;
      else if (len > 10) counts.remedial += 1;
    });

    let message = 'No major cracks detected in this session.';
    if (counts.not_safe > 0) {
      message = `Warning: ${counts.not_safe} crack(s) exceed 2.0 cm — immediate professional evaluation recommended.`;
    } else if (counts.months_1_2 > 0) {
      message = `Notice: ${counts.months_1_2} crack(s) exceed 1.5 cm — consider repair within 1-2 months.`;
    } else if (counts.remedial > 0) {
      message = `Advisory: ${counts.remedial} crack(s) exceed 1.0 cm — remedial action recommended.`;
    }

    // Add a contextual natural language closing sentence
    if (counts.not_safe + counts.months_1_2 + counts.remedial > 0) {
      message += ' Based on the number of cracks at this level, please seek help from a professional before it\'s too late.';
    }

    setSessionSummaryText(message);
  };

  return (
    <div style={{ fontFamily: 'sans-serif', backgroundColor: '#0f172a', color: '#f8fafc', minHeight: '100vh', padding: '20px', display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #334155', paddingBottom: '15px', marginBottom: '20px' }}>
        <div>
          <h1 style={{ margin: 0, fontSize: '1.5rem', color: 'white' }}>Crack Inspection Workspace</h1>
        </div>

        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <button
            title="Upload disabled in this build"
            style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#334155', border: '1px solid #475569', color: '#fff', padding: '8px 16px', borderRadius: '6px', cursor: 'not-allowed' }}
            disabled
          >
            <Upload size={18} />
            Upload Disabled
          </button>

          {inspections.length > 0 && (
            <>
              <button
                onClick={() => setIsDualWindow(!isDualWindow)}
                style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#334155', border: '1px solid #475569', color: '#fff', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer' }}
              >
                {isDualWindow ? <Square size={18} /> : <Columns size={18} />}
                {isDualWindow ? 'Single Window' : 'Compare Side-by-Side'}
              </button>

              <button
                onClick={() => generateSessionSummary()}
                style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: '#064e3b', border: '1px solid #065f46', color: '#fff', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer' }}
              >
                Session Summary
              </button>
            </>
          )}

          
        </div>
      </header>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <aside style={{ width: '260px', borderRight: '1px solid #334155', backgroundColor: '#0f172a', padding: '15px', overflow: 'auto' }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {inspections.map((session, sIdx) => {
              const isSessionAnalyzed = session.is_processed_session || session.originals?.some(o => o.mask_url);

              return (
                <div key={session.sessionId || sIdx} style={{ display: 'flex', flexDirection: 'column', gap: '8px', borderBottom: '1px solid #1e293b', paddingBottom: '12px' }}>

                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 8px', color: '#64748b', fontSize: '0.75rem', fontWeight: 'bold' }}>
                      <Folder size={14} />
                      <span>SESSION: {session.sessionId}</span>
                    </div>
                    <span
                      title={isSessionAnalyzed ? "Processed" : "Pending Analysis"}
                      style={{ fontSize: '0.6rem', color: isSessionAnalyzed ? '#10b981' : '#f59e0b' }}
                    >
                      {isSessionAnalyzed ? '● PROCESSED' : '○ PENDING'}
                    </span>
                  </div>

                  {!isSessionAnalyzed && (
                    <button
                      onClick={() => handleAnalyzedSession(session)}
                      disabled={loading}
                      style={{
                        backgroundColor: '#0284c7', color: '#fff', border: 'none', borderRadius: '4px',
                        padding: '5px 8px', fontSize: '0.7rem', fontWeight: 'bold', cursor: loading ? 'not-allowed' : 'pointer',
                        transition: 'background-color 0.2s', width: '100%', marginBottom: '4px'
                      }}
                    >
                      {loading ? 'Processing Workspace...' : 'Run CV Analysis'}
                    </button>
                  )}

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', paddingLeft: '12px', borderLeft: '1px solid #1e293b' }}>
                    {session.originals?.map((orig, oIdx) => {
                      const isSelected = selectedIdx === sIdx && selectedOriginalIdx === oIdx;
                      return (
                        <button
                          key={orig.id}
                          onClick={() => {
                            setSelectedIdx(sIdx);
                            setSelectedOriginalIdx(oIdx);
                          }}
                          style={{
                            width: '100%', padding: '8px 10px', borderRadius: '6px', border: 'none', textAlign: 'left', cursor: 'pointer',
                            backgroundColor: isSelected ? '#2563eb' : '#1e293b',
                            color: '#fff', transition: 'background-color 0.15s', fontSize: '0.8rem'
                          }}
                        >
                          <div style={{ fontWeight: '500', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {orig.name || `Image #${orig.id.slice(0, 5)}`}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </aside>

        <main style={{ flex: 1, display: 'grid', gridTemplateColumns: isDualWindow ? '1fr 1fr' : '1fr', gap: '1px', backgroundColor: '#334155' }}>
          {activeInspection ? (
            <>
              <InteractiveWindow
                key={`alpha-${activeInspection?.id}`}
                title="Viewport Alpha"
                inspection={{
                  id: activeInspection.id,
                  original_url: activeInspection.url,
                  mask_url: activeInspection.mask_url,
                  crack_data: activeInspection.crack_data || { bounding_boxes: [], contours: [] }
                }}
                onCrackSelect={setActiveCrack}
              />
              {isDualWindow && (
                <InteractiveWindow
                  key={`beta-${activeInspection?.id}`}
                  title="Viewport Beta"
                  inspection={{
                    id: activeInspection.id,
                    original_url: activeInspection.url,
                    mask_url: activeInspection.mask_url,
                    crack_data: activeInspection.crack_data || { bounding_boxes: [], contours: [] }
                  }}
                  onCrackSelect={setActiveCrack}
                />
              )}
            </>
          ) : (
            <div style={{ display: 'flex', flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#090d16', color: '#64748b', gridColumn: '1 / -1', height: '100%' }}>
              No inspections loaded. Use the backend or import a session to populate inspections.
            </div>
          )}
        </main>

        <aside style={{ width: '320px', borderLeft: '1px solid #334155', backgroundColor: '#0f172a', padding: '15px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <h3 style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#64748b', letterSpacing: '0.05em', margin: '0 0 12px 0' }}>Assessment Summary</h3>

            {activeInspection ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
                  <div style={{ flex: 1, backgroundColor: '#1e293b', padding: '12px', borderRadius: '6px', border: '1px solid #334155' }}>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase' }}>Cracks</div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 'bold', color: '#38bdf8' }}>{(activeInspection.crack_data?.bounding_boxes || []).length}</div>
                  </div>
                  <div style={{ width: '120px', backgroundColor: '#1e293b', padding: '12px', borderRadius: '6px', border: '1px solid #334155', textAlign: 'center' }}>
                    <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px', textTransform: 'uppercase' }}>Severity</div>
                    {(() => {
                      const boxes = (activeInspection.crack_data?.bounding_boxes || []);
                      let maxLen = 0;
                      boxes.forEach(b => { const l = parseFloat(String(b.crackLength||'').split(' ')[0])||0; if (l>maxLen) maxLen = l; });
                      if (maxLen > 20) return <div style={{ color: '#ef4444', fontWeight: '700' }}>Not Safe</div>;
                      if (maxLen > 15) return <div style={{ color: '#f59e0b', fontWeight: '700' }}>1-2 Months</div>;
                      if (maxLen > 10) return <div style={{ color: '#facc15', fontWeight: '700' }}>Remedial</div>;
                      return <div style={{ color: '#10b981', fontWeight: '700' }}>Stable</div>;
                    })()}
                  </div>
                </div>

                <div style={{ display: 'flex', gap: '8px' }}>
                  <button onClick={() => handleCreateAssessment(activeSession, activeInspection)} style={{ flex: 1, backgroundColor: '#2563eb', color: '#fff', padding: '10px', borderRadius: '6px', border: 'none', cursor: 'pointer' }}>Save Assessment</button>
                  <button onClick={() => generateSessionSummary()} style={{ backgroundColor: '#475569', color: '#fff', padding: '10px', borderRadius: '6px', border: 'none', cursor: 'pointer' }}>Generate Summary</button>
                </div>

                <div style={{ backgroundColor: '#0b1220', padding: '12px', borderRadius: '6px', border: '1px solid #1f2937' }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '8px', textTransform: 'uppercase' }}>Session Message</div>
                  <div style={{ color: '#cbd5e1', fontSize: '0.9rem' }}>{sessionSummaryText || 'No summary generated yet. Click "Session Summary" to create one.'}</div>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: '#64748b', fontSize: '0.85rem', marginTop: '20px' }}>
                No inspection selected. Select a session and image to view assessment.
              </div>
            )}
          </div>
        </aside>
      </div>

      {activeCrack && (
        <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', backgroundColor: 'rgba(0,0,0,0.6)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 9999 }} onClick={() => setActiveCrack(null)}>
          <div style={{ backgroundColor: '#1e293b', border: '1px solid #475569', padding: '20px', borderRadius: '8px', minWidth: '280px' }} onClick={e => e.stopPropagation()}>
            <h4 style={{ margin: '0 0 10px 0', borderBottom: '1px solid #334155', paddingBottom: '6px' }}>Crack ID: #{activeCrack.id}</h4>
            <p style={{ margin: '6px 0' }}><strong>Length:</strong> {activeCrack.crackLength}</p>
            <p style={{ margin: '6px 0' }}><strong>Avg. Width:</strong> {activeCrack.avgWidth}</p>
            <p style={{ margin: '6px 0' }}><strong>Max. Width:</strong> {activeCrack.maxWidth || "N/A"}</p>
            <p style={{ margin: '6px 0' }}><strong>Orientation:</strong> {getOrientationLabel(activeCrack.orientation) || "N/A"}</p>
            <button onClick={() => setActiveCrack(null)} style={{ marginTop: '12px', width: '100%', padding: '6px', backgroundColor: '#ef4444', border: 'none', color: '#fff', borderRadius: '4px', cursor: 'pointer' }}>Dismiss</button>
          </div>
        </div>
      )}
    </div>
  );
}

function InteractiveWindow({ title, inspection, onCrackSelect }) {
  const [maskMode, setMaskMode] = useState("photo");
  const [showBoxes, setShowBoxes] = useState(true);
  const [showContours, setShowContours] = useState(true);

  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);

  const dragStart = useRef({ x: 0, y: 0 });
  const viewPortRef = useRef(null);

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomFactor = 0.15;
    const direction = e.deltaY < 0 ? 1 : -1;

    setScale((prevScale) => {
      const nextScale = prevScale + direction * zoomFactor;
      return Math.min(Math.max(nextScale, 0.4), 6.0);
    });
  }

  useEffect(() => {
    const viewportNode = viewPortRef.current;
    if (!viewportNode) return;
    viewportNode.addEventListener('wheel', handleWheel, { passive: false });

    return () => {
      viewportNode.removeEventListener('wheel', handleWheel);
    };
  }, [scale]);

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    dragStart.current = { x: e.clientX - position.x, y: e.clientY - position.y };
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragStart.current.x,
      y: e.clientY - dragStart.current.y
    });
  };

  const handleMouseUpOrLeave = () => {
    setIsDragging(false);
  };

  return (
    <div style={{ backgroundColor: '#090d16', borderRadius: '8px', border: '1px solid #334155', display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 12px', backgroundColor: '#131b2e', borderBottom: '1px solid #1e293b', zIndex: 10 }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 'bold', color: '#94a3b8' }}>{title}</span>
        <div style={{ display: 'flex', backgroundColor: '#1e293b', borderRadius: '6px', padding: '2px', gap: '6px' }}>
          <div style={{ display: 'flex', gap: '2px', backgroundColor: '#0f172a', padding: '2px', borderRadius: '4px' }}>
            {['photo', 'overlay', 'mask'].map((mode) => (
              <button
                key={mode}
                onClick={() => setMaskMode(mode)}
                style={{
                  border: 'none',
                  backgroundColor: maskMode === mode ? '#2563eb' : 'transparent',
                  color: maskMode === mode ? '#fff' : '#64748b',
                  padding: '3px 6px',
                  borderRadius: '3px',
                  fontSize: '0.65rem',
                  textTransform: 'capitalize',
                  cursor: 'pointer'
                }}
              >
                {mode}
              </button>
            ))}
          </div>
          <ToggleButton active={showBoxes} onClick={() => setShowBoxes(!showBoxes)} text="Boxes" />
          <ToggleButton active={showContours} onClick={() => setShowContours(!showContours)} text="Contours" />
        </div>
      </div>

      <div
        ref={viewPortRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUpOrLeave}
        onMouseLeave={handleMouseUpOrLeave}
        style={{
          flex: 1, overflow: 'hidden', position: 'relative', cursor: isDragging ? 'grabbing' : 'grab',
          display: 'flex', justifyContent: 'center', alignItems: 'center'
        }}
      >
        <div
          key={inspection.id}
          style={{
            position: 'relative',
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            userSelect: 'none',
            maxWidth: '90%',
            maxHeight: '90%'
          }}
        >
          <img
            src={inspection.original_url}
            alt="Original asset"
            draggable="false"
            style={{ display: 'block', maxWidth: '100%', height: 'auto', maxHeight: '70vh', pointerEvents: 'none', opacity: maskMode === "mask" ? 0.0 : (maskMode === "overlay" ? 0.4 : 1.0), transition: 'opacity 0.15s ease' }}
          />
          {(maskMode === "overlay" || maskMode === "mask") && inspection.mask_url && (
            <img
              src={inspection.mask_url}
              alt="Binary Crack Mask"
              draggable="false"
              style={{
                position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                pointerEvents: 'none'
              }}
            />
          )}

          {(showBoxes || showContours) && (
            <svg
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              {showContours && inspection.crack_data.contours.map(c => (
                <path
                  key={c.id}
                  d={c.path}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth="1.2"
                  strokeLinecap="round"
                />
              ))}
              {showBoxes && inspection.crack_data?.bounding_boxes?.map(b => (
                <rect
                  key={b.id}
                  x={b.x}
                  y={b.y}
                  width={b.width}
                  height={b.height}
                  fill="rgba(239, 68, 68, 0.1)"
                  stroke="#ef4444"
                  strokeWidth="0.8"
                  style={{ cursor: 'pointer', pointerEvents: 'auto' }}
                  onClick={(e) => { e.stopPropagation(); onCrackSelect(b); }}
                />
              ))}
            </svg>
          )}
          <div style={{ position: 'absolute', bottom: '8px', right: '12px', fontSize: '0.7rem', color: '#64748b', background: '#090d16cc', padding: '2px 6px', borderRadius: '4px', zIndex: 5 }}>
            Zoom: {Math.round(scale * 100)}%
          </div>
        </div>
      </div>
    </div>
  );
}

function ToggleButton({ active, onClick, text }) {
  return (
    <button
      onClick={onClick}
      style={{
        border: 'none',
        backgroundColor: active ? '#2563eb' : '#1e293b',
        color: active ? '#fff' : '#64748b',
        padding: '4px 8px',
        borderRadius: '4px',
        fontSize: '0.7rem',
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        gap: '4px'
      }}
    >
      <span style={{ fontSize: '0.6rem', opacity: 0.7 }}>{active ? '●' : '○'}</span>
      {text}
    </button>
  );
}