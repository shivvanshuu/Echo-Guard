"""
Audio Drop Simulation API
=========================
Drop a real audio or video clip at a point in the forest. Every sensor node
"hears" it according to its distance, and the nodes that detect a threat raise
real alerts through the normal alert pipeline.
"""
import json
import logging
import os
import tempfile
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.alerts import persist_and_broadcast_alert
from app.db.database import get_db
from app.schemas.payload import ThreatAlertCreate
from app.services.audio_classifier import get_yamnet_classifier
from app.services.audio_simulation import (
    LOUDNESS_OFFSET_DB,
    MIN_DETECTION_SNR_DB,
    SUPPORTED_EXTENSIONS,
    AudioDecodeError,
    decode_audio,
    detection_radius_m,
    get_source_level_spl,
    simulate_audio_drop,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/simulate", tags=["simulation"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_NODES = 50


class NodePosition(BaseModel):
    id: str = Field(..., min_length=1, max_length=50)
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    name: Optional[str] = Field(default=None, max_length=120)


_nodes_adapter = TypeAdapter(List[NodePosition])


async def _read_limited(upload: UploadFile) -> bytes:
    contents = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. The limit is 50 MB.")
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return contents


@router.get("/coverage")
async def get_listening_range(threat_type: Optional[str] = None):
    """How far a node can pick out a sound, for each loudness. Used to draw the ranges on the forest."""
    if not threat_type:
        radii = {
            "quiet": 1000.0,
            "normal": 2200.0,
            "loud": 5500.0,
        }
        return {
            "detection_radius_m": radii,
            "minimum_snr_db": MIN_DETECTION_SNR_DB,
            "threat_type": "CHAINSAW",
            "source_level_spl_db": get_source_level_spl(None),
            "note": "Simulated assumptions, not a measured field range.",
        }
    return {
        "detection_radius_m": {name: detection_radius_m(name, threat_type=threat_type) for name in LOUDNESS_OFFSET_DB},
        "minimum_snr_db": MIN_DETECTION_SNR_DB,
        "threat_type": threat_type,
        "source_level_spl_db": get_source_level_spl(threat_type),
        "note": "Simulated assumptions, not a measured field range.",
    }


@router.post("/audio-drop")
async def drop_audio_in_forest(
    file: UploadFile = File(...),
    nodes: str = Form(..., description="JSON list of sensors: [{id, lat, lon, name?}]"),
    source_lat: float = Form(..., ge=-90.0, le=90.0),
    source_lon: float = Form(..., ge=-180.0, le=180.0),
    loudness: str = Form("normal", description="quiet, normal or loud"),
    send_alerts: bool = Form(True, description="Store alerts for the nodes that detect a threat"),
    db: AsyncSession = Depends(get_db),
):
    filename = file.filename or ""
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Use one of: {', '.join(SUPPORTED_EXTENSIONS)}",
        )
    if loudness not in LOUDNESS_OFFSET_DB:
        raise HTTPException(status_code=422, detail=f"loudness must be one of {sorted(LOUDNESS_OFFSET_DB)}")

    try:
        node_list = [n.model_dump() for n in _nodes_adapter.validate_json(nodes)]
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=f"Invalid nodes list: {error.errors()[0]['msg']}")
    if not node_list:
        raise HTTPException(status_code=422, detail="Provide at least one sensor node.")
    if len(node_list) > MAX_NODES:
        raise HTTPException(status_code=422, detail=f"At most {MAX_NODES} nodes are supported.")

    contents = await _read_limited(file)

    suffix = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(contents)
        temp_path = handle.name

    try:
        try:
            waveform = await run_in_threadpool(decode_audio, temp_path)
        except AudioDecodeError as error:
            raise HTTPException(status_code=400, detail=str(error))

        classifier = get_yamnet_classifier()
        result, payloads = await run_in_threadpool(
            simulate_audio_drop, classifier, waveform, node_list, source_lat, source_lon, loudness
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    alert_ids = {}
    if send_alerts:
        for payload in payloads:
            saved = await persist_and_broadcast_alert(db, ThreatAlertCreate(**payload))
            alert_ids[payload["device_id"]] = str(saved.id)

    for node in result["nodes"]:
        node["alert_id"] = alert_ids.get(node["id"])
    result["alerts_sent"] = len(alert_ids)
    result["file_info"] = {"filename": filename, "size_bytes": len(contents)}

    logger.info(
        "Audio drop '%s': strongest node %s, %d alert(s) sent, model=%s",
        filename, result["strongest_node"], len(alert_ids), result["model"],
    )
    return result
