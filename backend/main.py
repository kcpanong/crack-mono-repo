import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.storage import InspectionRepository

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="storage"), name="static")

@app.get("/api/inspections")
async def list_inspections():
    return InspectionRepository.get_all_inspections()

@app.post("/api/upload")
async def upload_inspection(file: Optional[UploadFile] = File(None)):
    if file is None:
        return HTTPException(status_code=400, detail="No local file stream provided for processing.")
    
    unique_id = f"insp_{int(time.time())}"
    return InspectionRepository.save_new_inspection(file, unique_id)
    
@app.post("/api/analyze-session")
async def analyze_session(payload: dict = Body(...)):
    # Support a batch command from frontend to process all pending sessions
    if payload.get("runAllPending"):
        sessions = InspectionRepository.get_all_inspections()
        processed_count = 0
        for sess in sessions:
            s_id = sess.get("sessionId")
            originals = sess.get("originals", []) or []
            updated_originals = []
            any_processed = False
            for orig in originals:
                # skip already processed
                if orig.get("is_processed") or orig.get("mask_url"):
                    updated_originals.append(orig)
                    continue

                orig_id = orig.get("id")
                orig_url = orig.get("url") or orig.get("storageUrl") or orig.get("original_url")

                resized_variants = orig.get("resized_variants", []) or []
                if isinstance(resized_variants, list) and resized_variants:
                    variant_target = resized_variants[0]
                    resized_url = variant_target.get("url") or variant_target.get("storageUrl") or variant_target.get("original_url")
                else:
                    resized_url = orig_url

                if not resized_url:
                    orig["is_processed"] = False
                    updated_originals.append(orig)
                    continue

                # If an assessment already exists for this original, load it and skip work
                assessment_id = f"img_{orig_id}"
                try:
                    if InspectionRepository.assessment_exists(assessment_id):
                        try:
                            existing = InspectionRepository.load_assessment_record(assessment_id)
                            orig["mask_url"] = existing.get("masks3Url") or existing.get("masks3_url") or existing.get("mask_url")
                            orig["crack_data"] = existing.get("crack_data", {"bounding_boxes": [], "contours": []})
                            orig["is_processed"] = True
                            updated_originals.append(orig)
                            continue
                        except Exception as e:
                            print(f"Error loading existing assessment {assessment_id}: {e}")
                except Exception as e:
                    print(f"Error checking assessment existence for {assessment_id}: {e}")

                try:
                    result = InspectionRepository.process_cloud_session_images(
                        original_id=orig_id,
                        original_url=orig_url,
                        resized_url=resized_url
                    )
                    orig["mask_url"] = result.get("mask_url")
                    orig["crack_data"] = result.get("crack_data", {"bounding_boxes": [], "contours": []})
                    orig["is_processed"] = True

                    # persist an assessment record for this original
                    try:
                        assessment_payload = {
                            "id": f"img_{orig_id}",
                            "sessionId": s_id,
                            "type": orig.get("type", "original"),
                            "name": orig.get("name"),
                            "storageUrl": orig_url,
                            "masks3Url": orig.get("mask_url") or result.get("mask_url"),
                            "is_assessed": True,
                            "assessed_at": None,
                            "crack_data": {
                                "bounding_boxes": orig["crack_data"].get("bounding_boxes", []),
                                "contours": orig["crack_data"].get("contours", [])
                            }
                        }
                        InspectionRepository.save_assessment_record(assessment_payload)
                    except Exception as e:
                        print(f"Failed to save assessment for orig {orig_id}: {e}")

                    any_processed = True
                except Exception as e:
                    print(f"Failed to process asset {orig_id} during batch: {str(e)}")
                    orig["is_processed"] = False

                updated_originals.append(orig)

            if any_processed:
                processed_session = {
                    "sessionId": s_id,
                    "originals": updated_originals,
                    "is_processed_session": True
                }
                try:
                    InspectionRepository.save_session_metadata(processed_session)
                except Exception as e:
                    print(f"Failed saving session metadata for {s_id}: {e}")
                processed_count += 1

        return {"processed_sessions": processed_count}

    session_id = payload.get("sessionId")
    originals = payload.get("originals", [])

    if not session_id or not originals:
        raise HTTPException(status_code=400, detail="Missing sessionId or original assets list.")
    
    analyzed_originals = []

    for orig in originals:
        orig_id = orig.get("id")
        orig_url = orig.get("url") or orig.get("storageUrl") or orig.get("original_url")

        resized_variants = orig.get("resized_variants", []) or []
        if isinstance(resized_variants, list) and resized_variants:
            variant_target = resized_variants[0]
            resized_url = variant_target.get("url") or variant_target.get("storageUrl") or variant_target.get("original_url")
        else:
            resized_url = orig_url

        if not resized_url:
            print(f"Skipping original {orig_id} because no source URL is available.")
            orig["is_processed"] = False
            analyzed_originals.append(orig)
            continue

        # If an assessment already exists for this original, load it and skip work
        assessment_id = f"img_{orig_id}"
        try:
            if InspectionRepository.assessment_exists(assessment_id):
                try:
                    existing = InspectionRepository.load_assessment_record(assessment_id)
                    orig["mask_url"] = existing.get("masks3Url") or existing.get("masks3_url") or existing.get("mask_url")
                    orig["crack_data"] = existing.get("crack_data", {"bounding_boxes": [], "contours": []})
                    orig["is_processed"] = True
                    analyzed_originals.append(orig)
                    continue
                except Exception as e:
                    print(f"Error loading existing assessment {assessment_id}: {e}")
        except Exception as e:
            print(f"Error checking assessment existence for {assessment_id}: {e}")

        try:
            result = InspectionRepository.process_cloud_session_images(
                original_id=orig_id,
                original_url=orig_url,
                resized_url=resized_url
            )

            orig["mask_url"] = result.get("mask_url")
            orig["crack_data"] = result.get("crack_data", {"bounding_boxes": [], "contours": []})
            orig["is_processed"] = True
            # Create and persist assessment record (backend-owned)
            try:
                assessment_payload = {
                    "id": f"img_{orig_id}",
                    "sessionId": session_id,
                    "type": orig.get("type", "original"),
                    "name": orig.get("name"),
                    "storageUrl": orig_url,
                    "masks3Url": orig.get("mask_url") or result.get("mask_url"),
                    "is_assessed": True,
                    "assessed_at": None,
                    "crack_data": {
                        "bounding_boxes": orig["crack_data"].get("bounding_boxes", []),
                        "contours": orig["crack_data"].get("contours", [])
                    }
                }
                InspectionRepository.save_assessment_record(assessment_payload)
            except Exception as e:
                print(f"Failed to save assessment for orig {orig_id}: {e}")
        except Exception as e:
            print(f"Failed to process asset {orig_id}: {str(e)}")
            orig["is_processed"] = False

        analyzed_originals.append(orig)

    processed_session = {
        "sessionId": session_id,
        "originals": analyzed_originals,
        "is_processed_session": True
    }

    if hasattr(InspectionRepository, "save_session_metadata"):
        InspectionRepository.save_session_metadata(processed_session)

    return processed_session


@app.post("/api/assessments")
async def create_assessment(payload: dict = Body(...)):
    """Accept an assessment payload and persist it without re-running CV pipelines."""
    if not payload.get("id"):
        raise HTTPException(status_code=400, detail="Missing assessment id")

    try:
        saved = InspectionRepository.save_assessment_record(payload)
        return saved
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)