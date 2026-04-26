"""
Palm Biometric REST API — Frontend-Ready Edition
================================================

Frontend integration features added:
- CORS enabled (for cross-origin requests from web frontends)
- Base64 image support (alternative to multipart file upload — easier from JS)
- Structured JSON responses with consistent format
- Detailed error codes for frontend handling
- Image preview/thumbnail returned in responses for UI feedback
- Confidence percentage (0-100) instead of raw cosine
- All endpoints support both file upload and base64 input
- OPTIONS preflight handled correctly
- Request ID tracking for debugging
- Detailed success/failure status

ENDPOINTS:
    GET    /                          — API root, returns endpoint list
    GET    /health                    — Quick health check
    GET    /info                      — Full system info
    
    POST   /enroll/full               — 3 RGB + 3 IR (best, 92-95%)
    POST   /enroll/rgb_only           — 3 RGB only (90-93%)
    POST   /enroll/quick              — 1 RGB + 1 IR (85-90%)
    POST   /enroll/quick_rgb          — 1 RGB only (82-88%)
    
    POST   /identify                  — 1 RGB + 1 IR
    POST   /identify/rgb_only         — 1 RGB only
    POST   /verify                    — 1-to-1 with claimed name
    
    GET    /enrolled                  — List + count
    POST   /enrolled/remove           — Remove one
    DELETE /enrolled/clear            — Clear all (requires confirm)

INPUT FORMATS:
  Each endpoint accepts EITHER:
    1. multipart/form-data with file uploads (rgb, ir, rgb_1, etc.)
    2. application/json with base64-encoded images (rgb_b64, ir_b64, etc.)

CORS: Allowed for all origins by default. Configure CORS_ALLOWED_ORIGINS for production.
"""

import os
import sys
import logging
import tempfile
import base64
import binascii
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================
class APIConfig:
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    
    MODEL_PATH = 'models/best_model.pth'
    DATABASE_PATH = 'enrolled_database/database.pkl'
    DEVICE = 'cpu'
    
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'webp'}
    
    UPLOAD_DIR = Path(tempfile.gettempdir()) / 'palm_api_uploads'
    
    # CORS — set to specific domains in production:
    # CORS_ALLOWED_ORIGINS = ['https://your-frontend.com', 'http://localhost:3000']
    CORS_ALLOWED_ORIGINS = '*'  # Allow all for development
    

# ============================================================
# FLASK APP WITH CORS
# ============================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = APIConfig.MAX_CONTENT_LENGTH

# Enable CORS for all routes — required for frontend access
CORS(
    app,
    resources={r"/*": {"origins": APIConfig.CORS_ALLOWED_ORIGINS}},
    methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allow_headers=['Content-Type', 'Authorization', 'X-Request-ID'],
    expose_headers=['X-Request-ID'],
    supports_credentials=False,
)


_palm_system = None
_enroller = None


# ============================================================
# REQUEST LOGGING / ID TRACKING
# ============================================================
@app.before_request
def add_request_id():
    """Add unique ID to each request for tracing in logs."""
    g.request_id = request.headers.get('X-Request-ID') or str(uuid.uuid4())[:8]
    if request.method != 'OPTIONS':
        logger.info(f"[{g.request_id}] {request.method} {request.path}")


@app.after_request
def add_response_headers(response):
    """Add request ID and CORS headers to all responses."""
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    return response


# ============================================================
# RESPONSE HELPERS
# ============================================================
def success_response(data: dict, status_code: int = 200):
    """Standard success response format."""
    return jsonify({
        'success': True,
        'request_id': g.request_id,
        'timestamp': datetime.now().isoformat(),
        **data,
    }), status_code


def error_response(message: str, code: str = 'ERROR', status_code: int = 400, **extra):
    """Standard error response format."""
    return jsonify({
        'success': False,
        'request_id': g.request_id,
        'error_code': code,
        'error': message,
        'timestamp': datetime.now().isoformat(),
        **extra,
    }), status_code


# ============================================================
# THUMBNAIL GENERATOR
# ============================================================
def _make_thumbnail(image_path: str, size: int = 120) -> Optional[str]:
    """
    Create a small base64 thumbnail from an image file.
    Returns base64 string like 'data:image/jpeg;base64,...'
    or None if failed.
    Size: thumbnail will be size x size pixels (square crop).
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]

        # Square center crop first
        s = min(h, w)
        y0 = (h - s) // 2
        x0 = (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]

        # Resize to thumbnail size
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        # Encode to JPEG bytes
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 70]
        _, buffer = cv2.imencode('.jpg', img, encode_param)

        # Convert to base64
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"

    except Exception:
        return None


# ============================================================
# IMAGE INPUT HANDLING
# ============================================================
def _allowed_file(filename: str) -> bool:
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in APIConfig.ALLOWED_EXTENSIONS


def _save_file_upload(file_storage, prefix: str = "upload") -> Optional[str]:
    """Save a multipart file upload to disk."""
    if file_storage is None or not file_storage.filename:
        return None
    if not _allowed_file(file_storage.filename):
        return None
    
    APIConfig.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file_storage.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    save_path = APIConfig.UPLOAD_DIR / f"{prefix}_{g.request_id}_{timestamp}_{filename}"
    file_storage.save(str(save_path))
    return str(save_path)


def _save_base64(b64_string: str, prefix: str = "upload") -> Optional[str]:
    """Decode and save a base64-encoded image."""
    if not b64_string:
        return None
    
    # Strip data URL prefix if present (e.g., "data:image/jpeg;base64,...")
    if ',' in b64_string:
        b64_string = b64_string.split(',', 1)[1]
    
    try:
        img_bytes = base64.b64decode(b64_string, validate=True)
    except (binascii.Error, ValueError):
        return None
    
    # Verify it's a valid image
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    APIConfig.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    save_path = APIConfig.UPLOAD_DIR / f"{prefix}_{g.request_id}_{timestamp}.jpg"
    cv2.imwrite(str(save_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return str(save_path)


def _get_image_input(field_name: str, prefix: str = "upload") -> Optional[str]:
    """
    Unified image input handler. Tries:
      1. Multipart file upload
      2. Base64 in JSON body  (field_name + '_b64')
      3. Base64 in form data
    
    Returns: file path on disk, or None
    """
    # Try multipart first
    if request.files and field_name in request.files:
        return _save_file_upload(request.files[field_name], prefix=prefix)
    
    # Try JSON body for base64
    json_data = request.get_json(silent=True) or {}
    b64_data = json_data.get(f'{field_name}_b64') or json_data.get(field_name)
    if b64_data and isinstance(b64_data, str):
        return _save_base64(b64_data, prefix=prefix)
    
    # Try form data for base64
    b64_form = request.form.get(f'{field_name}_b64')
    if b64_form:
        return _save_base64(b64_form, prefix=prefix)
    
    return None


def _get_string_input(field_name: str) -> Optional[str]:
    """Get a string field from JSON or form data."""
    if request.is_json:
        json_data = request.get_json(silent=True) or {}
        val = json_data.get(field_name)
        if val:
            return str(val).strip()
    
    val = request.form.get(field_name)
    if val:
        return val.strip()
    
    return None


def _cleanup_files(paths):
    for p in paths:
        if p:
            try:
                os.remove(p)
            except Exception:
                pass


# ============================================================
# CONFIDENCE CONVERSION
# ============================================================
def _confidence_to_percent(cosine_sim: float) -> float:
    """Convert cosine similarity (0-1) to confidence percentage (0-100)."""
    return round(max(0.0, min(1.0, float(cosine_sim))) * 100, 2)


# ============================================================
# INITIALIZATION
# ============================================================
def initialize_system():
    """Load model once at startup."""
    global _palm_system, _enroller
    
    from identify import PalmBiometricSystem
    from enroll import FlexibleEnroller
    
    if not Path(APIConfig.MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model not found at {APIConfig.MODEL_PATH}. "
            f"Train the model first: python3 train.py"
        )
    
    logger.info("Loading palm biometric system...")
    _palm_system = PalmBiometricSystem(
        model_path=APIConfig.MODEL_PATH,
        database_path=APIConfig.DATABASE_PATH,
        device=APIConfig.DEVICE,
    )
    _enroller = FlexibleEnroller(_palm_system)
    
    logger.info(f"System ready. {_palm_system.num_enrolled()} people enrolled.")


# ============================================================
# ROOT / HEALTH / INFO
# ============================================================
@app.route('/', methods=['GET'])
def root():
    return success_response({
        'service': 'Palm Biometric API',
        'version': '2.0',
        'status': 'running',
        'endpoints': {
            'health': 'GET /health',
            'info': 'GET /info',
            'enrollment': {
                'full': 'POST /enroll/full',
                'rgb_only': 'POST /enroll/rgb_only',
                'quick': 'POST /enroll/quick',
                'quick_rgb': 'POST /enroll/quick_rgb',
            },
            'identification': {
                'identify': 'POST /identify',
                'identify_rgb_only': 'POST /identify/rgb_only',
                'verify': 'POST /verify',
            },
            'database': {
                'list': 'GET /enrolled',
                'remove': 'POST /enrolled/remove',
                'clear': 'DELETE /enrolled/clear',
            },
        },
    })


@app.route('/health', methods=['GET'])
def health():
    return success_response({
        'status': 'healthy',
        'enrolled_count': _palm_system.num_enrolled() if _palm_system else 0,
        'model_loaded': _palm_system is not None,
    })


@app.route('/info', methods=['GET'])
def info():
    return success_response({
        'model_path': APIConfig.MODEL_PATH,
        'enrolled_count': _palm_system.num_enrolled(),
        'match_threshold': float(_palm_system.match_threshold),
        'device': APIConfig.DEVICE,
        'max_upload_mb': APIConfig.MAX_CONTENT_LENGTH // (1024 * 1024),
        'allowed_extensions': sorted(list(APIConfig.ALLOWED_EXTENSIONS)),
        'supports': {
            'multipart_upload': True,
            'base64_input': True,
            'cors': True,
        },
    })


# ============================================================
# ENROLLMENT ENDPOINTS
# ============================================================
@app.route('/enroll/full', methods=['POST'])
def enroll_full():
    """
    Best-accuracy enrollment: 3 RGB + 3 IR images.
    
    Multipart form fields:
        name (str), rgb_1, rgb_2, rgb_3, ir_1, ir_2, ir_3 (files)
        overwrite (str, optional): 'true' or 'false'
    
    OR JSON body with base64:
        {
            "name": "...",
            "rgb_1_b64": "data:image/jpeg;base64,...",
            "rgb_2_b64": "...",
            "rgb_3_b64": "...",
            "ir_1_b64": "...",
            "ir_2_b64": "...",
            "ir_3_b64": "...",
            "overwrite": false
        }
    """
    saved_files = []
    try:
        name = _get_string_input('name')
        if not name:
            return error_response('Missing required field: name', 'MISSING_NAME', 400)
        
        overwrite = (_get_string_input('overwrite') or 'false').lower() == 'true'
        
        rgb_paths = []
        ir_paths = []
        for i in [1, 2, 3]:
            rgb = _get_image_input(f'rgb_{i}', prefix=f'enroll_{name}_rgb{i}')
            ir = _get_image_input(f'ir_{i}', prefix=f'enroll_{name}_ir{i}')
            if not rgb:
                return error_response(
                    f'Missing or invalid image: rgb_{i}',
                    'MISSING_IMAGE',
                    400,
                    missing_field=f'rgb_{i}'
                )
            if not ir:
                return error_response(
                    f'Missing or invalid image: ir_{i}',
                    'MISSING_IMAGE',
                    400,
                    missing_field=f'ir_{i}'
                )
            rgb_paths.append(rgb)
            ir_paths.append(ir)
        
        saved_files = rgb_paths + ir_paths
        
        # Check duplicate
        if not overwrite and _palm_system.is_enrolled(name):
            return error_response(
                f'{name} is already enrolled. Set overwrite=true to replace.',
                'ALREADY_ENROLLED',
                409,
                name=name
            )
        
        success = _enroller.enroll_full(name, rgb_paths, ir_paths, overwrite=True)
        
        if success:
            # Generate thumbnail from first RGB image
            thumbnail = _make_thumbnail(rgb_paths[0]) if rgb_paths else None
            return success_response({
                'mode': 'full',
                'name': name,
                'images_used': 3,
                'enrolled_count': _palm_system.num_enrolled(),
                'expected_accuracy': '92-95% for new people, 97-99% for trained',
                'thumbnail': thumbnail,
            })
        else:
            return error_response(
                'Enrollment failed. Hand may not be detectable in images.',
                'ENROLLMENT_FAILED',
                500
            )
    
    except Exception as e:
        logger.exception(f"[{g.request_id}] Enrollment error")
        return error_response(str(e), 'INTERNAL_ERROR', 500)
    
    finally:
        _cleanup_files(saved_files)


@app.route('/enroll/rgb_only', methods=['POST'])
def enroll_rgb_only():
    """3 RGB images, IR auto-generated. For when no IR camera available."""
    saved_files = []
    try:
        name = _get_string_input('name')
        if not name:
            return error_response('Missing required field: name', 'MISSING_NAME', 400)
        
        overwrite = (_get_string_input('overwrite') or 'false').lower() == 'true'
        
        rgb_paths = []
        for i in [1, 2, 3]:
            rgb = _get_image_input(f'rgb_{i}', prefix=f'enroll_{name}_rgb{i}')
            if not rgb:
                return error_response(
                    f'Missing or invalid image: rgb_{i}',
                    'MISSING_IMAGE',
                    400,
                    missing_field=f'rgb_{i}'
                )
            rgb_paths.append(rgb)
        
        saved_files = rgb_paths
        
        if not overwrite and _palm_system.is_enrolled(name):
            return error_response(
                f'{name} already enrolled', 'ALREADY_ENROLLED', 409, name=name
            )
        
        success = _enroller.enroll_rgb_only(name, rgb_paths, overwrite=True)
        
        if success:
            thumbnail = _make_thumbnail(rgb_paths[0]) if rgb_paths else None
            return success_response({
                'mode': 'rgb_only',
                'name': name,
                'images_used': 3,
                'ir_generated': True,
                'enrolled_count': _palm_system.num_enrolled(),
                'expected_accuracy': '90-93% for new people, 95-97% for trained',
                'thumbnail': thumbnail,
            })
        else:
            return error_response('Enrollment failed', 'ENROLLMENT_FAILED', 500)
    
    except Exception as e:
        logger.exception(f"[{g.request_id}] Enrollment error")
        return error_response(str(e), 'INTERNAL_ERROR', 500)
    
    finally:
        _cleanup_files(saved_files)


@app.route('/enroll/quick', methods=['POST'])
def enroll_quick():
    """Quick: 1 RGB + 1 IR. Best for already-trained people."""
    saved_files = []
    try:
        name = _get_string_input('name')
        if not name:
            return error_response('Missing field: name', 'MISSING_NAME', 400)
        
        overwrite = (_get_string_input('overwrite') or 'false').lower() == 'true'
        
        rgb_path = _get_image_input('rgb', prefix=f'enroll_{name}_rgb')
        ir_path = _get_image_input('ir', prefix=f'enroll_{name}_ir')
        
        if not rgb_path:
            return error_response('Missing or invalid: rgb', 'MISSING_IMAGE', 400, missing_field='rgb')
        if not ir_path:
            return error_response('Missing or invalid: ir', 'MISSING_IMAGE', 400, missing_field='ir')
        
        saved_files = [rgb_path, ir_path]
        
        if not overwrite and _palm_system.is_enrolled(name):
            return error_response(
                f'{name} already enrolled', 'ALREADY_ENROLLED', 409, name=name
            )
        
        success = _enroller.enroll_quick(name, rgb_path, ir_path, overwrite=True)
        
        if success:
            thumbnail = _make_thumbnail(rgb_path) if rgb_path else None
            return success_response({
                'mode': 'quick',
                'name': name,
                'images_used': 1,
                'enrolled_count': _palm_system.num_enrolled(),
                'expected_accuracy': '85-90% for new people, 95-98% for trained',
                'thumbnail': thumbnail,
            })
        else:
            return error_response('Enrollment failed', 'ENROLLMENT_FAILED', 500)
    
    except Exception as e:
        logger.exception(f"[{g.request_id}] Enrollment error")
        return error_response(str(e), 'INTERNAL_ERROR', 500)
    
    finally:
        _cleanup_files(saved_files)


@app.route('/enroll/quick_rgb', methods=['POST'])
def enroll_quick_rgb():
    """Fastest: 1 RGB only, IR auto-generated."""
    saved_files = []
    try:
        name = _get_string_input('name')
        if not name:
            return error_response('Missing field: name', 'MISSING_NAME', 400)
        
        overwrite = (_get_string_input('overwrite') or 'false').lower() == 'true'
        
        rgb_path = _get_image_input('rgb', prefix=f'enroll_{name}_rgb')
        if not rgb_path:
            return error_response('Missing or invalid: rgb', 'MISSING_IMAGE', 400, missing_field='rgb')
        
        saved_files = [rgb_path]
        
        if not overwrite and _palm_system.is_enrolled(name):
            return error_response(
                f'{name} already enrolled', 'ALREADY_ENROLLED', 409, name=name
            )
        
        success = _enroller.enroll_quick_rgb(name, rgb_path, overwrite=True)
        
        if success:
            thumbnail = _make_thumbnail(rgb_path) if rgb_path else None
            return success_response({
                'mode': 'quick_rgb',
                'name': name,
                'images_used': 1,
                'ir_generated': True,
                'enrolled_count': _palm_system.num_enrolled(),
                'expected_accuracy': '82-88% for new people, 92-96% for trained',
                'thumbnail': thumbnail,
            })
        else:
            return error_response('Enrollment failed', 'ENROLLMENT_FAILED', 500)
    
    except Exception as e:
        logger.exception(f"[{g.request_id}] Enrollment error")
        return error_response(str(e), 'INTERNAL_ERROR', 500)
    
    finally:
        _cleanup_files(saved_files)


# ============================================================
# IDENTIFICATION ENDPOINTS
# ============================================================
@app.route('/identify', methods=['POST'])
def identify():
    """
    Identify a person from 1 RGB + 1 IR.
    
    Returns rich response with confidence as percentage (0-100), top matches,
    and clear status for frontend handling.
    """
    saved_files = []
    try:
        rgb_path = _get_image_input('rgb', prefix='query_rgb')
        ir_path = _get_image_input('ir', prefix='query_ir')
        
        if not rgb_path:
            return error_response('Missing or invalid: rgb', 'MISSING_IMAGE', 400)
        if not ir_path:
            return error_response('Missing or invalid: ir', 'MISSING_IMAGE', 400)
        
        saved_files = [rgb_path, ir_path]
        
        try:
            top_k = int(_get_string_input('top_k') or 5)
        except (ValueError, TypeError):
            top_k = 5
        
        try:
            t = _get_string_input('threshold')
            threshold = float(t) if t else None
        except (ValueError, TypeError):
            threshold = None
        
        result = _palm_system.identify(rgb_path, ir_path, top_k=top_k, threshold=threshold)
        
        # Empty database
        if result.status == 'no_database':
            return error_response(
                'No people enrolled. Enroll someone first.',
                'EMPTY_DATABASE',
                404,
            )
        
        # No hand detected
        if result.status == 'no_hand':
            return error_response(
                'Hand could not be detected in the image. Please retake the photo.',
                'NO_HAND_DETECTED',
                422,
            )
        
        return success_response({
            'identified': result.status == 'accepted',
            'name': result.name if result.status == 'accepted' else None,
            'best_guess': result.name,
            'confidence': _confidence_to_percent(result.confidence),
            'confidence_raw': float(result.confidence),
            'status': result.status,
            'threshold': float(threshold if threshold is not None else _palm_system.match_threshold),
            'thumbnail': _make_thumbnail(rgb_path) if rgb_path else None,
            'top_matches': [
                {
                    'rank': i + 1,
                    'name': name,
                    'confidence': _confidence_to_percent(score),
                    'confidence_raw': float(score),
                }
                for i, (name, score) in enumerate(result.top_k_matches)
            ],
            'message': _format_identify_message(result),
        })
    
    except Exception as e:
        logger.exception(f"[{g.request_id}] Identify error")
        return error_response(str(e), 'INTERNAL_ERROR', 500)
    
    finally:
        _cleanup_files(saved_files)


@app.route('/identify/rgb_only', methods=['POST'])
def identify_rgb_only():
    """Identify from RGB only — IR auto-generated."""
    saved_files = []
    try:
        rgb_path = _get_image_input('rgb', prefix='query_rgb')
        if not rgb_path:
            return error_response('Missing or invalid: rgb', 'MISSING_IMAGE', 400)
        saved_files = [rgb_path]
        
        try:
            top_k = int(_get_string_input('top_k') or 5)
        except (ValueError, TypeError):
            top_k = 5
        
        # Generate fake IR
        from rgb_to_ir import rgb_to_ir_style
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            return error_response('Cannot decode RGB image', 'INVALID_IMAGE', 400)
        
        ir_image = rgb_to_ir_style(rgb)
        ir_path = str(APIConfig.UPLOAD_DIR / f"query_ir_{g.request_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
        cv2.imwrite(ir_path, ir_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_files.append(ir_path)
        
        result = _palm_system.identify(rgb_path, ir_path, top_k=top_k)
        
        if result.status == 'no_database':
            return error_response('No people enrolled', 'EMPTY_DATABASE', 404)
        if result.status == 'no_hand':
            return error_response('Hand not detected', 'NO_HAND_DETECTED', 422)
        
        return success_response({
            'identified': result.status == 'accepted',
            'name': result.name if result.status == 'accepted' else None,
            'best_guess': result.name,
            'confidence': _confidence_to_percent(result.confidence),
            'confidence_raw': float(result.confidence),
            'status': result.status,
            'ir_generated': True,
            'top_matches': [
                {
                    'rank': i + 1,
                    'name': name,
                    'confidence': _confidence_to_percent(score),
                    'confidence_raw': float(score),
                }
                for i, (name, score) in enumerate(result.top_k_matches)
            ],
            'message': _format_identify_message(result),
        })
    
    except Exception as e:
        logger.exception(f"[{g.request_id}] Identify error")
        return error_response(str(e), 'INTERNAL_ERROR', 500)
    
    finally:
        _cleanup_files(saved_files)


@app.route('/verify', methods=['POST'])
def verify():
    """1-to-1 verification: 'Is this person who they claim to be?'"""
    saved_files = []
    try:
        name = _get_string_input('name')
        if not name:
            return error_response('Missing field: name', 'MISSING_NAME', 400)
        
        rgb_path = _get_image_input('rgb', prefix='verify_rgb')
        ir_path = _get_image_input('ir', prefix='verify_ir')
        
        if not rgb_path or not ir_path:
            return error_response('Missing rgb or ir', 'MISSING_IMAGE', 400)
        
        saved_files = [rgb_path, ir_path]
        
        if not _palm_system.is_enrolled(name):
            return error_response(
                f'{name} is not enrolled in the database',
                'NOT_ENROLLED',
                404,
                name=name
            )
        
        try:
            t = _get_string_input('threshold')
            threshold = float(t) if t else None
        except (ValueError, TypeError):
            threshold = None
        
        is_match, score = _palm_system.verify(name, rgb_path, ir_path, threshold=threshold)
        
        return success_response({
            'verified': bool(is_match),
            'claimed_name': name,
            'confidence': _confidence_to_percent(score),
            'confidence_raw': float(score),
            'threshold': float(threshold if threshold is not None else _palm_system.match_threshold),
            'message': f'Verified as {name}' if is_match else f'Does not match {name}',
        })
    
    except Exception as e:
        logger.exception(f"[{g.request_id}] Verify error")
        return error_response(str(e), 'INTERNAL_ERROR', 500)
    
    finally:
        _cleanup_files(saved_files)


# ============================================================
# DATABASE MANAGEMENT
# ============================================================
@app.route('/enrolled', methods=['GET'])
def list_enrolled():
    """List all enrolled people. Supports pagination via ?page=1&per_page=50."""
    names = _palm_system.list_enrolled()
    
    try:
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(500, max(1, int(request.args.get('per_page', len(names) if names else 50))))
    except (ValueError, TypeError):
        page = 1
        per_page = 50
    
    total = len(names)
    start = (page - 1) * per_page
    end = start + per_page
    
    return success_response({
        'count': total,
        'page': page,
        'per_page': per_page,
        'total_pages': max(1, (total + per_page - 1) // per_page),
        'names': names[start:end],
    })


@app.route('/enrolled/remove', methods=['POST'])
def remove_enrolled():
    """Remove a person."""
    name = _get_string_input('name')
    if not name:
        return error_response('Missing field: name', 'MISSING_NAME', 400)
    
    if not _palm_system.is_enrolled(name):
        return error_response(f'Not found: {name}', 'NOT_ENROLLED', 404, name=name)
    
    _palm_system.remove_enrolled(name)
    
    return success_response({
        'removed': name,
        'remaining_count': _palm_system.num_enrolled(),
    })


@app.route('/enrolled/clear', methods=['DELETE'])
def clear_enrolled():
    """DESTRUCTIVE: clears entire database. Requires ?confirm=yes."""
    confirm = request.args.get('confirm') or _get_string_input('confirm')
    
    if confirm != 'yes':
        return error_response(
            'Confirmation required. Add ?confirm=yes',
            'CONFIRMATION_REQUIRED',
            400
        )
    
    count = _palm_system.num_enrolled()
    for n in list(_palm_system.list_enrolled()):
        _palm_system.remove_enrolled(n)
    
    return success_response({
        'cleared_count': count,
        'remaining_count': 0,
        'message': f'Cleared {count} enrolled people',
    })


# ============================================================
# HELPER: build user-friendly messages
# ============================================================
def _format_identify_message(result) -> str:
    if result.status == 'accepted':
        return f'Identified as {result.name} ({_confidence_to_percent(result.confidence)}%)'
    if result.status == 'rejected_low_confidence':
        return (f'No confident match found. Best guess was {result.name} '
                f'({_confidence_to_percent(result.confidence)}%) but below threshold.')
    if result.status == 'no_hand':
        return 'Hand not detected. Please retake the photo with palm clearly visible.'
    if result.status == 'no_database':
        return 'No people enrolled yet.'
    return f'Status: {result.status}'


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found(e):
    return error_response('Endpoint not found', 'NOT_FOUND', 404, path=request.path)


@app.errorhandler(405)
def method_not_allowed(e):
    return error_response('Method not allowed for this endpoint', 'METHOD_NOT_ALLOWED', 405)


@app.errorhandler(413)
def too_large(e):
    return error_response(
        f'File too large. Max: {APIConfig.MAX_CONTENT_LENGTH // (1024*1024)} MB',
        'FILE_TOO_LARGE',
        413
    )


@app.errorhandler(500)
def server_error(e):
    return error_response('Internal server error', 'INTERNAL_ERROR', 500)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PALM BIOMETRIC API v2 (Frontend-Ready)")
    print("=" * 60)
    
    initialize_system()
    
    print(f"\nCORS: {APIConfig.CORS_ALLOWED_ORIGINS}")
    print(f"Max upload: {APIConfig.MAX_CONTENT_LENGTH // (1024*1024)} MB")
    print(f"Server: http://{APIConfig.HOST}:{APIConfig.PORT}")
    print(f"Test: curl http://localhost:{APIConfig.PORT}/health")
    print("=" * 60)
    
    app.run(
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        debug=APIConfig.DEBUG,
        threaded=True,
    )