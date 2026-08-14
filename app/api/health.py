from fastapi import APIRouter

router = APIRouter(prefix="/agent/v1/health", tags=["Health"])


@router.get("")
def health_check():
    return {"status": "ok"}
