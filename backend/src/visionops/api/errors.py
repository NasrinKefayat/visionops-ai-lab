"""Safe, consistent API errors without request inputs, paths or driver diagnostics."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, _exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {"code": "internal_error", "message": "Request could not be completed."}
            },
        )

    @app.exception_handler(APIError)
    async def api_error(_request: Request, exc: APIError):
        return JSONResponse(
            status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request parameters are invalid.",
                    "fields": [".".join(map(str, error["loc"])) for error in exc.errors()],
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": "not_found" if exc.status_code == 404 else "request_error",
                    "message": "Resource not found."
                    if exc.status_code == 404
                    else "Request could not be completed.",
                }
            },
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_error(_request: Request, _exc: SQLAlchemyError):
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "database_unavailable",
                    "message": "Experiment data is unavailable.",
                }
            },
        )

    @app.exception_handler(ResponseValidationError)
    async def response_error(_request: Request, _exc: ResponseValidationError):
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "data_invalid",
                    "message": "Stored experiment data is unavailable.",
                }
            },
        )
