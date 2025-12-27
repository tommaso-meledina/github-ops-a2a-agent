from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class BearerAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/.well-known"):
            return await call_next(request)
        auth = request.headers.get("Authorization")

        if not auth or not auth.lower().startswith("bearer "):
            print(f"Missing or invalid Authorization header: {auth}")
            return JSONResponse(
                {"detail": "Missing or invalid Authorization header"},
                status_code=401,
            )

        token = auth.split(" ", 1)[1]

        # TODO: validate token (OIDC / JWT / introspection)
        # If invalid:
        #   return JSONResponse(..., status_code=403)

        # Attach identity to request.state
        request.state.token = token
        # request.state.user = ...
        # request.state.scopes = ...

        return await call_next(request)
