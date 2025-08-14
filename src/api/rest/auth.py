"""
Authentication and Authorization for AI Detector API
JWT-based authentication with role-based access control
"""

from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import os
import logging

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Authentication error"""
    pass


class AuthorizationError(Exception):
    """Authorization error"""
    pass


# User model (in production, this would be from database)
class User:
    def __init__(self, user_id: str, username: str, email: str, 
                 hashed_password: str, roles: List[str] = None, 
                 is_active: bool = True, created_at: datetime = None):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.hashed_password = hashed_password
        self.roles = roles or ["user"]
        self.is_active = is_active
        self.created_at = created_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": self.roles,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }
    
    def has_role(self, role: str) -> bool:
        return role in self.roles
    
    def is_admin(self) -> bool:
        return self.has_role("admin")


# Mock user database (in production, use real database)
mock_users = {
    "test_user": User(
        user_id="user_123",
        username="test_user",
        email="test@example.com",
        hashed_password=pwd_context.hash("password123"),
        roles=["user"]
    ),
    "admin_user": User(
        user_id="admin_123", 
        username="admin_user",
        email="admin@example.com",
        hashed_password=pwd_context.hash("admin123"),
        roles=["admin", "user"]
    )
}


# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash password"""
    return pwd_context.hash(password)


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username/password"""
    user = mock_users.get(username)
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    if not user.is_active:
        return None
    
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def get_user_from_token(token: str) -> User:
    """Get user from JWT token"""
    try:
        payload = decode_token(token)
        username = payload.get("sub")
        
        if username is None:
            raise AuthenticationError("Invalid token payload")
        
        # Check token type
        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")
        
        # Get user (in production, query database)
        user = mock_users.get(username)
        if user is None:
            raise AuthenticationError("User not found")
        
        if not user.is_active:
            raise AuthenticationError("User account is inactive")
        
        return user
        
    except JWTError as e:
        raise AuthenticationError(f"Token validation failed: {str(e)}")


# FastAPI dependencies
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token (optional)"""
    if not credentials:
        return None
    
    try:
        user = get_user_from_token(credentials.credentials)
        return user.to_dict()
    except AuthenticationError:
        return None


async def require_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Require authentication (mandatory)"""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail={"message": "Authentication required", "error_code": "MISSING_TOKEN"}
        )
    
    try:
        user = get_user_from_token(credentials.credentials)
        return user.to_dict()
    except AuthenticationError as e:
        raise HTTPException(
            status_code=401,
            detail={"message": str(e), "error_code": "INVALID_TOKEN"}
        )


def require_role(required_role: str):
    """Require specific role"""
    async def role_dependency(current_user: Dict[str, Any] = Depends(require_auth)):
        user_roles = current_user.get("roles", [])
        if required_role not in user_roles:
            raise HTTPException(
                status_code=403,
                detail={
                    "message": f"Role '{required_role}' required", 
                    "error_code": "INSUFFICIENT_PERMISSIONS"
                }
            )
        return current_user
    
    return role_dependency


def require_admin():
    """Require admin role"""
    return require_role("admin")


# Login/logout functionality
class LoginRequest:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password


class TokenResponse:
    def __init__(self, access_token: str, refresh_token: str, token_type: str = "bearer", user: Dict[str, Any] = None):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_type = token_type
        self.user = user


def login(username: str, password: str) -> TokenResponse:
    """Login and generate tokens"""
    user = authenticate_user(username, password)
    if not user:
        raise AuthenticationError("Invalid username or password")
    
    # Create tokens
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "user_id": user.user_id, "roles": user.roles},
        expires_delta=access_token_expires
    )
    
    refresh_token = create_refresh_token(
        data={"sub": user.username, "user_id": user.user_id}
    )
    
    logger.info(f"User {user.username} logged in successfully")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user=user.to_dict()
    )


def refresh_access_token(refresh_token: str) -> str:
    """Refresh access token using refresh token"""
    try:
        payload = decode_token(refresh_token)
        username = payload.get("sub")
        
        if username is None:
            raise AuthenticationError("Invalid refresh token")
        
        # Check token type
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid token type for refresh")
        
        # Get user
        user = mock_users.get(username)
        if user is None or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username, "user_id": user.user_id, "roles": user.roles},
            expires_delta=access_token_expires
        )
        
        logger.info(f"Access token refreshed for user {user.username}")
        
        return access_token
        
    except JWTError as e:
        raise AuthenticationError(f"Refresh token validation failed: {str(e)}")


# API Key authentication (alternative to JWT)
class APIKeyAuth:
    """API Key authentication"""
    
    def __init__(self):
        # Mock API keys (in production, store in database)
        self.api_keys = {
            "api_key_user_123": {
                "user_id": "user_123",
                "username": "api_user",
                "roles": ["user"],
                "is_active": True
            },
            "api_key_admin_123": {
                "user_id": "admin_123", 
                "username": "api_admin",
                "roles": ["admin", "user"],
                "is_active": True
            }
        }
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key"""
        key_info = self.api_keys.get(api_key)
        if key_info and key_info.get("is_active", False):
            return key_info
        return None


api_key_auth = APIKeyAuth()


async def get_current_user_from_api_key(request: Request) -> Optional[Dict[str, Any]]:
    """Get user from API key header"""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        return None
    
    user_info = api_key_auth.validate_api_key(api_key)
    return user_info


async def get_current_user_flexible(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token OR API key"""
    
    # Try JWT token first
    if credentials:
        try:
            user = get_user_from_token(credentials.credentials)
            return user.to_dict()
        except AuthenticationError:
            pass
    
    # Try API key
    user_info = await get_current_user_from_api_key(request)
    if user_info:
        return user_info
    
    return None


# Rate limiting by user
class UserRateLimiter:
    """Rate limiter per authenticated user"""
    
    def __init__(self):
        self.user_requests = {}
    
    def check_user_rate_limit(self, user_id: str, endpoint: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if user has exceeded rate limit"""
        import time
        
        current_time = time.time()
        key = f"{user_id}:{endpoint}"
        
        if key not in self.user_requests:
            self.user_requests[key] = []
        
        # Remove old requests
        cutoff_time = current_time - window
        self.user_requests[key] = [t for t in self.user_requests[key] if t > cutoff_time]
        
        # Check limit
        if len(self.user_requests[key]) >= limit:
            return False
        
        # Record this request
        self.user_requests[key].append(current_time)
        return True


user_rate_limiter = UserRateLimiter()


# Permission decorators
def check_permission(permission: str):
    """Check specific permission"""
    async def permission_check(current_user: Dict[str, Any] = Depends(require_auth)):
        # In a real implementation, you'd check permissions from database
        # For now, just check basic role-based permissions
        user_roles = current_user.get("roles", [])
        
        permission_map = {
            "detect": ["user", "admin"],
            "train": ["admin"],
            "admin": ["admin"],
            "export": ["user", "admin"]
        }
        
        allowed_roles = permission_map.get(permission, [])
        if not any(role in user_roles for role in allowed_roles):
            raise HTTPException(
                status_code=403,
                detail={
                    "message": f"Permission '{permission}' required",
                    "error_code": "INSUFFICIENT_PERMISSIONS"
                }
            )
        
        return current_user
    
    return permission_check


# Session management
class SessionManager:
    """Simple session management"""
    
    def __init__(self):
        self.active_sessions = {}
    
    def create_session(self, user_id: str, token: str) -> str:
        """Create user session"""
        import uuid
        session_id = str(uuid.uuid4())
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow()
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return False
        
        # Check if session is expired (24 hours)
        if datetime.utcnow() - session["created_at"] > timedelta(hours=24):
            self.invalidate_session(session_id)
            return False
        
        # Update last activity
        session["last_activity"] = datetime.utcnow()
        return True
    
    def invalidate_session(self, session_id: str):
        """Invalidate session"""
        self.active_sessions.pop(session_id, None)
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """Get all sessions for user"""
        return [
            session_id for session_id, session in self.active_sessions.items()
            if session["user_id"] == user_id
        ]


session_manager = SessionManager()


__all__ = [
    'User', 'AuthenticationError', 'AuthorizationError',
    'verify_password', 'get_password_hash', 'authenticate_user',
    'create_access_token', 'create_refresh_token', 'decode_token', 'get_user_from_token',
    'get_current_user', 'require_auth', 'require_role', 'require_admin',
    'login', 'refresh_access_token',
    'APIKeyAuth', 'api_key_auth', 'get_current_user_from_api_key', 'get_current_user_flexible',
    'UserRateLimiter', 'user_rate_limiter',
    'check_permission', 'SessionManager', 'session_manager'
]