"""
Enterprise Authentication System
===============================
JWT-based authentication with role-based access control
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import secrets
import uuid
from sqlalchemy.orm import Session

# Import models (will be available after installation)
# from .models import User, UserSession, AuditLog, Organization
# from .database import get_db
# from .enterprise_config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handler
security = HTTPBearer()


class AuthenticationError(Exception):
    """Custom authentication error"""
    def __init__(self, message: str, error_code: str = "AUTH_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class AuthorizationError(Exception):
    """Custom authorization error"""
    def __init__(self, message: str, error_code: str = "AUTHORIZATION_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class PasswordManager:
    """Password management utilities"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def generate_secure_password(length: int = 12) -> str:
        """Generate a secure random password"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """Validate password strength"""
        issues = []
        score = 0
        
        if len(password) < 8:
            issues.append("Password must be at least 8 characters long")
        else:
            score += 1
            
        if not any(c.islower() for c in password):
            issues.append("Password must contain lowercase letters")
        else:
            score += 1
            
        if not any(c.isupper() for c in password):
            issues.append("Password must contain uppercase letters")
        else:
            score += 1
            
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain numbers")
        else:
            score += 1
            
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain special characters")
        else:
            score += 1
        
        strength_levels = ["Very Weak", "Weak", "Fair", "Good", "Strong"]
        strength = strength_levels[min(score, 4)]
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "strength": strength,
            "score": score
        }


class TokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=30)
            
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT refresh token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=7)
            
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise AuthenticationError("Invalid token")
    
    def create_password_reset_token(self, email: str) -> str:
        """Create a password reset token"""
        data = {
            "email": email,
            "type": "password_reset",
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(data, self.secret_key, algorithm=self.algorithm)


class SessionManager:
    """User session management"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_session(self, user_id: int, ip_address: str, user_agent: str) -> str:
        """Create a new user session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=8)
        
        # Create session record (when models are available)
        # session = UserSession(
        #     session_id=session_id,
        #     user_id=user_id,
        #     ip_address=ip_address,
        #     user_agent=user_agent,
        #     expires_at=expires_at
        # )
        # self.db.add(session)
        # self.db.commit()
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Validate if a session is active"""
        # Implementation when models are available
        # session = self.db.query(UserSession).filter(
        #     UserSession.session_id == session_id,
        #     UserSession.is_active == True,
        #     UserSession.expires_at > datetime.utcnow()
        # ).first()
        # return session is not None
        return True
    
    def invalidate_session(self, session_id: str):
        """Invalidate a user session"""
        # Implementation when models are available
        # self.db.query(UserSession).filter(
        #     UserSession.session_id == session_id
        # ).update({"is_active": False})
        # self.db.commit()
        pass
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        # Implementation when models are available
        # self.db.query(UserSession).filter(
        #     UserSession.expires_at < datetime.utcnow()
        # ).update({"is_active": False})
        # self.db.commit()
        pass


class RoleManager:
    """Role-based access control"""
    
    # Define role hierarchy
    ROLES = {
        "super_admin": {
            "level": 100,
            "permissions": ["*"]  # All permissions
        },
        "org_admin": {
            "level": 80,
            "permissions": [
                "manage_users", "manage_organization", "view_all_cases",
                "export_data", "view_audit_logs", "manage_api_keys"
            ]
        },
        "doctor": {
            "level": 70,
            "permissions": [
                "analyze_images", "view_cases", "review_cases", "create_cases",
                "view_own_cases", "download_reports"
            ]
        },
        "nurse": {
            "level": 50,
            "permissions": [
                "analyze_images", "create_cases", "view_own_cases", "basic_reports"
            ]
        },
        "technician": {
            "level": 40,
            "permissions": [
                "analyze_images", "create_cases", "view_own_cases"
            ]
        },
        "user": {
            "level": 30,
            "permissions": [
                "analyze_images", "view_own_cases"
            ]
        },
        "viewer": {
            "level": 10,
            "permissions": [
                "view_own_cases"
            ]
        }
    }
    
    @classmethod
    def has_permission(cls, user_role: str, required_permission: str, custom_permissions: Optional[List[str]] = None) -> bool:
        """Check if a role has a specific permission"""
        if user_role not in cls.ROLES:
            return False
        
        role_permissions = cls.ROLES[user_role]["permissions"]
        
        # Super admin has all permissions
        if "*" in role_permissions:
            return True
        
        # Check role permissions
        if required_permission in role_permissions:
            return True
        
        # Check custom permissions
        if custom_permissions and required_permission in custom_permissions:
            return True
        
        return False
    
    @classmethod
    def can_access_role(cls, current_role: str, target_role: str) -> bool:
        """Check if current role can manage target role"""
        if current_role not in cls.ROLES or target_role not in cls.ROLES:
            return False
        
        current_level = cls.ROLES[current_role]["level"]
        target_level = cls.ROLES[target_role]["level"]
        
        return current_level > target_level
    
    @classmethod
    def get_available_roles(cls, current_role: str) -> List[str]:
        """Get roles that current role can assign"""
        if current_role not in cls.ROLES:
            return []
        
        current_level = cls.ROLES[current_role]["level"]
        available_roles = []
        
        for role, config in cls.ROLES.items():
            if config["level"] < current_level:
                available_roles.append(role)
        
        return available_roles


class AuthService:
    """Main authentication service"""
    
    def __init__(self, secret_key: str):
        self.token_manager = TokenManager(secret_key)
        self.password_manager = PasswordManager()
        self.role_manager = RoleManager()
    
    def authenticate_user(self, db: Session, email: str, password: str, ip_address: str, user_agent: str) -> Dict[str, Any]:
        """Authenticate a user and return tokens"""
        # Implementation when models are available
        # user = db.query(User).filter(User.email == email).first()
        
        # Mock user for now
        user = None
        
        if not user:
            raise AuthenticationError("Invalid email or password")
        
        # Check if user is locked
        # if user.is_locked:
        #     raise AuthenticationError("Account is temporarily locked due to failed login attempts")
        
        # Verify password
        # if not self.password_manager.verify_password(password, user.hashed_password):
        #     self.handle_failed_login(db, user)
        #     raise AuthenticationError("Invalid email or password")
        
        # Reset failed attempts on successful login
        # user.failed_login_attempts = 0
        # user.locked_until = None
        # user.last_login = datetime.utcnow()
        # db.commit()
        
        # Create session
        session_manager = SessionManager(db)
        session_id = session_manager.create_session(1, ip_address, user_agent)  # user.id
        
        # Create tokens
        access_token = self.token_manager.create_access_token(
            data={"sub": email, "user_id": 1, "role": "doctor", "session_id": session_id}
        )
        refresh_token = self.token_manager.create_refresh_token(
            data={"sub": email, "user_id": 1, "session_id": session_id}
        )
        
        # Log audit event
        # log_audit_event(
        #     db, "login", user_id=user.id, organization_id=user.organization_id,
        #     ip_address=ip_address, user_agent=user_agent, session_id=session_id
        # )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": 1,  # user.id
                "email": email,
                "full_name": "Dr. Test User",  # user.full_name
                "role": "doctor",  # user.role
                "organization": "Test Clinic"  # user.organization.name
            }
        }
    
    def handle_failed_login(self, db: Session, user):
        """Handle failed login attempt"""
        # user.failed_login_attempts += 1
        
        # Lock account after max attempts
        # if user.failed_login_attempts >= 5:  # settings.auth.max_login_attempts
        #     user.locked_until = datetime.utcnow() + timedelta(minutes=15)
        
        # db.commit()
        pass
    
    def refresh_access_token(self, refresh_token: str) -> str:
        """Refresh an access token"""
        payload = self.token_manager.verify_token(refresh_token)
        
        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid refresh token")
        
        # Create new access token
        new_token = self.token_manager.create_access_token(
            data={"sub": payload["sub"], "user_id": payload["user_id"], "role": payload.get("role")}
        )
        
        return new_token
    
    def logout_user(self, db: Session, session_id: str, user_id: int):
        """Logout a user"""
        session_manager = SessionManager(db)
        session_manager.invalidate_session(session_id)
        
        # Log audit event
        # log_audit_event(db, "logout", user_id=user_id, session_id=session_id)


# Global auth service instance (will be initialized with proper config)
auth_service = None


def get_auth_service() -> AuthService:
    """Get the global auth service instance"""
    global auth_service
    if not auth_service:
        auth_service = AuthService("your-secret-key-change-in-production")
    return auth_service


# Dependency functions for FastAPI
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user"""
    token = credentials.credentials
    auth = get_auth_service()
    
    try:
        payload = auth.token_manager.verify_token(token)
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if email is None or user_id is None:
            raise AuthenticationError("Invalid token payload")
        
        # Return mock user for now
        return {
            "id": user_id,
            "email": email,
            "role": payload.get("role", "user"),
            "session_id": payload.get("session_id")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def permission_checker(current_user: dict = Depends(get_current_user)):
        role_manager = RoleManager()
        
        if not role_manager.has_permission(current_user["role"], permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {permission}"
            )
        
        return current_user
    
    return permission_checker


def require_role(required_role: str):
    """Decorator to require specific role"""
    def role_checker(current_user: dict = Depends(get_current_user)):
        role_manager = RoleManager()
        
        if not role_manager.can_access_role(current_user["role"], required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient role level. Required: {required_role}"
            )
        
        return current_user
    
    return role_checker