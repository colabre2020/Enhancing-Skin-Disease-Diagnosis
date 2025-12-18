"""
Enterprise Database Models
=========================
SQLAlchemy models for the enterprise dermatology AI platform
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, JSON, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import uuid
import json

Base = declarative_base()


class TimestampMixin:
    """Mixin for timestamp fields"""
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Organization(Base, TimestampMixin):
    """Organization/Clinic model"""
    __tablename__ = "organizations"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    
    # Contact information
    email = Column(String(255))
    phone = Column(String(50))
    address = Column(Text)
    website = Column(String(255))
    
    # Settings
    license_type = Column(String(50), default="basic")  # basic, professional, enterprise
    max_users = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    
    # Compliance settings
    hipaa_enabled = Column(Boolean, default=True)
    audit_enabled = Column(Boolean, default=True)
    data_retention_days = Column(Integer, default=2555)  # 7 years
    
    # Relationships
    users = relationship("User", back_populates="organization")
    cases = relationship("DiagnosisCase", back_populates="organization")
    audit_logs = relationship("AuditLog", back_populates="organization")


class User(Base, TimestampMixin):
    """User model with role-based access"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True)
    
    # Authentication
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    last_login = Column(DateTime(timezone=True))
    
    # Profile information
    first_name = Column(String(100))
    last_name = Column(String(100))
    title = Column(String(100))  # Dr., MD, etc.
    specialization = Column(String(100))
    license_number = Column(String(100))
    
    # Role and permissions
    role = Column(String(50), default="user")  # admin, doctor, nurse, technician, user
    permissions = Column(JSON)  # Custom permissions
    
    # Organization relationship
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    organization = relationship("Organization", back_populates="users")
    
    # Security
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True))
    password_reset_token = Column(String(255))
    password_reset_expires = Column(DateTime(timezone=True))
    
    # Relationships
    cases = relationship("DiagnosisCase", back_populates="user")
    sessions = relationship("UserSession", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    @property
    def full_name(self):
        return f"{self.first_name or ''} {self.last_name or ''}".strip()
    
    @property
    def is_locked(self):
        return self.locked_until and self.locked_until > datetime.utcnow()


class UserSession(Base, TimestampMixin):
    """User session tracking"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Session details
    ip_address = Column(String(45))
    user_agent = Column(Text)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Device information
    device_type = Column(String(50))
    browser = Column(String(100))
    location = Column(String(255))
    
    # Relationships
    user = relationship("User", back_populates="sessions")


class DiagnosisCase(Base, TimestampMixin):
    """Diagnosis case model"""
    __tablename__ = "diagnosis_cases"
    
    id = Column(Integer, primary_key=True, index=True)
    case_number = Column(String(50), unique=True, nullable=False)
    
    # Patient information (anonymized for HIPAA compliance)
    patient_id = Column(String(100))  # Internal patient ID
    patient_age = Column(Integer)
    patient_gender = Column(String(20))
    patient_skin_type = Column(String(50))
    
    # Clinical information
    clinical_history = Column(Text)
    lesion_location = Column(String(100))
    symptoms = Column(Text)
    
    # Image information
    image_filename = Column(String(255))
    image_path = Column(String(500))
    image_hash = Column(String(64))  # For deduplication
    image_metadata = Column(JSON)
    
    # AI Analysis results
    ai_predictions = Column(JSON)
    ai_confidence_scores = Column(JSON)
    ai_explanations = Column(Text)
    visual_concepts = Column(JSON)
    clinical_concepts = Column(JSON)
    
    # Clinical review
    reviewed_by_doctor = Column(Boolean, default=False)
    doctor_diagnosis = Column(String(255))
    doctor_notes = Column(Text)
    treatment_recommendations = Column(Text)
    
    # Status and workflow
    status = Column(String(50), default="pending")  # pending, analyzed, reviewed, completed
    priority = Column(String(20), default="normal")  # low, normal, high, urgent
    
    # Relationships
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    user = relationship("User", back_populates="cases")
    organization = relationship("Organization", back_populates="cases")
    audit_logs = relationship("AuditLog", back_populates="case")


class AuditLog(Base):
    """Audit log for compliance and tracking"""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Action details
    action = Column(String(100), nullable=False)  # login, logout, view_case, analyze_image, etc.
    resource_type = Column(String(50))  # user, case, organization, etc.
    resource_id = Column(String(100))
    
    # User and organization
    user_id = Column(Integer, ForeignKey("users.id"))
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    case_id = Column(Integer, ForeignKey("diagnosis_cases.id"))
    
    # Request details
    ip_address = Column(String(45))
    user_agent = Column(Text)
    session_id = Column(String(255))
    
    # Change tracking
    old_values = Column(JSON)
    new_values = Column(JSON)
    
    # Additional metadata
    metadata = Column(JSON)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    organization = relationship("Organization", back_populates="audit_logs")
    case = relationship("DiagnosisCase", back_populates="audit_logs")


class APIKey(Base, TimestampMixin):
    """API keys for external access"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False)
    
    # Permissions and limits
    permissions = Column(JSON)
    rate_limit = Column(Integer, default=1000)  # requests per hour
    is_active = Column(Boolean, default=True)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True))
    
    # Organization relationship
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)


class SystemSettings(Base, TimestampMixin):
    """System-wide settings"""
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    value_type = Column(String(20), default="string")  # string, integer, boolean, json
    description = Column(Text)
    
    # Category for grouping
    category = Column(String(50), default="general")
    
    # Access control
    is_public = Column(Boolean, default=False)
    requires_restart = Column(Boolean, default=False)


class NotificationTemplate(Base, TimestampMixin):
    """Email/notification templates"""
    __tablename__ = "notification_templates"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    type = Column(String(50), nullable=False)  # email, sms, push
    
    # Template content
    subject = Column(String(255))
    body_text = Column(Text)
    body_html = Column(Text)
    
    # Template variables
    variables = Column(JSON)  # Available template variables
    
    # Settings
    is_active = Column(Boolean, default=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))


class DataExport(Base, TimestampMixin):
    """Data export requests for compliance"""
    __tablename__ = "data_exports"
    
    id = Column(Integer, primary_key=True, index=True)
    export_type = Column(String(50), nullable=False)  # user_data, audit_logs, cases
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    
    # Export parameters
    date_from = Column(DateTime(timezone=True))
    date_to = Column(DateTime(timezone=True))
    filters = Column(JSON)
    
    # File information
    filename = Column(String(255))
    file_path = Column(String(500))
    file_size = Column(Integer)
    
    # Request details
    requested_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    organization_id = Column(Integer, ForeignKey("organizations.id"), nullable=False)
    
    # Expiration (for security)
    expires_at = Column(DateTime(timezone=True))


# Model utility functions
def generate_case_number() -> str:
    """Generate a unique case number"""
    import time
    timestamp = int(time.time())
    random_part = str(uuid.uuid4()).split('-')[0]
    return f"CASE-{timestamp}-{random_part.upper()}"


def create_user_session(user_id: int, ip_address: str, user_agent: str) -> UserSession:
    """Create a new user session"""
    session_id = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(hours=8)
    
    return UserSession(
        session_id=session_id,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        expires_at=expires_at
    )


def log_audit_event(
    session: Session,
    action: str,
    user_id: Optional[int] = None,
    organization_id: Optional[int] = None,
    case_id: Optional[int] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    old_values: Optional[Dict] = None,
    new_values: Optional[Dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None
):
    """Log an audit event"""
    audit_log = AuditLog(
        action=action,
        user_id=user_id,
        organization_id=organization_id,
        case_id=case_id,
        resource_type=resource_type,
        resource_id=resource_id,
        old_values=old_values,
        new_values=new_values,
        ip_address=ip_address,
        user_agent=user_agent,
        session_id=session_id,
        metadata=metadata
    )
    session.add(audit_log)
    session.commit()
    return audit_log