"""
Enterprise Configuration Management
==================================
Handles all configuration for the enterprise dermatology AI platform
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import validator, Field


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    db_url: str = Field(default="sqlite:///./dermatology_enterprise.db")
    db_echo: bool = Field(default=False)
    
    class Config:
        env_prefix = "DB_"


class AuthSettings(BaseSettings):
    """Authentication and security configuration"""
    secret_key: str = Field(default="your-secret-key-change-in-production")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)
    refresh_token_expire_days: int = Field(default=7)
    password_reset_expire_hours: int = Field(default=24)
    
    # Session configuration
    session_expire_hours: int = Field(default=8)
    max_login_attempts: int = Field(default=5)
    lockout_duration_minutes: int = Field(default=15)
    
    class Config:
        env_prefix = "AUTH_"


class EmailSettings(BaseSettings):
    """Email configuration for notifications"""
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    from_email: str = Field(default="noreply@dermatologyai.com")
    from_name: str = Field(default="Dermatology AI Platform")
    
    class Config:
        env_prefix = "EMAIL_"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and sessions"""
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_expire_seconds: int = Field(default=3600)
    session_expire_seconds: int = Field(default=28800)  # 8 hours
    
    class Config:
        env_prefix = "REDIS_"


class FileStorageSettings(BaseSettings):
    """File storage configuration"""
    upload_dir: str = Field(default="./uploads")
    max_file_size_mb: int = Field(default=10)
    allowed_extensions: List[str] = Field(default=["jpg", "jpeg", "png", "webp"])
    
    # AWS S3 configuration for production
    use_s3: bool = Field(default=False)
    s3_bucket: str = Field(default="")
    s3_access_key: str = Field(default="")
    s3_secret_key: str = Field(default="")
    s3_region: str = Field(default="us-east-1")
    
    class Config:
        env_prefix = "STORAGE_"


class AIModelSettings(BaseSettings):
    """AI model configuration"""
    model_cache_dir: str = Field(default="./models")
    batch_size: int = Field(default=1)
    confidence_threshold: float = Field(default=0.5)
    max_concurrent_analyses: int = Field(default=5)
    
    # Model paths
    vision_model_path: str = Field(default="")
    text_model_path: str = Field(default="")
    
    class Config:
        env_prefix = "AI_"


class MonitoringSettings(BaseSettings):
    """Monitoring and logging configuration"""
    log_level: str = Field(default="INFO")
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=8001)
    
    # External monitoring
    sentry_dsn: str = Field(default="")
    datadog_api_key: str = Field(default="")
    
    class Config:
        env_prefix = "MONITORING_"


class AppSettings(BaseSettings):
    """Main application configuration"""
    app_name: str = Field(default="Enterprise Dermatology AI Platform")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=True)
    
    # Server configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    
    # CORS configuration
    cors_origins: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window_seconds: int = Field(default=60)
    
    @validator('environment')
    def validate_environment(cls, v):
        if v not in ['development', 'staging', 'production']:
            raise ValueError('Environment must be development, staging, or production')
        return v
    
    class Config:
        env_prefix = "APP_"


class EnterpriseSettings(BaseSettings):
    """Main enterprise settings class that combines all configurations"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.database = DatabaseSettings()
        self.auth = AuthSettings()
        self.email = EmailSettings()
        self.redis = RedisSettings()
        self.storage = FileStorageSettings()
        self.ai = AIModelSettings()
        self.monitoring = MonitoringSettings()
        self.app = AppSettings()
    
    # License and enterprise features
    license_key: str = Field(default="")
    max_users: int = Field(default=100)
    max_organizations: int = Field(default=10)
    enable_audit_logs: bool = Field(default=True)
    enable_data_export: bool = Field(default=True)
    enable_api_access: bool = Field(default=True)
    
    # HIPAA compliance features
    enable_encryption_at_rest: bool = Field(default=True)
    enable_audit_trail: bool = Field(default=True)
    data_retention_days: int = Field(default=2555)  # 7 years
    enable_user_consent_tracking: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = EnterpriseSettings()


def get_settings() -> EnterpriseSettings:
    """Get the global settings instance"""
    return settings