"""
Security and Privacy Management for Darrell Agent
Handles credential encryption, session management, and privacy protection
"""

import os
import json
import hashlib
import secrets
import base64
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
from dataclasses import dataclass


@dataclass
class SecurityContext:
    """Security context for operations"""
    user_id: str
    session_id: str
    permissions: List[str]
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class CredentialManager:
    """Secure credential management"""
    
    def __init__(self, storage_path: str = "credentials", use_keyring: bool = True):
        """
        Initialize credential manager
        
        Args:
            storage_path: Path for encrypted credential storage
            use_keyring: Whether to use system keyring for master key
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.use_keyring = use_keyring
        self.service_name = "darrell_agent"
        
        # Initialize encryption
        self._master_key = self._get_or_create_master_key()
        self._cipher = Fernet(self._master_key)
    
    def _get_or_create_master_key(self) -> bytes:
        """Get or create master encryption key"""
        if self.use_keyring:
            try:
                # Try to get key from system keyring
                key_b64 = keyring.get_password(self.service_name, "master_key")
                if key_b64:
                    return base64.urlsafe_b64decode(key_b64.encode())
                
                # Create new key and store in keyring
                key = Fernet.generate_key()
                keyring.set_password(self.service_name, "master_key", 
                                   base64.urlsafe_b64encode(key).decode())
                return key
            except Exception:
                # Fall back to file-based storage
                pass
        
        # File-based key storage
        key_file = self.storage_path / ".master_key"
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        
        # Create new key
        key = Fernet.generate_key()
        with open(key_file, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions
        os.chmod(key_file, 0o600)
        return key
    
    def store_credential(self, service: str, username: str, password: str, 
                        metadata: Dict[str, Any] = None) -> bool:
        """
        Store encrypted credential
        
        Args:
            service: Service name (e.g., 'zoom', 'elevenlabs')
            username: Username or identifier
            password: Password or API key
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        try:
            credential_data = {
                'username': username,
                'password': password,
                'metadata': metadata or {},
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Encrypt credential data
            encrypted_data = self._cipher.encrypt(
                json.dumps(credential_data).encode()
            )
            
            # Store encrypted data
            credential_file = self.storage_path / f"{service}.cred"
            with open(credential_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set restrictive permissions
            os.chmod(credential_file, 0o600)
            return True
            
        except Exception as e:
            print(f"Failed to store credential for {service}: {e}")
            return False
    
    def get_credential(self, service: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt credential
        
        Args:
            service: Service name
        
        Returns:
            Credential data or None if not found
        """
        try:
            credential_file = self.storage_path / f"{service}.cred"
            if not credential_file.exists():
                return None
            
            # Read and decrypt data
            with open(credential_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._cipher.decrypt(encrypted_data)
            credential_data = json.loads(decrypted_data.decode())
            
            return credential_data
            
        except Exception as e:
            print(f"Failed to retrieve credential for {service}: {e}")
            return None
    
    def delete_credential(self, service: str) -> bool:
        """
        Delete stored credential
        
        Args:
            service: Service name
        
        Returns:
            Success status
        """
        try:
            credential_file = self.storage_path / f"{service}.cred"
            if credential_file.exists():
                credential_file.unlink()
            return True
        except Exception as e:
            print(f"Failed to delete credential for {service}: {e}")
            return False
    
    def list_services(self) -> List[str]:
        """List all stored services"""
        services = []
        for cred_file in self.storage_path.glob("*.cred"):
            services.append(cred_file.stem)
        return services


class SessionManager:
    """Secure session management"""
    
    def __init__(self, session_timeout: int = 3600):
        """
        Initialize session manager
        
        Args:
            session_timeout: Session timeout in seconds
        """
        self.session_timeout = session_timeout
        self.active_sessions: Dict[str, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 3
        self.lockout_duration = timedelta(minutes=15)
    
    def create_session(self, user_id: str, permissions: List[str],
                      ip_address: str = None, user_agent: str = None) -> str:
        """
        Create new secure session
        
        Args:
            user_id: User identifier
            permissions: List of permissions
            ip_address: Client IP address
            user_agent: Client user agent
        
        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            permissions=permissions,
            created_at=now,
            expires_at=now + timedelta(seconds=self.session_timeout),
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.active_sessions[session_id] = context
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """
        Validate session and return context
        
        Args:
            session_id: Session ID to validate
        
        Returns:
            Security context or None if invalid
        """
        if session_id not in self.active_sessions:
            return None
        
        context = self.active_sessions[session_id]
        
        # Check expiration
        if datetime.now() > context.expires_at:
            self.destroy_session(session_id)
            return None
        
        return context
    
    def refresh_session(self, session_id: str) -> bool:
        """
        Refresh session expiration
        
        Args:
            session_id: Session ID to refresh
        
        Returns:
            Success status
        """
        context = self.validate_session(session_id)
        if not context:
            return False
        
        context.expires_at = datetime.now() + timedelta(seconds=self.session_timeout)
        return True
    
    def destroy_session(self, session_id: str) -> bool:
        """
        Destroy session
        
        Args:
            session_id: Session ID to destroy
        
        Returns:
            Success status
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if identifier is rate limited
        
        Args:
            identifier: IP address or user ID
        
        Returns:
            True if allowed, False if rate limited
        """
        now = datetime.now()
        
        if identifier not in self.failed_attempts:
            return True
        
        # Clean old attempts
        cutoff = now - self.lockout_duration
        self.failed_attempts[identifier] = [
            attempt for attempt in self.failed_attempts[identifier]
            if attempt > cutoff
        ]
        
        # Check if still locked out
        return len(self.failed_attempts[identifier]) < self.max_failed_attempts
    
    def record_failed_attempt(self, identifier: str):
        """
        Record failed authentication attempt
        
        Args:
            identifier: IP address or user ID
        """
        now = datetime.now()
        if identifier not in self.failed_attempts:
            self.failed_attempts[identifier] = []
        
        self.failed_attempts[identifier].append(now)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now()
        expired_sessions = [
            session_id for session_id, context in self.active_sessions.items()
            if now > context.expires_at
        ]
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]


class PrivacyManager:
    """Privacy protection and data handling"""
    
    def __init__(self, data_retention_days: int = 7):
        """
        Initialize privacy manager
        
        Args:
            data_retention_days: Days to retain sensitive data
        """
        self.data_retention_days = data_retention_days
        self.sensitive_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
        ]
    
    def sanitize_text(self, text: str) -> str:
        """
        Sanitize text by removing sensitive information
        
        Args:
            text: Text to sanitize
        
        Returns:
            Sanitized text
        """
        import re
        
        sanitized = text
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized
    
    def hash_identifier(self, identifier: str) -> str:
        """
        Create privacy-preserving hash of identifier
        
        Args:
            identifier: Identifier to hash
        
        Returns:
            Hashed identifier
        """
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def should_retain_data(self, created_at: datetime) -> bool:
        """
        Check if data should be retained based on retention policy
        
        Args:
            created_at: When data was created
        
        Returns:
            True if should retain, False if should delete
        """
        cutoff = datetime.now() - timedelta(days=self.data_retention_days)
        return created_at > cutoff


class SecurityManager:
    """Main security manager coordinating all security components"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize security manager
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.credential_manager = CredentialManager(
            storage_path=config.get('credential_storage_path', 'credentials'),
            use_keyring=config.get('use_keyring', True)
        )
        self.session_manager = SessionManager(
            session_timeout=config.get('session_timeout', 3600)
        )
        self.privacy_manager = PrivacyManager(
            data_retention_days=config.get('data_retention_days', 7)
        )
    
    def authenticate_user(self, service: str, username: str, password: str) -> Optional[str]:
        """
        Authenticate user and create session
        
        Args:
            service: Service name
            username: Username
            password: Password
        
        Returns:
            Session ID if successful, None otherwise
        """
        # Check rate limiting
        if not self.session_manager.check_rate_limit(username):
            return None
        
        # Verify credentials
        stored_cred = self.credential_manager.get_credential(service)
        if not stored_cred or stored_cred['username'] != username or stored_cred['password'] != password:
            self.session_manager.record_failed_attempt(username)
            return None
        
        # Create session
        permissions = ['meeting_access', 'voice_synthesis', 'automation']
        return self.session_manager.create_session(username, permissions)
    
    def validate_operation(self, session_id: str, operation: str) -> bool:
        """
        Validate if operation is allowed for session
        
        Args:
            session_id: Session ID
            operation: Operation to validate
        
        Returns:
            True if allowed, False otherwise
        """
        context = self.session_manager.validate_session(session_id)
        if not context:
            return False
        
        # Check permissions (simplified)
        required_permissions = {
            'join_meeting': ['meeting_access'],
            'synthesize_voice': ['voice_synthesis'],
            'control_automation': ['automation']
        }
        
        required = required_permissions.get(operation, [])
        return all(perm in context.permissions for perm in required)
    
    def cleanup(self):
        """Perform security cleanup tasks"""
        self.session_manager.cleanup_expired_sessions()
        # Additional cleanup tasks would go here
