from datetime import datetime, timedelta
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from api.models import AccessLevel, TokenPayload
from config.settings import settings

pwd_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return pwd_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        pwd_hasher.verify(hashed_password, plain_password)
        return True
    except VerifyMismatchError:
        return False


USERS_DB = {
    "admin@company.com": {
        "email": "admin@company.com",
        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$9mO0IBrEB9A1ULYe4V7zlQ$R4/7X0GDwZR8BptJ587tGoT7jsWJ99RmWZZss3j0VCo",
        "access_level": AccessLevel.ADMIN,
        "full_name": "Admin User",
    },
    "user@company.com": {
        "email": "user@company.com",
        "hashed_password": "$argon2id$v=19$m=65536,t=3,p=4$sCkZDH8Fktu3ko72Ncx/RA$h7WcvC2wruvAcknwv0KOjoaxHcpd/8Ggfr6SGu6wpHg",
        "access_level": AccessLevel.USER,
        "full_name": "Regular User",
    },
}


def authenticate_user(email: str, password: str) -> dict | None:
    user = USERS_DB.get(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(email: str, access_level: AccessLevel) -> str:
    expire = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": email, "access_level": access_level.value, "exp": expire}
    encoded_jwt = jwt.encode(
        payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> TokenPayload:
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        email: str = payload.get("sub")
        access_level: str = payload.get("access_level")
        exp: int = payload.get("exp")
        if email is None or access_level is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing required claims",
            )
        return TokenPayload(
            sub=email,
            access_level=AccessLevel(access_level),
            exp=datetime.fromtimestamp(exp),
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e!s}"
        )


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenPayload:
    token = credentials.credentials
    return decode_access_token(token)


async def require_admin(
    current_user: TokenPayload = Depends(get_current_user),
) -> TokenPayload:
    if current_user.access_level != AccessLevel.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return current_user


def get_user_by_email(email: str) -> dict | None:
    return USERS_DB.get(email)


def create_test_users():
    return {
        "admin": {
            "email": "admin@company.com",
            "password": "admin123",
            "access_level": "admin",
        },
        "user": {
            "email": "user@company.com",
            "password": "user123",
            "access_level": "user",
        },
    }


if __name__ == "__main__":
    user = authenticate_user("admin@company.com", "admin123")
    if user:
        print(f"Authenticated: {user['email']} ({user['access_level']})")
        token = create_access_token(user["email"], user["access_level"])
        print(f"Token: {token[:50]}...")
        payload = decode_access_token(token)
        print(f"Decoded: {payload.sub} ({payload.access_level})")
    else:
        print("Authentication failed")
