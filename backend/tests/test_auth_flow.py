import os
import sys
import uuid
import requests

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_password_hashing():
    print("Testing password hashing PBKDF2 helpers...")
    from main import hash_password, verify_password
    
    pwd = "MySecretPassword123"
    hashed = hash_password(pwd)
    print(f"Hashed format: {hashed}")
    
    assert verify_password(pwd, hashed) is True, "Password verification should pass"
    assert verify_password("WrongPassword", hashed) is False, "Password verification should fail with wrong password"
    print("Password hashing tests passed!")

def test_tokens():
    print("\nTesting HMAC token helpers...")
    from main import generate_token, verify_token
    
    user_id = "test-user-123"
    token = generate_token(user_id)
    print(f"Generated token: {token}")
    
    verified_id = verify_token(token)
    assert verified_id == user_id, f"Verified user ID should be {user_id}, got {verified_id}"
    
    # Test tampering
    tampered_token = token.replace("test-user-123", "tampered-user")
    assert verify_token(tampered_token) is None, "Verification should fail for tampered token"
    print("Token helper tests passed!")

def test_endpoints():
    print("\nTesting registration & login endpoints...")
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Generate unique emails
    test_id = str(uuid.uuid4())[:8]
    email = f"user_{test_id}@test.com"
    name = "Test User"
    password = "SuperSecretPassword"
    
    # 1. Register User
    reg_payload = {"name": name, "email": email, "password": password}
    response = client.post("/auth/register", json=reg_payload)
    print("Register Response:", response.status_code, response.json())
    assert response.status_code == 200, "Registration should succeed"

    # 2. Register Duplicate Email (should fail)
    response_dup = client.post("/auth/register", json=reg_payload)
    print("Register Duplicate Response:", response_dup.status_code)
    assert response_dup.status_code == 400, "Duplicate registration should fail with 400"

    # 3. Login with Correct Password
    login_payload = {"email": email, "password": password}
    response_login = client.post("/auth/login", json=login_payload)
    print("Login Correct Response:", response_login.status_code)
    assert response_login.status_code == 200, "Login should succeed"
    login_data = response_login.json()
    assert "access_token" in login_data, "Login response must contain access_token"
    
    # 4. Login with Incorrect Password
    login_bad_payload = {"email": email, "password": "WrongPassword"}
    response_bad_login = client.post("/auth/login", json=login_bad_payload)
    print("Login Bad Response:", response_bad_login.status_code)
    assert response_bad_login.status_code == 400, "Login with wrong password should fail with 400"

    print("Register & Login endpoint tests passed!")

if __name__ == "__main__":
    test_password_hashing()
    test_tokens()
    test_endpoints()
    print("\nALL PORTABLE CRYPTOGRAPHIC AUTH TESTS PASSED SUCCESSFULLY!")
