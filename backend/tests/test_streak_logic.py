from datetime import datetime, timedelta, timezone
from pymongo import MongoClient
import os

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = MongoClient(MONGODB_URI)
db = client["speechai_db"]
users_collection = db["users"]

# Import function directly from main
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import update_user_streak

def test_streaks():
    test_user = "test_user_streak_verification"
    
    # 1. Clean up first
    users_collection.delete_one({"_id": test_user})
    print("Cleaned up old test user.")

    # 2. Case A: First practice ever
    streak = update_user_streak(test_user)
    print(f"Case A (First practice) - Expected: 1, Got: {streak}")
    assert streak == 1, "First practice streak should be 1"

    # 3. Case B: Practice again on the same day
    # Same day, streak should remain 1
    streak = update_user_streak(test_user)
    print(f"Case B (Same day practice) - Expected: 1, Got: {streak}")
    assert streak == 1, "Same day practice should not increment streak"

    # 4. Case C: Practice on consecutive days
    # Mock last practice as yesterday
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    users_collection.update_one(
        {"_id": test_user},
        {"$set": {"last_practice_date": yesterday, "streak_count": 1}}
    )
    streak = update_user_streak(test_user)
    print(f"Case C (Consecutive days practice) - Expected: 2, Got: {streak}")
    assert streak == 2, "Consecutive days practice should increment streak"

    # 5. Case D: Practice after missing a day
    # Mock last practice as 3 days ago
    three_days_ago = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    users_collection.update_one(
        {"_id": test_user},
        {"$set": {"last_practice_date": three_days_ago, "streak_count": 2}}
    )
    streak = update_user_streak(test_user)
    print(f"Case D (Missed days practice) - Expected: 1, Got: {streak}")
    assert streak == 1, "Missed days practice should reset streak to 1"

    # Clean up test user
    users_collection.delete_one({"_id": test_user})
    print("Verification completed successfully! All assertions passed.")

if __name__ == "__main__":
    test_streaks()
