#!/usr/bin/env python3
"""
Check if the mood_history table exists in Supabase
"""

from supabase import create_client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
SUPABASE_URL = "https://thvugbyogpuvcljlcplf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRodnVnYnlvZ3B1dmNsamxjcGxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MjQxODgsImV4cCI6MjA2NTEwMDE4OH0.zM9jr8RXEqTDSjFcFhOVBzpi-iSg8-BkxvdULwqAxww"

def check_table_status():
    """Check if mood_history table exists and is accessible"""
    try:
        # Create Supabase client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Try to query the table
        logger.info("Checking if mood_history table exists...")
        result = client.table("mood_history").select("*").limit(1).execute()
        
        if result.data is not None:
            logger.info(f"✅ Table exists! Found {len(result.data)} records")
            if result.data:
                logger.info(f"Sample record: {result.data[0]}")
            else:
                logger.info("Table is empty (no records yet)")
            return True, "exists"
        else:
            logger.warning("Table query returned None")
            return False, "unknown"
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Table check failed: {error_msg}")
        
        if "permission denied" in error_msg.lower():
            return False, "permission_denied"
        elif "does not exist" in error_msg.lower() or "relation" in error_msg.lower():
            return False, "not_exists"
        else:
            return False, "connection_error"

def create_table_if_needed():
    """Try to create the table if it doesn't exist"""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Try to insert a test record (this will fail if table doesn't exist)
        test_record = {
            "emotion": "test",
            "score": 0.5,
            "created_at": "2025-01-17T10:00:00Z"
        }
        
        logger.info("Attempting to insert test record...")
        result = client.table("mood_history").insert(test_record).execute()
        
        if result.data:
            logger.info("✅ Test record inserted successfully - table exists and is writable")
            
            # Clean up test record
            if result.data[0].get('id'):
                client.table("mood_history").delete().eq('id', result.data[0]['id']).execute()
                logger.info("Test record cleaned up")
            
            return True
        else:
            logger.warning("Insert returned no data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Table creation/test failed: {e}")
        return False

def main():
    """Main function to check table status"""
    print("🔍 Checking Supabase table status...")
    
    exists, status = check_table_status()
    
    if exists:
        print("✅ mood_history table exists and is accessible!")
        return True
    else:
        print(f"❌ Table issue detected: {status}")
        
        if status == "not_exists":
            print("📝 Table doesn't exist. Attempting to create by inserting test data...")
            if create_table_if_needed():
                print("✅ Table created successfully!")
                return True
            else:
                print("❌ Failed to create table")
        elif status == "permission_denied":
            print("🔒 Permission denied - you may need to:")
            print("   1. Check your Supabase RLS (Row Level Security) policies")
            print("   2. Create the table manually in Supabase dashboard")
            print("   3. Verify your API key has the correct permissions")
        elif status == "connection_error":
            print("🌐 Connection error - check your internet and Supabase URL")
        
        print("\n💡 Don't worry! The app will still work with local storage.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n🔧 To create the table manually in Supabase:")
        print("1. Go to your Supabase dashboard")
        print("2. Navigate to Table Editor")
        print("3. Create a new table called 'mood_history'")
        print("4. Add these columns:")
        print("   - id (int8, primary key, auto-increment)")
        print("   - created_at (timestamptz, default: now())")
        print("   - emotion (text)")
        print("   - score (float8)")
        print("   - confidence (float8, nullable)")
        print("   - all_emotions (jsonb, nullable)")
        print("   - analysis_type (text, nullable)")
        print("   - session_id (text, nullable)")