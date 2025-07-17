#!/usr/bin/env python3
"""
Comprehensive Supabase diagnostics
"""

from supabase import create_client
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
SUPABASE_URL = "https://thvugbyogpuvcljlcplf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRodnVnYnlvZ3B1dmNsamxjcGxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MjQxODgsImV4cCI6MjA2NTEwMDE4OH0.zM9jr8RXEqTDSjFcFhOVBzpi-iSg8-BkxvdULwqAxww"

def test_basic_connection():
    """Test basic connection to Supabase"""
    print("🔍 Testing basic connection...")
    try:
        # Test direct HTTP request
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/",
            headers=headers,
            timeout=10
        )
        
        print(f"   HTTP Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Basic connection successful")
            return True
        else:
            print(f"❌ Connection failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_supabase_client():
    """Test Supabase client initialization"""
    print("\n🔍 Testing Supabase client...")
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Client created successfully")
        return client
    except Exception as e:
        print(f"❌ Client creation failed: {e}")
        return None

def test_table_permissions():
    """Test different approaches to access the table"""
    print("\n🔍 Testing table permissions...")
    
    try:
        # Test with direct HTTP request
        headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json'
        }
        
        # Try to access mood_history table
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/mood_history?limit=1",
            headers=headers,
            timeout=10
        )
        
        print(f"   Direct HTTP Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Direct HTTP access successful")
            data = response.json()
            print(f"   Records found: {len(data)}")
            return True
        else:
            print(f"❌ Direct HTTP failed: {response.text}")
            
            # Check if it's a specific error
            if response.status_code == 401:
                print("   🔒 Authentication issue")
            elif response.status_code == 404:
                print("   📋 Table not found")
            elif response.status_code == 403:
                print("   🚫 Permission denied")
                
            return False
            
    except Exception as e:
        print(f"❌ HTTP test error: {e}")
        return False

def provide_solutions():
    """Provide comprehensive solutions"""
    print("\n🛠️ COMPREHENSIVE SOLUTIONS:")
    print("="*60)
    
    print("\n1️⃣ GRANT PERMISSIONS TO ANON ROLE:")
    print("Run this SQL in Supabase SQL Editor:")
    print("""
-- Grant usage on schema
GRANT USAGE ON SCHEMA public TO anon;

-- Grant permissions on mood_history table
GRANT ALL ON TABLE public.mood_history TO anon;

-- Grant permissions on sequence (for auto-increment)
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon;
""")
    
    print("\n2️⃣ DISABLE RLS COMPLETELY:")
    print("Run this SQL in Supabase SQL Editor:")
    print("""
-- Disable RLS on the table
ALTER TABLE public.mood_history DISABLE ROW LEVEL SECURITY;
""")
    
    print("\n3️⃣ CHECK TABLE OWNERSHIP:")
    print("Run this SQL to check table ownership:")
    print("""
-- Check table owner
SELECT schemaname, tablename, tableowner 
FROM pg_tables 
WHERE tablename = 'mood_history';

-- Check permissions
SELECT grantee, privilege_type 
FROM information_schema.role_table_grants 
WHERE table_name = 'mood_history';
""")
    
    print("\n4️⃣ RECREATE TABLE WITH PROPER PERMISSIONS:")
    print("If all else fails, run this SQL:")
    print("""
-- Drop and recreate table
DROP TABLE IF EXISTS public.mood_history;

CREATE TABLE public.mood_history (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    emotion TEXT NOT NULL,
    score FLOAT DEFAULT 0.0,
    confidence FLOAT DEFAULT 0.0,
    all_emotions JSONB,
    analysis_type TEXT DEFAULT 'text',
    session_id TEXT,
    user_input_length INTEGER DEFAULT 0,
    processing_time FLOAT DEFAULT 0.0,
    weather_condition TEXT,
    time_of_day INTEGER,
    day_of_week INTEGER,
    user_location TEXT,
    user_id TEXT
);

-- Disable RLS
ALTER TABLE public.mood_history DISABLE ROW LEVEL SECURITY;

-- Grant permissions
GRANT ALL ON TABLE public.mood_history TO anon;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO anon;
""")

def main():
    """Main diagnostic function"""
    print("🔬 COMPREHENSIVE SUPABASE DIAGNOSTICS")
    print("="*50)
    
    # Test 1: Basic connection
    if not test_basic_connection():
        print("\n❌ Basic connection failed - check your URL and API key")
        return False
    
    # Test 2: Client creation
    client = test_supabase_client()
    if not client:
        print("\n❌ Client creation failed")
        return False
    
    # Test 3: Table permissions
    if not test_table_permissions():
        print("\n❌ Table access failed")
        provide_solutions()
        return False
    
    print("\n🎉 ALL TESTS PASSED! Your Supabase is working correctly!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print(f"\n💡 Your app will continue working with local storage.")
        print(f"   Try the solutions above to enable Supabase integration.")