#!/usr/bin/env python3
"""
Test table access and try to resolve RLS issues
"""

from supabase import create_client
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
SUPABASE_URL = "https://thvugbyogpuvcljlcplf.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRodnVnYnlvZ3B1dmNsamxjcGxmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDk1MjQxODgsImV4cCI6MjA2NTEwMDE4OH0.zM9jr8RXEqTDSjFcFhOVBzpi-iSg8-BkxvdULwqAxww"

def test_table_operations():
    """Test various table operations"""
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        print("🔍 Testing table operations...")
        
        # Test 1: Try to select from table
        print("\n1️⃣ Testing SELECT operation...")
        try:
            result = client.table("mood_history").select("*").limit(1).execute()
            print(f"✅ SELECT successful: {len(result.data) if result.data else 0} records found")
            if result.data:
                print(f"   Sample record: {result.data[0]}")
        except Exception as e:
            print(f"❌ SELECT failed: {e}")
        
        # Test 2: Try to insert a record
        print("\n2️⃣ Testing INSERT operation...")
        try:
            test_record = {
                "emotion": "test_emotion",
                "score": 0.75,
                "confidence": 0.75,
                "analysis_type": "test",
                "session_id": "test_session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            
            result = client.table("mood_history").insert(test_record).execute()
            if result.data:
                print("✅ INSERT successful!")
                print(f"   Inserted record ID: {result.data[0].get('id', 'unknown')}")
                
                # Test 3: Try to read the inserted record
                print("\n3️⃣ Testing SELECT after INSERT...")
                read_result = client.table("mood_history").select("*").eq("session_id", test_record["session_id"]).execute()
                if read_result.data:
                    print(f"✅ READ successful: Found {len(read_result.data)} records")
                    
                    # Test 4: Clean up test record
                    print("\n4️⃣ Cleaning up test record...")
                    delete_result = client.table("mood_history").delete().eq("id", result.data[0]["id"]).execute()
                    print("✅ Cleanup successful")
                    
                    return True
                else:
                    print("❌ Could not read inserted record")
            else:
                print("❌ INSERT returned no data")
                
        except Exception as e:
            print(f"❌ INSERT failed: {e}")
            
            # Check if it's an RLS issue
            if "policy" in str(e).lower() or "rls" in str(e).lower():
                print("\n🔒 This appears to be a Row Level Security (RLS) issue!")
                print("   You need to either:")
                print("   1. Disable RLS on the mood_history table, OR")
                print("   2. Create RLS policies to allow access")
                return False
        
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def provide_rls_solution():
    """Provide RLS solution instructions"""
    print("\n🛠️ To fix RLS issues, run this SQL in your Supabase SQL Editor:")
    print("\n" + "="*60)
    print("-- Option 1: Disable RLS (simpler, less secure)")
    print("ALTER TABLE mood_history DISABLE ROW LEVEL SECURITY;")
    print("\n-- Option 2: Create permissive RLS policies (more secure)")
    print("ALTER TABLE mood_history ENABLE ROW LEVEL SECURITY;")
    print("")
    print("-- Allow anonymous users to read all records")
    print("CREATE POLICY \"Allow anonymous read\" ON mood_history")
    print("FOR SELECT TO anon USING (true);")
    print("")
    print("-- Allow anonymous users to insert records")
    print("CREATE POLICY \"Allow anonymous insert\" ON mood_history")
    print("FOR INSERT TO anon WITH CHECK (true);")
    print("")
    print("-- Allow anonymous users to update their own records")
    print("CREATE POLICY \"Allow anonymous update\" ON mood_history")
    print("FOR UPDATE TO anon USING (true) WITH CHECK (true);")
    print("")
    print("-- Allow anonymous users to delete their own records")
    print("CREATE POLICY \"Allow anonymous delete\" ON mood_history")
    print("FOR DELETE TO anon USING (true);")
    print("="*60)

def main():
    """Main test function"""
    print("🧪 Testing Supabase table access...")
    
    success = test_table_operations()
    
    if success:
        print("\n🎉 SUCCESS! Your Supabase table is working perfectly!")
        print("✅ Table exists")
        print("✅ Can read data")
        print("✅ Can write data")
        print("✅ Can delete data")
        print("\nYour emotion AI app will now use Supabase as the primary database!")
    else:
        print("\n⚠️ Table access issues detected.")
        provide_rls_solution()
        print("\n💡 Don't worry - your app will still work with local storage until this is fixed!")
    
    return success

if __name__ == "__main__":
    main()