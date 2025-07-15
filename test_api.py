import streamlit as st
import requests

st.title("🧪 Tenor API Test")

TENOR_API_KEY = "AIzaSyAARkiZD-Qo59Pji8ihow4hJCZDOsdchzE"
url = "https://tenor.googleapis.com/v2/search"

if st.button("Test API"):
    params = {
        "q": "happy",
        "key": TENOR_API_KEY,
        "limit": 1
    }
    
    try:
        response = requests.get(url, params=params)
        st.write(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            st.write("✅ Success!")
            st.json(data)
            
            if data.get("results"):
                gif_data = data["results"][0]
                st.write("First GIF:")
                st.json(gif_data)
        else:
            st.error(f"❌ Error: {response.status_code}")
            st.write(response.text)
            
    except Exception as e:
        st.error(f"❌ Exception: {e}")