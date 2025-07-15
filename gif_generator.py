# gif_generator_fixed.py
import requests
import random
import streamlit as st
import json

from streamlit.runtime.scriptrunner import RerunException

def rerun():
    raise RerunException

# Tenor API Configuration
try:
    with open("key.txt", encoding="utf-8") as f:
        TENOR_API_KEY = f.read().strip()
except FileNotFoundError:
    TENOR_API_KEY = None
    st.error("🔑 Tenor API key not found. Create a 'key.txt' file with your Tenor API key.")
    st.stop()

TENOR_BASE_URL = "https://tenor.googleapis.com/v2/search"

# Emotion to GIF search terms mapping
EMOTION_GIF_MAP = {
    "sadness": ["sad cat", "crying", "comfort hug", "rain sad"],
    "anger": ["angry cat", "rage", "mad face", "frustrated"],
    "joy": ["happy dance", "celebration", "party", "excited"],
    "fear": ["scared cat", "hiding", "nervous", "anxiety"],
    "surprise": ["surprised", "shocked", "wow", "mind blown"],
    "disgust": ["disgusted", "eww", "gross", "yuck"],
    "love": ["love", "heart eyes", "romantic", "cute couple"],
    "excitement": ["excited", "hyped", "pumped", "enthusiastic"],
    "neutral": ["thinking", "contemplating", "neutral", "okay"],
    "stress": ["stressed", "overwhelmed", "tired", "exhausted"],
    "gratitude": ["thank you", "grateful", "appreciation", "blessed"],
    "disappointment": ["disappointed", "let down", "sad face", "sigh"],
    "confusion": ["confused", "question mark", "puzzled", "what"],
    "embarrassment": ["embarrassed", "shy", "awkward", "cringe"],
    "pride": ["proud", "achievement", "success", "winning"],
    "relief": ["relief", "phew", "relaxed", "calm"],
    "nervousness": ["nervous", "anxious", "worried", "fidgeting"],
    "optimism": ["optimistic", "positive", "hopeful", "bright"],
    "caring": ["caring", "compassion", "kindness", "support"],
    "curiosity": ["curious", "interested", "wondering", "exploring"],
    "approval": ["thumbs up", "approval", "good job", "yes"],
    "disapproval": ["thumbs down", "no", "disapproval", "not good"],
    "annoyance": ["annoyed", "irritated", "eye roll", "ugh"],
    "grief": ["grief", "mourning", "loss", "sad memorial"],
    "remorse": ["sorry", "regret", "apologetic", "guilt"],
    "admiration": ["admiration", "impressed", "wow", "amazing"],
    "desire": ["want", "craving", "longing", "wishful"],
    "realization": ["lightbulb", "aha moment", "realization", "eureka"]
}

def test_tenor_api():
    """Test function to debug Tenor API connection"""
    st.markdown("### 🔧 API Test")
    
    if st.button("Test Tenor API", key="test_api"):
        if not TENOR_API_KEY:
            st.error("No API key found!")
            return
        
        params = {
            "q": "happy",
            "key": TENOR_API_KEY,
            "limit": 1,
            "media_filter": "gif",
            "contentfilter": "medium"
        }
        
        try:
            st.write("Making API request...")
            response = requests.get(TENOR_BASE_URL, params=params, timeout=10)
            st.write(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                st.write("✅ API Response received!")
                st.json(data)
                
                if data.get("results"):
                    st.write("✅ Results found!")
                    gif_data = data["results"][0]
                    st.write("First GIF data:")
                    st.json(gif_data)
                else:
                    st.error("❌ No results in response")
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def fetch_emotion_gif(emotion: str, limit: int = 10) -> dict:
    """
    Fetch GIF from Tenor API based on emotion
    Returns dict with gif_url, title, and success status
    """
    try:
        # Get search terms for the emotion
        search_terms = EMOTION_GIF_MAP.get(emotion.lower(), ["neutral", "okay"])
        search_query = random.choice(search_terms)
        
        # API parameters
        params = {
            "q": search_query,
            "key": TENOR_API_KEY,
            "limit": limit,
            "media_filter": "gif",
            "contentfilter": "medium"
        }
        
        # Make API request
        response = requests.get(TENOR_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("results"):
            # Pick a random GIF from results
            gif_data = random.choice(data["results"])
            
            # Extract GIF URL from the correct structure
            gif_url = gif_data["media_formats"]["gif"]["url"]
            title = gif_data.get("content_description", f"{emotion} GIF")
            
            return {
                "success": True,
                "gif_url": gif_url,
                "title": title,
                "search_term": search_query
            }
        else:
            return {
                "success": False,
                "error": "No GIFs found for this emotion"
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"API request failed: {str(e)}"
        }
    except KeyError as e:
        return {
            "success": False,
            "error": f"GIF URL not found in expected format: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

# Also update your display function to handle errors better
def display_emotion_gif(emotion: str, confidence: float) -> None:
    """
    Display emotion-based GIF in Streamlit UI
    """
    st.markdown("---")
    st.subheader(f"🎭 {emotion.title()} Mood GIF")
    
    # Initialize session state for gif url if not present
    if f"gif_url_{emotion}" not in st.session_state:
        with st.spinner("🎬 Finding the perfect GIF..."):
            gif_result = fetch_emotion_gif(emotion)
            if gif_result["success"]:
                st.session_state[f"gif_url_{emotion}"] = gif_result["gif_url"]
                st.session_state[f"gif_title_{emotion}"] = gif_result["title"]
                st.session_state[f"gif_search_term_{emotion}"] = gif_result["search_term"]
            else:
                st.session_state[f"gif_url_{emotion}"] = None
                st.session_state[f"gif_error_{emotion}"] = gif_result["error"]
    
    # Create columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🔄 Get New GIF", key=f"gif_refresh_{emotion}"):
            # Fetch new gif and update session state
            with st.spinner("🎬 Finding the perfect GIF..."):
                gif_result = fetch_emotion_gif(emotion)
                if gif_result["success"]:
                    st.session_state[f"gif_url_{emotion}"] = gif_result["gif_url"]
                    st.session_state[f"gif_title_{emotion}"] = gif_result["title"]
                    st.session_state[f"gif_search_term_{emotion}"] = gif_result["search_term"]
                    # Force rerun to update UI
                    st.experimental_rerun()
                else:
                    st.session_state[f"gif_url_{emotion}"] = None
                    st.session_state[f"gif_error_{emotion}"] = gif_result["error"]
                    st.experimental_rerun()
    
    with col2:
        st.markdown(f"*Confidence: {confidence:.1%}*")
    
    gif_url = st.session_state.get(f"gif_url_{emotion}")
    gif_title = st.session_state.get(f"gif_title_{emotion}")
    gif_search_term = st.session_state.get(f"gif_search_term_{emotion}")
    gif_error = st.session_state.get(f"gif_error_{emotion}")
    
    if gif_url:
        st.markdown(f"**{gif_title}**")
        st.caption(f"Search term: *{gif_search_term}*")
        
        # Display the GIF with smaller width for better layout
        st.image(gif_url, width=350)
        
        # Add some fun emojis based on emotion
        emoji_map = {
            "sadness": "😢💙", "anger": "😡🔥", "joy": "😊🎉",
            "fear": "😨💭", "surprise": "😲✨", "love": "❤️💕",
            "excitement": "🤩⚡", "neutral": "😐⚖️", "stress": "😰💆‍♀️"
        }
        emoji = emoji_map.get(emotion.lower(), "🎭✨")
        st.markdown(f"<div style='text-align: center; font-size: 2rem;'>{emoji}</div>", 
                   unsafe_allow_html=True)
    else:
        st.error(f"❌ {gif_error if gif_error else 'Failed to load GIF'}")
        st.info("💡 Try analyzing another emotion or check your internet connection")
         #--------------------------------------------------------------------------------------#       
        # Debug info for failed requests
        with st.expander("🔍 Debug Info"):
            st.json({})
#-----------------------------------------------------------------------------------------#
def create_gif_tab() -> None:
    """
    Create a dedicated GIF tab for standalone GIF generation
    """
    st.markdown("### 🎭 Emotion GIF Generator")
    st.markdown("Generate fun GIFs and memes based on any emotion!")
    
    # Add API test section
    test_tenor_api()
    
    st.markdown("---")
    
    # Manual emotion selection
    col1, col2 = st.columns(2)
    with col1:
        selected_emotion = st.selectbox(
            "Choose an emotion:",
            list(EMOTION_GIF_MAP.keys()),
            index=0
        )
    with col2:
        confidence = st.slider("Confidence level:", 0.1, 1.0, 0.8)
    
    if st.button("🎬 Generate GIF", use_container_width=True):
        display_emotion_gif(selected_emotion, confidence)
    
    # Show available emotions
    st.markdown("#### Available Emotions:")
    emotions_text = ", ".join([f"**{emotion}**" for emotion in sorted(EMOTION_GIF_MAP.keys())])
    st.markdown(emotions_text)

# Quick integration function for existing tabs
def add_gif_to_emotion_display(emotion: str, confidence: float, show_button: bool = True) -> None:
    """
    Quick function to add GIF display to existing emotion results
    Call this after displaying emotion scores in your existing tabs
    """
    if show_button:
        if st.button("🎭 Show Emotion GIF", key=f"show_gif_{emotion}_{confidence}"):
            display_emotion_gif(emotion, confidence)
    else:
        display_emotion_gif(emotion, confidence)

# Main function for testing
if __name__ == "__main__":
    st.title("🎭 GIF Generator Debug")
    create_gif_tab()