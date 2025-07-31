import streamlit as st
import sys
import importlib.util

def check_tensorflow_compatibility():
    """
    Check TensorFlow compatibility and provide status information.
    """
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        
        # Check if tf-keras is available
        try:
            import tf_keras
            tf_keras_version = tf_keras.__version__
            tf_keras_available = True
        except ImportError:
            tf_keras_version = "Not installed"
            tf_keras_available = False
        
        # Check compatibility
        compatibility_status = "Compatible" if tf_keras_available else "Needs tf-keras"
        
        return {
            "tensorflow_version": tf_version,
            "tf_keras_version": tf_keras_version,
            "tf_keras_available": tf_keras_available,
            "compatibility_status": compatibility_status,
            "python_version": sys.version
        }
        
    except ImportError as e:
        return {
            "tensorflow_version": "Not installed",
            "tf_keras_version": "Not installed", 
            "tf_keras_available": False,
            "compatibility_status": "TensorFlow not found",
            "error": str(e)
        }

def display_dependency_status():
    """
    Display dependency status in Streamlit sidebar.
    """
    with st.sidebar.expander("🔧 System Status", expanded=False):
        status = check_tensorflow_compatibility()
        
        st.write("**Dependencies:**")
        st.write(f"• TensorFlow: {status['tensorflow_version']}")
        st.write(f"• tf-keras: {status['tf_keras_version']}")
        st.write(f"• Python: {status['python_version'][:5]}")
        
        if status['tf_keras_available']:
            st.success("✅ All dependencies OK")
        else:
            st.warning("⚠️ tf-keras needed")
            if st.button("Install tf-keras"):
                st.code("pip install tf-keras")

def resolve_dependency_conflicts():
    """
    Provide instructions for resolving dependency conflicts.
    """
    st.info("""
    **To resolve TensorFlow compatibility issues:**
    
    1. Uninstall existing packages:
    ```bash
    pip uninstall tensorflow tf-keras -y
    ```
    
    2. Install compatible versions:
    ```bash
    pip install tensorflow tf-keras
    ```
    
    3. If issues persist, try:
    ```bash
    pip install tensorflow==2.15.0 tf-keras==2.15.0
    ```
    """)

if __name__ == "__main__":
    # Test the dependency checker
    status = check_tensorflow_compatibility()
    print("Dependency Status:", status)