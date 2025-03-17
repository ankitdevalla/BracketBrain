import streamlit as st

def is_mobile():
    """Detect if user is on a mobile device based on viewport width"""
    # Get viewport width from query params (if available)
    try:
        # Use the new API instead of the deprecated one
        viewport_width = st.query_params.get("vw", ["1000"])[0]
        return int(viewport_width) < 768
    except:
        # Default to desktop if we can't determine
        return False

def inject_mobile_js():
    """Inject JavaScript to detect mobile devices and set viewport"""
    mobile_js = """
    <script>
    // Set viewport meta tag for mobile devices
    const meta = document.createElement('meta');
    meta.name = 'viewport';
    meta.content = 'width=device-width, initial-scale=1, maximum-scale=1';
    document.head.appendChild(meta);
    
    // Function to detect mobile and set query param
    function detectMobile() {
        const viewportWidth = window.innerWidth;
        const currentUrl = new URL(window.location.href);
        currentUrl.searchParams.set('vw', viewportWidth);
        window.history.replaceState({}, '', currentUrl);
        
        // Add mobile class to body if viewport is narrow
        if (viewportWidth < 768) {
            document.body.classList.add('mobile-view');
        } else {
            document.body.classList.remove('mobile-view');
        }
    }
    
    // Run on load and resize
    window.addEventListener('load', detectMobile);
    window.addEventListener('resize', detectMobile);
    </script>
    """
    st.markdown(mobile_js, unsafe_allow_html=True) 