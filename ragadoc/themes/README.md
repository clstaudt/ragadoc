# 🎨 Darkstreaming Theme

This directory contains the darkstreaming CSS theme for the ragadoc Streamlit application.

## Theme

**darkstreaming.css** - Modern dark theme with streaming-inspired design and gold accents

## Usage

The darkstreaming theme is automatically applied when you run the application:

```python
from ragadoc import setup_streamlit_config
setup_streamlit_config()  # Always uses darkstreaming theme
```

## Theme Features

- **Modern Dark Design**: Sleek dark background with gold accents
- **Streaming-Inspired**: Professional appearance suitable for AI applications  
- **Responsive**: Mobile-friendly responsive design
- **Custom Components**: Styled Streamlit components including buttons, inputs, and chat messages
- **Typography**: Clean Inter font with proper contrast ratios
- **Visual Effects**: Gradients, shadows, and hover animations

## Theme Structure

### CSS Variables
The theme uses CSS variables for consistency:
- `--primary-gold: #FFB000` - Main brand color (gold)
- `--accent-gold: #FFC947` - Secondary accent color
- `--dark-bg: #121212` - Main background color  
- `--darker-bg: #0a0a0a` - Darker background for gradients
- `--surface-bg: #1a1a1a` - Surface/card backgrounds
- `--elevated-bg: #242424` - Input and elevated element backgrounds
- `--text-primary: #ffffff` - Primary text color
- `--text-secondary: #b3b3b3` - Secondary text color
- `--border-color: #2a2a2a` - Border color
- `--hover-color: #2a2a2a` - Hover state color

### Key Styled Components
- `.stApp` - Main application container with gradient background
- `.stButton > button` - Golden gradient buttons with hover effects
- `.stSelectbox > div > div` - Dark select boxes with gold focus states
- `.stTextInput > div > div > input` - Dark text inputs with gold accents
- `.stChatMessage` - Chat message styling with user/assistant differentiation
- `.css-1d391kg` - Sidebar with dark surface background

## Fallback System

If the darkstreaming.css file is missing, the system automatically falls back to a basic dark theme to ensure the app always has proper styling.

## Customization

To customize the theme, edit the CSS variables at the top of `darkstreaming.css`:

```css
:root {
    --primary-gold: #your-new-color;
    --accent-gold: #your-accent-color;
    /* Modify other variables as needed */
}
```

## Technical Details

- Uses modern CSS features including CSS Grid, Flexbox, and CSS Variables
- Includes webkit scrollbar styling for consistent appearance
- Responsive design with mobile breakpoints
- Proper accessibility considerations with sufficient color contrast
- Optimized for Streamlit's component structure

---

**The darkstreaming theme provides a professional, modern appearance perfect for AI document processing applications.** 🎨 