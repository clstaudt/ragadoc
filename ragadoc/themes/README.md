# 🎨 Theme System

This directory contains CSS theme files for the ragadoc Streamlit application.

## Available Themes

- **darkstreaming.css** - Modern dark theme with streaming-inspired design and green accents
- **classic.css** - Clean, professional light theme with blue accents

## How to Use

### Default Usage
The app uses the `darkstreaming` theme by default:

```python
from ragadoc import setup_streamlit_config
setup_streamlit_config()  # Uses darkstreaming theme
```

### Switch Themes
To use a different theme:

```python
from ragadoc import setup_streamlit_config
setup_streamlit_config(theme="classic")  # Uses classic theme
```

### Environment-Based Theme Selection
You can also set themes via environment variables:

```bash
export RAGADOC_THEME=classic
streamlit run app.py
```

## Creating New Themes

### 1. Create a CSS File
Create a new `.css` file in this directory (e.g., `mytheme.css`):

```css
/* My Custom Theme */
:root {
    --primary-color: #your-color;
    --bg-color: #your-bg;
    /* Define your variables */
}

.stApp {
    background: var(--bg-color);
    /* Your styling */
}

/* Style other components */
.stButton > button {
    background: var(--primary-color);
}
```

### 2. Use Your Theme
```python
setup_streamlit_config(theme="mytheme")
```

## Theme Structure

### CSS Variables
Use CSS variables for easy customization:
- `--primary-color` - Main brand color
- `--accent-color` - Secondary accent color  
- `--bg-color` - Background color
- `--text-color` - Text color
- `--border-color` - Border color

### Key Streamlit Classes
Important CSS classes to style:
- `.stApp` - Main application container
- `.stButton > button` - Buttons
- `.stSelectbox > div > div` - Select boxes
- `.stTextInput > div > div > input` - Text inputs
- `.stChatMessage` - Chat messages
- `.css-1d391kg` - Sidebar

## Fallback System

If a theme file doesn't exist, the system automatically falls back to a basic dark theme to ensure the app always has styling.

## Best Practices

1. **Use CSS Variables** - Makes themes easy to customize
2. **Keep It Focused** - Only style what you need to change
3. **Test Responsiveness** - Include mobile-friendly styles
4. **Comment Your Code** - Help others understand your theme
5. **Follow Naming Conventions** - Use descriptive theme names

## Contributing

To contribute a new theme:
1. Create a new CSS file following the naming convention
2. Test it thoroughly with the app
3. Add it to the list of available themes in this README
4. Submit a pull request

Happy theming! 🎨 