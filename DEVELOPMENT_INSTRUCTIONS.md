
- Make sure you never destroy existing functionality when adding new functionality, unless it is a replacement or the old functionality is no longer needed.
- Never forget to update the conda environment config file when you update the requirements.txt
- Make sure there are concise and up to date docstrings that document usage.
- Debug information belongs into the command line logs, not in the app UI/UX.
- Always develop a generic solution, do not use content from specific examples in the code
- Never include content from example documents in the source code. Never leak content from provided examples into test code!