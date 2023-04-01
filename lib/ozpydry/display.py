"""Display wrapper."""

from IPython.display import display, Markdown

def md(*s):
    """Render markdown formated strings."""
    display(Markdown(*s))