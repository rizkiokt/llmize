import textwrap

def pretty_print(prompt=None, response=None, width=150):
    print("=" * width)
    
    def wrap_text(text):
        """Wrap text while preserving explicit newlines."""
        return "\n".join(textwrap.fill(line, width) if line.strip() else "" for line in text.splitlines())

    if prompt:
        print("Prompt:")
        print(wrap_text(prompt))
        print()
    
    if response:
        print("Response:")
        print(wrap_text(response))
    
    print("=" * width)
