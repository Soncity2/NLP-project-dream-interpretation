import re

def format_model_name(name):
    # Convert to lowercase
    name = name.lower()
    # Remove non-alphanumeric characters except numbers
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

# Example usage
formatted_name = format_model_name("GPT-2")
print(formatted_name)  # Output: gpt2
