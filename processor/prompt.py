

prompt_template = """Given the user's historical interactive items arranged in chronological order, please recommend a suitable item for the user. Please output the item {index}.
User's historical interactive items: 
"""

item_template = {
    "title": """Title: {title}
Description: {description}""",
    "sem_id": """Item Index: {sem_id}"""
}

prediction_template = {
    "title": """Title: {title}""",
    "sem_id": """Item Index: {sem_id}"""
}

t2i_prompt = """Given the following item title and desctription, please output the corresponding Item Index.
Examples:
{examples}
"""

i2t_prompt = """Given the following item index, please output the corresponding item title and desctription.
Examples:
{examples}
"""

text_template = """Title: {title}
Description: {description}"""

index_template = """Item Index: {index}"""