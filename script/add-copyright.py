import os
import datetime
import logging

current_year = datetime.datetime.now().year
owner = "liangyuwang"
logging.basicConfig(filename='license_addition_errors.log', level=logging.ERROR)

def add_license_header(file_path, comment_style):
    try:
        with open(file_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            license_snippet = "Licensed under the Apache License, Version 2.0"
            if license_snippet not in content:
                header = "# Copyright Notice\n"
                if comment_style == "block":
                    header = f"/* Copyright (c) {current_year} {owner}\n * Licensed under the Apache License, Version 2.0\n */\n\n"
                elif comment_style == "line":
                    header = f"# Copyright (c) {current_year} {owner}\n# Licensed under the Apache License, Version 2.0\n\n"
                file.seek(0, 0)
                file.write(header + content)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")

file_map = {
    '.cpp': 'block',
    '.h': 'block',
    '.cu': 'block',
    '.py': 'line',
    '.cmake': 'line'
}

for root, dirs, files in os.walk("."):
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext in file_map:
            add_license_header(os.path.join(root, file), file_map[ext])
        elif 'CMakeLists.txt' in file:
            add_license_header(os.path.join(root, file), 'line')
