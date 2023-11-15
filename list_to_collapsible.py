"""This script turns a list in a markdown file into a collapsible dropdown.

The markdown file is assumed to be exported from Notion (particularly at this
link: https://chawins.github.io/llm-sp). The collapsible dropdown is in HTML.
"""


def reformat_header(line):
    dropdown_header = line.strip()[2:]
    title_end_idx = dropdown_header.find("[[")
    title = dropdown_header[:title_end_idx]

    # Find all the links
    links = []
    link_string = dropdown_header
    while True:
        # Not very efficient but will do
        _idx = link_string.find("[[")
        if _idx == -1:
            break
        name_start_idx = _idx + 2
        name_end_idx = link_string.find("]")
        name = link_string[name_start_idx:name_end_idx]
        url_start_idx = name_end_idx + 2
        url_start = link_string[url_start_idx:]
        url_len = url_start.find(")")
        url = link_string[url_start_idx : url_start_idx + url_len]
        new_link = f'[<a href="{url}">{name}</a>]'
        links.append(new_link)
        link_string = link_string[url_start_idx + url_len + 2 :]

    # What's left is the emojis
    emojis = link_string
    links = " ".join(links)
    return f"{title}{links}{emojis}"


def markdown_to_collapsible_dropdown(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.readlines()

    new_content = []
    in_list = False
    dropdown_content = []

    line_idx = 0
    for line in content:
        if "Other resources" in line:
            break
        if line.startswith("-") and not line.startswith("---") and not in_list:
            # Start new paper
            dropdown_content = []
            dropdown_content.append("<details>")
            new_header = reformat_header(line)
            dropdown_content.append(f"<summary>{new_header}</summary>\n\n")
            in_list = True
        elif line.startswith("-") and not line.startswith("---"):
            dropdown_content.append("</details>\n")
            dropdown_content.append("\n")
            new_content.extend(dropdown_content)
            dropdown_content = []
            dropdown_content.append("<details>")
            new_header = reformat_header(line)
            dropdown_content.append(f"<summary>{new_header}</summary>\n\n")
            in_list = True
        elif line.startswith("    "):
            assert in_list, line
            dropdown_content.append(line[4:])
        else:
            if dropdown_content:
                dropdown_content.append("</details>\n")
                dropdown_content.append("\n")
                new_content.extend(dropdown_content)
                dropdown_content = []
            new_content.append(line)
            in_list = False
        line_idx += 1

    new_content.extend(content[line_idx:])

    # Write to a new file
    with open("converted_" + filename, "w", encoding="utf-8") as f:
        for line in new_content:
            f.write(line)


if __name__ == "__main__":
    markdown_file = input("Enter the Markdown filename: ")
    markdown_to_collapsible_dropdown(markdown_file)
    print(f"Converted content written to converted_{markdown_file}")
