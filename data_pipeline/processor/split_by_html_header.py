import re


def parse_section_content(section_body):
    """
    Recursively parse the section body to extract a flat list of content blocks.
    Handles headers, paragraphs, lists, code blocks, and nested structure.
    """
    blocks = []
    
    if not hasattr(section_body, 'children'):
        return blocks

    for element in section_body.children:
        # 0. Headers (h1-h6)
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text = element.get_text(" ", strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                level = int(element.name[1])
                blocks.append({"type": "header", "level": level, "text": text})
            continue

        if element.name == 'div':
            classes = element.get('class', [])
            
            # 1. Paragraph
            if 'paragraph' in classes:
                text = element.get_text(" ", strip=True) 
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    blocks.append({"type": "paragraph", "text": text})
            
            # 2. Listing Block (Code)
            elif 'listingblock' in classes:
                code_tag = element.select_one('code')
                language = code_tag.get('data-lang', 'unknown') if code_tag else 'unknown'
                
                pre = element.find('pre')
                if pre:
                    text = pre.get_text()
                else:
                    text = element.get_text()
                    
                blocks.append({
                    "type": "code", 
                    "language": language,
                    "text": text
                })
                
            # 3. List
            elif 'ulist' in classes or 'olist' in classes:
                items = []
                for li in element.select('li'):
                    p = li.find('p')
                    if p:
                        txt = p.get_text(" ", strip=True)
                    else:
                        txt = li.get_text(" ", strip=True)
                    
                    txt = re.sub(r'\s+', ' ', txt).strip()
                    items.append("- " + txt)
                        
                if items:
                    blocks.append({
                        "type": "list",
                        "text": "\n".join(items)
                    })
            
            # 4. Admonition
            elif 'admonitionblock' in classes:
                adm_type = 'note'
                for c in classes:
                    if c in ['tip', 'note', 'important', 'warning', 'caution']:
                        adm_type = c
                        break
                
                content_cell = element.select_one('td.content')
                if content_cell:
                    text = content_cell.get_text(" ", strip=True)
                else:
                    text = element.get_text(" ", strip=True)
                
                text = re.sub(r'\s+', ' ', text).strip()
                
                blocks.append({
                    "type": "admonition",
                    "subtype": adm_type,
                    "text": text
                })

            # 5. Structural/Container Divs - Recurse
            # preamble, sect1-5, sectionbody, openblock, tabs, content, tabpanel
            # This ensures we penetrate all wrappers to find content
            else:
                # Check known container classes or valid structure
                recursion_targets = [
                    'sectionbody', 'preamble', 
                    'sect1', 'sect2', 'sect3', 'sect4', 'sect5',
                    'openblock', 'content', 'tabpanel', 'tabs'
                ]
                
                # Special case: id="preamble" might not have class
                is_preamble = element.get('id') == 'preamble'
                
                if is_preamble or any(c in classes for c in recursion_targets):
                    inner_blocks = parse_section_content(element)
                    blocks.extend(inner_blocks)
                
                # If no class matches but it's a generic div, we might want to peek inside?
                # Sometimes styling divs exist. But ignoring them is safer unless we miss content.
                # For now, explicit recursion is safer.

    return blocks