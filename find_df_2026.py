import json
nb = json.load(open('projet_NBA.ipynb', encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'df_2026 = df' in source:
            print(f'Cell index {i}: {cell.get("id", "no_id")}')
            print("Source preview:")
            print(source[:200])
            break
