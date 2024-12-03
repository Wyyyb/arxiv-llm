import json
import os
from tqdm import tqdm


def generate_bibtex(metadata):
    """Generate BibTeX entry from arXiv metadata with proper field handling"""

    # Handle required fields first
    try:
        # Get year from versions or from journal-ref as fallback
        year = None
        if 'versions' in metadata and metadata['versions']:
            year = metadata['versions'][0]['created'].split(',')[1].strip().split(" ")[2]
        elif 'journal-ref' and ':' in metadata.get('journal-ref', ''):
            year = metadata['journal-ref'].split(',')[-1].strip()
            print("using journal-ref", year)

        # Get first author's lastname for citation key
        if 'authors_parsed' in metadata and metadata['authors_parsed']:
            first_author = metadata['authors_parsed'][0][0].replace('\\', '')
        else:
            first_author = metadata['authors'].split(',')[0].split()[-1].replace('\\', '')

        title = metadata['title'].strip()
        title_first_word = title.lower().split(" ")[0]
        # Create citation key
        citation_key = f"{first_author.lower()}{year}{title_first_word}" if year else f"{first_author.lower()}_arxiv"

        # Start building BibTeX
        bibtex = [f"@article{{{citation_key},"]

        # Handle essential fields
        if 'title' in metadata and metadata['title']:
            bibtex.append(f"    title={{{metadata['title'].strip()}}},")
        if 'authors' in metadata and metadata['authors']:
            authors = metadata['authors'].replace('\\', '')
            bibtex.append(f"    author={{{authors}}},")

        # Handle journal reference
        if metadata.get('journal-ref'):
            bibtex.append(f"    journal={{{metadata['journal-ref']}}},")
        else:
            bibtex.append(f"    journal={{arXiv preprint arXiv:{metadata['id']}}},")

        # Add year if available
        if year:
            bibtex.append(f"    year={{{year}}},")

        # Handle optional fields
        if metadata.get('doi'):
            bibtex.append(f"    doi={{{metadata['doi']}}},")

        # if metadata.get('abstract'):
        #     # Clean up abstract: remove newlines and extra spaces
        #     abstract = ' '.join(metadata['abstract'].split())
        #     bibtex.append(f"    abstract={{{abstract}}},")

        if metadata.get('report-no'):
            bibtex.append(f"    number={{{metadata['report-no']}}},")

        # Add arXiv information
        bibtex.append(f"    archivePrefix={{arXiv}},")
        bibtex.append(f"    eprint={{{metadata['id']}}},")
        if metadata.get('categories'):
            bibtex.append(f"    primaryClass={{{metadata['categories']}}},")

        # Remove trailing comma from last entry and close the bibtex entry
        bibtex[-1] = bibtex[-1].rstrip(',')
        bibtex.append("}")

        return '\n'.join(bibtex), citation_key, title

    except Exception as e:
        print(f"Error generating BibTeX: {str(e)}")
        # Return a minimal BibTeX entry with available information
        return f"""@article{{{metadata.get('id', 'unknown')},
    title={{{metadata.get('title', 'Unknown Title')}}},
    journal={{arXiv preprint arXiv:{metadata.get('id', 'unknown')}}},
    author={{{metadata.get('authors', 'Unknown Authors')}}}
}}""", metadata.get('id', 'unknown'), metadata.get('title', 'Unknown Title')


def check_cs(cat, cat_map):
    cat = cat.lower().replace("physics", "").replace("optics", "")
    if "cs" in cat:
        if cat not in cat_map:
            cat_map[cat] = 1
        else:
            cat_map[cat] += 1
        # cs_list = cat.split(" ")
        # for each in cs_list:
        #     if each not in cat_map:
        #         cat_map[each] = 1
        #     else:
        #         cat_map[each] += 1
        return True
    return False


def load_local_metadata(metadata_path):
    res = []
    exist_id = set()
    cat_map = {}
    with open(metadata_path) as fi:
        for line in tqdm(fi):
            curr = json.loads(line)
            cat = curr.get('categories', "")
            if not check_cs(cat, cat_map):
                continue
            arxiv_id = curr.get('id', None)
            bibtex, citation_key, title = generate_bibtex(curr)
            if arxiv_id is not None and arxiv_id not in exist_id:
                res.append({"id": arxiv_id, "bibtex": bibtex, "title": title,
                            "citation_key": citation_key})
            elif arxiv_id is None:
                print("invalid arxiv_id", line)
            elif arxiv_id in exist_id:
                print("arxiv_id repeated", arxiv_id)
            else:
                print("unknown error", line)
    print(cat_map)
    print("length", len(res))
    with open("bibtex_info_1202.jsonl", "w") as fo:
        for each in res:
            fo.write(json.dumps(each) + "\n")


load_local_metadata("/Users/yubowang/Downloads/arxiv-metadata-oai-snapshot_1122.json")






