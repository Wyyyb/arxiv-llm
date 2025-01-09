from step_3_extract_parts import extract_abstract

with open("test.txt", "r") as fi:
    text = fi.read()
abstract = extract_abstract(text)
print("abstract ", abstract )
