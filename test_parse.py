import re

filepath = "C:\\Users\\Leandro Almeida\\Downloads\\DSL-AG-hybrid\\novo_experimento2.txt"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Test patterns
chunk_start = re.compile(r'CHUNK (\d+) - INÍCIO')
chunk_final = re.compile(r'CHUNK (\d+) - FINAL')
tempo_total = re.compile(r'Tempo total: ([\d.]+)s \(([\d.]+)min\)')
train_gmean = re.compile(r'Train G-mean: ([\d.]+)')
test_gmean = re.compile(r'Test G-mean:\s+([\d.]+)')

lines = content.split('\n')

print("Testing patterns...")
print()

chunk_count = 0
tempo_count = 0
train_count = 0
test_count = 0

for line in lines:
    if chunk_start.search(line):
        chunk_count += 1
        print(f"Found chunk start: {line.strip()}")
    if chunk_final.search(line):
        print(f"Found chunk final: {line.strip()}")

    match = tempo_total.search(line)
    if match:
        tempo_count += 1
        print(f"Found tempo: s={match.group(1)}, min={match.group(2)}")

    match = train_gmean.search(line)
    if match:
        train_count += 1
        print(f"Found train gmean: {match.group(1)}")

    match = test_gmean.search(line)
    if match:
        test_count += 1
        print(f"Found test gmean: {match.group(1)}")
        print()

print(f"\nTotals: chunks={chunk_count}, tempo={tempo_count}, train={train_count}, test={test_count}")
