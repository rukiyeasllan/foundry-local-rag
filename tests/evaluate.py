import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import LocalRAG, FALLBACK

TESTS = [
    ('Foundry Local ne ise yarar?', True, ['yerel']),
    ('RAG sistemi nasil calisir?', True, ['belge']),
    ('Embedding nedir?', True, ['vektor']),
    ('SQLite bu projede ne icin kullaniliyor?', True, ['sqlite', 'saklamak']),
    ('Turkiye Cumhuriyetinin baskenti neresidir?', False, []),
    ('Dunyanin en yuksek dagi hangisidir?', False, []),
]

def normalize(text):
    table = str.maketrans('çğıöşüÇĞİÖŞÜ', 'cgiosuCGIOSU')
    return text.translate(table).lower()

def contains_fallback(text):
    text = normalize(text)
    return ('bu bilgi mevcut belgelerde bulunamadi' in text or
            'bu bilgi mevcut belgelerde bulunmadi' in text)

def main():
    rag = LocalRAG()
    passed = 0
    total_time = 0.0

    print()
    print('=' * 70)
    print('RAG EVALUATION - STRICT')
    print('=' * 70)

    for i, (question, should_answer, keywords) in enumerate(TESTS, 1):
        start = time.perf_counter()
        answer, sources = rag.ask(question)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        normalized_answer = normalize(answer)
        fallback_found = contains_fallback(answer)

        if should_answer:
            keyword_ok = all(normalize(k) in normalized_answer for k in keywords)
            success = (not fallback_found) and keyword_ok and bool(sources)
        else:
            success = fallback_found and not sources

        if success:
            passed += 1

        print()
        print(f'TEST {i}: {question}')
        print('Durum:', 'PASS' if success else 'FAIL')
        print(f'Sure: {elapsed:.2f} saniye')
        print('Cevap:', answer)
        if sources:
            print(f"Benzerlik: {sources[0]['score']:.4f}")

    print()
    print('=' * 70)
    print('OZET')
    print('=' * 70)
    print(f'Basarili test: {passed}/{len(TESTS)}')
    print(f'Basari orani: {passed / len(TESTS) * 100:.1f}%')
    print(f'Ortalama sure: {total_time / len(TESTS):.2f} saniye')

if __name__ == '__main__':
    main()

