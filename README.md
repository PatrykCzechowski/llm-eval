# llm-eval

## Porównanie wyników Tender Scout

Porównaj dwa kompletne eksporty JSON z `TenderScout.Evaluation` na tym samym zbiorze pytań.

```bash
uv run --locked python -m llm_eval.results.tender_scout baseline.json candidate.json
```

Raport podaje liczbę zaliczonych pytań w obu przebiegach oraz identyfikatory regresji i popraw. Pytanie z odpowiedzią w dokumentach przechodzi, gdy system odpowiada i wskazuje właściwy cytat. Pytanie bez odpowiedzi przechodzi przy odmowie. Eksporty muszą zawierać te same identyfikatory i etykiety. Do porównania użyj tej samej wersji zbioru i udokumentuj różnice konfiguracji.

### Raport A/B w GitHub Actions

Workflow `Tender Scout A/B report` porównuje dwa gotowe eksporty. Evaluator Tender Scout i lokalna Ollama generują je poza GitHubem. Zestaw pytań, dokumenty i model mogą zostać lokalnie. GitHub dostaje tylko aliasy przypadków, etykiety logiczne i liczniki cytatów. Raport w podsumowaniu uruchomienia jest publiczny, tak jak repozytorium.

Po dodaniu workflow do domyślnej gałęzi uruchom z katalogu `llm-eval`:

```bash
set -o pipefail
uv run --locked python scripts/tender_scout_ci.py prepare baseline.json candidate.json /tmp/tender-scout-case-map.json \
  | gh workflow run tender-scout-report.yml --json
```

Mapa łączy aliasy z lokalnymi identyfikatorami. Podaj nową ścieżkę dla każdej pary porównań i zachowaj mapę lokalnie. Workflow służy do diagnozy i może zakończyć się sukcesem mimo znalezienia regresji. Nie jest wymaganym statusem pull requesta. Kod wyjścia błędu oznacza problem z wejściem lub środowiskiem.
