.PHONY: verify audit test
verify:
	python scripts/verify_release.py
audit:
	python scripts/reproduce.py --out reanalysis
test:
	python -m pytest -q
