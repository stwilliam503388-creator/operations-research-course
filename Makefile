# 运筹学与优化方法课程系列 — 一键验证
# 用法:
#   make          → 列出所有代码
#   make python   → 运行官方 smoke checks
#   make check    → 运行官方 smoke checks
#   make test     → 运行官方 smoke checks
#   make cpp      → 运行包含 C++ 编译的官方 smoke checks
#   make list     → 列出所有可运行代码
#   make clean    → 清理编译产物

.PHONY: all python cpp check test list clean

PYTHON := python3

# 查找所有 Python 代码（排除 venv、cache 和 prompts 目录），仅用于列清单。
PY_FILES := $(shell find . -name '*.py' -not -path '*/venv/*' -not -path '*/.venv/*' -not -path '*/__pycache__/*' -not -path '*/prompts/*' | sort)

# 默认: 列出所有代码
all: list

list:
	@echo "========================================"
	@echo "  运筹学与优化方法课程系列 — 代码清单"
	@echo "========================================"
	@echo ""
	@echo "Python 代码 ($(words $(PY_FILES)) 个):"
	@for f in $(PY_FILES); do \
		printf "  %s\n" "$$f"; \
	done
	@echo ""
	@echo "用法:"
	@echo "  make python  — 运行官方 smoke checks"
	@echo "  make check   — 运行官方 smoke checks"
	@echo "  make test    — 运行官方 smoke checks"
	@echo "  make cpp     — 运行包含 C++ 编译的官方 smoke checks"
	@echo "  make clean   — 清理缓存文件"

python:
	@echo "========================================"
	@echo "  运行官方 smoke checks"
	@echo "========================================"
	$(PYTHON) run_checks.py

check: python

test: python

cpp: python

clean:
	find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	find . -path '*/code/*' -type f -name '*.png' -delete 2>/dev/null || true
	@echo "已清理缓存文件"
